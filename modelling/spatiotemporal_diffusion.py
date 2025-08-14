import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import convolve
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def spatiotemporal_diffusion_ode(t, y, grid_shape, dx, diffusion_rate, removal_rate, infectious_positions, generation_rates):
    """
    Defines the system of ODEs for the spatiotemporal diffusion model.

    Parameters:
        t (float): Current time (not used explicitly, as the system is autonomous).
        y (numpy.ndarray): Flattened 1D array of the grid's quanta concentrations.
        grid_shape (tuple): Shape of the grid (rows, cols).
        dx (float): Grid spacing (assumed dx = dy).
        diffusion_rate (float): Diffusion rate (in m^2/s).
        removal_rate (float): Removal rate (in s^-1).
        infectious_positions (list): List of (x, y) positions of infectious people in the grid.
        generation_rates (list): List of quanta generation rates (in quanta per second).

    Returns:
        numpy.ndarray: Flattened 1D array of the time derivatives of the grid's quanta concentrations.
    """
    # Reshape the 1D array back into a 2D grid
    grid = y.reshape(grid_shape)
    rows, cols = grid_shape

    # Initialize the time derivative of the grid
    dgrid_dt = np.zeros_like(grid)

    # Diffusion coefficient
    alpha = diffusion_rate / (dx**2)

    # Apply the 5-point stencil for diffusion
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dgrid_dt[i, j] = alpha * (
                grid[i - 1, j] + grid[i + 1, j] + grid[i, j - 1] + grid[i, j + 1] - 4 * grid[i, j]
            )

    # Add removal term
    dgrid_dt -= removal_rate * grid

    # Add quanta generation at infectious positions
    for (x, y), rate in zip(infectious_positions, generation_rates):
        dgrid_dt[x, y] += rate

    # Flatten the 2D array back into a 1D array
    return dgrid_dt.flatten()

def solve_diffusion(grid, dx, diffusion_rate, removal_rate, infectious_positions, generation_rates, t_span, dt):
    """
    Solves the spatiotemporal diffusion model using an ODE solver.

    Parameters:
        grid (numpy.ndarray): 2D array representing the initial quanta concentration.
        dx (float): Grid spacing (assumed dx = dy).
        diffusion_rate (float): Diffusion rate (in m^2/s).
        removal_rate (float): Removal rate (in s^-1).
        infectious_positions (list): List of (x, y) positions of infectious people in the grid.
        generation_rates (list): List of quanta generation rates (in quanta per second).
        t_span (tuple): Time span for the simulation (start, end) in seconds.
        dt (float): Time step for the solver output.

    Returns:
        tuple: (times, results), where `times` is an array of time points and `results` is a 3D array of quanta concentrations over time.
    """
    # Flatten the initial grid into a 1D array
    y0 = grid.flatten()

    # Time points for the solver
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    # Solve the ODE system
    sol = solve_ivp(
        spatiotemporal_diffusion_ode,
        t_span,
        y0,
        t_eval=t_eval,
        args=(grid.shape, dx, diffusion_rate, removal_rate, infectious_positions, generation_rates),
        method='LSODA'  # Similar to lsodes in R
    )

    # Reshape the solution back into a 3D array (time, rows, cols)
    results = sol.y.T.reshape(-1, grid.shape[0], grid.shape[1])

    return sol.t, results


def spatiotemporal_diffusion_ode2(t, y, grid_shape, dx, diffusion_rate, removal_rate, infectious_positions, mask):
    grid = y.reshape(grid_shape)
    
    # Diffusion coefficient
    alpha = diffusion_rate / dx**2

    # Define 5-point stencil kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # Apply convolution only where mask is True
    diffused = convolve(grid * mask, kernel, mode='constant', cval=0.0)

    # Compute change, apply mask to keep outside values zero
    dgrid_dt = alpha * diffused
    dgrid_dt -= removal_rate * grid
    dgrid_dt *= mask  # Zero out outside domain

    # Add quanta generation
    for x, y, rate in infectious_positions:
        if mask[x, y]:
            dgrid_dt[x, y] += rate

    return dgrid_dt.flatten()

def solve_diffusion2(grid, dx, diffusion_rate, removal_rate, infectious_positions, t_span, dt, mask):
    y0 = grid.flatten()
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    sol = solve_ivp(
        spatiotemporal_diffusion_ode2,
        t_span,
        y0,
        t_eval=t_eval,
        args=(grid.shape, dx, diffusion_rate, removal_rate, infectious_positions, mask),
        method='LSODA'
    )

    results = sol.y.T.reshape(-1, grid.shape[0], grid.shape[1])
    return sol.t, results



def prepare_be_solver(mask, dx, diffusion_rate, removal_rate, dt):
    """
    Prepare LU-factorized solver for backward Euler diffusion.

    Parameters
    ----------
    mask : 2D bool ndarray
        True where simulation is active.
    dx : float
        Cell size (m).
    diffusion_rate : float
        Eddy diffusion rate (m^2/s).
    removal_rate : float
        First-order removal rate (1/s).
    dt : float
        Timestep (s).

    Returns
    -------
    solver : callable
        Function that solves the reduced system.
    idx_map : 1D ndarray
        Maps full grid flat indices to reduced indices (or -1).
    keep : 1D ndarray
        Flat indices of active cells.
    """
    nrows, ncols = mask.shape
    N = nrows * ncols

    # Identify active cells
    keep = np.flatnonzero(mask.ravel())
    nk = len(keep)

    idx_map = -np.ones(N, dtype=int)
    idx_map[keep] = np.arange(nk)

    # Diffusion coefficient
    alpha = diffusion_rate / dx**2

    # Build sparse Laplacian for active cells (5-point stencil)
    data, rows, cols = [], [], []
    for full_idx in keep:
        r, c = divmod(full_idx, ncols)
        center = idx_map[full_idx]

        # Center
        rows.append(center)
        cols.append(center)
        data.append(-4 * alpha)

        # Neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols:
                neighbor_full_idx = rr * ncols + cc
                neighbor_reduced_idx = idx_map[neighbor_full_idx]
                if neighbor_reduced_idx >= 0:
                    rows.append(center)
                    cols.append(neighbor_reduced_idx)
                    data.append(alpha)

    K = sp.csr_matrix((data, (rows, cols)), shape=(nk, nk))

    # Backward Euler system matrix: (I - dt * operator)
    M = sp.eye(nk, format='csr') - dt * (K - removal_rate * sp.eye(nk, format='csr'))

    # LU factorization
    solver = spla.factorized(M)

    return solver, idx_map, keep

def solve_diffusion_be(grid, infectious_positions, t_span, dt,
                       solver, idx_map, keep, mask):
    """
    Backward-Euler timestepper for multiple calls without refactorizing.

    Parameters
    ----------
    grid : 2D ndarray
        Initial concentration grid.
    infectious_positions : list of (i_row, j_col, rate)
        Sources of quanta (rate in same units as grid/time).
    t_span : tuple (t0, t1)
        Time range for simulation.
    dt : float
        Time step size (same units as rates).
    solver : callable
        Pre-factorized solver from `prepare_be_solver`.
    idx_map : ndarray
        Maps full grid indices to reduced indices (from `prepare_be_solver`).
    keep : ndarray
        Flat indices of active cells (from `prepare_be_solver`).
    mask : 2D ndarray (bool)
        Active cell mask.

    Returns
    -------
    times : 1D ndarray
        Simulation times.
    results : 3D ndarray
        Concentration fields over time, shape = (nt, nrows, ncols)
    """
    nrows, ncols = grid.shape
    N = nrows * ncols

    # initial reduced state
    y_full = grid.ravel().astype(float)
    y_reduced = y_full[keep].copy()   # length nk

    t0, t1 = t_span
    times = np.arange(t0, t1 + dt, dt)
    results = []

    # Precompute reduced source positions
    src_reduced = np.zeros(len(keep), dtype=float)
    for (i_row, j_col, rate) in infectious_positions:
        if not (0 <= i_row < nrows and 0 <= j_col < ncols):
            continue
        full_idx = i_row * ncols + j_col
        reduced_idx = idx_map[full_idx]
        if reduced_idx >= 0:
            src_reduced[reduced_idx] += rate

    for t in times:
        # Save snapshot
        y_full[keep] = y_reduced
        results.append(y_full.reshape((nrows, ncols)).copy())

        # Backward-Euler RHS
        rhs = y_reduced + dt * src_reduced
        y_reduced = solver(rhs)  # uses pre-factorized LU

    return np.array(times), np.array(results)

