"""Finite-difference solver for 2D diffusion with first-order removal.

Implements a backward-Euler scheme on a masked 2D grid using LU-factorized
sparse linear algebra. The 5-point Laplacian stencil approximates isotropic
eddy diffusion of airborne quanta, and a first-order term represents combined
removal by ventilation, settling, and inactivation.

Consumed by run_model.py to simulate spatiotemporal quanta concentrations.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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

    # Discretized diffusion coefficient: D / dx^2 (1/s)
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

    # Backward Euler system matrix: M = I - dt*(K - lambda*I)
    # where K is the Laplacian and lambda is the first-order removal rate.
    # Solving M * y_{n+1} = y_n + dt*S gives the implicit time step.
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

        # Backward-Euler RHS: y_n + dt * S (source term)
        rhs = y_reduced + dt * src_reduced
        y_reduced = solver(rhs)  # uses pre-factorized LU

    return np.array(times), np.array(results)

