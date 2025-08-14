import numpy as np

def compute_risk(df_id, quanta, V, ihr_wait, ihr_walk, n_active):
    """
    Computes the risk of infection for a patient based on their spatiotemporal positions,
    quanta concentration, and activity status.

    Parameters:
        df_id (pd.DataFrame): DataFrame containing patient positions and activity.
        quanta (np.ndarray): 3D array of quanta concentration [time, y, x].
        V (float): Room volume (m^3).
        ihr_wait (float): Inhalation rate when waiting (m^3/s).
        ihr_walk (float): Inhalation rate when walking (m^3/s).
        n_active (int): Number of active cells in the room.

    Returns:
        T (int): Duration of exposure (number of time steps).
        Q_diffu (float): Total inhaled quanta (spatiotemporal).
        P_diffu (float): Risk of infection (spatiotemporal).
        Q_mixed (float): Total inhaled quanta (well-mixed).
        P_mixed (float): Risk of infection (well-mixed).
    """
    # Spatiotemporal positions of the patient
    t_idx = (df_id['time'].values).astype(int)
    x_idx = df_id['x_i'].values.astype(int)
    y_idx = df_id['y_k'].values.astype(int)
    is_walking = df_id['is_walking'].values

    # Quanta concentration at the positions
    c_diffu = quanta[t_idx, y_idx, x_idx]
    c_diffu = np.maximum(c_diffu, 0)
    ihr = np.where(is_walking == 1, ihr_walk, ihr_wait)
    Q_diffu = np.sum(c_diffu * ihr / V)

    # Quanta concentration if well-mixed
    c_mixed = quanta[t_idx, :, :]
    c_mixed = np.maximum(c_mixed, 0)
    c_mixed = np.sum(c_mixed, axis=(1, 2)) / n_active
    Q_mixed = np.sum(c_mixed * ihr / V)

    # Duration of exposure
    T = len(df_id)

    # Risks of infection
    P_diffu = 1 - np.exp(-Q_diffu)
    P_mixed = 1 - np.exp(-Q_mixed)

    # Return the results
    return T, Q_diffu, P_diffu, Q_mixed, P_mixed