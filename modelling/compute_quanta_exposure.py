import numpy as np

def compute_exposure(
    df_id,
    quanta,
    V,
    ihr_wait,
    ihr_walk,
    n_active,
    *,
    steps_per_hour=3600,
    dt_seconds=1.0,
    track_id=None,
):
    """
    Compute hourly inhaled quanta for a single individual and return rows ready for a DataFrame.

    Returns a list of tuples:
        (new_track_id, hour_idx, T_hour, Q_diffu_hour, Q_mixed_hour)

    Only hours where the individual is present at least once are returned.

    Parameters
    ----------
    df_id : pd.DataFrame
        Must contain columns: 'time' (int), 'x_i' (int), 'y_k' (int), 'is_walking' (0/1).
        If `track_id` is None, the function tries to read the individual's id from
        df_id['new_track_id'].iloc[0] (if present).
    quanta : np.ndarray
        3D array [time, y, x] with quanta concentration per simulation step.
    V : float
        Room volume (m^3). Must be > 0.
    ihr_wait : float
        Inhalation rate when waiting (m^3/s).
    ihr_walk : float
        Inhalation rate when walking (m^3/s).
    n_active : int
        Number of active cells for the well-mixed baseline. Must be > 0.
    steps_per_hour : int, optional
        Simulation steps per hour (default 360; e.g., if one step = 10 s).
    dt_seconds : float, optional
        Duration of one simulation step in seconds. Set to your step size (e.g., 10.0).
        If your previous totals intentionally omitted step duration, leave as 1.0.
    track_id : any, optional
        The individual's identifier to attach to each row. If None, the function
        uses df_id['new_track_id'].iloc[0] if available; otherwise uses None.

    Returns
    -------
    list[tuple]
        Each tuple is (new_track_id, hour_idx, T_hour, Q_diffu_hour, Q_mixed_hour).
    """
    if V <= 0:
        raise ValueError("V must be > 0")
    if n_active <= 0:
        raise ValueError("n_active must be > 0")
    if len(df_id) == 0:
        return []

    # Pull id if not provided
    if track_id is None:
        if 'new_track_id' in df_id.columns and len(df_id['new_track_id']) > 0:
            track_id = df_id['new_track_id'].iloc[0]
        else:
            track_id = None

    # Indices & activity
    t_idx = df_id['time'].to_numpy(dtype=int, copy=False)
    x_idx = df_id['x_i'].to_numpy(dtype=int, copy=False)
    y_idx = df_id['y_k'].to_numpy(dtype=int, copy=False)
    is_walking = df_id['is_walking'].to_numpy(dtype=np.int8, copy=False)

    # Hour bin for each observation
    hour_idx = t_idx // steps_per_hour

    # Per-row inhalation rate
    ihr = np.where(is_walking == 1, ihr_walk, ihr_wait)

    results = []
    # Iterate unique hours present
    for h in np.unique(hour_idx):
        mask = (hour_idx == h)
        if not np.any(mask):
            continue

        t_h = t_idx[mask]
        x_h = x_idx[mask]
        y_h = y_idx[mask]
        ihr_h = ihr[mask]

        # Spatiotemporal (diffusive) concentration at the person's locations
        c_diffu_h = quanta[t_h, y_h, x_h]
        # Clamp negatives to zero
        c_diffu_h = np.maximum(c_diffu_h, 0.0)

        # Well-mixed concentration at those times (mean over active cells)
        c_all_h = quanta[t_h, :, :]
        c_all_h = np.maximum(c_all_h, 0.0)
        c_mixed_h = np.sum(c_all_h, axis=(1, 2)) / float(n_active)

        # Integrate over steps present in this hour
        Q_diffu_hour = float(np.sum(c_diffu_h * ihr_h / V) * dt_seconds)
        Q_mixed_hour = float(np.sum(c_mixed_h * ihr_h / V) * dt_seconds)

        T_hour = int(mask.sum())  # number of steps present in this hour

        if T_hour > 0:
            results.append((track_id, int(h), T_hour, Q_diffu_hour, Q_mixed_hour))

    return results
