
"""
tracking_utils.py
------------------
Utilities for reading linked tracking data and padding missing per-second timesteps.

Functions
---------
- read_linked_tracking_data(date, base_path=..., start_hour=6)
    Read, merge, and aggregate tracking data for a specific date.
- pad_track_data(df)
    Vectorized padding of missing seconds per track with forward-filled positions.
"""
from __future__ import annotations

import os
from typing import Optional
import numpy as np
import pandas as pd
import time


__all__ = ["read_linked_tracking_data", "pad_track_data"]


def read_linked_tracking_data(date: str, *, base_path: str = "data-clean/tracking/", start_hour: int = 6) -> pd.DataFrame:
    """
    Read and merge *linked* tracking data for a given date, then aggregate to one
    row per (time, new_track_id). Designed to feed directly into `pad_track_data`.

    Parameters
    ----------
    date : str
        Date string matching the filenames in the tracking folders, e.g. "2025-10-17".
    base_path : str, optional (keyword-only)
        Root directory that contains the "unlinked" and "linked-clinical" subfolders.
        Defaults to "data-clean/tracking/".
    start_hour : int, optional (keyword-only)
        Hour-of-day used as the zero point for the returned `time` values (in seconds).
        For example, with start_hour=6 (default), 06:00:00 is time==0, 06:00:01 is 1, etc.

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with columns:
            - time (int64): seconds since {start_hour}:00:00 (can skip seconds if gaps exist)
            - new_track_id (int32): the linked track identifier
            - position_x (float32): mean x at that second for the linked track
            - position_y (float32): mean y at that second for the linked track
            - clinic_id (object or numeric): first clinic_id observed for that second (if present)

    Notes
    -----
    - This function **does not** fill missing seconds. Use `pad_track_data(df)` afterward
      to create continuous per-second rows by forward-filling positions across gaps.
    - Input files:
        * {base_path}/unlinked/{date}.csv
            expected columns: ["time","track_id","position_x","position_y"]
            where "time" is epoch milliseconds.
        * {base_path}/linked-clinical/{date}.csv
            expected columns include ["raw_track_id","new_track_id"] and optionally "clinic_id".
    - All casting is done to memory-friendly dtypes (int32/float32 where appropriate).

    Examples
    --------
    >>> df_agg = read_linked_tracking_data("2025-10-17")
    >>> df_full = pad_track_data(df_agg)  # fill missing seconds per track
    """
    t0 = time.time()
    
    # Column dtypes for the unlinked input file
    dtypes_unlinked = {
        "time": "int64",       # epoch milliseconds
        "track_id": "int32",
        "position_x": "float32",
        "position_y": "float32",
    }

    # Resolve file paths
    unlinked_file = os.path.join(base_path, "unlinked", f"{date}.csv")
    linked_file   = os.path.join(base_path, "linked-clinical", f"{date}.csv")

    # Read inputs
    df = pd.read_csv(unlinked_file, usecols=list(dtypes_unlinked.keys()), dtype=dtypes_unlinked)
    linked_df = pd.read_csv(linked_file, dtype={"raw_track_id": "int32", "new_track_id": "int32"})

    # Merge "new_track_id" (and potentially "clinic_id") into the unlinked records
    df = df.rename(columns={"track_id": "raw_track_id"}).sort_values("raw_track_id")
    linked_df = linked_df.sort_values("raw_track_id")
    df = pd.merge(df, linked_df, on="raw_track_id", how="inner", sort=False)

    # Convert epoch ms -> seconds since `start_hour`:00:00
    ts = pd.to_datetime(df["time"], unit="ms")
    seconds_since_midnight = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    start_seconds = int(start_hour) * 3600
    df["time"] = (seconds_since_midnight - start_seconds).astype("int64")

    # Drop raw_track_id (no longer needed)
    df.drop(columns=["raw_track_id"], inplace=True, errors="ignore")

    # Aggregate to one row per (time, new_track_id)
    df = (
        df.groupby(["time", "new_track_id"], as_index=False)
          .agg(
              position_x=("position_x", "mean"),
              position_y=("position_y", "mean"),
              clinic_id =("clinic_id",  "first"),
          )
    )

    # Ensure compact dtypes
    df["new_track_id"] = df["new_track_id"].astype("int32")
    df["time"] = df["time"].astype("int64")
    df[["position_x", "position_y"]] = df[["position_x", "position_y"]].astype("float32")

    # Sort consistently
    df = df.sort_values(["new_track_id", "time"], kind="mergesort").reset_index(drop=True)
    
    print(f'read_linked_tracking_data {time.time() - t0:.2f} seconds')
    
    return df


def pad_track_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pad missing 1-second timesteps for each `new_track_id` by forward-filling
    positions (and clinic_id) from the previous timestep. Highly optimized,
    vectorized implementation with no per-group Python loops.

    Parameters
    ----------
    df : pandas.DataFrame
        The aggregated DataFrame produced by `read_linked_tracking_data` with
        at least the following columns:
            - "new_track_id" (int32)
            - "time" (int64): seconds since reference start
            - "position_x" (float32)
            - "position_y" (float32)
        Optionally:
            - "clinic_id" (any dtype)

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing *all* seconds from each track's first to last
        observed second. Newly created rows have positions (and clinic_id) 
        copied forward from the previous observed second in the same track.

    Performance
    -----------
    - Builds all per-track ranges via NumPy (`np.arange`) and concatenates once.
    - One merge and one grouped forward-fill over the full table.
    - Suitable for large datasets; avoids `.groupby().apply()` per track.

    Examples
    --------
    >>> df_agg = read_linked_tracking_data("2025-10-17")
    >>> df_full = pad_track_data(df_agg)
    """
    t0 = time.time()
    
    if df.empty:
        return df.copy()

    required_cols = {"new_track_id", "time", "position_x", "position_y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"pad_track_data: missing required columns: {sorted(missing)}")

    # Ensure expected ordering before constructing ranges
    df = df.sort_values(["new_track_id", "time"], kind="mergesort").reset_index(drop=True)

    # Compute time spans per track
    grp = df.groupby("new_track_id", sort=False)["time"]
    min_time = grp.min()
    max_time = grp.max()

    # Pre-allocate complete ranges (vectorized; no Python loops over rows)
    lengths = (max_time - min_time + 1).to_numpy(dtype=np.int64)
    # Repeat each track id by its span length
    full_track_ids = np.repeat(min_time.index.to_numpy(), lengths)
    # Create the concatenated time vector by adding offsets per block
    # Build per-track ranges and then concatenate once
    full_times = np.concatenate([np.arange(start, end + 1, dtype=np.int64)
                                 for start, end in zip(min_time.to_numpy(), max_time.to_numpy())])

    full_df = pd.DataFrame({"new_track_id": full_track_ids, "time": full_times})

    # Merge original observations; missing rows will be NaN and then ffilled
    df_full = full_df.merge(df, on=["new_track_id", "time"], how="left", sort=False)

    # Forward fill per track for position columns (and clinic_id if present)
    cols_to_ffill = ["position_x", "position_y"] + (["clinic_id"] if "clinic_id" in df_full.columns else [])
    df_full[cols_to_ffill] = (
        df_full.groupby("new_track_id", sort=False)[cols_to_ffill]
               .ffill()
    )

    # Cast back to compact dtypes where appropriate
    df_full["new_track_id"] = df_full["new_track_id"].astype("int32", copy=False)
    df_full["time"] = df_full["time"].astype("int64", copy=False)
    df_full[["position_x", "position_y"]] = df_full[["position_x", "position_y"]].astype("float32", copy=False)
    
    print(f'pad_track_data {time.time() - t0:.2f} seconds')
    
    return df_full
