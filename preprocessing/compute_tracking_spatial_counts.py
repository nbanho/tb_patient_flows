#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from read_tracking_linked_data import read_linked_tracking_data, pad_track_data
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

EXTENT_DEFAULT = (0.0, 51.0, -0.02, 14.214)  # (min_x, max_x, min_y, max_y)

def rasterize_person_time(date: str, extent, cell_size: float) -> np.ndarray:
    """
    Vectorized rasterization: each row inside extent contributes 0.5s (2 Hz sampling)
    to the cell it falls in. Returns a (num_cells_y, num_cells_x) float32 array of seconds.
    """
    
    # dimensions
    min_x, max_x, min_y, max_y = extent
    
    # number of columns (x) and rows (y)
    num_cells_x = int((max_x - min_x) / cell_size)
    num_cells_y = int((max_y - min_y) / cell_size)
    total_cells = num_cells_x * num_cells_y

    # Read linked tracking data
    df = read_linked_tracking_data(date)
    df = pad_track_data(df)

    # Filter to extent (half-open on the max side, like your original code)
    x = df["position_x"].to_numpy()
    y = df["position_y"].to_numpy()
    in_bounds = (x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y)
    if not np.any(in_bounds):
        # No data in the extent; return zeros
        return np.zeros((num_cells_y, num_cells_x), dtype=np.float32)

    x = x[in_bounds]
    y = y[in_bounds]

    # Compute cell indices (integer bins)
    x_idx = ((x - min_x) / cell_size).astype(np.int32)
    y_idx = ((y - min_y) / cell_size).astype(np.int32)

    # Linearize 2D indices for bincount
    lin_idx = y_idx * num_cells_x + x_idx

    # Each observation represents 0.5 seconds at 2 Hz
    weights = np.full(lin_idx.shape[0], 0.5, dtype=np.float32)

    # Sum person-time per cell
    accum = np.bincount(lin_idx, weights=weights, minlength=total_cells).astype(np.float32)

    # Reshape back to (rows, cols) i.e., (y, x)
    return accum.reshape((num_cells_y, num_cells_x))


def process_one(date: str, extent, cell_size: float):
    counts = rasterize_person_time(date, extent, cell_size)
    return date, counts


def main():
    # parser arguments
    parser = argparse.ArgumentParser(description="Compute per-cell person time (seconds) from 2 Hz tracks.")
    parser.add_argument("--output", default="data-clean/tracking/spatial-density.h5", help="Output HDF5 file path")
    parser.add_argument("--cellsize", type=float, default=0.25, help="Cell size in same units as positions")
    parser.add_argument("--cores", type=int, default=os.cpu_count(), help="Number of worker processes")
    parser.add_argument("--extent", nargs=4, type=float, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
                        default=EXTENT_DEFAULT, help="Spatial extent (min_x max_x min_y max_y)")
    args = parser.parse_args()
    extent = tuple(args.extent)
    cell_size = args.cellsize
    
    # study dates
    linked_tb_path = 'data-clean/tracking/linked-tb/'
    dates = [
        file.replace('.csv', '') 
        for file in os.listdir(linked_tb_path) 
        if file.endswith('.csv')
    ]

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.cores) as ex:
        fut2file = {ex.submit(process_one, d, extent, cell_size): d for d in dates}
        for fut in as_completed(fut2file):
            date, counts = fut.result()
            results.append((date, counts))

    # Save to HDF5 (one dataset per date). Values are person-seconds per cell (float32).
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.output, "w") as hf:
        # Stash some metadata for reproducibility
        hf.attrs["extent"] = np.asarray(extent, dtype=np.float32)
        hf.attrs["cellsize"] = np.float32(cell_size)
        hf.attrs["units"] = "seconds"
        for date, counts in results:
            hf.create_dataset(date, data=counts, compression="gzip")

    print(f"Combined person-time grids saved to {args.output}")


if __name__ == "__main__":
    main()
