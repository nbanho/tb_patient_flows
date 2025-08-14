from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import os
from numba import njit
import time
from scipy.spatial import cKDTree

# Saving data
base_path = 'data-clean/tracking/'
save_path_tb = os.path.join(base_path, 'tb-positions/')
save_path_non_tb = os.path.join(base_path, 'non-tb-positions/')
def save_large_csv(df, filepath, chunk_size=500_000):
    # Always overwrite the file (mode='w') for the first chunk
    df.iloc[:0].to_csv(filepath, index=False)
    for i, start in enumerate(range(0, len(df), chunk_size)):
        mode = 'w' if i == 0 else 'a'
        header = (i == 0)
        df.iloc[start:start+chunk_size].to_csv(filepath, index=False, header=header, mode=mode)

# Study dates
linked_tb_path = 'data-clean/tracking/linked-tb/'
dates = [
    file.replace('.csv', '') 
    for file in os.listdir(linked_tb_path) 
    if file.endswith('.csv')
]
# # Exclude dates where both output files already exist
# dates = [
#     date for date in dates
#     if not (
#         os.path.exists(os.path.join(save_path_tb, f'{date}.csv')) and
#         os.path.exists(os.path.join(save_path_non_tb, f'{date}.csv'))
#     )
# ]
# dates = dates[0:1]

# Tracking data
dtypes = {
    'time': 'int64',       # since it's ms since epoch
    'track_id': 'int32',
    'position_x': 'float32',
    'position_y': 'float32'
}
start_seconds = 6 * 3600  # 06:00:00 in seconds
def read_linked_tracking_data(date):
    unlinked_file = os.path.join(base_path, 'unlinked', f'{date}.csv')
    linked_file = os.path.join(base_path, 'linked-clinical', f'{date}.csv')
    df = pd.read_csv(unlinked_file, usecols=list(dtypes.keys()), dtype=dtypes)
    linked_df = pd.read_csv(linked_file, dtype={'raw_track_id': 'int32', 'new_track_id': 'int32'})
    df.rename(columns={'track_id': 'raw_track_id'}, inplace=True)
    df.sort_values('raw_track_id', inplace=True)
    linked_df.sort_values('raw_track_id', inplace=True)
    df = pd.merge(df, linked_df, on='raw_track_id', how='inner', sort=False)
    ts = pd.to_datetime(df['time'], unit='ms')
    seconds_since_midnight = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    start_seconds = 6 * 3600
    df['time_int'] = (seconds_since_midnight - start_seconds).astype(int)
    df.drop(columns=['time'], inplace=True)
    df.rename(columns={'time_int': 'time'}, inplace=True)
    return df

# Grid coordinates
mask = np.load('data-clean/building/building-grid-mask.npy')
valid_indices = np.argwhere(mask)
valid_positions = np.load('data-clean/building/building-grid-mask-valid-positions.npy')
tree = cKDTree(valid_positions)
def get_cell_indices_batch(x_points, y_points):
    dists, nearest_idx = tree.query(np.column_stack((x_points, y_points)), k=1)
    nearest_cells = valid_indices[nearest_idx]
    x_idx = nearest_cells[:, 1]
    y_idx = nearest_cells[:, 0]
    return x_idx, y_idx

# Determine activity status using Numba for performance
@njit
def compute_is_walking(coords, track_ids, threshold=0.25):
    n = coords.shape[0]
    is_walk = np.zeros(n, dtype=np.int8)
    thr_sq = threshold * threshold
    for i in range(1, n):
        if track_ids[i] == track_ids[i - 1]:
            dx = coords[i, 0] - coords[i - 1, 0]
            dy = coords[i, 1] - coords[i - 1, 1]
            if dx * dx + dy * dy > thr_sq:
                is_walk[i] = 1
            else:
                is_walk[i] = 0
        else:
            dx = coords[i, 0] - coords[i - 1, 0]
            dy = coords[i, 1] - coords[i - 1, 1]
            is_walk[i] = 1 if (dx * dx + dy * dy) > thr_sq else 0
    if n > 1:
        is_walk[0] = is_walk[1]
    return is_walk


# Aggregate TB patient positions
n_seconds = 12 * 60 * 60 + 1  # 0..12h inclusive
positions_array = [[] for _ in range(n_seconds)]
@njit
def find_change_points(arr):
    n = len(arr)
    change_points = [0]
    for i in range(1, n):
        if arr[i] != arr[i - 1]:
            change_points.append(i)
    change_points.append(n)
    return np.array(change_points, dtype=np.int64)


def process_date(date):
    # Load and merge data and compute time integer
    start_comp_time = time.time()
    df = read_linked_tracking_data(date)
    print(f"Read data for {date} in {time.time() - start_comp_time:.2f} seconds")
    
    # Determine cell indices
    start_comp_time = time.time()
    df['x_i'], df['y_k'] = get_cell_indices_batch(df['position_x'].to_numpy(), df['position_y'].to_numpy())
    print(f"Determined grid cells for {date} in {time.time() - start_comp_time:.2f} seconds")
    
    # Determine activity
    start_comp_time = time.time()
    coords = df[['position_x', 'position_y']].to_numpy()
    track_ids = df['new_track_id'].to_numpy()
    df['is_walking'] = compute_is_walking(coords, track_ids, threshold=0.25)
    print(f"Determined activity {date} in {time.time() - start_comp_time:.2f} seconds")
    
    # Subset non-tb patients
    start_comp_time = time.time()
    non_tb_df = df[df['clinic_id'].isna()]
    non_tb_df = non_tb_df[['time', 'new_track_id', 'x_i', 'y_k', 'is_walking']]
    non_tb_output_file = os.path.join(save_path_non_tb, f'{date}.csv')
    save_large_csv(non_tb_df, non_tb_output_file)
    print(f"Saved non-TB track positions for {date} took {time.time() - start_comp_time:.2f} seconds")

    # Subset tb patients
    start_comp_time = time.time()
    df = df[~df['clinic_id'].isna()]
    
    # Group once and fill positions
    for t, g in df.groupby('time', sort=False):
        tuples = [tuple(map(int, row)) for row in g[['new_track_id', 'x_i', 'y_k', 'is_walking']].to_numpy()]
        positions_array[t] = tuples

    # Build df_list
    df_list = pd.DataFrame({
        'time': np.arange(n_seconds, dtype=int),
        'positions': positions_array,
        'dt': 1
    })

    # Change point detection
    positions_tuple = [tuple(sorted(p)) for p in df_list['positions']]
    pos_str = np.array([str(p) for p in positions_tuple], dtype=np.unicode_)
    change_idx = find_change_points(pos_str)  

    # Aggregate consecutive identical positions
    out = pd.DataFrame({
        'time': df_list['time'].iloc[change_idx[:-1]].astype(int).to_numpy(),
        'positions': [df_list['positions'][start] for start in change_idx[:-1]],
        'dt': np.diff(change_idx)
    })
    df_list = out.reset_index(drop=True)
    print(f"Reshaped data for TB track positions for {date} took {time.time() - start_comp_time:.2f} seconds")

    # Save the resulting df_list to a CSV file
    start_comp_time = time.time()
    output_file = os.path.join(save_path_tb, f'{date}.csv')
    save_large_csv(df_list, output_file)
    print(f"Saved TB track positions for {date} took {time.time() - start_comp_time:.2f} seconds")

# Run the for loop in parallel
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_date, dates)