# Libraries
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import os
import time
from numba import njit
import argparse
import concurrent.futures
from functools import partial

# Dates
linked_tb_path = 'data-clean/tracking/linked-tb/'
dates = [
    file.replace('.csv', '') 
    for file in os.listdir(linked_tb_path) 
    if file.endswith('.csv')
]

# Command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run risk model simulation.")
    parser.add_argument('--date', type=str, required=True, help='Date  run')
    parser.add_argument('--cores', type=int, default=4, help='Number of cores to use for parallel processing of multiple dates.')
    return parser.parse_args()


# File paths
base_path = 'data-clean/tracking/'
save_path_tb = os.path.join(base_path, 'features/')

# Tracking data columns
dtypes = {
    'time': 'int64',       # since it's ms since epoch
    'track_id': 'int32',
    'position_x': 'float32',
    'position_y': 'float32', 
    'gender': 'string',
    'tag': 'string'
}

# Air exchange rates
aer_path = 'data-clean/environmental/air-exchange-rate.csv'
aer_df = pd.read_csv(aer_path)

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


def add_aer(result_df, date):
    aer_row = aer_df[(aer_df['date'] == date) & (aer_df['device'] == 'Aranet4 272D2')]
    aer_value = aer_row.iloc[0]['aer_tmb']
    result_df['aer'] = aer_value
    return result_df

def add_daytime(result_df, non_tb_df):
    # Output column names
    hour_labels = [f'hour_{h}' for h in range(6, 18)]  # hour_6 .. hour_17 (12 hours)

    df = non_tb_df[['new_track_id', 'time']].copy()

    # Compute hour index: 0→6–7am, 1→7–8am, ..., 11→5–6pm
    if np.issubdtype(df['time'].dtype, np.datetime64):
        # If 'time' is datetime, use hour-of-day and offset by 6
        hour_idx = df['time'].dt.hour.to_numpy() - 6
    else:
        # If 'time' is numeric seconds since 6am (0..43200), bucket by 3600s
        # (If your 'time' is seconds since midnight, use: (df['time'].to_numpy() - 6*3600) // 3600)
        hour_idx = (df['time'].to_numpy() // 3600).astype(np.int16)

    # Keep only rows within the 12-hour window [6am, 6pm)
    mask = (hour_idx >= 0) & (hour_idx < 12)
    if not np.any(mask):
        # No rows in window: just add zero columns
        for col in hour_labels:
            result_df[col] = 0
        return result_df

    # Reduce to unique (track_id, hour) pairs → guarantees binary per hour
    slim = pd.DataFrame({
        'new_track_id': df.loc[mask, 'new_track_id'].to_numpy(),
        'hour_idx': hour_idx[mask]
    }).drop_duplicates()

    # Make a wide binary matrix: rows=track_id, cols=hour_idx
    pivot = (
        slim.assign(val=1)
            .pivot(index='new_track_id', columns='hour_idx', values='val')
            .fillna(0)
            .astype('uint8')
            .reindex(columns=range(12), fill_value=0)  # ensure all 12 hours present
    )

    # Rename columns to hour_6..hour_17
    pivot.columns = hour_labels

    # Merge to result_df and fill missing with 0
    out = result_df.merge(pivot, left_on='new_track_id', right_index=True, how='left')
    out[hour_labels] = out[hour_labels].fillna(0).astype('uint8')

    return out

def add_person_features(result_df, non_tb_df):
    # --- Gender ---
    if non_tb_df['gender'].isna().all():
        # whole column missing → all NA
        result_df['female'] = pd.NA
    else:
        def majority_gender(genders):
            # drop NA
            genders = genders.dropna()
            # drop NOT_SURE
            genders = genders[genders.isin(['MALE', 'FEMALE'])]
            if genders.empty:
                return pd.NA
            # majority vote
            return genders.value_counts().idxmax()

        gender_map = non_tb_df.groupby('new_track_id')['gender'].apply(majority_gender)
        result_df = result_df.merge(gender_map.rename('gender'),
                                    left_on='new_track_id', right_index=True, how='left')
        result_df['female'] = result_df['gender'].map({'FEMALE': 1, 'MALE': 0}).astype('Int64')
        result_df.drop(columns=['gender'], inplace=True)

    # --- Tag / Healthcare worker ---
    if non_tb_df['gender'].isna().all(): # use gender because tag was still provided as False
        result_df['healthcare_worker'] = pd.NA
    else:
        def has_healthcare_worker(tags):
            tags = tags.dropna().astype(str)
            return 1 if (tags == 'True').any() else 0

        tag_map = non_tb_df.groupby('new_track_id')['tag'].apply(has_healthcare_worker)
        result_df = result_df.merge(tag_map.rename('healthcare_worker'),
                                    left_on='new_track_id', right_index=True, how='left')
        result_df['healthcare_worker'] = result_df['healthcare_worker'].astype('Int64')

    return result_df

def add_visit_time(result_df, non_tb_df):
    # Compute (max_time - min_time) per track_id and convert ms → seconds
    time_spent = (non_tb_df.groupby('new_track_id')['time'].max()
                   - non_tb_df.groupby('new_track_id')['time'].min())

    time_spent = time_spent.rename('time_spent_seconds')

    result_df = result_df.merge(time_spent, left_on='new_track_id', right_index=True, how='left')
    result_df['time_spent_seconds'] = result_df['time_spent_seconds'].fillna(0).astype(int)

    return result_df

@njit
def update_proximities(ids, coords, min_dist, proximity_seconds, contact_seconds, id_map):
    n = len(ids)
    seen = np.zeros(n, dtype=np.uint8)  # mark if this id had any close partner this timestamp
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_dist:
                tidx = id_map[ids[i]]
                pidx = id_map[ids[j]]
                seen[i] = 1
                seen[j] = 1
                contact_seconds[tidx, pidx] += 1
                contact_seconds[pidx, tidx] += 1
    # add at most one second for each participant in this timestamp
    for i in range(n):
        if seen[i]:
            tidx = id_map[ids[i]]
            proximity_seconds[tidx] += 1

def add_close_proximity(result_df, non_tb_df, min_dist=1.0, min_time=60):
    df = non_tb_df[['new_track_id', 'position_x', 'position_y', 'time']].copy()
    # only one observation per second per person
    df = (df.groupby(['time', 'new_track_id'], as_index=False)
        .agg(position_x=('position_x', 'mean'),
             position_y=('position_y', 'mean')))
    df.sort_values('time', inplace=True)

    unique_ids = df['new_track_id'].unique()
    n_ids = len(unique_ids)

    # Build mapping: track_id -> integer index
    # (Numba-friendly: use max_id+1 sized array for lookup)
    max_id = unique_ids.max()
    id_map = np.full(max_id + 1, -1, dtype=np.int32)
    for i, tid in enumerate(unique_ids):
        id_map[tid] = i

    # Results
    proximity_seconds = np.zeros(n_ids, dtype=np.int32)
    contact_seconds = np.zeros((n_ids, n_ids), dtype=np.int32)

    # presence time caps
    presence_times = (df.groupby('new_track_id')['time'].max()
     - df.groupby('new_track_id')['time'].min()).reindex(unique_ids).to_numpy(dtype=np.int32)


    # fast grouping by time
    times, idx_start = np.unique(df['time'].to_numpy(), return_index=True)
    idx_end = np.append(idx_start[1:], len(df))

    for start, end in zip(idx_start, idx_end):
        group = df.iloc[start:end]
        ids = group['new_track_id'].to_numpy().astype(np.int64)
        coords = group[['position_x', 'position_y']].to_numpy()
        if len(ids) < 2:
            continue
        update_proximities(ids, coords, min_dist, proximity_seconds, contact_seconds, id_map)

    # cap proximity time
    for i in range(n_ids):
        if proximity_seconds[i] > presence_times[i]:
            proximity_seconds[i] = presence_times[i]

    # count valid contacts
    contact_counts = (contact_seconds >= min_time).sum(axis=1)

    # wrap into Series
    prox_series = pd.Series(proximity_seconds, index=unique_ids,
                            name=f'close_proximity_time_{min_dist}m')
    contact_series = pd.Series(contact_counts, index=unique_ids,
                               name=f'close_proximity_contacts_{min_dist}m_{min_time}s')

    result_df = result_df.merge(prox_series, left_on='new_track_id', right_index=True, how='left')
    result_df = result_df.merge(contact_series, left_on='new_track_id', right_index=True, how='left')

    result_df.fillna({f'close_proximity_time_{min_dist}m': 0,
                      f'close_proximity_contacts_{min_dist}m_{min_time}s': 0}, inplace=True)

    result_df[f'close_proximity_time_{min_dist}m'] = result_df[f'close_proximity_time_{min_dist}m'].astype(int)
    result_df[f'close_proximity_contacts_{min_dist}m_{min_time}s'] = result_df[f'close_proximity_contacts_{min_dist}m_{min_time}s'].astype(int)

    return result_df

@njit
def accumulate_person_time(times, ids, counts, id_map, out):
    for i in range(len(times)):
        t = times[i]
        tidx = id_map[ids[i]]
        out[tidx] += counts[t]
    return

def add_person_time(result_df, non_tb_df):
    # unique IDs
    unique_ids = non_tb_df['new_track_id'].unique()
    n_ids = len(unique_ids)
    id_to_idx = {tid: i for i, tid in enumerate(unique_ids)}

    # map track_ids to integer indices
    ids = non_tb_df['new_track_id'].to_numpy()
    ids_idx = np.array([id_to_idx[i] for i in ids], dtype=np.int64)

    # map times to integer indices
    times_unique, times_idx = np.unique(non_tb_df['time'].to_numpy(), return_inverse=True)

    # counts per unique time
    counts = np.bincount(times_idx)

    # output array
    out = np.zeros(n_ids, dtype=np.int64)

    # JIT accumulation
    accumulate_person_time(times_idx, ids_idx, counts, np.arange(n_ids), out)

    # wrap back into Series
    person_time = pd.Series(out, index=unique_ids, name='person_time_seconds')

    result_df = result_df.merge(person_time, left_on='new_track_id', right_index=True, how='left')
    result_df['person_time_seconds'] = result_df['person_time_seconds'].fillna(0).astype(int)
    return result_df

def add_features(date):
    print(f'Processing date: {date}')
    tt = time.time()
    
    # Read tracking data
    t0 = time.time()
    df = read_linked_tracking_data(date)
    print(f'read_linked_tracking_data {time.time() - t0:.2f} seconds')
    
    # Filter non-TB patients
    non_tb_df = df[df['clinic_id'].isna()]
    
    # Resulting dataframe with features
    result_df = pd.DataFrame({'new_track_id': non_tb_df['new_track_id'].unique()})
    result_df['date'] = date
    
    # Add air exchange rate
    result_df = add_aer(result_df, date)
    
    # Add person features
    t0 = time.time()
    result_df = add_person_features(result_df, non_tb_df)
    print(f'add_person_features took {time.time() - t0:.2f} seconds')
    
    # Add daytime features
    t0 = time.time()
    result_df = add_daytime(result_df, non_tb_df)
    print(f'add_daytime took {time.time() - t0:.2f} seconds')

    # Add visit time
    t0 = time.time()
    result_df = add_visit_time(result_df, non_tb_df)
    print(f'add_visit_time took {time.time() - t0:.2f} seconds')

    # Add close proximity time and contact partners
    t0 = time.time()
    result_df = add_close_proximity(result_df, non_tb_df, min_dist=1.0, min_time=60)
    print(f'add_close_proximity_contacts took {time.time() - t0:.2f} seconds')

    # Add person-time
    t0 = time.time()
    result_df = add_person_time(result_df, non_tb_df)
    print(f'add_person_time took {time.time() - t0:.2f} seconds')
    
    # Save to CSV
    output_file = os.path.join(save_path_tb, f'{date}.csv')
    result_df.to_csv(output_file, index=False)
    print(f'Processed date {date} in {time.time() - tt:.2f} seconds')
    
if __name__ == "__main__":
    args = parse_args()
    if args.date == 'all':
        cores = getattr(args, 'cores', 4)
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            executor.map(add_features, dates)
    else:
        add_features(date=args.date)