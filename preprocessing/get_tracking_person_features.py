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
save_path_tb = os.path.join(base_path, 'person-features/')

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
    
    # Add person features
    t0 = time.time()
    result_df = add_person_features(result_df, non_tb_df)
    print(f'add_person_features took {time.time() - t0:.2f} seconds')
    
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