import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from scipy.spatial import cKDTree
from multiprocessing import Pool

def compute_close_contacts(date, cc_dist, cc_time, min_time):
    """
    Computes the number of close contacts for each track_id in the dataframe.
    
    Parameters:
    date (str): Date in the format '%Y-%m-%d' to identify the file.
    cc_dist (float): Distance threshold for close contact
    cc_time (int): Time threshold in seconds for close contact
    min_time (int): Minimum time in seconds that a track_id must have spent in the clinic to be considered.
                       
    Returns:
    pd.DataFrame: DataFrame with 'track_id', 'close_contacts', and 'other_contacts' columns.
    """
    
    # Construct file paths
    base_path = 'data-clean/tracking/'
    unlinked_file = os.path.join(base_path, 'unlinked', f'{date}.csv')
    linked_file = os.path.join(base_path, 'linked', f'{date}.csv')
    
    # Read the unlinked data
    df = pd.read_csv(unlinked_file, usecols=['time', 'track_id', 'position_x', 'position_y'])
    
    # Check if the linked-tb file exists and merge if it does
    linked_df = pd.read_csv(linked_file)
    linked_df.rename(columns={'track_id': 'new_track_id', 'raw_track_id': 'track_id'}, inplace=True)
    df = pd.merge(df, linked_df, on='track_id')
    
    # Drop the old track_id column and rename new_track_id to track_id
    df.drop(columns=['track_id'], inplace=True)
    df.rename(columns={'new_track_id': 'track_id'}, inplace=True)
    
    # Convert unix timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    
    # Filter track_ids based on min_time
    time_spent = df.groupby('track_id')['time'].agg(lambda x: (x.max() - x.min()).total_seconds())
    valid_track_ids = time_spent[time_spent >= min_time].index
    df = df[df['track_id'].isin(valid_track_ids)]
    
    # Initialize dictionaries to store the number of close contacts and other contacts for each track_id
    close_contacts = {track_id: 0 for track_id in df['track_id'].unique()}
    other_contacts = {track_id: set() for track_id in df['track_id'].unique()}
    
    # Dictionary to store cumulative contact times
    cumulative_contact_times = {track_id: {} for track_id in df['track_id'].unique()}
    
    # Group by time to process each timestamp separately
    grouped_by_time = df.groupby('time')
    
    for time, group in grouped_by_time:
        positions = group[['position_x', 'position_y']].values
        track_ids = group['track_id'].values
        
        # Use KD-Tree for efficient spatial queries
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=cc_dist)  # Find all pairs within cc_dist meters
        
        for i, j in pairs:
            track_id_i = track_ids[i]
            track_id_j = track_ids[j]
            
            if track_id_j not in cumulative_contact_times[track_id_i]:
                cumulative_contact_times[track_id_i][track_id_j] = 0
            if track_id_i not in cumulative_contact_times[track_id_j]:
                cumulative_contact_times[track_id_j][track_id_i] = 0
            
            cumulative_contact_times[track_id_i][track_id_j] += 1
            cumulative_contact_times[track_id_j][track_id_i] += 1
        
        # Update other contacts
        for track_id in track_ids:
            other_contacts[track_id].update(track_ids)
            other_contacts[track_id].remove(track_id)
    
    # Update close contacts if cumulative time exceeds cc_time
    for track_id, contacts in cumulative_contact_times.items():
        for other_track_id, time_count in contacts.items():
            if time_count >= cc_time:
                close_contacts[track_id] += 1
    
    # Calculate other contacts
    for track_id in other_contacts:
        other_contacts[track_id] = len(other_contacts[track_id]) - close_contacts[track_id]
    
    # Convert the result to a DataFrame
    result_df = pd.DataFrame(list(close_contacts.items()), columns=['track_id', 'close_contacts'])
    result_df['other_contacts'] = result_df['track_id'].map(other_contacts)
    result_df['date'] = date
    
    return result_df

def process_file(file):
    date = file.replace('.csv', '')
    return compute_close_contacts(date, cc_dist=1.0, cc_time=60, min_time=60)

if __name__ == '__main__':
    base_path = 'data-clean/tracking/unlinked'
    output_file = 'data-clean/tracking/close-contacts.csv'

    # List all files in the unlinked directory
    files = [f for f in os.listdir(base_path) if f.endswith('.csv')]

    # Use multiprocessing to process files in parallel
    with Pool(8) as pool:
        all_data = pool.map(process_file, files)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False, header=True)