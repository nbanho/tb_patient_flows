import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from multiprocessing import Pool

def compute_occupancy(date):
    """
    Computes the number of close contacts for each track_id in the dataframe.
    
    Parameters:
    date (str): Date in the format '%Y-%m-%d' to identify the file.
    
    Returns:
    pd.DataFrame: DataFrame with 'track_id', 'time_minute', and 'track_id_count' columns.
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
    
    # Round time to the nearest second
    df['time_second'] = df['time'].dt.floor('S')
    
    # Count the number of unique track_id per second
    second_counts = df.groupby('time_second')['track_id'].nunique().reset_index()
    second_counts.rename(columns={'track_id': 'track_id_count'}, inplace=True)
    
    # Round time to the nearest minute
    second_counts['time_minute'] = second_counts['time_second'].dt.floor('T')
    
    # Compute the average count per minute
    track_id_counts = second_counts.groupby('time_minute')['track_id_count'].mean().reset_index()
    
    return track_id_counts
    

def process_file(file):
    date = file.replace('.csv', '')
    return compute_occupancy(date)

if __name__ == '__main__':
    base_path = 'data-clean/tracking/unlinked'
    output_file = 'data-clean/tracking/occupancy.csv'

    # List all files in the unlinked directory
    files = [f for f in os.listdir(base_path) if f.endswith('.csv')]

    # Use multiprocessing to process files in parallel
    with Pool(8) as pool:
        all_data = pool.map(process_file, files)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False, header=True)