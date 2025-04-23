import pandas as pd
import os
from datetime import datetime

def compute_total_time_in_clinic(date):
    """
    Computes the total time spent in the clinic for each track_id in the dataframe.
    
    Parameters:
    date (str): Date in the format '%Y-%m-%d' to identify the file.
                       
    Returns:
    pd.DataFrame: DataFrame with 'track_id' and 'total_time_in_clinic' columns.
    """
    
    # Construct file paths
    base_path = 'data-clean/tracking/'
    unlinked_file = os.path.join(base_path, 'unlinked', f'{date}.csv')
    linked_file = os.path.join(base_path, 'linked', f'{date}.csv')
    linked_tb_file = os.path.join(base_path, 'linked-tb', f'{date}.csv')
    
    # Read the unlinked data
    df = pd.read_csv(unlinked_file, usecols=['time', 'track_id', 'position_x', 'position_y'])
    
    # Check if the linked-tb file exists and merge if it does
    if os.path.exists(linked_tb_file):
        linked_tb_df = pd.read_csv(linked_tb_file)
        df = pd.merge(df, linked_tb_df, on='track_id')
    else: # otherwise merge automatic links
        linked_df = pd.read_csv(linked_file)
        linked_df.rename(columns={'track_id': 'new_track_id', 'raw_track_id': 'track_id'}, inplace=True)
        df = pd.merge(df, linked_df, on='track_id')
    
    # Drop the old track_id column and rename new_track_id to track_id
    df.drop(columns=['track_id'], inplace=True)
    df.rename(columns={'new_track_id': 'track_id'}, inplace=True)
    
    # Convert unix timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    
    # Compute the total time spent in the clinic for each track_id
    total_time = df.groupby('track_id')['time'].agg(lambda x: (x.max() - x.min()).total_seconds())
    
    # Convert the result to a DataFrame
    result_df = total_time.reset_index()
    result_df.columns = ['track_id', 'total_time_in_clinic']
    result_df['date'] = date
    return result_df

def process_all_dates():
    """
    Processes all dates in the unlinked directory and combines the results into a single DataFrame.
    Stores the combined DataFrame in 'data-clean/tracking/time-in-clinic.csv'.
    """
    
    base_path = 'data-clean/tracking/unlinked'
    output_file = 'data-clean/tracking/time-in-clinic.csv'
    
    # List all files in the unlinked directory
    files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
    
    # Initialize an empty list to store DataFrames
    all_data = []
    
    # Process each file
    for file in files:
        date = file.replace('.csv', '')
        df = compute_total_time_in_clinic(date)
        all_data.append(df)
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Store the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False, header=True)

process_all_dates()