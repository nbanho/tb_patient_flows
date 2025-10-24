import pandas as pd
import os
import time
from read_tracking_linked_data import read_linked_tracking_data

def compute_duration(date):
    """
    Computes the total time spent in the clinic for each track_id in the dataframe.
    
    Parameters:
    date (str): Date in the format '%Y-%m-%d' to identify the file.
                       
    Returns:
    pd.DataFrame: DataFrame with 'track_id' and 'total_time_in_clinic' columns.
    """
    
    # Read data
    df = read_linked_tracking_data(date)
    
    # Compute the total time spent in the clinic for each track_id
    total_time = df.groupby('new_track_id')['time'].agg(lambda x: x.max() - x.min())
    
    # Convert the result to a DataFrame
    result_df = total_time.reset_index()
    result_df.columns = ['new_track_id', 'duration']
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
        t0 = time.time()
        df = compute_duration(date)
        print(f'compute_duration {time.time() - t0:.2f} seconds')
        all_data.append(df)
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Store the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False, header=True)

process_all_dates()