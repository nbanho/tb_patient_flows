import pandas as pd
import os

# List all date.csv files in the folder
base_path = 'data-clean/tracking/unlinked'
files = [f for f in os.listdir(base_path) if f.endswith('.csv')]

# Initialize an empty list to store DataFrames
all_data = []

# Process each file
for file in files:
    date = file.replace('.csv', '')
    
    # Construct file paths
    unlinked_file = os.path.join(base_path, f'{date}.csv')
    linked_file = os.path.join(base_path, '../linked', f'{date}.csv')
    linked_tb_file = os.path.join(base_path, '../linked-tb', f'{date}.csv')
    
    # Read the unlinked data
    df = pd.read_csv(unlinked_file, usecols=['time', 'track_id'])
    
    # Check if the linked-tb file exists and merge if it does
    if os.path.exists(linked_tb_file):
        linked_tb_df = pd.read_csv(linked_tb_file)
        df = pd.merge(df, linked_tb_df, on='track_id')
        df['tb_patient'] = df['category'].apply(lambda x: 1 if x in ['sure', 'sputum only'] else 0)
    else: # otherwise merge automatic links
        linked_df = pd.read_csv(linked_file)
        linked_df.rename(columns={'track_id': 'new_track_id', 'raw_track_id': 'track_id'}, inplace=True)
        df = pd.merge(df, linked_df, on='track_id')
        df['tb_patient'] = float('nan')
    
    # Add the date column
    df['date'] = date
    
    # Select relevant columns
    df = df[['date', 'new_track_id', 'tb_patient']]
    df.rename(columns={'new_track_id': 'track_id'}, inplace=True)
    
    # Select the first row per date and track_id
    df_first_row = df.groupby(['date', 'track_id']).first().reset_index()
    
    # Append to the list of DataFrames
    all_data.append(df_first_row)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Save the combined DataFrame to a CSV file
output_file = 'data-clean/mapping_clinical_tracking.csv'
combined_df.to_csv(output_file, index=False, header=True)