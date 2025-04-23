import os
import json
import pandas as pd
from datetime import datetime

def read_counts_json(file_path):
    # Load JSON data from file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize an empty list to hold the data for the DataFrame
    data_list = []
    # Loop through each logic in the 'logics' list
    for logic in data['logics_data']['logics']:
        # Loop through each record in the 'records' list
        for record in logic['records']:
            # Extract the 'from' and 'to' timestamps and convert them to datetime objects
            from_timestamp = datetime.fromtimestamp(record['from'] / 1000.0)
            to_timestamp = datetime.fromtimestamp(record['to'] / 1000.0)
            
            # Loop through each count in the 'counts' list
            for count in record['counts']:
                # Extract the 'name' field
                name = count['name']
                # Extract the 'value' field
                value = count['value']
                
                # Append the data to the data list
                data_list.append([logic['id'], from_timestamp, to_timestamp, name, value])
    
    # Convert the data list into a pandas DataFrame
    df = pd.DataFrame(data_list, columns=['id', 'from', 'to', 'type', 'count'])
    return df

# a = read_counts_json("test-data/2.1_logics_sn001EC0A01198_2024-04-15T11-30-00Z_id3413.json")

def process_files_in_folder(folder_path, output_csv_path):
    # Initialize an empty list to hold the dataframes
    df_list = []
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Convert the JSON file to a dataframe and append it to the list
            df = read_counts_json(file_path)
            df_list.append(df)
    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(df_list, ignore_index=True)
    # Write the combined dataframe to a CSV file
    combined_df.to_csv(output_csv_path, index=False)

process_files_in_folder("test-data/line", "test-data/line.csv")