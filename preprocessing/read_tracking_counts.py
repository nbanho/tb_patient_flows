import os
import json
import pandas as pd
import re
from collections import defaultdict
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
                data_list.append([logic['id'], logic['name'], from_timestamp, to_timestamp, name, value])
    
    # Convert the data list into a pandas DataFrame
    df = pd.DataFrame(data_list, columns=['id', 'name', 'from', 'to', 'type', 'count'])
    return df

# a = read_counts_json("data-raw/xovis.nosynch/final-data/LINE/logics_ms000732AB8D11_2_2024-05-17T13-05-00Z_id0.json")

# Define the folder path
folder_path = "data-raw/xovis.nosynch/final-data/LINE/"
all_dfs = []
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = read_counts_json(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs("data-clean/tracking/counts", exist_ok=True)
    final_df.to_csv("data-clean/tracking/counts/counts.csv", index=False)
else:
    print("No valid dataframes to concatenate.")