import os
import json
import pandas as pd

import json
import pandas as pd

def event_json_to_dataframe(file_path):
    # Open the file and load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize an empty list to hold the data for the DataFrame
    data_list = []
    
    # Loop through each frame in the 'frames' list
    for frame in data['live_data']['frames']:
        # Extract the 'tracked_objects' list
        tracked_objects = frame['tracked_objects']
        
        # Extract the time
        time = frame['time']
        
        # Loop through each object in the 'tracked_objects' list
        for obj in tracked_objects:
            # Extract the required fields
            track_id = obj['track_id']
            obj_type = obj['type']
            position_x = obj['position'][0]
            position_y = obj['position'][1]
            attributes = obj['attributes']
            person_height = attributes.get('person_height')
            gender = attributes.get('gender')
            tag = attributes.get('tag')
            face_mask = attributes.get('face_mask')
            view_direction_x = attributes.get('view_direction')[0] if attributes.get('view_direction') else None
            view_direction_y = attributes.get('view_direction')[1] if attributes.get('view_direction') else None
            members = attributes.get('members')
            members_with_tag = attributes.get('members_with_tag')
            # Append the fields to the data list
            data_list.append([time, track_id, obj_type, position_x, position_y, person_height, gender, tag, face_mask, view_direction_x, view_direction_y, members, members_with_tag])
    
    # Convert the data list into a pandas DataFrame
    df = pd.DataFrame(data_list, columns=['time', 'track_id', 'type', 'position_x', 'position_y', 'person_height', 'gender', 'tag', 'face_mask', 'view_direction_x', 'view_direction_y', 'members', 'members_with_tag'])
    
    return df

# a = json_to_dataframe("test-data/2.0_live_data_ms3CECEFEB9AAE_1_2024-04-15T11-17-59Z_id3876608.json")

def process_event_files_in_folder(folder_path, output_csv_path):
    # Initialize an empty list to hold the dataframes
    df_list = []
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Convert the JSON file to a dataframe and append it to the list
            df = event_json_to_dataframe(file_path)
            df_list.append(df)
    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(df_list, ignore_index=True)
    # Write the combined dataframe to a CSV file
    combined_df.to_csv(output_csv_path, index=False)

process_event_files_in_folder("data-raw/xovis/june-25/Event/", "data-raw/tracking/june-25-events.csv")