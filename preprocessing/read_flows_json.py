import os
import json
import pandas as pd
import re
from collections import defaultdict
from datetime import datetime

def event_json_to_dataframe(file_path):
	"""
	Reads a JSON file containing event data and converts it into a pandas DataFrame.
	
	:param file_path: The file path to the JSON file.
	:return: A pandas DataFrame containing the event data.
	"""
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
	# Shift time by two hours
	df['time'] = df['time'] + 2 * 60 * 60 * 1000
	return df

# load data
def separate_by_day(df: pd.DataFrame) -> list:
	"""
	Reads a CSV file into a pandas DataFrame and separates it into a list of DataFrames for each day based on a Unix timestamp column named 'time', assuming 'time' is in milliseconds, without converting to datetime.

	:param df: A pandas DataFrame.
	:return: A list of pandas DataFrames, each containing the data for a separate day.
	"""
	# Calculate the day number since the Unix epoch for each timestamp
	# Assuming 'time' is in milliseconds, convert to days by dividing by (1000 * 60 * 60 * 24)
	# Use floor division to get the day number
	df['day_number'] = (df['time'] // (1000 * 60 * 60 * 24))
	
	# Group the DataFrame by the day number
	grouped = df.groupby('day_number')
	
	# Create a list of DataFrames, each containing data for a separate day
	df_list = [group for _, group in grouped]
	
	return df_list

def filter_daytime(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
	"""
	Filters a pandas DataFrame to include only rows within the time window of 7am to 5pm and sorts the resulting DataFrame in ascending order by the 'time' column, without converting to datetime.
	
	:param df: The input pandas DataFrame with a Unix timestamp column named 'time' in milliseconds.
	:param start_hour: The start hour of the time window (inclusive).
	:param end_hour: The end hour of the time window (inclusive).
	:return: A pandas DataFrame filtered by the specified time window and sorted by 'time'.
	"""
	# Convert hours to milliseconds
	start_time_ms = start_hour * 3600 * 1000
	end_time_ms = end_hour * 3600 * 1000
	
	# Filter rows based on the time window, adjusting for the date
	# Assuming 'time' is the Unix timestamp in milliseconds
	# This example assumes that the timestamp at midnight (start of the day) can be found by modulo operation with a day's milliseconds
	day_in_ms = 24 * 3600 * 1000
	filtered_df = df[((df['time'] % day_in_ms) >= start_time_ms) & ((df['time'] % day_in_ms) <= end_time_ms)]
	
	# Sort the DataFrame by 'time'
	filtered_df = filtered_df.sort_values(by=['track_id','time'])
	
	return filtered_df

def process_event_files_in_folder(folder_path, files, output_csv_path, date):
	# Initialize an empty list to hold the dataframes
	df_list = []
	# Loop through each file in the folder
	for f in files:
		# Check if the file is a JSON file
		if f.endswith('.json'):
			# Construct the full file path
			file_path = os.path.join(folder_path, f)
			# Convert the JSON file to a dataframe and append it to the list
			try:
				df = event_json_to_dataframe(file_path)
				df_list.append(df)
			except Exception as e:
				print(f"Error processing file {f}: {e}")
				continue
	# Concatenate all dataframes into a single dataframe
	combined_df = pd.concat(df_list, ignore_index=True)
	# Filter daily dataframe to include only daytime data
	filtered_df = filter_daytime(combined_df, 6, 18)
	# Save data frame to a separate CSV file
	filtered_df.to_csv(f"{output_csv_path}{date}.csv", index=False)

# Define the folder path
folder_path = "data-raw/xovis.nosynch/final-data/EVENT/"

# Get all file names in the specified folder
file_names = os.listdir(folder_path)

# Regular expression to extract date in the format %Y-%m-%d
date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

# Dictionary to store grouped file names by date
grouped_files = defaultdict(list)

for file_name in file_names:
    # Extract date from file name
    match = date_pattern.search(file_name)
    if match:
        date = match.group(0)
        grouped_files[date].append(file_name)

# Convert defaultdict to a regular dictionary and print the result
grouped_files = dict(grouped_files)
cutoff_date = datetime.strptime('2024-06-16', '%Y-%m-%d')
grouped_files = {date: files for date, files in grouped_files.items() if datetime.strptime(date, '%Y-%m-%d') > cutoff_date}
grouped_files = dict(sorted(grouped_files.items(), key=lambda item: datetime.strptime(item[0], '%Y-%m-%d')))

# process files by date
for date in list(grouped_files.keys()):
		print(f"Processing files for date: {date}")
		process_event_files_in_folder(folder_path, grouped_files[date], "data-clean/tracking/unlinked/", date)

