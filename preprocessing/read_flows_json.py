import os
import json
import pandas as pd

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
	# Separate the combined dataframe into a list of dataframes by day
	daily_dfs = separate_by_day(combined_df)
	# Filter each daily dataframe to include only daytime data
	filtered_dfs = [filter_daytime(df, 4, 16) for df in daily_dfs]
	# Save each data frame to a separate CSV file
	for df in filtered_dfs:
		if df.empty:
			continue
		# Get the date from the first row of the dataframe
		date = pd.to_datetime(df['time'].iloc[0], unit='ms').strftime('%Y-%m-%d')
		df.to_csv(f"{output_csv_path}{date}.csv", index=False)

process_event_files_in_folder("data-raw/xovis/june-25/Event/", "data-clean/tracking/unlinked/")