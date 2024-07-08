import os
import pandas as pd
import json
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# Ignore the SettingWithCopyWarning in pandas
pd.options.mode.chained_assignment = None  # default='warn'

# filter parameters
filt_duration = 10 * 1000 # in milliseconds
filt_distance = 5 # in m; currently not considered

# annotation parameters
cons_buff_dist = 0.5
main_buff_dist = 1

# processing parameters
max_time_lookahead_walk = [1, 2, 3, 5, 7, 10]
min_dist_lookahead_walk = [0.05, 0.075, 0.1, 0.25, 0.5, 1]
max_time_lookahead_sit = [10, 30, 60, 90, 180, 300]
max_distance_lookahead_sit = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3]
params = [max_time_lookahead_walk, min_dist_lookahead_walk, max_time_lookahead_sit, max_distance_lookahead_sit]
all_same_length = all(len(sublist) == len(params[0]) for sublist in params)
if not all_same_length:
	raise ValueError("Parameter lists must be of the same length.")

# load data
def separate_by_day(csv_file_path: str) -> list:
	"""
	Reads a CSV file into a pandas DataFrame and separates it into a list of DataFrames for each day based on a Unix timestamp column named 'time', assuming 'time' is in milliseconds, without converting to datetime.

	:param csv_file_path: The file path to the CSV file.
	:return: A list of pandas DataFrames, each containing the data for a separate day.
	"""
	import pandas as pd
	import numpy as np

	# Read the CSV file into a pandas DataFrame
	df = pd.read_csv(csv_file_path)

	# Calculate the day number since the Unix epoch for each timestamp
	# Assuming 'time' is in milliseconds, convert to days by dividing by (1000 * 60 * 60 * 24)
	# Use floor division to get the day number
	df['day_number'] = (df['time'] // (1000 * 60 * 60 * 24))

	# Group the DataFrame by the day number
	grouped = df.groupby('day_number')

	# Create a list of DataFrames, each containing data for a separate day
	df_list = [group for _, group in grouped]

	return df_list

a = separate_by_day("data-raw/tracking/june-25-events.csv")

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

b = filter_daytime(a[1], 4, 16)

# filter data

def compute_group_duration(group: pd.DataFrame) -> int:
	"""Computes the duration of a single group."""
	return group['time'].max() - group['time'].min()

def compute_group_distance(group: pd.DataFrame) -> float:
	"""Computes the total moving distance for a single group."""
	distances = ((group['position_x'].diff()**2 + group['position_y'].diff()**2)**0.5).fillna(0)
	return distances.sum()

def filter_tracks(df: pd.DataFrame, min_duration: int, min_distance: int = None) -> pd.DataFrame:
	"""
	Filters tracks based on minimum duration and distance criteria.
	
	:param: df: The input DataFrame with columns 'time', 'track_id', 'position_x', and 'position_y'.
	:param: min_duration: The minimum duration of a track in milliseconds.
	:param: min_distance: The minimum distance traveled by a track in meters.
	:return: A filtered DataFrame containing only tracks that meet the criteria.
	"""
	# Pre-compute duration and distance for each group and store in a DataFrame
	if min_duration is not None:
		duration_df = df.groupby('track_id').apply(lambda x: compute_group_duration(x)).reset_index(name='duration')
	if min_distance is not None:
		distance_df = df.groupby('track_id').apply(lambda x: compute_group_distance(x)).reset_index(name='distance')
	
	# Filter based on duration and distance criteria
	if min_duration and min_distance is not None:
		summary_df = pd.merge(duration_df, distance_df, on='track_id')
		valid_tracks = summary_df[(summary_df['duration'] > min_duration) & (summary_df['distance'] > min_distance)]
	elif min_duration is not None:
		summary_df = duration_df
		valid_tracks = summary_df[(summary_df['duration'] > min_duration)]
	else:
		summary_df = distance_df
		valid_tracks = summary_df[(summary_df['distance'] > min_distance)]
	
	# Filter the original DataFrame to include only valid tracks
	filtered_df = df[df['track_id'].isin(valid_tracks['track_id'])]
	
	return filtered_df


c = filter_tracks(b, min_duration=filt_duration)

# annotate data

# Load the JSON data
with open('data-raw/background/config.json') as f:
	data = json.load(f)

# Navigate to the singlesensors within bern-multi01
geometries = data['multisensors']['bern-multi01']['geometries']

# Filter geometries for zones containing "entry" in their name
entry_geometries = [g for g in geometries if g['type'] == 'ZONE' and 'entry' in g['name'].lower()]

# Prepare polygons with adjusted boundaries
polygons = []
for geometry in entry_geometries:
	# Adjust the indentation of the code inside the loop
	# to match the expected indentation level
	polygons.append(Polygon(geometry['geometry']))

# Combine all polygons into a single geometry for efficient querying
combined_polygon = unary_union(polygons)ame or 'reg' in name else 1
buffered_polygon = polygon.buffer(buffer_distance)
polygons.append(buffered_polygon)

# Combine all polygons into a single geometry for efficient querying
combined_polygon = unary_union(polygons)

def annotate_near_entry(df: pd.DataFrame, combined_polygon) -> pd.DataFrame:
	"""
	Annotates rows in the DataFrame if they are near an entry zone.

	Parameters:
	- df (pd.DataFrame): The DataFrame to annotate.
	- combined_polygon: The pre-computed combined polygon of all relevant entry zones.

	Returns:
	- pd.DataFrame: The annotated DataFrame with a new column 'near_entry'.
	"""
	# Convert DataFrame columns to Shapely Points and check if they are within the combined polygon
	points = [Point(x, y) for x, y in zip(df['position_x'], df['position_y'])]
	df['near_entry'] = [combined_polygon.contains(point) for point in points]
	
	return df

# Create the new geometry dictionary
seating_geometries = []
seating_area_geometry = {
	'geometry': seating_area_coords,
	'type': 'ZONE',
	'name': 'seating area 1'
}

# Append the new geometry to the geometries list
seating_geometries.append(seating_area_geometry)

# Define the coordinates for the new rectangle
seating_area_2_coords = [[13.5, 0], [47.5, 0], [47.5, 1.75], [13.5, 1.75]]

# Create the new geometry dictionary for the second seating area
seating_area_2_geometry = {
	'geometry': seating_area_2_coords,
	'type': 'ZONE',
	'name': 'seating area 2'
}

# Append the new geometry to the geometries list
seating_geometries.append(seating_area_2_geometry)
combined_seating_polygon = unary_union([Polygon(geometry['geometry']) for geometry in seating_geometries])

def annotate_in_seat_area(df: pd.DataFrame, combined_polygon) -> pd.DataFrame:
	"""
	Annotates rows in the DataFrame if they are near an entry zone.
	
	Parameters:
	- df (pd.DataFrame): The DataFrame to annotate.
	- combined_polygon: The pre-computed combined polygon of all relevant seating zones.
	
	Returns:
	- pd.DataFrame: The annotated DataFrame with a new column 'near_entry'.
	"""
	# Convert DataFrame columns to Shapely Points and check if they are within the combined polygon
	points = [Point(x, y) for x, y in zip(df['position_x'], df['position_y'])]
	df['in_seating'] = [combined_polygon.contains(point) for point in points]
	
	return df

# compute the distance between consecutive points
def compute_dxy(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Computes the Euclidean distance between consecutive points within each track_id
	and adds it as a new column 'dxy' to the DataFrame.
	
	Parameters:
	- df (pd.DataFrame): The input DataFrame with columns 'time', 'track_id', 'position_x', and 'position_y'.
	
	Returns:
	- pd.DataFrame: The DataFrame with an additional column 'dxy'.
	"""
	# Group by 'track_id' and calculate the differences within each group
	df['delta_x'] = df.groupby('track_id')['position_x'].diff()
	df['delta_y'] = df.groupby('track_id')['position_y'].diff()
	
	# Compute the Euclidean distance (dxy) using the differences
	df['dxy'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
	
	return df

ca = annotate_near_entry(c, combined_polygon)
ca = annotate_in_seat_area(ca, combined_seating_polygon)
ca = compute_dxy(ca)
ca.to_csv("data-clean/tracking/unlinked-event-20-06-2024.csv", index=False)

# Assuming 'df' is your DataFrame and it already has the 'dxy' column computed
plt.hist(df['dxy'].dropna(), bins=50)  # Drop NA values to avoid errors, adjust bins as needed
plt.title('Histogram of dxy')
plt.xlabel('dxy')
plt.ylabel('Frequency')
plt.show()

quantiles = df['dxy'].quantile([i / 20.0 for i in range(21)])
print(quantiles)
quantiles_walk = ca[ca['in_seating'] == False]['dxy'].quantile([i / 20.0 for i in range(21)])
print(quantiles_walk)

# process data

def is_direction_continued(second_last_point: pd.Series, last_point: pd.Series, potential_match_point: pd.Series) -> bool:
	"""
	Determines if the direction from the second_last_point to the last_point
	towards the potential_match_point continues in the general direction of movement.
	
	Parameters:
	- second_last_point: A dict or similar structure with 'position_x' and 'position_y' for the second last point.
	- last_point: A dict or similar structure with 'position_x' and 'position_y' for the last point.
	- potential_match_point: A dict or similar structure with 'position_x' and 'position_y' for the potential match point.
	
	Returns:
	- True if the direction is continued (angle > 90 degrees), False otherwise.
	"""
	# Calculate the direction vector for the last two points of the current track
	direction_vector = (last_point['position_x'] - second_last_point['position_x'], last_point['position_y'] - second_last_point['position_y'])
	
	# Calculate the vector from the last point to the potential match point
	match_vector = (potential_match_point['position_x'] - last_point['position_x'], potential_match_point['position_y'] - last_point['position_y'])
	
	# Calculate the dot product of the two vectors
	dot_product = direction_vector[0] * match_vector[0] + direction_vector[1] * match_vector[1]
	
	# Calculate the magnitudes of the vectors
	magnitude_direction_vector = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
	magnitude_match_vector = np.sqrt(match_vector[0]**2 + match_vector[1]**2)
	
	# Check if either magnitude is zero to avoid division by zero
	if magnitude_direction_vector == 0 or magnitude_match_vector == 0:
		# Handle the case as per your application logic, here assuming direction is continued
		return True
	
	# Calculate the cosine of the angle between the two vectors
	cos_theta = dot_product / (magnitude_direction_vector * magnitude_match_vector)
	
	# If cos_theta is less than 0, the angle is greater than 90 degrees
	return cos_theta > 0


# # Generate a random sequence of points
# np.random.seed(43) # For reproducibility
# points = pd.DataFrame({
#     'position_x': np.random.rand(10),
#     'position_y': np.random.rand(10)
# })

# # Apply the function to each consecutive triplet of points
# results = []
# for i in range(len(points) - 2):
#     result = is_direction_continued(points.iloc[i], points.iloc[i+1], points.iloc[i+2])
#     results.append(result)

# # Plot the sequence of points
# plt.figure(figsize=(10, 6))
# plt.plot(points['position_x'], points['position_y'], marker='o')

# # Annotate the plot with the result of the function for each triplet
# for i, result in enumerate(results):
#     plt.annotate(f'{result}', (points.iloc[i+2]['position_x'], points.iloc[i+2]['position_y']), textcoords="offset points", xytext=(0,10), ha='center')

# plt.title('Random Sequence of Points with Direction Continuity Annotation')
# plt.xlabel('Position X')
# plt.ylabel('Position Y')
# plt.grid(True)
# plt.show()


def find_potential_matches(df: pd.DataFrame, potential_matches: pd.DataFrame, second_last_point: pd.Series, last_point: pd.Series, time: int, distance: float) -> pd.DataFrame:
	"""
	Calculates potential matches for a track based on time and distance criteria, considering seating area and direction.
	
	Parameters:
	- df: DataFrame with track data.
	- last_point: The last point of the current track.
	- time: Maximum time to look ahead for linking tracks.
	- distance: Minimum distance to look ahead for linking tracks.
	
	Returns:
	- DataFrame of potential matches.
	"""
	# Filter potential matches based on time
	potential_matches['time_diff'] = potential_matches['time'] - last_point['time']
	potential_matches = potential_matches[potential_matches['time_diff'] <= time]
	
	# Filter potential matches based on distance
	potential_matches['distance'] = np.sqrt((potential_matches['position_x'] - last_point['position_x'])**2 + (potential_matches['position_y'] - last_point['position_y'])**2)
	potential_matches = potential_matches[potential_matches['distance'] <= distance]
	
	# Filter potential matches based on moving direction
	if not last_point['in_seating']:
		potential_matches = potential_matches[potential_matches.apply(lambda row: is_direction_continued(second_last_point, last_point, row), axis=1)]
	
	return potential_matches


def link_interrupted_tracks(df: pd.DataFrame, track_id: int, max_time_walk: int, min_dist_walk: float, max_time_sit: int, max_dist_sit: float) -> pd.DataFrame:
	"""
	Iteratively links interrupted tracks based on time and distance criteria, ensuring all linked tracks
	receive the earliest track_id among them. The process is repeated until no more links can be made.
	
	Parameters:
	- df: DataFrame with columns ['time', 'track_id', 'position_x', 'position_y'].
	- max_time_walk: Maximum time to look ahead for linking tracks.
	- min_dist_walk: Minimum distance to look ahead for linking tracks if the last two points' distance is lower than this value.
	- max_time_sit: Maximum time to look ahead for linking tracks if the last point is within the seating area.
	- max_dist_sit: Maximum distance to look ahead for linking tracks if the last point is within the seating area.
	
	Returns:
	- DataFrame with updated 'track_id' for linked tracks.
	"""
	# current track with last two racks
	current_track = df[df['track_id'] == track_id]
	last_point = current_track.iloc[-1]
	second_last_point = current_track.iloc[-2]
	
	# check if near entry
	if last_point['near_entry']:
		return False, df
	
	# Filter based on time condition and then select the first occurrence of each track_id
	potential_matches = df.drop_duplicates(subset='track_id', keep='first')
	potential_matches = potential_matches[potential_matches['time'] > last_point['time']]
	potential_matches = potential_matches[potential_matches['near_entry'] == False]
	if potential_matches.empty:
		return False, df
	
	# filter potential matches based on time, distance, and walking direction
	max_dist_walk = max(min_dist_walk, last_point['dxy'])
	potential_matches = find_potential_matches(df, potential_matches, second_last_point, last_point, max_time_walk, max_dist_walk)
	
	# consider additional matches in seating area
	if potential_matches.empty:
		if last_point['in_seating']:
			potential_matches = find_potential_matches(df, potential_matches, second_last_point, last_point, max_time_sit, max_dist_sit)
	
	if potential_matches.empty:
		return False, df
	else:
		# Sort by time_diff and distance, then take the first match
		best_match = potential_matches.sort_values(by=['time_diff', 'distance']).iloc[0]
		best_match_track_id = best_match['track_id']
		
		# Update new_track_id_mapping to reflect the link
		df['track_id'] = df['track_id'].replace(track_id, best_match_track_id)
		return True, df

def process_tracks(initial_dataset: pd.DataFrame, parameters: list) -> pd.DataFrame:
	"""
	Processes tracks with varying lookahead values using the first values from seconds_array and distance_array
	for the initial call, and then uses the subsequent values in the loop.

	Parameters:
	- initial_dataset (pd.DataFrame): The initial dataset to process.
	- parameters (list): An array of lists of the same length containing the parameters for linking tracks.

	Returns:
	- pd.DataFrame: The final updated dataset.
	"""
	# Initialize the updated_dataset with the initial_dataset before the loop
	updated_dataset = initial_dataset.copy()
	updated_dataset['raw_track_id'] = updated_dataset['track_id']
	link_count = 0  # Initialize a counter for the number of links made
	
	for i in range(len(parameters[0])):
		more_links = True
		while more_links:
			# Copy the current state of updated_dataset to check for changes after processing
			previous_dataset = updated_dataset.copy()
			last_entries = updated_dataset.drop_duplicates(subset='track_id', keep='last')
			last_entries = last_entries[last_entries['near_entry'] == False]
			track_ids = last_entries['track_id'].unique()
			
			for track_id in track_ids:
				# Attempt to link interrupted tracks
				updated, updated_dataset = link_interrupted_tracks(updated_dataset, track_id, parameters[0][i] * 1000, parameters[1][i], parameters[2][i] * 1000, parameters[3][i])
				# Check if the operation resulted in a change
				if updated:
					link_count += 1  # Increment the link count if a change was made
			
			# Check if at least one track_id has been replaced by comparing the datasets before and after processing
			more_links = not previous_dataset.equals(updated_dataset)
		
		# Print the total number of links made
		print(f"Total number of links made after round {i}: {link_count}")
	
	# Subset columns
	return updated_dataset

d = process_tracks(ca, params)
d.to_csv("data-clean/tracking/linked-event-20-06-2024.csv", index=False)