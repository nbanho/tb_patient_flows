import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Ignore the SettingWithCopyWarning in pandas
pd.options.mode.chained_assignment = None  # default='warn'

# check if direction is continued
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

# filter potential matches based on time and distance
def filter_potential_matches(potential_matches: pd.DataFrame, last_point: pd.Series, time: int, distance: float) -> pd.DataFrame:
	"""
	Calculates potential matches for a track based on time and distance criteria, considering seating area and direction.
	
	Parameters:
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
	
	return potential_matches

# link interrupted tracks
def link_interrupted_tracks(df: pd.DataFrame, track_id: int, max_time_walk: int, max_dist_walk: float, max_time_sit: int, max_dist_sit: float, max_time_quick: int, max_dist_quick: float, max_time_tba: int, max_dist_tba: float) -> pd.DataFrame:
	"""
	Iteratively links interrupted tracks based on time and distance criteria, ensuring all linked tracks
	receive the earliest track_id among them. The process is repeated until no more links can be made.
	
	Parameters:
	- df: DataFrame with columns ['time', 'track_id', 'position_x', 'position_y'].
	- max_time_walk: Maximum time to look ahead for linking tracks.
	- max_dist_walk: Maximum distance to look ahead for linking tracks.
	- max_time_sit: Maximum time to look ahead for linking tracks if the last point is within the seating area.
	- max_dist_sit: Maximum distance to look ahead for linking tracks if the last point is within the seating area.
	- max_time_quick: Maximum time to look ahead for linking tracks if the last point is in the seating area but walking.
	- max_dist_quick: Maximum distance to look ahead for linking tracks if the last point is in the seating area but walking.
	- max_time_tba: Maximum time to look ahead for linking tracks if the last point is in the TB area.
	- max_dist_tba: Maximum distance to look ahead for linking tracks if the last point is in the TB area.
	
	Returns:
	- DataFrame with updated 'track_id' for linked tracks.
	"""
	# current track with last two racks
	current_track = df[df['track_id'] == track_id]
	first_point = current_track.iloc[0]
	last_point = current_track.iloc[-1]
	
	# check if near entry
	if last_point['near_entry']:
		return False, 0, df
	
	# check if in TB area for clinic staff
	if first_point['in_tb_cs']:
		return False, 0, df
	
	# Filter based on time condition and then select the first occurrence of each potential track_id
	potential_matches = df.drop_duplicates(subset='track_id', keep='first')
	potential_matches = potential_matches[potential_matches['time'] > last_point['time']]
	potential_matches = potential_matches[potential_matches['near_entry'] == False]
	potential_matches = potential_matches[potential_matches['in_tb_cs'] == False]
	if potential_matches.empty:
		return False, 0, df
	
	# filter potential matches based on time, distance, and walking direction
	if last_point['in_seating']:
		potential_matches_sit = potential_matches[potential_matches['in_seating'] == True]
		potential_matches_sit = filter_potential_matches(potential_matches_sit, last_point, max_time_sit, max_dist_sit)
		if potential_matches_sit.empty:
			link_type = 1
			potential_matches = filter_potential_matches(potential_matches, last_point, max_time_quick, max_dist_quick)
		else:
			link_type = 3
			potential_matches = potential_matches_sit
	elif last_point['in_tb_pat'] or last_point['in_vitals_pat']:
		if last_point['in_tb_pat']:
			potential_matches_tba = potential_matches[potential_matches['in_tb_pat']]
		else:
			potential_matches_tba = potential_matches[potential_matches['in_vitals_pat']]
		potential_matches_tba = filter_potential_matches(potential_matches_tba, last_point, max_time_tba, max_dist_tba)
		if potential_matches_tba.empty:
			link_type = 1
			potential_matches = filter_potential_matches(potential_matches, last_point, max_time_quick, max_dist_quick)
		else:
			link_type = 4
			potential_matches = potential_matches_tba
	else:
		link_type = 2
		potential_matches = filter_potential_matches(potential_matches, last_point, max_time_walk, max_dist_walk)
		# second_last_point = current_track.iloc[-2]
		# potential_matches = potential_matches[potential_matches.apply(lambda row: is_direction_continued(second_last_point, last_point, row), axis=1)]
	
	if potential_matches.empty:
		return False, link_type, df
	else:
		# Sort by time_diff and distance, then take the first match
		best_match = potential_matches.sort_values(by=['time_diff', 'distance']).iloc[0]
		best_match_track_id = best_match['track_id']
		
		# Update new_track_id_mapping to reflect the link
		df['track_id'] = df['track_id'].replace(track_id, best_match_track_id)
		return True, link_type, df

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
	
	for i in range(len(parameters[0])):
		link_walk_count = 0  # Initialize a counter for the number of links made in walking area
		link_quick_count = 0 # Initialize a counter for the number of links made in quick area
		link_sit_count = 0  # Initialize a counter for the number of links made in seating area
		link_tba_count = 0 # Initialize a counter for the number of links made in TB area
		more_links = True
		while more_links:
			# Copy the current state of updated_dataset to check for changes after processing
			previous_dataset = updated_dataset.copy()
			last_entries = updated_dataset.drop_duplicates(subset='track_id', keep='last')
			last_entries = last_entries[last_entries['near_entry'] == False]
			track_ids = last_entries['track_id'].unique()
			
			for track_id in track_ids:
				# Attempt to link interrupted tracks
				updated, link_type, updated_dataset = link_interrupted_tracks(updated_dataset, track_id, parameters[0][i], parameters[1][i], parameters[2][i], parameters[3][i], parameters[4][i], parameters[5][i], parameters[6][i], parameters[7][i])
				# Check if the operation resulted in a change
				if updated:
					if link_type == 1:
						link_quick_count += 1
					elif link_type == 2:
						link_walk_count += 1
					elif link_type == 3:
						link_sit_count += 1
					elif link_type == 4:
						link_tba_count += 1
			
			# Check if at least one track_id has been replaced by comparing the datasets before and after processing
			more_links = not previous_dataset.equals(updated_dataset)
		
		# Print the total number of links made
		print(f"... total number of links made after round {i+1}: {link_quick_count} (quick), {link_walk_count} (walk), {link_sit_count} (sit), {link_tba_count} (tba)")
	
	return updated_dataset

# testing
# test_old = pd.read_csv("data-clean/tracking/unlinked/2024-06-26.csv")
# test_new = process_tracks(test_old, params)
# unique_mappings = test_new[['raw_track_id', 'track_id']].drop_duplicates()
# unique_mappings.to_csv("data-clean/tracking/linked/2024-06-26.csv", index=False)

def read_and_process_tracks(file: str, pars = list):
	print(f"Processing file: {file}")
	uldf = pd.read_csv(os.path.join('../data-clean/tracking/unlinked/', file))
	ldf = process_tracks(uldf, pars)
	mappings = ldf[['raw_track_id', 'track_id']].drop_duplicates()
	mappings.to_csv(os.path.join('../data-clean/tracking/linked/', file), index=False)
	print(f"Saving file: {file}")


if __name__ == "__main__":
		# files
		unlinked_files = ["2024-06-20.csv"] # unlinked_files = [f for f in os.listdir('../data-clean/tracking/unlinked/') if f.endswith('.csv')]
		
		# processing parameters
		max_time_quick = np.arange(1, 11, 1).tolist()
		max_time_quick = [x * 1000 for x in max_time_quick]
		max_dist_quick = np.repeat(1, 10).tolist()
		max_time_walk = [5, 5, 5, 5, 10, 10, 10, 10, 30, 30]
		max_time_walk = [x * 1000 for x in max_time_walk]
		max_dist_walk = [0.5, 1, 1.5, 2, 0.5, 1, 1.5, 2, 0.25, 0.5]
		max_time_sit = [60, 60, 60, 60, 300, 300, 300, 300, 600, 600]
		max_time_sit = [x * 1000 for x in max_time_sit]
		max_dist_sit = [0.25, 0.5, 0.75, 1, 0.25, 0.5, 0.75, 1.0, 0.25, 0.5]
		max_time_tba = [5, 5, 5, 5, 10, 10, 10, 10, 20, 20]
		max_time_tba = [x * 1000 for x in max_time_tba]
		max_dist_tba = [0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4]
		params = [max_time_walk, max_dist_walk, max_time_sit, max_dist_sit, max_time_quick, max_dist_quick, max_time_tba, max_dist_tba]
		all_same_length = all(len(sublist) == len(params[0]) for sublist in params)
		if not all_same_length:
			raise ValueError("Parameter lists must be of the same length.")
		
		# settings
		num_cores = 1
		
		with ProcessPoolExecutor(max_workers=num_cores) as executor:
				executor.map(
					read_and_process_tracks,
					unlinked_files,
					[params] * len(unlinked_files)
				)