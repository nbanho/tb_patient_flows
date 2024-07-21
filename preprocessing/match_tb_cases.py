import pandas as pd

def match_tb_cases(clin_df, date, min_time_in_area, max_time_diff):
	"""
	Matches clinical data with tracking data based on time in tb_area and time difference.
	
	Parameters:
	clin_df (pd.DataFrame): DataFrame containing clinical data with columns 'id', 'date', 'start_time', 'completion_time'.
	date (str): The date to filter the clinical data.
	max_time_diff (int): The maximum time difference in milliseconds.
	min_time_in_area (int): The minimum time in the TB area in milliseconds.
	
	Returns:
	None: The function saves the matched DataFrame to a CSV file.
	"""
	# Step 1: Filter clin_df rows according to the specified date
	clin_df_filtered = clin_df[clin_df['date'] == date]
	
	# Step 2: Load track_df file found in the folder "data-clean/tracking/linked/"
	mapping_track_file_path = f"data-clean/tracking/linked/{date}.csv"
	mapping_track_df = pd.read_csv(mapping_track_file_path)
	track_df = pd.read_csv(f"data-clean/tracking/unlinked/{date}.csv")
	track_df = track_df.merge(mapping_track_df, left_on='track_id', right_on='raw_track_id', how='left')
	track_df['track_id'] = track_df['track_id_y'].combine_first(track_df['track_id_x'])
	track_df = track_df.drop(columns=['track_id_x', 'track_id_y', 'raw_track_id'])
	
	# Initialize a list to store the match results
	match_results = []
	
	# Step 3: For each clin_id in clin_df, find a matching track_id in track_df
	for _, clin_row in clin_df_filtered.iterrows():
		clin_id = clin_row['id']
		start_time = clin_row['start_time']
		completion_time = clin_row['completion_time']
		
		# Filter track_df rows based on start_time and completion_time
		potential_matches = track_df[(track_df['time'] >= start_time) & (track_df['time'] <= completion_time) & (track_df['in_tb_pat'])]
		
		# Compute metrics
		potential_matches = potential_matches.groupby('track_id').agg(min_time_diff=('time', lambda x: (x - start_time).abs().min()), row_count=('time', 'size')).reset_index()
		
		# Filter minimum criteria
		potential_matches = potential_matches[(potential_matches['min_time_diff'] <= max_time_diff) & (potential_matches['row_count'] >= min_time_in_area)]
		
		# Consider most likely match if there are multiple potential matches
		potential_matches = potential_matches.sort_values(by=['min_time_diff', 'row_count'], ascending=[True, False])
		
		# Append the match result to the list
		match_results.append({
			'clin_id': clin_id,
			'start_time': start_time,
			'completion_time': completion_time,
			'track_id': potential_matches['track_id'].iloc[0] if not potential_matches.empty else None,
			'time_diff': potential_matches['min_time_diff'].iloc[0] / 1000 if not potential_matches.empty else None,
			'time_in_area': potential_matches['row_count'].iloc[0] if not potential_matches.empty else None
		})
	
	# Convert match results to a DataFrame
	match_df = pd.DataFrame(match_results)
	match_df['track_id'] = match_df['track_id'].fillna(-1).astype(int)
	
	# Step 4: Save the match DataFrame to a CSV file
	output_file_path = f"data-clean/tracking/matched/{date}.csv"
	match_df.to_csv(output_file_path, index=False)

# Example usage
clin_df = pd.read_csv('data-clean/clinical/tb_cases.csv')
date = "2024-06-20"
min_time_in_area = 1
max_time_diff = 300 * 1000
match_tb_cases(clin_df, date, min_time_in_area, max_time_diff)