import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Ignore the SettingWithCopyWarning in pandas
pd.options.mode.chained_assignment = None  # default='warn'

# filter potential matches based on time and distance
def filter_potential_matches(pot_mat: pd.DataFrame, lp: pd.Series, time: int, distance: float) -> pd.DataFrame:
    """
    Calculates potential matches for a track based on time and distance criteria, considering seating area and direction.
    
    Parameters:
    - pot_mat: DataFrame with all potential matches
    - lp: The last point of the current track.
    - time: Maximum time to look ahead for linking tracks.
    - distance: Minimum distance to look ahead for linking tracks.
    
    Returns:
    - DataFrame of potential matches.
    """
    # Filter potential matches based on time
    pot_mat['time_diff'] = pot_mat['time'] - lp['time']
    pot_mat = pot_mat[pot_mat['time_diff'] <= time]
    
    # Filter potential matches based on distance
    pot_mat['distance'] = np.sqrt((pot_mat['position_x'] - lp['position_x'])**2 + (pot_mat['position_y'] - lp['position_y'])**2)
    pot_mat = pot_mat[pot_mat['distance'] <= distance]
    
    return pot_mat


def has_alternative(df: pd.DataFrame, t: int, x: float, y: float, td: int, dist: float) -> bool:
    """
    Check if potential alternatives have a better alternative based on the time difference and distance of the potential match to the track.
    
    Parameters:
    - df: DataFrame with columns ['time', 'track_id', 'position_x', 'position_y'].
    - t: Time of potential match
    - x: Position x of potential match
    - y: Position y of potential match
    - td: Time difference between potential match and track
    - dist: Distance between potential match and track
    
    Return:
    - Bool whether there is an alternative (data frame of alternatives not empty)
    """
    alternatives = df.drop_duplicates(subset='track_id', keep='last')
    alternatives = alternatives[alternatives['time'] < t]
    if alternatives.empty:
        return False
    alternatives['time_diff'] = t - alternatives['time']
    alternatives = alternatives[alternatives['time_diff'] < td]
    if alternatives.empty:
        return False
    alternatives['distance'] = np.sqrt((alternatives['position_x'] - x)**2 + (alternatives['position_y'] - y)**2)
    alternatives = alternatives[alternatives['distance'] < dist]
    if alternatives.empty:
        return False
    return True


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
    last_point = current_track.iloc[-1]
    
    # check if near entry
    if last_point['near_entry']:
        return False, 0, df
    
    # filter based on time condition and then select the first occurrence of each potential track_id
    potential_matches = df.drop_duplicates(subset='track_id', keep='first')
    potential_matches = potential_matches[potential_matches['time'] > last_point['time']]
    potential_matches = potential_matches[potential_matches['near_entry'] == False]
    if potential_matches.empty:
        return False, 0, df
    
    # filter potential matches separately in seating, tb or walking area by time and distance
    if last_point['in_seating']:
        link_type = 3
        matches = potential_matches[potential_matches['in_seating'] == True]
        if not matches.empty:
            matches = filter_potential_matches(matches, last_point, max_time_sit, max_dist_sit)
    elif last_point['in_check']:
        link_type = 4
        matches = potential_matches[potential_matches['in_check'] == True]
        if not matches.empty:
            matches = filter_potential_matches(matches, last_point, max_time_tba, max_dist_tba)
    else:
        link_type = 2
        matches = potential_matches[potential_matches['in_check'] == False]
        matches = filter_potential_matches(matches, last_point, max_time_walk, max_dist_walk)
    
    # if no matches, check for between area matches
    if matches.empty:
        link_type = 1
        matches = filter_potential_matches(potential_matches, last_point, max_time_quick, max_dist_quick)
    
    if matches.empty:
        return False, 0, df
    
    # check if there are better alternatives
    matches['alt'] = matches.apply(
        lambda row: has_alternative(df, row['time'], row['position_x'], row['position_y'], row['time_diff'], row['distance']),
        axis=1)
    matches = matches[matches['alt'] == False]
    
    # if there is none or more than one match do not link, otherwise link with single match
    if matches.empty:
        return False, 0, df
    elif len(matches) > 1:
        return False, 0, df
    else:
        match = matches.iloc[0]
        match_track_id = match['track_id']
        df['track_id'] = df['track_id'].replace(track_id, match_track_id)
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
# max_time_quick = [5, 5, 10, 10, 20, 20]
# max_time_quick = [x * 1000 for x in max_time_quick]
# max_dist_quick = [0.5, 1, 0.5, 1, 0.5, 1]
# max_time_walk = [10, 10, 20, 20, 30, 30]
# max_time_walk = [x * 1000 for x in max_time_walk]
# max_dist_walk = [1, 2, 1, 2, 1, 2]
# max_time_sit = [60, 60, 120, 120, 300, 300]
# max_time_sit = [x * 1000 for x in max_time_sit]
# max_dist_sit = [0.25, 0.5, 0.25, 0.5, 0.25, 0.5]
# max_time_tba = [30, 30, 60, 60, 120, 120]
# max_time_tba = [x * 1000 for x in max_time_tba]
# max_dist_tba = [0.25, 0.5, 0.25, 0.5, 0.25, 0.5]
# params = [max_time_walk, max_dist_walk, max_time_sit, max_dist_sit, max_time_quick, max_dist_quick, max_time_tba, max_dist_tba]
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
        unlinked_files = ["2024-06-24.csv"] # unlinked_files = [f for f in os.listdir('../data-clean/tracking/unlinked/') if f.endswith('.csv')]
        
        # processing parameters
        max_time_quick = [5, 5, 10, 10, 20, 20, 30, 30]
        max_time_quick = [x * 1000 for x in max_time_quick]
        max_dist_quick = [0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1]
        max_time_walk = [5, 5, 10, 10, 30, 30, 60, 60]
        max_time_walk = [x * 1000 for x in max_time_walk]
        max_dist_walk = [1, 2, 1, 2, 1, 2, 1, 2]
        max_time_sit = [30, 30, 60, 60, 120, 120, 300, 300]
        max_time_sit = [x * 1000 for x in max_time_sit]
        max_dist_sit = [0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5]
        max_time_tba = [10, 10, 30, 30, 60, 60, 120, 120]
        max_time_tba = [x * 1000 for x in max_time_tba]
        max_dist_tba = [0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5]
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