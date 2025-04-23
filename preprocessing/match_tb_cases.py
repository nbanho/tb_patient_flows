import pandas as pd
import os

def tb_area_tracks(df, min_time=5, max_time=float('inf')):
        """
        Calculate the total time spent in the TB area per track_id, along with start and end times.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing columns 'track_id' (int), 'time' (int, unix timestamp in ms), and 'in_tb_pat' (bool).
        min_time (int): Minimum time spent in the TB area in minutes for filtering. Default is 5 minutes.
        max_time (float): Maximum time spent in the TB area in minutes for filtering. Default is infinity.
        
        Returns:
        pd.DataFrame: DataFrame with columns 'track_id', 'total_time_in_tb', 'start_time', and 'end_time'.
        """
        # Group by track_id
        grouped = df.groupby('track_id')
        
        # Initialize lists to store results
        results = []
        
        for track_id, group in grouped:
                # Filter the group to include only rows where in_tb_pat is True
                tb_pat_group = group[group['in_tb_pat']]
                
                if not tb_pat_group.empty:
                        # Calculate total time spent in TB area
                        tb_pat_time = group['in_tb_pat'].sum() / 60
                        
                        # Calculate total time spent in TB staff area:
                        tb_staff_time = group['in_tb_cs'].sum() / 60
                        
                        # Check if total time is within the specified range
                        if (min_time <= tb_pat_time <= max_time) & (tb_staff_time < 1):
                                start_time = tb_pat_group['time'].min()
                                end_time = tb_pat_group['time'].max()
                                
                                # Append results
                                results.append({
                                        'track_id': int(track_id),
                                        'total_time_in_tb': tb_pat_time,
                                        'total_time_in_tb_staff': tb_staff_time,
                                        'start_time_in_tb': pd.to_datetime(start_time, unit='ms').strftime('%H:%M:%S'),
                                        'end_time_in_tb': pd.to_datetime(end_time, unit='ms').strftime('%H:%M:%S')
                                })
        
        # Convert results to DataFrame
        result_df = pd.DataFrame(results).sort_values('start_time_in_tb')
        
        return result_df


def match_tracking_clinic(date_csv: str):
    print(date_csv)
    date = date_csv.replace('.csv', '')
    # unlinked data
    unlinked_data = pd.read_csv(os.path.join("data-clean/tracking/unlinked/", date_csv))
    mapping_data = pd.read_csv(os.path.join("data-clean/tracking/linked/", date_csv))
    linked_data = unlinked_data.merge(mapping_data, left_on='track_id', right_on='raw_track_id', how='left')
    linked_data['track_id'] = linked_data['track_id_y'].combine_first(linked_data['track_id_x'])
    linked_data = linked_data.drop(columns=['track_id_x', 'track_id_y'])
    
    # tracks in tb area
    tracks_in_tb = tb_area_tracks(linked_data)
    
    # filter clinical data
    clinic_date = clinic[clinic['date'] == date]
    
    # column bind two data frames with unequal number of rows
    max_rows = max(len(tracks_in_tb), len(clinic_date))
    tracks_in_tb = tracks_in_tb.reset_index(drop=True).reindex(range(max_rows))
    clinic_date = clinic_date.reset_index(drop=True).reindex(range(max_rows))
    combined_df = pd.concat([tracks_in_tb.reset_index(drop=True), clinic_date.reset_index(drop=True)], axis=1)
    
    return combined_df
    

# load clinical data
clinic = pd.read_csv("data-clean/clinical/tb_cases.csv")

# perform matching
unlinked_files = [f for f in os.listdir("data-clean/tracking/unlinked/") if f.endswith('.csv')]

for file in unlinked_files:
    matched_data = match_tracking_clinic(file)
    matched_data.to_csv(os.path.join("data-clean/tracking/matched/", file))
