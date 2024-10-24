import os
import pandas as pd
import json
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor

# filter data based on duration and distance

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


# annotate data

def annotate_xy(df: pd.DataFrame, polygon: Polygon, col_name: str) -> pd.DataFrame:
    """
    Annotates rows in the DataFrame if they are near an entry zone.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to annotate.
    - polygon: The combined polygon of all relevant entry zones.
    - col_name: The name of the new column to add.
    
    Returns:
    - pd.DataFrame: The annotated DataFrame with a new column 'near_entry'.
    """
    # Convert DataFrame columns to Shapely Points and check if they are within the combined polygon
    points = [Point(x, y) for x, y in zip(df['position_x'], df['position_y'])]
    df[col_name] = [polygon.contains(point) for point in points]
    
    return df


# process data

def process_file(filename, in_fold_path, out_fold_path, time_filter, distance_filter, entry_polygon, seating_polygon, check_polygon, check_tb_polygon, sputum_polygon):
        print(f"Processing file: {os.path.join(in_fold_path, filename)}")
        df = pd.read_csv(os.path.join(in_fold_path, filename))
        df = filter_tracks(df, time_filter, distance_filter)
        df = annotate_xy(df, entry_polygon, 'near_entry')
        df = annotate_xy(df, seating_polygon, 'in_seating')
        df = annotate_xy(df, check_polygon, 'in_check')
        df = annotate_xy(df, check_tb_polygon, 'in_check_tb')
        df = annotate_xy(df, sputum_polygon, 'in_sputum')
        df.to_csv(os.path.join(out_fold_path, filename), index=False)


if __name__ == "__main__":
        folder_path = '../data-clean/tracking/unlinked/'
        output_folder_path = '../data-clean/tracking/unlinked/'
        
        # Filter parameters
        filt_duration = 10 * 1000
        filt_distance = None
        
        # Annotation parameters
        cons_buff_dist = 0.15
        main_buff_dist = 0.25
        
        # Entries
        with open('../data-raw/background/config.json') as f:
            data = json.load(f)
        geometries = data['multisensors']['bern-multi01']['geometries']
        entry_geometries = [g for g in geometries if g['type'] == 'ZONE' and 'entry' in g['name'].lower()]
        entry_buffered_polygons = []
        for geometry in entry_geometries:
            # Determine buffer distance based on the name
            buffer_distance = cons_buff_dist if 'reg' in geometry['name'].lower() or 'cons' in geometry['name'].lower() else main_buff_dist
            # Create the polygon, apply the buffer, and add to the list
            entry_buffered_polygons.append(Polygon(geometry['geometry']).buffer(buffer_distance))
        entry_polygon = unary_union(entry_buffered_polygons)
        
        # Seating area
        seating_geometries = []
        seating_area_1_geometry = {
            'geometry': [[11.31, 2.6], [47.5, 2.6], [47.5, 6.5], [11.31, 6.5]],
            'type': 'ZONE',
            'name': 'seating area 1'
        }
        seating_geometries.append(seating_area_1_geometry)
        seating_area_2_geometry = {
            'geometry': [[13.5, 0], [47.5, 0], [47.5, 1.75], [13.5, 1.75]],
            'type': 'ZONE',
            'name': 'seating area 2'
        }
        seating_geometries.append(seating_area_2_geometry)
        seating_area_3_geometry = {
            'geometry': [[8-1.8-0.7, 8.8], [8-1.8, 8.8], [8-1.8, 8.8+4], [8-1.8-0.7, 8.8+4]],
            'type': 'ZONE',
            'name': 'seating area 3'
        }
        seating_geometries.append(seating_area_3_geometry)
        seating_area_4_geometry = {
            'geometry': [[8.2-2.4, 6.3], [8.2, 6.3], [8.2, 6.3+.5], [8.2-2.4, 6.3+.5]],
            'type': 'ZONE',
            'name': 'seating area 4'
        }
        seating_geometries.append(seating_area_4_geometry)
        seating_polygon = unary_union([Polygon(geometry['geometry']) for geometry in seating_geometries])

        # TB area
        check_area = [[8, .8], [11.3, .8], [11.3, 6.2], [8, 6.2]]
        check_area = {
            'geometry': check_area,
            'type': 'ZONE',
            'name': 'Check area'
        }
        check_area_polygon = Polygon(check_area['geometry'])
        check_tb_area = [[9.1, 2.6], [11.3, 2.6], [11.3, 3.7], [9.1, 3.7]]
        check_tb_area = {
            'geometry': check_tb_area,
            'type': 'ZONE',
            'name': 'TB area'
        }
        check_tb_area_polygon = Polygon(check_tb_area['geometry'])
        sputum_area = [[44.5, 0], [47.5, 0], [47.5, 1.5], [44.5, 1.5]]
        sputum_area = {
            'geometry': sputum_area,
            'type': 'ZONE',
            'name': 'Sputum'
        }
        sputum_area_polygon = Polygon(sputum_area['geometry'])
        
        # Get list of CSV files
        csv_files = ["2024-07-03.csv"] # [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        # Specify the number of cores to use
        num_cores = 3
        
        # Use ProcessPoolExecutor to parallelize the processing
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
                executor.map(
                    process_file,
                    csv_files,
                    [folder_path] * len(csv_files),
                    [output_folder_path] * len(csv_files),
                    [filt_duration] * len(csv_files),
                    [filt_distance] * len(csv_files),
                    [entry_polygon] * len(csv_files),
                    [seating_polygon] * len(csv_files),
                    [check_area_polygon] * len(csv_files),
                    [check_tb_area_polygon] * len(csv_files),
                    [sputum_area_polygon] * len(csv_files),
                )

