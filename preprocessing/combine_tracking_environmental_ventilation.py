import os
import pandas as pd
from modelling.create_building_mask import *
from concurrent.futures import ProcessPoolExecutor

#### Combine tracking and CO2 data ####

# Check if the combined file exists
combined_file_path = 'data-clean/environmental/co2-occupancy.csv'

def process_group(device, date, group, occupancy):
    mean_no_people_list = [0]  # Initialize with zero for the first row
    group = group.sort_values(by='datetime')  # Ensure datetime is ordered

    for i in range(1, len(group)):
        # Get the current and previous datetime
        current_datetime = group.iloc[i]['datetime']
        previous_datetime = group.iloc[i - 1]['datetime']

        # Filter the occupancy DataFrame for the current date and time interval
        occupancy_filtered = occupancy[(occupancy['datetime'].dt.date == current_datetime.date()) &
                                       (occupancy['datetime'] >= previous_datetime) &
                                       (occupancy['datetime'] < current_datetime)]

        # Compute the mean number of no_people for the interval
        mean_no_people = occupancy_filtered['no_people'].mean()

        # Append the mean number of no_people to the list
        mean_no_people_list.append(mean_no_people)

    group['no_people'] = mean_no_people_list
    return group

def process_group_wrapper(args):
    return process_group(*args)

if __name__ == "__main__":
    if not os.path.exists(combined_file_path):
        # Read the CSV files into DataFrames
        occupancy = pd.read_csv('data-clean/tracking/occupancy.csv')
        co2 = pd.read_csv('data-clean/environmental/co2-temp-humidity.csv')
        
        # Convert the time columns to datetime format
        occupancy['datetime'] = pd.to_datetime(occupancy['datetime'])
        co2['datetime'] = pd.to_datetime(co2['datetime'])
        
        # Group the co2 DataFrame by device and date
        co2['date'] = co2['datetime'].dt.date
        grouped_co2 = co2.groupby(['device', 'date'])
        
        # Use parallel processing to speed up the computation
        with ProcessPoolExecutor(max_workers=20) as executor:
            results = executor.map(process_group_wrapper, [(device, date, group, occupancy) for (device, date), group in grouped_co2])
            
        # Combine the results into a single DataFrame
        combined_df = pd.concat(results)
        
        # Subset
        combined_df = combined_df.rename(columns={"Carbon dioxide(ppm)": "co2"})
        combined_df = combined_df[['device', 'datetime', 'co2', 'no_people']]
        
        # Save the updated co2 DataFrame to a new CSV file
        combined_df.to_csv(combined_file_path, index=False, header=True)
    else:
        # Load the combined file
        co2 = pd.read_csv(combined_file_path)
        co2['datetime'] = pd.to_datetime(co2['datetime'])