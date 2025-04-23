import os
import pandas as pd

# Define the folder paths
co2_folder_path = 'data-raw/co2/'
tracking_folder_path = 'data-clean/tracking/unlinked/'

# List all CSV files in the co2 folder
co2_csv_files = [f for f in os.listdir(co2_folder_path) if f.endswith('.csv')]

# List all CSV files in the tracking folder to filter dates
tracking_csv_files = [f for f in os.listdir(tracking_folder_path) if f.endswith('.csv')]
tracking_dates = [f.split('.')[0] for f in tracking_csv_files]

# Initialize an empty list to store DataFrames
dataframes = []

# Initialize a dictionary to store the latest datetime for each device
latest_datetime = {'Aranet4 272D2': None, 'Aranet4 25247': None}

# Group files by device
device_files = {'Aranet4 272D2': [], 'Aranet4 25247': []}
for file in co2_csv_files:
    if '272D2' in file:
        device_files['Aranet4 272D2'].append(file)
    elif '25247' in file:
        device_files['Aranet4 25247'].append(file)

# Sort files by date for each device
for device in device_files:
    device_files[device].sort(key=lambda x: pd.to_datetime(x.split('_')[1][:10], format='%Y-%m-%d'))

# Loop through each device and its files
for device, files in device_files.items():
    for file in files:
        print(file)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(co2_folder_path, file))
        
        # Parse the datetime column with multiple formats
        datetime_col = df.columns[0]
        try:
            df['datetime'] = pd.to_datetime(df[datetime_col], format='%d/%m/%Y %I:%M:%S %p')
        except ValueError:
            try:
                df['datetime'] = pd.to_datetime(df[datetime_col], format='%d/%m/%Y %H:%M:%S')
            except ValueError:
                df['datetime'] = pd.to_datetime(df[datetime_col], format='%d.%m.%Y %H:%M')
        
        # Remove rows with datetime smaller than the latest datetime for the device
        if latest_datetime[device] is not None:
            df = df[df['datetime'] > latest_datetime[device]]
        
        # Update the latest datetime for the device
        if not df.empty:
            latest_datetime[device] = df['datetime'].max()
        
        # Add a column for the device
        df['device'] = device
        
        # Subset
        df = df[['device', 'datetime', 'Carbon dioxide(ppm)', 'Temperature(°C)', 'Relative humidity(%)', 'Atmospheric pressure(hPa)']]
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Rename columns for consistency
combined_df.columns = ['device', 'datetime', 'co2', 'temp', 'humidity', 'pressure']

# Compute the outdoor CO2 level as the bottom 5% of the CO2 levels per day and device
combined_df['co2_outdoor'] = combined_df.groupby(['device', combined_df['datetime'].dt.date])['co2'].transform(lambda x: x.quantile(0.05))

# Filter the data between 6am and 6pm
combined_df = combined_df[(combined_df['datetime'].dt.time >= pd.to_datetime('06:00:00').time()) & 
                          (combined_df['datetime'].dt.time <= pd.to_datetime('18:00:00').time())]

# Filter dates based on the tracking CSV files
combined_df['date'] = combined_df['datetime'].dt.strftime('%Y-%m-%d')
combined_df = combined_df[combined_df['date'].isin(tracking_dates)]

# Drop the temporary 'date' column
combined_df = combined_df.drop(columns=['date'])

# Save the combined DataFrame to a CSV file
combined_df.to_csv('data-clean/environmental/co2-temp-humidity.csv', index=False, header=True)