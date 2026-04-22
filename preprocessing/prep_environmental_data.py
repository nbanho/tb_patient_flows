"""Consolidate raw Aranet4 CO2 sensor exports into a single cleaned CSV.

Reads CO2, temperature, humidity, and pressure data from two Aranet4 devices
deployed in the hospital waiting area, removes duplicates from overlapping
export files, filters to study hours (6 AM - 6 PM) and study dates, and
computes an approximate outdoor CO2 level as the 5th percentile per day.

Reads from:  data-raw/co2/  (Aranet4 CSV exports)
Writes to:   data-clean/environmental/co2-temp-humidity.csv
"""

import os
import pandas as pd


if __name__ == "__main__":
    co2_folder_path = 'data-raw/co2/'
    tracking_folder_path = 'data-clean/tracking/unlinked/'

    co2_csv_files = [f for f in os.listdir(co2_folder_path) if f.endswith('.csv')]

    # Study dates are inferred from the tracking data files
    tracking_csv_files = [f for f in os.listdir(tracking_folder_path) if f.endswith('.csv')]
    tracking_dates = [f.split('.')[0] for f in tracking_csv_files]

    dataframes = []

    # Track the latest datetime per device to remove overlapping records
    # between consecutive Aranet4 export files
    latest_datetime = {'Aranet4 272D2': None, 'Aranet4 25247': None}

    # Two CO2 monitors deployed in the waiting area
    device_files = {'Aranet4 272D2': [], 'Aranet4 25247': []}
    for file in co2_csv_files:
        if '272D2' in file:
            device_files['Aranet4 272D2'].append(file)
        elif '25247' in file:
            device_files['Aranet4 25247'].append(file)

    for device in device_files:
        device_files[device].sort(key=lambda x: pd.to_datetime(x.split('_')[1][:10], format='%Y-%m-%d'))

    for device, files in device_files.items():
        for file in files:
            print(file)

            df = pd.read_csv(os.path.join(co2_folder_path, file))

            # Parse the datetime column (Aranet4 exports use varying formats)
            datetime_col = df.columns[0]
            try:
                df['datetime'] = pd.to_datetime(df[datetime_col], format='%d/%m/%Y %I:%M:%S %p')
            except ValueError:
                try:
                    df['datetime'] = pd.to_datetime(df[datetime_col], format='%d/%m/%Y %H:%M:%S')
                except ValueError:
                    df['datetime'] = pd.to_datetime(df[datetime_col], format='%d.%m.%Y %H:%M')

            # Remove rows overlapping with previous export file
            if latest_datetime[device] is not None:
                df = df[df['datetime'] > latest_datetime[device]]

            if not df.empty:
                latest_datetime[device] = df['datetime'].max()

            df['device'] = device
            df = df[['device', 'datetime', 'Carbon dioxide(ppm)', 'Temperature(°C)', 'Relative humidity(%)', 'Atmospheric pressure(hPa)']]
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.columns = ['device', 'datetime', 'co2', 'temp', 'humidity', 'pressure']

    # Approximate outdoor CO2 as the 5th percentile of daily readings,
    # assuming the lowest readings correspond to well-ventilated periods.
    combined_df['co2_outdoor'] = combined_df.groupby(['device', combined_df['datetime'].dt.date])['co2'].transform(lambda x: x.quantile(0.05))

    # Filter to study hours (6 AM - 6 PM) matching the tracking data window
    combined_df = combined_df[(combined_df['datetime'].dt.time >= pd.to_datetime('06:00:00').time()) &
                              (combined_df['datetime'].dt.time <= pd.to_datetime('18:00:00').time())]

    # Keep only dates present in the tracking data
    combined_df['date'] = combined_df['datetime'].dt.strftime('%Y-%m-%d')
    combined_df = combined_df[combined_df['date'].isin(tracking_dates)]
    combined_df = combined_df.drop(columns=['date'])

    combined_df.to_csv('data-clean/environmental/co2-temp-humidity.csv', index=False, header=True)
