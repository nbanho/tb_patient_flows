import pandas as pd
import os
from datetime import datetime

# Files
base_path = 'data-clean/tracking/unlinked'
files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
dates = [datetime.strptime(f.replace('.csv', ''), '%Y-%m-%d') for f in files]

# Map study phase
def determine_study_phase(date):
    if datetime(2024, 6, 17) <= date <= datetime(2024, 6, 30):
        return 'Baseline'
    elif datetime(2024, 7, 1) <= date <= datetime(2024, 7, 15):
        return 'First intervention'
    elif datetime(2024, 7, 16) <= date <= datetime(2024, 7, 29):
        return 'Second intervention'
    elif datetime(2024, 7, 30) <= date <= datetime(2024, 8, 23):
        return 'Baseline'
    else:
        return 'Unknown'
dates = {'date': dates, 'study_phase': [determine_study_phase(date) for date in dates]}
dates = pd.DataFrame(dates)

# Save the combined DataFrame to a CSV file
output_file = 'data-clean/mapping_dates_interventions.csv'
dates.to_csv(output_file, index=False, header=True)