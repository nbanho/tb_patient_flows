"""Create a date-to-study-phase mapping CSV for all study dates.

The study ran from June 17 to August 22, 2024, with four phases:
  - Baseline (pre-intervention):  June 17 - June 30
  - First intervention:           July 1  - July 15   (optimized waiting-area layout)
  - Second intervention:          July 16 - July 29   (added one-way patient flow)
  - Baseline (post-intervention): July 30 - August 22

Reads from:  data-clean/tracking/unlinked/  (date list inferred from filenames)
Writes to:   data-clean/mapping_dates_interventions.csv
"""

import pandas as pd
import os
from datetime import datetime


def determine_study_phase(date):
    """Map a date to its study phase based on the intervention timeline."""
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


if __name__ == "__main__":
    base_path = 'data-clean/tracking/unlinked'
    files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
    dates = [datetime.strptime(f.replace('.csv', ''), '%Y-%m-%d') for f in files]

    dates = {'date': dates, 'study_phase': [determine_study_phase(date) for date in dates]}
    dates = pd.DataFrame(dates)

    output_file = 'data-clean/mapping_dates_interventions.csv'
    dates.to_csv(output_file, index=False, header=True)
