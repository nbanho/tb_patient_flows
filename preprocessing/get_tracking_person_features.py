# Libraries
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import os
import time
from read_tracking_linked_data import read_linked_tracking_data 
import argparse
import concurrent.futures
from functools import partial

# Get gender and healthcare worker status per person
def add_person_features(result_df, non_tb_df):
    # --- Gender ---
    if non_tb_df['gender'].isna().all():
        # whole column missing → all NA
        result_df['female'] = pd.NA
    else:
        def majority_gender(genders):
            # drop NA
            genders = genders.dropna()
            # drop NOT_SURE
            genders = genders[genders.isin(['MALE', 'FEMALE'])]
            if genders.empty:
                return pd.NA
            # majority vote
            return genders.value_counts().idxmax()

        gender_map = non_tb_df.groupby('new_track_id')['gender'].apply(majority_gender)
        result_df = result_df.merge(gender_map.rename('gender'),
                                    left_on='new_track_id', right_index=True, how='left')
        result_df['female'] = result_df['gender'].map({'FEMALE': 1, 'MALE': 0}).astype('Int64')
        result_df.drop(columns=['gender'], inplace=True)

    # --- Tag / Healthcare worker ---
    if non_tb_df['gender'].isna().all(): # use gender because tag was still provided as False
        result_df['healthcare_worker'] = pd.NA
    else:
        def has_healthcare_worker(tags):
            tags = tags.dropna().astype(str)
            return 1 if (tags == 'True').any() else 0

        tag_map = non_tb_df.groupby('new_track_id')['tag'].apply(has_healthcare_worker)
        result_df = result_df.merge(tag_map.rename('healthcare_worker'),
                                    left_on='new_track_id', right_index=True, how='left')
        result_df['healthcare_worker'] = result_df['healthcare_worker'].astype('Int64')

    return result_df


# loop over dates and process
linked_tb_path = 'data-clean/tracking/linked-tb/'
dates = [
    file.replace('.csv', '') 
    for file in os.listdir(linked_tb_path) 
    if file.endswith('.csv')
]
result_list = []

for date in dates:
    print(f'Processing date: {date}')
    
    # Read tracking data
    df = read_linked_tracking_data(date)
    
    # Filter non-TB patients
    non_tb_df = df[df['clinic_id'].isna()]
    
    # Resulting dataframe with features
    result_df = pd.DataFrame({'new_track_id': non_tb_df['new_track_id'].unique()})
    result_df['date'] = date
    
    # Add person features
    result_df = add_person_features(result_df, non_tb_df)
    result_list.append(result_df)
    
# combine all dates
final_df = pd.concat(result_list, ignore_index=True)
output_file = os.path.join('data-clean/tracking/', 'person-features.csv')
final_df.to_csv(output_file, index=False)