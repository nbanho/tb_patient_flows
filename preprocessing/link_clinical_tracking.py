# libraries
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import os

# settings
save_path = 'data-clean/tracking/linked-clinical/'
linked_tb_path = 'data-clean/tracking/linked-tb/'
dates = [
    file.replace('.csv', '') 
    for file in os.listdir(linked_tb_path) 
    if file.endswith('.csv')
]

for date in dates:
    print(date)
    
    # clinical data
    clinical = pd.read_csv('data-clean/clinical/tb_cases.csv')
    clinical = clinical[clinical['date'] == date].sort_values(by='start_time', ascending=True)
    clinical['order'] = range(1, len(clinical) + 1)
    clinical = clinical[clinical['tb_status'].isin(['presumptive', 'infectious'])]
    clinical = clinical[['clinic_id', 'order']]

    # tracking data
    base_path = 'data-clean/tracking/'
    linked_file = os.path.join(base_path, 'linked', f'{date}.csv')
    linked_tb_file = os.path.join(base_path, 'linked-tb', f'{date}.csv')
    linked_df = pd.read_csv(linked_file)
    linked_tb_df = pd.read_csv(linked_tb_file)

    # link
    linked_tb_df['order'] = linked_tb_df['order'].replace("-", 0).fillna(0).astype(int)
    linked_tb_df = pd.merge(linked_tb_df, clinical, on='order', how='left')
    linked_tb_df = linked_tb_df[['track_id', 'new_track_id', 'clinic_id']]
    linked_df = pd.merge(linked_df, linked_tb_df, on='track_id', how='left')
    
    # save linked_df
    output_file = os.path.join(save_path, f'{date}.csv')
    linked_df.to_csv(output_file, index=False)