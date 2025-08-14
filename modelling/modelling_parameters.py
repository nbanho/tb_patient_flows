import numpy as np
import glob
import pandas as pd
import pickle
import os

# Seed
np.random.seed(42)  # For reproducibility

# Number of samples
n_samples = 1000

# Quanta generation rates
mean_waiting = 0.07162337
sigma_waiting = 2.99399
mean_walking = 0.2058702
sigma_walking = 2.99252

# Load and combine all CSV files
csv_files = glob.glob('data-clean/tracking/linked-tb/*.csv')
linked_tb_df = pd.concat(
    [df for df in (pd.read_csv(f) for f in csv_files) if 'category' in df.columns and (df['category'] == "sure").any()],
    ignore_index=True
)
linked_tb_df = linked_tb_df[linked_tb_df['category'] == "sure"]

# Get unique new_track_id
unique_track_ids = linked_tb_df['new_track_id'].unique()

# Sample 1,000 quanta generation rates per track_id from lognormal distributions
quanta_waiting = {
	track_id: np.random.lognormal(mean=mean_waiting, sigma=sigma_waiting, size=n_samples) / 3600
	for track_id in unique_track_ids
}
quanta_walking = {
	track_id: np.random.lognormal(mean=mean_walking, sigma=sigma_walking, size=n_samples) / 3600
	for track_id in unique_track_ids
}

output_dir = 'data-clean/assumptions'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'quanta_waiting.pkl'), 'wb') as f:
    pickle.dump(quanta_waiting, f)

with open(os.path.join(output_dir, 'quanta_walking.pkl'), 'wb') as f:
    pickle.dump(quanta_walking, f)

# Removal rate parameters
settling = np.random.lognormal(mean=0.3624846, sigma=0.517269, size=n_samples) 
settling = settling / 3600
inactivation = np.random.lognormal(mean=0.0008491922, sigma=0.9993368, size=n_samples) 
inactivation = inactivation / 3600

with open(os.path.join(output_dir, 'settling.pkl'), 'wb') as f:
    pickle.dump(settling, f)

with open(os.path.join(output_dir, 'inactivation.pkl'), 'wb') as f:
    pickle.dump(inactivation, f)
    
# Breathing rate parameters: mean across men and women
inhalation_rate_waiting = 0.5 * 0.4632 + 0.5 * 0.5580  # m^3/h
inhalation_rate_waiting = inhalation_rate_waiting / 3600  # m^3/s
inhalation_rate_walking = 0.5 * 1.2192 + 0.5 * 1.4478  # m^3/h
inhalation_rate_walking = inhalation_rate_walking / 3600  # m^3/s
inhalation_rates = (inhalation_rate_waiting, inhalation_rate_walking)
with open(os.path.join(output_dir, 'inhalation_rates.pkl'), 'wb') as f:
    pickle.dump(inhalation_rates, f)