import pandas as pd
import numpy as np
import os
import h5py

def rasterize_and_count(file_path, extent, cell_size):
    min_x, max_x, min_y, max_y = extent
    num_cells_x = int((max_x - min_x) / cell_size)
    num_cells_y = int((max_y - min_y) / cell_size)
    counts = np.zeros((num_cells_y, num_cells_x), dtype=int)
    df = pd.read_csv(file_path, usecols=['position_x', 'position_y'])
    df = df[(df['position_x'] >= min_x) & (df['position_x'] < max_x) & (df['position_y'] >= min_y) & (df['position_y'] < max_y)]
    x_indices = ((df['position_x'] - min_x) / cell_size).astype(int)
    y_indices = ((df['position_y'] - min_y) / cell_size).astype(int)
    for x, y in zip(x_indices, y_indices):
        counts[y, x] += 1
    return counts

# Define the extent and cell size
extent = (0, 51, -0.02, 14.214)
cell_size = 0.25  # Adjust the cell size as needed

# Define the directory path
directory_path = 'data-clean/tracking/unlinked/'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# Initialize a list to store the results
results = []

# Apply the function to each file and store the results
for file in csv_files:
    date = file.replace('.csv', '')
    file_path = os.path.join(directory_path, file)
    counts = rasterize_and_count(file_path, extent, cell_size)
    results.append({'date': date, 'counts': counts})

# Save the combined results to an HDF5 file
output_file = 'data-clean/tracking/spatial-density.h5'
with h5py.File(output_file, 'w') as hf:
    for result in results:
        hf.create_dataset(result['date'], data=result['counts'])

print(f"Combined results saved to {output_file}")