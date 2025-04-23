import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.dates import DateFormatter

def plot_with_background(ax, alpha = .5):
	"""
	Creates a plot with a background image.
	
	Parameters:
	- ax: The matplotlib axis object containing the plot.
	"""
	# Path to your image file
	image_path = 'data-raw/background/image3195.png'
	
	# Coordinates for the image placement
	image_extent = (0, 51, -0.02, 14.214)
	
	# Load the background image
	img = mpimg.imread(image_path)
	
	# If an extent is provided, use it to correctly scale and position the image
	if image_extent:
		ax.imshow(img, aspect='auto', extent=image_extent, zorder=-1, alpha = alpha)
	else:
		ax.imshow(img, aspect='auto', zorder=-1, alpha = alpha)
	return ax


def compute_track_duration(df):
	"""
	Computes the duration of each track.
	
	Parameters:
	- df: A pandas DataFrame with 'track_id' and 'time' columns.
	
	Returns:
	- A pandas DataFrame with 'track_id' and 'duration' for each track.
	"""
	# Calculate the duration by subtracting 4.the first time from the last time for each track
	duration_df = df.groupby('track_id')['time'].apply(lambda x: x.max() - x.min()).reset_index(name='duration')
	
	# Convert duration to a more readable format if needed, e.g., total seconds
	duration_df['duration'] = duration_df['duration'] / 1000 / 60  # Convert milliseconds to minutes
	return duration_df

def plot_track_duration_histogram(duration_df, ax=None):
	"""
	Plots a histogram of the duration of each track for a single dataset.
	
	Parameters:
	- duration_df: A pandas DataFrame with 'track_id' and 'duration' columns.
	"""
	# Plot the histogram
	ax.hist(duration_df['duration'], bins=20, color='skyblue', edgecolor='black', log=True)
	ax.set_title('Histogram of Track Durations')
	ax.set_xlabel('Duration (minutes)')
	ax.set_ylabel('Frequency')
	return ax

def compute_track_distance(df):
	"""
	Computes the total distance of each track using Euclidean distance in a more efficient manner.
	
	Parameters:
	- df: A pandas DataFrame with 'track_id', 'position_x', and 'position_y' columns.
	
	Returns:
	- A pandas DataFrame with 'track_id' and 'total_distance' for each track.
	"""
	# Calculate shifted positions for x and y
	df['shifted_x'] = df.groupby('track_id')['position_x'].shift(-1)
	df['shifted_y'] = df.groupby('track_id')['position_y'].shift(-1)
	
	# Vectorized calculation of the Euclidean distance between consecutive points within each track
	df['distance'] = np.sqrt((df['shifted_x'] - df['position_x'])**2 + (df['shifted_y'] - df['position_y'])**2)
	
	# Drop the last row of each track where the shift results in NaN values
	df.dropna(subset=['shifted_x', 'shifted_y'], inplace=True)
	
	# Sum the distances for each track to get the total distance
	total_distance_df = df.groupby('track_id')['distance'].sum().reset_index(name='total_distance')
	return total_distance_df

def plot_track_distance_histogram(distance_df, ax=None):
	"""
	Plots a histogram of the total distance of each track for a single dataset, using Euclidean distance.
	
	Parameters:
	- distance_df: A pandas DataFrame with 'track_id' and 'total_distance' columns.
	"""
	# Plot the histogram
	ax.hist(distance_df['total_distance'], bins=20, color='skyblue', edgecolor='black', log=True)
	ax.set_title('Histogram of Track Total Distances (Euclidean)')
	ax.set_xlabel('Total Distance (m)')
	ax.set_ylabel('Frequency')
	return ax

def plot_tracks_time(df, ax=None):
	"""
	Counts the number of unique track IDs per second, averages these counts per minute, and plots this on a line graph.
	
	Parameters:
	- df: A pandas DataFrame with columns 'time' (as datetime), 'track_id', 'position_x', and 'position_y'.
	"""
	# Work on a copy of the DataFrame to avoid modifying the original
	df_copy = df.copy()
	
	# Ensure 'time' is in datetime format
	df_copy['time'] = pd.to_datetime(df_copy['time'], unit='ms', origin='unix', utc=True)
	
	# Set 'time' as the index
	df_copy.set_index('time', inplace=True)
	
	# Resample to 1-second intervals, counting unique track IDs in each interval
	track_counts_per_second = df_copy['track_id'].resample('S').nunique()
	
	# Resample to 1-minute intervals, averaging the counts per second
	track_counts_per_minute = track_counts_per_second.resample('T').mean()
	
	# plot
	ax.plot(track_counts_per_minute.index, track_counts_per_minute, color='blue')
	ax.set_title('Average Number of Track IDs per Minute')
	ax.set_xlabel('Time')
	ax.set_ylabel('Average Number of Track IDs')
	ax.grid(True)
	ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	return ax

# Create a plot with the background image
def plot_first_last_tracks(df, ax=None):
	"""
	Plots the first and last track of each track_id in a scatter plot.
	
	Parameters:
	- df: A pandas DataFrame with at least 'track_id', 'position_x', and 'position_y' columns.
	
	Returns:
	- ax: A matplotlib axis object containing the scatter plot.
	"""
	# Ensure the DataFrame is sorted by track_id and then by the tracking time or equivalent
	df_sorted = df.sort_values(by=['track_id', 'time'])
	
	# Group by track_id and get the first and last entry for each track_id
	first_tracks = df_sorted.groupby('track_id').first().reset_index()
	last_tracks = df_sorted.groupby('track_id').last().reset_index()
	
	# Plot the first track points in green
	ax.scatter(first_tracks['position_x'], first_tracks['position_y'], color='green', label='First Track', s = 1)
	
	# Plot the last track points in red
	ax.scatter(last_tracks['position_x'], last_tracks['position_y'], color='red', label='Last Track', s = 1)
	
	# Adding legend to distinguish first and last tracks
	ax.legend()
	
	# Labeling the axes
	ax.set_xlabel('Position X')
	ax.set_ylabel('Position Y')
	ax.set_title('First and Last Tracks of Each Track ID')
	return ax


def check_file(file_path):
	# Extract the date from the file path
	date_str = file_path.split('/')[-1].split('.')[0]
	
	# Read the CSV file
	unlinked_df = pd.read_csv(file_path)
	n_unlink = unlinked_df['track_id'].nunique()
	
	# Create a 2x2 grid of subplots
	fig, axs = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
	
	# Plot each of the four plots in the respective subplot
	plot_track_duration_histogram(compute_track_duration(unlinked_df), ax=axs[0, 0])
	plot_track_distance_histogram(compute_track_distance(unlinked_df), ax=axs[0, 1])
	plot_tracks_time(unlinked_df, ax=axs[1, 0])
	plot_with_background(plot_first_last_tracks(unlinked_df, ax=axs[1, 1]))
	
	# Add a global title with the date and number of tracks
	fig.suptitle(f'Data Check for {date_str} - Number of Tracks: {n_unlink}', fontsize=16)
	
	# Adjust layout to prevent overlap
	plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for the suptitle
	
	# Save the combined plot to a file
	plt.savefig(f'data-check/checks/unlinked/{date_str}.png')

# Loop through all CSV files in the directory
file_paths = glob.glob('data-clean/tracking/unlinked/*.csv')
for file_path in file_paths:
	check_file(file_path)

