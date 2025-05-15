import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os

# Read the CSV data
df = pd.read_csv('all_data.csv')

# Apply Savitzky-Golay filter to smooth the data and remove spikes
# Window size of 15 (odd number) and polynomial order of 3
columns_to_filter = ['Knee', 'Elbow', 'Back', 'Neck', 'Reach', 'Hip_Ankle_Center_Distance']
filtered_df = df.copy()

for column in columns_to_filter:
    filtered_df[column] = savgol_filter(df[column], window_length=15, polyorder=3)

# Calculate min and max knee angle
min_knee = filtered_df['Knee'].min()
max_knee = filtered_df['Knee'].max()

# Calculate averages
avg_elbow = filtered_df['Elbow'].mean()
avg_neck = filtered_df['Neck'].mean()
avg_back = filtered_df['Back'].mean()
avg_reach = filtered_df['Reach'].mean()
avg_hip_ankle = filtered_df['Hip_Ankle_Center_Distance'].mean()

# Create a dictionary with the feature data
features_dict = {
    'min_knee_angle': [min_knee],
    'max_knee_angle': [max_knee],
    'avg_elbow_angle': [avg_elbow],
    'avg_neck_angle': [avg_neck],
    'avg_back_angle': [avg_back],
    'avg_reach': [avg_reach],
    'avg_saddle_offset': [avg_hip_ankle]
}

# Create a DataFrame with the features
features_df = pd.DataFrame(features_dict)
output_path = "side_features.csv"
# If the file does not exist, create it with the correct structure
if not os.path.exists(output_path):
    empty_df = pd.DataFrame(columns=features_df.columns)
    empty_df.to_csv(output_path, index=False)

# Now save the actual data (overwrites the file)
features_df.to_csv(output_path, index=False)
print(f"Side features saved to CSV: {output_path}")

# Delete the original file
os.remove('all_data.csv')

print("Processing complete. Results saved to side_features.csv")
