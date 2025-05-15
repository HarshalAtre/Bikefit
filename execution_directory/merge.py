import pandas as pd

def main():
    # Read the CSV files
    side_features = pd.read_csv('side_features.csv')
    back_features = pd.read_csv('back_features.csv')
    user_input = pd.read_csv('user_input.csv')
    
    # Create a new DataFrame with the required fields
    combined_data = pd.DataFrame({
        'min_knee_angle': [side_features['min_knee_angle'].iloc[0]],
        'max_knee_angle': [side_features['max_knee_angle'].iloc[0]],
        'avg_elbow_angle': [side_features['avg_elbow_angle'].iloc[0]],
        'avg_neck_angle': [side_features['avg_neck_angle'].iloc[0]],
        'avg_back_angle': [side_features['avg_back_angle'].iloc[0]],
        'reach': [side_features['avg_reach'].iloc[0]],  # renamed from avg_reach
        'horizontal_distance': [side_features['avg_saddle_offset'].iloc[0]],  # renamed from avg_saddle_offset
        'mean_pelvis_angle': [back_features['pelvis mean'].iloc[0]],
        'pelvis_angle_deviation': [back_features['pelvis range'].iloc[0]],  # renamed from pelvis range
        'mean_spine_angle': [back_features['spine mean'].iloc[0]],
        'rider_arm_length': [user_input['rider_arm_length'].iloc[0]],
        'rider_torso_length': [user_input['rider_torso_length'].iloc[0]],
        'rider_inseam_length': [user_input['rider_inseam_length'].iloc[0]]
    })
    
    # Write the combined data to a new CSV file
    combined_data.to_csv('13_samples_input.csv', index=False)
    print("Combined CSV file created successfully as '13_samples_input.csv'")

if __name__ == "__main__":
    main()
