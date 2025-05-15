import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os


def detect_outliers(data, window_size=5, std_threshold=3.0):
    """
    Detect outliers in the data series using a rolling window approach.
    
    Args:
        data: The data series to check for outliers
        window_size: Size of the rolling window
        std_threshold: Number of standard deviations to consider a point an outlier
        
    Returns:
        Boolean mask where True indicates outliers
    """
    # Calculate rolling median and standard deviation
    rolling_median = data.rolling(window=window_size, center=True).median()
    rolling_std = data.rolling(window=window_size, center=True).std()
    
    # For the first and last few points where rolling calculations don't work well
    rolling_median.fillna(method='bfill').fillna(method='ffill', inplace=True)
    rolling_std.fillna(rolling_std.mean(), inplace=True)
    
    # Identify outliers as points that are too far from the rolling median
    distance = np.abs(data - rolling_median)
    outlier_mask = distance > (std_threshold * rolling_std)
    
    # Handle cases where std is very small (near zero)
    min_std_threshold = 10.0  # Minimum threshold for absolute changes
    absolute_outliers = distance > min_std_threshold
    
    return outlier_mask | absolute_outliers


def smooth_data(data, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth the data
    
    Args:
        data: Data series to smooth
        window_length: Window length for the filter (must be odd)
        polyorder: Polynomial order for the filter
        
    Returns:
        Smoothed data series
    """
    # Make sure we have enough data points for the window
    if len(data) < window_length:
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        
    # Make sure window_length is odd
    if window_length % 2 == 0:
        window_length -= 1
        
    # Make sure polyorder is less than window_length
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    # If we have very few data points, revert to simple moving average
    if window_length < 5:
        return data.rolling(window=max(3, len(data)//2), center=True).mean().fillna(data.mean())
    
    return savgol_filter(data, window_length, polyorder)


def process_angle_data(csv_path, output_dir=None):
    """
    Process angle data from CSV file: filter, smooth and calculate key statistics only
    
    Args:
        csv_path: Path to the CSV file with angle data
        output_dir: Directory to save output file (default: same as input file)
        
    Returns:
        Dictionary with key statistics
    """
    print(f"Processing file: {csv_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    required_columns = ['frame', 'PelvicRocking', 'SpinalAsymmetry']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        # Try to infer column names
        if len(df.columns) >= 3:  # Expect at least frame, PelvicRocking, SpinalAsymmetry
            print(f"Warning: Missing columns {missing_columns}. Trying to infer column names.")
            new_columns = ['frame', 'PelvicRocking', 'SpinalAsymmetry']
            # Only replace first columns up to 3
            df.columns = new_columns + list(df.columns[3:]) if len(df.columns) > 3 else new_columns[:len(df.columns)]
        else:
            raise ValueError(f"Missing required columns and unable to infer: {missing_columns}")
    
    # Process each angle column
    angle_columns = ['PelvicRocking', 'SpinalAsymmetry']
    key_stats = {}
    
    for column in angle_columns:
        print(f"\nProcessing {column}...")
        
        # Step 1: Detect and handle outliers
        outliers = detect_outliers(df[column], window_size=5, std_threshold=3.0)
        outliers_count = outliers.sum()
        print(f"  - Detected {outliers_count} outliers ({outliers_count/len(df)*100:.1f}% of data)")
        
        # Create a copy of the data for processing
        processed_values = df[column].copy()
        
        # Replace outliers with NaN
        processed_values[outliers] = np.nan
        
        # Interpolate missing values
        processed_values = processed_values.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        
        # Step 2: Apply smoothing
        smoothed_values = smooth_data(processed_values, window_length=11, polyorder=3)
        
        # Step 3: Calculate statistics
        processed_mean = smoothed_values.mean()
        
        # Calculate the adjusted mean (mean + 90)
        adjusted_mean = processed_mean + 90
        
        # For PelvicRocking, also calculate max-min range
        if column == 'PelvicRocking':
            value_range = np.max(smoothed_values) - np.min(smoothed_values)
            
            key_stats[column] = {
                'adjusted_mean': adjusted_mean,
                'value_range': value_range
            }
            
            print(f"  - Adjusted Mean (Mean + 90): {adjusted_mean:.2f}°")
            print(f"  - Range (Max - Min): {value_range:.2f}°")
            
        else:  # SpinalAsymmetry
            key_stats[column] = {
                'adjusted_mean': adjusted_mean
            }
            
            print(f"  - Adjusted Mean (Mean + 90): {adjusted_mean:.2f}°")
    
    # Save key stats to CSV format with specified fields
    stats_df = pd.DataFrame({
        'pelvis mean': [key_stats['PelvicRocking']['adjusted_mean']],
        'pelvis range': [key_stats['PelvicRocking']['value_range']],
        'spine mean': [key_stats['SpinalAsymmetry']['adjusted_mean']]
    })
    
    # Define the CSV path
    csv_stats_path = os.path.join(output_dir, "back_features.csv")

        # If the file does not exist, create it with the correct columns (same as stats_df)
    if not os.path.exists(csv_stats_path):
            empty_df = pd.DataFrame(columns=stats_df.columns)
            empty_df.to_csv(csv_stats_path, index=False)

        # Now overwrite it with new stats
    stats_df.to_csv(csv_stats_path, index=False)
    print(f"Key statistics saved to CSV: {csv_stats_path}")
    
    return key_stats


def main():
    # MODIFY HERE: Hardcoded file path - change this to your desired CSV file
    csv_file = r"back_data.csv"
    
    # You can also set a specific output directory (optional)
    output_dir = None  # Set to None to use the same directory as the input file
    
    try:
        key_stats = process_angle_data(csv_file, output_dir)
        
        # Delete the original data file after processing
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"\nDeleted original data file: {csv_file}")
        else:
            print(f"\nOriginal data file not found: {csv_file}")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    main()
