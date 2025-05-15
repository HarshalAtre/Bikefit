import cv2
import numpy as np
import pandas as pd
import os
import glob
import shutil
import subprocess
from datetime import datetime
import math


class CalibratedSkeletonTracker:
    def __init__(self):
        # Marker ID to body part mapping
        self.marker_map = {
            0: "Head",
            1: "Shoulder",
            2: "Elbow",
            3: "Wrist",
            7: "MidBack",
            4: "Hip",
            5: "Knee",
            6: "Ankle"
        }

        # Define connections between body parts
        self.connections = [
            (0, 1),  # Head to Shoulder
            (1, 2),  # Shoulder to Elbow
            (2, 3),  # Elbow to Wrist
            (1, 7),  # Shoulder to MidBack
            (7, 4),  # MidBack to Hip
            (4, 5),  # Hip to Knee
            (5, 6)   # Knee to Ankle
        ]

        # Define angles to track with direction specification
        # Name, point1_id, vertex_id, point2_id, direction (clockwise or anticlockwise)
        self.angles_to_track = [
            ("Knee", 4, 5, 6, "anticlockwise"),        # Hip-Knee-Ankle (clockwise)
            ("Elbow", 1, 2, 3, "clockwise"),   # Shoulder-Elbow-Wrist (anticlockwise)
            ("Back", 1, 7, 4, "clockwise"),    # Shoulder-MidBack-Hip (anticlockwise)
            ("Neck", 0, 1, 7, "anticlockwise")         # Head-Shoulder-MidBack (clockwise)
        ]

        # Distances to track
        self.distances_to_track = [
            # Name, point1_id, point2_id
            ("Reach", 1, 3)  # Shoulder to Wrist (horizontal distance between vertical lines)
        ]

        # Colors for visualization
        self.colors = {
            "skeleton": (0, 255, 0),     # Green
            "markers": (0, 0, 255),      # Red
            "marker_text": (0, 255, 0),  # Green for marker names
            "angle_text": (0, 255, 0),   # Green for angles
            "angle_arc": (255, 165, 0),  # Orange for angle arcs
            "distance_text": (255, 200, 0),  # Yellow for distances
            "frame_text": (255, 255, 255),  # White
            "center_line": (255, 0, 255),   # Magenta for center lines
            "ankle_center": (0, 255, 255)   # Cyan for ankle center
        }

        # Data storage
        self.angle_data = []
        self.distance_data = []
        self.center_distance_data = []  # New data storage for center distance
        self.frame_count = 0
        
        # Camera calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        # Pixel to cm conversion (will be calculated when markers are detected)
        self.pixel_to_cm_ratio = None
        self.marker_size_cm = 6.4  # Default marker size is 6.4 cm
        
        # Ankle path tracking
        self.ankle_positions = []
        self.ankle_center = None
        self.ankle_center_calculated = False

    def create_calibration(self, calibration_folder, save_path="camera_calibration.npz"):
        """Create new calibration from charuco board images"""
        if not os.path.exists(calibration_folder):
            print(f"Calibration folder {calibration_folder} does not exist")
            return False
            
        # Get all calibration images
        images = glob.glob(os.path.join(calibration_folder, "*.jpg")) + \
                 glob.glob(os.path.join(calibration_folder, "*.png"))
                 
        if not images:
            print("No calibration images found")
            return False
            
        print(f"Found {len(images)} calibration images")
        
        # Set up calibration
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard((10, 7), 100, 80, aruco_dict)
        detector = cv2.aruco.CharucoDetector(board)
        
        # Storage for charuco corners and ids
        all_corners = []
        all_ids = []
        
        # Process each calibration image
        for img_path in images:
            print(f"Processing {img_path}")
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image {img_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect charuco board
            charucoCorners, charucoIds, markerCorners, markerIds = detector.detectBoard(gray)
            
            if charucoCorners is not None and charucoIds is not None and len(charucoIds) > 4:
                all_corners.append(charucoCorners)
                all_ids.append(charucoIds)
                
        
        if len(all_corners) < 5:
            print("Not enough valid calibration images (need at least 5)")
            return False
            
        # Get image size from first image
        img_sample = cv2.imread(images[0])
        img_size = (img_sample.shape[1], img_sample.shape[0])
        
        # Calibrate camera
        calibration_flags = cv2.CALIB_RATIONAL_MODEL
        
        try:
            print("Calibrating camera...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, board, img_size, None, None, flags=calibration_flags)
                
            print(f"Calibration completed with error: {ret}")
            
            # Save calibration parameters
            np.savez(save_path, 
                     camera_matrix=camera_matrix, 
                     dist_coeffs=dist_coeffs)
                     
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.calibrated = True
            
            print(f"Calibration saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False

    def calculate_angle(self, p1, p2, p3, direction="clockwise"):
        """
        Calculate angle between three points based on specified direction
        
        Args:
            p1, p2, p3: Three points where p2 is the vertex
            direction: 'clockwise' or 'anticlockwise' rotation from p1-p2 to p2-p3
            
        Returns:
            Angle in degrees (0-360)
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        # Create vectors
        ba = a - b
        bc = c - b
        
        # Calculate the raw angle using arctan2
        angle1 = np.arctan2(ba[1], ba[0])
        angle2 = np.arctan2(bc[1], bc[0])
        
        # Calculate the signed angle difference
        angle_diff = angle2 - angle1
        
        # Adjust for 2π periodicity
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
            
        # Convert to degrees
        angle_deg = np.degrees(angle_diff)
        
        # Adjust based on direction requirement
        if direction == "clockwise":
            if angle_deg > 0:
                angle_deg = 360 - angle_deg
            else:
                angle_deg = -angle_deg
        elif direction == "anticlockwise":
            if angle_deg < 0:
                angle_deg = 360 + angle_deg
            # else angle_deg is already positive, so keep it
            
        return angle_deg
    
    def calculate_horizontal_distance(self, p1, p2):
        """Calculate horizontal distance between two points in cm"""
        pixel_distance = abs(p2[0] - p1[0])
        if self.pixel_to_cm_ratio:
            return pixel_distance * self.pixel_to_cm_ratio
        return pixel_distance
        
    def calculate_vertical_distance(self, p1, p2):
        """Calculate vertical distance between two points in cm"""
        pixel_distance = abs(p2[1] - p1[1])
        if self.pixel_to_cm_ratio:
            return pixel_distance * self.pixel_to_cm_ratio
        return pixel_distance
        
    def calculate_distance(self, p1, p2, axis="euclidean"):
        """Calculate distance between two points based on specified axis in cm"""
        if axis == "horizontal":
            return self.calculate_horizontal_distance(p1, p2)
        elif axis == "vertical":
            return self.calculate_vertical_distance(p1, p2)
        else:  # euclidean
            pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
            if self.pixel_to_cm_ratio:
                return pixel_distance * self.pixel_to_cm_ratio
            return pixel_distance
    
    def calculate_marker_size(self, corners):
        """Calculate the average marker size in pixels"""
        if not corners:
            return None
            
        # Calculate the average side length of the marker
        total_side_length = 0
        num_sides = 0
        
        for marker_corners in corners:
            corner_points = marker_corners[0]
            # Calculate the length of each side
            for i in range(4):
                p1 = corner_points[i]
                p2 = corner_points[(i + 1) % 4]
                side_length = np.linalg.norm(p1 - p2)
                total_side_length += side_length
                num_sides += 1
                
        if num_sides > 0:
            average_side_length = total_side_length / num_sides
            # Calculate the pixel to cm ratio
            self.pixel_to_cm_ratio = self.marker_size_cm / average_side_length
            print(f"Pixel to cm ratio: {self.pixel_to_cm_ratio:.4f} cm/pixel")
            return average_side_length
        
        return None

    def calculate_ankle_center(self):
        """Calculate the center of ankle circular movement using least squares circle fit"""
        if len(self.ankle_positions) < 10:  # Need enough points for a good fit
            return None
            
        # Convert to numpy array for easier calculations
        points = np.array(self.ankle_positions)
        
        # Initial guess: mean of the points
        x_m = np.mean(points[:, 0])
        y_m = np.mean(points[:, 1])
        
        # Center the data
        u = points[:, 0] - x_m
        v = points[:, 1] - y_m
        
        # Linear system defining the center (uc, vc) in terms of
        # the first eigenvector of a data covariance matrix
        A = np.vstack([u, v]).T
        B = np.sum(u**2 + v**2) / 2.0 * np.ones(len(u))
        
        # Solve the linear system
        try:
            center, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            
            # Center of the circle
            center_x = center[0] + x_m
            center_y = center[1] + y_m
            
            return (int(center_x), int(center_y))
        except np.linalg.LinAlgError:
            # If the system is singular, return the mean as fallback
            return (int(x_m), int(y_m))

    def process_video(self, video_path):
        """Process video file with ArUco markers to collect skeleton data"""
        if not self.calibrated:
            print("Camera not calibrated. Please run calibration first.")
            return
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.frame_count = 0
        self.angle_data = []
        self.distance_data = []
        self.center_distance_data = []
        self.ankle_positions = []
        self.ankle_center = None
        self.ankle_center_calculated = False

        # First pass: Track ankle positions for center calculation
        print("First pass: Tracking ankle positions...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Undistort the frame using calibration parameters
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(gray)
            
            # Calculate pixel to cm ratio if not already calculated
            if corners and self.pixel_to_cm_ratio is None:
                self.calculate_marker_size(corners)
            
            if ids is not None:
                # Check if ankle marker (ID 6) is detected
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == 6:  # Ankle marker
                        # Calculate center of the marker
                        center_x = int(np.mean([corners[i][0][j][0] for j in range(4)]))
                        center_y = int(np.mean([corners[i][0][j][1] for j in range(4)]))
                        self.ankle_positions.append((center_x, center_y))
                        break
            
            self.frame_count += 1
            
        # Calculate the ankle center
        if self.ankle_positions:
            self.ankle_center = self.calculate_ankle_center()
            self.ankle_center_calculated = True
            print(f"Ankle center calculated at {self.ankle_center}")
        
        # Reset video capture for second pass
        cap.release()
        cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        
        print("Second pass: Processing video with ankle center...")

        # Create a list to store all data for each frame
        all_frame_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            
            # Undistort the frame using calibration parameters
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Convert to grayscale for ArUco detection (use undistorted frame)
            gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(gray)

            # Recalculate pixel to cm ratio if needed
            if corners and self.pixel_to_cm_ratio is None:
                self.calculate_marker_size(corners)

            # Process markers and collect data
            frame_data = self.process_markers_data(corners, ids)
            
            # Add the frame data to our collection
            if frame_data:
                all_frame_data.append(frame_data)

            # Display progress
            if self.frame_count % 100 == 0:
                print(f"Processed {self.frame_count} frames")

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        # Interpolate missing data
        all_frame_data = self.interpolate_data_combined(all_frame_data)

        # Save combined data
        base_path = os.path.splitext(video_path)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_combined_data(all_frame_data, base_path, timestamp)

        print(f"Processing complete. {self.frame_count} frames analyzed.")

    def process_markers_data(self, corners, ids):
        """Process detected markers to collect skeleton data"""
        # Initialize data dictionary for this frame
        frame_data = {"frame": self.frame_count}
        
        # If no markers detected, return early with just frame number
        if ids is None:
            return frame_data
            
        # Extract marker centers and create a dictionary
        centers = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.marker_map:
                # Calculate center of the marker
                center_x = int(np.mean([corners[i][0][j][0] for j in range(4)]))
                center_y = int(np.mean([corners[i][0][j][1] for j in range(4)]))
                centers[marker_id] = (center_x, center_y)


        # Calculate angles
        for angle_name, id1, id2, id3, direction in self.angles_to_track:
            if id1 in centers and id2 in centers and id3 in centers:
                angle = self.calculate_angle(centers[id1], centers[id2], centers[id3], direction)
                frame_data[angle_name] = angle
        
        # Calculate distances
        for dist_name, id1, id2 in self.distances_to_track:
            if id1 in centers and id2 in centers:
                horizontal_distance_px = abs(centers[id1][0] - centers[id2][0])
                horizontal_distance_cm = horizontal_distance_px
                
                if self.pixel_to_cm_ratio:
                    horizontal_distance_cm = horizontal_distance_px * self.pixel_to_cm_ratio
                
                frame_data[dist_name] = horizontal_distance_cm
        
        # Calculate distance between hip and ankle center
        if 4 in centers and self.ankle_center_calculated:  # Hip marker
            hip_x = centers[4][0]
            horizontal_distance_px = abs(hip_x - self.ankle_center[0])
            horizontal_distance_cm = horizontal_distance_px
            
            if self.pixel_to_cm_ratio:
                horizontal_distance_cm = horizontal_distance_px * self.pixel_to_cm_ratio
            
            frame_data["Hip_Ankle_Center_Distance"] = horizontal_distance_cm
        
        return frame_data

    def interpolate_data_combined(self, frame_data_list):
        """Interpolate missing values in the combined data"""
        if not frame_data_list:
            return []

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(frame_data_list)
        
        # Ensure frame numbers are complete and sequential
        max_frame = max(df['frame'])
        full_frames = pd.DataFrame({'frame': range(1, max_frame + 1)})
        df = pd.merge(full_frames, df, on='frame', how='left')

        # Interpolate missing values for each data column
        for col in df.columns:
            if col != "frame" and df[col].isna().any():
                # Linear interpolation
                df[col] = df[col].interpolate(method='linear')

                # Forward/backward fill for any remaining NaNs at the start/end
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        # Convert back to list of dictionaries
        return df.to_dict('records')

    def save_combined_data(self, frame_data_list, base_path, timestamp):
        """Save all tracking data to a single CSV file"""
        if not frame_data_list:
            print("No data to save.")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(frame_data_list)
        
        # Save to CSV
        csv_path = f"all_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"All data saved to: {csv_path}")


def get_marker_size_from_file(file_path="marker.txt"):
    """Read marker size in centimeters from a file"""
    try:
        with open(file_path, 'r') as f:
            marker_size = float(f.read().strip())
            print(f"Marker size from file: {marker_size} cm")
            return marker_size
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading marker size from file: {e}")
        print("Using default marker size of 6.4 cm")
        return 6.4  # default value


def cleanup_files(calibration_folder, calibration_file, video_path):
    """Clean up calibration folder and file after processing"""
    try:
        if os.path.exists(calibration_folder):
            print(f"Removing calibration folder: {calibration_folder}")
            shutil.rmtree(calibration_folder)
        
        if os.path.exists(calibration_file):
            print(f"Removing calibration file: {calibration_file}")
            os.remove(calibration_file)

        if os.path.exists(video_path):
            print(f"Removing video file: {video_path}")
            os.remove(video_path)
            
        return True
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False


def run_filter_script():
    """Run the filter.py script after processing is complete"""
    try:
        print("Running filter.py script...")
        result = subprocess.run(["python", "filter.py"], check=True)
        print("filter.py executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing filter.py: {e}")
        return False
    except FileNotFoundError:
        print("Error: filter.py not found in the current directory.")
        return False
def run_back_script():
    """Run the filter.py script after processing is complete"""
    try:
        print("Running back_final.py script...")
        result = subprocess.run(["python", "back_final.py"], check=True)
        print("back_final.py executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing back_final.py: {e}")
        return False
    except FileNotFoundError:
        print("Error: back_final.py not found in the current directory.")
        return False
def run_back_filter_script():
    """Run the back_filter.py script after processing is complete"""
    try:
        print("Running back_filter.py script...")
        result = subprocess.run(["python", "back_filter.py"], check=True)
        print("back_filter.py executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing back_filter.py: {e}")
        return False
    except FileNotFoundError:
        print("Error: back_filter.py not found in the current directory.")
        return False


def main():
    # Fixed values instead of command line arguments
    video_path = "side.mp4"
    calibration_folder = "calibration_images"
    calibration_file = "camera_calibration.npz"
    
    # Get marker size from file
    marker_size = get_marker_size_from_file()
    
    print(f"Calibration folder: {calibration_folder}")
    print(f"Calibration file: {calibration_file}")
    print(f"Marker size: {marker_size} cm")
    print(f"Processing video: {video_path}")
    
    tracker = CalibratedSkeletonTracker()
    tracker.marker_size_cm = marker_size
    
    # Always create fresh calibration
    if tracker.create_calibration(calibration_folder, calibration_file):
        # Process the video
        tracker.process_video(video_path)
        print("Processing complete.")
        
        # Run the filter.py script
        run_filter_script()
        run_back_script()
        run_back_filter_script()
        # Clean up calibration files
        # cleanup_files(calibration_folder, calibration_file, video_path)
        
    else:
        print("Camera calibration failed. Please check your calibration images.")


if __name__ == "__main__":
    main()
