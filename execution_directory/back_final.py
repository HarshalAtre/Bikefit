import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime


class BackViewSkeletonTracker:
    def __init__(self):
        # Marker ID to body part mapping for back view
        self.marker_map = {
            0: "BottomRod",
            1: "SeatCenter",
            2: "LowestTirePoint",
            3: "MiddleBack",
            4: "LowerBack",
            5: "LeftHipBone",
            6: "RightHipBone"
        }

        # Define minimal connections between body parts for clarity
        self.connections = [
            (2, 1),   # Tire point to Seat Center (reference line)
            (5, 6),   # Left Hip Bone to Right Hip Bone (pelvis line)
            (3, 4)    # Middle Back to Lower Back (spine line)
        ]

        # Colors for visualization
        self.colors = {
            "skeleton": (150, 150, 150),  # Gray for basic skeleton
            "markers": (0, 0, 255),       # Red for markers
            "marker_text": (0, 255, 0),   # Green for marker names
            "angle_text": (0, 255, 0),    # Green for angles
            "frame_text": (255, 255, 255),  # White for frame counter
            "reference_line": (255, 165, 0),  # Orange for tire-seat reference line
            "pelvis_line": (255, 0, 255),    # Magenta for pelvis line
            "spine_line": (0, 165, 255),      # Orange-red for spine line
            "perpendicular_line": (0, 255, 180),  # Aqua for spine perpendicular bisector
            "angle_arc": (0, 255, 255)       # Cyan for angle arcs
        }

        # Data storage
        self.angle_data = []
        self.frame_count = 0

    def calculate_angle_between_lines(self, line1_p1, line1_p2, line2_p1, line2_p2):
        """Calculate angle between two lines defined by two points each"""
        # Calculate vectors for each line
        vector1 = np.array([line1_p2[0] - line1_p1[0], line1_p2[1] - line1_p1[1]])
        vector2 = np.array([line2_p2[0] - line2_p1[0], line2_p2[1] - line2_p1[1]])
        
        # Normalize vectors
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        
        # Calculate dot product
        dot_product = np.dot(vector1, vector2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Handle numerical errors
        
        # Calculate angle in degrees
        angle_deg = np.degrees(np.arccos(dot_product))
        
        # We want the smallest angle between the lines (0-90 degrees)
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return angle_deg

    def calculate_clockwise_angle(self, ref_line_p1, ref_line_p2, line_p1, line_p2):
        """Calculate clockwise angle from reference line to the second line
        
        This method calculates the angle in a clockwise direction,
        from the top of the reference line to the right side of the intersecting line.
        
        Parameters:
        - ref_line_p1, ref_line_p2: Points defining the reference line (vertical line)
        - line_p1, line_p2: Points defining the second line
        
        Returns the angle in degrees (0-360)
        """
        # Calculate vectors for each line
        # Note: OpenCV's coordinate system has origin at top-left
        # with y-axis pointing downward
        
        # Calculate vectors (directed line segments)
        ref_vector = np.array([ref_line_p2[0] - ref_line_p1[0], ref_line_p2[1] - ref_line_p1[1]])
        target_vector = np.array([line_p2[0] - line_p1[0], line_p2[1] - line_p1[1]])
        
        # Normalize vectors
        ref_vector = ref_vector / np.linalg.norm(ref_vector)
        target_vector = target_vector / np.linalg.norm(target_vector)
        
        # Calculate the angle between vectors using arctan2
        # Get angles from positive y-axis (downward in OpenCV coordinates)
        ref_angle = np.degrees(np.arctan2(ref_vector[0], ref_vector[1]))
        target_angle = np.degrees(np.arctan2(target_vector[0], target_vector[1]))
        
        # Ensure angles are in the range [0, 360]
        if ref_angle < 0:
            ref_angle += 360
        if target_angle < 0:
            target_angle += 360
            
        # Calculate clockwise angle from reference to target
        clockwise_angle = target_angle - ref_angle
        
        # Ensure result is in the range [0, 360]
        if clockwise_angle < 0:
            clockwise_angle += 360
            
        return clockwise_angle

    def calculate_perpendicular_bisector(self, p1, p2, frame_shape):
        """Calculate perpendicular bisector line through two points"""
        # Calculate midpoint
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        midpoint = (int(mid_x), int(mid_y))
        
        # Calculate perpendicular slope (negative reciprocal)
        if p2[0] == p1[0]:  # Horizontal line
            # Perpendicular is vertical line
            perp_p1 = (midpoint[0], 0)
            perp_p2 = (midpoint[0], frame_shape[0])
        elif p2[1] == p1[1]:  # Vertical line
            # Perpendicular is horizontal line
            perp_p1 = (0, midpoint[1])
            perp_p2 = (frame_shape[1], midpoint[1])
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            perp_slope = -1 / slope
            
            # Calculate vector along perpendicular direction
            # We'll use a reasonably sized vector for initial line segment
            if abs(perp_slope) > 1:  # Steeper line, use y step
                dy = 100
                dx = dy / perp_slope
            else:  # Shallower line, use x step
                dx = 100
                dy = dx * perp_slope
                
            # Two points along the perpendicular line
            perp_p1 = (int(midpoint[0] - dx), int(midpoint[1] - dy))
            perp_p2 = (int(midpoint[0] + dx), int(midpoint[1] + dy))
            
            # Extend to frame boundaries
            perp_p1, perp_p2 = self.extend_line(perp_p1, perp_p2, frame_shape)
        
        return midpoint, perp_p1, perp_p2

    def extend_line(self, p1, p2, frame_shape):
        """Extend a line defined by two points to reach frame boundaries"""
        h, w = frame_shape[:2]
        
        # Calculate slope and intercept
        if p2[0] == p1[0]:  # Vertical line
            x1, x2 = p1[0], p1[0]
            y1, y2 = 0, h
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            intercept = p1[1] - slope * p1[0]
            
            # Find intersections with frame boundaries
            # Left boundary (x=0)
            y_left = intercept
            # Right boundary (x=w)
            y_right = slope * w + intercept
            # Top boundary (y=0)
            x_top = -intercept / slope if slope != 0 else w/2
            # Bottom boundary (y=h)
            x_bottom = (h - intercept) / slope if slope != 0 else w/2
            
            # Find which intersections are within frame bounds
            points = []
            if 0 <= y_left <= h:
                points.append((0, int(y_left)))
            if 0 <= y_right <= h:
                points.append((w-1, int(y_right)))
            if 0 <= x_top <= w:
                points.append((int(x_top), 0))
            if 0 <= x_bottom <= w:
                points.append((int(x_bottom), h-1))
            
            # Sort points to get endpoints
            if len(points) >= 2:
                # Sort by x coordinate
                points.sort()
                x1, y1 = points[0]
                x2, y2 = points[-1]
            else:
                # Fallback to original points
                x1, y1 = p1
                x2, y2 = p2
        
        return (int(x1), int(y1)), (int(x2), int(y2))

    def find_line_intersection(self, line1_p1, line1_p2, line2_p1, line2_p2):
        """Find the intersection point of two lines"""
        # Line 1 represented as a1x + b1y = c1
        a1 = line1_p2[1] - line1_p1[1]
        b1 = line1_p1[0] - line1_p2[0]
        c1 = a1 * line1_p1[0] + b1 * line1_p1[1]
        
        # Line 2 represented as a2x + b2y = c2
        a2 = line2_p2[1] - line2_p1[1]
        b2 = line2_p1[0] - line2_p2[0]
        c2 = a2 * line2_p1[0] + b2 * line2_p1[1]
        
        # Calculate determinant
        det = a1 * b2 - a2 * b1
        
        # If determinant is zero, lines are parallel
        if abs(det) < 1e-9:
            return None
        
        # Calculate intersection point
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        
        return (int(x), int(y))

    def draw_angle_arc(self, frame, vertex, p1, p2, angle, color, radius=50):
        """Draw an arc to visualize the angle between two lines"""
        # Calculate vectors from vertex to points
        v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]], dtype=np.float64)
        v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]], dtype=np.float64)
        
        # Calculate angles for each vector
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        
        # Convert to degrees
        start_angle = int(np.degrees(angle1))
        end_angle = int(np.degrees(angle2))
        
        # Ensure proper drawing direction
        if end_angle < start_angle:
            end_angle += 360
        
        # Draw the arc
        cv2.ellipse(frame, vertex, (radius, radius), 0, start_angle, end_angle, color, 2)
        
        # Add angle text at the arc
        mid_angle_rad = (np.radians(start_angle) + np.radians(end_angle)) / 2
        text_x = int(vertex[0] + (radius + 20) * np.cos(mid_angle_rad))
        text_y = int(vertex[1] + (radius + 20) * np.sin(mid_angle_rad))
        cv2.putText(frame, f"{angle:.1f}°", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def process_video(self, video_path):
        """Process video file with ArUco markers to measure pelvic and spinal angles"""
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        self.frame_count = 0
        self.angle_data = []
        
        print(f"Processing video: {video_path}")
        print("Processing frames...")

        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            
            # Print progress every 100 frames
            if self.frame_count % 100 == 0:
                print(f"Processed {self.frame_count} frames")

            # Make a copy for analysis
            display_frame = frame.copy()

            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None:
                # Process detected markers
                self.process_markers(display_frame, corners, ids)

        # Release resources
        cap.release()

        # Interpolate missing angle data
        self.interpolate_angle_data()

        # Save data
        self.save_angle_data(os.path.splitext(video_path)[0], datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"Processed a total of {self.frame_count} frames")
        print("Processing complete.")
        
        # Delete the input video file
        # try:
        #     # os.remove(video_path)
        #     # print(f"Deleted input video file: {video_path}")
        # except OSError as e:
        #     print(f"Error deleting input video file: {e}")

    def interpolate_angle_data(self):
        """Interpolate missing angle values in the data"""
        if not self.angle_data:
            return

        df = pd.DataFrame(self.angle_data)

        # Interpolate missing values for pelvic rocking and spinal asymmetry
        angle_columns = ["PelvicRocking", "SpinalAsymmetry"]

        for angle_name in angle_columns:
            if angle_name in df.columns:
                # Check if there are any NaN values to interpolate
                if df[angle_name].isna().any():
                    # Linear interpolation
                    df[angle_name] = df[angle_name].interpolate(method='linear')

                    # Forward/backward fill for any remaining NaNs at the start/end
                    df[angle_name] = df[angle_name].fillna(method='ffill').fillna(method='bfill')

        # Update the angle data
        self.angle_data = df.to_dict('records')

    def process_markers(self, frame, corners, ids):
        """Process detected markers to measure and visualize pelvic and spinal angles"""
        # Extract marker centers and create a dictionary
        centers = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.marker_map:
                # Calculate center of the marker
                center_x = int(np.mean([corners[i][0][j][0] for j in range(4)]))
                center_y = int(np.mean([corners[i][0][j][1] for j in range(4)]))
                centers[marker_id] = (center_x, center_y)

                # Draw marker center
                cv2.circle(frame, (center_x, center_y), 5, self.colors["markers"], -1)

                # Label the marker
                label = f"{self.marker_map[marker_id]} ({marker_id})"
                cv2.putText(frame, label, (center_x + 10, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["marker_text"], 1)

        # Initialize angles for this frame
        frame_angles = {"frame": self.frame_count}
        pelvic_rocking_angle = None
        spinal_asymmetry_angle = None
        
        # Get frame shape for line extension
        frame_shape = frame.shape
        
        # Store line data for intersection calculation
        lines = {}
        
        # Draw connections between markers
        for connection in self.connections:
            if connection[0] in centers and connection[1] in centers:
                p1 = centers[connection[0]]
                p2 = centers[connection[1]]
                
                # Determine line type and style
                if connection == (2, 1):  # Tire-Seat reference line
                    line_color = self.colors["reference_line"]
                    line_thickness = 3
                    line_type = "reference"
                    
                    # Extend reference line
                    ext_p1, ext_p2 = self.extend_line(p1, p2, frame_shape)
                    cv2.line(frame, ext_p1, ext_p2, line_color, line_thickness)
                    
                    # Store extended line
                    lines[line_type] = (ext_p1, ext_p2, p1, p2)
                    
                    # Add label
                    label_pos = (int((p1[0] + p2[0])/2) + 10, int((p1[1] + p2[1])/2))
                    cv2.putText(frame, "Reference Line", label_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
                    
                elif connection == (5, 6):  # Hip to Hip pelvis line
                    line_color = self.colors["pelvis_line"]
                    line_thickness = 3
                    line_type = "pelvis"
                    
                    # Extend pelvis line
                    ext_p1, ext_p2 = self.extend_line(p1, p2, frame_shape)
                    cv2.line(frame, ext_p1, ext_p2, line_color, line_thickness)
                    
                    # Store extended line
                    lines[line_type] = (ext_p1, ext_p2, p1, p2)
                    
                    # Add label
                    label_pos = (int((p1[0] + p2[0])/2) + 10, int((p1[1] + p2[1])/2))
                    cv2.putText(frame, "Pelvis Line", label_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
                    
                elif connection == (3, 4):  # Middle-Lower back spine line
                    line_color = self.colors["spine_line"]
                    line_thickness = 3
                    
                    # Draw the spine line (not extended)
                    cv2.line(frame, p1, p2, line_color, line_thickness)
                    
                    # Calculate perpendicular bisector for spine line
                    midpoint, perp_p1, perp_p2 = self.calculate_perpendicular_bisector(p1, p2, frame_shape)
                    
                    # Draw perpendicular bisector
                    cv2.line(frame, perp_p1, perp_p2, self.colors["perpendicular_line"], 2)
                    
                    # Store perpendicular bisector
                    lines["spine_perp"] = (perp_p1, perp_p2, midpoint, p2)
                    
                    # Add labels
                    spine_label_pos = (int((p1[0] + p2[0])/2) + 10, int((p1[1] + p2[1])/2))
                    cv2.putText(frame, "Spine Line", spine_label_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
                    
                    perp_label_pos = (int((midpoint[0] + perp_p2[0])/2), int((midpoint[1] + perp_p2[1])/2) + 20)
                    cv2.putText(frame, "Spine Perpendicular", perp_label_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["perpendicular_line"], 2)
                    
                else:
                    # Other connections
                    cv2.line(frame, p1, p2, self.colors["skeleton"], 2)
        
        # Calculate angles and draw angle markers at intersections
        if "reference" in lines:
            ref_ext_p1, ref_ext_p2, ref_p1, ref_p2 = lines["reference"]
            
            # Calculate and draw pelvic rocking angle at intersection
            if "pelvis" in lines:
                pelvis_ext_p1, pelvis_ext_p2, pelvis_p1, pelvis_p2 = lines["pelvis"]
                
                # Calculate pelvic rocking angle clockwise from reference line
                pelvic_rocking_angle = self.calculate_clockwise_angle(
                    ref_p1, ref_p2, pelvis_p1, pelvis_p2)
                
                # We can convert to a more intuitive range if needed
                # For example, if we want angles in the range [-180, 180]
                # where negative means counterclockwise from vertical
                if pelvic_rocking_angle > 180:
                    pelvic_rocking_angle = pelvic_rocking_angle - 360
                    
                frame_angles["PelvicRocking"] = pelvic_rocking_angle
                
                # Find intersection point
                intersection = self.find_line_intersection(
                    ref_ext_p1, ref_ext_p2, pelvis_ext_p1, pelvis_ext_p2)
                
                if intersection is not None:
                    # Draw angle arc at intersection
                    self.draw_angle_arc(
                        frame, intersection, ref_ext_p2, pelvis_ext_p2,
                        abs(pelvic_rocking_angle), self.colors["angle_arc"], radius=40)
                    
                    # Display the pelvic rocking angle on the frame (in corner)
                    cv2.putText(frame, f"Pelvic Rocking: {pelvic_rocking_angle:.1f}°", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["angle_text"], 2)
            
            # Calculate and draw spinal asymmetry angle using perpendicular bisector
            if "spine_perp" in lines:
                perp_p1, perp_p2, midpoint, spine_p2 = lines["spine_perp"]
                
                # Calculate spinal asymmetry angle clockwise from reference line
                spinal_asymmetry_angle = self.calculate_clockwise_angle(
                    ref_p1, ref_p2, perp_p1, perp_p2)
                
                # Convert to more intuitive range if needed
                if spinal_asymmetry_angle > 180:
                    spinal_asymmetry_angle = spinal_asymmetry_angle - 360
                    
                frame_angles["SpinalAsymmetry"] = spinal_asymmetry_angle
                
                # Find intersection point
                intersection = self.find_line_intersection(
                    ref_ext_p1, ref_ext_p2, perp_p1, perp_p2)
                
                if intersection is not None:
                    # Draw angle arc at intersection
                    self.draw_angle_arc(
                        frame, intersection, ref_ext_p2, perp_p2,
                        abs(spinal_asymmetry_angle), self.colors["angle_arc"], radius=40)
                    
                    # Display the spinal asymmetry angle on the frame (in corner)
                    y_pos = 100 if pelvic_rocking_angle is not None else 70
                    cv2.putText(frame, f"Spinal Asymmetry: {spinal_asymmetry_angle:.1f}°", 
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["angle_text"], 2)

        # Add angles to dataframe
        self.angle_data.append(frame_angles)

    def save_angle_data(self, base_path, timestamp):
        """Save angle data to CSV file"""
        if not self.angle_data:
            print("No angle data to save")
            return

        df = pd.DataFrame(self.angle_data)
        csv_path = f"back_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Angle data saved to: {csv_path}")


def main():
    # Get video path 
    video_path = r"back.mp4"
    print(f"Starting to process video: {video_path}")
    tracker = BackViewSkeletonTracker()
    tracker.process_video(video_path)


if __name__ == "__main__":
    main()
