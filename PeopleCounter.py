import cv2
from ultralytics import YOLO
import time
from VideoSource import *
from Logger import *
from Visualizer import  *
from ConfigLoader import *


class PeopleCounter:
    """Main class for people counting"""

    def __init__(self, config_path):
        """Initialize the people counter"""
        # Load configuration
        self.config = ConfigLoader.load_config(config_path)

        # Initialize video source
        self.video_source = VideoSource(self.config["video_source"])
        if not self.video_source.initialize():
            raise ValueError("Failed to initialize video source")

        # Get frame dimensions
        self.frame_width, self.frame_height = self.video_source.get_dimensions()

        # Initialize YOLO model
        self.model = YOLO(self.config["model_path"])

        # Set threshold line position
        self.threshold_y = int(self.frame_height * self.config["threshold_line_position"])

        # Initialize logger
        self.logger = Logger(self.config["log_dir"])

        # Debug settings
        self.debug_enabled = self.config["debug_enabled"]
        self.save_debug_frames = self.config.get("save_debug_frames", False)
        if self.save_debug_frames:
            self.debug_dir = self.config["debug_dir"]
            os.makedirs(self.debug_dir, exist_ok=True)

        # Counters
        self.entry_count = 0
        self.exit_count = 0

        # Track objects
        self.tracks = {}
        self.disappeared_tracks = {}
        self.counted_tracks = set()

        # Configure tracking parameters
        self.max_disappeared = self.config["max_disappeared"]
        self.direction_buffer_size = self.config["direction_buffer_size"]

        # FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def _count_person(self, track_id, positions):
        """Count a person crossing the threshold line"""
        if track_id in self.counted_tracks:
            return False

        if len(positions) < self.direction_buffer_size:
            return False

        # Check if the track crossed the line
        above_line = [pos < self.threshold_y for pos in positions]
        if not (True in above_line and False in above_line):
            return False

        # Calculate direction
        recent_positions = positions[-self.direction_buffer_size:]
        first_half = recent_positions[:len(recent_positions) // 2]
        second_half = recent_positions[len(recent_positions) // 2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        # Determine direction and count
        if avg_first > self.threshold_y > avg_second:  # Moving from bottom to top
            self.entry_count += 1
            print(f"Entry detected (ID: {track_id}), Total Entries: {self.entry_count}")
            self.logger.log_event("Entry", track_id, self.entry_count, self.exit_count)
            self.counted_tracks.add(track_id)
            return True
        elif avg_first < self.threshold_y < avg_second:  # Moving from top to bottom
            self.exit_count += 1
            print(f"Exit detected (ID: {track_id}), Total Exits: {self.exit_count}")
            self.logger.log_event("Exit", track_id, self.entry_count, self.exit_count)
            self.counted_tracks.add(track_id)
            return True

        return False

    def run(self):
        """Run the people counting system"""
        frame_count = 0
        start_time = time.time()
        fps = 0
        debug_frame_count = 0

        while True:
            ret, frame = self.video_source.read()
            if not ret:
                break

            # Calculate FPS
            frame_count += 1
            if (time.time() - start_time) > 1:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            # Create a copy of the frame for debug purposes
            if self.debug_enabled:
                debug_frame = frame.copy()

                # Draw the threshold line
                cv2.line(debug_frame, (0, self.threshold_y), (self.frame_width, self.threshold_y), (0, 255, 0), 2)

            # Run YOLOv8 detection with tracking
            results = self.model.track(frame, persist=True, classes=[0], verbose=False)  # 0 is class ID for person

            # Mark all current tracks as not updated
            current_track_ids = set()

            # Process detection results
            if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                # Update active tracks
                for box, track_id in zip(boxes, track_ids):
                    current_track_ids.add(track_id)
                    x1, y1, x2, y2 = box

                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Create or update track
                    if track_id not in self.tracks:
                        self.tracks[track_id] = {
                            'positions': [center_y],
                            'last_seen': 0
                        }
                    else:
                        self.tracks[track_id]['positions'].append(center_y)
                        self.tracks[track_id]['last_seen'] = 0

                        # Keep only the most recent positions to limit memory usage
                        if len(self.tracks[track_id]['positions']) > 30:
                            self.tracks[track_id]['positions'] = self.tracks[track_id]['positions'][-30:]

                    # Try to count this person if they've not been counted yet
                    self._count_person(track_id, self.tracks[track_id]['positions'])

                    # Draw bounding box and ID for debugging
                    if self.debug_enabled:
                        color = (0, 0, 255)  # Default color (red)
                        if track_id in self.counted_tracks:
                            # Change color based on direction
                            directions = self.tracks[track_id]['positions']
                            if len(directions) >= 2:
                                if directions[0] > directions[-1]:  # Moving up (entry)
                                    color = (0, 255, 0)  # Green for entries
                                else:
                                    color = (255, 0, 0)  # Blue for exits

                        cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.circle(debug_frame, (center_x, center_y), 5, color, -1)
                        cv2.putText(debug_frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update disappeared tracks and remove old ones
            for track_id in list(self.tracks.keys()):
                if track_id not in current_track_ids:
                    self.tracks[track_id]['last_seen'] += 1

                    # Move to disappeared tracks if not seen for too long
                    if self.tracks[track_id]['last_seen'] > self.max_disappeared:
                        # Try one last time to count this person before removing
                        self._count_person(track_id, self.tracks[track_id]['positions'])
                        # Store in disappeared tracks for a while before completely forgetting
                        self.disappeared_tracks[track_id] = self.tracks[track_id]
                        del self.tracks[track_id]

            # Clean up old disappeared tracks
            for track_id in list(self.disappeared_tracks.keys()):
                self.disappeared_tracks[track_id]['last_seen'] += 1
                if self.disappeared_tracks[track_id]['last_seen'] > self.max_disappeared * 2:
                    del self.disappeared_tracks[track_id]

            # Display debug information
            if self.debug_enabled:
                # Display counts and FPS
                cv2.putText(debug_frame, f"Entries: {self.entry_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_frame, f"Exits: {self.exit_count}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show the frame
                cv2.imshow("People Counter", debug_frame)

                # Save debug frames if enabled
                if self.save_debug_frames and debug_frame_count % 30 == 0:  # Save every 30 frames
                    cv2.imwrite(f"{self.debug_dir}/frame_{debug_frame_count:06d}.jpg", debug_frame)
                debug_frame_count += 1

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release resources
        self.video_source.release()
        if self.debug_enabled:
            cv2.destroyAllWindows()
        print(f"Counting results saved to {self.logger.log_filename}")