import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from collections import defaultdict
import csv
from datetime import datetime


class PeopleCounter:
    def __init__(self, model_path, video_source=0, threshold_line_position=0.5):
        """
        Initialize the people counter with YOLO model

        Args:
            model_path: Path to the YOLO model (.pt file)
            video_source: Camera index or video file path
            threshold_line_position: Relative position of counting line (0-1, default 0.5)
        """
        # Load YOLO model
        self.model = YOLO(model_path)

        # Video capture
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {video_source}")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set counting line position
        self.threshold_y = int(self.frame_height * threshold_line_position)

        # Counters
        self.entry_count = 0
        self.exit_count = 0

        # Track objects
        self.tracks = {}
        self.disappeared_tracks = {}
        self.counted_tracks = set()  # Keep track of IDs that have already been counted

        # Configure tracking parameters
        self.max_disappeared = 30  # Maximum frames to keep a disappeared track
        self.direction_buffer_size = 5  # Number of positions to keep for direction calculation

        # Logging
        self.log_filename = f"people_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Event", "Track ID", "Total Entries", "Total Exits"])

    def _log_event(self, event, track_id):
        """Log counting events to CSV file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event, track_id, self.entry_count, self.exit_count])

    def _count_person(self, track_id, positions):
        """
        Determine direction and count person

        Args:
            track_id: Unique identifier for the track
            positions: List of y-positions (oldest to newest)
        """
        if track_id in self.counted_tracks:
            return

        if len(positions) < self.direction_buffer_size:
            return

        # Check if the track has crossed the threshold line
        above_line = [pos < self.threshold_y for pos in positions]

        # Person must have positions on both sides of the line to be counted
        if not (True in above_line and False in above_line):
            return

        # Get the direction by comparing the first and last positions
        # We use the direction_buffer_size most recent positions for more stable direction detection
        recent_positions = positions[-self.direction_buffer_size:]

        # Calculate the average y position for the first half and second half of the buffer
        first_half = recent_positions[:len(recent_positions) // 2]
        second_half = recent_positions[len(recent_positions) // 2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        # Determine direction based on average movement
        if avg_first > self.threshold_y > avg_second:  # Moving from bottom to top
            self.entry_count += 1
            print(f"Entry detected (ID: {track_id}), Total Entries: {self.entry_count}")
            self._log_event("Entry", track_id)
            self.counted_tracks.add(track_id)

        elif avg_first < self.threshold_y < avg_second:  # Moving from top to bottom
            self.exit_count += 1
            print(f"Exit detected (ID: {track_id}), Total Exits: {self.exit_count}")
            self._log_event("Exit", track_id)
            self.counted_tracks.add(track_id)

    def run(self):
        """Run the people counting system"""
        frame_count = 0
        start_time = time.time()
        fps = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Calculate FPS
            frame_count += 1
            if (time.time() - start_time) > 1:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            # Draw counting line
            cv2.line(frame, (0, self.threshold_y), (self.frame_width, self.threshold_y), (0, 255, 0), 2)

            # Run YOLOv8 detection with tracking
            results = self.model.track(frame, persist=True, classes=[0])  # 0 is class ID for person

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

                    # Draw bounding box and ID
                    color = (0, 0, 255)  # Default color (red)
                    if track_id in self.counted_tracks:
                        # Change color based on direction
                        if track_id in self.counted_tracks:
                            directions = self.tracks[track_id]['positions']
                            if len(directions) >= 2:
                                if directions[0] > directions[-1]:  # Moving up (entry)
                                    color = (0, 255, 0)  # Green for entries
                                else:
                                    color = (255, 0, 0)  # Blue for exits

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, color, -1)
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
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

            # Display counts and FPS
            cv2.putText(frame, f"Entries: {self.entry_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Exits: {self.exit_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow("People Counter", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Counting results saved to {self.log_filename}")


if __name__ == "__main__":
    # Create and run the people counter
    counter = PeopleCounter(
        model_path="bestpele.pt",  # Path to your YOLO model
        video_source="20231207153936_839_2.avi",  # 0 for webcam, or path to video file
        threshold_line_position=0.5  # Adjust the position of counting line (0-1)
    )
    counter.run()