import cv2
from ultralytics import YOLO
import time
import os
from VideoSource import VideoSource
from Logger import Logger
from ConfigLoader import ConfigLoader
from utility import is_windows, is_raspberry_pi
from PeopleTracker import *
class PeopleCounter:
    """Main class for people counting system"""

    def __init__(self, config_path):
        """Initialize the people counter"""
        self.config = ConfigLoader.load_config(config_path)

        self._setup_model()
        self.logger = Logger(self.config["log_dir"])

        self._setup_debug()

    def _setup_video_source(self,video_source=0):
        """Initialize video source and dimensions"""
        self.video_source = VideoSource(video_source)
        if not self.video_source.initialize():
            raise ValueError("Failed to initialize video source")
        self.frame_width, self.frame_height = self.video_source.get_dimensions()

    def _setup_model(self):
        """Initialize YOLO model based on platform"""
        model_path = (self.config["desktop_model_path"] if is_windows()
                      else self.config["raspi_model_path"] if is_raspberry_pi()
        else None)
        if model_path:
            self.model = YOLO(model_path)
        else:
            raise ValueError("Unsupported platform for model selection")

    def _setup_debug(self):
        """Setup debug settings"""
        self.debug_enabled = self.config["debug_enabled"]
        self.save_debug_frames = self.config.get("save_debug_frames", False)
        if self.save_debug_frames:
            self.debug_dir = self.config["debug_dir"]
            os.makedirs(self.debug_dir, exist_ok=True)

    def _process_detections(self, frame, debug_frame):
        """Process YOLO detections and update tracking"""
        results = self.model.track(frame, show=False, iou=0.5, persist=True,
                                   classes=[0], verbose=False)

        current_track_ids = set()
        if (results[0].boxes is not None and
                hasattr(results[0].boxes, 'id') and
                results[0].boxes.id is not None):

            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                current_track_ids.add(track_id)
                x1, y1, x2, y2 = box
                center_y = int((y1 + y2) / 2)

                self.tracker.update_track(track_id, center_y)
                self.tracker.count_person(track_id, self.logger)

                if self.debug_enabled:
                    self._draw_detection(debug_frame, box, track_id)

        # Update tracks not in current frame
        self.tracker.tracks = {k: v for k, v in self.tracker.tracks.items()
                               if k in current_track_ids or v['last_seen'] == 0}
        self.tracker.update_disappeared()

    def _draw_detection(self, frame, box, track_id):
        """Draw detection bounding box and information"""
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        color = (0, 0, 255)  # Red default
        if track_id in self.tracker.counted_tracks:
            positions = self.tracker.tracks.get(track_id, self.tracker.disappeared_tracks.get(track_id))['positions']
            color = (0, 255, 0) if positions[0] > positions[-1] else (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def start(self, video_source):
        self._setup_video_source(video_source)
        self.tracker = PeopleTracker(self.frame_height, self.config)
        """Run the people counting system"""
        frame_count, start_time, debug_frame_count = 0, time.time(), 0

        while True:
            ret, frame = self.video_source.read()
            if not ret:
                break

            # FPS calculation
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 1 else 0
            if elapsed > 1:
                frame_count, start_time = 0, time.time()

            # Process frame
            debug_frame = frame.copy() if self.debug_enabled else None
            if self.debug_enabled:
                cv2.line(debug_frame, (0, self.tracker.threshold_y),
                         (self.frame_width, self.tracker.threshold_y), (0, 255, 0), 2)

            self._process_detections(frame, debug_frame)

            # Handle debug output
            if self.debug_enabled:
                self._display_debug_info(debug_frame, fps)
                if self.save_debug_frames and debug_frame_count % 30 == 0:
                    cv2.imwrite(f"{self.debug_dir}/frame_{debug_frame_count:06d}.jpg", debug_frame)
                debug_frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self._cleanup()

    def stop(self):
        """Stop the people counting system and clean up resources"""
        self.running = False
        self._cleanup()
        return {
            'entries': self.tracker.entry_count,
            'exits': self.tracker.exit_count,
            'log_file': self.logger.log_filename
        }

    def get_entry_count(self):
        """Return the current number of people who entered"""
        return self.tracker.entry_count

    def get_exit_count(self):
        """Return the current number of people who exited"""
        return self.tracker.exit_count

    def _display_debug_info(self, frame, fps):
        """Display debug information on frame"""
        cv2.putText(frame, f"Entries: {self.tracker.entry_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {self.tracker.exit_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("People Counter", frame)

    def _cleanup(self):
        """Release resources"""
        self.video_source.release()
        if self.debug_enabled:
            cv2.destroyAllWindows()
        print(f"Counting results saved to {self.logger.log_filename}")