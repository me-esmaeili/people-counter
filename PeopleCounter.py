import cv2
import os
import logging
import threading
import queue
import time
import numpy as np
from my_object_counter import ObjectCounter
from utility import get_horizontal_line_coordinates
from VideoSource import VideoSource
from ConfigManager import ConfigManager
from CountLogger import CountLogger
from MotionDetector import MotionDetector


class PeopleCounter:
    def __init__(self, config_file):
        self.config_manager = ConfigManager(config_file)
        self.model_config = self.config_manager.get_model_config()
        self.video_config = self.config_manager.get_video_config()
        self.log_config = self.config_manager.get_log_config()

        # Get motion detection config or use defaults
        self.motion_config = self.config_manager.get_motion_config()

    def _init(self):
        # Model settings
        self.model_filename = self.model_config["model_filename"]
        self.line_width = self.model_config["line_width"]
        self.device = self.model_config["device"]

        # Video processing settings
        self.skip_frames = self.video_config["skip_frames"]
        self.video_writer_codec = self.video_config["video_writer_codec"]
        self.show = self.video_config["show"]
        self.draw = self.video_config["draw"]
        self.save_video = self.video_config["save_video"]
        self.save_video_path = self.video_config["save_video_path"]
        self.input_width = self.video_config["input_width"]
        self.input_height = self.video_config["input_height"]
        self.swap_direction = self.video_config["swap_direction"]

        self.video_duration = self.video_config["video_duration"]
        self.max_video_files = self.video_config["max_video_files"]

        # Initialize state variables
        self.entry_count = 0
        self.exit_count = 0
        self.running = False
        self.video_writer = None
        self.counter = None
        self.last_annotated_frame = None
        self.source = None
        self.frame_queue = queue.Queue(maxsize=5)  # Buffer a few frames
        self.result_queue = queue.Queue(maxsize=5)  # Buffer for processed frames

        # Video writer state variables
        self.current_video_frames = 0
        self.frames_per_video = 0  # Will be set based on fps and duration
        self.video_file_counter = 0
        self.video_files = []

        # Initialize motion detection
        self.motion_detector = None
        if self.motion_config["motion_enabled"]:
            self._init_motion_detector()

        # Initialize logger
        self.logger = CountLogger(log_dir=self.log_config["log_dir"],
                                     console_log=self.log_config["console_log"],
                                     max_logs=self.log_config["max_logs"])

        # Flag to track if we should run the counter based on motion
        self.should_count = False

    def _init_motion_detector(self):
        """Initialize the motion detector"""
        self.motion_detector = MotionDetector(
            min_area_percent=self.motion_config["min_area_percent"],

            debug=False
        )

        logging.info("Motion detector initialized")

    def start(self, video_source):
        self._init()
        # Initialize video source
        self.source = VideoSource(video_source)
        if not self.source.initialize():
            raise Exception("Error reading video file")

        # Get video dimensions and fps
        self.fps = 25.0 if self.source.using_picam else self.source.cap.get(cv2.CAP_PROP_FPS)
        self.frames_per_video = int(self.fps * self.video_duration)  # Calculate frames per video segment

        # Calculate line coordinates for counting
        line_points = get_horizontal_line_coordinates(self.input_width, self.input_height)

        # Create output directory if it doesn't exist
        if self.save_video and not os.path.exists(self.save_video_path):
            os.makedirs(self.save_video_path)

        # Initialize first video writer if needed
        if self.save_video:
            self._create_new_video_writer()

        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        # Initialize object counter
        self.counter = ObjectCounter(
            show=self.show,
            draw=self.draw,
            region=line_points,
            model=self.model_filename,
            line_width=self.line_width,
            verbose=False,
            swap_direction=self.swap_direction,
            imgsz=(640, 480),
            device=self.device
        )

        # Start threads
        self.running = True
        capture_thread = threading.Thread(target=self._capture_frames)
        process_thread = threading.Thread(target=self._process_frames)
        output_thread = threading.Thread(target=self._output_frames)

        capture_thread.daemon = True
        process_thread.daemon = True
        output_thread.daemon = True

        capture_thread.start()
        process_thread.start()
        output_thread.start()

        # Wait for threads to complete
        capture_thread.join()
        process_thread.join()
        output_thread.join()

        self._cleanup_resources()

    def _create_new_video_writer(self):
        """Create a new video writer with timestamp-based filename"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(self.save_video_path, f"video_{timestamp}.mp4")

        # Create new video writer
        self.video_writer = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc(*self.video_writer_codec),
            self.fps,
            (self.input_width, self.input_height)
        )

        # Add to list of video files
        self.video_files.append(output_filename)
        self.video_file_counter += 1
        self.current_video_frames = 0

        # Manage circular buffer of files
        self._manage_video_files()

        return output_filename

    def _manage_video_files(self):
        """Maintain circular buffer of video files, removing oldest when exceeding max"""
        while len(self.video_files) > self.max_video_files:
            oldest_file = self.video_files.pop(0)
            try:
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
                    logging.info(f"Removed oldest video file: {oldest_file}")
            except Exception as e:
                logging.error(f"Error removing file {oldest_file}: {str(e)}")

    def _capture_frames(self):
        """Thread function to capture frames"""
        frame_count = 0
        while self.running:
            success, frame = self.source.read()
            if not success:
                self.running = False
                break

            frame = cv2.resize(frame, (self.input_width, self.input_height))
            timestamp = None
            if hasattr(self.source, "get_timestamp"):
                timestamp = self.source.get_timestamp()
            else:
                timestamp = time.time()

            frame_count += 1

            # Determine if this frame should be processed based on skip_frames
            should_process = self.source.isCamera() or (frame_count % (self.skip_frames + 1) == 0)

            # Only add frames that need processing to the queue
            if should_process:
                try:
                    self.frame_queue.put((frame, timestamp, frame_count), block=True, timeout=1)
                except queue.Full:
                    # If queue is full, skip this frame
                    pass

    def _process_frames(self):
        """Thread function to process frames"""
        while self.running:
            try:
                frame, timestamp, frame_count = self.frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # First, check for motion if motion detector is enabled
            if self.motion_detector is not None:
                motion_detected, motion_frame, motion_data = self.motion_detector.detect(frame, timestamp)

                # Only run counter if motion is detected
                if motion_detected:
                    # Set flag to indicate motion was detected
                    self.should_count = True

                    # Process frame with counter
                    processed_frame = self.counter.count(frame)

                    # Log the motion event
                    logging.info(
                        f"Motion detected ({motion_data['percent_area']:.2f}% of frame) - Processing with counter")

                    # # Add motion detection info to the frame
                    # cv2.putText(processed_frame,
                    #             f"Motion: {motion_data['percent_area']:.2f}%",
                    #             (10, processed_frame.shape[0] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #             (0, 255, 0), 2)
                    #
                    # # Draw motion contours
                    # cv2.drawContours(processed_frame, motion_data["contours"], -1, (0, 255, 0), 2)
                else:
                    # If no motion detected, just create a basic frame without running the counter
                    self.should_count = False
                    processed_frame = frame.copy()

                    # # Add no-motion info to the frame
                    # cv2.putText(processed_frame,
                    #             f"No Motion: {motion_data['percent_area']:.2f}%",
                    #             (10, processed_frame.shape[0] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #             (0, 0, 255), 2)
            else:
                # If no motion detector, always run counter
                self.should_count = True
                processed_frame = self.counter.count(frame)

            # cv2.imshow("view",processed_frame)

            self.last_annotated_frame = processed_frame.copy()

            # Only update counts if counter was run
            if self.should_count:
                # Get current counts
                current_entry_count = self.get_entry_count()
                current_exit_count = self.get_exit_count()

                # Log new entries
                if current_entry_count > self.entry_count:
                    for _ in range(current_entry_count - self.entry_count):
                        self.logger.log_entry(current_entry_count, current_exit_count, timestamp)

                # Log new exits
                if current_exit_count > self.exit_count:
                    for _ in range(current_exit_count - self.exit_count):
                        self.logger.log_exit(current_entry_count, current_exit_count, timestamp)

                # Update our counts
                self.entry_count = current_entry_count
                self.exit_count = current_exit_count

            try:
                self.result_queue.put((processed_frame, timestamp), block=True, timeout=1)
            except queue.Full:
                # If result queue is full, skip this frame output
                pass

            self.frame_queue.task_done()

    def _output_frames(self):
        """Thread function to handle frame output"""
        while self.running:
            try:
                frame, timestamp = self.result_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # Write frame to output video if enabled
            if self.save_video and self.video_writer is not None:
                # Write the current frame
                self.video_writer.write(frame)
                self.current_video_frames += 1

                # Check if we need to start a new video file
                if self.current_video_frames >= self.frames_per_video:
                    # Close current video writer
                    if self.video_writer is not None:
                        self.video_writer.release()

                    # Create new video writer
                    self._create_new_video_writer()
                    logging.info(f"Created new video file: {self.video_files[-1]}")

            self.result_queue.task_done()

    def _cleanup_resources(self):
        """Clean up resources"""
        if self.source:
            self.source.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def stop(self):
        """Stop processing and release resources"""
        self.running = False

    def get_entry_count(self):
        """Get total entry count across all classes"""
        total_in = 0
        if self.counter and hasattr(self.counter, "classwise_counts"):
            for key, value in self.counter.classwise_counts.items():
                if value["IN"] != 0 or value["OUT"] != 0:
                    total_in += value["IN"]
        return total_in

    def get_exit_count(self):
        """Get total exit count across all classes"""
        total_out = 0
        if self.counter and hasattr(self.counter, "classwise_counts"):
            for key, value in self.counter.classwise_counts.items():
                if value["IN"] != 0 or value["OUT"] != 0:
                    total_out += value["OUT"]
        return total_out