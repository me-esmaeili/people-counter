import cv2
import numpy as np
import time
import logging
from datetime import datetime


class MotionDetector:
    def __init__(self,
                 min_area_percent=0.5,  # Minimum percentage of image area to trigger motion
                 max_area_percent=50,  # Maximum percentage of image area to trigger motion
                 history=20,  # Number of frames to build background model
                 var_threshold=16,  # Variance threshold for background subtraction
                 detect_shadows=True,  # Whether to detect shadows
                 blur_size=5,  # Size of Gaussian blur kernel
                 dilation_iterations=2,  # Number of dilation iterations
                 min_seconds_between_alerts=60,  # Minimum seconds between alerts
                 debug=False  # Whether to enable debug logs
                 ):
        self.min_area_percent = min_area_percent
        self.max_area_percent = max_area_percent
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.blur_size = blur_size
        self.dilation_iterations = dilation_iterations
        self.min_seconds_between_alerts = min_seconds_between_alerts
        self.debug = debug

        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )

        # State variables
        self.last_motion_timestamp = 0
        self.last_no_motion_timestamp = 0
        self.motion_detected = False
        self.frame_width = 0
        self.frame_height = 0
        self.frame_area = 0
        self.min_area = 0
        self.max_area = 0
        self.no_motion_callbacks = []
        self.motion_callbacks = []
        self.warmup_frames = 0
        self.max_warmup_frames = 30  # Number of frames to warm up background model

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MotionDetector")

    def initialize(self, frame):
        """Initialize detector with the first frame dimensions"""
        self.frame_height, self.frame_width = frame.shape[:2]
        self.frame_area = self.frame_height * self.frame_width
        self.min_area = int(self.frame_area * (self.min_area_percent / 100))
        self.max_area = int(self.frame_area * (self.max_area_percent / 100))

        self.logger.info(f"Initialized motion detector with frame dimensions: {self.frame_width}x{self.frame_height}")
        self.logger.info(f"Motion detection area thresholds: min={self.min_area}, max={self.max_area}")

        return True

    def register_no_motion_callback(self, callback):
        """Register a callback function to be called when no motion is detected"""
        self.no_motion_callbacks.append(callback)

    def register_motion_callback(self, callback):
        """Register a callback function to be called when motion is detected"""
        self.motion_callbacks.append(callback)

    def detect(self, frame, timestamp=None):
        """
        Detect motion in the given frame

        Args:
            frame: The frame to analyze
            timestamp: Optional timestamp for the frame (defaults to current time)

        Returns:
            Tuple of (motion_detected, processed_frame, motion_data)
            where motion_data is a dictionary containing:
                - contours: List of motion contours
                - total_area: Total area of motion
                - percent_area: Percentage of frame covered by motion
        """
        if timestamp is None:
            timestamp = time.time()

        # Initialize if not already
        if self.frame_width == 0 or self.frame_height == 0:
            self.initialize(frame)

        # Warm up the background model
        if self.warmup_frames < self.max_warmup_frames:
            self.bg_subtractor.apply(frame)
            self.warmup_frames += 1
            return False, frame, {"contours": [], "total_area": 0, "percent_area": 0}

        # Create a working copy of the frame
        processed_frame = frame.copy()

        # Apply background subtraction
        mask = self.bg_subtractor.apply(frame)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(mask, (self.blur_size, self.blur_size), 0)

        # Threshold the image
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # Dilate the threshold image to fill in holes
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=self.dilation_iterations)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate total motion area
        total_area = sum(cv2.contourArea(c) for c in contours)
        percent_area = (total_area / self.frame_area) * 100

        # Check if motion is detected based on area thresholds
        motion_detected = (total_area >= self.min_area and total_area <= self.max_area)

        # Draw contours on the frame if motion is detected
        if motion_detected:
            cv2.drawContours(processed_frame, contours, -1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Motion: {percent_area:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(processed_frame, f"No Motion: {percent_area:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Handle state changes
        time_since_last_motion = timestamp - self.last_motion_timestamp
        time_since_last_no_motion = timestamp - self.last_no_motion_timestamp

        # Check for motion state changes
        if motion_detected and not self.motion_detected:
            self.motion_detected = True
            self.last_motion_timestamp = timestamp
            self._trigger_motion_callbacks(timestamp, percent_area)
            self.logger.info(f"Motion detected: {percent_area:.2f}% of frame")

        # Check for no-motion state changes
        elif not motion_detected and self.motion_detected:
            if time_since_last_no_motion >= self.min_seconds_between_alerts:
                self.motion_detected = False
                self.last_no_motion_timestamp = timestamp
                self._trigger_no_motion_callbacks(timestamp, time_since_last_motion)
                self.logger.info(f"No motion detected after {time_since_last_motion:.2f} seconds")

        # Return results
        motion_data = {
            "contours": contours,
            "total_area": total_area,
            "percent_area": percent_area
        }

        return motion_detected, processed_frame, motion_data

    def _trigger_no_motion_callbacks(self, timestamp, duration):
        """Trigger all registered no-motion callbacks"""
        for callback in self.no_motion_callbacks:
            try:
                callback(timestamp, duration)
            except Exception as e:
                self.logger.error(f"Error in no-motion callback: {str(e)}")

    def _trigger_motion_callbacks(self, timestamp, percent_area):
        """Trigger all registered motion callbacks"""
        for callback in self.motion_callbacks:
            try:
                callback(timestamp, percent_area)
            except Exception as e:
                self.logger.error(f"Error in motion callback: {str(e)}")

    def reset(self):
        """Reset the background subtractor and state variables"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        self.warmup_frames = 0
        self.motion_detected = False
        self.last_motion_timestamp = 0
        self.last_no_motion_timestamp = 0

        self.logger.info("Motion detector reset")