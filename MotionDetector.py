import cv2
import numpy as np
import time
import logging
from datetime import datetime


class MotionDetector:
    def __init__(self,
                 min_area_percent=0.5,
                 max_area_percent=50,
                 history=50,  # Reduced history frames
                 var_threshold=25,
                 detect_shadows=False,  # Shadows detection turned off
                 blur_size=5,
                 dilation_iterations=1,  # Reduced iterations
                 min_seconds_between_alerts=60,
                 learning_rate=0.05,  # Faster learning rate
                 threshold_value=30,
                 debug=False,
                 resize_factor=0.5  # Added resize factor to work with smaller frames
                 ):
        self.min_area_percent = min_area_percent
        self.max_area_percent = max_area_percent
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.blur_size = blur_size
        self.dilation_iterations = dilation_iterations
        self.min_seconds_between_alerts = min_seconds_between_alerts
        self.learning_rate = learning_rate
        self.threshold_value = threshold_value
        self.debug = debug
        self.resize_factor = resize_factor  # New parameter to resize frames

        # Initialize background subtractor (simplified - using only MOG2)
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

        # Simple warmup counter instead of tracking frames
        self.warmup_counter = 0
        self.warmup_needed = 10  # Reduced warmup frames

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("LightweightMotionDetector")

    def initialize(self, frame):
        """Initialize detector with the first frame dimensions"""
        # Resize frame if needed
        if self.resize_factor != 1.0:
            frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)

        self.frame_height, self.frame_width = frame.shape[:2]
        self.frame_area = self.frame_height * self.frame_width
        self.min_area = int(self.frame_area * (self.min_area_percent / 100))
        self.max_area = int(self.frame_area * (self.max_area_percent / 100))

        self.logger.info(f"Initialized detector with frame dimensions: {self.frame_width}x{self.frame_height}")
        return True

    def register_no_motion_callback(self, callback):
        """Register a callback function to be called when no motion is detected"""
        self.no_motion_callbacks.append(callback)

    def register_motion_callback(self, callback):
        """Register a callback function to be called when motion is detected"""
        self.motion_callbacks.append(callback)

    def detect(self, frame, timestamp=None):
        """
        Detect motion in the given frame using simplified approach
        """
        if timestamp is None:
            timestamp = time.time()


        # Initialize if not already
        if self.frame_width == 0 or self.frame_height == 0:
            self.initialize(frame)

            # Resize frame for performance
        if self.resize_factor != 1.0:
            frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)

        # Create a working copy only if we'll draw on it
        processed_frame = frame.copy() if self.debug else frame

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply blur with fixed kernel size
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        # Handle warmup period
        if self.warmup_counter < self.warmup_needed:
            self.bg_subtractor.apply(gray)
            self.warmup_counter += 1
            return False, processed_frame, {"motion_level": 0, "percent_area": 0}

        # Get foreground mask (use only MOG2 method for simplicity and performance)
        fg_mask = self.bg_subtractor.apply(gray, learningRate=self.learning_rate)

        # Simple morphological operations
        if self.dilation_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)  # Smaller fixed kernel
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=self.dilation_iterations)

        # Calculate motion level
        motion_level = cv2.countNonZero(fg_mask)
        percent_area = (motion_level / self.frame_area) * 100

        # Check if motion is detected based on area thresholds
        motion_detected = (motion_level >= self.min_area and motion_level <= self.max_area)

        # Draw information on the frame if in debug mode
        if self.debug:
            text = f"Motion: {percent_area:.1f}%" if motion_detected else f"No Motion: {percent_area:.1f}%"
            color = (0, 255, 0) if motion_detected else (0, 0, 255)
            cv2.putText(processed_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("fg_mask",fg_mask)
            cv2.imshow("processed_frame", processed_frame)

        # Handle state changes
        if motion_detected and not self.motion_detected:
            self.motion_detected = True
            self.last_motion_timestamp = timestamp
            self._trigger_motion_callbacks(timestamp, percent_area)
            if self.debug:
                self.logger.info(f"Motion detected: {percent_area:.2f}% of frame")
        elif not motion_detected and self.motion_detected:
            if (timestamp - self.last_no_motion_timestamp) >= self.min_seconds_between_alerts:
                self.motion_detected = False
                self.last_no_motion_timestamp = timestamp
                duration = timestamp - self.last_motion_timestamp
                self._trigger_no_motion_callbacks(timestamp, duration)
                if self.debug:
                    self.logger.info(f"No motion detected after {duration:.2f} seconds")

        # Return results
        motion_data = {
            "motion_level": motion_level,
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
        """Reset the detector"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        self.warmup_counter = 0
        self.motion_detected = False
        self.last_motion_timestamp = 0
        self.last_no_motion_timestamp = 0