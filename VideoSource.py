import cv2
import os
import platform


class VideoSource:
    """Handles video input from different sources"""

    def __init__(self, source, config):
        self.source = source
        self.cap = None
        self.using_picam = False
        self.picam2 = None
        self.config = config
        self.frame_width = config.get("video_width", 640)
        self.frame_height = config.get("video_height", 480)
        self.fps = config.get("video_fps", 25.0)

    def initialize(self):
        """Initialize the video source based on platform"""
        # Check if we're on Raspberry Pi
        if platform.system() == "Linux" and os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model") as f:
                model = f.read()
            if "Raspberry Pi" in model:
                try:
                    from picamera2 import Picamera2

                    self.picam2 = Picamera2()

                    # Configure the camera for preview
                    self.picam2.preview_configuration.sensor.output_size = (self.frame_width, self.frame_height)
                    self.picam2.preview_configuration.sensor.bit_depth = 10
                    self.picam2.preview_configuration.main.format = 'RGB888'

                    # Configure video settings
                    self.picam2.video_configuration.controls.FrameRate = self.fps
                    self.picam2.video_configuration.controls.AwbEnable = True
                    self.picam2.video_configuration.controls.Contrast = 0
                    self.picam2.video_configuration.controls.ExposureTime = 1  # in microseconds

                    # Apply configuration
                    config = self.picam2.create_video_configuration()
                    self.picam2.configure(config)
                    self.picam2.start()

                    print("Using PiCamera2 on Raspberry Pi")
                    self.using_picam = True

                    # Get a frame to determine dimensions
                    test_frame = self.picam2.capture_array()
                    self.frame_height, self.frame_width = test_frame.shape[:2]
                    return True
                except ImportError:
                    print("PiCamera2 not found, falling back to OpenCV")

        # Default to OpenCV VideoCapture
        print(f"Using OpenCV VideoCapture with source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Failed to open video source: {self.source}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        return True

    def read(self):
        """Read a frame from the video source"""
        if self.using_picam:
            frame = self.picam2.capture_array()
            return True, frame
        else:
            return self.cap.read()

    def get_dimensions(self):
        """Return the frame dimensions"""
        return self.frame_width, self.frame_height

    def release(self):
        """Release the video source"""
        if self.using_picam and self.picam2:
            self.picam2.stop()
        elif self.cap:
            self.cap.release()

    def isCamera(self):
        return (
                self.using_picam or
                (isinstance(self.source, str) and
                 (not os.path.isfile(self.source) or self.source.isdigit())) or
                isinstance(self.source, int)
        )
