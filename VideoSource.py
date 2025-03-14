import cv2
import os
import platform


class VideoSource:
    """Handles video input from different sources"""

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.using_picam = False
        self.picam2 = None

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
                    self.picam2.preview_configuration.sensor.output_size = (640, 480)
                    self.picam2.preview_configuration.sensor.bit_depth = 10
                    self.picam2.preview_configuration.main.format = 'RGB888'

                    # Configure video settings
                    self.picam2.video_configuration.controls.FrameRate = 25.0
                    self.picam2.video_configuration.controls.AwbEnable = True
                    # Uncomment and set custom gains if needed
                    # self.picam2.video_configuration.controls.ColourGains = (9, 5)
                    self.picam2.video_configuration.controls.Contrast = 0
                    self.picam2.video_configuration.controls.ExposureTime = 1  # in microseconds
                    # Uncomment and set AWB mode if needed
                    # self.picam2.video_configuration.controls.AwbMode = 'Cloudy'

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

        # Get video dimensions
        # self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = 640
        self.frame_height = 480
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

    def isCamera (self):
        return       (   self.using_picam or
            (isinstance(self.source, str) and
             (not os.path.isfile(self.source) or self.source.isdigit())) or
            isinstance(self.source, int))
