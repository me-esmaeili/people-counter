import cv2
import platform
import os

class VideoCapture:
    def __init__(self, source=0):
        self.source = source
        self.is_picamera = False
        self.picam = None
        self.cap = None
        self.initialize_capture()

    def initialize_capture(self):
        is_raspberry_pi = platform.machine().startswith('arm') or platform.machine().startswith('aarch')
        if isinstance(self.source, str) and os.path.isfile(self.source):
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.source}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.is_file = True
            self.is_picamera = False
        elif is_raspberry_pi and self.source in [0, '0', '/dev/video0'] and not isinstance(self.source, str):
            try:
                from picamera2 import Picamera2
                self.picam = Picamera2()
                config = self.picam.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
                self.picam.configure(config)
                self.picam.start()
                self.width, self.height = 1280, 720
                self.fps = 30
                self.is_file = False
                self.is_picamera = True
                print("Using Picamera2 on Raspberry Pi")
            except ImportError:
                self._fallback_to_opencv()
        else:
            self._fallback_to_opencv()

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) if not self.is_picamera else 1280)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if not self.is_picamera else 720)
        self.fps = self.fps if hasattr(self, 'fps') else 30
        print(f"Video source initialized: {self.width}x{self.height} at {self.fps}fps")

    def _fallback_to_opencv(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")
        self.fps = 30
        self.is_file = False
        self.is_picamera = False

    def capture_frame(self):
        if self.is_picamera:
            frame = self.picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        return self.cap.read()

    def release(self):
        if self.is_picamera and self.picam:
            self.picam.stop()
            print("Picamera2 stopped")
        elif self.cap:
            self.cap.release()
            print("OpenCV camera released")