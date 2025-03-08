import cv2
import numpy as np
import os
import platform
from ultralytics import YOLO
import time

# Conditional import for picamera2 on Raspberry Pi
if platform.system() == "Linux" and os.path.exists("/usr/lib/libcamera.so"):
    try:
        from picamera2 import Picamera2

        PICAMERA_AVAILABLE = True
    except ImportError:
        PICAMERA_AVAILABLE = False
else:
    PICAMERA_AVAILABLE = False

from hardware_controller import HardwareController
from system_logger import SystemLogger
from web_api_client import WebAPIClient
from webcam_stream import WebcamVideoStream


class PeopleCounter:
    """Main class for people counting functionality, compatible with video files, Windows, and Raspberry Pi"""

    def __init__(self, source, id_points, border, rotation, settings):
        self.source = source  # Can be video file path, RTSP URL, or None for default webcam
        self.id_points = id_points
        self.border = border
        self.rotation = rotation
        self.settings = settings
        self.counter_dict = {
            'counterin': [],
            'counterout': [],
            'pts_1': self._convert_to_array(id_points[1:5]),
            'pts_2': self._convert_to_array(id_points[5:9])
        }
        self.hardware = HardwareController()
        self.logger = SystemLogger()
        self.web_client = WebAPIClient(settings['webPath'], id_points[0])
        self.last_frame_time = time.time()
        self._initialize_source()

    def _initialize_source(self):
        """Initialize video source based on input and platform"""
        if os.path.isfile(self.source):  # Check if source is a video file
            self.cap = cv2.VideoCapture(self.source)
            self.is_video_file = True
            self.is_picamera = False
        elif PICAMERA_AVAILABLE and not self.source:  # Use PiCamera if available and no source specified
            self.cap = Picamera2()
            self.cap.preview_configuration.main.format = 'RGB888'
            self.cap.video_configuration.controls.FrameRate = 25.0
            self.cap.start()
            self.is_video_file = False
            self.is_picamera = True
        else:  # Use RTSP or default webcam
            self.cap = WebcamVideoStream(src=self.source if self.source else 0).start()
            self.is_video_file = False
            self.is_picamera = False

    @staticmethod
    def _convert_to_array(region):
        return np.array([[region[0], region[1]], [region[0], region[3]],
                         [region[2], region[3]], [region[2], region[1]]]).reshape(-1, 1, 2)

    def _calculate_counter(self, id, lst_first, lst_second):
        if id not in lst_first:
            lst_first.append(id)
            if id in lst_second:
                lst_first.remove(id)
                lst_second.remove(id)
                return True
        return False

    def process_region(self, cx, cy, ids):
        dist_1 = cv2.pointPolygonTest(self.counter_dict['pts_1'], (cx, cy), False)
        dist_2 = cv2.pointPolygonTest(self.counter_dict['pts_2'], (cx, cy), False)

        if dist_1 == 1:
            if self._calculate_counter(ids, self.counter_dict['counterin'], self.counter_dict['counterout']):
                self.web_client.add_event(1)
                print("State 1")
                self.hardware.set_led(True)
        if dist_2 == 1:
            if self._calculate_counter(ids, self.counter_dict['counterout'], self.counter_dict['counterin']):
                self.web_client.add_event(-1)
                print("State 2")
                self.hardware.set_led(True)

    def draw(self,img):
        cv2.polylines(img, [self.counter_dict['pts_1']], True, (0, 255, 255), 2)  # Yellow
        cv2.polylines(img, [self.counter_dict['pts_2']], True, (255, 0, 255), 2)  # Magenta

        return img

    def run(self):
        self.hardware.start()
        self.logger.start()
        self.web_client.start()

        model = YOLO('bestPele.pt', task='detect')

        # Initialize background subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,  # Number of frames to consider for background
            varThreshold=25,  # Threshold for foreground detection
            detectShadows=False  # Set to True if you want to detect shadows
        )

        # Alternative: Running average background
        background = None
        alpha = 0.01  # Learning rate for running average

        try:
            while True:
                if self.is_video_file:
                    ret, img = self.cap.read()
                    if not ret:
                        print("End of video file reached")
                        break
                elif self.is_picamera:
                    img = self.cap.capture_array()
                else:
                    grabbed, img = self.cap.read()
                    if not grabbed:
                        print("Failed to grab frame")
                        break

                # Process frame
                if self.rotation == 90:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                if self.border != 0:
                    img = cv2.copyMakeBorder(img, self.border, self.border, self.border, self.border,
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])

                self.last_frame_time = time.time()

                # Convert to grayscale for better subtraction
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

                # Update background model and get foreground mask
                if background is None:
                    background = gray.astype(float)

                # Running average update
                cv2.accumulateWeighted(gray, background, alpha)
                diff_frame = cv2.absdiff(gray, cv2.convertScaleAbs(background))

                # Alternative: MOG2 subtraction
                fg_mask = bg_subtractor.apply(gray)

                # Combine both methods for better results
                _, thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
                fg_mask = cv2.bitwise_and(fg_mask, thresh)

                # Morphological operations to clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

                cv2.imshow("mask",fg_mask)


                if self.settings['ShowLive']:
                    img2 = img.copy()

                # Calculate motion based on foreground mask
                motion_level = cv2.countNonZero(fg_mask) 
                print("motion_level:",motion_level)
                if motion_level >= self.settings['MotionSen']:
                    # print("Diff")

                    results = model.track(img, imgsz=(640, 480), show=False,
                                          conf=self.settings['set_confidence'],
                                          persist=True, iou=0.5, device='cpu',
                                          verbose=False)

                    if results:
                        for info in results[0].boxes.cpu().data.tolist():
                            if len(info) == 7:
                                x1, y1, x2, y2, ids, confidence, class_detect = info
                                if confidence >= 0.2:
                                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                                    if self.settings['ShowLive']:
                                        cv2.circle(img2, (int(cx), int(cy)), 50, (0, 255, 0), -1)
                                    self.process_region(cx, cy, ids)
                else:
                    # print("normal")
                    pass

                if self.settings['ShowLive']:
                    # Optional: Show foreground mask for debugging
                    # cv2.imshow("Foreground Mask", fg_mask)
                    img2 = self.draw(img2)
                    cv2.imshow(f"People Counter {self.id_points[0]}", img2)
                    if cv2.waitKey(1) == ord('q'):  # Changed from waitKey(0) to waitKey(1)
                        break

        finally:
            # Cleanup
            if self.is_video_file:
                self.cap.release()
            elif not self.is_picamera:
                self.cap.stop()
            self.hardware.cleanup()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test with a video file
    settings = {
        'MotionSen': 1000000,
        'set_confidence': 0.5,
        'webPath': 'http://example.com/api',
        'ShowLive': True
    }
    counter = PeopleCounter(
        source="test_video.mp4",  # Replace with your video file path
        id_points=[1, 100, 100, 200, 200, 300, 300, 400, 400],
        border=0,
        rotation=0,
        settings=settings
    )
    counter.run()