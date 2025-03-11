import cv2
import numpy as np
import os
import platform
from ultralytics import YOLO
import time

import concurrent.futures
from pathlib import Path
import cv2

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

    def draw_percentage_on_mask(self,mask, white_percentage):

        # Convert the mask to a 3-channel image for display (if it's grayscale)
        if len(mask.shape) == 2:  # If the image is grayscale
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_display = mask.copy()

        # Define text and position
        text = f"White: {white_percentage:.2f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # Green color
        thickness = 2
        position = (10, 50)  # Top-left corner

        # Draw the text on the image
        cv2.putText(mask_display, text, position, font, font_scale, font_color, thickness)

        return mask_display

    def run(self):
        # Initialize components
        print("Starting PeopleCounter...")
        self.initialize_components()

        print("Loading YOLO model...")
        try:
            # Try standard model first as a test
            model = YOLO('bestPele.pt', task='detect')
            print("Using standard YOLOv8n model for testing")
        except:
            try:
                # Fall back to custom model
                model = self.load_yolo_model()
                print("Using custom model: bestPele.pt")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                return

        bg_subtractor, background, alpha = self.initialize_background_model()

        detection_interval = 2  # Check more frequently
        frame_count = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future_results = None

                while True:
                    start_loop_time = time.time()

                    # Capture and process frame
                    img = self.capture_frame()
                    if img is None:
                        print("No frame captured")
                        break

                    img = self.preprocess_frame(img)

                    # Always make a copy for display
                    img_copy = img.copy()

                    # Run detection on every Nth frame
                    if frame_count % detection_interval == 0:
                        print(f"Running detection on frame {frame_count}")
                        # Run detection directly (no async for debugging)
                        results, scale_factors = self.async_detect_objects(model, img)
                        self.process_detection_results([results, scale_factors], img_copy)

                    # Show frame
                    img_copy = self.draw(img_copy)
                    cv2.imshow(f"People Counter {self.id_points[0]}", img_copy)

                    # Break on 'q' key
                    if cv2.waitKey(1) == ord('q'):
                        break

                    frame_count += 1
                    frame_time = time.time() - start_loop_time
                    print(f"Frame {frame_count} processed in {frame_time:.3f}s ({1 / frame_time:.1f} FPS)")

                    # Add a small delay if processing is too fast
                    if frame_time < 0.03:  # Aim for maximum ~30 FPS
                        time.sleep(0.03 - frame_time)

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def initialize_components(self):
        self.hardware.start()
        self.logger.start()
        self.web_client.start()

    def load_yolo_model(self):
        # Load YOLO model (optimized to use CUDA if available)
        return YOLO('bestPele.pt', task='detect')

    def initialize_background_model(self):
        # Configure background subtraction
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )
        background = None
        alpha = 0.01  # Learning rate for running average
        return bg_subtractor, background, alpha

    def capture_frame(self):
        if self.is_video_file:
            ret, img = self.cap.read()
            if not ret:
                print("End of video file reached")
                return None
        elif self.is_picamera:
            img = self.cap.capture_array()
        else:
            grabbed, img = self.cap.read()
            if not grabbed:
                print("Failed to grab frame")
                return None
        return img

    def preprocess_frame(self, img):
        if self.rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if self.border != 0:
            img = cv2.copyMakeBorder(
                img, self.border, self.border, self.border, self.border,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        self.last_frame_time = time.time()
        return img

    def to_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def calculate_foreground_mask(self, gray, bg_subtractor, background, alpha):
        if background is None:
            background = gray.astype(float)

        # Running average update
        cv2.accumulateWeighted(gray, background, alpha)
        diff_frame = cv2.absdiff(gray, cv2.convertScaleAbs(background))

        # MOG2 subtraction
        fg_mask = bg_subtractor.apply(gray)

        # Combine methods for better results
        _, thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(fg_mask, thresh)

        # Cleanup mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)  # Reduced iterations for speed
        return background, fg_mask

    def calculate_white_percentage(self, fg_mask):
        white_pixels = cv2.countNonZero(fg_mask)
        total_pixels = fg_mask.size
        return (white_pixels / total_pixels) * 100

    def show_mask_with_percentage(self, fg_mask, white_percentage):
        # print(f"White pixels make up {white_percentage:.2f}% of the image.")
        fg_mask_color = self.draw_percentage_on_mask(fg_mask, white_percentage)
        cv2.imshow("mask", fg_mask_color)

    def extract_motion_regions(self, fg_mask, img):
        # Find contours of motion regions
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            region = img[y:y + h, x:x + w]  # Crop region of interest
            regions.append(region)
        return regions

    def async_detect_objects(self, model, img):
        """Improved object detection with better parameters and error handling"""
        try:
            # Preserve aspect ratio during resize
            original_h, original_w = img.shape[:2]
            target_w, target_h = 640, 480  # Higher resolution for better accuracy

            # Create a better scaled version
            scale_w = target_w / original_w
            scale_h = target_h / original_h

            # Resize the image
            resized_img = cv2.resize(img, (target_w, target_h))

            # Run detection with better parameters
            detection = model.track(
                source=resized_img,
                imgsz=(target_w, target_h),
                conf=0.3,  # Lower confidence threshold
                iou=0.45,  # Better IOU threshold for overlapping detections
                persist=True,
                classes=0,  # Specifically target class 0 (person) if that's your model's class
                device='cpu',
                tracker="botsort_1.yaml",
                verbose=False
            )

            print(
                f"Detection results: {len(detection[0].boxes) if detection and len(detection) > 0 else 'No detections'}")

            # Return detections with scaling info for coordinate adjustment
            return [detection], (scale_w, scale_h)

        except Exception as e:
            print(f"Error in object detection: {e}")
            import traceback
            traceback.print_exc()
            return [], (1.0, 1.0)

    def process_detection_results(self, results, img_copy):
        """Improved detection processing with better visualization and debugging"""
        detection_results, scale_factors = results
        scale_w, scale_h = scale_factors

        # Check if YOLO detection results are non-empty
        detections_found = 0

        for detection in detection_results:
            if not detection or len(detection) == 0:
                continue

            # Process each box
            for box in detection[0].boxes:
                try:
                    # Extract box data
                    box_data = box.cpu().data.tolist()

                    # Print box data for debugging
                    print(f"Box data: {box_data}")

                    # Check for valid data structure
                    if len(box_data) >= 6:  # Should have at least x1,y1,x2,y2,conf,cls
                        x1, y1, x2, y2 = box_data[0:4]
                        confidence = box_data[4]
                        class_id = box_data[5]

                        # Get track ID if available (when using tracking)
                        track_id = int(box_data[6]) if len(box_data) > 6 else -1

                        # Skip if confidence is too low (secondary filter)
                        if confidence < self.settings['set_confidence']:
                            continue

                        # Scale coordinates back to original image size if needed
                        x1, x2 = x1 / scale_w, x2 / scale_w
                        y1, y2 = y1 / scale_h, y2 / scale_h

                        # Calculate center of the bounding box
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Draw the bounding box with more visibility
                        cv2.rectangle(
                            img_copy,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0), 2  # Green bounding box
                        )

                        # Draw the label with more information
                        label = f"Person ID:{track_id} Conf:{confidence:.2f}"
                        cv2.putText(
                            img_copy,
                            label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0), 2
                        )

                        # Draw a more visible center point
                        cv2.circle(img_copy, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # Red circle

                        # Process regions for counting logic
                        if track_id > 0:  # Only process valid track IDs
                            self.process_region(cx, cy, track_id)

                        detections_found += 1

                except Exception as e:
                    print(f"Error processing detection box: {e}")

        # Add overall detection count to the frame
        cv2.putText(
            img_copy,
            f"People detected: {detections_found}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    def show_live_feed(self, img_copy):
        img_copy = self.draw(img_copy)
        cv2.imshow(f"People Counter {self.id_points[0]}", img_copy)
        return cv2.waitKey(1) == ord('q')

    def cleanup(self):
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