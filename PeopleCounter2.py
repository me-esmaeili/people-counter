import cv2
import numpy as np
import os
import time
from pathlib import Path
import concurrent.futures
from ultralytics import YOLO


class PeopleCounter:
    def __init__(self, source=0, model_path='bestPele.pt', show_live=True):

        self.source = source
        self.model_path = model_path
        self.show_live = show_live


        self.initialize_capture()


        self.entry_zone_percent = [[20, 40], [60, 40], [60, 50], [20, 50]]
        self.exit_zone_percent = [[20, 60], [60, 60], [60, 70], [20, 70]]

        # Convert percentage zones to pixel coordinates
        self.entry_zone = self.percent_to_pixel(self.entry_zone_percent)
        self.exit_zone = self.percent_to_pixel(self.exit_zone_percent)

        # Tracking data
        self.people_in_entry = set()
        self.people_in_exit = set()
        self.count_in = 0
        self.count_out = 0


        self.model = self.load_model()

        # Batch processing parameters
        self.batch_size = 4  # Number of frames to process in a batch
        self.frame_buffer = []

        # Resize parameters for faster processing
        self.process_width = 640  # Width to resize frames to for processing
        self.process_height = 480  # Height to resize frames to for processing
        self.use_resize = True  # Whether to use resizing for processing (not for display)

        # Skip frames for higher throughput
        self.frame_skip = 0  # Process every nth frame (0 = process all frames)

        # Background subtraction
        self.use_bg_subtraction = True
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.min_contour_area = 500  # Minimum contour area to consider as movement

        # Motion detection parameters
        self.dilate_kernel = np.ones((5, 5), np.uint8)  # Kernel for dilation
        self.erode_kernel = np.ones((3, 3), np.uint8)  # Kernel for erosion
        self.motion_threshold_percent = 1.0  # Motion must cover at least 1% of frame

        # ROIs for movement detection (percentage coordinates)
        self.motion_roi_percent = None  # By default, use the entire frame
        self.motion_roi = None


        self.show_motion_mask = False  # Show motion detection visualization


        self.background = None
        self.alpha = 0.01  # Learning rate for background accumulation

    def calculate_expanded_roi(self):

        all_points = np.vstack((self.entry_zone, self.exit_zone))

        # Find bounding box
        x_min = np.min(all_points[:, 0])
        y_min = np.min(all_points[:, 1])
        x_max = np.max(all_points[:, 0])
        y_max = np.max(all_points[:, 1])


        width = x_max - x_min
        height = y_max - y_min
        x_expand = int(width * 0.3)
        y_expand = int(height * 0.3)


        x_min = max(0, x_min - x_expand)
        y_min = max(0, y_min - y_expand)
        x_max = min(self.width, x_max + x_expand)
        y_max = min(self.height, y_max + y_expand)

        # Convert to percentage coordinates for the ROI
        x_min_pct = (x_min / self.width) * 100
        y_min_pct = (y_min / self.height) * 100
        x_max_pct = (x_max / self.width) * 100
        y_max_pct = (y_max / self.height) * 100

        # Return as percentage coordinates for a rectangle
        return [
            [x_min_pct, y_min_pct],  # Top-left
            [x_max_pct, y_min_pct],  # Top-right
            [x_max_pct, y_max_pct],  # Bottom-right
            [x_min_pct, y_max_pct]  # Bottom-left
        ]
    def percent_to_pixel(self, percent_coords):

        if percent_coords is None:
            return None

        pixel_coords = []
        for x_percent, y_percent in percent_coords:
            x_pixel = int((x_percent / 100) * self.width)
            y_pixel = int((y_percent / 100) * self.height)
            pixel_coords.append([x_pixel, y_pixel])

        return np.array(pixel_coords, dtype=np.int32)

    def initialize_capture(self):
        """Initialize the video source"""
        # Check if source is a file path
        if isinstance(self.source, str) and os.path.isfile(self.source):
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.source}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.is_file = True
        else:
            # Assume webcam or RTSP stream
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video source: {self.source}")
            self.fps = 30  # Default FPS for live streams
            self.is_file = False

        # Get video dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video source initialized: {self.width}x{self.height} at {self.fps}fps")

    def load_model(self):
        """Load the YOLO model"""
        try:
            # Optimize model for inference
            model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")
            # Print model classes
            print(f"Model classes: {model.names}")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def preprocess_frame(self, frame):
        """Resize frame for faster processing"""
        if not self.use_resize:
            return frame

        return cv2.resize(frame, (self.process_width, self.process_height))

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

    def detect_motion(self, frame):

        if self.motion_roi is not None:
            # Create a mask for the ROI
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [self.motion_roi], 255)
            roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi_frame = frame

        # Convert to grayscale for processing
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Apply enhanced background subtraction
        self.background, fg_mask = self.calculate_foreground_mask(
            gray, self.bg_subtractor, self.background, self.alpha)

        # Find contours in the mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and get bounding rectangles
        motion_rects = []
        total_motion_area = 0
        frame_area = frame.shape[0] * frame.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_rects.append((x, y, w, h))
                total_motion_area += area

        # Calculate motion percentage
        motion_percent = (total_motion_area / frame_area) * 100

        # Determine if there is significant motion based on threshold
        has_motion = motion_percent >= self.motion_threshold_percent

        return fg_mask, has_motion, motion_rects, motion_percent
    def draw_motion_info(self, frame, motion_mask, motion_rects, motion_percent):
        """Draw motion information on frame"""
        # Overlay motion mask with transparency
        overlay = frame.copy()
        motion_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        motion_color[np.where((motion_color == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # Make motion areas red
        cv2.addWeighted(motion_color, 0.3, frame, 0.7, 0, frame)

        # Draw rectangles around motion areas
        for rect in motion_rects:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Add motion percentage text
        cv2.putText(frame, f"Motion: {motion_percent:.2f}%", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def process_detections(self, results, display_frame, orig_shape=None):
        """Process detection results and update tracking"""
        if not results or len(results) == 0:
            return display_frame

        # Process each detection
        boxes = results[0].boxes

        if len(boxes) == 0:
            return display_frame

        # Get all boxes as a tensor (faster than processing one by one)
        box_data = boxes.cpu().data.tolist()

        # Get scale factors if we resized the frame
        scale_x, scale_y = 1.0, 1.0
        if self.use_resize and orig_shape is not None:
            scale_x = orig_shape[1] / self.process_width
            scale_y = orig_shape[0] / self.process_height

        for data in box_data:
            # Check if we have enough data
            if len(data) >= 6:  # x1,y1,x2,y2,conf,cls,[track_id]
                x1, y1, x2, y2 = map(int, data[0:4])

                # Scale coordinates back if we resized
                if self.use_resize:
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                confidence = data[4]
                class_id = int(data[5])

                # Skip if this isn't a person (class 0)
                if class_id != 0:
                    continue

                # Get track ID if available
                track_id = int(data[6]) if len(data) > 6 else -1

                # Calculate center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f"ID:{track_id} {confidence:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add center point
                cv2.circle(display_frame, (cx, cy), 4, (0, 0, 255), -1)

                # Check if person is in entry or exit zone
                if track_id >= 0:  # Only track valid IDs
                    self.track_person(cx, cy, track_id)

        return display_frame

    def process_frame(self, frame):
        """Process a single frame with object detection"""
        # Make a copy for display
        display_frame = frame.copy()
        orig_shape = frame.shape

        # First apply background subtraction to detect motion
        if self.use_bg_subtraction:
            motion_mask, has_motion, motion_rects, motion_percent = self.detect_motion(frame)

            # Visualize motion if needed
            if self.show_motion_mask:
                display_frame = self.draw_motion_info(display_frame, motion_mask, motion_rects, motion_percent)

            # Only run detection if significant motion is detected
            if not has_motion:
                # Draw tracking zones
                cv2.polylines(display_frame, [self.entry_zone], True, (0, 255, 255), 2)
                cv2.polylines(display_frame, [self.exit_zone], True, (255, 0, 255), 2)

                # Add counter text
                cv2.putText(display_frame, f"IN: {self.count_in}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"OUT: {self.count_out}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Skip object detection if no motion
                return display_frame

        # Preprocess for faster detection
        proc_frame = self.preprocess_frame(frame)

        # Run detection
        results = self.model.track(proc_frame, persist=True, conf=0.4, iou=0.45)

        # Process detections and update display frame
        display_frame = self.process_detections(results, display_frame, orig_shape)

        # Draw tracking zones
        cv2.polylines(display_frame, [self.entry_zone], True, (0, 255, 255), 2)  # Entry zone in yellow
        cv2.polylines(display_frame, [self.exit_zone], True, (255, 0, 255), 2)  # Exit zone in magenta

        # Add counter text
        cv2.putText(display_frame, f"IN: {self.count_in}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"OUT: {self.count_out}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return display_frame

    def process_frame_batch(self, frames):
        """Process a batch of frames using YOLO's batching capability"""
        # Store original shapes for scaling
        orig_shapes = [frame.shape for frame in frames]

        # Create copies for display
        display_frames = [frame.copy() for frame in frames]

        # Check for motion in each frame if bg subtraction is enabled
        if self.use_bg_subtraction:
            frames_with_motion = []
            frames_without_motion = []
            motion_indices = []

            for i, frame in enumerate(frames):
                motion_mask, has_motion, motion_rects, motion_percent = self.detect_motion(frame)

                # Visualize motion if needed
                if self.show_motion_mask:
                    display_frames[i] = self.draw_motion_info(display_frames[i], motion_mask, motion_rects,
                                                              motion_percent)

                if has_motion:
                    frames_with_motion.append(frame)
                    motion_indices.append(i)
                else:
                    frames_without_motion.append(i)

            # Only process frames with motion
            if len(frames_with_motion) == 0:
                # No motion in any frame, just draw zones and counters
                for i in range(len(frames)):
                    # Draw tracking zones
                    cv2.polylines(display_frames[i], [self.entry_zone], True, (0, 255, 255), 2)
                    cv2.polylines(display_frames[i], [self.exit_zone], True, (255, 0, 255), 2)

                    # Add counter text
                    cv2.putText(display_frames[i], f"IN: {self.count_in}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frames[i], f"OUT: {self.count_out}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                return display_frames

            # Process only frames with motion
            proc_frames = [self.preprocess_frame(frame) for frame in frames_with_motion]
            results = self.model.track(proc_frames, persist=True, conf=0.4, iou=0.45)

            # Map results back to original frames
            for i, result_idx in enumerate(motion_indices):
                if i < len(results):
                    # Process detections for this frame
                    display_frames[result_idx] = self.process_detections(
                        [results[i]], display_frames[result_idx], orig_shapes[result_idx])
        else:
            # No bg subtraction, process all frames
            proc_frames = [self.preprocess_frame(frame) for frame in frames]
            results = self.model.track(proc_frames, persist=True, conf=0.4, iou=0.45)

            # Process results for each frame
            for i, frame in enumerate(frames):
                if i < len(results):
                    display_frames[i] = self.process_detections(
                        [results[i]], display_frames[i], orig_shapes[i])

        # Draw zones and counters on all frames
        for i in range(len(frames)):
            # Draw tracking zones
            cv2.polylines(display_frames[i], [self.entry_zone], True, (0, 255, 255), 2)
            cv2.polylines(display_frames[i], [self.exit_zone], True, (255, 0, 255), 2)

            # Add counter text
            cv2.putText(display_frames[i], f"IN: {self.count_in}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frames[i], f"OUT: {self.count_out}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return display_frames

    def track_person(self, cx, cy, track_id):
        """Track a person's movement through zones"""
        # Check if person is in entry zone
        in_entry = cv2.pointPolygonTest(self.entry_zone, (cx, cy), False) >= 0
        in_exit = cv2.pointPolygonTest(self.exit_zone, (cx, cy), False) >= 0

        if in_entry:
            if track_id not in self.people_in_entry:
                self.people_in_entry.add(track_id)
                print(f"Person {track_id} entered entry zone")

        # Check if person is in exit zone
        if in_exit:
            if track_id not in self.people_in_exit:
                self.people_in_exit.add(track_id)
                print(f"Person {track_id} entered exit zone")

            # If they were previously in entry zone, count as IN
            if track_id in self.people_in_entry:
                self.people_in_entry.remove(track_id)
                self.count_in += 1
                print(f"Person {track_id} counted as IN")

        # If they move from exit to entry, count as OUT
        if track_id in self.people_in_exit and in_entry:
            self.people_in_exit.remove(track_id)
            self.count_out += 1
            print(f"Person {track_id} counted as OUT")

    def set_zones_percent(self, entry_zone_percent=None, exit_zone_percent=None):

        if entry_zone_percent is not None:
            self.entry_zone_percent = entry_zone_percent
            self.entry_zone = self.percent_to_pixel(entry_zone_percent)

        if exit_zone_percent is not None:
            self.exit_zone_percent = exit_zone_percent
            self.exit_zone = self.percent_to_pixel(exit_zone_percent)

        print("Tracking zones updated using percentage coordinates")

    def set_motion_roi(self, motion_roi_percent=None):

        if motion_roi_percent is not None:
            self.motion_roi_percent = motion_roi_percent
            self.motion_roi = self.percent_to_pixel(motion_roi_percent)
            print("Motion ROI updated using percentage coordinates")

    def set_performance_params(self, use_resize=True, process_width=640, process_height=480,
                               frame_skip=0, batch_size=4):

        self.use_resize = use_resize
        self.process_width = process_width
        self.process_height = process_height
        self.frame_skip = frame_skip
        self.batch_size = batch_size
        print(f"Performance parameters updated: resize={use_resize}, size={process_width}x{process_height}, " +
              f"skip={frame_skip}, batch={batch_size}")

    def set_bg_subtraction_params(self, enabled=True, history=500, threshold=16, min_area=500,
                                  motion_threshold_percent=1.0, show_mask=False, alpha=0.01):

        self.use_bg_subtraction = enabled
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=threshold, detectShadows=False)
        self.min_contour_area = min_area
        self.motion_threshold_percent = motion_threshold_percent
        self.show_motion_mask = show_mask
        self.alpha = alpha

        print(f"Background subtraction updated: enabled={enabled}, threshold={motion_threshold_percent}%, " +
              f"min_area={min_area}, history={history}, alpha={alpha}")
    def run(self):
        """Run the people counter"""
        try:
            # Main processing loop
            frame_count = 0
            skip_count = 0
            processed_count = 0
            detected_motion_count = 0
            start_time_total = time.time()
            fps_avg = 0

            # For batch processing
            batch_frames = []
            batch_timestamps = []

            while True:
                # Capture frame
                ret, frame = self.cap.read()

                if not ret:
                    print("End of video stream")
                    break

                frame_count += 1

                # Skip frames if needed
                if self.frame_skip > 0 and skip_count < self.frame_skip:
                    skip_count += 1
                    continue
                else:
                    skip_count = 0

                # Add frame to batch
                batch_frames.append(frame)
                batch_timestamps.append(time.time())

                # Process batch when it reaches batch_size
                if len(batch_frames) >= self.batch_size:
                    start_time_batch = time.time()

                    # Process the batch
                    processed_frames = self.process_frame_batch(batch_frames)

                    # Display and handle processed frames
                    for i, (proc_frame, timestamp) in enumerate(zip(processed_frames, batch_timestamps)):
                        if self.show_live:
                            cv2.imshow("People Counter", proc_frame)

                            # Exit on 'q' key (check only on last frame for efficiency)
                            if i == len(processed_frames) - 1:
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    return

                    # Calculate batch FPS
                    batch_time = time.time() - start_time_batch
                    batch_fps = len(batch_frames) / batch_time if batch_time > 0 else 0

                    # Update stats
                    processed_count += len(batch_frames)
                    fps_avg = processed_count / (time.time() - start_time_total)

                    # Print stats
                    print(f"Frames: {frame_count}, Processed: {processed_count}, " +
                          f"Batch FPS: {batch_fps:.1f}, Avg FPS: {fps_avg:.1f}, " +
                          f"In: {self.count_in}, Out: {self.count_out}")

                    # Clear the batch
                    batch_frames = []
                    batch_timestamps = []

                # Process remaining frames individually if not using batching
                if self.batch_size <= 1 and self.show_live:
                    # Process single frame
                    start_time = time.time()
                    processed_frame = self.process_frame(frame)

                    # Display the result
                    cv2.imshow("People Counter", processed_frame)

                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # Calculate FPS
                    frame_time = time.time() - start_time
                    fps = 1 / frame_time if frame_time > 0 else 0

                    # Print stats periodically
                    processed_count += 1
                    if processed_count % 30 == 0:
                        print(f"Frame {frame_count}: {fps:.1f} FPS, In: {self.count_in}, Out: {self.count_out}")

        except Exception as e:
            print(f"Error running people counter: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Process any remaining frames in the batch
            if batch_frames:
                processed_frames = self.process_frame_batch(batch_frames)

                # Display remaining frames
                if self.show_live:
                    for proc_frame in processed_frames:
                        cv2.imshow("People Counter", proc_frame)
                        cv2.waitKey(1)

            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            print("People counter stopped")

            # Print final statistics
            total_time = time.time() - start_time_total
            overall_fps = processed_count / total_time if total_time > 0 else 0
            print(f"Final stats: {frame_count} frames, {processed_count} processed, " +
                  f"{overall_fps:.1f} FPS, In: {self.count_in}, Out: {self.count_out}")


# Example usage
if __name__ == "__main__":
    # Create the counter
    counter = PeopleCounter(
        source="20231207153936_839_2.avi",  # Video file path
        model_path="bestPele.pt",  # Your custom model
        show_live=True  # Show the video with detection results
    )

    # Set performance parameters for faster processing
    counter.set_performance_params(
        use_resize=True,  # Resize frames for faster processing
        process_width=640,  # Process at 640x480 resolution
        process_height=480,
        frame_skip=2,  # Process every 3rd frame
        batch_size=4  # Process 4 frames at once
    )

    # Configure background subtraction
    counter.set_bg_subtraction_params(
        enabled=True,  # Enable background subtraction
        history=500,  # Background history
        threshold=16,  # Detection threshold
        min_area=0.2,  # Minimum contour area
        motion_threshold_percent=1.0,  # Motion must cover at least 1% of frame
        show_mask=True  # Show motion visualization
    )



    # Set custom zones using percentage coordinates
    counter.set_zones_percent(
        entry_zone_percent=[[15, 40],
                            [80, 40],
                            [80, 45],
                            [15, 45]],  # Upper region: 25-75% width, 40-45% height
        exit_zone_percent=[[15, 60],
                           [80, 60],
                           [80, 65],
                           [15, 65]]  # Lower region: 25-75% width, 60-65% height
    )
    motion_roi = counter.calculate_expanded_roi()
    counter.set_motion_roi(motion_roi)
    # Run the counter
    counter.run()