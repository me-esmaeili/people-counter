import cv2
import numpy as np

class FrameProcessor:
    def __init__(self, width, height, zone_tracker, motion_detector):
        self.width = width
        self.height = height
        self.zone_tracker = zone_tracker
        self.motion_detector = motion_detector
        self.use_resize = True
        self.process_width = 640
        self.process_height = 480

    def set_performance_params(self, params):
        self.use_resize = params.get("use_resize", self.use_resize)
        self.process_width = params.get("process_width", self.process_width)
        self.process_height = params.get("process_height", self.process_height)

    def preprocess_frame(self, frame):
        if self.use_resize:
            return cv2.resize(frame, (self.process_width, self.process_height))
        return frame

    def process_detections(self, results, display_frame, orig_shape):
        if not results or len(results) == 0:
            return display_frame
        boxes = results[0].boxes
        if len(boxes) == 0:
            return display_frame

        scale_x, scale_y = (orig_shape[1] / self.process_width, orig_shape[0] / self.process_height) if self.use_resize else (1.0, 1.0)
        for data in boxes.cpu().data.tolist():
            if len(data) >= 6:
                x1, y1, x2, y2 = map(int, data[0:4])
                if self.use_resize:
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                confidence, class_id = data[4], int(data[5])
                if class_id != 0:
                    continue
                track_id = int(data[6]) if len(data) > 6 else -1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID:{track_id} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(display_frame, (cx, cy), 4, (0, 0, 255), -1)
                if track_id >= 0:
                    self.zone_tracker.track_person(cx, cy, track_id)
        return display_frame

    def process_frame(self, frame, model_handler):
        display_frame = frame.copy()
        orig_shape = frame.shape

        if self.motion_detector.use_bg_subtraction:
            motion_mask, has_motion, motion_rects, motion_percent = self.motion_detector.detect_motion(frame)
            if self.motion_detector.show_motion_mask:
                overlay = display_frame.copy()
                motion_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                motion_color[np.where((motion_color == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
                cv2.addWeighted(motion_color, 0.3, display_frame, 0.7, 0, display_frame)
                for rect in motion_rects:
                    x, y, w, h = rect
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display_frame, f"Motion: {motion_percent:.2f}%", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if not has_motion:
                return self._draw_zones_and_counters(display_frame)

        proc_frame = self.preprocess_frame(frame)
        results = model_handler.detect(proc_frame)
        display_frame = self.process_detections(results, display_frame, orig_shape)
        return self._draw_zones_and_counters(display_frame)

    def _draw_zones_and_counters(self, frame):
        cv2.polylines(frame, [self.zone_tracker.entry_zone], True, (0, 255, 255), 2)
        cv2.polylines(frame, [self.zone_tracker.exit_zone], True, (255, 0, 255), 2)
        cv2.putText(frame, f"IN: {self.zone_tracker.count_in}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.zone_tracker.count_out}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame