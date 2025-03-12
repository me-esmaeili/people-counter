import cv2
import numpy as np

class MotionDetector:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.use_bg_subtraction = True
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.min_contour_area = 500
        self.motion_threshold_percent = 1.0
        self.show_motion_mask = False
        self.background = None
        self.alpha = 0.01
        self.motion_roi = None

    def detect_motion(self, frame):
        if self.motion_roi is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [self.motion_roi], 255)
            roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi_frame = frame

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        self.background, fg_mask = self.calculate_foreground_mask(gray)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_rects = []
        total_motion_area = 0
        frame_area = frame.shape[0] * frame.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_rects.append((x, y, w, h))
                total_motion_area += area

        motion_percent = (total_motion_area / frame_area) * 100
        has_motion = motion_percent >= self.motion_threshold_percent
        return fg_mask, has_motion, motion_rects, motion_percent

    def calculate_foreground_mask(self, gray):
        if self.background is None:
            self.background = gray.astype(float)
        cv2.accumulateWeighted(gray, self.background, self.alpha)
        diff_frame = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        fg_mask = self.bg_subtractor.apply(gray)
        _, thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(fg_mask, thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        return self.background, fg_mask

    def set_motion_roi(self, motion_roi_percent):
        self.motion_roi = self.percent_to_pixel(motion_roi_percent)
        print("Motion ROI updated")

    def percent_to_pixel(self, percent_coords):
        if percent_coords is None:
            return None
        return np.array([[int((x / 100) * self.width), int((y / 100) * self.height)]
                         for x, y in percent_coords], dtype=np.int32)

    def set_bg_subtraction_params(self, params):
        self.use_bg_subtraction = params.get("enabled", self.use_bg_subtraction)
        if "history" in params or "threshold" in params:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=params.get("history", 500),
                varThreshold=params.get("threshold", 25),
                detectShadows=False
            )
        self.min_contour_area = params.get("min_area", self.min_contour_area)
        self.motion_threshold_percent = params.get("motion_threshold_percent", self.motion_threshold_percent)
        self.show_motion_mask = params.get("show_mask", self.show_motion_mask)
        self.alpha = params.get("alpha", self.alpha)