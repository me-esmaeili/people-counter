from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque


class PeopleCounter:
    def __init__(self, model_path, video_source=0, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.video = cv2.VideoCapture(video_source)

        self.confidence_threshold = confidence_threshold
        self.entry_count = 0
        self.exit_count = 0
        self.tracked_objects = {}  # Store object IDs and their positions
        self.crossing_status = {}  # Track crossing state (above/below center line)
        self.next_id = 0
        self.previous_positions = deque(maxlen=2)

        ret, frame = self.video.read()
        if ret:
            self.frame_height = frame.shape[0]
            self.center_line = self.frame_height // 2
        else:
            raise ValueError("Could not read from video source")

    def detect_people(self, frame):
        results = self.model(frame, conf=self.confidence_threshold, classes=[0])
        boxes = []
        confidences = []

        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)

                x1, y1, x2, y2 = map(int, xyxy)
                w = x2 - x1
                h = y2 - y1

                boxes.append([x1, y1, w, h])
                confidences.append(conf)

        return boxes, confidences

    def track_and_count(self, frame):
        boxes, confidences = self.detect_people(frame)
        current_positions = {}

        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x, y, w, h = box
            center_x = x + w // 2
            center_y = y + h // 2

            obj_id = self.assign_id(center_x, center_y)
            current_positions[obj_id] = center_y

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.count_movement(current_positions)
        self.previous_positions.append(current_positions.copy())

        cv2.line(frame, (0, self.center_line), (frame.shape[1], self.center_line), (0, 0, 255), 2)
        cv2.putText(frame, f"Entries: {self.entry_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {self.exit_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def assign_id(self, x, y):
        if not self.tracked_objects:
            self.tracked_objects[self.next_id] = (x, y)
            self.crossing_status[self.next_id] = None  # Initial state
            self.next_id += 1
            return self.next_id - 1

        min_dist = float('inf')
        closest_id = None

        for obj_id, (prev_x, prev_y) in self.tracked_objects.items():
            dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            if dist < min_dist and dist < 100:
                min_dist = dist
                closest_id = obj_id

        if closest_id is not None:
            self.tracked_objects[closest_id] = (x, y)
            return closest_id
        else:
            self.tracked_objects[self.next_id] = (x, y)
            self.crossing_status[self.next_id] = None
            self.next_id += 1
            return self.next_id - 1

    def count_movement(self, current_positions):
        if len(self.previous_positions) < 2:
            return

        prev_positions = self.previous_positions[0]

        for obj_id, curr_y in current_positions.items():
            if obj_id in prev_positions:
                prev_y = prev_positions[obj_id]

                # Update crossing status
                curr_above = curr_y < self.center_line
                prev_above = prev_y < self.center_line

                # Check previous status
                prev_status = self.crossing_status[obj_id]

                # Entry: from below to above
                if not prev_above and curr_above and prev_status != "above":
                    self.entry_count += 1
                    self.crossing_status[obj_id] = "above"

                # Exit: from above to below
                elif prev_above and not curr_above and prev_status != "below":
                    self.exit_count += 1
                    self.crossing_status[obj_id] = "below"

    def run(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            frame = self.track_and_count(frame)
            cv2.imshow('People Counter', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()


def main():
    counter = PeopleCounter(
        model_path='bestpele.pt',
        video_source="20231207153936_839_2.avi",
        confidence_threshold=0.5
    )
    counter.run()


if __name__ == "__main__":
    main()