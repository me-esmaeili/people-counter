import time
import cv2
from VideoCapture import *
from ModelHandler import *
from ZoneTracker import *
from MotionDetector import *
from FrameProcessor import *
class PeopleCounter:
    def __init__(self, source=0, model_path='bestPele.pt', show_live=True):
        self.video_capture = VideoCapture(source)
        self.model_handler = ModelHandler(model_path)
        self.zone_tracker = ZoneTracker(self.video_capture.width, self.video_capture.height)
        self.motion_detector = MotionDetector(self.video_capture.width, self.video_capture.height)
        self.frame_processor = FrameProcessor(self.video_capture.width, self.video_capture.height,
                                             self.zone_tracker, self.motion_detector)
        self.show_live = show_live
        self.batch_size = 4
        self.frame_skip = 0

    def run(self):
        frame_count = 0
        skip_count = 0
        processed_count = 0
        start_time_total = time.time()
        batch_frames = []

        try:
            while True:
                ret, frame = self.video_capture.capture_frame()
                if not ret:
                    print("End of video stream")
                    break

                frame_count += 1
                if self.frame_skip > 0 and skip_count < self.frame_skip:
                    skip_count += 1
                    continue
                skip_count = 0

                batch_frames.append(frame)
                if len(batch_frames) >= self.batch_size:
                    start_time_batch = time.time()
                    for i, frame in enumerate(batch_frames):
                        batch_frames[i] = self.frame_processor.process_frame(frame, self.model_handler)
                    if self.show_live:
                        for frame in batch_frames:
                            cv2.imshow("People Counter", frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                return
                    batch_time = time.time() - start_time_batch
                    processed_count += len(batch_frames)
                    batch_fps = len(batch_frames) / batch_time if batch_time > 0 else 0
                    fps_avg = processed_count / (time.time() - start_time_total)
                    # print(f"Batch FPS: {batch_fps:.1f}, Avg FPS: {fps_avg:.1f}")
                    batch_frames = []

                if self.batch_size <= 1 and self.show_live:
                    processed_frame = self.frame_processor.process_frame(frame, self.model_handler)
                    cv2.imshow("People Counter", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            if batch_frames and self.show_live:
                for frame in batch_frames:
                    cv2.imshow("People Counter", self.frame_processor.process_frame(frame, self.model_handler))
                    cv2.waitKey(1)
            self.video_capture.release()
            cv2.destroyAllWindows()
            total_time = time.time() - start_time_total
            overall_fps = processed_count / total_time if total_time > 0 else 0
            print(f"Final stats: {frame_count} frames, {processed_count} processed, {overall_fps:.1f} FPS")

    def set_batch_params(self, params):
        self.frame_skip = params.get("frame_skip", self.frame_skip)
        self.batch_size = params.get("batch_size", self.batch_size)

if __name__ == "__main__":
    counter = PeopleCounter(source="20231207153936_839_2.avi", model_path="bestPele.pt", show_live=True)
    counter.run()