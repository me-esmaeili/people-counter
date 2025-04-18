import cv2
import logging
import threading
import queue
import time
import numpy as np
from my_object_counter import ObjectCounter
from utility import get_horizontal_line_coordinates
from VideoSource import VideoSource
from ConfigManager import ConfigManager
from CountLogger import CountLogger
from MotionDetector import MotionDetector
from VideoWriter import VideoWriter


class PeopleCounter:
    def __init__(self, config_file):
        logging.info("Initializing PeopleCounter...")
        self.config_manager = ConfigManager(config_file)
        self.model_config = self.config_manager.get_model_config()
        self.video_config = self.config_manager.get_video_config()
        self.log_config = self.config_manager.get_log_config()
        self.motion_config = self.config_manager.get_motion_config()
        logging.info("Configuration loaded.")

    def _init(self):
        logging.info("Initializing internal components...")

        self.model_filename = self.model_config["model_filename"]
        self.line_width = self.model_config["line_width"]
        self.device = self.model_config["device"]

        self.skip_frames = self.video_config["skip_frames"]
        self.show = self.video_config["show"]
        self.draw = self.video_config["draw"]
        self.save_video = self.video_config["save_video"]
        self.input_width = self.video_config["input_width"]
        self.input_height = self.video_config["input_height"]
        self.swap_direction = self.video_config["swap_direction"]

        self.entry_count = 0
        self.exit_count = 0
        self.running = False
        self.counter = None
        self.last_annotated_frame = None
        self.source = None

        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        self.video_writer = None
        if self.save_video:
            self.video_writer = VideoWriter(self.video_config)
            logging.info("VideoWriter initialized.")

        self.motion_detector = None
        if self.motion_config["motion_enabled"]:
            self._init_motion_detector()

        self.logger = CountLogger(
            log_dir=self.log_config["log_dir"],
            console_log=self.log_config["console_log"],
            max_logs=self.log_config["max_logs"]
        )
        self.should_count = False
        logging.info("Initialization complete.")

    def _init_motion_detector(self):
        self.motion_detector = MotionDetector(
            min_area_percent=self.motion_config["min_area_percent"],
            debug=False
        )
        logging.info("Motion detector initialized.")

    def start(self, video_source):
        logging.info("Starting PeopleCounter with video source: %s", video_source)
        self._init()
        self.source = VideoSource(video_source)
        if not self.source.initialize():
            logging.error("Failed to initialize video source.")
            raise Exception("Error reading video file")
        logging.info("Video source initialized.")

        # تنظیم FPS برای استفاده از 15 FPS
        if self.source.using_picam:
            self.fps = 15.0
        else:
            source_fps = self.source.cap.get(cv2.CAP_PROP_FPS)
            self.fps = min(source_fps, 15.0) if source_fps and source_fps > 0 else 15.0

        if self.video_writer:
            self.video_writer.set_fps(self.fps)

        line_points = get_horizontal_line_coordinates(self.input_width, self.input_height)
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        self.counter = ObjectCounter(
            show=self.show,
            draw=self.draw,
            region=line_points,
            model=self.model_filename,
            line_width=self.line_width,
            verbose=False,
            swap_direction=self.swap_direction,
            imgsz=(self.input_width, self.input_height),
            device=self.device
        )
        logging.info("ObjectCounter initialized.")

        self.running = True

        capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        process_thread = threading.Thread(target=self._process_frames, daemon=True)
        output_thread = threading.Thread(target=self._output_frames, daemon=True)

        capture_thread.start()
        process_thread.start()
        output_thread.start()

        capture_thread.join()
        process_thread.join()
        output_thread.join()

        self._cleanup_resources()

    def _capture_frames(self):
        logging.info("Capture thread started.")
        frame_count = 0
        while self.running:
            success, frame = self.source.read()
            if not success:
                logging.warning("Failed to read frame. Stopping...")
                self.running = False
                break

            frame = cv2.resize(frame, (self.input_width, self.input_height))
            timestamp = getattr(self.source, "get_timestamp", lambda: time.time())()
            frame_count += 1

            should_process = self.source.isCamera() or (frame_count % (self.skip_frames + 1) == 0)
            if should_process:
                try:
                    self.frame_queue.put_nowait((frame, timestamp, frame_count))
                except queue.Full:
                    logging.warning("Frame queue full. Dropping frame %d", frame_count)

    def _process_frames(self):
        logging.info("Processing thread started.")
        while self.running:
            try:
                frame, timestamp, frame_count = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            if self.motion_detector:
                motion_detected, _, motion_data = self.motion_detector.detect(frame, timestamp)
                self.should_count = motion_detected
                if motion_detected:
                    logging.info("Motion detected (%.2f%%) at frame %d", motion_data["percent_area"], frame_count)

            else:
                self.should_count = True

            processed_frame = self.counter.count(frame) if self.should_count else frame.copy()
            self.last_annotated_frame = processed_frame.copy()

            if self.should_count:
                current_entry = self.get_entry_count()
                current_exit = self.get_exit_count()

                if current_entry > self.entry_count:
                    for _ in range(current_entry - self.entry_count):
                        self.logger.log_entry(current_entry, current_exit, timestamp)
                        logging.info("Entry logged. Total: %d", current_entry)

                if current_exit > self.exit_count:
                    for _ in range(current_exit - self.exit_count):
                        self.logger.log_exit(current_entry, current_exit, timestamp)
                        logging.info("Exit logged. Total: %d", current_exit)

                self.entry_count = current_entry
                self.exit_count = current_exit

            try:
                self.result_queue.put_nowait((processed_frame, timestamp))
            except queue.Full:
                logging.warning("Result queue full. Dropping processed frame %d", frame_count)

            self.frame_queue.task_done()

    def _output_frames(self):
        logging.info("Output thread started.")
        while self.running:
            try:
                frame, timestamp = self.result_queue.get(timeout=1)
            except queue.Empty:
                continue

            if self.save_video and self.video_writer:
                self.video_writer.write(frame)

            self.result_queue.task_done()

    def _cleanup_resources(self):
        logging.info("Cleaning up resources...")
        if self.source:
            self.source.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete.")

    def stop(self):
        logging.info("Stopping PeopleCounter...")
        self.running = False

    def get_entry_count(self):
        if not self.counter or not hasattr(self.counter, "classwise_counts"):
            return 0
        return sum(v["IN"] for v in self.counter.classwise_counts.values())

    def get_exit_count(self):
        if not self.counter or not hasattr(self.counter, "classwise_counts"):
            return 0
        return sum(v["OUT"] for v in self.counter.classwise_counts.values())
