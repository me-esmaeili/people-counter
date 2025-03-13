import cv2
import os
import logging
from my_object_counter import ObjectCounter
from utility import get_horizontal_line_coordinates
from VideoSource import VideoSource
from ConfigManager import ConfigManager
from CountLogger import  *


class PeopleCounter:
    def __init__(self, config_file):
        self.config_manager = ConfigManager(config_file)


        self.model_config = self.config_manager.get_model_config()
        self.video_config = self.config_manager.get_video_config()
        self.log_config = self.config_manager.get_log_config()


    def _init(self):
        # Model settings
        self.model_filename = self.model_config["model_filename"]
        self.line_width = self.model_config["line_width"]
        self.device = self.model_config["device"]

        # Video processing settings
        self.skip_frames = self.video_config["skip_frames"]
        self.video_writer_codec = self.video_config["video_writer_codec"]
        self.show = self.video_config["show"]
        self.save_video = self.video_config["save_video"]
        self.save_video_path = self.video_config["save_video_path"]
        self.input_width = self.video_config["input_width"]
        self.input_height = self.video_config["input_height"]
        self.swap_direction = self.video_config["swap_direction"]

        # Initialize state variables
        self.entry_count = 0
        self.exit_count = 0
        self.running = False
        self.video_writer = None
        self.counter = None
        self.last_annotated_frame = None
        self.source = None

        # Initialize logger
        self.logger = CSVCountLogger(log_dir=self.log_config["log_dir"], console_log=self.log_config["console_log"])
    def start(self, video_source):
        self._init()
        # Initialize video source
        self.source = VideoSource(video_source)
        if not self.source.initialize():
            raise Exception("Error reading video file")

        # Get video dimensions and fps
        # w, h = self.source.get_dimensions()
        fps = 25.0 if self.source.using_picam else self.source.cap.get(cv2.CAP_PROP_FPS)

        # Calculate line coordinates for counting
        line_points = get_horizontal_line_coordinates(self.input_width, self.input_height)

        # Initialize video writer if needed
        self._setup_video_writer(video_source,self.input_width, self.input_height, fps)

        logging.getLogger("ultralytics").setLevel(logging.ERROR)


        # Initialize object counter
        self.counter = ObjectCounter(
            show=self.show,
            region=line_points,
            model=self.model_filename,
            line_width=self.line_width,
            verbose = False,
            swap_direction= self.swap_direction,
            imgsz=(640, 480),
            device=self.device

        )

        # Process video frames
        self._process_frames()

    def _setup_video_writer(self, video_source, width, height, fps):
        """Set up video writer if saving is enabled"""
        if self.save_video:
            filename, extension = os.path.splitext(video_source)
            output_filename = self.save_video_path if self.save_video_path else f"{filename}_out{extension}"
            self.video_writer = cv2.VideoWriter(
                output_filename,
                cv2.VideoWriter_fourcc(*self.video_writer_codec),
                fps,
                (width, height)
            )
        else:
            self.video_writer = None

    def _process_frames(self):
        """Process video frames with object counting"""
        frame_count = 0
        self.running = True

        # w, h = self.source.get_dimensions()

        while self.running:
            success, frame = self.source.read()

            if not success:
                print("Video frame is empty or video processing has been completed.")
                break

            frame = cv2.resize(frame, (self.input_width, self.input_height))
            frame_count += 1

            # Process every nth frame based on skip_frames setting
            if frame_count % (self.skip_frames + 1) == 0:
                # Process frame with counter
                frame = self.counter.count(frame)
                self.last_annotated_frame = frame.copy()

                # Get current counts
                current_entry_count = self.get_entry_count()
                current_exit_count = self.get_exit_count()

                # Get timestamp
                timestamp = None
                if hasattr(self.source, "get_timestamp"):
                    timestamp = self.source.get_timestamp()
                else:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Check for new entries
                if current_entry_count > self.entry_count:
                    for _ in range(current_entry_count - self.entry_count):
                        self.logger.log_entry(current_entry_count, current_exit_count, timestamp)

                # Check for new exits
                if current_exit_count > self.exit_count:
                    for _ in range(current_exit_count - self.exit_count):
                        self.logger.log_exit(current_entry_count, current_exit_count, timestamp)

                # Update our counts
                self.entry_count = current_entry_count
                self.exit_count = current_exit_count

            elif self.last_annotated_frame is not None:
                frame = self.last_annotated_frame.copy()

            # Write frame to output video if enabled
            if self.video_writer is not None:
                self.video_writer.write(frame)

        self._cleanup_resources()

    def _cleanup_resources(self):
        """Clean up resources without altering the running flag."""
        if self.source:
            self.source.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def stop(self):
        """Stop processing and release resources"""
        self.running = False


    def get_entry_count(self):
        """Get total entry count across all classes"""
        total_in = 0
        if self.counter and hasattr(self.counter, "classwise_counts"):
            for key, value in self.counter.classwise_counts.items():
                if value["IN"] != 0 or value["OUT"] != 0:
                    total_in += value["IN"]
        return total_in

    def get_exit_count(self):
        """Get total exit count across all classes"""
        total_out = 0
        if self.counter and hasattr(self.counter, "classwise_counts"):
            for key, value in self.counter.classwise_counts.items():
                if value["IN"] != 0 or value["OUT"] != 0:
                    total_out += value["OUT"]
        return total_out