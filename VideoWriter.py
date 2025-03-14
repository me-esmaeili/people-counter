import cv2
import os
import time
import logging


class VideoWriter:
    def __init__(self, config):
        # Extract configuration
        self.video_writer_codec = config["video_writer_codec"]
        self.save_video_path = config["save_video_path"]
        self.input_width = config["input_width"]
        self.input_height = config["input_height"]
        self.video_duration = config["video_duration"]
        self.max_video_files = config["max_video_files"]
        self.fps = 25

        # Initialize state variables
        self.video_writer = None
        self.current_video_frames = 0
        self.frames_per_video = int(self.fps * self.video_duration)  # Calculate frames per video segment
        self.video_file_counter = 0
        self.video_files = []

        # Create output directory if it doesn't exist
        if not os.path.exists(self.save_video_path):
            os.makedirs(self.save_video_path)

        # Initialize first video writer
        self._create_new_video_writer()

    def write(self, frame):
        """Write a frame to the current video file"""
        if self.video_writer is not None:
            self.video_writer.write(frame)
            self.current_video_frames += 1

            # Check if we need to start a new video file
            if self.current_video_frames >= self.frames_per_video:
                # Close current video writer
                self.release()

                # Create new video writer
                self._create_new_video_writer()
                logging.info(f"Created new video file: {self.video_files[-1]}")

    def _create_new_video_writer(self):
        """Create a new video writer with timestamp-based filename"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(self.save_video_path, f"video_{timestamp}.mp4")

        # Create new video writer
        self.video_writer = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc(*self.video_writer_codec),
            self.fps,
            (self.input_width, self.input_height)
        )

        # Add to list of video files
        self.video_files.append(output_filename)
        self.video_file_counter += 1
        self.current_video_frames = 0

        # Manage circular buffer of files
        self._manage_video_files()

        return output_filename

    def _manage_video_files(self):
        """Maintain circular buffer of video files, removing oldest when exceeding max"""
        while len(self.video_files) > self.max_video_files:
            oldest_file = self.video_files.pop(0)
            try:
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
                    logging.info(f"Removed oldest video file: {oldest_file}")
            except Exception as e:
                logging.error(f"Error removing file {oldest_file}: {str(e)}")

    def set_fps(self, fps):
        """Update the FPS and recalculate frames per video"""
        self.fps = fps
        self.frames_per_video = int(self.fps * self.video_duration)

    def release(self):
        """Release the current video writer"""
        if self.video_writer is not None:
            # self.video_writer.write_done = True
            self.video_writer.release()
            self.video_writer = None