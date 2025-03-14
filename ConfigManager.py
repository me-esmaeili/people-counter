import json


class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading configuration: {e}")
            return {}

    def get(self, key, default=None):
        """Get configuration value with fallback to default"""
        return self.config.get(key, default)

    def get_model_config(self):
        """Get model related configuration"""
        return {
            "model_filename": self.get("model_filename", "assets/bestpele.pt"),
            "line_width": self.get("line_width", 2),
            "device": self.get("device", "cpu")
        }

    def get_video_config(self):
        """Get video processing configuration"""
        return {
            "skip_frames": self.get("skip_frames", 2),
            "video_writer_codec": self.get("video_writer_codec", "mp4v"),
            "show": self.get("show", True),
            "draw": self.get("draw", True),
            "save_video": self.get("save_video", False),
            "save_video_path": self.get("save_video_path", ""),
            "video_duration": self.get("video_duration", 300),
            "max_video_files": self.get("max_video_files", 5),
            "input_width": self.get("input_width", 640),
            "input_height": self.get("input_height", 480),
            "swap_direction": self.get("swap_direction", True)


        }

    def get_log_config(self):
        """Get video processing configuration"""
        return {
            "log_dir": self.get("log_dir", "logs"),
            "console_log": self.get("console_log", False),
            "max_logs": self.get("max_logs", 10)

        }