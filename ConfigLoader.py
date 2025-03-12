import json
import os

class ConfigLoader:

    @staticmethod
    def load_config(config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}, creating default config")
            config = {
                "desktop_model_path": "assets/bestpele.pt",
                "raspi_model_path": "assets/bestPele_ncnn_model-640",
                "video_source": 0,
                "threshold_line_position": 0.5,
                "debug_enabled": True,
                "save_debug_frames": False,
                "debug_dir": "./debug_frames",
                "log_dir": "./logs",
                "max_disappeared": 30,
                "direction_buffer_size": 5
            }

            # Save default config
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            return config