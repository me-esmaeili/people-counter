import os
import json

class ConfigLoader:
    """Handles loading configuration files"""
    @staticmethod
    def load_json(file_name):
        code_path = os.path.dirname(os.path.realpath(__file__))
        source = os.path.join(code_path, file_name)
        with open(source, 'r') as file:
            return json.load(file)

    @staticmethod
    def get_app_settings(file_name="AppSettings_peopleCounter.json"):
        return ConfigLoader.load_json(file_name).values()

    @staticmethod
    def get_camera_ips(file_name="cameraIPs.json"):
        return ConfigLoader.load_json(file_name)