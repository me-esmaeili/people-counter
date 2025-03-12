import json
import cv2
import numpy as np
from PeopleCounter import *
def calculate_expanded_roi(zone_tracker):
    all_points = np.vstack((zone_tracker.entry_zone, zone_tracker.exit_zone))
    x_min, y_min = np.min(all_points[:, 0]), np.min(all_points[:, 1])
    x_max, y_max = np.max(all_points[:, 0]), np.max(all_points[:, 1])
    width, height = x_max - x_min, y_max - y_min
    x_expand, y_expand = int(width * 0.3), int(height * 0.3)
    x_min, y_min = max(0, x_min - x_expand), max(0, y_min - y_expand)
    x_max, y_max = min(zone_tracker.width, x_max + x_expand), min(zone_tracker.height, y_max + y_expand)
    return [
        [x_min / zone_tracker.width * 100, y_min / zone_tracker.height * 100],
        [x_max / zone_tracker.width * 100, y_min / zone_tracker.height * 100],
        [x_max / zone_tracker.width * 100, y_max / zone_tracker.height * 100],
        [x_min / zone_tracker.width * 100, y_max / zone_tracker.height * 100]
    ]

if __name__ == "__main__":
    # Load configuration from JSON file
    with open("config.json", "r") as f:
        config = json.load(f)

    # Create the counter with settings from config
    counter = PeopleCounter(
        source=config.get("source", "0"),
        model_path=config.get("model_path", "bestPele.pt"),
        show_live=config.get("show_live", True)
    )

    # Set performance parameters
    if "performance" in config:
        counter.frame_processor.set_performance_params(config["performance"])
        counter.set_batch_params(config["performance"])
        print(f"Performance parameters: {config['performance']}")

    # Configure background subtraction
    if "background_subtraction" in config:
        counter.motion_detector.set_bg_subtraction_params(config["background_subtraction"])
        print(f"Background subtraction: {config['background_subtraction']}")

    # Set zones
    if "zones" in config:
        counter.zone_tracker.set_zones_percent(config["zones"])
        print(f"Zones: {config['zones']}")

    # Set motion ROI
    if "motion_roi" in config and config["motion_roi"].get("calculate_expanded", False):
        motion_roi = calculate_expanded_roi(counter.zone_tracker)
        counter.motion_detector.set_motion_roi(motion_roi)
        print(f"Motion ROI: {motion_roi}")

    # Run the counter
    counter.run()