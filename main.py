import os
from config_loader import ConfigLoader
from people_counter import PeopleCounter
from threading import Thread


def run_counter(counter):
    """Helper function to run a PeopleCounter instance"""
    counter.run()


def main():
    main_path = os.path.dirname(os.path.abspath(__file__))

    # Load configurations
    camera_ips = ConfigLoader.get_camera_ips()
    settings_values = ConfigLoader.get_app_settings()
    settings = dict(zip(['res', 'set_confidence', 'rtsp', 'MotionSen', 'ShowLive', 'delay',
                         'fromCenterMargineJSON', 'distance_thresholdJSON', 'trackerResetValueJSON',
                         'CameraId', 'down2topJSON', 'saveInterval', 'webPath'], settings_values))

    # List to keep track of threads
    threads = []

    # Iterate through each camera in cameraIPs_dictionary
    for camera in camera_ips.values():
        RTSP_URL, idPoints, Borders, Rotation = camera

        # Create a PeopleCounter instance for each camera
        counter = PeopleCounter(
            source=RTSP_URL,
            id_points=idPoints,
            border=Borders,
            rotation=Rotation,
            settings=settings
        )

        # Start each counter in a separate thread
        thread = Thread(target=run_counter, args=(counter,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete (optional, remove if you want the program to run indefinitely)
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()