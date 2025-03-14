import time
import threading
from PeopleCounter import PeopleCounter


def run_video(video_controller, video_source, duration):
    """
    Starts the video processing and stops it after the specified duration.
    """
    # Start video processing in a separate thread if start() is blocking.
    processing_thread = threading.Thread(target=video_controller.start, args=(video_source,))
    processing_thread.start()

    # Allow the video to run for 'duration' seconds.
    time.sleep(duration)

    # Stop the video processing.
    video_controller.stop()  # Assumes a stop() method exists to break the loop.

    # Wait for the thread to finish.
    processing_thread.join()


def main1():
    config_file = "config.json"
    video_controller = PeopleCounter(config_file)

    # Process the first video for 5 seconds.
    video_source1 = "assets/samples/20231207153936_839_2.avi"
    run_video(video_controller, video_source1, duration=15)

    print("After first video processing (5 seconds):")
    print("Entry Count:", video_controller.get_entry_count())
    print("Exit Count:", video_controller.get_exit_count())

    # (Optional) Reset counts if needed before processing the next video.
    # For example:
    # video_controller.reset_counts()  # If such a method exists.

    # # Process the second video for 10 seconds.
    # video_source2 = "assets/samples/20231205155546_456.avi"
    # run_video(video_controller, video_source2, duration=10)
    #
    # print("After second video processing (10 seconds):")
    # print("Entry Count:", video_controller.get_entry_count())
    # print("Exit Count:", video_controller.get_exit_count())

def main2():
    config_file = "config.json"
    video_controller = PeopleCounter(config_file)

    # Process the first video for 5 seconds.
    video_source = "assets/samples/20231205155546_456.avi"
    video_controller.start(video_source)
if __name__ == "__main__":
    # main1()
    main2()
