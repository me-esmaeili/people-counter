import csv
import os

from datetime import datetime


class Logger:
    """Handles logging of events"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_filename = os.path.join(log_dir, f"people_count_{self.timestamp}.csv")

        # Initialize CSV log file
        with open(self.log_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Event", "Track ID", "Total Entries", "Total Exits"])

    def log_event(self, event, track_id, entry_count, exit_count):
        """Log an event to the CSV file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event, track_id, entry_count, exit_count])