import os
import csv
import datetime


class CSVCountLogger:
    def __init__(self, log_dir="logs", console_log=False):
        self.console_log = console_log
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(log_dir, f"people_count_{timestamp}.csv")
        with open(self.log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Event', 'Track ID', 'Total Entries', 'Total Exits'])
        self.entry_count = 0
        self.exit_count = 0
        self.track_counter = 0
        if self.console_log:
            print(f"Logging to: {self.log_filename}")

    def log_entry(self, total_entries, total_exits, timestamp=None):
        self.track_counter += 1
        self._log_event("Entry", self.track_counter, total_entries, total_exits, timestamp)

    def log_exit(self, total_entries, total_exits, timestamp=None):
        self.track_counter += 1
        self._log_event("Exit", self.track_counter, total_entries, total_exits, timestamp)

    def _log_event(self, event_type, track_id, total_entries, total_exits, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, event_type, track_id, total_entries, total_exits])
        if self.console_log:
            print(f"[{timestamp}] {event_type}: Track ID={track_id}, Total Entries={total_entries}, Total Exits={total_exits}")