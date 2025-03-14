import os
import csv
import datetime


class CountLogger:
    def __init__(self, log_dir="logs", console_log=False, max_logs=10):
        self.console_log = console_log
        self.log_dir = log_dir
        self.max_logs = max_logs

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(log_dir, f"people_count_{timestamp}.csv")

        with open(self.log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Event', 'Track ID', 'Total Entries', 'Total Exits'])

        self.entry_count = 0
        self.exit_count = 0
        self.track_counter = 0

        # Clean up old log files
        self.cleanup_old_logs()

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
            print(
                f"[{timestamp}] {event_type}: Track ID={track_id}, Total Entries={total_entries}, Total Exits={total_exits}")

    def cleanup_old_logs(self):
        """Retain only the n most recent log files in the log directory."""
        if self.max_logs <= 0:  # Keep all logs if max_logs is 0 or negative
            return

        # Get all log files in the directory
        log_files = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith("people_count_") and filename.endswith(".csv"):
                file_path = os.path.join(self.log_dir, filename)
                file_creation_time = os.path.getctime(file_path)
                log_files.append((file_path, file_creation_time))

        # Sort by creation time (newest first)
        log_files.sort(key=lambda x: x[1], reverse=True)

        # Remove older files beyond max_logs limit
        if len(log_files) > self.max_logs:
            for file_path, _ in log_files[self.max_logs:]:
                try:
                    os.remove(file_path)
                    if self.console_log:
                        print(f"Removed old log file: {file_path}")
                except Exception as e:
                    if self.console_log:
                        print(f"Error removing log file {file_path}: {e}")