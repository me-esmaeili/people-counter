import numpy as np
import cv2

class ZoneTracker:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.entry_zone_percent = [[20, 40], [60, 40], [60, 50], [20, 50]]
        self.exit_zone_percent = [[20, 60], [60, 60], [60, 70], [20, 70]]
        self.entry_zone = self.percent_to_pixel(self.entry_zone_percent)
        self.exit_zone = self.percent_to_pixel(self.exit_zone_percent)
        self.people_in_entry = set()
        self.people_in_exit = set()
        self.count_in = 0
        self.count_out = 0

    def percent_to_pixel(self, percent_coords):
        if percent_coords is None:
            return None
        return np.array([[int((x / 100) * self.width), int((y / 100) * self.height)]
                         for x, y in percent_coords], dtype=np.int32)

    def track_person(self, cx, cy, track_id):
        """Track a person's movement through zones"""
        # Check if person is in entry zone
        in_entry = cv2.pointPolygonTest(self.entry_zone, (cx, cy), False) >= 0
        in_exit = cv2.pointPolygonTest(self.exit_zone, (cx, cy), False) >= 0

        # Handle entry zone
        if in_entry:
            if track_id not in self.people_in_entry:
                self.people_in_entry.add(track_id)
                print(f"Person {track_id} entered entry zone")

        # Handle exit zone
        if in_exit:
            if track_id not in self.people_in_exit:
                self.people_in_exit.add(track_id)
                print(f"Person {track_id} entered exit zone")

            # If the person was previously in the entry zone, count as IN
            if track_id in self.people_in_entry:
                self.people_in_entry.remove(track_id)
                self.count_in += 1
                print(f"Person {track_id} counted as IN")

        # Handle transitions from exit to entry (count as OUT)
        if track_id in self.people_in_exit and in_entry:
            self.people_in_exit.remove(track_id)
            self.count_out += 1
            print(f"Person {track_id} counted as OUT")
    def set_zones_percent(self, entry_zone_percent=None, exit_zone_percent=None):
        if entry_zone_percent:
            self.entry_zone_percent = entry_zone_percent
            self.entry_zone = self.percent_to_pixel(entry_zone_percent)
        if exit_zone_percent:
            self.exit_zone_percent = exit_zone_percent
            self.exit_zone = self.percent_to_pixel(exit_zone_percent)
        print("Tracking zones updated")

    def set_zones_percent(self, params):
        if "entry_zone_percent" in params:
            self.entry_zone_percent = params["entry_zone_percent"]
            self.entry_zone = self.percent_to_pixel(self.entry_zone_percent)
        if "exit_zone_percent" in params:
            self.exit_zone_percent = params["exit_zone_percent"]
            self.exit_zone = self.percent_to_pixel(self.exit_zone_percent)
        print("Tracking zones updated")