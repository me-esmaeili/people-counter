
class PeopleTracker:
    """Handles tracking and counting logic for people crossing a threshold"""

    def __init__(self, frame_height, config):
        self.threshold_y = int(frame_height * config["threshold_line_position"])
        self.max_disappeared = config["max_disappeared"]
        self.direction_buffer_size = config["direction_buffer_size"]

        # Tracking state
        self.entry_count = 0
        self.exit_count = 0
        self.tracks = {}
        self.disappeared_tracks = {}
        self.counted_tracks = set()

    def _determine_direction(self, positions):
        """Determine crossing direction based on position history"""
        if len(positions) < self.direction_buffer_size:
            return None

        recent_positions = positions[-self.direction_buffer_size:]
        first_half = recent_positions[:len(recent_positions) // 2]
        second_half = recent_positions[len(recent_positions) // 2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_first > self.threshold_y > avg_second:
            return "entry"
        elif avg_first < self.threshold_y < avg_second:
            return "exit"
        return None

    def update_track(self, track_id, center_y):
        """Update or create a track with new position"""
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'positions': [center_y],
                'last_seen': 0
            }
        else:
            self.tracks[track_id]['positions'].append(center_y)
            self.tracks[track_id]['last_seen'] = 0
            # Limit position history
            self.tracks[track_id]['positions'] = self.tracks[track_id]['positions'][-30:]

    def count_person(self, track_id, logger):
        """Count a person if they cross the threshold"""
        if track_id in self.counted_tracks or track_id not in self.tracks:
            return False

        positions = self.tracks[track_id]['positions']
        direction = self._determine_direction(positions)

        if direction == "entry":
            self.entry_count += 1
            print(f"Entry detected (ID: {track_id}), Total Entries: {self.entry_count}")
            logger.log_event("Entry", track_id, self.entry_count, self.exit_count)
            self.counted_tracks.add(track_id)
            return True
        elif direction == "exit":
            self.exit_count += 1
            print(f"Exit detected (ID: {track_id}), Total Exits: {self.exit_count}")
            logger.log_event("Exit", track_id, self.entry_count, self.exit_count)
            self.counted_tracks.add(track_id)
            return True
        return False

    def update_disappeared(self):
        """Manage tracks that are no longer detected"""
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['last_seen'] += 1
            if self.tracks[track_id]['last_seen'] > self.max_disappeared:
                self.disappeared_tracks[track_id] = self.tracks[track_id]
                del self.tracks[track_id]

        # Clean up old disappeared tracks
        for track_id in list(self.disappeared_tracks.keys()):
            self.disappeared_tracks[track_id]['last_seen'] += 1
            if self.disappeared_tracks[track_id]['last_seen'] > self.max_disappeared * 2:
                del self.disappeared_tracks[track_id]
