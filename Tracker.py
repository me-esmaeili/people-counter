class Tracker:
    """Tracks people and manages track data"""

    def __init__(self, config):
        self.max_disappeared = config.get("max_disappeared", 30)
        self.direction_buffer_size = config.get("direction_buffer_size", 5)
        self.max_positions = config.get("max_positions", 30)

        # Tracking data structures
        self.tracks = {}
        self.disappeared_tracks = {}
        self.counted_tracks = set()  # Keep track of IDs that have already been counted

    def update_tracks(self, track_ids, boxes, threshold_y):
        """Update tracks with new detection results"""
        current_track_ids = set(track_ids)

        # Process detected objects
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box

            # Calculate center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Create or update track
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'positions': [center_y],
                    'last_seen': 0,
                    'box': box,
                    'center': (center_x, center_y)
                }
            else:
                self.tracks[track_id]['positions'].append(center_y)
                self.tracks[track_id]['last_seen'] = 0
                self.tracks[track_id]['box'] = box
                self.tracks[track_id]['center'] = (center_x, center_y)

                # Keep only the most recent positions
                if len(self.tracks[track_id]['positions']) > self.max_positions:
                    self.tracks[track_id]['positions'] = self.tracks[track_id]['positions'][-self.max_positions:]

        # Update disappeared tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in current_track_ids:
                self.tracks[track_id]['last_seen'] += 1

                # Move to disappeared tracks if not seen for too long
                if self.tracks[track_id]['last_seen'] > self.max_disappeared:
                    # Store in disappeared tracks
                    self.disappeared_tracks[track_id] = self.tracks[track_id]
                    del self.tracks[track_id]

        # Clean up old disappeared tracks
        for track_id in list(self.disappeared_tracks.keys()):
            self.disappeared_tracks[track_id]['last_seen'] += 1
            if self.disappeared_tracks[track_id]['last_seen'] > self.max_disappeared * 2:
                del self.disappeared_tracks[track_id]

        return current_track_ids

    def count_person(self, track_id, threshold_y):
        """Determine if a person has crossed the threshold line"""
        # Skip if already counted or track doesn't exist
        if track_id in self.counted_tracks or track_id not in self.tracks:
            return None

        positions = self.tracks[track_id]['positions']

        # Need enough positions to determine direction
        if len(positions) < self.direction_buffer_size:
            return None

        # Check if the track crossed the line
        above_line = [pos < threshold_y for pos in positions]
        if not (True in above_line and False in above_line):
            return None

        # Calculate direction using recent positions
        recent_positions = positions[-self.direction_buffer_size:]
        first_half = recent_positions[:len(recent_positions) // 2]
        second_half = recent_positions[len(recent_positions) // 2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        # Determine direction and count
        if avg_first > threshold_y > avg_second:  # Moving from bottom to top
            self.counted_tracks.add(track_id)
            return "Entry"
        elif avg_first < threshold_y < avg_second:  # Moving from top to bottom
            self.counted_tracks.add(track_id)
            return "Exit"

        return None