import cv2
import os
class Visualizer:
    """Handles visualization of detection results"""

    def __init__(self, config):
        self.debug_enabled = config.get("debug_enabled", False)
        self.save_debug_frames = config.get("save_debug_frames", False)
        self.debug_dir = config.get("debug_dir", "./debug_frames")

        # Create debug directory if it doesn't exist and debug is enabled
        if self.save_debug_frames and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        self.frame_count = 0

    def draw_frame(self, frame, tracks, counted_tracks, threshold_y, frame_width, entry_count, exit_count, fps):
        """Draw visualization elements on the frame"""
        if not self.debug_enabled:
            return None

        # Draw threshold line
        cv2.line(frame, (0, threshold_y), (frame_width, threshold_y), (0, 255, 0), 2)

        # Draw tracks
        for track_id, track_data in tracks.items():
            box = track_data['box']
            center = track_data['center']
            x1, y1, x2, y2 = box

            # Determine color based on counting status
            color = (0, 0, 255)  # Default color (red)
            if track_id in counted_tracks:
                directions = track_data['positions']
                if len(directions) >= 2:
                    if directions[0] > directions[-1]:  # Moving up (entry)
                        color = (0, 255, 0)  # Green for entries
                    else:
                        color = (255, 0, 0)  # Blue for exits

            # Draw bounding box and ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, center, 5, color, -1)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display counts and FPS
        cv2.putText(frame, f"Entries: {entry_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {exit_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save debug frame if enabled
        if self.save_debug_frames and self.frame_count % 30 == 0:  # Save every 30 frames
            frame_path = os.path.join(self.debug_dir, f"frame_{self.frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

        self.frame_count += 1
        return frame