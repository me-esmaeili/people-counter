import cv2
import logging
from my_solutions import BaseSolution
from shapely.geometry import LineString, Polygon, Point

# تنظیمات لاگر
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ObjectCounter(BaseSolution):
    def __init__(self, draw=True, swap_direction=False, **kwargs):
        super().__init__(**kwargs)

        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.classwise_counts = {}
        self.region_initialized = False
        self.swap_direction = swap_direction
        self.draw = draw

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]

        self.line_width = 2
        self.track_history = {}
        self.track_line = []

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point

        logger.info("ObjectCounter initialized with draw=%s, swap_direction=%s", draw, swap_direction)

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        if prev_position is None or track_id in self.counted_ids:
            logger.debug("Skipping track_id=%s (prev_position=%s, already_counted=%s)", track_id, prev_position,
                         track_id in self.counted_ids)
            return

        logger.debug("Checking track_id=%s for counting", track_id)

        if len(self.region) == 2:
            line = self.LineString(self.region)
            if line.intersects(self.LineString([prev_position, current_centroid])):
                direction_is_horizontal = abs(self.region[0][0] - self.region[1][0]) < abs(
                    self.region[0][1] - self.region[1][1])
                movement_positive = current_centroid[0] > prev_position[0] if direction_is_horizontal else \
                    current_centroid[1] > prev_position[1]

                logger.debug("Line intersection detected. Horizontal: %s, Movement positive: %s",
                             direction_is_horizontal, movement_positive)

                if (movement_positive and not self.swap_direction) or (not movement_positive and self.swap_direction):
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                    logger.info("Counted IN for %s (track_id=%s)", self.names[cls], track_id)
                else:
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                    logger.info("Counted OUT for %s (track_id=%s)", self.names[cls], track_id)

                self.counted_ids.append(track_id)

        elif len(self.region) > 2:
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                movement_positive = (
                        (region_width < region_height and current_centroid[0] > prev_position[0]) or
                        (region_width >= region_height and current_centroid[1] > prev_position[1])
                )

                logger.debug("Polygon contains point. Width: %d, Height: %d, Movement positive: %s",
                             region_width, region_height, movement_positive)

                if (movement_positive and not self.swap_direction) or (not movement_positive and self.swap_direction):
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                    logger.info("Counted IN for %s (track_id=%s)", self.names[cls], track_id)
                else:
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                    logger.info("Counted OUT for %s (track_id=%s)", self.names[cls], track_id)

                self.counted_ids.append(track_id)

    def store_classwise_counts(self, cls):
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}
            logger.debug("Initialized classwise count for class %s", self.names[cls])

    def store_tracking_history(self, track_id, box):
        centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(centroid)
        logger.debug("Stored centroid %s for track_id %s", centroid, track_id)

    def draw_region(self, im):
        if len(self.region) >= 2:
            logger.debug("Drawing region with %d points", len(self.region))
            for i in range(len(self.region) - 1):
                pt1 = tuple(map(int, self.region[i]))
                pt2 = tuple(map(int, self.region[i + 1]))
                cv2.line(im, pt1, pt2, (104, 0, 123), self.line_width * 2)
            if len(self.region) > 2:
                cv2.line(im, tuple(map(int, self.region[-1])), tuple(map(int, self.region[0])), (104, 0, 123),
                         self.line_width * 2)

    def draw_box(self, im, box, label, color):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_centroid_and_tracks(self, im, track_id, cls):
        track = self.track_history[track_id]
        for i in range(1, len(track)):
            if track[i - 1] is None or track[i] is None:
                continue
            cv2.line(im, tuple(map(int, track[i - 1])), tuple(map(int, track[i])), (255, 255, 255), 2)
        if track:
            cv2.circle(im, tuple(map(int, track[-1])), 4, (0, 255, 255), -1)

    def display_counts(self, im):
        if not self.draw:
            return

        logger.debug("Displaying classwise counts")
        x, y = 10, 30
        for key, value in self.classwise_counts.items():
            if value["IN"] == 0 and value["OUT"] == 0:
                continue
            label = f"{key.capitalize()}:"
            if self.show_in:
                label += f" IN {value['IN']}"
            if self.show_out:
                label += f" OUT {value['OUT']}"
            cv2.putText(im, label.strip(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (104, 31, 17), 2)
            y += 25

    def display_output(self, im):
        logger.debug("Displaying output frame")
        cv2.imshow("Output", im)
        cv2.waitKey(1)

    def count(self, im0):
        logger.debug("Starting count on frame")

        if not self.region_initialized:
            logger.info("Initializing region for first time")
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)

        if self.draw:
            self.draw_region(im0)

        logger.debug("Processing %d objects", len(self.boxes))

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            if self.draw:
                self.draw_box(im0, box, self.names[cls], (0, 255, 0))

            self.store_tracking_history(track_id, box)
            self.store_classwise_counts(cls)

            if self.draw:
                self.draw_centroid_and_tracks(im0, track_id, cls)

            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]

            self.count_objects(current_centroid, track_id, prev_position, cls)

        self.display_counts(im0)
        self.display_output(im0)
        return im0
