from collections import defaultdict

import cv2

from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, DEFAULT_CFG_DICT, DEFAULT_SOL_DICT, LOGGER
from ultralytics.utils.checks import check_imshow, check_requirements


class BaseSolution:
    def __init__(self, IS_CLI=False, **kwargs):
        LOGGER.info("Initializing BaseSolution...")

        check_requirements("shapely>=2.0.0")
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep

        self.annotator = None
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.r_s = None

        DEFAULT_SOL_DICT.update(kwargs)
        DEFAULT_CFG_DICT.update(kwargs)
        self.CFG = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}
        LOGGER.debug(f"Configuration: {self.CFG}")

        self.region = self.CFG["region"]
        self.line_width = self.CFG["line_width"] if self.CFG["line_width"] is not None else 2

        if self.CFG["model"] is None:
            self.CFG["model"] = "yolo11n.pt"

        LOGGER.info(f"Loading YOLO model: {self.CFG['model']}")
        self.model = YOLO(self.CFG["model"])
        self.names = self.model.names

        self.track_add_args = {
            k: self.CFG[k]
            for k in ["verbose", "iou", "conf", "device", "max_det", "half", "tracker", "imgsz"]
        }

        if IS_CLI and self.CFG["source"] is None:
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            LOGGER.warning(f"WARNING: source not provided. Using default source {ASSETS_URL}/{d_s}")
            from ultralytics.utils.downloads import safe_download
            safe_download(f"{ASSETS_URL}/{d_s}")
            self.CFG["source"] = d_s

        self.env_check = check_imshow(warn=True)
        self.track_history = defaultdict(list)
        LOGGER.info("BaseSolution initialized.")

    def extract_tracks(self, im0):
        LOGGER.info("Running model.track() to extract tracks")
        self.tracks = self.model.track(
            source=im0,
            persist=True,
            classes=self.CFG["classes"],
            **self.track_add_args
        )

        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
            LOGGER.debug(f"Detected {len(self.track_ids)} tracks")
        else:
            self.boxes, self.clss, self.track_ids = [], [], []
            LOGGER.debug("No tracks detected")

    def store_tracking_history(self, track_id, box):
        LOGGER.debug(f"Storing tracking history for track ID: {track_id}")
        self.track_line = self.track_history.setdefault(track_id, [])
        self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        if len(self.track_line) > 30:
            self.track_line.pop(0)
        LOGGER.debug(f"Tracking history updated for ID {track_id}, total points: {len(self.track_line)}")

    def initialize_region(self):
        LOGGER.debug("Initializing region polygon or line")
        if self.region is None:
            self.region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
            LOGGER.warning("Region not provided. Using default region")
        self.r_s = self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        LOGGER.info(f"Region initialized: {self.r_s}")

    def display_output(self, im0):
        LOGGER.debug("Displaying output frame if show is enabled")
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Ultralytics Solutions", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                LOGGER.info("Display window closed by user")
                return
