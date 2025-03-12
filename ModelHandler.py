from ultralytics import YOLO

class ModelHandler:
    def __init__(self, model_path='bestPele.pt'):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        try:
            model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")
            print(f"Model classes: {model.names}")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def detect(self, frame, conf=0.4, iou=0.45, persist=True):
        return self.model.track(frame, persist=persist, conf=conf, iou=iou)