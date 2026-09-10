from ultralytics import YOLO


class BagDetector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, frame):
        results = self.model(frame, classes=[24], conf=0.40)
        return results[0].plot()