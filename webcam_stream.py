import cv2
from threading import Thread

class WebcamVideoStream:
    """Manages video stream capture"""
    def __init__(self, src='rtsp://admin:Noasa@123@10.10.200.11:554/main', width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.mp4Type = '.mp4' in src.lower()

    def start(self):
        self.started = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.started = False
        self.thread.join()

    def checkCameraPath(self):
        return self.stream.isOpened()