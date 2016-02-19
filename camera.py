import cv2
import os


class FrameException(Exception):
    pass


class VideoCamera:
    def __init__(self, camera_id):
        self.frame_name = "Camera"
        self.cam = cv2.VideoCapture(camera_id)

    def read(self):
        ret, frame = self.cam.read()
        if ret:
            return "Camera", frame
        else:
            raise FrameException


class FakeCamera:
    def __init__(self, directory):
        self.directory = directory
        self.full_paths = [os.path.join(directory, x) for x in os.listdir(directory)]

        self.i = 0

    def read(self):

        self.i += 1
        self.i %= len(self.full_paths)
        frame_name = self.full_paths[self.i].split("\\")[-1]
        return frame_name, cv2.imread(self.full_paths[self.i])

    def pull(self, name):
        path = os.path.join(self.directory, name)
        frame = cv2.imread(path)
        if frame is None:
            raise FrameException
        else:
            return frame
