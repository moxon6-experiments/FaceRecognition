import os

import cv2

from .imagereader import ImageReader


class Camera:
    class FrameException(Exception):
        pass


class WebCam(Camera):
    def __init__(self, camera_id):
        self.frame_name = "Camera"
        self.cam = None
        self.camera_id = camera_id
        self.num_images = None

    def read(self):

        # Don't initialise camera until needed
        if not self.cam:
            self.cam = cv2.VideoCapture(self.camera_id)

        ret, frame = self.cam.read()
        if ret:
            return "Camera", frame
        else:
            raise self.FrameException


class FileCamera(Camera):
    def __init__(self, directory):
        self.directory = directory
        self.image_reader = ImageReader(directory)
        self.file_list = self.image_reader.generate_file_list()
        self.num_images = len(self.file_list)

    def read(self):
        try:
            return self.image_reader.read()
        except IndexError:
            self.image_reader = ImageReader(self.directory)
            return self.read()

    def pull(self, name):
        path = os.path.join(self.directory, name)
        frame = cv2.imread(path)
        if frame is None:
            raise self.FrameException
        else:
            return frame
