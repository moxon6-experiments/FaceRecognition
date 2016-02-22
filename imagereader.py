import os
import cv2


class ImageReadException(Exception):
    pass


class ImageFile:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.img = None

    def read(self):
        if self.img is None:
            self.img = cv2.imread(self.path)
        return self.img

    def unload(self):
        del self.img
        self.img = None


class ImageReader:
    def __init__(self, base_dir, file_types=[]):
        self.base_dir = base_dir
        self.file_types = file_types
        self.i = 0

        self.images = self.generate_file_list()
        self.num_files = len(self.images)

    def generate_file_list(self):
        files = []
        for subject in os.listdir(self.base_dir):
            subject_dir = os.path.join(self.base_dir, subject)
            for img in os.listdir(subject_dir):
                full_path = os.path.join(subject_dir, img)
                if len(self.file_types) == 0 or img[-3:] in self.file_types:
                    files.append(ImageFile(img, full_path))
        return files

    def read(self):
        if self.i >= self.num_files:
            raise IndexError("Reset ImageReader")
        subject = self.images[self.i].name
        im = self.images[self.i].read()
        self.i += 1
        if im is None:
            raise ImageReadException
        return subject, im

    def __next__(self):
        if self.i >= self.num_files:
            self.i = 0
            raise StopIteration
        return self.read()

    def __iter__(self):
        return self


