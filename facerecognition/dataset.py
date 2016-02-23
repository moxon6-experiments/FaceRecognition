import os

import cv2
import numpy as np
from .imagealgorithms import ProjectionModel, DenseLBP, AlignedImageDetect
from .imageprocessing import GreyScale
from .imagereader import ImageReader
from .featurevector import FeatureVector


class DataSet:
    def __init__(self, dataset_directory=None, serial_directory=None):
        if serial_directory is not None:
            self.projection_model = self.load(serial_directory)
        elif dataset_directory is not None:
            self.projection_model = self.generate(dataset_directory)
        else:
            raise Exception("Must provide input data")

    def generate(self, dataset_directory):
        image_reader = ImageReader(dataset_directory)
        num_faces = image_reader.num_files
        projection_model = ProjectionModel(num_faces)
        for name, img in image_reader:
            try:
                face = AlignedImageDetect.extract(img)
                lbp = self.get_lbp(face)
                projection_model.add_vector(lbp)
            except AlignedImageDetect.AlgorithmError:
                continue
        projection_model.generate_models()
        return projection_model

    @staticmethod
    def load(serial_directory):
        return ProjectionModel.load(serial_directory)

    def save(self, serial_directory):
        self.projection_model.save(serial_directory)

    def extract(self, lbp_vector):
        lbp_vector = self.projection_model.extract(lbp_vector)
        lbp_vector = lbp_vector / np.linalg.norm(lbp_vector)
        return FeatureVector(lbp_vector)

    @classmethod
    def get_lbp_from_file(cls, dataset_directory, face_filename):
        face_path = os.path.join(dataset_directory, face_filename)
        face_img = cv2.imread(face_path)
        lbp_out = cls.get_lbp(face_img)

        return lbp_out

    @staticmethod
    def get_lbp(face_img):
        face = GreyScale.extract(face_img)
        lbp_out = DenseLBP.extract(face)
        return lbp_out
