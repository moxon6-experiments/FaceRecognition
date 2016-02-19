import os
import numpy as np
import cv2

from imagealgorithms import ProjectionModel, DenseLBP
from imageprocessing import GreyScale
from featurevector import FeatureVector


class DataSet:
    def __init__(self):
        self.projection_model = NotImplemented

    def save(self, serial_directory):
        self.projection_model.save(serial_directory)

    def generate(self, dataset_directory):
        faces = os.listdir(dataset_directory)
        num_faces = len(faces)

        projection_model = ProjectionModel(num_faces)
        for face_filename in faces:
            lbp = self.get_lbp_from_file(dataset_directory, face_filename)
            projection_model.add_vector(lbp)
        projection_model.generate_models()
        self.projection_model = projection_model

    def load(self, serial_directory):
        self.projection_model = ProjectionModel.load(serial_directory)

    def extract(self, lbp_vector):
        lbp_vector = self.projection_model.extract(lbp_vector)
        lbp_vector = lbp_vector / np.linalg.norm(lbp_vector)
        return FeatureVector(lbp_vector)

    @staticmethod
    def get_lbp_from_file(dataset_directory, face_filename):
        face_path = os.path.join(dataset_directory, face_filename)
        face_img = cv2.imread(face_path)
        lbp_out = DataSet.get_lbp(face_img)

        return lbp_out

    @staticmethod
    def get_lbp(face_img):
        face = GreyScale.extract(face_img)
        lbp_out = DenseLBP.extract(face)
        return lbp_out
