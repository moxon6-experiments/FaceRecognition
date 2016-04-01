import numpy as np
from .imagealgorithms import DenseLBP, AlignedImageDetect
from facerecognition.projectionmodel import ProjectionModel
from .imageprocessing import GreyScale
from .imagereader import ImageReader
from .featurevector import FeatureVector


class DataSet:
    def __init__(self, dataset_directory=None, serial_directory=None):
        if serial_directory is not None:
            self.projection_model = ProjectionModel.load(serial_directory)
        elif dataset_directory is not None:
            self.projection_model = self.generate_projection_model(dataset_directory)
        else:
            raise Exception("Must provide input databases")

    def generate_projection_model(self, dataset_directory):
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

    def save(self, serial_directory):
        self.projection_model.save(serial_directory)

    def extract(self, lbp_vector):
        lbp_vector = self.projection_model.extract(lbp_vector)
        lbp_vector = lbp_vector / np.linalg.norm(lbp_vector)
        return FeatureVector(lbp_vector)

    @staticmethod
    def get_lbp(face_img):
        face = GreyScale.extract(face_img)
        lbp_out = DenseLBP.extract(face)
        return lbp_out

    def extract_feature(self, face):
        lbp_vector = self.get_lbp(face)
        feature_vector = self.extract(lbp_vector)
        return feature_vector
