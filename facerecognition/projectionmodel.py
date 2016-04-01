import os
import numpy as np
from facerecognition.imagealgorithms import ImageAlgorithm
from facerecognition.pca import PCA


class ProjectionModel(ImageAlgorithm):
    def __init__(self, num_images=None, pca_models=None):

        if pca_models is None:
            self.pca_models = []
            self.index = 0
            self.num_images = num_images


            #TODO
            self.pca_cube = np.zeros((231, 59, num_images), dtype=np.float32)

        else:
            self.pca_models = pca_models
            self.index = self.num_images = pca_models[0].eigenvectors.shape[1]
            self.pca_cube = NotImplemented

    def add_vector(self, vector):
        self.pca_cube[:, :, self.index] = vector
        self.index += 1

    def generate_models(self):
        for i in range(231):
            pca_model = PCA(i, self.pca_cube[i])
            self.pca_models.append(pca_model)

    def extract(self, vector):
        total_dim = sum([x.dimension for x in self.pca_models])
        output_vector = np.zeros(total_dim, dtype=np.float32)

        start_index = 0

        for i, pca_model in enumerate(self.pca_models):

            subvector = pca_model.extract(vector[i])
            subvector_dimension = pca_model.dimension

            output_vector[start_index:start_index+subvector_dimension] = subvector
            start_index += pca_model.dimension

        return output_vector

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for pca_model in self.pca_models:
            pca_model.save(directory)

    @staticmethod
    def load(directory):
        pca_dirs = [x for x in os.listdir(directory) if "pca" in x]
        pca_models = [None] * len(pca_dirs)
        for pca_dir in pca_dirs:

            pca_index = int(pca_dir.split("_")[1])

            full_dir_path = os.path.join(directory, pca_dir)
            pca_model = PCA.load(full_dir_path)
            pca_models[pca_index] = pca_model

        return ProjectionModel(pca_models=pca_models)