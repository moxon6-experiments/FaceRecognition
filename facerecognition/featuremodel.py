import os

from .dataset import DataSet
from .imagereader import ImageReader
from .resultset import ResultSet

from .featurevector import FeatureVector
from .imagealgorithms import AlignedImageDetect


class FeatureModel:
    def __init__(self, dataset_directory=None, dataset=None):
        self.feature_vector_map = {}

        if dataset is not None:
            self.dataset = dataset

        elif dataset_directory is not None:
            self.dataset = DataSet(dataset_directory=dataset_directory)

        else:
            raise Exception("Must Provide Data Set Information")

    def train_dataset(self, dataset_directory):
        self.feature_vector_map = {}
        image_reader = ImageReader(dataset_directory)

        num_images = image_reader.num_files
        num_processed = 0

        for name, img in image_reader:
            num_processed += 1
            print("Pct Trained", (num_processed*100.0/num_images), "%")
            try:
                face = AlignedImageDetect.extract(img)
                lbp_out = self.dataset.get_lbp(face)

                feature_vector = self.dataset.extract(lbp_out)
                self.feature_vector_map[name] = feature_vector
            except AlignedImageDetect.AlgorithmError:
                continue



    def predict_nearest(self, subject_name, img, num_results=None):
        lbp_vector = self.dataset.get_lbp(img)

        feature_vector = self.dataset.extract(lbp_vector)

        distance_map = {}


        if subject_name[:5] not in [x[:5] for x in self.feature_vector_map.keys()]:
            raise Exception("Not Even There! %s, %s"%(subject_name, self.feature_vector_map.keys()))


        for vector_name, vector in self.feature_vector_map.items():
            dist = feature_vector.distance(vector)
            distance_map[vector_name] = dist

        items = list(distance_map.items())
        items.sort(key=lambda x: x[1])

        return ResultSet(subject_name, items, num_results)

    def save(self, serial_directory):
        self.dataset.save(serial_directory)
        if not os.path.exists(serial_directory):
            os.makedirs(serial_directory)
        for vector_name, vector in self.feature_vector_map.items():
            path = os.path.join(serial_directory, vector_name)
            vector.save(path)

    @classmethod
    def load(cls, serial_directory):
        dataset = DataSet(serial_directory=serial_directory)
        feature_space = FeatureModel(dataset=dataset)
        lbp_paths = [x for x in os.listdir(serial_directory) if ".npy" in x]
        for lbp_path in lbp_paths:
            lbp_name = lbp_path.split(".npy")[0]
            vector = FeatureVector.load(serial_directory+"/"+lbp_path)
            feature_space.feature_vector_map[lbp_name] = vector
        return feature_space
