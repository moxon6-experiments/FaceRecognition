import os

from .dataset import DataSet
from .imagereader import ImageReader
from .resultset import ResultSet

from .featurevector import FeatureVector
from .imagealgorithms import AlignedImageDetect


class FaceRecognitionModel:
    def __init__(self, dataset_directory=None, dataset=None, serial_directory=None):
        self.threshold = 92.3

        self.feature_vector_map = {}
        if dataset is not None:
            self.dataset = dataset
        elif dataset_directory is not None:
            self.dataset = DataSet(dataset_directory=dataset_directory)
        else:
            raise Exception("No Dataset Provided")

        self.serial_directory = serial_directory

    def train_dataset(self, dataset_directory):
        self.feature_vector_map = {}
        image_reader = ImageReader(dataset_directory)
        for name, img in image_reader:
            try:
                face = AlignedImageDetect.extract(img)
                feature_vector = self.dataset.extract_feature(face)
                self.feature_vector_map[name] = feature_vector
            except AlignedImageDetect.AlgorithmError:
                continue

    def get_user_profile(self, subject_name):
        return UserProfile(subject_name, self.serial_directory, self.threshold, self.dataset.extract_feature)

    def predict_nearest(self, subject_name, img, num_results=None):
        feature_vector = self.dataset.extract_feature(img)
        distance_map = {}
        if subject_name[:5] not in [x[:5] for x in self.feature_vector_map.keys()]:
            raise Exception("Not Even There! %s, %s" % (subject_name, self.feature_vector_map.keys()))

        for vector_name, vector in self.feature_vector_map.items():
            dist = feature_vector.distance(vector)
            distance_map[vector_name] = dist

        items = list(distance_map.items())
        items.sort(key=lambda x: x[1])

        return ResultSet(subject_name, items, num_results)

    @classmethod
    def load(cls, serial_directory):
        dataset = DataSet(serial_directory=serial_directory)
        feature_space = cls(dataset=dataset, serial_directory=serial_directory)
        lbp_paths = [x for x in os.listdir(serial_directory) if ".npy" in x]
        for lbp_path in lbp_paths:
            lbp_name = lbp_path.split(".npy")[0]
            feature_vector = FeatureVector.load(serial_directory+"/"+lbp_path)
            feature_space.feature_vector_map[lbp_name] = feature_vector
        return feature_space

    def save(self):
        self.dataset.save(self.serial_directory)
        self.save_trained(self.serial_directory)

    def save_trained(self, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for vector_name, vector in self.feature_vector_map.items():
            path = os.path.join(output_directory, vector_name)
            vector.save(path)


class UserProfile:
    def __init__(self, subject_name, serial_directory, threshold, extract_feature):
        self.features = self.load_user_features(serial_directory, subject_name)
        self.i = len(self.features) + 1
        self.threshold = threshold
        self.extract_feature = extract_feature

        self.serial_directory = serial_directory
        self.subject_name = subject_name

    def verify(self, face):
        query_vector = self.extract_feature(face)

        accepted = False

        for fv in self.features:
            vector_distance = query_vector.distance(fv)

            distance_string = "{:.4f}".format(vector_distance)

            if vector_distance < self.threshold:
                print("Threshold:", self.threshold, "Distance:", distance_string, "User:", self.subject_name)
                accepted = True
            else:
                print("Threshold:", self.threshold, "Distance:", distance_string, "User:", self.subject_name)
        return accepted

    def register(self, face):
        output_vector = self.extract_feature(face)
        full_path = self.serial_directory + "/users/" + self.subject_name + "/" + self.subject_name + "_%s" % self.i
        self.i += 1
        output_vector.save(full_path)

    @staticmethod
    def load_user_features(serial_directory, subject_name):
        file_names = [x for x in os.listdir(serial_directory + "/users/" + subject_name) if "npy" in x]
        feature_vectors = []
        for file_name in file_names:
            fv = FeatureVector.load(serial_directory + "/users/" + subject_name + "/" + file_name)
            feature_vectors.append(fv)
        return feature_vectors
