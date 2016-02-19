from dataset import DataSet
import os


class FeatureSpace:
    def __init__(self, dataset_directory):
        dataset = DataSet()
        dataset.generate(dataset_directory)
        self.dataset = dataset

        self.feature_vector_map = {}

        for img in os.listdir(dataset_directory):
            lbp_out = dataset.get_lbp_from_file(dataset_directory, img)
            feature_vector = dataset.extract(lbp_out)
            self.feature_vector_map[img] = feature_vector

    def nearest(self, img):
        lbp_vector = self.dataset.get_lbp(img)

        vector = self.dataset.extract(lbp_vector)

        distance_map = {}

        for vector_name in self.feature_vector_map:
            dist = vector.distance(self.feature_vector_map[vector_name])

            distance_map[vector_name] = dist

        items = list(distance_map.items())
        items.sort(key=lambda x: x[1])
        return items[0:3]