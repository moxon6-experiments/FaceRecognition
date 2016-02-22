import numpy as np


class FeatureVector:
    def __init__(self, vector):
        self.vector = vector

    def distance(self, other):

        dist = np.linalg.norm(self.vector - other.vector)
        return (dist*10)**3

    def save(self, path):
        np.save(path, self.vector)

    @staticmethod
    def load(path):
        vector = np.load(path)
        return FeatureVector(vector)

