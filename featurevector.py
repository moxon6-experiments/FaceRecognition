import numpy as np


class FeatureVector:
    def __init__(self, vector):
        self.vector = vector

    def distance(self, other):
        l1_dist = np.linalg.norm(self.vector - other.vector)
        return (l1_dist * 10)**3
