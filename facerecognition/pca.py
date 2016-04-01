import os
import numpy as np


class PCA:
    def __init__(self, index, vectors=None, eigenvectors=None, eigenvalues=None, mean=None):

        if eigenvectors is None or eigenvalues is None or mean is None:
            self.eigenvectors = NotImplemented
            self.eigenvalues = NotImplemented
            self.mean = NotImplemented
            self.compute(vectors)
        else:
            self.eigenvectors = eigenvectors
            self.eigenvalues = eigenvalues
            self.mean = mean

        self.index = index

    def compute(self, vectors):

        mean = vectors.mean(axis=1)

        vectors = (vectors.T - mean).T

        eigenvectors, singular_values, variances = np.linalg.svd(vectors, full_matrices=False)

        # Sort Descending
        indices = np.argsort(-singular_values)
        singular_values, eigenvectors = singular_values[indices].copy(), eigenvectors[:, indices].copy()

        self.eigenvalues = np.power(singular_values, 2) / vectors.shape[1]

        self.mean = mean

        for i in range(len(self.eigenvalues)):
            eigenvectors[:, i] = eigenvectors[:, i]/np.linalg.norm(eigenvectors[:, i])

        self.eigenvectors = eigenvectors

        self.retain_variance(0.95)

    def retain_variance(self, variance):

        eigenvalue_total = sum(self.eigenvalues)
        required = eigenvalue_total * variance

        current_variance = 0

        for i, x in enumerate(self.eigenvalues):
            current_variance += x
            if current_variance >= required:
                self.eigenvalues = self.eigenvalues[:i+1]
                self.eigenvectors = self.eigenvectors[:, :i+1]
                break

    def extract(self, vector):
        return self.subspace_project(vector)

    def subspace_project(self, vector):
        vector = vector - self.mean
        return np.dot(self.eigenvectors.T, vector)

    @property
    def dimension(self):
        return self.eigenvectors.shape[1]

    def save(self, directory):
        subdir = "pca_%s" % self.index
        subdir_path = os.path.join(directory, subdir)

        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

        eigenvector_path = os.path.join(subdir_path, "eigenvectors.npy")
        np.save(eigenvector_path, self.eigenvectors)

        eigenvalues_path = os.path.join(subdir_path, "eigenvalues.npy")
        np.save(eigenvalues_path, self.eigenvalues)

        mean_path = os.path.join(subdir_path, "mean.npy")
        np.save(mean_path, self.mean)

    @staticmethod
    def load(subdir_path):

        index = int(subdir_path.split("pca_")[1])

        eigenvector_path = os.path.join(subdir_path, "eigenvectors.npy")
        eigenvectors = np.load(eigenvector_path)

        eigenvalues_path = os.path.join(subdir_path, "eigenvalues.npy")
        eigenvalues = np.load(eigenvalues_path)

        mean_path = os.path.join(subdir_path, "mean.npy")
        mean = np.load(mean_path)

        return PCA(index, eigenvectors=eigenvectors, eigenvalues=eigenvalues, mean=mean)
