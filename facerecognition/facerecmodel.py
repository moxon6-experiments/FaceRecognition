from .featuremodel import FeatureModel
from .validation import ROCCalculation


class FaceRecognitionModel:
    def __init__(self, feature_model_directory, num_images=None, validation=False):
        self.feature_space = FeatureModel.load(feature_model_directory)

        if validation and num_images is not None:
            self.num_images = num_images
            self.roc = ROCCalculation()
        else:
            self.num_images = None
            self.roc = None
        self.count = 0
        self.threshold = 92.3

    def compare(self, user, face):
        """
        Returns:
            bool face_validation
        """

        result_set = self.feature_space.predict_nearest(user, face, 10)

        if self.roc is not None:
            self.roc.add_result(user, result_set)
            self.count += 1
            if self.count == self.num_images:
                self.roc.save_data()
        result_set.print()

        t1 = user[:5].lower() == result_set.best_result.name[:5].lower()
        t2 = result_set.best_result.distance < self.threshold
        return t1 and t2