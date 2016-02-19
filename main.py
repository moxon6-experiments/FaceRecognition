import cv2
from imageprocessing import ProcessingError, ImageResize, GreyScale, RotateImage
from imagealgorithms import AlgorithmError, DenseLBP, AlignedImageDetect, ProjectionModel
import os
import numpy as np
from util import VideoCamera, FakeCamera, FrameException


from imagealgorithms import ProjectionModel

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


class FeatureVector:
    def __init__(self, vector):
        self.vector = vector

    def distance(self, other):
        l1_dist = np.linalg.norm(self.vector - other.vector)
        return (l1_dist * 10)**3



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


class App:
    def __init__(self):
        pass


def main():
    feature_space = FeatureSpace("smallset")


    cam = VideoCamera(1)
    cam2 = FakeCamera("impostors")

    cam3 = FakeCamera("C:\\faces\\images")
    smallcam = FakeCamera("smallset")
    #image_aligner = AlignedImageDetect()



    while True:
        try:
            frame = cam.read()
            face = AlignedImageDetect.extract(frame)

            items = feature_space.nearest(face)
            #print(items)
            name = items[0][0]
            dist = items[0][1]
            if items[0][1] < 2000:
                print("SUCCESS:", "Frame Name:", cam.frame_name, "Closest Face:", name, "Distance", dist)
            else:
                pass
                #print("FAIL:", "Frame Name:", cam.frame_name, "Closest Face:", name, "Distance", dist)

            cv2.imshow("Extracted Face", face)

        except FrameException:
            continue
        except AlgorithmError:
            continue

        if frame is not None:
            try:
                pass

            except ProcessingError as e:
                pass

            ch = cv2.waitKey(10)
            if ch == 27:
                break
            if ch == ord("r"):
                cam, cam2, cam3 = cam2, cam3, cam


def main3():
    dataset = DataSet()
    dataset.generate("smallset")
    dataset.save("serialised")
    dataset.load("serialised")

if __name__ == "__main__":
    main()
