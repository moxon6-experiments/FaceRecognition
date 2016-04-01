import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from facerecognition.uniform import get_uniform_value
from .config import eye_cascade_path, face_cascade_path
from .imageprocessing import RotateImage, ImageResize, GreyScale, GaussianBlur,\
    GammaCorrect, GaussianDifference, ContrastEq


class ImageAlgorithm:
    class AlgorithmError(Exception):
        pass

AlgorithmError = ImageAlgorithm.AlgorithmError


class EyeDetect(ImageAlgorithm):
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    @classmethod
    def extract(cls, face, x, y):

        eyes = cls.eye_cascade.detectMultiScale(face, scaleFactor=1.2)
        eyes = list(eyes)
        if len(eyes) == 2:
            eyes.sort(key=lambda eye_coord: eye_coord[0])
            eye_positions = []
            for eye in eyes:
                eye_x, eye_y, eye_width, eye_height = eye
                eye_position = [eye_x+0.5*eye_width+x, eye_y+0.5*eye_height+y]
                eye_position = np.array(eye_position).astype(np.int)
                eye_positions.append(eye_position)

            return eye_positions[0], eye_positions[1]
        else:
            raise AlgorithmError("Not Exactly Two Eyes")


class AlignEyes(ImageAlgorithm):

    @staticmethod
    def extract(frame, eye_left, eye_right):

        final_width = 128
        final_height = 128

        offset_pc_x = 0.33
        offset_pc_y = 0.45

        eye_direction = eye_right - eye_left
        eye_distance = np.linalg.norm(eye_direction)

        rotation_angle = -np.arctan2(eye_direction[1], eye_direction[0])
        rotated_image = RotateImage.extract(frame, rotation_angle, tuple(eye_left))
        size = eye_distance/(1-2*offset_pc_x)
        x_position = eye_left[0] - offset_pc_x * size
        y_position = eye_left[1] - offset_pc_y * size

        cropped_rotated = rotated_image[y_position:y_position+size, x_position:x_position+size]

        if cropped_rotated.shape[0]*cropped_rotated.shape[1] <= 0:
            raise AlgorithmError("Area Must be Positive")

        cropped_rotated = ImageResize.extract(cropped_rotated, final_width, final_height)

        return cropped_rotated


class AlignedImageDetect(ImageAlgorithm):

    @staticmethod
    def extract(frame):
        frame = GreyScale.extract(frame)
        face, (x, y, w, h) = FaceDetect.extract(frame)
        eye_left, eye_right = EyeDetect.extract(face, x, y)
        fixed_face = AlignEyes.extract(frame, eye_left, eye_right)

        return fixed_face


class FaceDetect(ImageAlgorithm):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    @classmethod
    def extract(cls, frame):
        faces = cls.face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            raise AlgorithmError("No Faces Detected")
        else:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            return face, (x, y, w, h)


class DenseLBP(ImageAlgorithm):

    @staticmethod
    def extract(aligned_face):
        blurred_face = GaussianBlur.extract(aligned_face)

        masked_face = blurred_face
        tantriggs_face = Tantriggs.extract(masked_face)
        vector = BlockWiseBinaryPatternHistogram.extract(tantriggs_face)
        return vector


class Tantriggs(ImageAlgorithm):

    @staticmethod
    def extract(frame):
        frame = np.array(frame, dtype=np.float32) / 255
        frame = GammaCorrect.extract(frame)
        frame = GaussianDifference.extract(frame*255)
        frame = ContrastEq.extract(frame)
        return frame


class BinaryPatternHistogram(ImageAlgorithm):
    guf = np.vectorize(get_uniform_value)

    @staticmethod
    def extract(subarray):
        lbp = local_binary_pattern(subarray, 8, 1, "default")[1:-1, 1:-1]

        lbp = BinaryPatternHistogram.guf(lbp)

        hist, _ = np.histogram(lbp, density=False, bins=59)
        return hist


class BlockWiseBinaryPatternHistogram(ImageAlgorithm):

    @staticmethod
    def extract(image):

        circle = np.zeros((128, 128), dtype=np.int)
        cv2.ellipse(circle, (64, 64), (48, 64), 0, 0, 360, 1, thickness=-1)

        if image.shape[0] != 128 or image.shape[1] != 128:
            raise AlgorithmError("Must be 128x128")

        total_vectors = 0
        for i in range(21):
            for j in range(21):
                if np.sum(circle[6*j:6*j+8, 6*i:6*i+8]) == 64:
                    total_vectors += 1

        histogram_array = np.zeros((total_vectors, 59), dtype=np.float32)

        index = 0
        for i in range(21):
            for j in range(21):
                if np.sum(circle[6*j:6*j+8, 6*i:6*i+8]) == 64:
                    sub_array = image[6*j:6*j+8, 6*i:6*i+8]
                    if sub_array.shape[0] != 8 or sub_array.shape[1] != 8:
                        raise AlgorithmError("SubArray Must be 8x8")

                    histogram_array[index] = BinaryPatternHistogram.extract(sub_array)
                    index += 1
        if index != total_vectors:
            raise AlgorithmError("Invalid Index/Total Values: %s, %s" % (index, total_vectors))
        return histogram_array
