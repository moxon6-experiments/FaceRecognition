import cv2
import numpy as np
from scipy.ndimage import gaussian_filter as gaussian, rotate


class ProcessingError(Exception):
    pass


class ImageProcessing:
    pass


class FaceMask:

    @staticmethod
    def extract(frame):
        circle = np.zeros_like(frame)
        cv2.ellipse(circle, (64, 64), (48, 64), 0, 0, 360, (255, 255, 255), thickness=-1)
        zeroed_frame = np.multiply(frame, circle)
        circle[circle == 0] = 127
        circle[circle == 255] = 0
        zeroed_frame += circle
        return zeroed_frame


class ImageResize(ImageProcessing):

    @staticmethod
    def extract(frame, width, height):
        return cv2.resize(frame, (width, height))


class RotateImage(ImageProcessing):

    @staticmethod
    def extract(img, angle=0, pivot=(0, 0)):
        padX = [img.shape[1] - pivot[0], pivot[0]]
        padY = [img.shape[0] - pivot[1], pivot[1]]
        imgP = np.pad(img, [padY, padX], 'constant')
        imgR = rotate(imgP, -angle*180/np.pi, reshape=False)
        return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

class ContrastEq(ImageProcessing):

    @staticmethod
    def extract(frame):
        alpha = 0.1
        tau = 10
        frame = frame / np.power(np.mean(np.power(np.abs(frame), alpha)), 1.0 / alpha)
        frame = frame / np.power(np.mean(np.power(np.minimum(np.abs(frame), tau), alpha)), 1.0 / alpha)
        frame = tau * np.tanh(frame / tau)
        return frame


class GaussianBlur(ImageProcessing):

    @staticmethod
    def extract(frame, sigma=1.1):
        return gaussian(frame, sigma)


class GaussianDifference(ImageProcessing):

    @staticmethod
    def extract(frame):
        _sigma0 = 1.0
        _sigma1 = 2.0
        frame = np.asarray(GaussianBlur.extract(frame, _sigma1) - GaussianBlur.extract(frame, _sigma0))
        return frame


class GammaCorrect(ImageProcessing):

    @staticmethod
    def extract(frame):
        gamma = 0.2
        frame = np.power(frame, gamma)
        return frame


class GreyScale(ImageProcessing):

    @staticmethod
    def extract(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            return frame
        else:
            raise ProcessingError


