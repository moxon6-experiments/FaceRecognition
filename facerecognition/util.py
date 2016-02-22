import os

import cv2

from .imageprocessing import ImageResize


def display(window_title, image, width=None, height=None):
    if image is not None:
        if width is not None or height is not None:
            if width is None:
                width = image.shape[1]
            if height is None:
                height = image.shape[0]
            image = ImageResize.extract(image, width, height)

            cv2.imshow(window_title, image)


def get_key(time=10):
    return cv2.waitKey(time)


def clear():
    os.system('cls')
    print("\n")
