from imagealgorithms import AlignedImageDetect
from imageprocessing import ProcessingError
from imagealgorithms import AlgorithmError
from util import VideoCamera
import cv2
import os


class FrameException(Exception):
    pass


def main():
    if not os.path.exists("martin"):
        os.makedirs("martin")
    cam = VideoCamera(1)
    image_aligner = AlignedImageDetect()

    i = 0

    while True:
        try:
            frame = cam.read()
            face = image_aligner.extract(frame)
            cv2.imshow("Extracted Face", face)
            cv2.imwrite("martin/Martin_%s.tif" % i, face)
            i += 1

        except FrameException:
            continue
        except AlgorithmError:
            continue

        ch = cv2.waitKey(10)
        if ch == 27:
            break

if __name__ == "__main__":
    main()
