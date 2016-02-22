from imagealgorithms import AlignedImageDetect
from camera import WebCam
import cv2
import os


def main():
    if not os.path.exists("martin"):
        os.makedirs("martin")
    cam = WebCam(1)
    image_aligner = AlignedImageDetect()

    i = 0

    while True:
        try:
            name, frame = cam.read()
            face = image_aligner.extract(frame)
            cv2.imshow("Extracted Face", face)
            cv2.imwrite("martin/Martin_%s.tif" % i, face)
            i += 1
        except:
            continue

        ch = cv2.waitKey(10)
        if ch == 27:
            break

if __name__ == "__main__":
    main()

