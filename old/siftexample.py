from facerec.preprocessing import TanTriggsPreprocessing

import cv2
import numpy as np

class FakeCam:
    def __init__(self):
        self.im = cv2.imread("brick.png")
    def read(self):
        return True, self.im

cam = cv2.VideoCapture(0)
#cam = FakeCam()
tt = TanTriggsPreprocessing()


sift = cv2.xfeatures2d.SIFT_create()


while True:
    ret, frame = cam.read()
    if ret:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp = sift.detect(grey,None)
        img = cv2.drawKeypoints(grey, kp, None)

        cv2.imshow("Window", img)
        cv2.waitKey(10)