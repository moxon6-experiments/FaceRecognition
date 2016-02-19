from facerec.preprocessing import TanTriggsPreprocessing

import cv2

class FakeCam:
    def __init__(self):
        self.im = cv2.imread("brick.png")
    def read(self):
        return True, self.im

cam = cv2.VideoCapture(0)
#cam = FakeCam()
tt = TanTriggsPreprocessing()


while True:
    ret, frame = cam.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ttframe = tt.extract(frame)
        cv2.imshow("Window", ttframe)
        cv2.waitKey(10)