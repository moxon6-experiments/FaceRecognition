import cv2
from imagealgorithms import AlgorithmError, AlignedImageDetect
from camera import VideoCamera, FakeCamera, FrameException
from featurespace import FeatureSpace


class App:
    def __init__(self):

        self.cameras = self.create_cameras()
        self.feature_space = FeatureSpace("smallset")
        self.running = True

    def main(self):
        while self.running:
            self.handle_key(cv2.waitKey(10))

            try:
                frame_name, frame = self.cameras[0].read()
            except FrameException:
                continue
            try:
                face = AlignedImageDetect.extract(frame)
            except AlgorithmError:
                continue
            try:
                items = self.feature_space.nearest(face)
            except AlgorithmError:
                continue

            closest_name = items[0][0]
            closest_distance = items[0][1]

            if closest_distance < 1950:
                print("SUCCESS:",
                      "Frame Name:", frame_name,
                      "Closest Face:", closest_name,
                      "Distance", closest_distance)
            self.show_face(face)

    @staticmethod
    def show_face(face):
        cv2.imshow("Extracted Face", face)

    @staticmethod
    def create_cameras():
        cameras = list()
        cameras.append(VideoCamera(1))
        cameras.append(FakeCamera("impostors"))
        cameras.append(FakeCamera("C:\\faces\\images"))
        cameras.append(FakeCamera("smallset"))
        return cameras

    def handle_key(self, key):
        if key == 27:
            self.running = False
        if key == ord("r"):
            self.cameras = self.cameras[1:] + [self.cameras[0]]

if __name__ == "__main__":
    App().main()
