from imagealgorithms import AlignedImageDetect, AlgorithmError
from camera import WebCam, FileCamera
from featuremodel import FeatureModel
from util import display, get_key, clear


class FaceRecognitionApp:
    def __init__(self):

        self.cameras = self.create_cameras()
        self.feature_space = FeatureModel.load("serialised")

        self.running = True
        self.face = None
        self.frame = None

        self.correct = 0
        self.false = 0

    def main(self):
        while self.running:
            self.handle_key(get_key(10))

            result_set = self.get_result_set()
            display("Camera Frame", self.frame)
            display("Extracted Face", self.face, width=400, height=400)
            if result_set is None:
                continue

            closest_distance = result_set.best_result.distance

            if result_set.match:
                self.correct += 1
            else:
                self.false += 1

            print(
                "Correct:", self.correct, "\n"
                "False:", self.false, "\n"
                "Percentage:", self.correct*100.0/(self.correct+self.false), "%"
            )

            if closest_distance < 2100 or True:
                clear()
                result_set.print()

    def get_result_set(self):
        try:
            frame_name, self.frame = self.cam.read()
        except self.cam.FrameException:
            return None
        try:
            self.face = AlignedImageDetect.extract(self.frame)
            result_set = self.feature_space.get_nearest(frame_name, self.face, 10)
            return result_set
        except AlgorithmError:
            return None

    @property
    def cam(self):
        return self.cameras[0]

    @staticmethod
    def create_cameras():
        cameras = list()
        cameras.append(FileCamera("data/original/original_selected_subset"))
        cameras.append(WebCam(1))

        cameras.append(FileCamera("other/impostors"))

        return cameras

    def handle_key(self, key):
        if key == 27:
            self.running = False
        if key == ord("r"):
            self.cameras = self.cameras[1:] + [self.cameras[0]]

if __name__ == "__main__":
    FaceRecognitionApp().main()
