from facerecognition.camera import WebCam
from facerecognition.imagealgorithms import AlignedImageDetect
from viewstub import ViewStub
from facerecognition.util import display, get_key


class FaceException(Exception):
    pass


class FaceRecognitionClient:
    def __init__(self, camera):
        self.camera = camera
        self.view_stub = ViewStub()

    def get_face(self):
        try:
            _, frame = self.camera.read()
            face = AlignedImageDetect.extract(frame)
            return frame, face

        except self.camera.FrameException:
            raise FaceException

        except AlignedImageDetect.AlgorithmError:
            raise FaceException

    def verify(self):
        frame, face = self.get_face()
        display("Frame", frame)
        display("Face", face)
        get_key(1)
        self.view_stub.verify(face)

    def register(self):
        frame, face = self.get_face()
        display("Frame", frame)
        display("Face", face)
        ch = get_key(1)
        if ch == ord("r"):
            print("USER MESSAGE: Face Registered")
            self.view_stub.register(face)


def main():
    webcam = WebCam(1)
    frc = FaceRecognitionClient(webcam)
    while True:
        try:
            frc.verify()
            #frc.register()
        except FaceException:
            pass

if __name__ == "__main__":
    main()
