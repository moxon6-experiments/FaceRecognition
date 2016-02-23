from facerecognition.camera import WebCam, FileCamera
from facerecognition.facerecmodel import FaceRecognitionModel
from facerecognition.imagealgorithms import AlignedImageDetect
from facerecognition.util import display, get_key, clear


class FaceException(Exception):
    pass


class FaceRecognitionApp:
    def __init__(self):

        file_camera = FileCamera("data/original/original_selected_subset")
        web_cam = WebCam(1)

        self.cameras = [file_camera, web_cam]

        self.model = FaceRecognitionModel("serialised", num_images=self.cam.num_images, validation=True)
        self.display = False

    def main(self):
        while True:
            try:
                if self.display:
                    key = get_key(1)
                    if key == 27:
                        return
                    if key == ord("r"):
                        self.cameras = self.cameras[1:] + [self.cameras[0]]

                name, frame, face = self.get_face()

                valid = self.model.compare(name, face)

                if self.display:
                    display("Camera Frame", frame)
                    display("Extracted Face", face, width=400, height=400)

            except FaceException:
                continue

    def get_face(self):
        try:
            frame_name, frame = self.cam.read()
        except self.cam.FrameException:
            raise FaceException
        try:
            face = AlignedImageDetect.extract(frame)
        except AlignedImageDetect.AlgorithmError:
            raise FaceException
        return frame_name, frame, face

    @property
    def cam(self):
        return self.cameras[0]
