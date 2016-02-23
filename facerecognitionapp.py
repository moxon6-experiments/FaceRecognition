from facerecognition.camera import WebCam, FileCamera
from facerecognition.facerecmodel import FaceRecognitionModel
from facerecognition.imagealgorithms import AlignedImageDetect
from facerecognition.util import display, get_key


class FaceException(Exception):
    pass


class FaceRecServerStub:
    def __init__(self):
        pass


class FaceRecognitionApp:
    def __init__(self, camera_id, name):

        file_camera = FileCamera("data/original/original")
        web_cam = WebCam(camera_id)
        self.cameras = [file_camera, web_cam]
        self.name = name

        self.model = FaceRecognitionModel("full_feret_serialised",
                                          num_images=file_camera.num_images,
                                          validation=False)
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
