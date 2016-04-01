from facerecognition.facerecognitionmodel import FaceRecognitionModel
import os


class FaceRecognitionServer:
    def __init__(self, serialised_directory):
        self.model = FaceRecognitionModel.load(serialised_directory)
        self.serialised_directory = serialised_directory

    def get_user_profile(self, user):
        print(self.model.serial_directory)
        return self.model.get_user_profile(subject_name=user.face_id)

    def register_face(self, user, face):
        user_path = self.serialised_directory + "/users/" + user.face_id
        if not os.path.exists(user_path):
            os.makedirs(user_path)
        user_profile = self.get_user_profile(user)
        user_profile.register(face)






