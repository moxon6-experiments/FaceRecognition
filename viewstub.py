from server import FaceRecognitionServer

global_face_recognition_server = FaceRecognitionServer("databases/serialised/feret")

"""
Dummy Django View
"""


class UserStub:
    def __init__(self):
        self.face_id = "martin3"
        self.correct = 0
        self.incorrect = 0
        self.face_accepted = False


class ViewStub:
    def __init__(self):
        #Session.user
        self.user = UserStub()

    def verify(self, face):

        self.user_profile = global_face_recognition_server.get_user_profile(self.user)

        valid = self.user_profile.verify(face)

        if valid:
            #session.correct
            self.user.correct += 1
            print("Valid")
        else:
            #session.correct
            self.user.incorrect += 1
            print("Invalid")
        if self.user.correct + self.user.incorrect > 10:
            if self.user.correct > self.user.incorrect:
                print("FACE ACCEPTED")
                self.user.face_accepted = True
            else:
                print("FACE REJECTED!")
                self.user.face_accepted = False

    def register(self, face):
        global_face_recognition_server.register_face(self.user, face)
