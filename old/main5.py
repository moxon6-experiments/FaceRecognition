import os

import xmltodict

from old.main4 import *


class Face:
    def __init__(self, xml_path):
        xml_dict = xmltodict.parse(open(xml_path, 'rb'))

        eye_left, eye_right = self.get_eyes(xml_dict)
        if not self.valid:
            return

        face_name = xml_path.replace(".gnd.xml", ".tif").replace("xml", "images")
        self.face_image_path = os.path.join(face_name)

        face = cv2.imread(self.face_image_path)
        #print(self.face_image_path)
        if face is None:
            raise ProcessingError
        try:
            face = getGrey(face)
        except ProcessingError:
            pass


        try:
            self.face = cropFace(face, eye_left, eye_right)
        except:
            self.valid=False
            #print(eye_left, eye_right)
            #print(face_name)


    def get_face(self):
        return self.face

    def save(self):

        output_path = self.face_image_path.replace("images", "processed")
        #print(output_path)
        cv2.imwrite(output_path, self.face)


    def get_eyes(self, xml_dict):

        try:
            face_data = xml_dict['gnd']["subjects"]["subject"]["face"]
            left_eye = list(face_data["left-eye"].values())
            right_eye = list(face_data["right-eye"].values())
            self.valid = True

            left_eye = [int(x) for x in left_eye]
            right_eye = [int(x) for x in right_eye]

            # XML has from left and right from perspective of subject
            left_eye, right_eye = right_eye, left_eye

            return np.array(left_eye), np.array(right_eye)

        except Exception as e:
            self.valid = False
            return None, None


face_dir = "C:/faces"
def folder_iter(directory):
    for i, file in enumerate(os.listdir(directory)):
        if i % 100 ==0:
            print(file)
        full_path = os.path.join(directory, file)
        yield full_path


for xml_path in folder_iter(os.path.join(face_dir, "xml")):
    face = Face(xml_path)
    if face.valid:
        face.save()