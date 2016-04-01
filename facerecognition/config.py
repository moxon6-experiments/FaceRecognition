import os

package_dir = os.path.dirname(os.path.realpath(__file__))

face_cascade_path = os.path.join(package_dir, "cascades", "haarcascade_frontalface_alt2.xml")

eye_cascade_path = os.path.join(package_dir, "cascades", "haarcascade_eye.xml")

