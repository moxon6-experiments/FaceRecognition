import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from facerec.preprocessing import TanTriggsPreprocessing
tt = TanTriggsPreprocessing()
from scipy import ndimage
from uniform import get_uniform_value
np.set_printoptions(precision=3)



face_cascade_path = "haarcascade_frontalface_alt2.xml"
eye_cascade_path = "haarcascade_eye.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)


def blockwise_histogram(image):
    image = image-image.min()

    if image.shape[0] != 128 or image.shape[1] != 128:
        raise ProcessingError("Must be 128x128")
    histogram_array = np.zeros((21, 21, 59), dtype=np.uint8)
    guf = np.vectorize(get_uniform_value)
    for i in range(21):
        for j in range(21):
            subarray = image[6*j:6*j+8, 6*i:6*i+8]
            if subarray.shape[0] != 8 and subarray.shape[1] != 8:
                raise ProcessingError("SubArray Must be 8x8")
            lbp = local_binary_pattern(subarray, 8, 1, "default")[1:-1, 1:-1]

            lbp = guf(lbp)

            hist, _ = np.histogram(lbp, density=False, bins=59)
            histogram_array[i][j] = list(hist)
    return histogram_array.reshape(21*21, 59)



from imageprocessing import ProcessingError

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, -angle*180/np.pi, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def divofgauss(face):
    d1 = ndimage.gaussian_filter(face, 1.0)
    d2 = ndimage.gaussian_filter(face, 2.0)

    return d1 - d2



def contrastEq(X):
    alpha = 0.1
    tau = 10

    X = X / np.power(np.mean(np.power(np.abs(X),alpha)), 1.0/alpha)
    X = X / np.power(np.mean(np.power(np.minimum(np.abs(X),tau),alpha)), 1.0/alpha)
    X = tau*np.tanh(X/tau)
    return X

def correct_gamma(blurred):
    blurred = np.array(blurred, dtype=np.float32)
    return np.power(blurred, 0.2)


tantriggs = tt.extract

def cropFace(frame, eye_left, eye_right):

    final_width = 128
    final_height = 128

    offset_pc_x = 0.33
    offset_pc_y = 0.45

    eye_direction = eye_right - eye_left
    eye_distance = np.linalg.norm(eye_direction)

    rotation_angle = -np.arctan2(eye_direction[1], eye_direction[0])
    rotated_image = rotateImage(frame, rotation_angle, tuple(eye_left))
    size = eye_distance/(1-2*offset_pc_x)
    x_position = eye_left[0] - offset_pc_x * size
    y_position = eye_left[1] - offset_pc_y * size

    cropped_rotated = rotated_image[y_position:y_position+size, x_position:x_position+size]

    if cropped_rotated.shape[0]*cropped_rotated.shape[1] <=0:
        raise ProcessingError("Area Must be Positive")

    cropped_rotated = cv2.resize(cropped_rotated, (final_width, final_height))
    return cropped_rotated


def getGrey(frame):
    try:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        raise ProcessingError(str(e))


def get_eye_positions(face, x, y):
    eyes = eye_cascade.detectMultiScale(face)
    eyes = list(eyes)
    if len(eyes) == 2:
        eyes.sort(key=lambda x:x[0])
        eye_positions = []
        for eye in eyes:
            eyex, eyey, eyewidth, eyeheight = eye
            eye_position = [eyex+0.5*eyewidth+x, eyey+0.5*eyeheight+y]
            eye_position = np.array(eye_position).astype(np.int)
            eye_positions.append(eye_position)

        return eye_positions[0], eye_positions[1]
    else:
        raise ProcessingError("Not Exactly Two Eyes")

def getFace(frame):
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        raise ProcessingError("No Faces Detected")
    else:
        x,y,w,h = faces[0]
        face = frame[y:y+h, x:x+w]
        eye_left, eye_right = get_eye_positions(face, x, y)
        fixed_face = cropFace(frame, eye_left, eye_right)


        return fixed_face, faces[0]

def wprint(x):
    print(x)
    input("Press Enter")

def getLBP(tt_face):
    lbp = local_binary_pattern(tt_face, 24, 3, "uniform")
    return lbp


def dense_lbp(frame):
    grey = getGrey(frame)

    face, (x, y, width, height) = getFace(grey)
    blurred = ndimage.gaussian_filter(face, 1.1)
    tt_face = tantriggs(blurred)

    tt_face_resized = cv2.resize(tt_face, (width, height))

    grey[y:y+height:, x:x+width] = tt_face_resized

    vector = blockwise_histogram(tt_face)

    return grey


class FrameException:
    pass


class VideoCamera:
    def __init__(self, id):
        self.cam = cv2.VideoCapture(id)

    def read(self):
        ret, frame = self.cam.read()
        if ret:
            return frame
        else:
            raise FrameException

from imageprocessing import DenseLBP

def main():


    cam = VideoCamera(1)


    while True:
        try:
            frame = cam.read()
        except FrameException:
            continue
        if frame is not None:
            try:
                #output = dense_lbp(frame)
                lbpout = DenseLBP.extract(frame)

                #foutput = tantriggs(getGrey(frame))
                cv2.imshow("Window", frame)
            except ProcessingError as e:
                pass

            #cv2.imshow("Window2", frame)
            ch = cv2.waitKey(10)
            if ch == 27:
                break


if __name__ == "__main__":
    main()