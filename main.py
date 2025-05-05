import cv2 
import time
import numpy as np
from face import Face, FaceFinder 
from utils import VideoCapture

def recoverable(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Recovering: {e}")
    return inner

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

class EyeKeys:

    def __init__(self):
        self.finder = FaceFinder()
        self.face = Face()

    @recoverable
    def process(self,image):

        face_mesh = self.finder.find(image)
        if not face_mesh:
            return None

        self.face.process(image, face_mesh)

        face_landmarks = self.face.getLandmarks()
        l_eye = self.face.getLeftEye()
        r_eye = self.face.getRightEye()
        l_eye_landmarks = l_eye.getLandmarks()
        r_eye_landmarks = r_eye.getLandmarks()
        l_eye_pupil = l_eye.getPupil()
        r_eye_pupil = r_eye.getPupil()
        l_eye_minMax = l_eye.getMinMax()
        r_eye_minMax = r_eye.getMinMax()
        blink = l_eye.getBlink() and r_eye.getBlink()

        # print(face_landmarks,l_eye_landmarks,r_eye_landmarks,l_eye_pupil,r_eye_pupil,blink)

        # Define colors for each group
        color_face = (255, 0, 0)        # Blue
        color_l_eye = (0, 255, 0)       # Green
        color_r_eye = (0, 255, 255)     # Yellow
        color_l_pupil = (0, 0, 255)     # Red
        color_r_pupil = (255, 0, 255)   # Magenta

        radius = 2
        thickness = -1  # Filled circle

        # Draw face landmarks
        frame = image
        # Draw left eye landmarks
        for (x, y) in l_eye_landmarks:
            cv2.circle(frame, (int(x), int(y)), radius, color_l_eye, thickness)

        # Draw right eye landmarks
        for (x, y) in r_eye_landmarks:
            cv2.circle(frame, (int(x), int(y)), radius, color_r_eye, thickness)

        # Draw left pupil
        if len(l_eye_pupil):
            x, y = l_eye_pupil
            cv2.circle(frame, (int(x), int(y)), radius + 1, color_l_pupil, thickness)

        # Draw right pupil
        if len(r_eye_pupil):
            x, y = r_eye_pupil
            cv2.circle(frame, (int(x), int(y)), radius + 1, color_r_pupil, thickness)

        # Draw right pupil
        for (x, y) in r_eye_minMax:
            cv2.circle(frame, (int(x), int(y)), radius + 1, color_r_pupil, thickness)

        # Draw left pupil
        for (x, y) in l_eye_minMax:
            cv2.circle(frame, (int(x), int(y)), radius + 1, color_l_pupil, thickness)

        # Optional: draw blink status
        if blink:
            cv2.putText(frame, "Blink", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("landmarks",frame)
        cv2.waitKey(100)

        ll,lr,lu,ld = l_eye_minMax
        rl,rr,ru,rd = r_eye_minMax

        # todo this does not work
        width4 = distance(ll,lr)/4
        height4 = distance(lu,ld)/4
        print(distance(ll,l_eye_pupil))
        if distance(ll,l_eye_pupil) < width4:
            print("look left")

        if distance(rr,r_eye_pupil) < width4:
            print("look right")

        if distance(lu,l_eye_pupil) < height4:
            print("look up")

        if distance(ld,l_eye_pupil) < height4:
            print("look down")

        # Todo: detect which is closest




if __name__=="__main__":
    ekeys = EyeKeys()

    cap = cv2.VideoCapture(0)
    # Process each frame
    while True:
        ret, frame = cap.read()
        ekeys.process(frame)

