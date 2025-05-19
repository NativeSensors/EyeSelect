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

def compute_std(buffer):
    if len(buffer) < 2:
        return (0.0, 0.0)  # Not enough data
    arr = np.array(buffer)
    std_x = np.std(arr[:, 0])
    std_y = np.std(arr[:, 1])
    return std_x, std_y

l_buffer = []
r_buffer = []
class EyeKeys:

    def __init__(self):
        self.finder = FaceFinder()
        self.face = Face()

    # @recoverable
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

        lx, ly = l_eye_pupil
        lx = (lx-lr[0])/(ll[0]-lr[0])
        ly = (ly-ld[1])/(lu[1]-ld[1])

        rx, ry = r_eye_pupil
        rx = (rx-rr[0])/(rl[0]-rr[0])
        ry = (ry-rd[1])/(ru[1]-rd[1])

        # Window size
        width, height = 1920, 1080

        # Dot position and size
        l_dot_position = (lx*width, ly*height)  # (x, y)
        r_dot_position = (rx*width, ry*height)  # (x, y)
        dot_radius = 5
        dot_color = (0, 0, 255)  # Red in BGR
        background_color = (255, 255, 255)  # White

        # Create a white background image
        image = np.full((height, width, 3), background_color, dtype=np.uint8)
        # Ensure coordinates are integers
        if l_dot_position is not None and r_dot_position is not None:
            l_dot_position = (int(l_dot_position[0]), int(l_dot_position[1]))
            r_dot_position = (int(r_dot_position[0]), int(r_dot_position[1]))
            l_buffer.append(l_dot_position)
            r_buffer.append(r_dot_position)

            for n in range(len(r_buffer)):
                print(l_dot_position,l_eye_pupil)
                print(r_dot_position,r_eye_pupil)
                cv2.circle(image, l_buffer[n], dot_radius, (0, 0, 255), -1)
                cv2.circle(image, r_buffer[n], dot_radius, (255, 0, 0), -1)

            if len(r_buffer) > 20:
                r_buffer.pop(0)
                l_buffer.pop(0)

            l_std_x, l_std_y = compute_std(l_buffer)
            r_std_x, r_std_y = compute_std(r_buffer)

            print(f"Left eye std: X={l_std_x:.2f}, Y={l_std_y:.2f} x={int(l_dot_position[0]) - width/2} y={int(l_dot_position[1]) - height/2}")
            print(f"Right eye std: X={r_std_x:.2f}, Y={r_std_y:.2f} x={int(r_dot_position[0]) - width/2} y={int(r_dot_position[1]) - height/2}")

            # Display the image
            cv2.imshow("Dot Display", image)

            # Wait until a key is pressed
            cv2.waitKey(1)

        # Todo: detect which is closest

if __name__=="__main__":
    ekeys = EyeKeys()

    cap = cv2.VideoCapture(0)
    # Process each frame
    while True:
        ret, frame = cap.read()
        ekeys.process(frame)

