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

class EyeSelect:

    def __init__(self,
                 left_cb = lambda : 0, 
                 right_cb = lambda : 0,
                 blink_cb = lambda : 0,
                 up_cb = lambda : 0):
        self.finder = FaceFinder()
        self.face = Face()
        
        self.l_buffer = []
        self.r_buffer = []
        self.u_buffer = []
        self.d_buffer = []

        self.left_cb = left_cb
        self.right_cb = right_cb
        self.blink_cb = blink_cb
        self.up_cb = up_cb
        self.down_cb = down_cb
        self.debouncing = False
        self.debouncing_start = time.time()
        self.debouncing_relaxation = 3.0
        self.last_detection = time.time()
        self.baseline_y_u = 0.00
        self.baseline_y_d = 0.00

    def post_detection(self):
        for _ in range(len(self.l_buffer)):
            self.l_buffer.pop(0)
            self.r_buffer.pop(0)
        
        for _ in range(len(self.u_buffer)):
            self.u_buffer.pop(0)
            self.d_buffer.pop(0)


    # @recoverable
    def process(self,image,left_th=-100, right_th=100, up_th=1.5, blink_th=40):

        x_y_std = 40 # std deviation thershold for x_y move

        face_mesh = self.finder.find(image)
        if not face_mesh:
            return None

        self.face.process(image, face_mesh)

        # face_landmarks = self.face.getLandmarks()
        l_eye = self.face.getLeftEye()
        r_eye = self.face.getRightEye()
        # l_eye_landmarks = l_eye.getLandmarks()
        # r_eye_landmarks = r_eye.getLandmarks()
        l_eye_pupil = l_eye.getPupil()
        r_eye_pupil = r_eye.getPupil()
        l_eye_minMax = l_eye.getMinMax()
        r_eye_minMax = r_eye.getMinMax()

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
        # dot_color = (0, 0, 255)  # Red in BGR
        background_color = (255, 255, 255)  # White

        # Create a white background image
        image = np.full((height, width, 3), background_color, dtype=np.uint8)
        # Ensure coordinates are integers
        if l_dot_position is not None and r_dot_position is not None:
            l_dot_position = (int(l_dot_position[0]), int(l_dot_position[1]))
            r_dot_position = (int(r_dot_position[0]), int(r_dot_position[1]))
            self.l_buffer.append(l_dot_position)
            self.r_buffer.append(r_dot_position)

            if len(self.r_buffer) > 20:
                self.r_buffer.pop(0)
                self.l_buffer.pop(0)

            for n in range(len(self.r_buffer)):
                cv2.circle(image, self.l_buffer[n], dot_radius, (0, 0, 255), -1)
                cv2.circle(image, self.r_buffer[n], dot_radius, (255, 0, 0), -1)

            # Display the image
            cv2.imshow("Dot Display", image)

            # Wait until a key is pressed
            cv2.waitKey(1)

        l_std_x, l_std_y = compute_std(self.l_buffer)
        r_std_x, r_std_y = compute_std(self.r_buffer)

        
        x = ((int(l_dot_position[0]) - width/2) + (int(r_dot_position[0]) - width/2))/2
        y = ((int(l_dot_position[1]) - height/2) + (int(r_dot_position[1]) - height/2))/2

        std_x = (l_std_x + r_std_x)/2
        std_y = (l_std_y + r_std_y)/2

        self.u_buffer.append((distance(l_eye_pupil,lu)/distance(ld,lu),distance(r_eye_pupil,ru)/distance(rd,ru)))
        self.d_buffer.append((distance(l_eye_pupil,ld)/distance(ld,lu),distance(r_eye_pupil,rd)/distance(rd,ru)))
        u_std_r, u_std_l = compute_std(self.u_buffer)
        d_std_r, d_std_l = compute_std(self.d_buffer)

        if (u_std_r + u_std_l)/2 <= 0.02 and (self.debouncing):
            self.baseline_y_u = (distance(l_eye_pupil,lu) + distance(r_eye_pupil,ru))/2

        if len(self.d_buffer) > 20:
            self.u_buffer.pop(0)
            self.d_buffer.pop(0)

        # print(f"Eye std: u_std_r={u_std_r:.2f}, u_std_l={u_std_l:.2f} dist_u={(distance(l_eye_pupil,lu) + distance(r_eye_pupil,ru))/2:.2f} d_std_r={d_std_r:.2f}, d_std_l={d_std_l:.2f} dist_d={(distance(l_eye_pupil,ld) + distance(r_eye_pupil,rd))/2:.2f}")

        # print(f"Eye std: X={std_x:.2f}, Y={std_y:.2f} x={x} y={y} look_up = {(distance(l_eye_pupil,ld) + distance(r_eye_pupil,rd))/2}")

        if (x_y_std < std_x and x < left_th) and not self.debouncing:
            self.debouncing_start = time.time()
            self.debouncing = True
            self.left_cb()
            print(f"last detection: {time.time() - self.last_detection}")
            self.last_detection = time.time()
            self.post_detection()
        elif (x_y_std-10 > std_x and x > left_th + 40) and (self.debouncing_relaxation < (time.time() - self.debouncing_start)):
            self.debouncing = False
            pass

        if (x_y_std < std_x and x > right_th) and not self.debouncing:
            self.debouncing_start = time.time()
            self.debouncing = True
            self.right_cb()
            print(f"last detection: {time.time() - self.last_detection}")
            self.last_detection = time.time()
            self.post_detection()
        elif (x_y_std-10 > std_x and x < right_th - 40) and (self.debouncing_relaxation < (time.time() - self.debouncing_start)):
            self.debouncing = False
            pass

        u_std = (u_std_r + u_std_l)/2
        if ( up_th > abs((distance(l_eye_pupil,lu) + distance(r_eye_pupil,ru))/2 - self.baseline_y_u) and u_std >= 0.03) and not self.debouncing:
            self.debouncing_start = time.time()
            self.debouncing = True
            self.up_cb()
            print(f"last detection: {time.time() - self.last_detection}")
            self.last_detection = time.time()
            self.post_detection()
        elif self.debouncing and (self.debouncing_relaxation < (time.time() - self.debouncing_start)):
            self.debouncing = False


        if (std_y > blink_th) and not self.debouncing:
            self.debouncing_start = time.time()
            self.debouncing = True
            self.blink_cb()
            print(f"last detection: {time.time() - self.last_detection}")
            self.last_detection = time.time()
            self.post_detection()
        elif (std_y < blink_th-20) and (self.debouncing_relaxation < (time.time() - self.debouncing_start)):
            self.debouncing = False
            pass




if __name__=="__main__":
    ekeys = EyeSelect(
        left_cb=lambda : print("left"),
        right_cb=lambda : print("right"),
        blink_cb=lambda : print("blink"),
        up_cb=lambda : print("up")
    )

    cap = cv2.VideoCapture(0)
    # Process each frame
    while True:
        ret, frame = cap.read()
        ekeys.process(frame)

