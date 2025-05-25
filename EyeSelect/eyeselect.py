import cv2 
import time
import uuid
import numpy as np
from scipy.spatial.distance import cdist
from EyeSelect.face import Face, FaceFinder 
from EyeSelect.utils import VideoCapture

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

class EventSelector:

    def __init__(self,relaxation = 0.5):
        self.latched = False
        self.l_registry = dict()
        self.events = dict()
        self.unlatches = dict()

        self.debouncing_start = time.time()
        self.relaxation = relaxation
        self.last_detection = time.time()

    def register(self,event,unlatch):
        # Generate a random UUID (UUID4)
        unique_id = str(uuid.uuid4())
        self.events[unique_id] = event
        self.unlatches[unique_id] = unlatch
        self.l_registry[unique_id] = False

    def select(self, payload):
        if self.relaxation > (time.time() - self.debouncing_start):
            return

        if self.latched:
            for key, latched in self.l_registry.items():
                if latched:
                    ret = self.unlatches[key](payload)
                    if ret is not None:
                        self.l_registry[key] = ret
                        self.latched = self.l_registry[key]
        else:
            for key, func in self.events.items():
                ret = func(payload)
                if ret:
                   self.l_registry[key] = ret
                   self.latched = self.l_registry[key]
                   self.debouncing_start = time.time()
                   break

class EyeBaselineTracker:
    def __init__(self):
        # Initialize each baseline as an empty list and baseline median as class attribute
        self.std_x_values = []
        self.std_y_values = []
        self.x_y_std_values = []
        self.d_std_r_values = []
        self.d_std_l_values = []
        self.u_std_r_values = []
        self.u_std_l_values = []
        self.max_dist_x_values = []
        self.max_dist_y_values = []

        # Baseline medians (can be used directly)
        self.std_x = 9999.0
        self.std_y = 9999.0
        self.x_y_std = 9999.0
        self.d_std_r = 9999.0
        self.d_std_l = 9999.0
        self.u_std_r = 9999.0
        self.u_std_l = 9999.0
        self.max_dist_x = 9999.0
        self.max_dist_y = 9999.0

    def add_obj(self, eio):
        # Update lists with new values
        self.std_x_values.append(eio.std_x)
        self.std_y_values.append(eio.std_y)
        self.x_y_std_values.append(eio.x_y_std)
        self.d_std_r_values.append(eio.d_std_r)
        self.d_std_l_values.append(eio.d_std_l)
        self.u_std_r_values.append(eio.u_std_r)
        self.u_std_l_values.append(eio.u_std_l)
        self.max_dist_x_values.append(eio.max_dist_x)
        self.max_dist_y_values.append(eio.max_dist_y)


        # Update medians
        self.std_x = np.median(self.std_x_values)
        self.std_y = np.median(self.std_y_values)
        self.x_y_std = np.median(self.x_y_std_values)
        self.d_std_r = np.median(self.d_std_r_values)
        self.d_std_l = np.median(self.d_std_l_values)
        self.u_std_r = np.median(self.u_std_r_values)
        self.u_std_l = np.median(self.u_std_l_values)
        self.max_dist_x = np.median(self.max_dist_x_values)
        self.max_dist_y = np.median(self.max_dist_y_values)

class EyeIntermediateObject:

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.std_x = 0.0
        self.std_y = 0.0
        self.x_y_std = 0.0
        self.d_std_r = 0.0
        self.d_std_l = 0.0
        self.u_std_r = 0.0 
        self.u_std_l = 0.0
        self.left_th = 0.0
        self.right_th = 0.0
        self.up_th = 0.0
        self.blink_th = 0.0
        self.l_eye_pupil = [0.0,0.0]
        self.r_eye_pupil = [0.0,0.0]
        self.lu = 0.0
        self.ru = 0.0
        self.max_radius = 0.0
        self.max_dist_x = 0.0
        self.max_dist_y = 0.0

class EyeSelect:

    def __init__(self,
                 left_cb = None, 
                 right_cb = None,
                 blink_cb = None,
                 up_cb = None,
                 verbose = False):
        self.finder = FaceFinder()
        self.face = Face()

        self.verbose = verbose
        
        self.l_buffer = []
        self.r_buffer = []
        self.u_buffer = []
        self.d_buffer = []

        self.left_cb = left_cb
        self.right_cb = right_cb
        self.blink_cb = blink_cb
        self.up_cb = up_cb
        self.debouncing = False
        self.baseline_y_u = 0.00
        self.baseline_y_d = 0.00

        self.relaxation_tracker = 0.0

        self.eventSelector = EventSelector()
        if left_cb is not None:
            self.eventSelector.register(self.__right, self.__right_unlatch)
        if right_cb is not None:
            self.eventSelector.register(self.__left, self.__left_unlatch)
        if blink_cb is not None:
            self.eventSelector.register(self.__up, self.__up_unlatch)
        if up_cb is not None:
            self.eventSelector.register(self.__blink, self.__blink_unlatch)

        self.baseline = EyeBaselineTracker()

    def post_detection(self):
        for _ in range(len(self.l_buffer)):
            self.l_buffer.pop(0)
            self.r_buffer.pop(0)
        
        for _ in range(len(self.u_buffer)):
            self.u_buffer.pop(0)
            self.d_buffer.pop(0)

    def __left(self, eio : EyeIntermediateObject):
        if (eio.std_x > self.baseline.std_x and eio.x < eio.left_th):
            self.debouncing_start = time.time()
            self.debouncing = True
            self.left_cb()
            self.post_detection()
            return True

    def __left_unlatch(self,eio : EyeIntermediateObject):

        self.relaxation_tracker = (eio.std_x - self.baseline.std_x)/self.baseline.std_x + (eio.x - (eio.left_th + 40))/(eio.left_th + 40) 
        if (eio.std_x < self.baseline.std_x and eio.x > eio.left_th + 40):
            return False
        

    def __right(self, eio : EyeIntermediateObject):

        if (eio.std_x > self.baseline.std_x and eio.x > eio.right_th):
            self.debouncing_start = time.time()
            self.debouncing = True
            self.right_cb()
            self.post_detection()
            return True

    def __right_unlatch(self,eio : EyeIntermediateObject):

        self.relaxation_tracker = (eio.std_x - self.baseline.std_x)/self.baseline.std_x + (eio.x - (eio.right_th + 40))/(eio.right_th + 40) 
        if (eio.std_x < self.baseline.std_x and eio.x < eio.right_th - 40):
            return False

    def __up(self, eio : EyeIntermediateObject):

        if ( eio.max_dist_x < self.baseline.max_dist_x * 1.25 and eio.max_dist_y > self.baseline.max_dist_y * 1.5 and eio.std_y > self.baseline.std_y * 1.5):
            self.debouncing_start = time.time()
            self.up_cb()
            self.post_detection()
            return True

    def __up_unlatch(self,eio : EyeIntermediateObject):

        self.relaxation_tracker = (eio.std_y - (self.baseline.std_y * 1.25))/(self.baseline.std_y * 1.25)
        if eio.std_y < self.baseline.std_y * 1.25:
            return False

    def __blink(self, eio : EyeIntermediateObject):
        if (eio.max_radius > eio.blink_th and eio.std_x > self.baseline.std_x * 1.5 and eio.std_y > self.baseline.std_y * 1.5):
            self.debouncing_start = time.time()
            self.debouncing = True
            self.blink_cb()
            self.post_detection()
            return True

    def __blink_unlatch(self,eio : EyeIntermediateObject):
        
        self.relaxation_tracker = (eio.std_y - (self.baseline.std_y * 1.25))/(self.baseline.std_y * 1.25)
        if eio.std_y < self.baseline.std_y * 1.25:
            return False

    # @recoverable
    def process(self,image,left_th=-100, right_th=100, up_th=1.5, blink_th=100):

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
        width, height = 1000, 1000

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

            if self.verbose:
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

        max_radius = 0.0

        all_points = np.concatenate((self.l_buffer, self.r_buffer))
        distances = cdist(all_points, all_points)  # Shape: (len(a), len(b))
        max_radius = np.max(distances)

        # Max absolute difference in X
        max_dist_x = all_points[:, 0].max() - all_points[:, 0].min()  # shape broadcasted to (len(a), len(b))
        # Max absolute difference in Y
        max_dist_y = all_points[:, 1].max() - all_points[:, 1].min()
        max_radius = np.max(distances)
        max_radius = np.max(distances)


        eio = EyeIntermediateObject()
        eio.x = x
        eio.y = y
        eio.std_x = std_x
        eio.std_y = std_y
        eio.x_y_std = x_y_std
        eio.d_std_r = d_std_r
        eio.d_std_l = d_std_l
        eio.u_std_l = u_std_l
        eio.u_std_r = u_std_r
        eio.left_th = left_th
        eio.right_th = right_th
        eio.up_th = up_th
        eio.blink_th = blink_th
        eio.l_eye_pupil = l_eye_pupil
        eio.r_eye_pupil = r_eye_pupil
        eio.lu = lu
        eio.ru = ru
        eio.max_radius = max_radius
        eio.max_dist_x = max_dist_x
        eio.max_dist_y = max_dist_y

        self.eventSelector.select(eio)
        self.baseline.add_obj(eio)

        # print(f"Eye std: u_std_r={u_std_r:.2f}, u_std_l={u_std_l:.2f} dist_u={(distance(l_eye_pupil,lu) + distance(r_eye_pupil,ru))/2:.2f} d_std_r={d_std_r:.2f}, d_std_l={d_std_l:.2f} dist_d={(distance(l_eye_pupil,ld) + distance(r_eye_pupil,rd))/2:.2f}")

        # print(f"Eye std: X={std_x:.2f}, Y={std_y:.2f} x={x} y={y} look_up = {(distance(l_eye_pupil,ld) + distance(r_eye_pupil,rd))/2}")
        return self.relaxation_tracker

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

