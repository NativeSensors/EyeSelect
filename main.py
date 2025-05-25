import cv2 
import time
import numpy as np
from face import Face, FaceFinder 
from utils import VideoCapture
from eyeselect import EyeSelect

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
        ekeys.process(
            frame,
            left_th=-60,
            right_th=60,
            up_th=1.5,
            blink_th=100
        )

