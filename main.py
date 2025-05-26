import cv2 
import time
import numpy as np
from EyeSelect.eyeselect import EyeSelect

if __name__=="__main__":
    ekeys = EyeSelect(
        left_cb=lambda: print("\033[91mleft\033[0m"),   # Red
        right_cb=lambda: print("\033[94mright\033[0m"), # Blue
        blink_cb=lambda: print("\033[92mblink\033[0m"), # Green
        up_cb=lambda: print("\033[93mup\033[0m")        # Yellow
    )

    cap = cv2.VideoCapture(0)
    # Process each frame
    while True:
        ret, frame = cap.read()
        relaxation_tracker = ekeys.process(
            frame,
            left_th=-50,
            right_th=50,
            up_th=1.5,
            blink_th=40
        )
        print(f"Number below zero means software is relaxed and waiting for trigger event: {relaxation_tracker}, relaxed:{relaxation_tracker<0.0}")
