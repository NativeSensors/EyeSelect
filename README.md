# EyeSelect

EyeSelect is a Python module for detecting simple eye-based gestures (left, right, up, blink) from a webcam feed. It uses facial landmark detection to track eye pupil positions and determine directional movement or blinks.

## Features

* Detects and reacts to:

  * **Looking left**
  * **Looking right**
  * **Looking up**
  * **Blinking**
* Works in real-time using a webcam.
* Callback system to trigger custom actions for each gesture.

## Requirements

* Python 3.7+
* OpenCV
* NumPy
* A face detection module (like `face` and `utils` used internally, assumed to wrap something like MediaPipe or Dlib)

> Note: This code assumes the presence of `face.py` and `utils.py` which provide face mesh detection and video capture utilities. Ensure these are available and functional.

## How It Works

1. **Capture Video Feed** using OpenCV.
2. **Detect Face and Eyes** using the custom `FaceFinder` and `Face` classes.
3. **Track Pupil Movement** over time and compute relative positions within the eye bounds.
4. **Detect Movement Patterns** based on standard deviations and directional thresholds:

   * Left/Right → Horizontal pupil displacement.
   * Up → Vertical pupil position vs. baseline.
   * Blink → High variation in vertical pupil position.
5. **Trigger Callbacks** when a gesture is confidently detected.

## Example Usage

```python
from eyeselect import EyeSelect
import cv2

# Define what to do on each gesture
ekeys = EyeSelect(
    left_cb=lambda: print("left"),
    right_cb=lambda: print("right"),
    blink_cb=lambda: print("blink"),
    up_cb=lambda: print("up")
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    ekeys.process(
        frame,
        left_th=-100,     # Threshold for left detection
        right_th=100,     # Threshold for right detection
        up_th=1.5,        # Threshold for up detection
        blink_th=40       # Threshold for blink detection
    )
```

## Tuning Parameters

These thresholds can be tweaked in the `process()` function to better fit your lighting, camera, or user behavior:

* `left_th`, `right_th`: Horizontal offset from center for directional detection.
* `up_th`: Relative upward movement vs. a stable baseline.
* `blink_th`: Variance in vertical position to detect blinks.
* Internally uses standard deviation and position buffers for stability and debouncing.

## Notes

* A visual debug window (`Dot Display`) shows tracked pupil positions.
* The system uses basic debouncing to avoid repeated detections.
* The precision depends heavily on lighting and face mesh quality.
