import cv2 as cv
import numpy as np

# Returns whether the next frame is a keyframe and the actual frame
def keyframe(cap: cv.VideoCapture, kf_rate : int = 10) -> tuple[bool, np.ndarray]:
    # Get frame number
    frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    
    # Read the next frame
    _, frame = cap.read()
    
    is_kf = (frame_num % kf_rate == 0)
    return  (is_kf, frame)
