import cv2 as cv
import numpy as np
import os


def load_monocular(path_to_folder : str, imu : bool = False) -> tuple:
    files = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]
    if len(files) != (1 + int(imu)): raise Exception(f"Expected {1 + int(imu)} file(s), but got {len(files)}")
    
    for file_name in files:
        if (file_name.find(".mp4") and len(files) != 3) or file_name.find("left.mp4"): cap = cv.VideoCapture(file_name)
        elif file_name.find("imu.txt"): imu_data = 0
        else: continue
    
    # Return the loaded data
    if imu: 
        return (cap, imu_data)
    else:
        return (cap)

def load_stereo(path_to_folder : str, imu : bool = True) -> tuple:
    files = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]
    if len(files) != (2 + int(imu)): raise Exception(f"Expected {2 + int(imu)} files, but got {len(files)}")
    
    for file_name in files:
        if file_name.find("left.mp4"): left = cv.VideoCapture(file_name)
        elif file_name.find("right.mp4"): right = cv.VideoCapture(file_name)
        elif file_name.find("imu.txt"): imu_data = 0
        else: continue
    
    # Return the loaded data
    if imu: 
        return ([left, right], imu_data)
    else:
        return ([left, right])



def play_video(path : str, scale : float = 1) -> None:
    """Function to play back video

    Args:
        path (str): Relative path  
        scale (list): To scale the image
        
    Returns:
        None
    """
    cap = cv.VideoCapture(path)

    if not cap.isOpened():
        raise Exception("Cannot Open Camera")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame = cv.resize(frame, (0,0), fx=scale, fy=scale)
        
        #Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

