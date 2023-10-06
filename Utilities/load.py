import cv2 as cv
import numpy as np
import os

def load_monocular(path_to_folder : str, imu : bool = False) -> tuple[cv.VideoCapture,]:
    files = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]
    if len(files) != (1 + int(imu)): raise Exception(f"Expected {1 + int(imu)} file(s), but got {len(files)}")
    
    imu_data = 0
    for file_name in files:
        if (file_name.find(".mp4") and len(files) != 3) or file_name.find("left.mp4"): cap = cv.VideoCapture(file_name)
        elif file_name.find("imu.txt"): imu_data = 0 # Implement method to extract IMU data
        else: continue
    if not cap: raise Exception("Video capture not found")
    
    # Return the loaded data
    return (cap, imu_data)

def load_stereo(path_to_folder : str, imu : bool = True) -> tuple[cv.VideoCapture,]:
    files = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]
    if len(files) != (2 + int(imu)): raise Exception(f"Expected {2 + int(imu)} files, but got {len(files)}")
    
    imu_data = 0
    for file_name in files:
        if file_name.find("left.mp4"): left_cap = cv.VideoCapture(file_name)
        elif file_name.find("right.mp4"): right_cap = cv.VideoCapture(file_name)
        elif file_name.find("imu.txt"): imu_data = 0 # Implement method to extract IMU data
        else: continue
    if not left_cap or right_cap: raise Exception("Left and/or right video capture not found")
    
    # Return the loaded data
    return ([left_cap, right_cap], imu_data)

