import cv2
import numpy as np
import pickle
from tqdm import tqdm

# Parameters for camera calibration
chessboard_size = (8, 6)  # Checkerboard's dimensions
square_size = 0.025  # Checker size in m
frame_skip = 3  # Process every 1/n frames

# Create object points for the chessboard corners
obj_points = []
img_points = []
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

video_file = 'CalibrationVideos/GX010526.MP4'
cap = cv2.VideoCapture(video_file)

for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/frame_skip))):
    ret, frame = cap.read()
    if not ret:
        break

    if i % frame_skip != 0:
        continue  # Skip frames that are not multiples of frame_skip

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)

cap.release()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print(f'camera_matrix:\n{mtx}\ndistortion_coefficients:\n{dist}')

# Save the camera matrix and distortion coefficients as a pickle file
calibration_data = {'camera_matrix': mtx, 'distortion_coefficients': dist}

with open('camera_calibration.pickle', 'wb') as f:
    pickle.dump(calibration_data, f)

print("Camera calibration completed and saved to 'camera_calibration.pickle'")

# Load the first frame from the video and undistort it
cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()
cap.release()

if ret:
    undistorted_frame = cv2.undistort(frame, mtx, dist)

    # Resize the image to 1/4th the scale
    height, width = undistorted_frame.shape[:2]
    new_height = height // 4
    new_width = width // 4
    resized_frame = cv2.resize(undistorted_frame, (new_width, new_height))

    cv2.imshow('Undistorted Frame with Checkerboard', resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
