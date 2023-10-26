import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from guided_filter import _gf_gray
from time import time

### "Marine snow detection for real time feature detection" ###
# https://doi.org/10.1109/AUV53081.2022.9965895


# Function to get index for circular array
def wrap_index(index: int) -> int:
    return index % 3


### PARAMETERS ###
# Coefficients for Y-channel (BGR)
COEFFS = np.array([0.114, 0.587, 0.299])
# Radius of mean filter
r = 5
# Regularisation parameter that controls degree of smoothness for guided filter
eps = 0.05
# Subsampling ratio for guided filter
s = 4
# Kernel for tophat morphology
kernel1 = np.ones((3, 3), np.uint8)
# Kernel for density map
w = 50
kernel2 = np.ones((w, w)) / (w**2)
# Kernel for open/dilation operation of feature mask
kernel3 = cv.getStructuringElement(cv.MORPH_RECT, (w, w))
##################


cap = cv.VideoCapture("Grass.MP4")
# cap = cv.VideoCapture("C4_GX040003.MP4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

NUM_FRAMES = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# Create window with freedom of dimensions
cv.namedWindow("Output", cv.WINDOW_NORMAL)
# Create window with freedom of dimensions
cv.namedWindow("Original", cv.WINDOW_NORMAL)

start_time = time()
frame_counter = 0
images = []
idx = 1  # Index for accesing images
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # frame = cv.resize(frame, (0, 0), fx=1, fy=1)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_counter += 1
    # Converting input image to gray with weights
    Y = cv.transform(frame, COEFFS.reshape((1, 3)))
    # Y_blurred = cv.GaussianBlur(Y, (2 * r + 1, 2 * r + 1), cv.BORDER_REPLICATE)
    Y_blurred = cv.blur(Y, (2 * r + 1, 2 * r + 1), cv.BORDER_REPLICATE)

    # Y_lp = _gf_gray(Y, Y, r, eps, s)
    # Y_lp = Y_lp.astype(np.uint8)
    # Y_hp = Y - Y_lp
    # Y_hp = cv.subtract(Y, Y_lp)
    Y_hp = cv.subtract(Y, Y_blurred)

    if len(images) < 3:
        images.append(Y_hp)
        if len(images) < 3:
            disp_frame = np.zeros_like(frame)
            continue
    else:
        # Update the future image
        images[wrap_index(idx + 1)] = Y_hp

    prev_img = images[wrap_index(idx - 1)]
    curr_img = images[wrap_index(idx)]
    next_img = images[wrap_index(idx + 1)]

    # Keep moving particles using temporal information
    P = cv.subtract(curr_img, np.minimum(prev_img, next_img))
    # Threshold P to get binary image
    (_, P) = cv.threshold(P, 1, 255, cv.THRESH_BINARY)
    # Use tophat morphology to keep all small particles
    P_particles = cv.morphologyEx(P, cv.MORPH_TOPHAT, kernel1)
    # Subtract from original P to keep larger particles
    P = cv.subtract(P, P_particles)

    # Compute density map
    P_density = cv.filter2D(P, -1, kernel2, borderType=cv.BORDER_REPLICATE)
    # Threshold to only keep dense regions corresponding to features, i.e. locate areas of many features
    _, P_density = cv.threshold(P_density, 6, 255, cv.THRESH_BINARY)
    # Perform opening operation to remove noise
    P_density = cv.morphologyEx(P_density, cv.MORPH_OPEN, kernel3)
    # Perform closing to remove holes in blobs
    P_density = cv.morphologyEx(P_density, cv.MORPH_DILATE, kernel3)
    # Isolate snow by subtracting feature mask
    snow_mask = cv.subtract(P, P_density)

    # Create an RGB image so that it can be overlayed with the frame
    P_mask_rgb = cv.merge(
        [np.zeros_like(snow_mask), np.zeros_like(snow_mask), snow_mask]
    )

    # Overlay snow mask with original image
    P_overlay = cv.addWeighted(disp_frame, 1, P_mask_rgb, 0.5, 0)

    P_overlay = cv.putText(
        P_overlay,
        f"FPS: {(1 / (time() - start_time)):.2f}",
        (0, 100),
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        2,
        cv.LINE_AA,
    )

    # Concatenate the two images horizontally
    # concatenated_image = cv.hconcat(
    #     [P_overlay, cv.merge([snow_mask, snow_mask, snow_mask])]
    # )
    concatenaed_image = cv.hconcat(
        [P_overlay, cv.merge([snow_mask, snow_mask, snow_mask])]
    )
    # cv.imshow("P", P_density)
    cv.imshow("Output", concatenaed_image)
    cv.imshow("Original", disp_frame)
    if cv.waitKey(1) == ord("q"):
        break

    start_time = time()
    disp_frame = frame
    idx += 1

    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == NUM_FRAMES - 1:
        frame_counter = 0  # Or whatever as long as it is the same as next line
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
