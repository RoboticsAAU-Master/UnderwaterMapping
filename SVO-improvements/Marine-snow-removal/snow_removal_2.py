import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from time import time

### "Real-time marine snow noise removal from underwater video sequences" ###
# https://doi.org/10.1117/1.JEI.27.4.043002


# Function to get index for circular array
def wrap_index(index: int) -> int:
    return index % 3


### PARAMETERS ###
# Temporal number of frames
t = 3
# Patch width
w = 5
# Threshold (equation 3)
c = 70
# Spatial patch size (between 1 and 7)
s = 3
r = s // 2
# Threshold (equation 7)
d = 7
##################


cap = cv.VideoCapture("Grass.MP4")
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
    frame_counter += 1
    # frame = cv.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if len(images) < 3:
        images.append(frame)
        if len(images) < 3:
            disp_frame = np.zeros_like(frame)
            continue

    # Step 1-2
    prev_img = images[wrap_index(idx - 1)]
    curr_img = images[wrap_index(idx)]
    next_img = images[wrap_index(idx + 1)]

    # Step 3-4
    images_D = []
    for img in images:
        img_D = cv.add(
            cv.absdiff(img[:, :, 2], img[:, :, 1]),
            cv.add(
                cv.absdiff(img[:, :, 1], img[:, :, 0]),
                cv.absdiff(img[:, :, 0], img[:, :, 2]),
            ),
        )
        images_D.append(img_D)

    # Step 5-6
    img_criteria = cv.subtract(
        images_D[wrap_index(idx)],
        np.minimum(
            cv.blur(
                images_D[wrap_index(idx - 1)], (w, w), borderType=cv.BORDER_REPLICATE
            ),
            cv.blur(
                images_D[wrap_index(idx + 1)], (w, w), borderType=cv.BORDER_REPLICATE
            ),
        ),
    )

    # Step 7-11
    # _, candidate_mask = cv.threshold(img_criteria, c, 255, cv.THRESH_BINARY_INV)
    candidates = np.argwhere(img_criteria > c)
    candidates_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    if len(candidates) == 0:
        continue
    for cand in candidates:
        lower_x = max(cand[1] - r, 0)
        lower_y = max(cand[0] - r, 0)
        upper_x = min(cand[1] + r + 1, frame.shape[1])
        upper_y = min(cand[0] + r + 1, frame.shape[0])

        prev_patch = images[wrap_index(idx - 1)][lower_y:upper_y, lower_x:upper_x, :]
        next_patch = images[wrap_index(idx + 1)][lower_y:upper_y, lower_x:upper_x, :]
        prev_max = np.max(prev_patch, axis=2)
        next_max = np.max(next_patch, axis=2)

        if np.any((frame[cand[0], cand[1]] - prev_max) > d) or np.any(
            (frame[cand[0], cand[1]] - next_max) > d
        ):
            candidates_mask[cand[0], cand[1]] = 255

    # Step 12-14
    _, blobs = cv.connectedComponents(candidates_mask)

    # Update the future image
    images[wrap_index(idx - 1)] = frame

    # Show images
    cv.imshow("Original", disp_frame)
    cv.imshow("Output", candidates_mask)
    if cv.waitKey(1) == ord("q"):
        break

    start_time = time()
    idx += 1
    disp_frame = frame

    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == NUM_FRAMES - 1:
        frame_counter = 0  # Or whatever as long as it is the same as next line
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
