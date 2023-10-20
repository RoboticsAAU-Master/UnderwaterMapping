import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from guided_filter import _gf_gray 

# Function to get index for circular array
def wrap_index(index : int) -> int:
    return index % 3

# cap = cv.VideoCapture("marine_snow_large.MP4")
cap = cv.VideoCapture("C4_GX040003.MP4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
NUM_FRAMES = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

frame_counter = 0

images = []
COEFFS = np.array([0.114, 0.587, 0.299]) # Coefficients for Y-channel (BGR)
idx = 1 # Index for accesing images
r = 8 # Radius of mean filter
eps = 0.05 # Regularisation parameter that controls degree of smoothness 
s = 4 # Subsampling ratio

cv.namedWindow("Output", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv.namedWindow("Original", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame_counter += 1
    # Converting input image to gray with weights
    Y = cv.transform(frame, COEFFS.reshape((1,3)))
    Y_blurred = cv.blur(Y, (2*r+1,2*r+1), cv.BORDER_REPLICATE)
    Y_lp = _gf_gray(Y, Y, r, eps, s)
    Y_lp = Y_lp.astype(np.uint8)
    #Y_hp = Y - Y_lp
    # Y_hp = cv.subtract(Y, Y_lp)
    Y_hp = cv.subtract(Y, Y_blurred)
    
    if len(images) < 3:
        images.append(Y_hp)
        continue
    
    prev_img = images[wrap_index(idx-1)]
    curr_img = images[wrap_index(idx)]
    next_img = images[wrap_index(idx+1)]
    
    P = cv.subtract(curr_img, np.minimum(prev_img, next_img))
    
    images[wrap_index(idx-1)] = Y_hp
    
    idx += 1
    
    # Display the resulting frame
    #cv.imshow('Output', cv.medianBlur(Y_hp, 3))
    cv.imshow('Output', np.multiply(P,10))
    cv.imshow('Original', frame)
    if cv.waitKey(1) == ord('q'):
        break
    
    #If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == NUM_FRAMES - 1:
        frame_counter = 0 #Or whatever as long as it is the same as next line
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
