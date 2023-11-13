import cv2 as cv
import numpy as np
import glob

# Load all images
images = glob.glob("Input/*.tiff")

# Iterate through all images
for i, image in enumerate(images):
    cv.imwrite("images/image_" + i, image)
