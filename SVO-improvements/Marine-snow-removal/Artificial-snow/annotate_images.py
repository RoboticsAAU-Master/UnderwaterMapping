import cv2 as cv
import numpy as np
import glob
from tqdm import tqdm
import os
from generate_snow import generate_snow


def annotate_images(input_folder, image_folder, mask_folder, start_index=0):
    # Load all images
    images = [cv.imread(file) for file in glob.glob(input_folder + "/*")]

    # Create the output folders if they don't exist
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Iterate through all images
    for i, image in tqdm(enumerate(images)):
        snow_image, snow_mask = generate_snow(image, mask=True)

        cv.imwrite(image_folder + f"/image_{(start_index + i)}.png", snow_image)
        cv.imwrite(mask_folder + f"/mask_{(start_index + i)}.png", snow_mask)


if __name__ == "__main__":
    annotate_images(
        "Artificial-snow/Input",
        "Artificial-snow/images",
        "Artificial-snow/masks",
        start_index=0,
    )
