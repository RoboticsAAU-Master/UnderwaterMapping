import cv2 as cv
import glob
import os


def enumerate_images(input_folder, output_folder, start_index=0):
    # Load all images
    images = [cv.imread(file) for file in glob.glob(input_folder + "/*")]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all images
    for i, image in enumerate(images):
        cv.imwrite(output_folder + f"/original_image_{(start_index + i)}.png", image)


if __name__ == "__main__":
    enumerate_images("Artificial-snow/test", "Artificial-snow/Input", start_index=0)
