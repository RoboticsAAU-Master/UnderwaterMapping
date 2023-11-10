import cv2 as cv
import numpy as np
from scipy.stats import qmc
from enum import Enum
import math


class Motion_Direction(Enum):
    FOWARD_BACKWARD = 0
    UP_DOWN = 1
    LEFT_RIGHT = 2


def in_bounds(shape, x, y):
    return x >= 0 and x < shape[1] and y >= 0 and y < shape[0]


def get_bg_hue(image_hsv):
    # Get hue channel and flatten
    h = image_hsv[:, :, 0]
    h = h.flatten()

    # Calculate histogram
    hist = np.histogram(h, bins=180, range=(0, 180))

    # Obtain the background hue
    bg_hue = np.argmax(hist[0])

    return bg_hue


def get_random_samples(ref_image, num_samples, method="halton"):
    if method == "random":
        # Generate random samples
        x_vals = np.random.randint(0, ref_image.shape[1], size=num_samples)
        y_vals = np.random.randint(0, ref_image.shape[0], size=num_samples)
        samples_coords = np.column_stack((x_vals, y_vals)).astype(int)

    elif method == "halton":
        # Generate Halton sequence samples
        sampler = qmc.Halton(d=2, scramble=False)
        samples = sampler.random(n=num_samples)
        samples_coords = np.column_stack(
            (samples[:, 0] * ref_image.shape[1], samples[:, 1] * ref_image.shape[0])
        )
        samples_coords = samples_coords.astype(int)

    return samples_coords


def get_motion_direction(ref_image, motion_type, x, y, noise_var=0.0):
    if motion_type == Motion_Direction.FOWARD_BACKWARD.value:
        stretch_direction = np.array([x, y]) - np.array(
            [ref_image.shape[1] // 2, ref_image.shape[0] // 2]
        )

    elif motion_type == Motion_Direction.UP_DOWN.value:
        stretch_direction = np.array([x, y]) - np.array([x, 0])

    elif motion_type == Motion_Direction.LEFT_RIGHT.value:
        stretch_direction = np.array([x, y]) - np.array([0, y])

    else:
        raise ValueError("Invalid motion direction")

    if np.linalg.norm(stretch_direction) != 0:
        stretch_direction = stretch_direction / np.linalg.norm(stretch_direction)

    stretch_direction = stretch_direction + np.random.normal(0, noise_var, 2)
    return stretch_direction


def generate_snow(ref_image):
    # Convert to hsv color space
    hsv = cv.cvtColor(ref_image, cv.COLOR_BGR2HSV)

    # Get background hue
    bg_hue = get_bg_hue(hsv)

    # Calculate the number of snow particles
    num_range = (400, 600)
    num_snow = np.random.randint(num_range[0], num_range[1] + 1)

    # Generate Halton sequence
    samples_coords = get_random_samples(ref_image, num_snow, method="random")

    # Shape and size of snow particles
    snow_mask = np.zeros_like(hsv)
    motion_type = np.random.randint(0, 2 + 1)  # Define the motion type
    for x, y in samples_coords:
        # Color of snow particles
        val = np.array([bg_hue, np.random.randint(0, 80), np.random.randint(80, 200)])
        snow_mask[y, x, :] = val

        # Get direction along which we stretch the marine snow particle
        stretch_direction = get_motion_direction(
            ref_image, motion_type, x, y, noise_var=0.1
        )
        width_direction = np.array([-stretch_direction[1], stretch_direction[0]])

        # Determine the stretch size using a beta distribution (higher probability of smaller sizes)
        size_limits = (1, 30)
        stretch_size = np.random.beta(2, 8, 1)[0]
        stretch_size = (
            (size_limits[1] - size_limits[0]) * stretch_size + size_limits[0]
        ).astype(int)

        width_size = np.random.beta(2, 4 * size_limits[1] // stretch_size, 1)[0]
        width_size = (
            (size_limits[1] / 2 - size_limits[0] / 2) * width_size + size_limits[0] / 2
        ).astype(int)

        # Stretch the pixel along the stretch direction and add noise to the color
        new_val = val
        for i in range(1, stretch_size):
            new_pixel = (np.array([x, y]) + i * stretch_direction).astype(int)

            for j in range(-width_size // 2, width_size // 2 + 1):
                new_pixel = (new_pixel + j * width_direction).astype(int)
                if in_bounds(ref_image.shape, new_pixel[0], new_pixel[1]):
                    new_val[2:] = new_val[2:] + np.random.normal(0, 20, 1)
                    snow_mask[new_pixel[1], new_pixel[0], :] = np.clip(new_val, 0, 255)

    # Dilate the pixels
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    snow_mask = cv.dilate(snow_mask, kernel, iterations=1)

    # Convert snow to BGR color space
    snow_mask = cv.cvtColor(snow_mask, cv.COLOR_HSV2BGR)

    # Overlay snow mask with original image
    out_image = cv.addWeighted(ref_image, 1, snow_mask, 0.3, 0)

    # out_image = ref_image.copy()
    # non_zero_mask = snow_mask[:, :, 2] != 0
    # out_image[non_zero_mask] = snow_mask[non_zero_mask]

    return out_image


if __name__ == "__main__":
    # Read image and obtain snow image
    img = cv.imread("Input/UW_img.tif", cv.IMREAD_COLOR)
    img_snow = generate_snow(img)

    # Resize images
    scale = 0.5
    img_resized = cv.resize(img, (0, 0), fx=scale, fy=scale)
    img_snow_resized = cv.resize(img_snow, (0, 0), fx=scale, fy=scale)

    # Concatenate images
    concatenated = cv.hconcat([img_resized, img_snow_resized])

    # Show images
    cv.imshow("Original vs Snow", concatenated)
    cv.waitKey(0)
    cv.destroyAllWindows()
