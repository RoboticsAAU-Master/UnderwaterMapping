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


def get_background_hue(image_hsv):
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


# def get_motion_direction(ref_image, motion_type, main_direc, x, y, noise_var=0.0):
#     if motion_type == Motion_Direction.FOWARD_BACKWARD.value:
#         centre = np.array([ref_image.shape[1] // 2, ref_image.shape[0] // 2])
#         stretch_direction = np.array([x, y]) - centre

#     elif motion_type == Motion_Direction.UP_DOWN.value:
#         if main_direc:
#             top = np.array([ref_image.shape[1] // 2, -ref_image.shape[0]])
#         else:
#             top = np.array([ref_image.shape[1] // 2, 2 * ref_image.shape[0]])
#         stretch_direction = np.array([x, y]) - top

#     elif motion_type == Motion_Direction.LEFT_RIGHT.value:
#         if main_direc:
#             side = np.array([-ref_image.shape[1], ref_image.shape[0] // 2])
#         else:
#             side = np.array([2 * ref_image.shape[1], ref_image.shape[0] // 2])

#         side = np.array([-ref_image.shape[1], ref_image.shape[0] // 2])
#         stretch_direction = np.array([x, y]) - side

#     else:
#         raise ValueError("Invalid motion direction")

#     if np.linalg.norm(stretch_direction) != 0:
#         stretch_direction = stretch_direction / np.linalg.norm(stretch_direction)

#     stretch_direction = stretch_direction + np.random.normal(0, noise_var, 2)
#     return stretch_direction


def get_motion_direction(ref_image, centre_type, x, y, noise_var=0.0):
    # Generate x and y values for the centre types
    x_vals = np.linspace(-ref_image.shape[1], 2 * ref_image.shape[1], 4)
    y_vals = np.linspace(-ref_image.shape[0], 2 * ref_image.shape[0], 4)

    # Define the centre of the stretch as a point on a "+" shape on the image
    if centre_type < 4:  # Along centre x-axis
        centre = np.array([x_vals[centre_type], ref_image.shape[0] // 2])
    elif centre_type < 8:  # Along centre y-axis
        centre = np.array([ref_image.shape[1] // 2, y_vals[centre_type - 4]])
    elif centre_type == 8:  # Centre of image
        centre = np.array([ref_image.shape[1] // 2, ref_image.shape[0] // 2])
    else:
        raise ValueError("Invalid centre type")

    # Obtain the stretch direction
    stretch_direction = np.array([x, y]) - centre

    # Normalise the stretch direction
    if np.linalg.norm(stretch_direction) != 0:
        stretch_direction = stretch_direction / np.linalg.norm(stretch_direction)

    # Add noise to the stretch direction
    stretch_direction = stretch_direction + np.random.normal(0, noise_var, 2)
    return stretch_direction


def generate_snow(ref_image):
    # Convert to hsv color space
    hsv = cv.cvtColor(ref_image, cv.COLOR_BGR2HSV)

    # Calculate the number of snow particles
    num_range = (100, 300)
    num_snow = np.random.randint(num_range[0], num_range[1] + 1)

    # Generate Halton sequence
    samples_coords = get_random_samples(ref_image, num_snow, method="random")

    # Shape and size of snow particles
    snow_mask = np.zeros_like(hsv)
    centre_type = np.random.randint(0, 8 + 1)  # Define the centre type
    speed = np.random.randint(1, 6 + 1)  # Define the speed of the snow particles
    for x, y in samples_coords:
        # Color of snow particles
        val = np.array(
            [hsv[y, x, 0], np.random.randint(0, 80), np.random.randint(50, 255)]
        )
        snow_mask[y, x, :] = val

        # Get direction along which we stretch the marine snow particle
        stretch_direction = get_motion_direction(
            ref_image, centre_type, x, y, noise_var=0.2 / speed
        )
        width_direction = np.array([-stretch_direction[1], stretch_direction[0]])

        # Determine the stretch size using a beta distribution (higher probability of smaller sizes)
        size_limits = (1, 30)
        stretch_size = np.random.beta(2, 6 // speed, 1)[0]
        stretch_size = (
            (size_limits[1] - size_limits[0]) * stretch_size + size_limits[0]
        ).astype(int)
        # Determine the width size using a beta distribution based on the stretch size
        width_size = np.random.beta(2, speed * 2 * size_limits[1] // stretch_size, 1)[0]
        width_size = (
            (size_limits[1] / 2 - size_limits[0] / 2) * width_size + size_limits[0] / 2
        ).astype(int)

        # Stretch the pixel along the stretch direction and width direction and add noise to the value-component
        for i in range(1, stretch_size):
            new_val = val
            new_pixel = (np.array([x, y]) + i * stretch_direction).astype(int)

            for j in range(-width_size // 2, width_size // 2 + 1):
                new_pixel = (new_pixel + j * width_direction).astype(int)
                if in_bounds(ref_image.shape, new_pixel[0], new_pixel[1]):
                    # Add noise to the pixel value based on the previous pixel value
                    new_val[2:] = new_val[2:] + np.random.normal(0, 20, 1)
                    snow_mask[new_pixel[1], new_pixel[0], :] = np.clip(new_val, 0, 255)

    # Dilate the pixels with an elliptical kernel inversely proportional to the speed
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3 + 2 * (3 // speed), 3))
    snow_mask = cv.dilate(snow_mask, kernel, iterations=1)

    # Blur the pixels with a Gaussian kernel proportional to the speed (motion blur)
    snow_mask = cv.GaussianBlur(
        snow_mask, (3 + 2 * (speed // 2), 3 + 2 * (speed // 2)), 0
    )

    # Convert snow to BGR color space
    snow_mask = cv.cvtColor(snow_mask, cv.COLOR_HSV2BGR)

    # Overlay snow mask with original image
    out_image = cv.addWeighted(ref_image, 1, snow_mask, 0.3, 0)

    # out_image = ref_image.copy()
    # non_zero_mask = snow_mask[:, :, 2] != 0
    # out_image[non_zero_mask] = snow_mask[non_zero_mask]

    return out_image


if __name__ == "__main__":
    # Resize factor
    scale = 0.5

    # Read image
    img = cv.imread("Input/UW_img.tif", cv.IMREAD_COLOR)
    img_resized = cv.resize(img, (0, 0), fx=scale, fy=scale)

    # Resize images
    while True:
        img_snow = generate_snow(img)
        img_snow_resized = cv.resize(img_snow, (0, 0), fx=scale, fy=scale)

        # Concatenate images
        concatenated = cv.hconcat([img_resized, img_snow_resized])

        # Show images
        cv.imshow("Original vs Snow", concatenated)
        if cv.waitKey(0) == ord("q"):
            break

    cv.destroyAllWindows()
