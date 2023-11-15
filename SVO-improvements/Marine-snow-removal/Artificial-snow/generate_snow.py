import cv2 as cv
import numpy as np
from scipy.stats import qmc
from enum import Enum
import math

### PARAMETERS ###
NUM_PARTICLES_RANGE = (0, 100)  # range:[0, inf)
SAMPLING_METHOD = "random"  # "halton" or "random
SPEED_RANGE = (-2, 6)  # range:[1, inf)
SATURATION_RANGE = (40, 150)  # range:[0, 255]
VALUE_RANGE = (100, 255)  # range:[0, 255]
MEDIAN_BLUR_KERNEL_SIZE = 5  # range:[1, inf), (odd)
MEDIAN_BLUR_SATURATION_THRESH = 50  # range:[0, 255]
MEDIAN_BLUR_VALUE_THRESH = 240  # range:[0, 255]
SIGMA_SATURATION_THRESH = 5  # range:[0, 255]
SIGMA_VALUE_THRESH = 5  # range:[0, 255]
MOTION_DIRECTION_NOISE_VAR_MAX = 0.2  # range:[0, 1]
STRETCH_SIZE_RANGE = (1, 60)  # range:[1, inf)
WIDTH_SIZE_RANGE = (1, 30)  # range:[1, inf)
SNOW_PATCH_VALUE_NOISE = 20  # range:[0, 255]
OVERLAY_PERCENTAGE_RANGE = (0.1, 0.4)  # range:[0, 1]
##################


class Motion_Direction(Enum):
    FOWARD_BACKWARD = 0
    UP_DOWN = 1
    LEFT_RIGHT = 2


def in_bounds(shape, x, y):
    return x >= 0 and x < shape[1] and y >= 0 and y < shape[0]


def get_background(image_hsv, channel):
    # Get hue channel and flatten
    img_channel = image_hsv[:, :, channel]
    img_channel = img_channel.flatten()

    # Calculate histogram
    hist = np.histogram(img_channel, bins=180, range=(0, 180))

    # Obtain the background hue
    bg_val = np.argmax(hist[0])

    return bg_val


def get_std(image_hsv):
    hsv_32 = np.float32(image_hsv)

    # Calculate mean of image
    mu = cv.blur(hsv_32, (7, 7), borderType=cv.BORDER_CONSTANT)

    # Calculate mean of image squared
    mu2 = cv.blur(np.multiply(hsv_32, hsv_32), (7, 7), borderType=cv.BORDER_CONSTANT)

    # Calculate variance of image
    sigma = np.sqrt(np.subtract(mu2, np.multiply(mu, mu)))

    return sigma


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
    stretch_direction = stretch_direction / np.linalg.norm(stretch_direction)

    return stretch_direction


def generate_snow(ref_image, mask=False):
    # Convert to hsv color space
    hsv = cv.cvtColor(ref_image, cv.COLOR_BGR2HSV)

    background_val = get_background(hsv, 2)
    median_blur = cv.medianBlur(hsv, MEDIAN_BLUR_KERNEL_SIZE)
    sigma = get_std(hsv)

    # Calculate the number of snow particles
    num_snow = np.random.randint(NUM_PARTICLES_RANGE[0], NUM_PARTICLES_RANGE[1] + 1)

    # Generate random samples
    samples_coords = get_random_samples(ref_image, num_snow, method=SAMPLING_METHOD)

    # Concatenate images
    # test = cv.resize(hsv, (0, 0), fx=0.4, fy=0.4)
    # hsv_concat = cv.hconcat([test[:, :, 0], test[:, :, 1], test[:, :, 2]])
    # cv.imshow("HSV", hsv_concat)
    # cv.waitKey(0)

    # Shape and size of snow particles
    snow_mask = np.zeros_like(hsv)
    weighting_mask = np.zeros_like(hsv).astype(np.float32)

    centre_type = np.random.randint(0, 8 + 1)  # Define the centre type
    speed = np.random.randint(
        SPEED_RANGE[0], SPEED_RANGE[1] + 1
    )  # Define the speed of the snow particles
    if speed < 1:
        speed = 1
    for x, y in samples_coords:
        # Color of snow particles
        val = np.array(
            [
                hsv[y, x, 0],
                np.random.randint(SATURATION_RANGE[0], SATURATION_RANGE[1]),
                np.random.randint(VALUE_RANGE[0], VALUE_RANGE[1]),
            ]
        )

        # if (
        #     median_blur[y, x, 1] < MEDIAN_BLUR_SATURATION_THRESH
        #     and median_blur[y, x, 2] > MEDIAN_BLUR_VALUE_THRESH
        # ) or (
        #     (
        #         sigma[y, x, 1] > SIGMA_SATURATION_THRESH
        #         or sigma[y, x, 2] > SIGMA_VALUE_THRESH
        #     )
        # ):
        #     s_thresh = (hsv[:, :, 1] < 50).astype(np.uint8)
        #     v_thresh = (hsv[:, :, 2] > 240).astype(np.uint8)
        #     temp = cv.bitwise_and(s_thresh, v_thresh)
        #     temp = cv.bitwise_or(temp, (sigma[:, :, 1] > 5).astype(np.uint8))
        #     temp = cv.bitwise_or(temp, (sigma[:, :, 2] > 5).astype(np.uint8))
        #     cv.imshow("temp", temp * 255)
        #     cv.waitKey(0)
        #     continue

        if (
            median_blur[y, x, 1] < MEDIAN_BLUR_SATURATION_THRESH
            and median_blur[y, x, 2] > MEDIAN_BLUR_VALUE_THRESH
        ):
            continue

        # snow_mask[y, x, :] = val

        # Get direction along which we stretch the marine snow particle
        stretch_direction = get_motion_direction(
            ref_image,
            centre_type,
            x,
            y,
            noise_var=MOTION_DIRECTION_NOISE_VAR_MAX / speed,
        )
        width_direction = np.array([-stretch_direction[1], stretch_direction[0]])

        # Determine the stretch size using a beta distribution (higher probability of smaller sizes)
        stretch_size = np.random.beta(2, 2 * 6 // speed, 1)[0]
        stretch_size = (
            (STRETCH_SIZE_RANGE[1] - STRETCH_SIZE_RANGE[0]) * stretch_size
            + STRETCH_SIZE_RANGE[0]
        ).astype(int)
        # Determine the width size using a beta distribution based on the stretch size
        width_size = np.random.beta(
            2, speed * 3 * STRETCH_SIZE_RANGE[1] // stretch_size, 1
        )[0]
        width_size = (
            (WIDTH_SIZE_RANGE[1] - WIDTH_SIZE_RANGE[0]) * width_size
            + WIDTH_SIZE_RANGE[0]
        ).astype(int)

        # Stretch the pixel along the stretch direction and width direction and add noise to the value-component
        add_weight = np.random.beta(2, 6)
        end_shape_width = (np.random.beta(2, 4) * width_size) + 1

        for i in range((-stretch_size // 2) + 1, (stretch_size // 2) + 1):
            new_val = val
            new_pixel = (np.array([x, y]) + i * stretch_direction).astype(int)

            shape_width = (
                width_size * (1 - (abs(i) / (stretch_size // 2)))
                + end_shape_width * (abs(i) / (stretch_size // 2))
            ).astype(int)
            # shape_width = width_size + (1 - (i / (stretch_size // 2)))
            for j in range((-shape_width // 2) + 1, (shape_width // 2) + 1):
                new_pixel = (new_pixel + j * width_direction).astype(int)
                if in_bounds(ref_image.shape, new_pixel[0], new_pixel[1]):
                    # Add noise to the pixel value based on the previous pixel value
                    new_val[2] = new_val[2] + np.random.normal(
                        0, SNOW_PATCH_VALUE_NOISE, 1
                    )
                    snow_mask[new_pixel[1], new_pixel[0], :] = np.clip(new_val, 0, 255)
                    weighting_mask[new_pixel[1], new_pixel[0], :] = add_weight

    # Dilate the pixels with an elliptical kernel inversely proportional to the speed
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3 + 2 * (3 // speed), 3))
    snow_mask = cv.dilate(snow_mask, kernel, iterations=1)
    weighting_mask = cv.dilate(weighting_mask, kernel, iterations=1)

    _, non_zero_mask = cv.threshold(snow_mask[:, :, 2], 0, 255, cv.THRESH_BINARY)

    # Blur the pixels with a Gaussian kernel proportional to the speed (motion blur)
    # snow_mask = cv.GaussianBlur(
    #     snow_mask, (3 + 2 * (speed // 2), 3 + 2 * (speed // 2)), 0
    # )
    snow_mask = cv.blur(
        snow_mask,
        (3 + 2 * (speed // 2), 3 + 2 * (speed // 2)),
        borderType=cv.BORDER_CONSTANT,
    )
    # Convert snow to BGR color space
    snow_mask = cv.cvtColor(snow_mask, cv.COLOR_HSV2BGR)

    # Overlay snow mask with original image
    overlay_percentage = (background_val / 255) * (
        OVERLAY_PERCENTAGE_RANGE[1] - OVERLAY_PERCENTAGE_RANGE[0]
    ) + OVERLAY_PERCENTAGE_RANGE[0]
    weighting_mask = weighting_mask + overlay_percentage
    # out_image = cv.addWeighted(ref_image, 1, snow_mask, overlay_percentage, 0)
    out_image = cv.add(
        ref_image,
        cv.multiply(snow_mask, weighting_mask, dtype=cv.CV_32F),
        dtype=cv.CV_8U,
    )

    # non_zero_mask = snow_mask[:, :, 2] != 0
    # out_image = ref_image.copy()
    # out_image[non_zero_mask] = snow_mask[non_zero_mask]

    if mask:
        return out_image, non_zero_mask
    else:
        return out_image


if __name__ == "__main__":
    # Resize factor
    scale = 0.7

    # Read image
    img = cv.imread(
        r"C:\Users\danie\Downloads\dataset1\16233155503747041.tiff",
        cv.IMREAD_COLOR,
    )
    img_resized = cv.resize(img, (0, 0), fx=scale, fy=scale)

    # Resize images
    while True:
        img_snow, mask = generate_snow(img, mask=True)
        img_snow_resized = cv.resize(img_snow, (0, 0), fx=scale, fy=scale)

        # Concatenate images
        concatenated = cv.hconcat([img_resized, img_snow_resized])

        # Show images
        cv.imshow("Original vs Snow", concatenated)
        cv.imshow("Mask", mask)
        if cv.waitKey(0) == ord("q"):
            break

    cv.destroyAllWindows()
