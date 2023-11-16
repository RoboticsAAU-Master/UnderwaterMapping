import os
import random
import cv2 as cv
import numpy as np


# Function to get a list of corresponding image pairs
def get_image_pairs(folder_path):
    images = os.listdir(folder_path)
    pairs = {}
    for img in images:
        img_name, img_ext = os.path.splitext(img)
        pair_name = img_name.split("_")[1]  # Extracting common part of the name
        if pair_name not in pairs:
            pairs[pair_name] = []
        pairs[pair_name].append(img)
    return pairs


# Function to display random pair of images
def show_random_pair():
    pair_names = list(image_pairs.keys())
    random_pair_name = random.choice(pair_names)
    prediction_img_path = prediction_path + "/" + "image_" + random_pair_name + ".bmp"
    gt_img_path = gt_path + "/" + "mask_" + random_pair_name + ".png"
    original_img_path = original_path + "/" + "image_" + random_pair_name + ".png"

    prediction_img = cv.imread(prediction_img_path, cv.IMREAD_GRAYSCALE)
    gt_img = cv.imread(gt_img_path, cv.IMREAD_GRAYSCALE)
    gt_img = cv.resize(gt_img, (prediction_img.shape[1], prediction_img.shape[0]))
    original_img = cv.imread(original_img_path, cv.IMREAD_COLOR)
    original_img = cv.resize(
        original_img, (prediction_img.shape[1], prediction_img.shape[0])
    )

    # Calculate precision and recall
    ms_count = np.count_nonzero(prediction_img)
    gt_ms_count = np.count_nonzero(gt_img)
    tp_count = np.count_nonzero(cv.bitwise_and(prediction_img, gt_img))

    if gt_ms_count == 0:
        precision = float("nan")
    else:
        precision = tp_count / gt_ms_count

    if ms_count == 0:
        recall = float("nan")
    else:
        recall = tp_count / ms_count

    # Resize images to fit in the window if needed
    prediction_img = cv.resize(prediction_img, (300, 300))
    gt_img = cv.resize(gt_img, (300, 300))
    original_img = cv.resize(original_img, (300, 300))

    # Overlay precision and recall
    cv.putText(
        original_img,
        f"Precision: {precision:.2f}, Recall: {recall:.2f}",
        (10, 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    # Create a blank space as a separator
    spacing = (
        np.ones((original_img.shape[0], 20, 3), dtype=np.uint8) * 255
    )  # Horizontal spacing of 20 pixels (white color)

    # For concatenation, all images must have the same number of channels
    prediction_3channel = cv.cvtColor(prediction_img, cv.COLOR_GRAY2BGR)
    gt_3channel = cv.cvtColor(gt_img, cv.COLOR_GRAY2BGR)

    # Concatenate images
    all_images = np.concatenate(
        (original_img, spacing, gt_3channel, spacing, prediction_3channel), axis=1
    )

    cv.imshow(f"Results for {random_pair_name}.png", all_images)

    # Set positions for each image and spacing
    cv.moveWindow(f"Results for {random_pair_name}.png", 200, 200)

    k = cv.waitKey(0) & 0xFF
    cv.destroyAllWindows()
    if k == ord("q"):
        exit()


# Set seed for reproducibility
random.seed(1)

# Update these paths with your folder locations
prediction_path = "output/MS_vgg_21e"
gt_path = "data/test/masks"
original_path = "data/test/images"

# Get pairs of images from the folders
image_pairs = get_image_pairs(prediction_path)

# Display random pair of images when the script runs
while True:
    show_random_pair()
