import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import fnmatch
import splitfolders
from matplotlib import pyplot as plt


def dataGenerator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    aug_dict,
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
    target_size=(256, 256),
):
    # data generator function for driving the training
    image_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=None,
        save_prefix="image",
        seed=1,
    )
    # Modify the aug_dict for mask, since we must not change brightness
    # aug_dict["brightness_range"] = None

    # mask generator function for corresponding ground truth
    mask_datagen = ImageDataGenerator(**aug_dict)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=None,
        save_prefix="mask",
        seed=1,
    )
    # make pairs and return
    for img, mask in zip(image_generator, mask_generator):
        img, mask_indiv = ImgToBinary(img, mask)
        yield (img, mask_indiv)


def ImgToBinary(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    m = []
    for i in range(mask.shape[0]):
        # Perform OR operation across color channels for mask
        mask_reduced = np.logical_or.reduce(mask[i, :, :, :], axis=-1)
        # Store as (width, height, 1) array
        m.append(np.expand_dims(mask_reduced, axis=-1))

        # Visualize the original RGB image and the resulting monochrome image
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask[i, :, :, :])
        # plt.title("RGB Image")

        # plt.subplot(1, 2, 2)
        # plt.imshow(mask_reduced, cmap="gray")
        # plt.title("Monochrome Image (OR Operation)")
        # plt.waitforbuttonpress()

    m = np.array(m)

    return (img, m)


def getPaths(data_dir):
    # read image files from directory
    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.JPEG", "*.bmp"]
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if fnmatch.fnmatch(filename, pattern):
                    fname_ = os.path.join(d, filename)
                    image_paths.append(fname_)
    return image_paths
