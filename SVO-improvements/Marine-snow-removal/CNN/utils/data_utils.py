import numpy as np
import itertools as it
from keras.preprocessing.image import ImageDataGenerator


def trainDataGenerator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    aug_dict,
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
    target_size=(256, 256),
    sal=False,
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
    for img, mask in it.izip(image_generator, mask_generator):
        img, mask_indiv = ImgToBinary(img, mask, sal)
        yield (img, mask_indiv)


def ImgToBinary(img, mask, sal):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)
