"""
# Script for marine snow segmentation by transfer learning from SUIM-Net
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import numpy as np
from os.path import join, exists
from keras.models import Model
from keras.optimizers.legacy import Adam
from keras.layers import Conv2D
from keras import callbacks

# local libs
from models.suim_net import SUIM_Net
from utils.data_utils import *

# Directory for dataset
train_dir = "data/train/"
val_dir = "data/val/"

# ckpt directory
ckpt_base_dir = "ckpt/original/"
ckpt_custom_dir = "ckpt/custom/"
base_ = "RSB"  # or 'RSB'

# Define name for model which will be loaded and the name for model that is stored 
if base_ == "RSB":
    im_res_ = (320, 240, 3)
    ckpt_name_base = "suimnet_rsb.hdf5"
    ckpt_name_custom = "custom_suimnet_rsb.hdf5"
else:
    im_res_ = (320, 256, 3)
    ckpt_name_base = "suimnet_vgg.hdf5"
    ckpt_name_custom = "custom_suimnet_vgg.hdf5"

base_model_path = join(ckpt_base_dir, ckpt_name_base)
custom_model_path = join(ckpt_custom_dir, ckpt_name_custom)

if not exists(ckpt_custom_dir):
    os.makedirs(ckpt_custom_dir)

# Define width and height for input and output
im_h, im_w = im_res_[1], im_res_[0]

# Initialize suimnet object
original_suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=5)

# Get the model
original_model = original_suimnet.model

# Print model summary
print(original_model.summary())

# Load the weights from the saved file
original_model.load_weights(base_model_path)

# Freeze the weights of the original model, as we will only train the custom top layer
original_model.trainable = False

# Number of classes for the custom top layer
n_classes = 1  # Marine snow (Everything else besides this class is background)

# Create custom top layer
custom_top = Conv2D(n_classes, (3, 3), padding="same", activation="sigmoid")

# Get the output tensor of the second-to-last layer
headless_model = original_model.layers[-2].output

# Connect the output tensor to your custom layer
custom_output = custom_top(headless_model)

# Create the modified model
custom_model = Model(
    inputs=original_model.inputs, outputs=custom_output, name="custom_model"
)

# "Freeze" the behavior of this model
custom_model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Low learning rate
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Print model summary
print(custom_model.summary())

# Note that we must make sure to pass training=False when calling the base model, so that it runs in inference mode,
# so that batchnorm statistics don't get updated even after we unfreeze the base model for fine-tuning. By default,
# training=false for all layers?

# Training parameters
batch_size = 8
num_epochs = 50

# Setup data augmentation parameters.
data_gen_args = dict(
    rotation_range=0.2,
    # width_shift_range=0.05,
    # height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    # brightness_range=[0.5, 1],
    horizontal_flip=True,
    fill_mode="nearest",
)

# Setup model checkpoint
model_checkpoint = callbacks.ModelCheckpoint(
    custom_model_path,
    monitor="loss",
    verbose=1,
    mode="auto",
    save_weights_only=True,
    save_best_only=True,
)
# Early stopping callback
early_stop_callback = callbacks.EarlyStopping(monitor="loss", patience=3)

# Create data generator, which gives tuples of images and masks
train_gen = dataGenerator(
    batch_size,  # batch_size
    train_dir,  # train-data dir
    "images",  # image_folder
    "masks",  # mask_folder
    data_gen_args,  # aug_dict
    image_color_mode="rgb",
    mask_color_mode="rgb",
    target_size=(im_res_[1], im_res_[0]),
)

val_gen = dataGenerator(
    batch_size,  # batch_size
    val_dir,  # train-data dir
    "images",  # image_folder
    "masks",  # mask_folder
    data_gen_args,  # aug_dict
    image_color_mode="rgb",
    mask_color_mode="rgb",
    target_size=(im_res_[1], im_res_[0]),
)

# Get a batch of images and masks from the generator (for visualisation)
image_batch, mask_batch = next(train_gen)

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(12, 6))

for i in range(4):
    # Display the original image
    axs[0, i].imshow(
        image_batch[i].squeeze(), cmap="gray"
    )  # Assuming images are grayscale
    axs[0, i].set_title(f"Image {i+1}")
    axs[0, i].axis("off")

for i in range(4):
    # Display the corresponding mask
    axs[1, i].imshow(
        mask_batch[i].squeeze(), cmap="gray"
    )  # Assuming masks are grayscale
    axs[1, i].set_title(f"Mask {i+1}")
    axs[1, i].axis("off")

plt.tight_layout()
plt.show()

# Train the model
custom_model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=num_epochs,
    callbacks=[early_stop_callback, model_checkpoint],
    validation_data=val_gen,
    validation_steps=10,
)
