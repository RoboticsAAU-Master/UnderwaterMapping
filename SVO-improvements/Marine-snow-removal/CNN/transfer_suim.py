"""
# Script for marine snow segmentation by transfer learning from SUIM-Net
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import numpy as np
from os.path import join, exists
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras import callbacks

# local libs
from models.suim_net import SUIM_Net
from utils.data_utils import *

# Directory for dataset
train_dir = "data/"

# ckpt directory
ckpt_dir = "ckpt/custom/"
base_ = "VGG"  # or 'RSB'

# Define model which will be loaded
if base_ == "RSB":
    im_res_ = (320, 240, 3)
    ckpt_name = "custom_suimnet_rsb.hdf5"
else:
    im_res_ = (320, 256, 3)
    ckpt_name = "custom_suimnet_vgg.hdf5"

model_ckpt_name = join(ckpt_dir, ckpt_name)

if not exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Directory to save the output
samples_dir = "data/test/output/"
BG_dir = samples_dir + "BG/"  # Background
MS_dir = samples_dir + "MS/"  # Marine Snow
if not exists(samples_dir):
    os.makedirs(samples_dir)
if not exists(BG_dir):
    os.makedirs(BG_dir)
if not exists(MS_dir):
    os.makedirs(MS_dir)

# input/output shapes
base_ = "RSB"  # or 'VGG'
if base_ == "RSB":
    im_res_ = (320, 240, 3)
    ckpt_name = "suimnet_rsb5.hdf5"
else:
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg5.hdf5"

# Define width and height for input and output
im_h, im_w = im_res_[1], im_res_[0]

# Initialize suimnet object
original_suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=5)

# Get the model
original_model = original_suimnet.model

# Print model summary
print(original_model.summary())

# Load the weights from the saved file
original_model.load_weights(join("ckpt/original/", ckpt_name))

# Freeze the weights of the original model, as we will only train the custom top layer
original_model.trainable = False

# Number of classes for the custom top layer
n_classes = 2  # Marine snow and background

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
    optimizer=Adam(learning_rate=1e-5),  # Low learning rate
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print(custom_model.summary())

# Note that we must make sure to pass training=False when calling the base model, so that it runs in inference mode,
# so that batchnorm statistics don't get updated even after we unfreeze the base model for fine-tuning. By default,
# training=false for all layers?

# Training parameters
batch_size = 8
num_epochs = 50

# Setup data augmentation parameters. Consider adding shear, contrast, etc.
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Setup model checkpoint
model_checkpoint = callbacks.ModelCheckpoint(
    model_ckpt_name,
    monitor="loss",
    verbose=1,
    mode="auto",
    save_weights_only=True,
    save_best_only=True,
)

# Create data generator, which gives tuples of images and masks
train_gen = trainDataGenerator(
    batch_size,  # batch_size
    train_dir,  # train-data dir
    "images",  # image_folder
    "masks",  # mask_folder
    data_gen_args,  # aug_dict
    image_color_mode="rgb",
    mask_color_mode="rgb",
    target_size=(im_res_[1], im_res_[0]),
)

# Train the model
custom_model.fit(
    train_gen,
    steps_per_epoch=5000,
    epochs=num_epochs,
    callbacks=[model_checkpoint],
    verbose=1,
)
