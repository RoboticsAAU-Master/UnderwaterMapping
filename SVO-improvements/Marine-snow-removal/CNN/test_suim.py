import os
import numpy as np
from os.path import join, exists
import ntpath
from PIL import Image
from keras.models import Model
from keras.layers import Conv2D

# Local libs
from models.suim_net import SUIM_Net
from utils.data_utils import getPaths

# Directory with test data
test_dir = "data/test/images/"

# Directory to save the output
samples_dir = "output/"
BG_dir = samples_dir + "BG/"  # Background
MS_dir = samples_dir + "MS/"  # Marine Snow

if not exists(samples_dir):
    os.makedirs(samples_dir)

if not exists(BG_dir):
    os.makedirs(BG_dir)

if not exists(MS_dir):
    os.makedirs(MS_dir)

## input/output shapes
base_ = "VGG"  # or 'VGG'
if base_ == "RSB":
    im_res_ = (320, 240, 3)
    ckpt_name = "custom_suimnet_rsb5.hdf5"
else:
    im_res_ = (320, 256, 3)
    ckpt_name = "custom_suimnet_vgg5.hdf5"

im_h, im_w = im_res_[1], im_res_[0]

original_suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=5)

original_model = original_suimnet.model

# Get the model
original_model = original_suimnet.model

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

custom_model.load_weights(join("ckpt/custom/", ckpt_name))


def testGenerator():
    # test all images in the directory
    assert exists(test_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(test_dir):
        # read and scale inputs
        img = Image.open(p).resize((im_w, im_h))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        # inference
        out_img = custom_model.predict(img)
        # thresholding
        out_img[out_img > 0.5] = 1.0
        out_img[out_img <= 0.5] = 0.0
        print("tested: {0}".format(p))
        # get filename
        img_name = ntpath.basename(p).split(".")[0] + ".bmp"
        # save individual output masks
        BGs = np.reshape(out_img[0, :, :, 0], (im_h, im_w))
        MSs = np.reshape(out_img[0, :, :, 1], (im_h, im_w))
        Image.fromarray(np.uint8(BGs * 255.0)).save(BG_dir + img_name)
        Image.fromarray(np.uint8(MSs * 255.0)).save(MS_dir + img_name)


# test images
testGenerator()
