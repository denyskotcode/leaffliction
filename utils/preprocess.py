"""
The single preprocessing path shared by train.py and predict.py.

A mismatch between how images are prepared at training time and at
prediction time silently destroys accuracy, and it is invisible in the
metrics because training still converges. The defence is structural:
there is exactly one function, and both programs import *this* one.

Its name and parameters are recorded in labels.json ("preprocess" and
"input_size") so a saved model always states how it expects to be fed.
"""

import cv2
import numpy as np

# (H, W) fed to the network. The dataset is 256x256; 128x128 keeps every
# lesion clearly visible while making CPU-only training roughly four
# times cheaper.
IMAGE_SIZE = (128, 128)

# ImageNet statistics, because the backbone is pretrained on ImageNet.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Written into labels.json; bump it if anything below changes so an old
# model can never be silently paired with new preprocessing.
PREPROCESS_NAME = "resize128_rgb_float_imagenet_norm_v1"


def preprocess(img_rgb):
    """
    RGB uint8 (H, W, 3) -> float32 (3, IMAGE_SIZE) ready for the model.

    Resize to IMAGE_SIZE, scale to [0, 1], normalise with the ImageNet
    statistics, then move channels first (what torch expects).
    """
    if not isinstance(img_rgb, np.ndarray) or img_rgb.ndim != 3:
        raise ValueError("preprocess expects an (H, W, 3) RGB ndarray")
    if img_rgb.shape[2] != 3:
        raise ValueError("preprocess expects 3 channels in RGB order")

    height, width = IMAGE_SIZE
    resized = cv2.resize(img_rgb, (width, height),
                         interpolation=cv2.INTER_AREA)
    scaled = resized.astype(np.float32) / 255.0
    normalised = (scaled - MEAN) / STD
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))
