"""
The one preprocessing path shared by train.py and predict.py.

Preparing images one way while training and another way while predicting
destroys accuracy silently: training still converges, and the metrics
look fine. The defence here is structural -- a single function, imported
by both programs, with its name recorded in labels.json so a saved model
always states how it expects to be fed.
"""

import cv2
import numpy as np

# (H, W) fed to the network. The dataset is 256x256; 128x128 keeps every
# lesion clearly visible while making CPU-only training about four times
# cheaper.
IMAGE_SIZE = (128, 128)

# ImageNet statistics, because the backbone is pretrained on ImageNet.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Stored in labels.json; bump it whenever anything below changes, so an
# old model can never be silently paired with new preprocessing.
PREPROCESS_NAME = "resize128_rgb_float_imagenet_norm_v1"


def preprocess(img_rgb):
    """
    RGB uint8 (H, W, 3) -> float32 (3, *IMAGE_SIZE), ready for the model.

    Resize, scale to [0, 1], normalise with the ImageNet statistics, then
    put the channels first, which is the layout torch expects.
    """
    if not isinstance(img_rgb, np.ndarray) or img_rgb.ndim != 3:
        raise ValueError("preprocess expects an (H, W, 3) RGB ndarray")
    if img_rgb.shape[2] != 3:
        raise ValueError("preprocess expects 3 channels in RGB order")

    height, width = IMAGE_SIZE
    resized = cv2.resize(img_rgb, (width, height),
                         interpolation=cv2.INTER_AREA)
    unit = resized.astype(np.float32) / 255.0
    channels_first = ((unit - MEAN) / STD).transpose(2, 0, 1)
    return np.ascontiguousarray(channels_first)
