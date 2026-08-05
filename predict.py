#!/usr/bin/env python3
"""
predict.py <image> -- Part 4 of Leaffliction: classify one leaf image.

Reads model.pt and labels.json out of learnings.zip, prepares the image
with the *same* utils.preprocess used during training, prints the
predicted class and shows the original beside Part 3's transformed
rendering.
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile

import matplotlib

# The backend has to be picked before pyplot is imported anywhere, and
# importing Transformation pulls pyplot in. Fall back to the file-only
# backend when there is no display (ssh, CI, --save-to).
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import models  # noqa: E402

from utils.dataset import load_image  # noqa: E402
from utils.preprocess import PREPROCESS_NAME, preprocess  # noqa: E402

DEFAULT_MODEL = "learnings.zip"
NEEDED = ("model.pt", "labels.json")


def locate_artifacts(model_source, temp_dir):
    """
    Paths to model.pt and labels.json, from a folder or a learnings.zip.

    Only those two are needed: metrics.json and the augmented images are
    training-time artifacts and are never read here.
    """
    if os.path.isdir(model_source):
        paths = [os.path.join(model_source, name) for name in NEEDED]
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise ValueError(f"missing {', '.join(missing)}")
        return paths

    if not zipfile.is_zipfile(model_source):
        raise ValueError(f"not a zip archive or directory: {model_source}")

    with zipfile.ZipFile(model_source) as archive:
        missing = [name for name in NEEDED if name not in archive.namelist()]
        if missing:
            raise ValueError(
                f"{model_source} is missing {', '.join(missing)}")
        for name in NEEDED:
            archive.extract(name, temp_dir)
    return [os.path.join(temp_dir, name) for name in NEEDED]


def load_model(model_source, temp_dir):
    """Return (model, classes) ready for :func:`predict`."""
    model_path, labels_path = locate_artifacts(model_source, temp_dir)

    with open(labels_path, encoding="utf-8") as handle:
        labels = json.load(handle)
    classes = labels.get("classes")
    if not classes:
        raise ValueError("labels.json has no 'classes' list")

    trained_with = labels.get("preprocess")
    if trained_with and trained_with != PREPROCESS_NAME:
        print(f"warning: model was trained with preprocessing "
              f"'{trained_with}' but this build uses '{PREPROCESS_NAME}'",
              file=sys.stderr)

    checkpoint = torch.load(model_path, map_location="cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def predict(model, classes, img_rgb):
    """Return (label, confidence) for one RGB uint8 image."""
    batch = torch.from_numpy(preprocess(img_rgb)).unsqueeze(0)
    with torch.no_grad():
        scores = torch.softmax(model(batch), dim=1)[0]
    best = int(scores.argmax())
    return classes[best], float(scores[best])


def transformed(img_rgb):
    """Part 3's rendering, degrading to the original if it is missing."""
    try:
        from Transformation import transformed_for_display
        return transformed_for_display(img_rgb), "Transformed"
    except Exception as error:  # noqa: BLE001 - the picture is optional
        print(f"warning: transformation unavailable ({error})",
              file=sys.stderr)
        return img_rgb, "Transformed (unavailable)"


def show(img_rgb, label, confidence, save_to=None):
    """Display the original and the transformed image side by side."""
    picture, subtitle = transformed(img_rgb)

    figure, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(np.asarray(picture).astype(np.uint8))
    axes[1].set_title(subtitle)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Class predicted : {label}   ({confidence:.1%})",
                    fontsize=14)
    figure.tight_layout()

    headless = matplotlib.get_backend().lower() == "agg"
    if save_to or headless:
        destination = save_to or "prediction.png"
        figure.savefig(destination, dpi=120)
        prefix = "" if save_to else "no display available; "
        print(f"{prefix}saved figure to {destination}")
    else:
        plt.show()
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict the disease class of a single leaf image.")
    parser.add_argument("image", help="path to a leaf image")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"learnings.zip or folder "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--save-to", default=None,
                        help="write the figure to this file instead of "
                             "opening a window")
    parser.add_argument("--no-display", action="store_true",
                        help="print the prediction only")
    return parser


def main():
    args = build_parser().parse_args()

    if not os.path.isfile(args.image):
        print(f"error: no such file: {args.image}", file=sys.stderr)
        return 1
    if not os.path.exists(args.model):
        print(f"error: no model at {args.model}; run train.py first",
              file=sys.stderr)
        return 1

    try:
        img_rgb = load_image(args.image)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            model, classes = load_model(args.model, temp_dir)
        except (ValueError, OSError, KeyError) as error:
            print(f"error: cannot load model: {error}", file=sys.stderr)
            return 1
        label, confidence = predict(model, classes, img_rgb)

    print(f"Class predicted : {label}")
    print(f"Confidence      : {confidence:.2%}")

    if not args.no_display:
        show(img_rgb, label, confidence, args.save_to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
