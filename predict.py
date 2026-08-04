#!/usr/bin/env python3
"""
predict.py <image> -- Part 4 of Leaffliction: classify one leaf image.

Reads model.pt + labels.json out of learnings.zip, preprocesses the
image with the *same* utils.preprocess used during training, shows the
original next to Person B's transformed rendering, and prints the
predicted class.
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile

import matplotlib

# A backend must be chosen before pyplot is imported anywhere, and
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


def _load_from_zip(zip_path, temp_dir):
    """Extract model.pt and labels.json from the archive."""
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        missing = {"model.pt", "labels.json"} - names
        if missing:
            raise ValueError(
                f"{zip_path} is missing {', '.join(sorted(missing))}")
        archive.extract("model.pt", temp_dir)
        archive.extract("labels.json", temp_dir)
    return (os.path.join(temp_dir, "model.pt"),
            os.path.join(temp_dir, "labels.json"))


def _load_from_dir(directory):
    model_path = os.path.join(directory, "model.pt")
    labels_path = os.path.join(directory, "labels.json")
    for path in (model_path, labels_path):
        if not os.path.isfile(path):
            raise ValueError(f"missing {path}")
    return model_path, labels_path


def load_model(model_source, temp_dir):
    """
    Return (model, classes) from a learnings.zip or an unpacked folder.

    Only model.pt and labels.json are read -- metrics.json and the
    augmented images are training-time artifacts.
    """
    if os.path.isdir(model_source):
        model_path, labels_path = _load_from_dir(model_source)
    elif zipfile.is_zipfile(model_source):
        model_path, labels_path = _load_from_zip(model_source, temp_dir)
    else:
        raise ValueError(f"not a zip archive or directory: {model_source}")

    with open(labels_path, encoding="utf-8") as handle:
        labels = json.load(handle)
    classes = labels.get("classes")
    if not classes:
        raise ValueError("labels.json has no 'classes' list")

    saved = labels.get("preprocess")
    if saved and saved != PREPROCESS_NAME:
        print(f"warning: model was trained with preprocessing '{saved}' "
              f"but this build uses '{PREPROCESS_NAME}'", file=sys.stderr)

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
        probabilities = torch.softmax(model(batch), dim=1)[0]
    index = int(probabilities.argmax())
    return classes[index], float(probabilities[index])


def _transformed(img_rgb):
    """Person B's rendering; degrade to the original if unavailable."""
    try:
        from Transformation import transformed_for_display
        return transformed_for_display(img_rgb), "Transformed"
    except Exception as error:  # noqa: BLE001 - display is optional
        print(f"warning: transformation unavailable ({error})",
              file=sys.stderr)
        return img_rgb, "Transformed (unavailable)"


def show(img_rgb, label, confidence, save_to=None):
    """Display the original and the transformed image side by side."""
    transformed, subtitle = _transformed(img_rgb)

    figure, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(np.asarray(transformed).astype(np.uint8))
    axes[1].set_title(subtitle)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Class predicted : {label}   ({confidence:.1%})",
                    fontsize=14)
    figure.tight_layout()

    if save_to:
        figure.savefig(save_to, dpi=120)
        print(f"saved figure to {save_to}")
    elif matplotlib.get_backend().lower() == "agg":
        fallback = "prediction.png"
        figure.savefig(fallback, dpi=120)
        print(f"no display available; saved figure to {fallback}")
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
