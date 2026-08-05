"""
Dataset helpers shared by every part of the project.

OWNER: Person A. This is a contract-faithful implementation written by
Person C to stay unblocked (see 00_TEAM_BRIEF.md, "Day-1 shared
contracts"). When A ships the real module it should drop in verbatim:
the five public functions below keep exactly the documented signatures.

In-memory image format, everywhere in this project:
    numpy ndarray, shape (H, W, 3), dtype uint8, RGB order.
OpenCV is BGR, so the BGR<->RGB flip happens here and nowhere else.
"""

import os
import random

import cv2
import numpy as np

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _is_image(filename):
    return filename.lower().endswith(VALID_EXTENSIONS)


def list_images(root, include_root=False):
    """
    Walk ``root`` and return [(path, label), ...] sorted by path.

    The label is the name of the directory holding the image, e.g.
    ``Apple_healthy``. Labels live in the folder name and nowhere else.

    Images lying directly in ``root`` are skipped by default, because
    for a data set root they are strays with no class directory to name
    them. ``include_root=True`` instead labels them with ``root``'s own
    name, which is what Part 1 needs when it is pointed straight at a
    single class directory. Training keeps the default, so the split
    every accuracy claim rests on is unaffected.
    """
    if not os.path.isdir(root):
        raise NotADirectoryError(f"not a directory: {root}")

    root_label = os.path.basename(os.path.abspath(root))
    items = []
    for dirpath, _dirnames, filenames in os.walk(root):
        label = os.path.basename(dirpath)
        if os.path.abspath(dirpath) == os.path.abspath(root):
            if not include_root:
                continue
            label = root_label
        for filename in filenames:
            if _is_image(filename):
                items.append((os.path.join(dirpath, filename), label))
    return sorted(items)


def load_image(path):
    """Read an image from disk as RGB uint8 (H, W, 3)."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image(img, path):
    """Write an RGB uint8 (H, W, 3) image to ``path``."""
    if not isinstance(img, np.ndarray) or img.ndim != 3:
        raise ValueError("save_image expects an (H, W, 3) ndarray")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, img_bgr):
        raise IOError(f"cannot write image: {path}")


def class_counts(root, include_root=False):
    """Return {label: number_of_images} for every class under ``root``."""
    counts = {}
    for _path, label in list_images(root, include_root=include_root):
        counts[label] = counts.get(label, 0) + 1
    return counts


def split_dataset(root, val_ratio=0.2, seed=42):
    """
    Stratified, seeded train/validation split.

    Every class contributes the same *proportion* of its images to the
    validation set, so rare classes stay represented. The seed makes the
    split reproducible, which is what makes the >=90% accuracy claim
    defensible at evaluation time.

    Returns (train_items, val_items), each a list of (path, label).
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    by_label = {}
    for path, label in list_images(root):
        by_label.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    train_items, val_items = [], []
    for label in sorted(by_label):
        items = sorted(by_label[label])
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        # Never let a class vanish from either side of the split.
        n_val = max(1, min(n_val, len(items) - 1)) if len(items) > 1 else 0
        val_items.extend(items[:n_val])
        train_items.extend(items[n_val:])

    return train_items, val_items
