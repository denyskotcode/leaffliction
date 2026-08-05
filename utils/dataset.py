"""
Dataset helpers shared by every part of the project.

In-memory image format, everywhere in this project:
    numpy ndarray, shape (H, W, 3), dtype uint8, RGB order.
OpenCV works in BGR, so the BGR<->RGB flip happens in this module and
nowhere else -- one conversion in, one conversion out.
"""

import os
import random

import cv2
import numpy as np

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _images_in(folder):
    """Names of the image files sitting directly in ``folder``."""
    return [name for name in os.listdir(folder)
            if name.lower().endswith(VALID_EXTENSIONS)
            and os.path.isfile(os.path.join(folder, name))]


def list_images(root, include_root=False):
    """
    Walk ``root`` and return [(path, label), ...] sorted by path.

    The label is the name of the directory holding the image, e.g.
    ``Apple_healthy``: labels live in the folder name and nowhere else.

    Images lying loose in ``root`` are skipped by default -- under a
    dataset root they are strays with no class directory to name them.
    ``include_root=True`` labels them with ``root``'s own name instead,
    which is what Part 1 needs when pointed straight at a single class
    directory. Training keeps the default, so the split every accuracy
    claim rests on is unaffected.
    """
    if not os.path.isdir(root):
        raise NotADirectoryError(f"not a directory: {root}")

    root_path = os.path.abspath(root)
    items = []
    for folder, _subfolders, _files in os.walk(root):
        at_root = os.path.abspath(folder) == root_path
        if at_root and not include_root:
            continue
        label = os.path.basename(root_path if at_root else folder)
        items += [(os.path.join(folder, name), label)
                  for name in _images_in(folder)]
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
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, img_bgr):
        raise IOError(f"cannot write image: {path}")


def class_counts(root, include_root=False):
    """Return {label: number_of_images} for every class under ``root``."""
    counts = {}
    for _path, label in list_images(root, include_root=include_root):
        counts[label] = counts.get(label, 0) + 1
    return counts


def group_by_label(items):
    """[(path, label), ...] -> {label: [item, ...]}, insertion ordered."""
    groups = {}
    for path, label in items:
        groups.setdefault(label, []).append((path, label))
    return groups


def split_dataset(root, val_ratio=0.2, seed=42):
    """
    Stratified, seeded train/validation split.

    Every class hands over the same *proportion* of its images to the
    validation set, so rare classes stay represented, and the seed makes
    the whole split reproducible -- which is what makes the >=90%
    accuracy claim checkable at evaluation time.

    Returns (train_items, val_items), each a list of (path, label).
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    groups = group_by_label(list_images(root))
    rng = random.Random(seed)
    train_items, val_items = [], []

    for label in sorted(groups):
        items = sorted(groups[label])
        rng.shuffle(items)
        held_out = int(round(len(items) * val_ratio))
        if len(items) > 1:
            # Never let a class vanish from either side of the split.
            held_out = min(max(held_out, 1), len(items) - 1)
        else:
            held_out = 0
        val_items += items[:held_out]
        train_items += items[held_out:]

    return train_items, val_items
