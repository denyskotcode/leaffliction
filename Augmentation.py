#!/usr/bin/env python3
"""
Part 2 -- data augmentation.

The data set is not balanced: the largest class holds several times the
images of the smallest, and a classifier trained on it would learn that
imbalance as a prior. This program is the fix, and it has two modes.

    ./Augmentation.py "<image>"
        Display the six augmentations of one image and save each beside
        the original as ``<base>_<Aug>.JPG``.

    ./Augmentation.py -src <dir> -dst augmented_directory
        Balance a whole data set: copy every image across, then augment
        the smaller classes until each one matches the largest.

``train.py`` imports the six functions and ``build_augmented_dir`` from
here, so the images the model learns from are produced by exactly this
code rather than a second copy of it that could drift.
"""

import argparse
import math
import os
import random
import shutil
import sys

import cv2
import numpy as np
import matplotlib

# A backend has to be chosen before pyplot is imported: on a machine with
# no display the default one raises instead of drawing.
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from utils.dataset import list_images, load_image, save_image  # noqa: E402
from utils.naming import AUG_NAMES, augmented_filename  # noqa: E402

SEED = 42


# --------------------------------------------------------------------------
# The six augmentations required by the subject.
# Each takes and returns RGB uint8 (H, W, 3).
#
# All six reflect at the border rather than padding with black: a flat
# black wedge is a feature no real leaf photo has, and the network would
# happily learn it as a shortcut to whichever classes needed the most
# augmenting.
# --------------------------------------------------------------------------
def flip(img, rng=None):
    """Mirror left-to-right."""
    return cv2.flip(img, 1)


def rotate(img, rng=None):
    """Rotate about the centre by up to 30 degrees either way."""
    rng = rng or random.Random(SEED)
    height, width = img.shape[:2]
    angle = rng.uniform(-30.0, 30.0)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (width, height),
                          borderMode=cv2.BORDER_REFLECT_101)


def skew(img, rng=None):
    """Perspective skew, as though the leaf were tilted away."""
    rng = rng or random.Random(SEED)
    height, width = img.shape[:2]
    shift = rng.uniform(0.08, 0.20) * width
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = np.float32([[shift, 0], [width - shift, 0],
                      [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height),
                               borderMode=cv2.BORDER_REFLECT_101)


def shear(img, rng=None):
    """Slant the image horizontally, keeping its height."""
    rng = rng or random.Random(SEED)
    height, width = img.shape[:2]
    factor = rng.uniform(-0.25, 0.25)
    matrix = np.float32([[1, factor, -factor * height / 2], [0, 1, 0]])
    return cv2.warpAffine(img, matrix, (width, height),
                          borderMode=cv2.BORDER_REFLECT_101)


def crop(img, rng=None):
    """Take a random 70-88% window and scale it back up."""
    rng = rng or random.Random(SEED)
    height, width = img.shape[:2]
    keep = rng.uniform(0.70, 0.88)
    new_h, new_w = int(height * keep), int(width * keep)
    top = rng.randint(0, height - new_h)
    left = rng.randint(0, width - new_w)
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (width, height),
                      interpolation=cv2.INTER_LINEAR)


def distortion(img, rng=None):
    """Barrel / pincushion lens distortion."""
    rng = rng or random.Random(SEED)
    height, width = img.shape[:2]
    strength = rng.uniform(-0.35, 0.35)
    ys, xs = np.indices((height, width), dtype=np.float32)
    norm_x = (xs - width / 2) / (width / 2)
    norm_y = (ys - height / 2) / (height / 2)
    radius = norm_x ** 2 + norm_y ** 2
    scale = 1.0 + strength * radius
    map_x = (norm_x * scale) * (width / 2) + width / 2
    map_y = (norm_y * scale) * (height / 2) + height / 2
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


AUGMENTATIONS = {
    "Flip": flip,
    "Rotate": rotate,
    "Skew": skew,
    "Shear": shear,
    "Crop": crop,
    "Distortion": distortion,
}


def augment_image(img_rgb, rng=None):
    """Return {name: variant} for all six augmentation types."""
    rng = rng or random.Random(SEED)
    return {name: fn(img_rgb, rng) for name, fn in AUGMENTATIONS.items()}


# --------------------------------------------------------------------------
# Single image: display the six variants and save them beside the source
# --------------------------------------------------------------------------
def save_variants(source, variants, dst_dir=None):
    """Write each variant as ``<base>_<Aug>.JPG``; return the paths."""
    written = []
    for name, image in variants.items():
        path = os.path.join(dst_dir if dst_dir else os.path.dirname(source),
                            augmented_filename(source, name))
        save_image(image, path)
        written.append(path)
    return written


def show_variants(source, original, variants):
    """Original first, then the six augmentations, on one row."""
    panels = [("Original", original)] + list(variants.items())
    figure, axes = plt.subplots(1, len(panels),
                                figsize=(2.4 * len(panels), 3.2))
    for axis, (name, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(name, fontsize=11, color="#0b0b0b")
        axis.axis("off")
    figure.suptitle(os.path.basename(source), fontsize=12, color="#52514e")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Whole data set: balance every class up to the largest
# --------------------------------------------------------------------------
def build_augmented_dir(items, out_dir, seed=SEED):
    """
    Copy every image in ``items`` into ``out_dir/<label>/`` and augment
    the smaller classes until each one matches the largest.

    ``items`` is [(path, label), ...]. Only the images handed in reach
    the output, which is what lets ``train.py`` balance its training
    split alone and keep the validation split untouched.

    Returns [(path, label), ...] for the balanced set.
    """
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    by_label = {}
    for path, label in items:
        by_label.setdefault(label, []).append(path)
    if not by_label:
        raise ValueError("no images to balance")

    target = max(len(paths) for paths in by_label.values())
    print(f"balancing {len(by_label)} classes up to {target} images each")

    balanced = []
    rng = random.Random(seed)
    for label in sorted(by_label):
        paths = sorted(by_label[label])
        class_dir = os.path.join(out_dir, label)
        os.makedirs(class_dir, exist_ok=True)

        for path in paths:
            dest = os.path.join(class_dir, os.path.basename(path))
            shutil.copyfile(path, dest)
            balanced.append((dest, label))

        needed = target - len(paths)
        # (source, augmentation) pairs repeat every lcm(n_paths, 6)
        # steps, not every n_paths * 6 steps. Numbering the rounds by
        # the wrong period lets a later file overwrite an earlier one,
        # which silently under-fills exactly the rarest classes.
        period = math.lcm(len(paths), len(AUG_NAMES))
        for index in range(needed):
            source = paths[index % len(paths)]
            name = AUG_NAMES[index % len(AUG_NAMES)]
            round_id = index // period
            dest = os.path.join(
                class_dir, augmented_filename(source, name, round_id))
            variant = AUGMENTATIONS[name](load_image(source), rng)
            save_image(variant, dest)
            balanced.append((dest, label))

        written = len(os.listdir(class_dir))
        if written != target:
            raise RuntimeError(
                f"{label}: wrote {written} images but expected {target}; "
                "augmented filenames are colliding")
        print(f"  {label}: {len(paths)} -> {written}")

    return balanced


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="Augmentation.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate the six augmentations of a leaf image, or "
                    "balance a whole data set.",
        epilog="""examples:
  ./Augmentation.py "leaves/images/Apple_healthy/image (1).JPG"
      -> display Flip, Rotate, Skew, Shear, Crop and Distortion, and
         save each one beside the original as <base>_<Aug>.JPG

  ./Augmentation.py -src leaves/images -dst augmented_directory
      -> copy every image across and augment the smaller classes until
         every class matches the largest one
""")
    parser.add_argument("image", nargs="?",
                        help="a single image to augment and display")
    parser.add_argument("-src", metavar="DIR",
                        help="data set directory to balance")
    parser.add_argument("-dst", metavar="DIR", default="augmented_directory",
                        help="where the balanced set is written "
                             "(default: augmented_directory)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"random seed (default: {SEED})")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a window (useful over ssh)")
    return parser


def run_single(args):
    """Augment one image: save the six variants, then show them."""
    if not os.path.isfile(args.image):
        print(f"error: not a file: {args.image}", file=sys.stderr)
        return 1

    original = load_image(args.image)
    variants = augment_image(original, random.Random(args.seed))

    for path in save_variants(args.image, variants):
        print(os.path.basename(path))

    if not args.no_display:
        show_variants(args.image, original, variants)
        plt.show()
    return 0


def run_balance(args):
    """Balance a whole data set into ``-dst``."""
    items = list_images(args.src)
    if not items:
        items = list_images(args.src, include_root=True)
    if not items:
        print(f"error: no images found under {args.src}", file=sys.stderr)
        return 1

    balanced = build_augmented_dir(items, args.dst, seed=args.seed)
    print(f"{len(balanced)} images written to {args.dst}")
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.src and args.image:
        parser.error("give either a single image or -src, not both")
    if not args.src and not args.image:
        parser.error("give an image to augment, or -src to balance a set")

    try:
        if args.src:
            return run_balance(args)
        return run_single(args)
    except (NotADirectoryError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
