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
here, so the images the model learns from come from exactly this code
rather than a second copy of it that could drift.
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

# The backend has to be picked before pyplot is imported: with no display
# attached, the default one raises instead of drawing.
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from utils.dataset import list_images, load_image, save_image  # noqa: E402
from utils.naming import AUG_NAMES, augmented_filename  # noqa: E402

SEED = 42

# Every geometric augmentation reflects at the border rather than padding
# with black: a flat black wedge is a feature no real leaf photo has, and
# the network would happily learn it as a shortcut to whichever classes
# needed the most augmenting.
BORDER = cv2.BORDER_REFLECT_101


def _rng(rng):
    """Fall back to the project seed when no generator is handed in."""
    return random.Random(SEED) if rng is None else rng


def _warp(img, matrix):
    """
    Apply a geometric matrix, keeping the original frame size.

    A 2x3 matrix is affine (rotate, shear), a 3x3 one is a perspective
    transform (skew); the two OpenCV calls differ in name only, so the
    shape of the matrix picks between them.
    """
    height, width = img.shape[:2]
    apply = cv2.warpAffine if matrix.shape[0] == 2 else cv2.warpPerspective
    return apply(img, matrix, (width, height), borderMode=BORDER)


# --------------------------------------------------------------------------
# The six augmentations required by the subject.
# Each takes and returns RGB uint8 (H, W, 3).
# --------------------------------------------------------------------------
def flip(img, rng=None):
    """Mirror left-to-right."""
    return cv2.flip(img, 1)


def rotate(img, rng=None):
    """Turn about the centre by up to 30 degrees either way."""
    height, width = img.shape[:2]
    angle = _rng(rng).uniform(-30.0, 30.0)
    return _warp(img, cv2.getRotationMatrix2D((width / 2, height / 2),
                                              angle, 1.0))


def skew(img, rng=None):
    """Perspective tilt, as though the leaf leaned away from the lens."""
    height, width = img.shape[:2]
    shift = _rng(rng).uniform(0.08, 0.20) * width
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    tilted = np.float32([[shift, 0], [width - shift, 0],
                         [width, height], [0, height]])
    return _warp(img, cv2.getPerspectiveTransform(corners, tilted))


def shear(img, rng=None):
    """Slant horizontally, keeping the height and the centre line."""
    height, width = img.shape[:2]
    factor = _rng(rng).uniform(-0.25, 0.25)
    return _warp(img, np.float32([[1, factor, -factor * height / 2],
                                  [0, 1, 0]]))


def crop(img, rng=None):
    """Take a random 70-88% window and scale it back up."""
    rng = _rng(rng)
    height, width = img.shape[:2]
    keep = rng.uniform(0.70, 0.88)
    box_h, box_w = int(height * keep), int(width * keep)
    top = rng.randint(0, height - box_h)
    left = rng.randint(0, width - box_w)
    window = img[top:top + box_h, left:left + box_w]
    return cv2.resize(window, (width, height),
                      interpolation=cv2.INTER_LINEAR)


def distortion(img, rng=None):
    """Barrel or pincushion lens distortion, radial about the centre."""
    height, width = img.shape[:2]
    strength = _rng(rng).uniform(-0.35, 0.35)
    half_h, half_w = height / 2, width / 2

    rows, columns = np.indices((height, width), dtype=np.float32)
    unit_x = (columns - half_w) / half_w
    unit_y = (rows - half_h) / half_h
    # Pixels move outward (or inward) in proportion to their squared
    # distance from the centre, which is what bends straight edges.
    stretch = 1.0 + strength * (unit_x ** 2 + unit_y ** 2)

    return cv2.remap(img,
                     unit_x * stretch * half_w + half_w,
                     unit_y * stretch * half_h + half_h,
                     cv2.INTER_LINEAR, borderMode=BORDER)


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
    rng = _rng(rng)
    return {name: make(img_rgb, rng) for name, make in AUGMENTATIONS.items()}


# --------------------------------------------------------------------------
# Single image: display the six variants and save them beside the source
# --------------------------------------------------------------------------
def save_variants(source, variants, dst_dir=None):
    """Write each variant as ``<base>_<Aug>.JPG``; return the paths."""
    folder = dst_dir or os.path.dirname(source)
    written = []
    for name, image in variants.items():
        path = os.path.join(folder, augmented_filename(source, name))
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
def _fill_class(paths, class_dir, target, rng):
    """
    Augment ``paths`` into ``class_dir`` until it holds ``target`` images,
    and return the paths written.

    Sources and augmentations are walked in step, so the copies stay
    spread evenly over both instead of piling onto one image or one
    effect.
    """
    # (source, augmentation) pairs start repeating after lcm(n, 6) steps,
    # not n * 6 steps. Numbering the rounds by the wrong period lets a
    # later file overwrite an earlier one, which silently under-fills
    # exactly the rarest classes.
    period = math.lcm(len(paths), len(AUG_NAMES))
    written = []
    for step in range(target - len(paths)):
        source = paths[step % len(paths)]
        name = AUG_NAMES[step % len(AUG_NAMES)]
        dest = os.path.join(class_dir,
                            augmented_filename(source, name, step // period))
        save_image(AUGMENTATIONS[name](load_image(source), rng), dest)
        written.append(dest)
    return written


def build_augmented_dir(items, out_dir, seed=SEED):
    """
    Copy every image in ``items`` into ``out_dir/<label>/`` and augment
    the smaller classes until each one matches the largest.

    ``items`` is [(path, label), ...]. Only the images handed in reach
    the output, which is what lets ``train.py`` balance its training
    split alone and leave the validation split untouched.

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

    rng = random.Random(seed)
    balanced = []
    for label in sorted(by_label):
        paths = sorted(by_label[label])
        class_dir = os.path.join(out_dir, label)
        os.makedirs(class_dir, exist_ok=True)

        copies = []
        for path in paths:
            dest = os.path.join(class_dir, os.path.basename(path))
            shutil.copyfile(path, dest)
            copies.append(dest)
        copies += _fill_class(paths, class_dir, target, rng)

        written = len(os.listdir(class_dir))
        if written != target:
            raise RuntimeError(
                f"{label}: wrote {written} images but expected {target}; "
                "augmented filenames are colliding")
        balanced += [(dest, label) for dest in copies]
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
    items = list_images(args.src) or list_images(args.src, include_root=True)
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
