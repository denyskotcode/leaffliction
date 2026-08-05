"""
File-naming and label-parsing conventions.

Nothing on disk records these rules, so they are spelled out once here
instead of at every call site: Augmentation.py writes augmented names one
image at a time, train.py writes them in bulk while balancing, and
Distribution.py reads plant types back out of labels.
"""

import os

# Title-case, in the order the subject lists them.
AUG_NAMES = ("Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion")

# Augmented copies are always written as JPEG, whatever the source was.
AUG_EXTENSION = ".JPG"

DIGITS = "0123456789"


def stem(path):
    """``a/b/image (1).JPG`` -> ``image (1)``."""
    return os.path.splitext(os.path.basename(path))[0]


def augmented_filename(source, aug, round_id=0):
    """
    ``image (1).JPG`` + ``Flip`` -> ``image (1)_Flip.JPG``.

    ``round_id`` separates the second and later passes over the same
    (image, augmentation) pair, which balancing has to make once a class
    is small enough that its sources run out. Round 0 keeps the plain
    name the subject shows; later rounds gain a number, so no earlier
    file is ever silently overwritten.
    """
    if aug not in AUG_NAMES:
        raise ValueError(f"unknown augmentation: {aug!r}")
    if round_id < 0:
        raise ValueError("round_id must not be negative")
    tag = aug if round_id == 0 else f"{aug}{round_id}"
    return f"{stem(source)}_{tag}{AUG_EXTENSION}"


def augmented_path(source, aug, dst_dir=None, round_id=0):
    """
    Full path of an augmented copy.

    With no ``dst_dir`` the copy lands beside its source, which is what
    Part 2 asks for when handed a single image; pass one to redirect the
    copy into a balanced output directory instead.
    """
    folder = os.path.dirname(source) if dst_dir is None else dst_dir
    return os.path.join(folder, augmented_filename(source, aug, round_id))


def split_augmented(filename):
    """
    Inverse of :func:`augmented_filename`.

    Returns ``(original_stem, aug_name)``, or ``(stem, None)`` when the
    name carries no augmentation tag. Labels contain underscores of their
    own (``Apple_Black_rot``), so the last segment counts only when it
    really names one of the six augmentations.
    """
    base = stem(filename)
    head, underscore, tail = base.rpartition("_")
    aug = tail.rstrip(DIGITS) or tail
    if underscore and aug in AUG_NAMES:
        return head, aug
    return base, None


def is_augmented(filename):
    """True when ``filename`` was produced by the augmenter."""
    return split_augmented(filename)[1] is not None


def plant_type(label):
    """
    Plant of a class label: whatever precedes the first underscore.

    ``Apple_Black_rot`` -> ``Apple``. Part 1 groups its charts by this.
    The directory name is the only record of the plant, so a label with
    no underscore is simply its own plant type.
    """
    return label.split("_", 1)[0]
