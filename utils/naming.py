"""
Naming conventions: how an augmented file is named, and how a class
label encodes its plant type.

Both are conventions rather than data -- nothing on disk records them --
so they live in one module instead of being re-spelled at every call
site. ``Augmentation.py`` writes these names one image at a time,
``train.py`` writes them in bulk while balancing, and ``Distribution.py``
reads plant types back out of labels. One definition keeps all three
agreeing.
"""

import os

# Title-case, in the order the subject lists them.
AUG_NAMES = ("Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion")

# Augmented copies are always written as JPEG, whatever the source was.
AUG_EXTENSION = ".JPG"


def augmented_filename(source, aug, round_id=0):
    """
    ``image (1).JPG`` + ``Flip`` -> ``image (1)_Flip.JPG``.

    ``round_id`` disambiguates the second and later passes over the same
    (image, augmentation) pair, which happens once a class is small
    enough that balancing has to reuse its sources. Round 0 keeps the
    plain name the subject shows; later rounds take a numeric suffix so
    an earlier file is never silently overwritten.
    """
    if aug not in AUG_NAMES:
        raise ValueError(f"unknown augmentation: {aug!r}")
    if round_id < 0:
        raise ValueError("round_id must not be negative")
    base = os.path.splitext(os.path.basename(source))[0]
    suffix = aug if round_id == 0 else f"{aug}{round_id}"
    return f"{base}_{suffix}{AUG_EXTENSION}"


def augmented_path(source, aug, dst_dir=None, round_id=0):
    """
    Full path for an augmented copy.

    Defaults to sitting beside the source, which is what Part 2 asks for
    when it is handed a single image; pass ``dst_dir`` to redirect the
    copy into a balanced output directory instead.
    """
    if dst_dir is None:
        dst_dir = os.path.dirname(source)
    return os.path.join(dst_dir, augmented_filename(source, aug, round_id))


def split_augmented(filename):
    """
    Inverse of :func:`augmented_filename`.

    Returns ``(original_base, aug_name)``, or ``(base, None)`` when the
    name carries no augmentation suffix. Class labels contain
    underscores too (``Apple_Black_rot``), so a trailing segment only
    counts when it actually names one of the six augmentations.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    head, separator, tail = base.rpartition("_")
    if not separator:
        return base, None
    name = tail.rstrip("0123456789") or tail
    if name in AUG_NAMES:
        return head, name
    return base, None


def is_augmented(filename):
    """True when ``filename`` was produced by the augmenter."""
    return split_augmented(filename)[1] is not None


def plant_type(label):
    """
    Plant type of a class label: the part before the first underscore.

    ``Apple_Black_rot`` -> ``Apple``. Part 1 groups its charts by this,
    and the directory name is the only place the plant is recorded, so a
    label without an underscore is simply its own plant type.
    """
    return label.split("_", 1)[0]
