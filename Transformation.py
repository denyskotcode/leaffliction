#!/usr/bin/env python3
"""
Part 3 -- image transformation.

Six leaf-image transformations (Gaussian blur, mask, ROI objects,
analyze object, pseudolandmarks, plus a colour histogram), built on
OpenCV and NumPy in the spirit of the PlantCV pipeline the subject
shows.

    ./Transformation.py <image>
        Display the whole set for one image.

    ./Transformation.py -src <dir> -dst <dir> [-mask ...]
        Save the requested transformations for every image under -src,
        naming each one <base>_<Transformation><ext>.

    ./Transformation.py -h
        Usage.

Passing no per-transformation flag means all of them, histogram
included.

``predict.py`` imports ``transformed_for_display`` from here, so the
picture it shows beside a prediction is produced by this code and not by
a second copy of it.
"""

import argparse
import os
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

# Order matters twice over: it lays out the display grid, and it maps the
# CLI flags onto the keys of transform_image().
TRANSFORMS = {
    "gaussian_blur": "GaussianBlur",
    "mask": "Mask",
    "roi_objects": "RoiObjects",
    "analyze_object": "AnalyzeObject",
    "pseudolandmarks": "Pseudolandmarks",
}
HISTOGRAM_SUFFIX = "Histogram"

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


# --------------------------------------------------------------------------
# Core transformations
# --------------------------------------------------------------------------
def leaf_outline(img_rgb):
    """
    Separate the leaf from its (light, fairly uniform) background.

    Saturation does the work: leaf tissue is coloured and the background
    is not, so an Otsu threshold over the saturation channel splits them
    without a hand-tuned constant. Morphology closes the holes left by
    specular highlights, and keeping only the largest contour drops the
    stray flecks that survive.

    Returns (mask, contour); ``contour`` is None when nothing was found.
    """
    saturation = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    smoothed = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, rough = cv2.threshold(smoothed, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, kernel, iterations=2)
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rough, None

    leaf = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(rough)
    cv2.drawContours(mask, [leaf], -1, 255, thickness=cv2.FILLED)
    return mask, leaf


def _roi_objects(img_rgb, mask, contour):
    """The leaf cut out of its background, boxed."""
    isolated = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    if contour is not None:
        left, top, width, height = cv2.boundingRect(contour)
        cv2.rectangle(isolated, (left, top),
                      (left + width, top + height), RED, 3)
    return isolated


def _analyze_object(img_rgb, contour):
    """Outline, centre of mass, and the two shape numbers behind them."""
    canvas = img_rgb.copy()
    if contour is None:
        return canvas

    cv2.drawContours(canvas, [contour], -1, GREEN, 2)
    moments = cv2.moments(contour)
    if moments["m00"]:
        centre = (int(moments["m10"] / moments["m00"]),
                  int(moments["m01"] / moments["m00"]))
        cv2.drawMarker(canvas, centre, RED, markerType=cv2.MARKER_CROSS,
                       markerSize=20, thickness=2)
    caption = (f"area={int(cv2.contourArea(contour))} "
               f"perim={int(cv2.arcLength(contour, True))}")
    cv2.putText(canvas, caption, (10, canvas.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    return canvas


def _pseudolandmarks(img_rgb, contour, points=30):
    """
    Points spaced evenly along the outline, coloured top / middle /
    bottom by height -- PlantCV's pseudolandmarks in spirit.
    """
    canvas = img_rgb.copy()
    if contour is None:
        return canvas

    outline = contour.reshape(-1, 2)
    steps = np.linspace(0, len(outline) - 1, min(points, len(outline)))
    sampled = outline[steps.astype(int)]

    bands = np.array_split(np.argsort(sampled[:, 1]), 3)
    for colour, band in zip((RED, GREEN, BLUE), bands):
        for x, y in sampled[band]:
            cv2.circle(canvas, (int(x), int(y)), 4, colour, thickness=-1)
    return canvas


def transform_image(img_rgb):
    """
    Run the whole pipeline on an RGB uint8 image.

    Returns {original, gaussian_blur, mask, roi_objects, analyze_object,
    pseudolandmarks}, every value an RGB uint8 array the same shape as
    the input.
    """
    img_rgb = np.asarray(img_rgb, dtype=np.uint8)
    mask, contour = leaf_outline(img_rgb)
    return {
        "original": img_rgb.copy(),
        "gaussian_blur": cv2.GaussianBlur(img_rgb, (9, 9), 0),
        "mask": cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
        "roi_objects": _roi_objects(img_rgb, mask, contour),
        "analyze_object": _analyze_object(img_rgb, contour),
        "pseudolandmarks": _pseudolandmarks(img_rgb, contour),
    }


def color_histogram(img_rgb):
    """
    Per-channel colour histograms (RGB plus HSV saturation), as in
    Figure IV.7 of the subject. Counts are turned into proportions so
    images of different sizes stay comparable.
    """
    saturation = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    channels = (("Red", img_rgb[:, :, 0], "red"),
                ("Green", img_rgb[:, :, 1], "green"),
                ("Blue", img_rgb[:, :, 2], "blue"),
                ("Saturation", saturation, "orange"))

    figure, axis = plt.subplots(figsize=(6, 4))
    for label, channel, colour in channels:
        counts = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
        axis.plot(counts / counts.sum(), color=colour, label=label)

    axis.set_xlabel("Pixel intensity")
    axis.set_ylabel("Proportion of pixels")
    axis.set_title("Colour histogram")
    axis.legend()
    figure.tight_layout()
    return figure


def transformed_for_display(img_rgb):
    """
    The single transformed picture predict.py shows beside the original:
    the analyze-object rendering, which carries the most information at a
    glance (outline, centre, size).
    """
    return transform_image(img_rgb)["analyze_object"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def requested(args):
    """
    (keys, include_histogram) for this run.

    With no per-transformation flag the answer is everything, which makes
    the plain ``-src/-dst`` form do the obvious thing.
    """
    chosen = [key for key in TRANSFORMS if getattr(args, key)]
    if not chosen and not args.histogram:
        return list(TRANSFORMS), True
    return chosen, args.histogram


def display_single(img_rgb, keys, include_histogram):
    """Draw the original plus every requested transformation."""
    results = transform_image(img_rgb)
    panels = [("Original", results["original"])]
    panels += [(key.replace("_", " ").title(), results[key]) for key in keys]

    columns = 3
    rows = -(-len(panels) // columns)
    figure, axes = plt.subplots(rows, columns,
                                figsize=(5 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()

    if include_histogram:
        color_histogram(img_rgb)
    plt.show()


def process_batch(src, dst, keys, include_histogram):
    """Transform every image under ``src`` into ``dst``."""
    images = list_images(src, include_root=True)
    if not images:
        raise ValueError(f"no images found under: {src}")

    os.makedirs(dst, exist_ok=True)
    for path, _label in images:
        base, extension = os.path.splitext(os.path.basename(path))
        try:
            img_rgb = load_image(path)
        except (ValueError, OSError) as error:
            print(f"skipping {path}: {error}", file=sys.stderr)
            continue

        results = transform_image(img_rgb)
        for key in keys:
            save_image(results[key],
                       os.path.join(dst, f"{base}_{TRANSFORMS[key]}"
                                         f"{extension}"))
        if include_histogram:
            figure = color_histogram(img_rgb)
            figure.savefig(os.path.join(dst,
                                        f"{base}_{HISTOGRAM_SUFFIX}.png"))
            plt.close(figure)

        print(f"processed: {path}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="Transformation.py",
        description="Apply leaf-image transformations (Gaussian blur, "
                    "mask, ROI objects, analyze object, pseudolandmarks, "
                    "colour histogram). A single image path displays the "
                    "results; -src with -dst saves them for a whole "
                    "directory.",
    )
    parser.add_argument("image", nargs="?",
                        help="a single image to transform and display")
    parser.add_argument("-src", help="source directory (use with -dst)")
    parser.add_argument("-dst", help="directory the results are written to")
    parser.add_argument("-blur", dest="gaussian_blur", action="store_true",
                        help="include the Gaussian blur")
    parser.add_argument("-mask", dest="mask", action="store_true",
                        help="include the leaf mask")
    parser.add_argument("-roi", dest="roi_objects", action="store_true",
                        help="include the ROI objects")
    parser.add_argument("-object", dest="analyze_object",
                        action="store_true",
                        help="include the analyzed object")
    parser.add_argument("-landmarks", dest="pseudolandmarks",
                        action="store_true",
                        help="include the pseudolandmarks")
    parser.add_argument("-histogram", action="store_true",
                        help="include the colour histogram")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    keys, include_histogram = requested(args)

    try:
        if args.src and args.dst:
            # Nothing is displayed in batch mode, and a machine with no
            # display would refuse to open a window at all.
            matplotlib.use("Agg", force=True)
            process_batch(args.src, args.dst, keys, include_histogram)
        elif args.image:
            if not os.path.isfile(args.image):
                raise ValueError(f"no such file: {args.image}")
            display_single(load_image(args.image), keys, include_histogram)
        else:
            parser.print_help()
            return 1
    except (NotADirectoryError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
