#!/usr/bin/env python3
"""
transformation.py — Leaffliction, Part 3 (Person B).

Implements >=6 leaf-image transformations (Gaussian blur, mask, ROI objects,
analyze object, pseudolandmarks, plus a color histogram) using OpenCV/NumPy,
in the spirit of the PlantCV pipeline shown in the subject.

Public API for Person C (predict.py / train.py):

    transform_image(img_rgb) -> dict[str, np.ndarray]
        keys: original, gaussian_blur, mask, roi_objects,
              analyze_object, pseudolandmarks

    color_histogram(img_rgb) -> matplotlib.figure.Figure

    transformed_for_display(img_rgb) -> np.ndarray
        single "nice looking" transformed image for predict.py to show
        next to the original.

CLI usage:

    ./transformation.py path/to/image.jpg
        -> displays the full set of transformations in a matplotlib window.

    ./transformation.py -src leaves/images/Apple_healthy/ -dst dst_directory -mask
        -> batch mode: for every image found under -src (recursively),
           save the requested transformation(s) into -dst, using the
           original filename + "_<transformation>" suffix.

    ./transformation.py -h
        -> usage / help.
"""

import argparse
import os
import sys
import glob

import cv2
import numpy as np
import matplotlib

# Use a non-interactive backend automatically when we are only saving files
# (batch mode) or when there is no display available. We decide this at
# runtime in main(); importing pyplot after backend selection.
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# I/O helpers — try to reuse Person A's utils.dataset if it exists, else
# fall back to local, self-contained implementations. Convention (per the
# team contract): RGB uint8 ndarray (H, W, 3) at every boundary.
# --------------------------------------------------------------------------
try:
    from utils.dataset import load_image, save_image  # type: ignore
except Exception:  # pragma: no cover - fallback if utils/dataset.py absent
    def load_image(path):
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def save_image(path, img_rgb):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Order matters: this defines both the display grid order and the
# CLI flag -> key mapping used in batch mode.
TRANSFORM_ORDER = [
    "gaussian_blur",
    "mask",
    "roi_objects",
    "analyze_object",
    "pseudolandmarks",
]

# Human-readable suffixes used when saving files, matching subject's style
# (e.g. image (1)_Flip.JPG in the Augmentation example).
SUFFIXES = {
    "gaussian_blur": "GaussianBlur",
    "mask": "Mask",
    "roi_objects": "RoiObjects",
    "analyze_object": "AnalyzeObject",
    "pseudolandmarks": "Pseudolandmarks",
    "histogram": "Histogram",
}


# --------------------------------------------------------------------------
# Core transformations
# --------------------------------------------------------------------------
def _leaf_mask(img_rgb):
    """
    Build a binary mask isolating the leaf from the (usually light/uniform)
    background, using HSV saturation + Otsu thresholding, cleaned up with
    morphological operations and largest-contour selection.
    Returns a single-channel uint8 mask (0 / 255).
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    blurred = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    clean_mask = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        clean_mask = mask

    return clean_mask


def _largest_contour(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def transform_image(img_rgb):
    """
    Run the full transformation pipeline on an RGB uint8 image.

    Returns a dict with keys:
        original, gaussian_blur, mask, roi_objects,
        analyze_object, pseudolandmarks
    All values are RGB uint8 ndarrays of the same (H, W, 3) shape as input.
    """
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)

    h, w = img_rgb.shape[:2]
    result = {"original": img_rgb.copy()}

    # 1. Gaussian blur
    gaussian_blur = cv2.GaussianBlur(img_rgb, (9, 9), 0)
    result["gaussian_blur"] = gaussian_blur

    # 2. Mask (binary leaf mask, shown as a 3-channel image for display)
    mask = _leaf_mask(img_rgb)
    result["mask"] = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    contour = _largest_contour(mask)

    # 3. ROI objects: leaf isolated on background + bounding box
    roi_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    if contour is not None:
        x, y, cw, ch = cv2.boundingRect(contour)
        cv2.rectangle(roi_img, (x, y), (x + cw, y + ch), (255, 0, 0), 3)
    result["roi_objects"] = roi_img

    # 4. Analyze object: contour outline + centroid + shape stats overlay
    analyze_img = img_rgb.copy()
    if contour is not None:
        cv2.drawContours(analyze_img, [contour], -1, (0, 255, 0), 2)
        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            cv2.drawMarker(
                analyze_img, (cx, cy), (255, 0, 0),
                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2,
            )
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        cv2.putText(
            analyze_img, f"area={int(area)} perim={int(perimeter)}",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
        )
    result["analyze_object"] = analyze_img

    # 5. Pseudolandmarks: evenly-sampled contour points, 3 colored groups
    #    (top / left+right / bottom), similar in spirit to PlantCV's
    #    pseudolandmark output.
    landmarks_img = img_rgb.copy()
    if contour is not None:
        pts = contour.reshape(-1, 2)
        n_points = min(30, len(pts))
        idx = np.linspace(0, len(pts) - 1, n_points).astype(int)
        sampled = pts[idx]

        ys = sampled[:, 1]
        thirds = np.array_split(np.argsort(ys), 3)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # top, mid, bottom
        for group_idx, group in enumerate(thirds):
            for pi in group:
                x, y = sampled[pi]
                cv2.circle(
                    landmarks_img, (int(x), int(y)), 4,
                    colors[group_idx], thickness=-1,
                )
    result["pseudolandmarks"] = landmarks_img

    return result


def color_histogram(img_rgb):
    """
    Build a matplotlib Figure with per-channel color histograms
    (RGB + HSV saturation), matching Figure IV.7 in the subject.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    fig, ax = plt.subplots(figsize=(6, 4))
    channels = [
        ("Red", img_rgb[:, :, 0], "red"),
        ("Green", img_rgb[:, :, 1], "green"),
        ("Blue", img_rgb[:, :, 2], "blue"),
        ("Saturation", hsv[:, :, 1], "orange"),
    ]
    for label, channel, color in channels:
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()  # normalize -> proportion of pixels
        ax.plot(hist, color=color, label=label)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels")
    ax.set_title("Color histogram")
    ax.legend()
    fig.tight_layout()
    return fig


def transformed_for_display(img_rgb):
    """
    Single 'nice looking' transformed image for predict.py to display
    next to the original — the analyze_object rendering (contour +
    centroid + shape stats), which is the most visually informative.
    """
    transforms = transform_image(img_rgb)
    return transforms["analyze_object"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _find_images(src_dir):
    files = []
    for ext in VALID_EXTENSIONS:
        files.extend(
            glob.glob(os.path.join(src_dir, "**", f"*{ext}"), recursive=True)
        )
    return sorted(set(files))


def _requested_keys(args):
    """
    Which transformation keys were requested via flags. If none of the
    per-transform flags were passed, default to ALL of them (+ histogram).
    """
    flag_map = {
        "gaussian_blur": args.blur,
        "mask": args.mask,
        "roi_objects": args.roi,
        "analyze_object": args.object,
        "pseudolandmarks": args.landmarks,
    }
    requested = [k for k, v in flag_map.items() if v]
    include_hist = args.histogram

    any_flag = any(flag_map.values()) or args.histogram
    if not any_flag:
        return TRANSFORM_ORDER.copy(), True

    return requested, include_hist


def _display_single(img_rgb, keys, include_hist):
    n_plots = len(keys) + 1 + (1 if include_hist else 0)  # +1 for original
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    transforms = transform_image(img_rgb)

    plot_idx = 0
    axes[plot_idx].imshow(transforms["original"])
    axes[plot_idx].set_title("Original")
    axes[plot_idx].axis("off")
    plot_idx += 1

    for key in keys:
        axes[plot_idx].imshow(transforms[key])
        axes[plot_idx].set_title(key.replace("_", " ").title())
        axes[plot_idx].axis("off")
        plot_idx += 1

    for ax in axes[plot_idx:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()

    if include_hist:
        hist_fig = color_histogram(img_rgb)
        plt.show()
        return hist_fig
    return None


def _process_batch(src, dst, keys, include_hist):
    os.makedirs(dst, exist_ok=True)
    images = _find_images(src)
    if not images:
        print(f"No images found under: {src}", file=sys.stderr)
        sys.exit(1)

    for path in images:
        base = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1]
        try:
            img_rgb = load_image(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue

        transforms = transform_image(img_rgb)

        for key in keys:
            out_name = f"{base}_{SUFFIXES[key]}{ext}"
            save_image(os.path.join(dst, out_name), transforms[key])

        if include_hist:
            hist_fig = color_histogram(img_rgb)
            hist_path = os.path.join(dst, f"{base}_{SUFFIXES['histogram']}.png")
            hist_fig.savefig(hist_path)
            plt.close(hist_fig)

        print(f"Processed: {path}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="transformation.py",
        description=(
            "Apply leaf-image transformations (Gaussian blur, mask, "
            "ROI objects, analyze object, pseudolandmarks, color "
            "histogram). Single image path -> display results. "
            "-src/-dst -> batch-save results to a directory."
        ),
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to a single image (display mode).",
    )
    parser.add_argument(
        "-src", dest="src", default=None,
        help="Source directory of images (batch mode, used with -dst).",
    )
    parser.add_argument(
        "-dst", dest="dst", default=None,
        help="Destination directory to save transformed images.",
    )
    parser.add_argument(
        "-blur", action="store_true", help="Include Gaussian blur.",
    )
    parser.add_argument(
        "-mask", action="store_true", help="Include leaf mask.",
    )
    parser.add_argument(
        "-roi", action="store_true", help="Include ROI objects.",
    )
    parser.add_argument(
        "-object", action="store_true", help="Include analyze object.",
    )
    parser.add_argument(
        "-landmarks", action="store_true", help="Include pseudolandmarks.",
    )
    parser.add_argument(
        "-histogram", action="store_true", help="Include color histogram.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    keys, include_hist = _requested_keys(args)

    if args.src and args.dst:
        # Batch mode: no display needed, force a non-interactive backend
        # for safety on headless machines.
        matplotlib.use("Agg", force=True)
        _process_batch(args.src, args.dst, keys, include_hist)
    elif args.image:
        if not os.path.isfile(args.image):
            print(f"No such file: {args.image}", file=sys.stderr)
            sys.exit(1)
        img_rgb = load_image(args.image)
        _display_single(img_rgb, keys, include_hist)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
