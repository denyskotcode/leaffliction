#!/usr/bin/env python3
"""
Part 1 -- analysis of the data set.

``./Distribution.py <dir>`` fetches the images in the subdirectories of
``<dir>``, counts them per class, and draws a pie chart and a bar chart
for each plant type. Every name on the charts comes from a directory
name and nowhere else, so the program describes whatever subtree it is
pointed at: the whole data set, one plant, or a single class directory.
"""

import argparse
import os
import sys

import matplotlib

# A backend has to be chosen before pyplot is imported: on a machine with
# no display the default one raises instead of drawing.
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from utils.dataset import class_counts  # noqa: E402
from utils.naming import plant_type  # noqa: E402

# Categorical slots, checked against colour-vision-deficiency simulation
# on a light surface. This data set has four classes per plant and these
# four stay distinguishable in every simulation, including as pie slices
# where any two may end up side by side. Past four we fall back to the
# eight-slot order, which is only guaranteed between neighbours -- which
# is why every chart here also carries a direct label, never colour
# alone.
PALETTE_4 = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
PALETTE_8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
             "#e87ba4", "#008300", "#4a3aa7", "#e34948")

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def collect_counts(root):
    """
    Return {label: image count} for ``root``.

    Falls back to counting ``root``'s own images when the walk finds
    nothing nested, so pointing the program at a single class directory
    works as well as pointing it at the data set root.
    """
    counts = class_counts(root)
    if not counts:
        counts = class_counts(root, include_root=True)
    return counts


def group_by_plant(counts):
    """{plant type: {label: count}}, both levels ordered by name."""
    groups = {}
    for label in sorted(counts):
        groups.setdefault(plant_type(label), {})[label] = counts[label]
    return groups


def colors_for(count):
    """Pick one colour per class, in a fixed order that never shifts."""
    palette = PALETTE_4 if count <= len(PALETTE_4) else PALETTE_8
    return [palette[index % len(palette)] for index in range(count)]


def _draw_pie(axis, labels, values, colors):
    """Share of the plant's images held by each class."""
    wedges, _texts, percents = axis.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%",
        # A thin surface-coloured gap keeps adjacent slices legible even
        # when their colours are close.
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"color": "white", "fontsize": 10, "weight": "bold"},
    )
    axis.set_title("share of images", color=INK_MUTED, fontsize=11)
    return wedges, percents


def _draw_bar(axis, labels, values, colors):
    """Absolute count per class, labelled from the directory names."""
    positions = range(len(labels))
    bars = axis.bar(positions, values, color=colors, width=0.62)

    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=20, ha="right",
                         color=INK_MUTED, fontsize=9)
    axis.set_ylabel("images", color=INK_MUTED, fontsize=10)
    axis.set_title("images per class", color=INK_MUTED, fontsize=11)

    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(axis="both", length=0, colors=INK_MUTED)

    headroom = max(values) * 0.02 if values else 0
    for rectangle, value in zip(bars, values):
        axis.annotate(
            str(value),
            (rectangle.get_x() + rectangle.get_width() / 2,
             rectangle.get_height() + headroom),
            ha="center", va="bottom", fontsize=9, color=INK,
        )
    axis.set_ylim(0, max(values) * 1.12 if values else 1)


def draw_plant(plant, counts):
    """Build the pie + bar figure for one plant type."""
    labels = list(counts)
    values = [counts[label] for label in labels]
    colors = colors_for(len(labels))

    figure, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(13, 6))
    figure.suptitle(f"{plant} class distribution",
                    fontsize=15, color=INK, x=0.02, ha="left")

    wedges, _percents = _draw_pie(ax_pie, labels, values, colors)
    _draw_bar(ax_bar, labels, values, colors)

    # The legend names the classes once for both charts, so identity
    # never depends on remembering a colour.
    figure.legend(wedges, labels, loc="lower center", ncol=min(4, len(labels)),
                  frameon=False, fontsize=9, labelcolor=INK_MUTED,
                  bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    return figure


def report(groups):
    """Print the counts, so the analysis survives without a display."""
    for plant in sorted(groups):
        counts = groups[plant]
        total = sum(counts.values())
        print(f"{plant}: {total} images across {len(counts)} classes")
        for label in sorted(counts):
            share = counts[label] / total * 100
            print(f"  {label:<24} {counts[label]:>6}  ({share:5.1f}%)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="Distribution.py",
        description="Count the images of a leaf data set and chart the "
                    "class distribution of each plant type.",
    )
    parser.add_argument("directory",
                        help="data set directory to analyse; its "
                             "subdirectories name the classes")
    parser.add_argument("--save-dir", metavar="DIR",
                        help="also write each figure to DIR as a PNG")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a window (useful over ssh)")
    return parser


def main():
    args = build_parser().parse_args()

    try:
        counts = collect_counts(args.directory)
    except NotADirectoryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: cannot read {args.directory}: {error}",
              file=sys.stderr)
        return 1

    if not counts:
        print(f"error: no images found under {args.directory}",
              file=sys.stderr)
        return 1

    groups = group_by_plant(counts)
    report(groups)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for plant in sorted(groups):
        figure = draw_plant(plant, groups[plant])
        if args.save_dir:
            destination = os.path.join(args.save_dir,
                                       f"{plant}_distribution.png")
            figure.savefig(destination, dpi=120, bbox_inches="tight")
            print(f"saved {destination}")

    if args.no_display:
        plt.close("all")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
