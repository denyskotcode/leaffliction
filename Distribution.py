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

# The backend has to be picked before pyplot is imported: with no display
# attached, the default one raises instead of drawing.
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from utils.dataset import class_counts  # noqa: E402
from utils.naming import plant_type  # noqa: E402

# Categorical slots, checked against colour-vision-deficiency simulation
# on a light surface. The first four stay apart in every simulation, even
# as neighbouring pie slices, and four is what this data set needs per
# plant; the rest are only guaranteed between neighbours. That is why
# every chart below also carries a written label -- never colour alone.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
           "#eda100", "#e87ba4", "#008300", "#e34948")

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def collect_counts(root):
    """
    Return {label: image count} for ``root``.

    When the walk finds nothing nested it counts ``root``'s own images
    instead, so aiming the program at one class directory works as well
    as aiming it at the data set root.
    """
    return class_counts(root) or class_counts(root, include_root=True)


def group_by_plant(counts):
    """{plant type: {label: count}}, both levels ordered by name."""
    groups = {}
    for label in sorted(counts):
        groups.setdefault(plant_type(label), {})[label] = counts[label]
    return groups


def colors_for(count):
    """One colour per class, in a fixed order that never shifts."""
    return [PALETTE[slot % len(PALETTE)] for slot in range(count)]


def _style_axis(axis, title):
    """Muted title, no box, horizontal rules only."""
    axis.set_title(title, color=INK_MUTED, fontsize=11)
    axis.tick_params(axis="both", length=0, colors=INK_MUTED)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)


def _draw_pie(axis, values, colors):
    """Share of the plant's images held by each class."""
    wedges, _labels, _percents = axis.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%",
        # A thin surface-coloured gap keeps adjacent slices readable even
        # where their colours are close.
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"color": "white", "fontsize": 10, "weight": "bold"},
    )
    _style_axis(axis, "share of images")
    return wedges


def _draw_bar(axis, labels, values, colors):
    """Absolute count per class, named from the directories."""
    slots = range(len(labels))
    tallest = max(values)
    bars = axis.bar(slots, values, color=colors, width=0.62)

    axis.set_xticks(list(slots))
    axis.set_xticklabels(labels, rotation=20, ha="right",
                         color=INK_MUTED, fontsize=9)
    axis.set_ylabel("images", color=INK_MUTED, fontsize=10)
    axis.set_ylim(0, tallest * 1.12)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    _style_axis(axis, "images per class")
    axis.spines["bottom"].set_color(GRID)

    for bar, value in zip(bars, values):
        axis.annotate(str(value),
                      (bar.get_x() + bar.get_width() / 2,
                       value + tallest * 0.02),
                      ha="center", va="bottom", fontsize=9, color=INK)


def draw_plant(plant, counts):
    """Build the pie + bar figure for one plant type."""
    labels = list(counts)
    values = list(counts.values())
    colors = colors_for(len(labels))

    figure, (pie_axis, bar_axis) = plt.subplots(1, 2, figsize=(13, 6))
    figure.suptitle(f"{plant} class distribution",
                    fontsize=15, color=INK, x=0.02, ha="left")

    wedges = _draw_pie(pie_axis, values, colors)
    _draw_bar(bar_axis, labels, values, colors)

    # The legend names the classes once for both charts, so identity
    # never rests on remembering a colour.
    figure.legend(wedges, labels, loc="lower center", frameon=False,
                  ncol=min(4, len(labels)), fontsize=9,
                  labelcolor=INK_MUTED, bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    return figure


def report(groups):
    """Print the counts, so the analysis survives without a display."""
    for plant in sorted(groups):
        counts = groups[plant]
        total = sum(counts.values())
        print(f"{plant}: {total} images across {len(counts)} classes")
        for label, count in sorted(counts.items()):
            print(f"  {label:<24} {count:>6}  ({count / total * 100:5.1f}%)")


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
