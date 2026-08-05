# Leaffliction — Code Guide

A file-by-file, function-by-function walkthrough of the codebase, with a
runnable test for every piece.

Everything here has been executed against the real dataset. Commands
assume you are at the repo root and use the project virtualenv
(`.venv/bin/python`).

---

## Contents

1. [Architecture at a glance](#1-architecture-at-a-glance)
2. [Environment setup](#2-environment-setup)
3. [`utils/dataset.py`](#3-utilsdatasetpy--io-and-splitting)
4. [`utils/naming.py`](#4-utilsnamingpy--filename-and-label-conventions)
5. [`utils/preprocess.py`](#5-utilspreprocesspy--the-single-preprocessing-path)
6. [`Distribution.py`](#6-distributionpy--part-1)
7. [`Augmentation.py`](#7-augmentationpy--part-2)
8. [`Transformation.py`](#8-transformationpy--part-3)
9. [`train.py`](#9-trainpy--part-4-training)
10. [`predict.py`](#10-predictpy--part-4-prediction)
11. [End-to-end test recipe](#11-end-to-end-test-recipe)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture at a glance

```
leaves/images/<class>/*.JPG          raw dataset (never committed)
        │
        ▼
utils/dataset.py                     list / load / save / count / split
        │
        ├────────────► Distribution.py      pie + bar charts per plant
        │
        ├────────────► Augmentation.py ──► augmented_directory/
        │                    ▲                (balanced training images)
        │                    │ imported by
        │              train.py ──────────► learnings.zip
        │                                       ├── model.pt
        │                                       ├── labels.json
        │                                       ├── metrics.json
        │                                       └── augmented_directory/
utils/preprocess.py                                      │
   (imported by BOTH train and predict)                  │
        │                                                ▼
        └────────────► predict.py ◄───────────────────────
                              │
                              └──► Transformation.py (display only)
```

**Two rules hold the design together.**

*One preprocessing path.* `utils/preprocess.py` is imported by both
`train.py` and `predict.py`. Images must be prepared identically at
training and prediction time; a mismatch silently destroys accuracy while
training still appears to converge. There is exactly one code path, so
the mismatch cannot happen.

*One augmentation implementation.* `train.py` imports the six
augmentations and `build_augmented_dir` from `Augmentation.py`, so the
images the model learns from are produced by exactly the code Part 2
ships — not by a second copy of it that could drift.

**The image format contract**, everywhere in this project: numpy
`ndarray`, shape `(H, W, 3)`, dtype `uint8`, **RGB order**. OpenCV is
BGR, so the BGR↔RGB flip happens only inside `utils/dataset.py`.

**Ownership** (see `00_TEAM_BRIEF.md`): A owns `utils/dataset.py`,
`utils/naming.py`, `Distribution.py`, `Augmentation.py`; B owns
`Transformation.py` and the release; C owns `utils/preprocess.py`,
`train.py`, `predict.py`.

---

## 2. Environment setup

This project runs on **Python 3.14**, for which TensorFlow publishes no
wheel. Part 4 therefore uses **PyTorch**, and the model artifact is
`model.pt` rather than the `model.keras` named in the original team
brief.

```bash
python3 -m venv .venv

# PyTorch first, from the CPU index (avoids ~2 GB of unused CUDA wheels)
.venv/bin/python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu torch torchvision

.venv/bin/python -m pip install -r requirements.txt
```

**Verify the environment:**

```bash
.venv/bin/python -c "import cv2, numpy, matplotlib, torch, torchvision; print('OK')"
```

**Lint everything** (must be clean — this is the project's "norminette").
Default settings, i.e. 79 columns; do not relax it with
`--max-line-length`:

```bash
.venv/bin/python -m flake8 . --exclude=.venv,augmented_directory
```

### Build a small test set

Most tests below are fast on the full dataset, but a mini set makes
iteration quicker. This one is deliberately **imbalanced**, so it
exercises the class-balancing code:

```bash
MINI=/tmp/mini
rm -rf "$MINI"
for d in leaves/images/*/; do
  c=$(basename "$d"); mkdir -p "$MINI/$c"
  ls "$d" | head -25 | while read f; do cp "$d$f" "$MINI/$c/"; done
done
# make two classes rare
ls "$MINI/Apple_rust"    | tail -18 | while read f; do rm "$MINI/Apple_rust/$f"; done
ls "$MINI/Grape_healthy" | tail -12 | while read f; do rm "$MINI/Grape_healthy/$f"; done

for d in "$MINI"/*/; do echo -n "$(basename $d): "; ls "$d" | wc -l; done
```

---

## 3. `utils/dataset.py` — I/O and splitting

The shared foundation. Every other module reads and writes images through
this one, which is what keeps the RGB contract enforceable.

### `VALID_EXTENSIONS`
Tuple of accepted lowercase extensions: `.jpg`, `.jpeg`, `.png`. Matching
is case-insensitive, so `.JPG` and `.jpg` both pass.

### `_images_in(folder) -> list[str]`
Private. The image filenames sitting directly in `folder` — extension
check plus an `isfile` check, so a directory named `foo.jpg` is not
mistaken for an image.

### `list_images(root, include_root=False) -> list[tuple[str, str]]`

Walks `root` recursively and returns `[(path, label), ...]` **sorted by
path**.

- The **label is the name of the directory holding the image** (e.g.
  `Apple_healthy`). Labels live in the folder name and nowhere else.
- Images sitting loose in `root` are skipped by default — under a dataset
  root they are strays with no class directory to name them.
- `include_root=True` labels those images with `root`'s own name instead.
  `Distribution.py` and `Augmentation.py` fall back to it so that aiming
  them at a single class directory works; **training never uses it**, so
  the split behind the accuracy claim is unaffected.
- Sorting matters: it makes the downstream split deterministic.
- Raises `NotADirectoryError` if `root` isn't a directory.

```bash
.venv/bin/python -c "
from utils.dataset import list_images
items = list_images('leaves/images')
print('total:', len(items))
print('first:', items[0])
print('labels:', sorted({l for _, l in items}))

# a single class directory: empty by default, labelled with include_root
one = 'leaves/images/Apple_rust'
print('default    :', len(list_images(one)))
print('include_root:', len(list_images(one, include_root=True)),
      list_images(one, include_root=True)[0][1])
"
```
Expected: `total: 7221`, 8 labels, then `0` and `275 Apple_rust`.

**Error case:**
```bash
.venv/bin/python -c "
from utils.dataset import list_images
try: list_images('/nope')
except NotADirectoryError as e: print('raised correctly:', e)
"
```

### `load_image(path) -> np.ndarray`

Reads an image as **RGB uint8 `(H, W, 3)`**. This is the only place
BGR→RGB conversion happens. Raises `ValueError` if the file cannot be
decoded (missing, corrupt, or not an image).

```bash
.venv/bin/python -c "
from utils.dataset import load_image
img = load_image('leaves/images/Apple_healthy/image (1).JPG')
print(img.shape, img.dtype)
"
```
Expected: `(256, 256, 3) uint8`.

**Error case** — a text file with a `.JPG` name:
```bash
echo "not an image" > /tmp/bad.JPG
.venv/bin/python -c "
from utils.dataset import load_image
try: load_image('/tmp/bad.JPG')
except ValueError as e: print('raised correctly:', e)
"
```

### `save_image(img, path) -> None`

Writes an RGB uint8 array to disk, converting RGB→BGR on the way out.
Creates parent directories automatically. Raises `ValueError` for a
non-3D array, `IOError` if the write fails.

Note the argument order: **image first, path second**. Calling it the
other way round is the one mistake this signature invites, and it is what
broke `Transformation.py`'s batch mode until it was fixed.

**Round-trip test** — save then reload and compare. JPEG is lossy, so
compare with a tolerance rather than exact equality:

```bash
.venv/bin/python -c "
import numpy as np
from utils.dataset import load_image, save_image
img = load_image('leaves/images/Apple_healthy/image (1).JPG')
save_image(img, '/tmp/rt/out.JPG')
back = load_image('/tmp/rt/out.JPG')
print('shape match:', img.shape == back.shape)
print('mean abs diff:', float(np.abs(img.astype(int) - back).mean()))
"
```
A small non-zero diff is correct (JPEG compression). A **large** diff, or
red/blue looking swapped, would mean a channel-order bug.

### `class_counts(root, include_root=False) -> dict[str, int]`

`{label: number_of_images}`. Used by `Distribution.py`, and the quickest
way to see the imbalance.

```bash
.venv/bin/python -c "
from utils.dataset import class_counts
c = class_counts('leaves/images')
for k, v in sorted(c.items()): print(f'{k:<20}{v}')
print('total', sum(c.values()))
"
```
Expected: 8 classes, total 7221, ranging from `Apple_rust` (275) to
`Apple_healthy` (1640) — i.e. roughly 6× imbalance, which is why
balancing exists.

### `group_by_label(items) -> dict[str, list]`

`[(path, label), ...]` → `{label: [item, ...]}`, insertion-ordered. Small
shared helper; `split_dataset` builds its per-class buckets with it.

### `split_dataset(root, val_ratio=0.2, seed=42) -> (train, val)`

**Stratified, seeded** train/validation split. Returns two lists of
`(path, label)`.

- **Stratified**: every class contributes the same *proportion* to
  validation, so rare classes stay represented.
- **Seeded**: the same seed always produces the same split, which is what
  makes the ≥90% accuracy claim defensible at evaluation.
- **Never empties a class**: the held-out count is clamped so a class with
  ≥2 images always appears on both sides.
- Raises `ValueError` unless `0 < val_ratio < 1`.

```bash
.venv/bin/python -c "
from utils.dataset import split_dataset
tr, va = split_dataset('leaves/images', 0.2, 42)
print('train', len(tr), 'val', len(va))

# determinism: same seed -> identical split
tr2, va2 = split_dataset('leaves/images', 0.2, 42)
print('deterministic:', tr == tr2 and va == va2)

# different seed -> different split
tr3, _ = split_dataset('leaves/images', 0.2, 7)
print('seed changes split:', tr != tr3)

# no overlap
print('overlap:', len({p for p,_ in tr} & {p for p,_ in va}))

# stratification: per-class val fraction should all be ~0.2
from collections import Counter
ctr, cva = Counter(l for _,l in tr), Counter(l for _,l in va)
for k in sorted(cva):
    print(f'  {k:<20} val fraction {cva[k]/(ctr[k]+cva[k]):.3f}')
"
```
Expected: `train 5777 val 1444`, `deterministic: True`,
`seed changes split: True`, `overlap: 0`, and every per-class fraction
close to 0.200.

---

## 4. `utils/naming.py` — filename and label conventions

Nothing on disk records these rules, so they are defined once here.
`Augmentation.py` writes augmented names one image at a time, `train.py`
writes them in bulk while balancing, and `Distribution.py` reads plant
types back out of labels — one definition keeps all three agreeing.

| Name | Value |
|---|---|
| `AUG_NAMES` | `("Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion")` — title-case, in the order the subject lists them |
| `AUG_EXTENSION` | `".JPG"` — augmented copies are always JPEG, whatever the source was |

### `stem(path) -> str`
`a/b/image (1).JPG` → `image (1)`. Basename without the extension.

### `augmented_filename(source, aug, round_id=0) -> str`

`image (1).JPG` + `Flip` → `image (1)_Flip.JPG`.

`round_id` separates the second and later passes over the same
`(image, augmentation)` pair, which balancing must make once a class is
small enough that its sources run out. Round 0 keeps the plain name the
subject shows; later rounds gain a number (`image (1)_Flip1.JPG`), so no
earlier file is ever silently overwritten. Raises `ValueError` for an
unknown augmentation or a negative round.

### `augmented_path(source, aug, dst_dir=None, round_id=0) -> str`
Full path for the copy. With no `dst_dir` it lands beside its source —
what Part 2 does when handed a single image.

### `split_augmented(filename) -> (stem, aug | None)`
Inverse of `augmented_filename`. Class labels contain underscores of
their own (`Apple_Black_rot`), so the trailing segment counts only when
it really names one of the six augmentations.

### `is_augmented(filename) -> bool`
True when the augmenter produced this file.

### `plant_type(label) -> str`
`Apple_Black_rot` → `Apple`. Whatever precedes the first underscore; a
label without one is its own plant type. Part 1 groups its charts by it.

```bash
.venv/bin/python -c "
from utils.naming import (AUG_NAMES, augmented_filename, augmented_path,
                          split_augmented, is_augmented, plant_type, stem)
print(stem('a/b/image (1).JPG'))
print(augmented_filename('image (1).JPG', 'Flip'))
print(augmented_filename('image (1).JPG', 'Flip', 2))
print(augmented_path('leaves/x/image (1).JPG', 'Crop'))
for f in ['image (1).JPG', 'image (1)_Flip.JPG', 'image (1)_Flip2.JPG',
          'Apple_Black_rot.JPG']:
    print(f'{f:<24} {split_augmented(f)}  augmented={is_augmented(f)}')
print(plant_type('Apple_Black_rot'), plant_type('Grape_spot'), plant_type('x'))
try: augmented_filename('x.JPG', 'Nope')
except ValueError as e: print('raised correctly:', e)
"
```
Note `Apple_Black_rot.JPG` must come back as *not* augmented — `rot` is
part of the label, not an augmentation tag.

---

## 5. `utils/preprocess.py` — the single preprocessing path

The most correctness-critical file in the project. One function, shared
by training and prediction.

### Constants

| Name | Value | Why |
|---|---|---|
| `IMAGE_SIZE` | `(128, 128)` | Dataset is 256×256; 128 keeps lesions clearly visible while making CPU training ~4× cheaper. |
| `MEAN` / `STD` | ImageNet stats | The backbone is pretrained on ImageNet, so inputs must match its expected distribution. |
| `PREPROCESS_NAME` | `resize128_rgb_float_imagenet_norm_v1` | Written into `labels.json`. Bump it if anything changes, so an old model can never be silently paired with new preprocessing. |

### `preprocess(img_rgb) -> np.ndarray`

RGB uint8 `(H, W, 3)` → float32 `(3, 128, 128)`, ready for the model.

Four steps, in order:
1. Resize to `IMAGE_SIZE` with `INTER_AREA` (the correct choice for
   downscaling).
2. Scale to `[0, 1]`.
3. Normalise with the ImageNet mean/std.
4. Transpose to channels-first, which is what torch expects.

Raises `ValueError` for a non-3D array or a non-3-channel image.

```bash
.venv/bin/python -c "
import numpy as np
from utils.dataset import load_image
from utils.preprocess import preprocess, IMAGE_SIZE

img = load_image('leaves/images/Apple_healthy/image (1).JPG')
out = preprocess(img)
print('in ', img.shape, img.dtype)
print('out', out.shape, out.dtype)          # (3, 128, 128) float32
print('range %.2f..%.2f' % (out.min(), out.max()))
print('mean %.2f' % out.mean())
print('contiguous:', out.flags['C_CONTIGUOUS'])

# deterministic: same input -> identical output
print('deterministic:', np.array_equal(out, preprocess(img)))

# any input size works
small = np.zeros((64, 100, 3), dtype=np.uint8)
print('resizes anything:', preprocess(small).shape)
"
```
Expected `(3, 128, 128) float32` and `deterministic: True`.

Note the values are **not** in `0..1` — normalisation shifts them. The
theoretical bound is −2.12..2.64 (that is `(0−mean)/std` to
`(1−mean)/std`); any single image occupies part of that, e.g. −1.19..2.03
with mean 0.26 for the image above. A leaf photo is mostly mid-tone green
on a light background, so the mean sits above zero rather than at it.
Output pinned to `0..1` would mean the normalisation step was skipped.

**Error cases:**
```bash
.venv/bin/python -c "
import numpy as np
from utils.preprocess import preprocess
for bad, why in [(np.zeros((10,10)), '2D'), (np.zeros((10,10,4)), '4 channels')]:
    try: preprocess(bad)
    except ValueError as e: print(f'{why}: raised correctly')
"
```

**The regression test that matters most** — train and predict must agree.
This asserts both programs use the very same function:

```bash
.venv/bin/python -c "
import numpy as np, train, predict
from utils.dataset import load_image
img = load_image('leaves/images/Apple_rust/image (1).JPG')
a = train.preprocess(img)
b = predict.preprocess(img)
print('train/predict preprocessing identical:', np.array_equal(a, b))
print('same function object:', train.preprocess is predict.preprocess)
"
```
Both must be `True`.

---

## 6. `Distribution.py` — Part 1

```bash
./Distribution.py <dir>
```

Counts the images per class and draws a **pie chart plus a bar chart per
plant type**. Every name on the charts comes from a directory name and
nowhere else, so the program describes whatever subtree it is pointed at:
the whole dataset, one plant, or a single class directory.

### Backend selection (top of file)

`matplotlib.use("Agg")` is called **before** `pyplot` is imported, when no
`DISPLAY` is set — on a headless machine the default backend raises
instead of drawing. This is why the imports below it carry `# noqa: E402`.
`Augmentation.py`, `Transformation.py` and `predict.py` open the same way.

### Colours

`PALETTE` holds eight categorical slots checked against colour-vision
deficiency simulation. The **first four stay distinguishable in every
simulation**, even as neighbouring pie slices, and four is what this
dataset needs per plant; beyond that only neighbours are guaranteed. That
is why every chart also carries a written label and a legend — identity
never rests on colour alone.

### `collect_counts(root) -> dict[str, int]`
`class_counts(root)`, falling back to `class_counts(root,
include_root=True)` when the walk finds nothing nested. That fallback is
what makes a single class directory a valid argument.

### `group_by_plant(counts) -> dict[str, dict[str, int]]`
`{plant: {label: count}}`, both levels sorted by name, keyed via
`naming.plant_type`.

### `colors_for(count) -> list[str]`
One colour per class, cycling `PALETTE` in a fixed order that never
shifts between runs.

### `_style_axis` / `_draw_pie` / `_draw_bar`
Private drawing helpers. `_style_axis` applies the shared look (muted
title, no box, ticks without marks); `_draw_pie` returns the wedges so the
figure legend can reuse them; `_draw_bar` annotates each bar with its
exact count.

### `draw_plant(plant, counts) -> Figure`
The two-panel figure for one plant, with a shared legend beneath.

### `report(groups)`
Prints the counts, so the analysis survives with no display attached.

### `build_parser()` / `main()`

| Flag | Purpose |
|---|---|
| `directory` | dataset directory to analyse (required) |
| `--save-dir DIR` | also write each figure to `DIR` as a PNG |
| `--no-display` | do not open a window (useful over ssh) |

`main()` returns `1` with a one-line `error: ...` on a missing directory
or one containing no images.

```bash
# whole dataset, written to PNGs instead of windows
.venv/bin/python Distribution.py leaves/images --save-dir /tmp/charts --no-display
ls /tmp/charts

# any subtree works, including one class directory
.venv/bin/python Distribution.py leaves/images/Apple_rust --no-display

# error paths (exit 1, no traceback)
.venv/bin/python Distribution.py /nope;  echo "exit=$?"
.venv/bin/python Distribution.py docs;   echo "exit=$?"
```
Expected on the full set: `Apple: 3164 images across 4 classes`,
`Grape: 4057 images across 4 classes`, and two PNGs in `/tmp/charts`.

---

## 7. `Augmentation.py` — Part 2

```bash
./Augmentation.py "<image>"                       # one image, six variants
./Augmentation.py -src <dir> -dst augmented_directory   # balance a dataset
```

`train.py` imports the six functions and `build_augmented_dir` from here,
so the images the model learns from come from exactly this code.

### Constants

- `SEED = 42` — the project seed; `train.py` imports it from here.
- `BORDER = cv2.BORDER_REFLECT_101` — every geometric augmentation
  reflects at the border rather than padding with black. A flat black
  wedge is a feature no real leaf photo has, and the network would happily
  learn it as a shortcut to whichever classes needed the most augmenting.

### `_rng(rng)` / `_warp(img, matrix)`
Private helpers. `_rng` falls back to `random.Random(SEED)` when no
generator is handed in. `_warp` applies a geometric matrix at the original
frame size — a 2×3 matrix goes to `warpAffine`, a 3×3 one to
`warpPerspective`, so the shape of the matrix picks the call.

### The six augmentations

Each takes `(img, rng=None)` and returns RGB uint8 of the **same shape**.

| Function | What it does |
|---|---|
| `flip(img, rng)` | Horizontal mirror. The only deterministic one. |
| `rotate(img, rng)` | Rotation about the centre, random angle in ±30°. |
| `skew(img, rng)` | Perspective tilt, top edge pinched inward 8–20%. |
| `shear(img, rng)` | Affine shear, factor ±0.25, recentred. |
| `crop(img, rng)` | Random 70–88% window, resized back up. |
| `distortion(img, rng)` | Barrel/pincushion lens distortion via `cv2.remap`. |

`AUGMENTATIONS` maps each title-case name to its function.

**Test — all six preserve shape and dtype, and actually change the image:**
```bash
.venv/bin/python -c "
import random, numpy as np
from utils.dataset import load_image
from Augmentation import AUGMENTATIONS
img = load_image('leaves/images/Apple_healthy/image (1).JPG')
rng = random.Random(42)
for name, fn in AUGMENTATIONS.items():
    out = fn(img, rng)
    print(f'{name:<12} {str(out.shape):<16} {out.dtype}  changed={not np.array_equal(out, img)}')
"
```
All six must keep `(256, 256, 3) uint8` and report `changed=True`.

**Visual check** — write them out and look at them:
```bash
.venv/bin/python -c "
import random
from utils.dataset import load_image, save_image
from Augmentation import AUGMENTATIONS
img = load_image('leaves/images/Apple_healthy/image (1).JPG')
rng = random.Random(42)
for name, fn in AUGMENTATIONS.items():
    save_image(fn(img, rng), f'/tmp/aug/{name}.JPG')
print('wrote /tmp/aug/')
"
ls /tmp/aug/
```

### `augment_image(img_rgb, rng=None) -> dict[str, np.ndarray]`

All six variants at once, `{name: image}` — the API shape the team brief
specifies.

```bash
.venv/bin/python -c "
from utils.dataset import load_image
from Augmentation import augment_image
out = augment_image(load_image('leaves/images/Apple_rust/image (1).JPG'))
print(sorted(out)); print(len(out), 'variants')
"
```

### `save_variants(source, variants, dst_dir=None)` / `show_variants(...)`
Single-image mode. `save_variants` writes each variant as
`<base>_<Aug>.JPG` beside the source (or into `dst_dir`) and returns the
paths; `show_variants` lays the original and the six variants out on one
row.

### `_fill_class(paths, class_dir, target, rng) -> list[str]`

Private. Augments `paths` into `class_dir` until it holds `target`
images. Sources and augmentations are walked in step, so copies stay
spread over both instead of piling onto one image or one effect.

**Round numbering uses `math.lcm(len(paths), 6)`.** A
`(source, augmentation)` pair repeats every `lcm(n_paths, 6)` steps, *not*
every `n_paths × 6` steps. Using the wrong period lets a later file
silently overwrite an earlier one, which under-fills precisely the rarest
classes — the ones balancing exists to help.

### `build_augmented_dir(items, out_dir, seed=SEED) -> list`

Copies every image in `items` into `out_dir/<label>/`, then fills the
smaller classes up to the largest.

- Wipes `out_dir` first, so runs are reproducible rather than cumulative.
- `items` is `[(path, label)]` — **only the images handed in** reach the
  output. That is what lets `train.py` balance its training split alone
  and leave the validation split untouched.
- **Self-checking**: after each class it counts the files actually on disk
  and raises `RuntimeError` if that disagrees with the target. The printed
  `220 -> 1312` lines are measured counts, not intentions.

```bash
MINI=/tmp/mini   # from section 2
.venv/bin/python -c "
from Augmentation import build_augmented_dir
from utils.dataset import split_dataset
tr, _ = split_dataset('$MINI', 0.2, 42)
print('balanced total:', len(build_augmented_dir(tr, '/tmp/aug_test')))
"
for d in /tmp/aug_test/*/; do echo -n "$(basename $d): "; ls "$d" | wc -l; done
```
Every class must show the **same** count. The rare classes
(`Apple_rust`, `Grape_healthy`) are the ones to watch — they exercise the
lcm logic.

**Uniqueness check** (guards against the collision bug returning):
```bash
.venv/bin/python -c "
import os
for cls in sorted(os.listdir('/tmp/aug_test')):
    fs = os.listdir(f'/tmp/aug_test/{cls}')
    assert len(fs) == len(set(fs)), cls
print('all filenames unique within each class')
"
```

### `build_parser()` / `run_single()` / `run_balance()` / `main()`

| Flag | Default | Purpose |
|---|---|---|
| `image` | — | a single image to augment and display |
| `-src DIR` | — | dataset directory to balance |
| `-dst DIR` | `augmented_directory` | where the balanced set is written |
| `--seed N` | 42 | random seed |
| `--no-display` | off | do not open a window |

Giving both an image and `-src`, or neither, is an argparse error.
`main()` returns `1` with a one-line `error: ...` for a missing file, a
missing directory, or an empty one.

```bash
# single image: prints the six filenames it wrote
cp "leaves/images/Apple_rust/image (1).JPG" /tmp/one/ 2>/dev/null || \
  { mkdir -p /tmp/one && cp "leaves/images/Apple_rust/image (1).JPG" /tmp/one/; }
.venv/bin/python Augmentation.py "/tmp/one/image (1).JPG" --no-display
ls /tmp/one

# whole dataset
.venv/bin/python Augmentation.py -src /tmp/mini -dst /tmp/bal --no-display | tail -3

# error paths
.venv/bin/python Augmentation.py;                echo "exit=$?"   # expect 2 (argparse)
.venv/bin/python Augmentation.py x.JPG -src y;   echo "exit=$?"   # expect 2 (argparse)
.venv/bin/python Augmentation.py /nope.JPG;      echo "exit=$?"   # expect 1
```

---

## 8. `Transformation.py` — Part 3

```bash
./Transformation.py <image>                              # display
./Transformation.py -src <dir> -dst <dir> [-mask ...]    # batch save
./Transformation.py -h
```

Six transformations built on OpenCV and NumPy, in the spirit of the
PlantCV pipeline the subject shows. `predict.py` imports one function
from it, for display only — it is **not** part of the training path.

### `leaf_outline(img_rgb) -> (mask, contour)`

Separates the leaf from its light, fairly uniform background.

Saturation does the work: leaf tissue is coloured and the background is
not, so an **Otsu threshold over the HSV saturation channel** splits them
with no hand-tuned constant. Morphological close/open fills the holes left
by specular highlights, and keeping only the **largest contour** drops the
stray flecks that survive. Returns the filled mask and that contour
together — every caller needs both, and `contour` is `None` when nothing
was found, which each drawing helper handles by returning the image
untouched.

### The rendering helpers

| Function | Output |
|---|---|
| `_roi_objects(img, mask, contour)` | Leaf cut out of its background, red bounding box. |
| `_analyze_object(img, contour)` | Green outline, red centroid cross, `area=… perim=…` caption. |
| `_pseudolandmarks(img, contour, points=30)` | 30 points spaced along the outline, coloured top / middle / bottom. |

### `transform_image(img_rgb) -> dict[str, np.ndarray]`

The full pipeline. Keys: `original`, `gaussian_blur`, `mask`,
`roi_objects`, `analyze_object`, `pseudolandmarks` — all RGB uint8 of the
input's shape.

### `color_histogram(img_rgb) -> matplotlib.Figure`
Per-channel RGB plus HSV-saturation histograms (Figure IV.7 in the
subject). Counts are turned into proportions, so images of different sizes
stay comparable.

### `transformed_for_display(img_rgb) -> np.ndarray`
The single rendering `predict.py` shows beside the original: the
`analyze_object` view, which carries the most information at a glance.

```bash
.venv/bin/python -c "
from utils.dataset import load_image
from Transformation import transform_image, transformed_for_display, leaf_outline
img = load_image('leaves/images/Apple_scab/image (1).JPG')
t = transform_image(img)
print('keys:', sorted(t))
for k, v in t.items(): print(f'  {k:<18}{v.shape} {v.dtype}')
mask, contour = leaf_outline(img)
print('mask coverage: %.1f%%' % (100 * (mask > 0).mean()))
print('display:', transformed_for_display(img).shape)
"
```

### `requested(args) -> (keys, include_histogram)`
Which transformations this run wants. With **no** per-transformation flag
the answer is everything plus the histogram, which makes the plain
`-src/-dst` form do the obvious thing.

### `display_single(img_rgb, keys, include_histogram)`
Original plus every requested transformation on a 3-column grid, with the
histogram as a second figure.

### `process_batch(src, dst, keys, include_histogram)`
Lists images with `utils.dataset.list_images(src, include_root=True)` —
recursive, and it works when `-src` is a single class directory. Each
result is saved as `<base>_<Transformation><ext>`; the histogram goes to
`<base>_Histogram.png`. An unreadable file is skipped with a warning
rather than aborting the run.

> **Fixed:** batch mode used to call `save_image(path, img)` while
> `utils.dataset.save_image` takes `(img, path)`, so `-src/-dst` crashed
> on every invocation. The arguments are now the right way round, and the
> local I/O fallback that hid the mismatch is gone.

### `build_parser()` / `main()`

| Flag | Purpose |
|---|---|
| `image` | single image to transform and display |
| `-src` / `-dst` | batch mode: source and destination directories |
| `-blur` | include the Gaussian blur |
| `-mask` | include the leaf mask |
| `-roi` | include the ROI objects |
| `-object` | include the analyzed object |
| `-landmarks` | include the pseudolandmarks |
| `-histogram` | include the colour histogram |

Batch mode forces the `Agg` backend — nothing is displayed, and a machine
with no display would refuse to open a window at all. `main()` returns `1`
with a one-line `error: ...` for a missing file or directory, and prints
the help (exit `1`) when given no arguments at all.

```bash
# batch: 6 transforms + histogram for every image
.venv/bin/python Transformation.py -src /tmp/mini/Apple_rust -dst /tmp/tr_all | tail -2
ls /tmp/tr_all | head -6

# a single transform
.venv/bin/python Transformation.py -src /tmp/mini/Apple_rust -dst /tmp/tr_mask -mask | tail -1

# display mode (writes nothing; headless prints a matplotlib warning)
.venv/bin/python Transformation.py "leaves/images/Apple_healthy/image (1).JPG"

# error paths
.venv/bin/python Transformation.py /nope.JPG;               echo "exit=$?"
.venv/bin/python Transformation.py -src /nope -dst /tmp/x;  echo "exit=$?"
```
With 12 source images, `/tmp/tr_all` must hold 72 files: five saved
renderings plus one histogram each (`original` is displayed, never
written).

---

## 9. `train.py` — Part 4, training

```bash
./train.py <dir>
```

Pipeline: split → balance the training split only → fine-tune ResNet-18 →
evaluate on the held-out split → write `learnings.zip`.

> **The ordering is the whole design.** The split happens *before*
> augmentation. Balancing the full dataset first and splitting afterwards
> would scatter augmented copies of validation images into training —
> accuracy would look excellent and mean nothing. This is exactly the
> "results shouldn't look suspicious" failure the subject warns about.

The six augmentations and the balancing pass are **imported from
`Augmentation.py`** (§7), together with `SEED`.

### `class LeafDataset(items, classes)`

A torch `Dataset`. `items` is `[(path, label)]`; `classes` is the
index-ordered class list. `__getitem__` loads via `load_image`, runs
`preprocess`, and returns `(tensor, class_index)`.

The `classes` list is what ties label strings to output indices — the
same list is written to `labels.json`, which is why `predict.py` can map
indices back to names correctly.

```bash
.venv/bin/python -c "
from train import LeafDataset
from utils.dataset import split_dataset
tr, _ = split_dataset('leaves/images', 0.2, 42)
classes = sorted({l for _, l in tr})
ds = LeafDataset(tr, classes)
x, y = ds[0]
print('len', len(ds)); print('x', x.shape, x.dtype); print('y', y, '->', classes[y])
"
```
Expected: `x torch.Size([3, 128, 128]) torch.float32`, and the label name
must match the directory the first image came from.

### `build_model(num_classes)`

ImageNet-pretrained ResNet-18 with the 1000-class head replaced by a
fresh `Linear(512, num_classes)`. Transfer learning is what makes ≥90%
reachable in a handful of CPU epochs.

```bash
.venv/bin/python -c "
from train import build_model
m = build_model(8)
print('output features:', m.fc.out_features)
import torch
print('forward:', m(torch.zeros(2, 3, 128, 128)).shape)   # (2, 8)
"
```
First run downloads ~45 MB of weights and caches them in
`~/.cache/torch/`.

### `evaluate(model, loader, num_classes, device)`

Returns `(accuracy, images scored, confusion_matrix)`. Runs under
`torch.no_grad()` in `eval()` mode. The confusion matrix is
`confusion[true][pred]`, so **rows are ground truth, columns are
predictions**; accuracy is its trace over its sum, so the two numbers
cannot disagree.

### `run_epoch(model, loader, criterion, optimiser, device, epoch)`
One pass over the training set, printing a running loss every 20 steps.
Returns the mean loss.

### `train_model(model, train_loader, val_loader, classes, device, epochs)`

The training loop. AdamW (`lr=3e-4`, `weight_decay=1e-4`) with a cosine
LR schedule, cross-entropy loss.

After every epoch it evaluates on the held-out set and **keeps a clone of
the weights from the best epoch**, which are reloaded at the end. So the
saved model is the best one seen, not merely the last. Returns a dict:
`accuracy`, `state`, `confusion`, `count`, `epoch`.

### `per_class_recall(classes, confusion)` / `_write_json(path, payload)`
Small helpers: the per-class `support`/`correct`/`recall` table read off
the confusion rows, and the two-line JSON writer used for both metadata
files.

### `write_artifacts(model, classes, best, work_dir, aug_dir, zip_path)`

Writes the three metadata files into `work_dir` (default `learnings/`),
then zips them together with `augmented_directory/`.

| File | Contents |
|---|---|
| `model.pt` | `state_dict`, `architecture`, `num_classes` |
| `labels.json` | `classes` (index-ordered), `input_size`, `preprocess` |
| `metrics.json` | `val_accuracy`, `val_count`, `best_epoch`, `classes`, `per_class` (support/correct/recall), `confusion` |

`classes` order **must** match the model's output index order — it is the
same list used to build `LeafDataset`, so it does by construction.

### `seed_everything(seed)`
Seeds Python, numpy and torch in one call.

### `build_parser()` / `main()`

| Flag | Default | Purpose |
|---|---|---|
| `directory` | — | dataset root, `<dir>/<class>/*.JPG` |
| `--epochs` | 6 | training epochs |
| `--batch-size` | 64 | mini-batch size |
| `--val-ratio` | 0.2 | validation fraction |
| `--augmented-dir` | `augmented_directory` | where balanced images go |
| `--out` | `learnings.zip` | output archive |
| `--workers` | 4 | data-loading processes |

`main()` returns `1` (and prints to stderr) on: a non-directory argument,
an empty dataset, or a final accuracy below 90%. It warns if the
validation set has fewer than 100 images.

**Fast smoke test** (~1 min on the mini set):
```bash
.venv/bin/python train.py /tmp/mini --epochs 1 --batch-size 16 --workers 2 \
    --augmented-dir /tmp/aug_mini --out /tmp/mini_learnings.zip
```
Note this rewrites `learnings/` in the current directory — run it from a
scratch copy if you want to keep the real artifacts.

**Error handling:**
```bash
.venv/bin/python train.py /nope;        echo "exit=$?"   # expect 1
mkdir -p /tmp/empty
.venv/bin/python train.py /tmp/empty;   echo "exit=$?"   # expect 1
```
Both must print a clean `error: ...` with no traceback.

**Full run** (~20 min, 12-core CPU):
```bash
.venv/bin/python -u train.py leaves/images --epochs 6 --batch-size 64 --workers 4
```

---

## 10. `predict.py` — Part 4, prediction

```bash
./predict.py <image>
```

Loads the model, classifies one image, displays original + transformed,
prints the class.

### `locate_artifacts(model_source, temp_dir)`

Returns the paths to `model.pt` and `labels.json`, extracting them from an
archive or reading them out of an unpacked folder. Raises `ValueError`
naming what is missing, or saying the source is neither a zip nor a
directory.

### `load_model(model_source, temp_dir) -> (model, classes)`

Accepts **either** a `learnings.zip` or an unpacked directory. Reads only
`model.pt` and `labels.json` — `metrics.json` and the augmented images are
training-time artifacts and are deliberately ignored.

Rebuilds a ResNet-18 with `len(classes)` outputs and loads the weights.
If `labels.json` records a different `preprocess` name than the current
build, it **warns to stderr** — the early alarm for a train/predict
mismatch.

```bash
.venv/bin/python -c "
import tempfile
from predict import load_model
with tempfile.TemporaryDirectory() as td:
    m, c = load_model('learnings.zip', td)
print('classes:', c)
print('head outputs:', m.fc.out_features)
print('eval mode:', not m.training)
"
```

### `predict(model, classes, img_rgb) -> (label, confidence)`

Preprocesses, adds a batch dimension, runs a softmax, returns the
top class and its probability.

```bash
.venv/bin/python -c "
import tempfile
from utils.dataset import load_image
from predict import load_model, predict
with tempfile.TemporaryDirectory() as td:
    m, c = load_model('learnings.zip', td)
    for cls in ['Apple_rust', 'Grape_Esca']:
        lab, conf = predict(m, c, load_image(f'leaves/images/{cls}/image (1).JPG'))
        print(f'{cls:<18} -> {lab:<18} {conf:.2%}  {\"OK\" if lab==cls else \"WRONG\"}')
"
```

### `transformed(img_rgb)`

Calls `Transformation.transformed_for_display`. Wrapped in `try/except`:
if that module is missing or fails, prediction still works and the
original is shown instead. Display is a presentation concern and must
never take down classification.

### `show(img_rgb, label, confidence, save_to=None)`

Builds the two-panel figure (original | transformed) titled
`Class predicted : <label>`. Behaviour:

- `--save-to PATH` → writes there.
- Headless (`Agg` backend) → falls back to `prediction.png`.
- Otherwise → opens an interactive window.

### `build_parser()` / `main()`

| Flag | Purpose |
|---|---|
| `image` | image to classify (required) |
| `--model` | `learnings.zip` or folder (default `learnings.zip`) |
| `--save-to` | write the figure to a file |
| `--no-display` | print the prediction only |

Returns `1` on a missing image, missing model, unreadable image, or
unloadable model.

**All 8 classes:**
```bash
for c in Apple_Black_rot Apple_healthy Apple_rust Apple_scab \
         Grape_Black_rot Grape_Esca Grape_healthy Grape_spot; do
  f=$(ls "leaves/images/$c" | head -1)
  echo -n "$c -> "
  .venv/bin/python predict.py "leaves/images/$c/$f" --no-display | head -1
done
```

**Figure output:**
```bash
.venv/bin/python predict.py "leaves/images/Grape_Esca/image (5).JPG" --save-to /tmp/pred.png
```

**Error handling** — none of these may produce a traceback:
```bash
.venv/bin/python predict.py /nope.JPG --no-display;                  echo "exit=$?"
.venv/bin/python predict.py "leaves/images/Apple_rust/image (1).JPG" \
    --model /nope.zip --no-display;                                  echo "exit=$?"
echo "junk" > /tmp/bad.JPG
.venv/bin/python predict.py /tmp/bad.JPG --no-display;               echo "exit=$?"
.venv/bin/python predict.py "leaves/images/Apple_rust/image (1).JPG" \
    --model /tmp/bad.JPG --no-display;                               echo "exit=$?"
```
All must exit `1` with a one-line `error: ...`.

---

## 11. End-to-end test recipe

The full sequence, from clean checkout to verified result.

```bash
# 1. environment
.venv/bin/python -c "import cv2, numpy, matplotlib, torch; print('OK')"

# 2. lint the whole repo
.venv/bin/python -m flake8 . --exclude=.venv,augmented_directory && echo "lint clean"

# 3. dataset sanity (Part 1)
.venv/bin/python Distribution.py leaves/images --no-display

# 4. augmentation (Part 2)
.venv/bin/python Augmentation.py "leaves/images/Apple_rust/image (1).JPG" --no-display

# 5. transformation (Part 3)
.venv/bin/python Transformation.py -src leaves/images/Apple_rust -dst /tmp/tr -mask | tail -1

# 6. train (~20 min) (Part 4)
.venv/bin/python -u train.py leaves/images --epochs 6 --batch-size 64 --workers 4

# 7. inspect the proof
unzip -p learnings.zip metrics.json | .venv/bin/python -m json.tool | head -20

# 8. verify archive layout
unzip -l learnings.zip | head -6

# 9. predict on every class
for c in $(ls leaves/images); do
  f=$(ls "leaves/images/$c" | head -1)
  echo -n "$c -> "; .venv/bin/python predict.py "leaves/images/$c/$f" --no-display | head -1
done
```

### Proving the accuracy is honest

A near-perfect score invites scrutiny, so these are the checks to run —
and to be able to reproduce on demand at defense.

**No leakage** (validation images must never appear in training):
```bash
.venv/bin/python -c "
import os
from utils.dataset import split_dataset
from utils.naming import split_augmented
tr, va = split_dataset('leaves/images', 0.2, 42)
val = {(l, os.path.basename(p)) for p, l in va}
trn = {(l, os.path.basename(p)) for p, l in tr}
print('train/val overlap:', len(val & trn))
leak = 0
for cls in os.listdir('augmented_directory'):
    for f in os.listdir(os.path.join('augmented_directory', cls)):
        if (cls, split_augmented(f)[0] + '.JPG') in val: leak += 1
print('val images inside augmented_directory:', leak)
"
```
Both must be `0`. (`split_augmented` is the same function that built those
names, so the check cannot drift from the convention.)

**Duplicates in the raw dataset** (present as delivered, not introduced
here — but you must know the number):
```bash
.venv/bin/python -c "
import hashlib, collections
from utils.dataset import split_dataset
tr, va = split_dataset('leaves/images', 0.2, 42)
side = {p: 'train' for p, _ in tr}; side.update({p: 'val' for p, _ in va})
h = collections.defaultdict(list)
for p in side: h[hashlib.md5(open(p,'rb').read()).hexdigest()].append(p)
cross = [g for g in h.values() if len(g) > 1 and len({side[x] for x in g}) > 1]
print('byte-identical pairs spanning train/val:', len(cross))
"
```

**Accuracy with near-duplicates removed** — the clean number:
```bash
.venv/bin/python -c "
import collections, cv2, tempfile
from utils.dataset import split_dataset, load_image
from predict import load_model, predict
tr, va = split_dataset('leaves/images', 0.2, 42)
side = {p: 'train' for p, _ in tr}; side.update({p: 'val' for p, _ in va})
def ahash(p):
    g = cv2.cvtColor(load_image(p), cv2.COLOR_RGB2GRAY)
    s = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    return (s > s.mean()).tobytes()
h = collections.defaultdict(list)
for p in side: h[ahash(p)].append(p)
tainted = {x for g in h.values() if len({side[y] for y in g}) > 1
           for x in g if side[x] == 'val'}
clean = [(p, l) for p, l in va if p not in tainted]
with tempfile.TemporaryDirectory() as td:
    m, c = load_model('learnings.zip', td)
    ok = sum(predict(m, c, load_image(p))[0] == l for p, l in clean)
print(f'clean val: {ok}/{len(clean)} = {ok/len(clean):.4f}')
"
```

Measured results on this dataset (`learnings/metrics.json`,
`val_accuracy: 1.0`, `best_epoch: 6`):

| Check | Result |
|---|---|
| Validation images | 1444 (requirement: ≥100) |
| Train/val overlap | 0 |
| Val images in `augmented_directory/` | 0 |
| Byte-identical duplicates spanning train/val | 3 pairs |
| Near-duplicates (aHash) spanning train/val | 17 val images (1.18%) |
| Accuracy, all near-duplicates removed | **100% (1427/1427)** |

### Refactoring safety net

Any change to this code can be checked against an earlier commit rather
than by eye. Three properties must survive a refactor: the **split** is
unchanged (so the accuracy claim still refers to the same images),
**balancing** writes byte-identical files, and **`transform_image`** is
pixel-identical.

```bash
# check out the reference commit beside the working tree
OLD=/tmp/old_leaffliction
rm -rf $OLD && mkdir -p $OLD && git archive <commit> | tar -x -C $OLD

# 1. the split
.venv/bin/python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('old_ds', '$OLD/utils/dataset.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
sys.path.insert(0, '.')
from utils import dataset as new
print('split identical:',
      old.split_dataset('leaves/images') == new.split_dataset('leaves/images'))
"

# 2. balancing (use the mini set from section 2)
(cd $OLD && /path/to/repo/.venv/bin/python Augmentation.py \
    -src /tmp/mini -dst /tmp/bal_old --no-display >/dev/null)
.venv/bin/python Augmentation.py -src /tmp/mini -dst /tmp/bal_new --no-display >/dev/null
diff -r /tmp/bal_old /tmp/bal_new && echo "balancing byte-identical"

# 3. the transformations
.venv/bin/python -c "
import importlib.util, sys, numpy as np
sys.path.insert(0, '.')
from utils.dataset import load_image
import Transformation as new
spec = importlib.util.spec_from_file_location('old_T', '$OLD/Transformation.py')
old = importlib.util.module_from_spec(spec)
sys.modules['old_T'] = old; spec.loader.exec_module(old)
img = load_image('leaves/images/Apple_rust/image (1).JPG')
a, b = old.transform_image(img), new.transform_image(img)
print({k: bool(np.array_equal(a[k], b[k])) for k in a})
"
```
All three passed for the simplification commit (`ff46f3e`), which is why
it could be called behaviour-preserving.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: torch` | Using system `python3` instead of `.venv/bin/python`. |
| `No matching distribution found for tensorflow` | Expected — no TF wheel for Python 3.14. This project uses PyTorch. |
| `error: no model at learnings.zip` | Run `train.py` first. |
| `warning: model was trained with preprocessing '...'` | `labels.json` disagrees with `utils/preprocess.py`. **Retrain** — do not ignore this; it means train/predict have diverged. |
| Predictions all one class | Almost always a preprocessing mismatch. Run the train/predict identity test in §5. |
| Colours look wrong (red/blue swapped) | A BGR/RGB flip outside `utils/dataset.py`. Conversion belongs there and nowhere else. |
| `save_image expects an (H, W, 3) ndarray` | Arguments swapped: it is `save_image(img, path)`, not `(path, img)`. |
| `RuntimeError: wrote N images but expected M` | The filename-collision guard fired in `build_augmented_dir`. |
| `UserWarning: FigureCanvasAgg is non-interactive` | Headless display mode. Expected; use `--no-display`, `--save-dir` or `--save-to`. |
| No window appears | Headless environment; `predict.py` wrote the figure to `prediction.png`. |
| Training very slow | Lower `--batch-size`, reduce `--epochs`, or lower `IMAGE_SIZE` (requires retraining). |
| First run stalls at `build_model` | Downloading ~45 MB of pretrained weights to `~/.cache/torch/`. |
