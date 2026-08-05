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
4. [`utils/preprocess.py`](#4-utilspreprocesspy--the-single-preprocessing-path)
5. [`train.py`](#5-trainpy--training-part-4)
6. [`predict.py`](#6-predictpy--prediction-part-4)
7. [`Transformation.py`](#7-transformationpy--person-bs-module)
8. [End-to-end test recipe](#8-end-to-end-test-recipe)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture at a glance

```
leaves/images/<class>/*.JPG          raw dataset (never committed)
        │
        ▼
utils/dataset.py                     list / load / save / count / split
        │
        ├──────────────► train.py ──► augmented_directory/  (balanced train set)
        │                    │
        │                    └──────► learnings.zip
        │                                 ├── model.pt
        │                                 ├── labels.json
        │                                 ├── metrics.json
        │                                 └── augmented_directory/
        │                                          │
utils/preprocess.py ◄────────────────────────────┐ │
   (imported by BOTH train and predict)          │ │
        │                                        │ ▼
        └──────────────► predict.py ◄────────────┴─┘
                              │
                              └──► Transformation.py (Person B, display only)
```

**The one rule that matters:** `utils/preprocess.py` is imported by both
`train.py` and `predict.py`. Images must be prepared identically at
training and prediction time; a mismatch silently destroys accuracy while
training still appears to converge. There is exactly one code path, so
the mismatch cannot happen.

**The image format contract**, everywhere in this project: numpy
`ndarray`, shape `(H, W, 3)`, dtype `uint8`, **RGB order**. OpenCV is
BGR, so the BGR↔RGB flip happens only inside `utils/dataset.py`.

---

## 2. Environment setup

This project runs on **Python 3.14**, for which TensorFlow publishes no
wheel. Part 4 therefore uses **PyTorch**, and the model artifact is
`model.pt` rather than the `model.keras` named in the team brief.

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

**Lint everything** (must be clean — this is the project's "norminette"):

```bash
.venv/bin/python -m flake8 train.py predict.py utils/*.py
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

**Owner:** Person A. This is a contract-faithful implementation written
by Person C to stay unblocked. The five public functions keep exactly the
signatures agreed in `00_TEAM_BRIEF.md`, so A's real module drops in
without changing any calling code.

### `VALID_EXTENSIONS`
Tuple of accepted lowercase extensions: `.jpg`, `.jpeg`, `.png`. Matching
is case-insensitive via `_is_image`.

### `_is_image(filename) -> bool`
Private. True when the filename ends with a valid extension, compared
lowercase so `.JPG` and `.jpg` both pass.

### `list_images(root) -> list[tuple[str, str]]`

Walks `root` recursively and returns `[(path, label), ...]` **sorted by
path**.

- The **label is the name of the directory holding the image** (e.g.
  `Apple_healthy`). Labels live in the folder name and nowhere else.
- Images sitting directly in `root` are skipped — they have no class
  directory, so they have no label.
- Sorting matters: it makes the downstream split deterministic.
- Raises `NotADirectoryError` if `root` isn't a directory.

```bash
.venv/bin/python -c "
from utils.dataset import list_images
items = list_images('leaves/images')
print('total:', len(items))
print('first:', items[0])
print('labels:', sorted({l for _, l in items}))
"
```
Expected: `total: 7221`, 8 labels.

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

### `class_counts(root) -> dict[str, int]`

`{label: number_of_images}`. Used by Person A's `Distribution.py` and
useful for spotting imbalance.

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

### `split_dataset(root, val_ratio=0.2, seed=42) -> (train, val)`

**Stratified, seeded** train/validation split. Returns two lists of
`(path, label)`.

- **Stratified**: every class contributes the same *proportion* to
  validation, so rare classes stay represented.
- **Seeded**: the same seed always produces the same split, which is what
  makes the ≥90% accuracy claim defensible at evaluation.
- **Never empties a class**: `n_val` is clamped so a class with ≥2 images
  always appears on both sides.
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

## 4. `utils/preprocess.py` — the single preprocessing path

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
This asserts both programs produce byte-identical tensors:

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

## 5. `train.py` — training (Part 4)

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

### Constants

- `SEED = 42` — seeds Python, numpy and torch for reproducibility.
- `AUG_NAMES` — `("Flip", "Rotate", "Skew", "Shear", "Crop",
  "Distortion")`, the six augmentation types required by the subject.

### The six augmentation functions

Each takes `(img, rng)` and returns RGB uint8 of the **same shape**. All
use `BORDER_REFLECT_101` so geometric transforms don't introduce black
borders that the model could learn as a shortcut.

| Function | What it does |
|---|---|
| `_flip(img, rng)` | Horizontal mirror. The only deterministic one. |
| `_rotate(img, rng)` | Rotation, random angle in ±30°. |
| `_skew(img, rng)` | Perspective warp, top edge pinched inward 8–20%. |
| `_shear(img, rng)` | Affine shear, factor ±0.25, recentred. |
| `_crop(img, rng)` | Random crop keeping 70–88%, resized back up. |
| `_distortion(img, rng)` | Barrel/pincushion lens distortion via `cv2.remap`. |

`AUGMENTATIONS` maps each name to its function.

**Test — all six preserve shape and dtype, and actually change the image:**
```bash
.venv/bin/python -c "
import random, numpy as np
from utils.dataset import load_image
from train import AUGMENTATIONS
img = load_image('leaves/images/Apple_healthy/image (1).JPG')
rng = random.Random(42)
for name, fn in AUGMENTATIONS.items():
    out = fn(img, rng)
    changed = not np.array_equal(out, img)
    print(f'{name:<12} {str(out.shape):<16} {out.dtype}  changed={changed}')
"
```
All six must keep `(256, 256, 3) uint8` and report `changed=True`.

**Visual check** — write them out and look at them:
```bash
.venv/bin/python -c "
import random
from utils.dataset import load_image, save_image
from train import AUGMENTATIONS
img = load_image('leaves/images/Apple_healthy/image (1).JPG')
rng = random.Random(42)
for name, fn in AUGMENTATIONS.items():
    save_image(fn(img, rng), f'/tmp/aug/{name}.JPG')
print('wrote /tmp/aug/')
"
ls /tmp/aug/
```

### `augment_image(img_rgb, rng=None) -> dict[str, np.ndarray]`

Returns all six variants at once, `{name: image}`. This is the API shape
the team brief specifies for Person A's augmentation module.

```bash
.venv/bin/python -c "
from utils.dataset import load_image
from train import augment_image
out = augment_image(load_image('leaves/images/Apple_rust/image (1).JPG'))
print(sorted(out)); print(len(out), 'variants')
"
```

### `build_augmented_dir(train_items, out_dir, seed=SEED) -> list`

Copies every training image into `out_dir/<label>/`, then augments the
minority classes until **every class matches the largest one**.

- Wipes `out_dir` first, so runs are reproducible rather than cumulative.
- Only images from the **training** split ever reach this directory.
- Naming follows Person A's contract: `<base>_<Aug>.JPG`.
- **Round numbering uses `math.lcm(len(paths), 6)`.** A
  `(source, augmentation)` pair repeats every `lcm(n_paths, 6)` steps,
  *not* every `n_paths × 6` steps. Using the wrong period lets a later
  file silently overwrite an earlier one, which under-fills precisely the
  rarest classes — the ones balancing exists to help. Extra rounds get a
  numeric suffix (`_Flip1.JPG`).
- **Self-checking**: after each class it counts the files actually on disk
  and raises `RuntimeError` if that disagrees with the target. The printed
  `220 -> 1312` lines are measured counts, not intentions.

```bash
MINI=/tmp/mini   # from section 2
.venv/bin/python -c "
from train import build_augmented_dir
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

Returns `(accuracy, total, confusion_matrix)`. Runs under
`torch.no_grad()` in `eval()` mode. The confusion matrix is
`confusion[true][pred]`, so **rows are ground truth, columns are
predictions**.

### `train_model(model, train_loader, val_loader, classes, device, epochs)`

The training loop. AdamW (`lr=3e-4`, `weight_decay=1e-4`) with a cosine
LR schedule, cross-entropy loss.

After every epoch it evaluates on the held-out set and **keeps a clone of
the weights from the best epoch**, which are reloaded at the end. So the
saved model is the best one seen, not merely the last. Returns a dict:
`accuracy`, `state`, `confusion`, `count`, `epoch`.

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

## 6. `predict.py` — prediction (Part 4)

```bash
./predict.py <image>
```

Loads the model, classifies one image, displays original + transformed,
prints the class.

### Backend selection (top of file)

`matplotlib.use("Agg")` is called **before** `pyplot` is imported, when no
`DISPLAY` is set. This ordering is required: importing `Transformation`
pulls in `pyplot`, and the backend cannot be changed afterwards. This is
why the imports below it carry `# noqa: E402`.

### `_load_from_zip(zip_path, temp_dir)` / `_load_from_dir(directory)`

Private. Locate `model.pt` and `labels.json` — extracting from an archive,
or reading an unpacked folder. Both raise `ValueError` naming what is
missing.

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

### `_transformed(img_rgb)`

Calls Person B's `transformed_for_display`. Wrapped in `try/except`: if
B's module is missing or fails, prediction still works and the original
is shown instead. Display is a presentation concern and must never take
down classification.

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

## 7. `Transformation.py` — Person B's module

Owned by Person B. Person C imports one function from it, for display
only — it is **not** part of the training path.

### Functions C depends on

- **`transform_image(img_rgb) -> dict`** — the full pipeline. Keys:
  `original`, `gaussian_blur`, `mask`, `roi_objects`, `analyze_object`,
  `pseudolandmarks`. All values are RGB uint8 of the input's shape.
- **`transformed_for_display(img_rgb) -> np.ndarray`** — the single
  "nice-looking" rendering `predict.py` shows: the `analyze_object` view
  (contour outline, centroid cross, area/perimeter overlay).
- **`color_histogram(img_rgb) -> matplotlib.Figure`** — per-channel RGB +
  HSV-saturation histograms.

Internally, `_leaf_mask` isolates the leaf via HSV saturation + Otsu
thresholding, morphological cleanup, and largest-contour selection;
`_largest_contour` picks the biggest contour by area.

```bash
.venv/bin/python -c "
from utils.dataset import load_image
from Transformation import transform_image, transformed_for_display
img = load_image('leaves/images/Apple_scab/image (1).JPG')
t = transform_image(img)
print('keys:', sorted(t))
for k, v in t.items(): print(f'  {k:<18}{v.shape} {v.dtype}')
d = transformed_for_display(img)
print('display:', d.shape, d.dtype)
"
```

### CLI (Person B's part 3)

```bash
./Transformation.py -h
./Transformation.py "leaves/images/Apple_healthy/image (1).JPG"        # display
./Transformation.py -src leaves/images/Apple_healthy -dst out -mask    # batch
```

> **Resolved:** the two over-length lines here were wrapped, and the
> team no longer declares `--max-line-length=100`. The whole repo passes
> `flake8` on default settings, which is the norm the subject specifies.

---

## 8. End-to-end test recipe

The full sequence, from clean checkout to verified result.

```bash
# 1. environment
.venv/bin/python -c "import cv2, numpy, matplotlib, torch; print('OK')"

# 2. lint
.venv/bin/python -m flake8 train.py predict.py utils/*.py && echo "lint clean"

# 3. dataset sanity
.venv/bin/python -c "
from utils.dataset import class_counts, split_dataset
print(class_counts('leaves/images'))
tr, va = split_dataset('leaves/images', 0.2, 42)
print('train', len(tr), 'val', len(va))
"

# 4. train (~20 min)
.venv/bin/python -u train.py leaves/images --epochs 6 --batch-size 64 --workers 4

# 5. inspect the proof
unzip -p learnings.zip metrics.json | .venv/bin/python -m json.tool | head -20

# 6. verify archive layout
unzip -l learnings.zip | head -6

# 7. predict on every class
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
tr, va = split_dataset('leaves/images', 0.2, 42)
val = {(l, os.path.basename(p)) for p, l in va}
trn = {(l, os.path.basename(p)) for p, l in tr}
print('train/val overlap:', len(val & trn))
leak = 0
for cls in os.listdir('augmented_directory'):
    for f in os.listdir(os.path.join('augmented_directory', cls)):
        b = f
        for a in ('Flip','Rotate','Skew','Shear','Crop','Distortion'):
            i = b.find('_' + a)
            if i != -1: b = b[:i] + '.JPG'; break
        if (cls, b) in val: leak += 1
print('val images inside augmented_directory:', leak)
"
```
Both must be `0`.

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

Measured results on this dataset:

| Check | Result |
|---|---|
| Validation images | 1444 (requirement: ≥100) |
| Train/val overlap | 0 |
| Val images in `augmented_directory/` | 0 |
| Byte-identical duplicates spanning train/val | 3 pairs |
| Near-duplicates (aHash) spanning train/val | 17 val images (1.18%) |
| Accuracy, all near-duplicates removed | **100% (1427/1427)** |

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: torch` | Using system `python3` instead of `.venv/bin/python`. |
| `No matching distribution found for tensorflow` | Expected — no TF wheel for Python 3.14. This project uses PyTorch. |
| `error: no model at learnings.zip` | Run `train.py` first. |
| `warning: model was trained with preprocessing '...'` | `labels.json` disagrees with `utils/preprocess.py`. **Retrain** — do not ignore this; it means train/predict have diverged. |
| Predictions all one class | Almost always a preprocessing mismatch. Run the train/predict identity test in §4. |
| Colours look wrong (red/blue swapped) | A BGR/RGB flip outside `utils/dataset.py`. Conversion belongs there and nowhere else. |
| `RuntimeError: wrote N images but expected M` | The filename-collision guard fired in `build_augmented_dir`. |
| No window appears | Headless environment; the figure was written to `prediction.png`. Use `--save-to`. |
| Training very slow | Lower `--batch-size`, reduce `--epochs`, or lower `IMAGE_SIZE` (requires retraining). |
| First run stalls at `build_model` | Downloading ~45 MB of pretrained weights to `~/.cache/torch/`. |
