# Leaffliction — Leaf Disease Recognition

Computer vision project: image classification for plant leaf disease
recognition. The pipeline covers data set analysis, data augmentation,
image transformation, and a CNN-based classifier with prediction.

## Authors

| Login | Owns |
|---|---|
| `dkot` | Part 1 (`Distribution.py`), Part 2 (`Augmentation.py`), `utils/dataset.py`, `utils/naming.py` |
| `anmakaro` | Part 3 (`Transformation.py`), the release archive and `signature.txt` |
| `msylaiev` | Part 4 (`train.py`, `predict.py`), `utils/preprocess.py` |

## Project structure

Only the code and `signature.txt` are versioned. The data set, the
archives and the trained model are deliberately absent — see
[Release](#release-datasetzip--signaturetxt).

```
leaffliction/
├── Distribution.py       # Part 1 — data set analysis (pie/bar charts)
├── Augmentation.py       # Part 2 — data augmentation (balancing)
├── Transformation.py     # Part 3 — image transformations
├── train.py              # Part 4 — model training
├── predict.py            # Part 4 — prediction / evaluation
├── utils/
│   ├── dataset.py        # load_image / save_image, split, path helpers
│   ├── naming.py         # file naming conventions
│   └── preprocess.py     # shared preprocessing for train + predict
├── requirements.txt
├── signature.txt         # sha1 of dataset.zip (generated at release time)
└── .gitignore
```

The data set is expected at `leaves/images/`, one directory per class,
the directory name *being* the label — no class name is hardcoded
anywhere:

```
leaves/images/
├── Apple_Black_rot/   ├── Grape_Black_rot/
├── Apple_healthy/     ├── Grape_Esca/
├── Apple_rust/        ├── Grape_healthy/
└── Apple_scab/        └── Grape_spot/
```

## Setup

Use a project-local virtual environment so the system Python stays
untouched and everyone on the team gets the same versions.

```bash
python3 -m venv .venv

# PyTorch first, from the CPU index (avoids ~2 GB of unused CUDA wheels)
.venv/bin/python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu torch torchvision

.venv/bin/python -m pip install -r requirements.txt

# sanity check
.venv/bin/python -c "import cv2, numpy, matplotlib, torch; print('OK')"
```

Then run every script through that interpreter, e.g.
`.venv/bin/python train.py leaves/images`.

`requirements.txt` includes `opencv-python-headless`, `numpy`,
`matplotlib`, `scikit-learn`, `flake8`, and `torch` / `torchvision`
(Part 4). See the framework note under Part 4 for why it is PyTorch and
not TensorFlow.

---

## Part 1 — Distribution.py

Analyze a data set directory and display a pie chart and a bar chart per
plant type, labeled from the subdirectory names.

```bash
./Distribution.py ./leaves/images

# also write the figures as PNGs; skip the window entirely (ssh, CI)
./Distribution.py ./leaves/images --save-dir charts --no-display

# any subtree works, down to a single class directory
./Distribution.py ./leaves/images/Apple_rust
```

It prints the counts as well as charting them, so the analysis survives
with no display attached:

```
Apple: 3164 images across 4 classes
  Apple_Black_rot             620  ( 19.6%)
  Apple_healthy              1640  ( 51.8%)
  Apple_rust                  275  (  8.7%)
  Apple_scab                  629  ( 19.9%)
Grape: 4057 images across 4 classes
  ...
```

```bash
# actual directory layout:
tree --filelimit=13 leaves/images
# leaves/images/
# ├── Apple_Black_rot
# ├── Apple_healthy
# ├── Apple_rust
# ├── Apple_scab
# ├── Grape_Black_rot
# ├── Grape_Esca
# ├── Grape_healthy
# └── Grape_spot
```

---

## Part 2 — Augmentation.py

Balance the data set by generating 6 augmented versions of a given image
(Flip, Rotate, Skew, Shear, Crop, Distortion), saved next to the original.

```bash
./Augmentation.py "leaves/images/Apple_healthy/image (1).JPG"
```

```bash
# expected output files:
ls
# image (1)_Flip.JPG
# image (1)_Rotate.JPG
# image (1)_Skew.JPG
# image (1)_Shear.JPG
# image (1)_Crop.JPG
# image (1)_Distortion.JPG
```

Run it over a whole data set to balance it: every image is copied across,
then the smaller classes are augmented until each one matches the
largest.

```bash
./Augmentation.py -src leaves/images -dst augmented_directory
```

```
balancing 8 classes up to 1640 images each
  Apple_Black_rot: 620 -> 1640
  Apple_healthy: 1640 -> 1640
  ...
```

Flags: `-src`, `-dst` (default `augmented_directory`), `--seed`
(default 42), `--no-display`. `train.py` imports these same functions, so
the images the model learns from are produced by exactly this code.

---

## Part 3 — Transformation.py

Apply leaf-image transformations: Gaussian blur, mask, ROI objects,
analyze object, pseudolandmarks, plus a color histogram over the nine
channels the subject's Figure IV.7 names (blue, blue-yellow, green,
green-magenta, hue, lightness, red, saturation, value — drawn from RGB,
HSV and LAB).

### Help

```bash
./Transformation.py -h
python3 ./Transformation.py -h
```

### Single image — display mode

Shows all transformations (+ histogram) in a matplotlib window.

```bash
./Transformation.py "leaves/images/Apple_healthy/image (1).JPG"
python3 ./Transformation.py "leaves/images/Apple_healthy/image (1).JPG"
```

### Directory — batch save mode

Saves every requested transformation for every image found (recursively)
under `-src` into `-dst`, named `<original>_<Transformation><ext>`.

```bash
# save everything (all 5 transforms + histogram)
./Transformation.py -src leaves/images/Apple_healthy -dst dst_directory

# save only the mask, per subject's example command
./Transformation.py -src leaves/images/Apple_healthy -dst dst_directory -mask
```

### Individual transformation flags

Combine any of these; if none are given, all are produced.

```bash
./Transformation.py -src leaves/images/Apple_healthy -dst out -blur
./Transformation.py -src leaves/images/Apple_healthy -dst out -mask
./Transformation.py -src leaves/images/Apple_healthy -dst out -roi
./Transformation.py -src leaves/images/Apple_healthy -dst out -object
./Transformation.py -src leaves/images/Apple_healthy -dst out -landmarks
./Transformation.py -src leaves/images/Apple_healthy -dst out -histogram

# combine several
./Transformation.py -src leaves/images/Apple_healthy -dst out -mask -histogram
```

### Using it as a library

`predict.py` imports these to render the transformed panel beside the
original:

```python
from Transformation import transform_image, color_histogram, transformed_for_display
from utils.dataset import load_image

img_rgb = load_image("leaves/images/Apple_healthy/image (1).JPG")

transforms = transform_image(img_rgb)
# transforms.keys() -> original, gaussian_blur, mask, roi_objects,
#                      analyze_object, pseudolandmarks

fig = color_histogram(img_rgb)     # matplotlib Figure
display_img = transformed_for_display(img_rgb)  # np.ndarray, RGB uint8
```

---

## Part 4 — train.py / predict.py

> **Framework note.** This project runs on Python 3.14, for which
> TensorFlow publishes no wheel. Part 4 uses **PyTorch** instead, so the
> artifact inside `learnings.zip` is `model.pt` rather than the
> `model.keras` named in the team brief. Everything else in the archive
> is unchanged.

### Training

```bash
./train.py leaves/images
```

What it does, in order:

1. **Splits first** — `split_dataset(root, val_ratio=0.2, seed=42)`,
   stratified per class. The validation set is held out *before* any
   augmentation, so no augmented copy of a validation image can ever
   leak into training.
2. **Balances the training split only** — minority classes are augmented
   (Flip, Rotate, Skew, Shear, Crop, Distortion) up to the largest class
   and written to `augmented_directory/<class>/<base>_<Aug>.JPG`.
3. **Fine-tunes an ImageNet-pretrained ResNet-18** on the balanced set.
4. **Evaluates on the untouched validation split** and keeps the weights
   from the best epoch.
5. **Writes `learnings.zip`**:

```
model.pt              # weights + architecture name
labels.json           # {"classes":[...], "input_size":[H,W], "preprocess":"..."}
metrics.json          # {"val_accuracy":..., "val_count":N, "confusion":[[...]], ...}
augmented_directory/  # the modified images used for training
```

Useful flags:

```bash
./train.py leaves/images --epochs 6 --batch-size 64 --workers 4
./train.py leaves/images --augmented-dir augmented_directory --out learnings.zip
```

### Prediction

```bash
./predict.py "leaves/images/Apple_healthy/image (1).JPG"
```

Reads **only** `model.pt` + `labels.json` from `learnings.zip`,
preprocesses with the same `utils/preprocess.py` used at training time,
displays the original beside Part 3's `analyze_object` rendering, and
prints:

```
===          DL classification          ===

Class predicted : Apple_healthy
Confidence      : 99.87%
```

```bash
# alternative model location, headless use
./predict.py <image> --model learnings.zip
./predict.py <image> --save-to prediction.png   # write figure to a file
./predict.py <image> --no-display               # print the class only
```

Validation accuracy must be **> 90%** on a validation set of **at least
100 images**, provable during defense. The proof is stored in
`metrics.json` inside `learnings.zip`.

### Defending the result

Fine-tuning a pretrained ResNet-18 on this dataset reaches **100%
(1444/1444)** on the held-out split. That is a strong claim, so here is
the evidence that it is honest rather than leaked. All of it is
reproducible with `seed=42`.

| Check | Result |
|---|---|
| Validation images | 1444 (requirement: >= 100) |
| Train/validation overlap | 0 images |
| Validation images inside `augmented_directory/` | 0 |
| Byte-identical duplicates spanning train and val | 3 pairs |
| Near-duplicates (aHash) spanning train and val | 17 val images (1.18%) |
| Accuracy with all 17 near-duplicates removed | **100% (1427/1427)** |

Why it holds up:

- **The split happens before augmentation.** No augmented copy of a
  validation image can reach the training set, verified above.
- **The duplicates are in the raw dataset as delivered**, not introduced
  here. Removing every one of them does not move the accuracy, so the
  score is not resting on them.
- **The dataset is genuinely easy.** These are lab-condition PlantVillage
  images: one centred leaf, uniform grey background, consistent lighting.
  Near-ceiling accuracy from an ImageNet backbone is the expected result,
  not an anomaly.
- **The classes it could plausibly confuse, it does.** In an earlier run
  the only 2 errors were `Grape_Esca` -> `Grape_Black_rot`, which is a
  botanically sensible confusion rather than random noise.

Reproduce the leakage and duplicate checks at defense time with the
`split_dataset(..., seed=42)` call and a hash of each file; the split is
deterministic, so the same images land on the same side every run.

### Why preprocessing lives in one file

`utils/preprocess.py` exposes a single `preprocess()` that both
`train.py` and `predict.py` import. A train/predict preprocessing
mismatch is the classic silent accuracy killer — it cannot happen here,
because there is only one code path. `labels.json` also records the
preprocessing name, and `predict.py` warns if a model was trained with a
different one.

---

## Release (dataset.zip + signature.txt)

Built once the augmented data set (Part 2) and the trained model /
`learnings.zip` (Part 4) are ready.

```bash
# 1. assemble the release archive
zip -r dataset.zip leaves/images learnings.zip

# 2. generate the signature (choose the command matching your OS)
sha1sum dataset.zip            # Linux
shasum dataset.zip             # macOS
certUtil -hashfile dataset.zip sha1   # Windows

# 3. write the hash into signature.txt at the repo root
sha1sum dataset.zip | awk '{print $1}' > signature.txt
cat signature.txt
```

**Important:** `dataset.zip` itself must **never** be committed to git —
only the code and `signature.txt` belong in the repository. At defense
time, the signature of `signature.txt` is compared against a fresh hash
of your data set; a mismatch results in a grade of 0.

```bash
# verify before defense that everything still matches
sha1sum dataset.zip
cat signature.txt
```

---

## Linting

All Python files must pass `flake8` on its **default settings** — the
subject calls it the norm (`alias norminette_python=flake8`), and an
evaluator runs it bare, so the 79-column default is the limit that
counts. Do not relax it with `--max-line-length`.

```bash
.venv/bin/python -m flake8 *.py utils/*.py
```

---

## Quick reference — every command in one place

```bash
# setup
python3 -m venv .venv
.venv/bin/python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/python -m pip install -r requirements.txt

# part 1
./Distribution.py ./leaves/images

# part 2
./Augmentation.py "leaves/images/Apple_healthy/image (1).JPG"
./Augmentation.py -src leaves/images -dst augmented_directory

# part 3
./Transformation.py -h
./Transformation.py "leaves/images/Apple_healthy/image (1).JPG"
./Transformation.py -src leaves/images/Apple_healthy -dst dst_directory -mask

# part 4
./train.py leaves/images                      # -> learnings.zip + augmented_directory/
./predict.py "leaves/images/Apple_healthy/image (1).JPG"
unzip -p learnings.zip metrics.json           # proof of >=90% on the held-out set

# release
zip -r dataset.zip leaves/images learnings.zip
sha1sum dataset.zip | awk '{print $1}' > signature.txt

# lint (default settings: 79 columns, as the subject requires)
.venv/bin/python -m flake8 *.py utils/*.py
```

---

## Data set integrity

The four checks that matter, runnable at any time:

```bash
# 1. nothing forbidden is versioned  -> 0
git ls-files | grep -icE '\.(jpg|jpeg|png|zip)$'

# 2. the signature still matches the data set  -> two identical hashes
sha1sum dataset.zip; cat signature.txt

# 3. the split is deterministic, stratified and leak-free
.venv/bin/python -c "
from utils.dataset import split_dataset
tr, va = split_dataset('leaves/images', 0.2, 42)
print('train', len(tr), 'val', len(va))
print('overlap:', len({p for p, _ in tr} & {p for p, _ in va}))"

# 4. no validation image reached the augmented training set  -> 0
.venv/bin/python -c "
import os
from utils.dataset import split_dataset
from utils.naming import split_augmented
val = {(l, os.path.basename(p))
       for p, l in split_dataset('leaves/images', 0.2, 42)[1]}
print(sum((c, split_augmented(f)[0] + '.JPG') in val
          for c in os.listdir('augmented_directory')
          for f in os.listdir(os.path.join('augmented_directory', c))))"
```

Expected: `0`, matching hashes, `train 5777 val 1444` / `overlap: 0`,
and `0`.