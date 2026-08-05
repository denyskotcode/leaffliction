# Leaffliction — Team Brief (read this first)

Computer-vision project: classify leaf diseases from images.
Four parts, **three people**, chained together (Part 4 eats the output of Parts 2 & 3).

**Status: all four parts delivered.** This document is the contract *as
built* — every signature below matches the code in the repo. Where the
original plan and the shipped code differ, the difference is called out
in [Deviations from the original plan](#deviations-from-the-original-plan).
For a function-by-function walkthrough with runnable tests, see
[`guide.md`](guide.md).

---

## Who owns what

| Person | Files | Scope |
|---|---|---|
| **A** | `utils/dataset.py`, `utils/naming.py`, `Distribution.py`, `Augmentation.py` | Shared foundation + dataset analysis + augmentation/balancing |
| **B** | `Transformation.py`, plus final `dataset.zip` + `signature.txt` | PlantCV-style transforms + release engineering |
| **C** | `utils/preprocess.py`, `train.py`, `predict.py` | Model training + prediction (the ≥90% accuracy gate) |

**Why this split:** A owns the loader everyone builds on, so A also takes the two parts that lean hardest on directory-walking (Distribution, Augmentation). C is pure ML because it's the largest and riskiest chunk. B is the most independent, so B also becomes release engineer at the end.

**Critical path:** A ships `utils/` first → B and C unblock immediately. The only hard serialization is at the very end: A's balancing code **and** C's model must both exist before B builds the final zip + signature.

---

## The four parts (from the subject PDF)

1. **Distribution** — `Distribution.py <dir>` walks subdirectories, counts images per class, and shows a **pie chart + bar chart** per plant type. Chart columns are named from the directory names. Works on any subtree of the dataset, down to a single class directory.
2. **Augmentation** — `Augmentation.py <image>` displays and saves **6 augmentation types** (Flip, Rotate, Skew, Shear, Crop, Distortion), each saved as `<base>_<Aug>.JPG`. `Augmentation.py -src <dir> -dst <dir>` runs the balancing pass, augmenting minority classes up to the largest class count.
3. **Transformation** — `Transformation.py`: 6 transforms (Gaussian blur, mask, ROI objects, analyze object, pseudolandmarks, plus a colour histogram). Single image → display; `-src <dir> -dst <dir>` → save all. Per-transform flags and `-h` supported.
4. **Classification** — `train.py <dir>` trains on the balanced set and writes `learnings.zip` (model + the modified images). `predict.py <image>` shows original + transformed image and prints the predicted class. **Delivered: 100% validation accuracy on 1444 held-out images** (requirement: ≥90% on ≥100).

---

## Shared contracts (as implemented)

These are the seams. Everything below is what the code actually exposes.

### Repo layout
```
leaffliction/
  Distribution.py  Augmentation.py     # Person A
  Transformation.py                    # Person B
  train.py  predict.py                 # Person C
  utils/dataset.py  utils/naming.py    # Person A
  utils/preprocess.py                  # Person C (shared by train + predict)
  requirements.txt                     # A owns, others append
  signature.txt                        # B generates
  docs/                                # this brief, the code guide, the subject
```

### In-memory image format (the #1 source of cross-module bugs)
- numpy `ndarray`, shape `(H, W, 3)`, dtype `uint8`, **RGB order**.
- OpenCV is BGR; the BGR↔RGB flip happens **once**, inside `utils/dataset.py`.

### On-disk dataset
- `root/<class_dir>/<image>.JPG`
- **label** = the leaf directory name, e.g. `Apple_healthy`.
- **plant type** = prefix before the first `_`, e.g. `Apple` (`naming.plant_type`).
- Labels live in the folder name and nowhere else.

### `utils/dataset.py`
```python
list_images(root, include_root=False) -> list[tuple[str path, str label]]
load_image(path)                      -> np.ndarray            # RGB uint8
save_image(img, path)                 -> None                  # image FIRST, path second
class_counts(root, include_root=False)-> dict[str label, int]
group_by_label(items)                 -> dict[str label, list]
split_dataset(root, val_ratio=0.2, seed=42) -> (train_list, val_list)  # stratified per class
```
`include_root=True` labels images lying directly in `root` with `root`'s
own directory name, so Parts 1 and 2 accept a single class directory.
Training never uses it.

### `utils/naming.py`
```python
AUG_NAMES = ("Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion")
stem(path)                                        -> str
augmented_filename(source, aug, round_id=0)       -> "<base>_<Aug>.JPG"
augmented_path(source, aug, dst_dir=None, round_id=0) -> str
split_augmented(filename)                         -> (stem, aug | None)
is_augmented(filename)                            -> bool
plant_type(label)                                 -> str
```
`round_id > 0` appends a number (`<base>_<Aug>1.JPG`), so a class small
enough to reuse its sources cannot overwrite its own earlier copies.

### `Augmentation.py` (Part 2, imported by `train.py`)
```python
flip / rotate / skew / shear / crop / distortion (img_rgb, rng=None) -> np.ndarray
AUGMENTATIONS                     # {title-case name: function}
augment_image(img_rgb, rng=None)  -> dict[str name, np.ndarray]   # all 6 variants
build_augmented_dir(items, out_dir, seed=42) -> list[tuple[str, str]]
SEED = 42
```

### `Transformation.py` (Part 3, imported by `predict.py`)
```python
transform_image(img_rgb) -> dict[str, np.ndarray]
#   keys: original, gaussian_blur, mask, roi_objects, analyze_object, pseudolandmarks
color_histogram(img_rgb) -> matplotlib.Figure
transformed_for_display(img_rgb) -> np.ndarray          # what predict.py shows
leaf_outline(img_rgb) -> (mask, contour)                # the segmentation both share
```

### `utils/preprocess.py` (Part 4, shared)
```python
IMAGE_SIZE = (128, 128)
PREPROCESS_NAME = "resize128_rgb_float_imagenet_norm_v1"
preprocess(img_rgb) -> np.ndarray   # float32 (3, 128, 128)
```

### Model artifact — contents of `learnings.zip`
```
model.pt              # torch: state_dict + architecture + num_classes
labels.json           # {"classes":[...index-ordered...], "input_size":[H,W], "preprocess":"..."}
metrics.json          # {"val_accuracy":1.0, "val_count":1444, "best_epoch":6, "per_class":{...}, "confusion":[[...]]}
augmented_directory/  # the modified images used for training
```
- `predict.py` reads `model.pt` + `labels.json` only.
- `train.py` and `predict.py` **preprocess identically** — both import `utils/preprocess.py`.

---

## Deviations from the original plan

| Planned | Shipped | Why |
|---|---|---|
| `model.keras`, TensorFlow | `model.pt`, PyTorch | This machine runs Python 3.14; TensorFlow publishes no wheel for it. Nothing else in the archive changed. |
| A hands C a pre-built `augmented_directory/` | `train.py` calls A's `build_augmented_dir` itself | The split must happen **before** augmentation, or augmented copies of validation images leak into training. The seam is now an imported function instead of a folder — same code, safer ordering. |
| `flake8 --max-line-length=100` | `flake8` on defaults (79) | An evaluator runs it bare, so 79 is the limit that counts. |

---

## Entry / Exit contracts

**Entry — what each person needed to start:**

| Person | Needs (entry) | Format |
|---|---|---|
| A | Raw dataset path only | Dataset on disk, `root/<class>/*.JPG` |
| B | `utils/dataset.py` from A | Python module (RGB uint8 ndarrays) |
| C | `utils/dataset.py` + `split_dataset` (A); `build_augmented_dir` (A); `transformed_for_display` (B) | Loader module + importable functions |

**Exit — what each person handed over:**

| From → To | Deliverable | Format |
|---|---|---|
| A → B, C | `utils/dataset.py`, `utils/naming.py` | Python module, signatures above; images RGB `uint8 (H,W,3)` |
| A → C | `Augmentation.build_augmented_dir` + the six augmentations | Importable functions; output `<class>/<base>_<Aug>.JPG` |
| A → B | Balanced dataset (for release zip) | `augmented_directory/<class>/*.JPG` |
| B → C | `transform_image` / `transformed_for_display` | Python fn: RGB ndarray in → RGB ndarray/dict out |
| C → B | `learnings.zip` | Zip: `model.pt` + `labels.json` + `metrics.json` + `augmented_directory/` |
| B → team | `dataset.zip` + `signature.txt` | Zip (dataset + model) + one-line `sha1sum` file at repo root |

---

## Turn-in rules (don't lose the whole grade here)

- Only **code** + `signature.txt` go in the Git repo. **The dataset must NOT be committed** — committing it means grade 0.
- `signature.txt` = the `sha1` hash of your `dataset.zip` (dataset + trained model). Generate with `sha1sum dataset.zip` (Linux) / `shasum dataset.zip` (Mac). At defense the signature must match the zip exactly.
- Python must pass `flake8` (their "norminette") on default settings. Programs must not crash on bad input.
- Keep the validation set genuinely held out — no peeking, no overfitting tricks. The leakage and duplicate checks are in `guide.md` §11 and should be reproducible on demand.

---

## Where things stand

| Part | Owner | State |
|---|---|---|
| 1 — Distribution | A | Done — charts per plant, any subtree, `--save-dir` / `--no-display` |
| 2 — Augmentation | A | Done — six augmentations, balancing with collision-safe naming |
| 3 — Transformation | B | Done — six transforms, display + batch modes, per-transform flags |
| 4 — Classification | C | Done — 100% on 1444 held-out images, `learnings.zip` written |
| Release | B | `signature.txt` present at repo root (sha1 of `dataset.zip`) |
| Lint | all | `flake8` clean on defaults across the repo |
