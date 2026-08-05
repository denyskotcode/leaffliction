# Leaffliction — Team Brief (read this first)

Computer-vision project: classify leaf diseases from images.
Four parts, **three people**, chained together (Part 4 eats the output of Parts 2 & 3).

---

## Who owns what

| Person | Files | Scope |
|---|---|---|
| **A** | `utils/dataset.py`, `utils/naming.py`, `Distribution.py`, `Augmentation.py` | Shared foundation + dataset analysis + augmentation/balancing |
| **B** | `Transformation.py`, plus final `dataset.zip` + `signature.txt` | PlantCV transforms + release engineering |
| **C** | `train.py`, `predict.py` | Model training + prediction (the ≥90% accuracy gate) |

**Why this split:** A owns the loader everyone builds on, so A also takes the two parts that lean hardest on directory-walking (Distribution, Augmentation). C is pure ML because it's the largest and riskiest chunk. B is the most independent, so B also becomes release engineer at the end.

**Critical path:** A ships `utils/` on Day 1–2 → B and C unblock immediately. The only hard serialization is at the very end: A's balanced dataset **and** C's model must both exist before B builds the final zip + signature.

---

## The four parts (from the subject PDF)

1. **Distribution** — `Distribution.py <dir>` walks subdirectories, counts images per class, and shows a **pie chart + bar chart** per plant type. Chart columns are named from the directory names. Must work on any subtree of the dataset.
2. **Augmentation** — `Augmentation.py <image>` displays and saves **6 augmentation types** (Flip, Rotate, Skew, Shear, Crop, Distortion), each saved as `<base>_<Aug>.JPG`. A balancing pass augments minority classes up to the largest class count, written into `augmented_directory/`.
3. **Transformation** — `Transformation.py` (PlantCV recommended): ≥6 transforms (Gaussian blur, Mask, ROI objects, Analyze object, Pseudolandmarks, plus a color histogram). Single image → display; `-src <dir> -dst <dir>` → save all. Supports per-transform flags and `-h`.
4. **Classification** — `train.py <dir>` trains on the balanced set and saves a `.zip` (model + the modified images). `predict.py <image>` shows original + transformed image and prints the predicted class. **Validation accuracy ≥ 90%**, provable on **≥ 100 held-out images**.

---

## Day-1 shared contracts (lock these before anyone codes)

These are the seams. Agree on them and everyone can work in parallel against stubs.

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
```

### In-memory image format (the #1 source of cross-module bugs)
- numpy `ndarray`, shape `(H, W, 3)`, dtype `uint8`, **RGB order**.
- If you use OpenCV, convert BGR→RGB at load and RGB→BGR at save. Do it **once**, inside `utils/dataset.py`.

### On-disk dataset
- `root/<class_dir>/<image>.JPG`
- **label** = the leaf directory name, e.g. `apple_healthy`.
- **category** = prefix before the first `_`, e.g. `Apple`.
- Labels live in the folder name and nowhere else.

### `utils/dataset.py` API (Person A ships first)
```python
list_images(root)     -> list[tuple[str path, str label]]
load_image(path)      -> np.ndarray            # RGB uint8
save_image(img, path) -> None                  # expects RGB uint8
class_counts(root)    -> dict[str label, int]
split_dataset(root, val_ratio=0.2, seed=42) -> (train_list, val_list)  # stratified per class
```

### `utils/naming.py`
- Augmented file = `<basename>_<Aug>.JPG`, `Aug ∈ {Flip, Rotate, Skew, Shear, Crop, Distortion}` (title-case).

### Augmentation API
```python
augment_image(img_rgb) -> dict[str name, np.ndarray]   # all 6 variants
```

### Transformation API (Person B, imported by Person C)
```python
transform_image(img_rgb) -> dict[str, np.ndarray]
#   keys: original, gaussian_blur, mask, roi_objects, analyze_object, pseudolandmarks
color_histogram(img_rgb) -> matplotlib.Figure
transformed_for_display(img_rgb) -> np.ndarray          # what predict.py shows
```

### Model artifact — contents of `learnings.zip` (Person C)
```
model.keras           # trained model
labels.json           # {"classes":[...index-ordered...], "input_size":[H,W], "preprocess":"..."}
metrics.json          # {"val_accuracy":0.9x, "val_count":N, "confusion":[[...]]}
augmented_directory/  # the modified images used for training
```
- `predict.py` reads `model.keras` + `labels.json` only.
- `train.py` and `predict.py` **must preprocess identically** — both import `utils/preprocess.py`.

---

## Entry / Exit contracts

**Entry — what each person needs to start:**

| Person | Needs (entry) | Format |
|---|---|---|
| A | Raw dataset path only | Dataset on disk, `root/<class>/*.JPG` |
| B | `utils/dataset.py` from A | Python module (RGB uint8 ndarrays) |
| C | `utils/dataset.py` + `split_dataset` (A); `augmented_directory/` (A); `transformed_for_display` (B) | Loader module + folder of `.JPG` + importable fn |

**Exit — what each person hands over:**

| From → To | Deliverable | Format |
|---|---|---|
| A → B, C | `utils/dataset.py`, `utils/naming.py` | Python module, signatures above; images RGB `uint8 (H,W,3)` |
| A → C | `augmented_directory/` (balanced) | On-disk `<class>/*.JPG`, names `<base>_<Aug>.JPG` |
| A → B | Balanced dataset (for release zip) | Same folder as above |
| B → C | `transform_image` / `transformed_for_display` | Python fn: RGB ndarray in → RGB ndarray/dict out |
| C → B | `learnings.zip` | Zip: `model.keras` + `labels.json` + `metrics.json` + `augmented_directory/` |
| B → team | `dataset.zip` + `signature.txt` | Zip (dataset + model) + one-line `sha1sum` file at repo root |

---

## Turn-in rules (don't lose the whole grade here)

- Only **code** + `signature.txt` go in the Git repo. **The dataset must NOT be committed** — committing it means grade 0.
- `signature.txt` = the `sha1` hash of your `dataset.zip` (dataset + trained model). Generate with `sha1sum dataset.zip` (Linux) / `shasum dataset.zip` (Mac). At defense the signature must match the zip exactly.
- Python must pass `flake8` (their "norminette"). Programs must not crash on bad input.
- Keep the validation set genuinely held out — no peeking, no overfitting tricks.

---

## Suggested timeline

- **Day 1–2:** All three lock contracts. A ships `utils/`. B and C start against stubs (C trains on raw/mock data first).
- **Mid:** A finishes Distribution + Augmentation + balancing. B finishes Transformation. C iterates the model to ≥90%.
- **End:** A hands balanced `augmented_directory/` → C retrains/finalizes → C hands `learnings.zip` → B builds `dataset.zip` + `signature.txt` and audits the repo.
