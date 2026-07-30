# Leaffliction — Leaf Disease Recognition

Computer vision project: image classification for plant leaf disease
recognition. The pipeline covers data set analysis, data augmentation,
image transformation, and a CNN-based classifier with prediction.

## Project structure

```
leaffliction/
├── Distribution.py        # Part 1 — data set analysis (pie/bar charts)
├── Augmentation.py        # Part 2 — data augmentation (balancing)
├── transformation.py      # Part 3 — image transformations
├── train.py                # Part 4 — model training
├── predict.py               # Part 4 — prediction / evaluation
├── utils/
│   ├── dataset.py          # load_image / save_image, path helpers
│   ├── naming.py            # file naming conventions
│   └── preprocess.py        # shared preprocessing for train + predict
├── requirements.txt
├── signature.txt            # sha1 of dataset.zip (generated at release time)
└── .gitignore
```

## Setup

```bash
# make sure pip installs into the SAME interpreter you run scripts with
python3 -m pip install --user --break-system-packages -r requirements.txt

# sanity check
python3 -c "import cv2, numpy, matplotlib; print('OK')"
```

`requirements.txt` typically includes: `opencv-python-headless`, `numpy`,
`matplotlib`, `scikit-learn` / `tensorflow` or `torch` (whichever Person C
uses for training).

---

## Part 1 — Distribution.py

Analyze a data set directory and display a pie chart and a bar chart per
plant type, labeled from the subdirectory names.

```bash
./Distribution.py ./Apple
./Distribution.py ./Grape
```

```bash
# example directory layout expected:
find . -maxdepth 2
# ./Apple
# ./Apple/apple_healthy
# ./Apple/apple_apple_scab
# ./Apple/apple_black_rot
# ./Apple/apple_cedar_apple_rust
```

---

## Part 2 — Augmentation.py

Balance the data set by generating 6 augmented versions of a given image
(Flip, Rotate, Skew, Shear, Crop, Distortion), saved next to the original.

```bash
./Augmentation.py "./Apple/apple_healthy/image (1).JPG"
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

Run in bulk over a whole directory to balance every class (exact flag
names depend on Person A's implementation):

```bash
./Augmentation.py -src ./Apple/apple_black_rot -dst ./Apple/apple_black_rot
```

---

## Part 3 — transformation.py

Apply leaf-image transformations: Gaussian blur, mask, ROI objects,
analyze object, pseudolandmarks, plus a color histogram.

### Help

```bash
./transformation.py -h
python3 ./transformation.py -h
```

### Single image — display mode

Shows all transformations (+ histogram) in a matplotlib window.

```bash
./transformation.py ./Apple/apple_healthy/image\ \(1\).JPG
python3 ./transformation.py "./leaves/images/Apple_healthy/image (1).JPG"
```

### Directory — batch save mode

Saves every requested transformation for every image found (recursively)
under `-src` into `-dst`, named `<original>_<transformation><ext>`.

```bash
# save everything (all 5 transforms + histogram)
./transformation.py -src leaves/images/Apple_healthy/ -dst dst_directory

# save only the mask, per subject's example command
./transformation.py -src leaves/images/Apple_healthy/ -dst dst_directory -mask
```

### Individual transformation flags

Combine any of these; if none are given, all are produced.

```bash
./transformation.py -src leaves/images/Apple_healthy/ -dst out -blur
./transformation.py -src leaves/images/Apple_healthy/ -dst out -mask
./transformation.py -src leaves/images/Apple_healthy/ -dst out -roi
./transformation.py -src leaves/images/Apple_healthy/ -dst out -object
./transformation.py -src leaves/images/Apple_healthy/ -dst out -landmarks
./transformation.py -src leaves/images/Apple_healthy/ -dst out -histogram

# combine several
./transformation.py -src leaves/images/Apple_healthy/ -dst out -mask -histogram
```

### Using it as a library (for Person C)

```python
from transformation import transform_image, color_histogram, transformed_for_display
from utils.dataset import load_image

img_rgb = load_image("./leaves/images/Apple_healthy/image (1).JPG")

transforms = transform_image(img_rgb)
# transforms.keys() -> original, gaussian_blur, mask, roi_objects,
#                      analyze_object, pseudolandmarks

fig = color_histogram(img_rgb)     # matplotlib Figure
display_img = transformed_for_display(img_rgb)  # np.ndarray, RGB uint8
```

---

## Part 4 — train.py / predict.py

### Training

Trains on a directory of subdirectories (one per class), augmenting /
preprocessing images as needed. Saves the learned model + augmented
images into a `.zip`.

```bash
./train.py ./Apple/
```

### Prediction

Loads a saved model, runs on a single image, displays original +
transformed image, and prints the predicted disease class.

```bash
./predict.py "./Apple/apple_healthy/image (1).JPG"
```

Validation accuracy must be **> 90%** on a validation set of **at least
100 images**, provable during defense.

---

## Release (dataset.zip + signature.txt)

Built once the augmented data set (Person A) and trained model /
`learnings.zip` (Person C) are ready.

```bash
# 1. assemble the release archive
zip -r dataset.zip augmented_directory/ learnings.zip

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

All Python files must pass `flake8`:

```bash
python3 -m pip install --user --break-system-packages flake8
flake8 *.py utils/*.py --max-line-length=100
```

---

## Quick reference — every command in one place

```bash
# setup
python3 -m pip install --user --break-system-packages -r requirements.txt

# part 1
./Distribution.py ./Apple

# part 2
./Augmentation.py "./Apple/apple_healthy/image (1).JPG"

# part 3
./transformation.py -h
./transformation.py "./Apple/apple_healthy/image (1).JPG"
./transformation.py -src Apple/apple_healthy/ -dst dst_directory -mask

# part 4
./train.py ./Apple/
./predict.py "./Apple/apple_healthy/image (1).JPG"

# release
zip -r dataset.zip augmented_directory/ learnings.zip
sha1sum dataset.zip | awk '{print $1}' > signature.txt

# lint
flake8 *.py utils/*.py --max-line-length=100
```