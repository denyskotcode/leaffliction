# EVALUATION.md — Defense Walkthrough

A step-by-step script for the peer evaluation: what to run, in what
order, what the evaluator should see, and what to say. Every command
below has been executed against the real dataset; the expected outputs
are real outputs, not sketches.

**Total demo time: ~25 minutes** without a retrain, ~45 with one.

---

## Contents

- [0. Before the evaluation starts](#0-before-the-evaluation-starts)
- [1. Repo audit — the instant-zero checks](#1-repo-audit--the-instant-zero-checks)
- [2. Lint](#2-lint)
- [3. Part 1 — Distribution](#3-part-1--distribution)
- [4. Part 2 — Augmentation](#4-part-2--augmentation)
- [5. Part 3 — Transformation](#5-part-3--transformation)
- [6. Part 4 — Prediction](#6-part-4--prediction)
- [7. The ≥90% proof](#7-the-90-proof)
- [8. Proving the score is honest](#8-proving-the-score-is-honest)
- [9. Error handling sweep](#9-error-handling-sweep)
- [10. Optional — retrain from scratch](#10-optional--retrain-from-scratch)
- [11. Who answers what](#11-who-answers-what)
- [12. Failure playbook](#12-failure-playbook)

---

## 0. Before the evaluation starts

Do this **the night before**, not in the room.

```bash
cd /path/to/leaffliction

# the dataset is present and unpacked
ls leaves/images | wc -l                 # 8

# the environment works
.venv/bin/python -c "import cv2, numpy, matplotlib, torch, torchvision; print('OK')"

# the artifacts exist
ls -lh learnings.zip dataset.zip signature.txt
ls learnings/                            # model.pt labels.json metrics.json

# the signature still matches the zip
sha1sum dataset.zip; cat signature.txt   # the two hashes must be identical

# the working tree is clean and nothing forbidden is tracked
git status --short                       # empty
git ls-files | wc -l                     # 23
```

**Rules of the room**

- Never demo augmentation on a file inside `leaves/images/` — it writes
  six new files next to it and changes every count you just showed. Copy
  the image to `/tmp` first. Every command below already does.
- Never rebuild `dataset.zip` during the defense. Zip archives embed
  timestamps, so a rebuild changes the hash and breaks `signature.txt`.
- Have this file open in a second terminal.

---

## 1. Repo audit — the instant-zero checks

Two rules can zero the whole project regardless of the code. Get them out
of the way first; it sets a good tone.

```bash
git ls-files
```

Expected — **23 files, no images, no archives:**
```
.gitignore  README.md  requirements.txt  signature.txt
Distribution.py  Augmentation.py  Transformation.py  train.py  predict.py
utils/__init__.py  utils/dataset.py  utils/naming.py  utils/preprocess.py
docs/00_TEAM_BRIEF.md  docs/guide.md  docs/EVALUATION.md  docs/Leaffliction.md
docs/PERSON_A.md  docs/PERSON_B.md  docs/PERSON_C.md
docs/PERSON_A_GUIDE.md  docs/PERSON_B_GUIDE.md  docs/PERSON_C_GUIDE.md
```

```bash
git ls-files | grep -icE '\.(jpg|jpeg|png|zip)$'    # 0
```

> **Say:** "Only code and `signature.txt` are committed. `.gitignore`
> blocks `*.JPG`, `*.zip`, `leaves/` and `augmented_directory/`."

Then the signature:

```bash
sha1sum dataset.zip
cat signature.txt
```
Both must print `ee2c0426cb2f14ebc001554cb593813efe191e3c`.

> **Say:** "`dataset.zip` is the images plus `learnings.zip` — 393 MB,
> 7231 entries. `signature.txt` is its sha1, generated with `sha1sum`,
> never typed by hand."

*(Owner: **B**.)*

---

## 2. Lint

The subject calls `flake8` the Python norminette. An evaluator runs it
bare, so run it bare — **no `--max-line-length`**.

```bash
.venv/bin/python -m flake8 . --exclude=.venv,augmented_directory && echo CLEAN
```
Expected: `CLEAN`, no output above it.

> **Say:** "Default settings, 79 columns, whole repo."

---

## 3. Part 1 — Distribution

*(Owner: **A**. Background: [`PERSON_A_GUIDE.md`](PERSON_A_GUIDE.md))*

### 3.1 The whole dataset

```bash
.venv/bin/python Distribution.py leaves/images
```

Expected text output (two figures also open):
```
Apple: 3164 images across 4 classes
  Apple_Black_rot             620  ( 19.6%)
  Apple_healthy              1640  ( 51.8%)
  Apple_rust                  275  (  8.7%)
  Apple_scab                  629  ( 19.9%)
Grape: 4057 images across 4 classes
  Grape_Black_rot            1178  ( 29.0%)
  Grape_Esca                 1382  ( 34.1%)
  Grape_healthy               422  ( 10.4%)
  Grape_spot                 1075  ( 26.5%)
```

Each plant gets **one figure with a pie chart and a bar chart**, labelled
from the directory names, with a shared legend.

> **Say:** "7221 images, 8 classes. The largest class has 6× the images
> of the smallest — that imbalance is exactly what Part 2 fixes."

### 3.2 It works on any subtree

```bash
.venv/bin/python Distribution.py leaves/images/Apple_rust
```
Expected: `Apple: 275 images across 1 classes`.

> **Say:** "No class name is hardcoded anywhere. The labels are directory
> names; point it at any folder and it describes that folder."

### 3.3 Headless / saving figures

```bash
.venv/bin/python Distribution.py leaves/images --save-dir /tmp/charts --no-display
ls /tmp/charts
```
Expected: `Apple_distribution.png  Grape_distribution.png`.

---

## 4. Part 2 — Augmentation

*(Owner: **A**.)*

### 4.1 Six augmentations of one image

Copy the image out of the dataset first — this command writes files next
to it.

```bash
mkdir -p /tmp/demo && cp "leaves/images/Apple_healthy/image (1).JPG" /tmp/demo/
.venv/bin/python Augmentation.py "/tmp/demo/image (1).JPG"
ls /tmp/demo
```

Expected — the six names the subject's `ls` example shows, plus a window
with all seven panels:
```
image (1).JPG
image (1)_Crop.JPG        image (1)_Distortion.JPG
image (1)_Flip.JPG        image (1)_Rotate.JPG
image (1)_Shear.JPG       image (1)_Skew.JPG
```

> **Say:** "Flip, Rotate, Skew, Shear, Crop, Distortion. All six reflect
> at the border rather than padding with black — a black wedge is a
> feature no real photo has, and the network would learn it as a shortcut
> to whichever classes were augmented most."

### 4.2 Balancing a whole dataset

```bash
.venv/bin/python Augmentation.py -src leaves/images -dst /tmp/balanced --no-display
```

Expected (~15 seconds, 13120 images written):
```
balancing 8 classes up to 1640 images each
  Apple_Black_rot: 620 -> 1640
  Apple_healthy: 1640 -> 1640
  Apple_rust: 275 -> 1640
  ...
13120 images written to /tmp/balanced
```

```bash
for d in /tmp/balanced/*/; do echo -n "$(basename $d): "; ls "$d" | wc -l; done
```
Every class must print **1640**.

> **Say:** "Every class now matches the largest. The code counts the files
> actually on disk afterwards and raises if the number is wrong — that
> guard exists because filename collisions would silently under-fill
> precisely the rarest classes."

If asked how collisions are avoided: `math.lcm(n_sources, 6)`. A
`(source, augmentation)` pair repeats every `lcm(n, 6)` steps, not
`n × 6`; later rounds get a numeric suffix (`_Flip1.JPG`).

### 4.3 Reproducibility (if asked)

```bash
.venv/bin/python Augmentation.py -src /tmp/mini -dst /tmp/b1 --no-display >/dev/null
.venv/bin/python Augmentation.py -src /tmp/mini -dst /tmp/b2 --no-display >/dev/null
diff -r /tmp/b1 /tmp/b2 && echo "byte-identical"
```

---

## 5. Part 3 — Transformation

*(Owner: **B**. Background: [`PERSON_B_GUIDE.md`](PERSON_B_GUIDE.md))*

### 5.1 Help

```bash
.venv/bin/python Transformation.py -h
```
Shows both modes and all six per-transformation flags.

### 5.2 Display mode

```bash
.venv/bin/python Transformation.py "leaves/images/Apple_healthy/image (1).JPG"
```
A grid with **Original, Gaussian Blur, Mask, Roi Objects, Analyze Object,
Pseudolandmarks**, plus a second figure with the colour histogram.

> **Say:** "Everything starts from one segmentation: an Otsu threshold on
> the HSV **saturation** channel. The leaf is coloured, the background is
> flat grey — grey has near-zero saturation whatever its brightness. A
> green-channel threshold would fail on exactly the yellowed and
> brown-spotted leaves we have to classify."

### 5.3 Batch mode — the subject's example command

```bash
.venv/bin/python Transformation.py -src leaves/images/Apple_rust -dst /tmp/tr_mask -mask
ls /tmp/tr_mask | head -3
```
Expected: `image (1)_Mask.JPG`, `image (10)_Mask.JPG`, … (275 files).

### 5.4 Batch mode — everything

```bash
mkdir -p /tmp/tr_src && cp "leaves/images/Apple_scab/image (1).JPG" /tmp/tr_src/
.venv/bin/python Transformation.py -src /tmp/tr_src -dst /tmp/tr_all
ls /tmp/tr_all
```
Expected for one source image — five renderings plus the histogram:
```
image (1)_AnalyzeObject.JPG   image (1)_GaussianBlur.JPG
image (1)_Histogram.png       image (1)_Mask.JPG
image (1)_Pseudolandmarks.JPG image (1)_RoiObjects.JPG
```

> **Say:** "No flag means all of them. `original` is displayed but never
> saved — writing a copy of the input under a new name would be
> pointless."

---

## 6. Part 4 — Prediction

*(Owner: **C**. Background: [`PERSON_C_GUIDE.md`](PERSON_C_GUIDE.md))*

### 6.1 One image

```bash
.venv/bin/python predict.py "leaves/images/Apple_healthy/image (1).JPG"
```
Expected — a two-panel figure (original | transformed) titled with the
class, and on stdout the subject's banner and verdict:
```
===          DL classification          ===

Class predicted : Apple_healthy
Confidence      : 99.9x%
```

> **Say:** "It reads only `model.pt` and `labels.json` out of
> `learnings.zip`, and preprocesses with the same
> `utils/preprocess.py` that training used — the same function object,
> not a copy."

### 6.2 One image per class

```bash
for c in $(ls leaves/images); do
  f=$(ls "leaves/images/$c" | head -1)
  echo -n "$c -> "
  .venv/bin/python predict.py "leaves/images/$c/$f" --no-display \
    | grep "Class predicted"
done
```
All eight must print `Class predicted : <the same class>`.

### 6.3 The transformed panel

The right-hand panel is Person B's `analyze_object` rendering — contour,
centroid, area/perimeter — imported from `Transformation.py`. It is
wrapped in `try/except`: if that module failed, the prediction would
still print and the original would be shown instead. Classification never
depends on presentation.

---

## 7. The ≥90% proof

*(Owner: **C**.)*

```bash
unzip -p learnings.zip metrics.json | .venv/bin/python -m json.tool | head -12
```
Expected:
```json
{
    "val_accuracy": 1.0,
    "val_count": 1444,
    "best_epoch": 6,
    ...
}
```

Two requirements, both visible in that one file:

| Requirement | Value |
|---|---|
| Validation accuracy ≥ 90% | **100%** |
| Held-out images ≥ 100 | **1444** |

```bash
unzip -l learnings.zip | head -6
```
Shows `model.pt`, `labels.json`, `metrics.json`, then
`augmented_directory/…` — the modified images used for training, as the
subject requires.

`metrics.json` also carries per-class recall and the full confusion
matrix (`confusion[true][pred]` — rows are truth), so a sceptical
evaluator can check any single class.

---

## 8. Proving the score is honest

**Expect to be challenged on 100%.** Do not get defensive — run these.
*(Owner: **C**, with **A** on the split.)*

### 8.1 The split is deterministic and stratified

```bash
.venv/bin/python -c "
from collections import Counter
from utils.dataset import split_dataset
tr, va = split_dataset('leaves/images', 0.2, 42)
tr2, va2 = split_dataset('leaves/images', 0.2, 42)
print('train', len(tr), 'val', len(va))
print('deterministic:', tr == tr2 and va == va2)
print('overlap:', len({p for p,_ in tr} & {p for p,_ in va}))
c1, c2 = Counter(l for _,l in tr), Counter(l for _,l in va)
for k in sorted(c2): print(f'  {k:<18} val fraction {c2[k]/(c1[k]+c2[k]):.3f}')
"
```
Expected: `train 5777 val 1444`, `deterministic: True`, `overlap: 0`, and
every per-class fraction ≈ 0.200.

### 8.2 No validation image leaked into the augmented set

```bash
.venv/bin/python -c "
import os
from utils.dataset import split_dataset
from utils.naming import split_augmented
tr, va = split_dataset('leaves/images', 0.2, 42)
val = {(l, os.path.basename(p)) for p, l in va}
print('train/val overlap:', len(val & {(l, os.path.basename(p)) for p, l in tr}))
leak = sum((cls, split_augmented(f)[0] + '.JPG') in val
           for cls in os.listdir('augmented_directory')
           for f in os.listdir(os.path.join('augmented_directory', cls)))
print('val images inside augmented_directory:', leak)
"
```
Expected: **both 0**.

> **Say:** "The split happens before augmentation, and balancing is handed
> a list of training items — not a directory — so it structurally cannot
> see the validation half."

### 8.3 Duplicates were already in the data, and don't carry the score

The raw PlantVillage set contains 3 byte-identical pairs and 17
near-duplicates spanning train and val (1.18% of validation). Removing
every one of them leaves accuracy at **100% (1427/1427)**. The full
commands are in [`guide.md`](guide.md) §11 — run them if pushed, they
take about a minute each.

### 8.4 The four-sentence answer

1. The validation set was held out **before** any augmentation, with a
   seeded stratified split you can reproduce right now.
2. The known duplicates are in the dataset as delivered; deleting all of
   them doesn't move the number.
3. These are lab-condition PlantVillage images — one centred leaf,
   uniform background — so an ImageNet backbone is *expected* to sit near
   ceiling.
4. The mistakes it does make are botanically sensible
   (`Grape_Esca → Grape_Black_rot`), not random, which is what a leak or
   a label bug would produce.

---

## 9. Error handling sweep

"Programs must not crash on bad input." Run the lot — every one exits `1`
with a single `error: …` line and **no traceback**.

```bash
.venv/bin/python Distribution.py /nope;                          echo "exit=$?"
.venv/bin/python Distribution.py docs;                           echo "exit=$?"
.venv/bin/python Augmentation.py /nope.JPG;                      echo "exit=$?"
.venv/bin/python Augmentation.py -src /nope -dst /tmp/x;         echo "exit=$?"
.venv/bin/python Transformation.py /nope.JPG;                    echo "exit=$?"
.venv/bin/python Transformation.py -src /nope -dst /tmp/x;       echo "exit=$?"
.venv/bin/python predict.py /nope.JPG --no-display;              echo "exit=$?"
echo junk > /tmp/bad.JPG
.venv/bin/python predict.py /tmp/bad.JPG --no-display;           echo "exit=$?"
.venv/bin/python predict.py "leaves/images/Apple_rust/image (1).JPG" \
    --model /tmp/bad.JPG --no-display;                           echo "exit=$?"
.venv/bin/python train.py /nope;                                 echo "exit=$?"
```

Argparse misuse exits `2`, which is also correct:
```bash
.venv/bin/python Augmentation.py;                    echo "exit=$?"   # 2
.venv/bin/python Augmentation.py x.JPG -src y;       echo "exit=$?"   # 2
```

---

## 10. Optional — retrain from scratch

Only if the evaluator asks and there is time.

**Fast version (~1 minute)** — a small subset, 1 epoch. It proves the
pipeline runs end to end; it will *not* hit 90%.

```bash
# build a small imbalanced subset
MINI=/tmp/mini; rm -rf "$MINI"
for d in leaves/images/*/; do
  c=$(basename "$d"); mkdir -p "$MINI/$c"
  ls "$d" | head -25 | while read f; do cp "$d$f" "$MINI/$c/"; done
done
ls "$MINI/Apple_rust" | tail -18 | while read f; do rm "$MINI/Apple_rust/$f"; done

# run it from a scratch copy so the real learnings/ survives
mkdir -p /tmp/run && cp -r *.py utils /tmp/run/
cd /tmp/run && /path/to/leaffliction/.venv/bin/python train.py "$MINI" \
    --epochs 1 --batch-size 16 --workers 2 \
    --augmented-dir /tmp/run/aug --out /tmp/run/learnings.zip
```

**Full version (~20 minutes, 12-core CPU)**:
```bash
.venv/bin/python -u train.py leaves/images --epochs 6 --batch-size 64 --workers 4
```
Prints per-epoch loss and validation accuracy, then writes
`learnings.zip`. **Warning:** this overwrites `learnings/`,
`learnings.zip` and `augmented_directory/` in the working directory. It
does **not** touch `dataset.zip`, so the signature stays valid — but do
not rebuild the release afterwards.

---

## 11. Who answers what

| Topic | Owner | Their guide |
|---|---|---|
| `utils/dataset.py`, `utils/naming.py`, Parts 1 & 2 | **A** | [`PERSON_A_GUIDE.md`](PERSON_A_GUIDE.md) |
| Part 3, `dataset.zip`, `signature.txt`, the repo audit | **B** | [`PERSON_B_GUIDE.md`](PERSON_B_GUIDE.md) |
| `utils/preprocess.py`, Part 4, the accuracy defence | **C** | [`PERSON_C_GUIDE.md`](PERSON_C_GUIDE.md) |

Cross-cutting answers everyone should have ready:

- **"Why PyTorch when the brief says `model.keras`?"** Python 3.14 has no
  TensorFlow wheel. The artifact is `model.pt`; nothing else changed.
- **"Where are the class labels stored?"** In the directory names, nowhere
  else.
- **"What is the image format between modules?"** numpy `uint8 (H, W, 3)`,
  **RGB**. OpenCV is BGR, so the conversion happens once, inside
  `utils/dataset.py`.
- **"Why is `augmented_directory/` built by `train.py`?"** Because the
  split must come first. `train.py` calls A's `build_augmented_dir` with
  only the training items.

---

## 12. Failure playbook

| It goes wrong | Do this |
|---|---|
| `ModuleNotFoundError: torch` | You used system `python3`. Every command is `.venv/bin/python`. |
| No window opens | Headless session. Use `--no-display` / `--save-dir` / `--save-to`; `predict.py` writes `prediction.png` by itself. |
| `UserWarning: FigureCanvasAgg is non-interactive` | Same cause, harmless. Say so and move on. |
| `error: no model at learnings.zip` | `learnings.zip` isn't in the working directory. Pass `--model /path/to/learnings.zip`, or `--model learnings/`. |
| Signature mismatch | Do **not** rebuild the zip to "fix" it — that changes the hash again. Show `sha1sum dataset.zip` against `signature.txt` from the copy you prepared. |
| A prediction is wrong on some image | Expected on near-duplicates or odd crops. Show `metrics.json`: the claim is 1444 held-out images, not "never wrong on anything". |
| Distribution counts changed mid-demo | Someone ran `Augmentation.py` on a file inside `leaves/images/`. Delete the `*_Flip.JPG`-style files: `find leaves/images -name '*_[A-Z]*.JPG' -delete` — then re-check the counts. |
| A command hangs at `build_model` | First run downloads ~45 MB of pretrained ResNet weights into `~/.cache/torch/`. Pre-warm it before the defense. |
