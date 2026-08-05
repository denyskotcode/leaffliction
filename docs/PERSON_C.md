# Person C — Classification (train + predict)

The biggest and riskiest chunk: the model and the **≥90% validation
accuracy gate**.

**Your files:** `train.py`, `predict.py`, `utils/preprocess.py`

**Status: delivered — 100% validation accuracy on 1444 held-out images**
(`best_epoch: 6`). Function-by-function detail and runnable tests live in
[`guide.md`](guide.md) §5, §9, §10.

> **Framework note.** This machine runs Python 3.14, for which TensorFlow
> publishes no wheel, so Part 4 uses **PyTorch**. The artifact is
> `model.pt` rather than the `model.keras` in the original brief;
> everything else in `learnings.zip` is unchanged.

---

## Task 1 — `utils/preprocess.py` (shared, defined early)

```python
IMAGE_SIZE = (128, 128)
PREPROCESS_NAME = "resize128_rgb_float_imagenet_norm_v1"
preprocess(img_rgb) -> np.ndarray    # float32 (3, 128, 128)
```

Resize (`INTER_AREA`) → scale to `[0, 1]` → ImageNet mean/std → channels
first. **Both `train.py` and `predict.py` import this one function.**
Identical preprocessing at train and predict time is non-negotiable — a
mismatch quietly tanks accuracy while training still converges. The name
is written into `labels.json`, and `predict.py` warns if a loaded model
was trained under a different one.

---

## Task 2 — `train.py <dir>`

```bash
./train.py leaves/images --epochs 6 --batch-size 64 --workers 4
```

1. **Split first** — `split_dataset(root, val_ratio=0.2, seed=42)` from
   A's `utils`, stratified and seeded.
2. **Balance the training split only** — calls A's
   `build_augmented_dir(train_items, augmented_directory)`. Importing A's
   function rather than receiving a pre-built folder is what keeps
   augmented copies of validation images out of training.
3. **Fine-tune an ImageNet-pretrained ResNet-18** (AdamW `lr=3e-4`,
   cosine schedule, cross-entropy).
4. **Evaluate on the untouched split each epoch** and keep the weights
   from the best one.
5. **Write `learnings.zip`:**
```
model.pt              # state_dict + architecture + num_classes
labels.json           # {"classes":[...index-ordered...], "input_size":[H,W], "preprocess":"..."}
metrics.json          # {"val_accuracy":1.0, "val_count":1444, "best_epoch":6, "per_class":{...}, "confusion":[[...]]}
augmented_directory/  # the modified images used for training
```
`classes` order **must** match the model's output index order — it is the
same list used to build `LeafDataset`, so it does by construction.

Flags: `--epochs`, `--batch-size`, `--val-ratio`, `--augmented-dir`,
`--out`, `--workers`. Returns `1` on a bad directory, an empty dataset, or
accuracy below 90%; warns when the validation set is smaller than 100
images.

---

## Task 3 — `predict.py <image>`

```bash
./predict.py "leaves/images/Apple_healthy/image (1).JPG"
./predict.py <image> --model learnings.zip --save-to prediction.png
./predict.py <image> --no-display
```

1. Reads **only** `model.pt` + `labels.json`, from a `learnings.zip` or an
   unpacked folder.
2. Preprocesses with the **same** `utils/preprocess.py`.
3. Displays the original beside B's `transformed_for_display` rendering —
   wrapped in `try/except`, so a display failure never takes down the
   classification.
4. Prints the class and its confidence:
```
Class predicted : Apple_healthy
Confidence      : 99.87%
```
Headless, it writes `prediction.png` instead of opening a window. Returns
`1` on a missing image, missing model, unreadable image or unloadable
model — never a traceback.

---

## Task 4 — Hit and prove ≥90%

Delivered: **1444 validation images, 100% correct**, proof in
`metrics.json` (accuracy, count, best epoch, per-class recall, confusion
matrix).

A near-perfect score invites scrutiny, so the honesty checks in
`guide.md` §11 are reproducible on demand:

| Check | Result |
|---|---|
| Validation images | 1444 (requirement: ≥100) |
| Train/val overlap | 0 |
| Val images inside `augmented_directory/` | 0 |
| Byte-identical duplicates spanning train/val | 3 pairs (present in the raw dataset as delivered) |
| Near-duplicates (aHash) spanning train/val | 17 val images (1.18%) |
| Accuracy with all near-duplicates removed | **100% (1427/1427)** |

The dataset is genuinely easy — lab-condition PlantVillage images, one
centred leaf, uniform background — so near-ceiling accuracy from an
ImageNet backbone is the expected result, not an anomaly.

---

## Entry / Exit

**Entry:** `utils/dataset.py` + `split_dataset` from **A**;
`build_augmented_dir` and the six augmentations from **A**;
`transformed_for_display` from **B**.

| To | Deliverable | Format |
|---|---|---|
| B | `learnings.zip` | Zip: `model.pt` + `labels.json` + `metrics.json` + `augmented_directory/` |

---

## Watch out
- Preprocessing mismatch between train and predict is the classic silent
  accuracy killer — one shared function, and `labels.json` records its
  name so a stale model announces itself.
- Keep the val split reproducible (`seed=42`) so the accuracy claim is
  defensible at evaluation. Any refactor of `utils/dataset.py` should be
  checked against the previous commit (`guide.md` §11, "Refactoring
  safety net").
- All images arrive as RGB `uint8 (H,W,3)` from A — don't re-flip
  channels.
- `flake8` on default settings (79 columns); `predict.py` must not crash
  on an unseen valid image.
