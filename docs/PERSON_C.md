# Person C — Classification (train + predict)

You own the biggest and riskiest chunk: the model and the **≥90% validation accuracy gate**. You can start early against a stub/mock and swap in the real data later.

**Your files:** `train.py`, `predict.py`, `utils/preprocess.py`

---

## Task 1 — `utils/preprocess.py` (shared, define this early)

- One function that turns an RGB `uint8 (H,W,3)` image into model input (resize, scale, etc.).
- **Both `train.py` and `predict.py` import it.** Identical preprocessing at train and predict time is non-negotiable — a mismatch quietly tanks accuracy.

---

## Task 2 — `train.py <dir>`

1. `split_dataset(root, val_ratio=0.2, seed=42)` from A's `utils` → stratified, seeded train/val split.
2. Build and train a classifier on the **augmented** training data.
3. Save `learnings.zip` containing:
```
model.keras           # trained model
labels.json           # {"classes":[...index-ordered...], "input_size":[H,W], "preprocess":"..."}
metrics.json          # {"val_accuracy":0.9x, "val_count":N, "confusion":[[...]]}
augmented_directory/  # the modified images used for training
```
- `classes` order in `labels.json` **must** match the model's output index order.

---

## Task 3 — `predict.py <image>`

1. Load `model.keras` + `labels.json` only.
2. Preprocess with the **same** `utils/preprocess.py`.
3. Display the **original** image and the **transformed** image (call B's `transformed_for_display`).
4. Print the predicted class (e.g. `Class predicted : apple_healthy`).

---

## Task 4 — Hit and prove ≥90%

- Validation accuracy **≥ 90%** on a genuinely held-out set of **≥ 100 images**.
- Store the proof in `metrics.json` (accuracy, count, confusion matrix).
- **No peeking / no overfitting to the val set** — the subject explicitly warns results shouldn't "look suspicious."

---

## Entry (what you need to start)
- `utils/dataset.py` + `split_dataset` from **A**.
- `augmented_directory/` from **A** (the real balanced data).
- `transformed_for_display` from **B** (for `predict.py`).

**Unblock trick:** start immediately. Train against the raw dataset or a tiny hand-balanced mock, and stub B's transform (`return img`) until it lands. Swap in the real `augmented_directory/` and B's function later — your training loop won't change.

## Exit (what you hand over)

| To | Deliverable | Format |
|---|---|---|
| B | `learnings.zip` | Zip: `model.keras` + `labels.json` + `metrics.json` + `augmented_directory/` |

---

## Watch out
- Preprocessing mismatch between train and predict is the classic silent accuracy killer — share one function.
- Keep the val split reproducible (`seed=42`) so your ≥90% claim is defensible at evaluation.
- All images arrive as RGB `uint8 (H,W,3)` from A — don't re-flip channels.
- Code must pass `flake8`; `predict.py` must not crash on an unseen valid image.
