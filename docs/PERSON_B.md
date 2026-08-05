# Person B — Transformation + Release

You're the most independent part, so you also own the final release (zip + signature). **Your work + A's balanced data + C's model all converge at the end — you assemble it.**

**Your files:** `Transformation.py`, and the final `dataset.zip` + `signature.txt`

---

## Task 1 — `Transformation.py` (PlantCV recommended)

Implement **≥6 transformations** of a leaf image. The subject shows:
- **Original, Gaussian blur, Mask, ROI objects, Analyze object, Pseudolandmarks**, plus a **color histogram** (Figure IV.7).

Expose these for Person C to import:
```python
transform_image(img_rgb) -> dict[str, np.ndarray]
#   keys: original, gaussian_blur, mask, roi_objects, analyze_object, pseudolandmarks
color_histogram(img_rgb) -> matplotlib.Figure
transformed_for_display(img_rgb) -> np.ndarray   # single "nice" transformed image predict.py will show
```

---

## Task 2 — Dual mode + CLI

- **Single image path** → **display** the full set of transformations.
- **`-src <dir> -dst <dir>`** → **save** all transformations for every image into the destination directory.
- Support per-transform flags like `-mask` (as in the subject's example command).
- Provide `-h` usage that explains the arguments clearly.

Example the subject expects to work:
```
./Transformation.py -src Apple/apple_healthy/ -dst dst_directory -mask
```

---

## Task 3 — Release engineering (END of project)

Once A's balanced dataset and C's `learnings.zip` exist:
1. Assemble `dataset.zip` = the augmented dataset (from A) + the trained model / `learnings.zip` (from C) — everything needed to reproduce results.
2. Generate the signature: `sha1sum dataset.zip` (Linux) or `shasum dataset.zip` (Mac).
3. Write that hash into `signature.txt` at the repo root.
4. **Audit the repo:** only code + `signature.txt` may be committed. The dataset must NOT be in Git (committing it = grade 0). Signature must match the zip at defense.

---

## Entry (what you need to start)
- `utils/dataset.py` from **A** (for `load_image` / `save_image`, RGB uint8 ndarrays).
- For the release step only: A's `augmented_directory/` and C's `learnings.zip`.

## Exit (what you hand over)

| To | Deliverable | Format |
|---|---|---|
| C | `transform_image` / `transformed_for_display` | Python fns: RGB ndarray in → RGB ndarray / dict out |
| Team | `dataset.zip` + `signature.txt` | Zip (dataset + model) + one-line `sha1` file at repo root |

---

## Watch out
- Keep all image I/O in A's RGB `uint8 (H,W,3)` convention. PlantCV and OpenCV like BGR — convert at the boundary so C gets clean RGB back.
- `transformed_for_display` is the one C actually renders in `predict.py` — keep it stable and quick.
- Code must pass `flake8`; `-h` and both modes must not crash.
