# Person B — Transformation + Release

The most independent part, which is why it also owns the final release
(zip + signature). Your work, A's balanced data and C's model converge at
the end — you assemble it.

**Your files:** `Transformation.py`, plus `dataset.zip` + `signature.txt`

**Status: delivered.** Function-by-function detail and runnable tests
live in [`guide.md`](guide.md) §8.

---

## Task 1 — `Transformation.py`

Six transformations, built on OpenCV/NumPy in the spirit of the PlantCV
pipeline the subject shows: **original, Gaussian blur, mask, ROI objects,
analyze object, pseudolandmarks**, plus a **colour histogram**
(Figure IV.7).

Exposed for Person C:
```python
transform_image(img_rgb) -> dict[str, np.ndarray]
#   keys: original, gaussian_blur, mask, roi_objects, analyze_object, pseudolandmarks
color_histogram(img_rgb) -> matplotlib.Figure
transformed_for_display(img_rgb) -> np.ndarray   # the single image predict.py shows
leaf_outline(img_rgb) -> (mask, contour)         # the segmentation the rest build on
```

`leaf_outline` is the core: an Otsu threshold over the **HSV saturation**
channel separates coloured leaf tissue from the uncoloured background with
no hand-tuned constant, morphology fills highlight holes, and the largest
contour drops stray flecks. It returns the mask and the contour together
because every caller needs both; `contour` is `None` when nothing is
found, and each renderer then returns the image untouched rather than
raising.

---

## Task 2 — Dual mode + CLI

```bash
./Transformation.py -h
./Transformation.py "leaves/images/Apple_healthy/image (1).JPG"        # display
./Transformation.py -src leaves/images/Apple_healthy -dst out -mask    # batch
./Transformation.py -src leaves/images/Apple_healthy -dst out          # everything
```

- **Single image path** → displays the full set on a 3-column grid, plus
  the histogram as a second figure.
- **`-src` + `-dst`** → saves each requested transformation as
  `<base>_<Transformation><ext>`, and the histogram as
  `<base>_Histogram.png`. Recursive, and it accepts a single class
  directory.
- Per-transform flags: `-blur`, `-mask`, `-roi`, `-object`, `-landmarks`,
  `-histogram`. **No flag means all of them**, histogram included.
- Batch mode forces the `Agg` backend; nothing is displayed and a headless
  machine would otherwise refuse to open a window.
- Bad file or directory → one-line `error: ...`, exit `1`. No arguments →
  help, exit `1`.

> **Fixed since the first version:** batch mode called
> `save_image(path, img)` while `utils.dataset.save_image` takes
> `(img, path)`, so `-src/-dst` crashed on every invocation. Arguments are
> now correct and the local I/O fallback that hid the mismatch is gone —
> the module imports A's `utils.dataset` unconditionally.

---

## Task 3 — Release engineering (END of project)

Once the augmented dataset (A) and `learnings.zip` (C) exist:

```bash
# 1. assemble the release archive
zip -r dataset.zip leaves/images learnings.zip

# 2. generate the signature
sha1sum dataset.zip | awk '{print $1}' > signature.txt   # Linux
shasum  dataset.zip                                      # macOS

# 3. verify before defense
sha1sum dataset.zip; cat signature.txt
```

4. **Audit the repo:** only code + `signature.txt` may be committed. The
   dataset must NOT be in Git (committing it = grade 0). The signature
   must match the zip exactly at defense.

`signature.txt` is present at the repo root and holds the sha1 of the
current `dataset.zip`.

---

## Entry / Exit

**Entry:** `utils/dataset.py` from **A** (`load_image` / `save_image`,
RGB uint8 ndarrays). For the release step only: A's
`augmented_directory/` and C's `learnings.zip`.

| To | Deliverable | Format |
|---|---|---|
| C | `transform_image` / `transformed_for_display` | Python fns: RGB ndarray in → RGB ndarray / dict out |
| Team | `dataset.zip` + `signature.txt` | Zip (dataset + model) + one-line `sha1` file at repo root |

---

## Watch out
- Keep all image I/O in A's RGB `uint8 (H,W,3)` convention, and remember
  the argument order: `save_image(img, path)`.
- `transformed_for_display` is the one C actually renders in `predict.py`
  — keep it stable and quick. It is wrapped in a `try/except` there, so a
  failure degrades the picture rather than the prediction, but that is a
  safety net, not a licence.
- `flake8` on default settings (79 columns); `-h` and both modes must not
  crash.
