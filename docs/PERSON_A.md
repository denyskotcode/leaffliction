# Person A — Foundations + Distribution + Augmentation

You own the shared plumbing everyone else depends on, plus the two parts that lean hardest on directory-walking. **Ship `utils/` first (Day 1–2)** — B and C are blocked until you do.

**Your files:** `utils/dataset.py`, `utils/naming.py`, `Distribution.py`, `Augmentation.py`, `requirements.txt`

---

## Task 1 — `utils/` (do this FIRST, it unblocks the team)

### `utils/dataset.py`
```python
list_images(root)     -> list[tuple[str path, str label]]
load_image(path)      -> np.ndarray            # RGB, uint8, shape (H, W, 3)
save_image(img, path) -> None                  # expects RGB uint8
class_counts(root)    -> dict[str label, int]
split_dataset(root, val_ratio=0.2, seed=42) -> (train_list, val_list)  # stratified per class
```
- **label** = leaf directory name (e.g. `apple_healthy`). **category** = prefix before first `_` (e.g. `Apple`).
- If you use OpenCV: convert **BGR→RGB on load** and **RGB→BGR on save**, inside these functions. Nobody else should touch color order.
- `split_dataset` must be **stratified** (each class split by `val_ratio`) and **seeded** (reproducible).

### `utils/naming.py`
- Helper that builds `<basename>_<Aug>.JPG` for `Aug ∈ {Flip, Rotate, Skew, Shear, Crop, Distortion}` (title-case, matches the subject's `ls` example).

**Exit for this task:** committed module with the exact signatures above. Ping B and C the moment it lands.

---

## Task 2 — `Distribution.py`

`Distribution.py <dir>` walks subdirectories, counts images per class, and displays:
- a **pie chart** and a **bar chart** per plant type,
- chart columns/labels named from the directory names.

Must work on **any subtree** of the dataset, not just `Apple/`. Use `class_counts` from your own `utils`.

---

## Task 3 — `Augmentation.py`

`Augmentation.py <image>` displays and saves **6 augmentations** of the given image:
- **Flip, Rotate, Skew, Shear, Crop, Distortion**
- each saved next to the original as `<base>_<Aug>.JPG` (use `utils/naming.py`).
- expose `augment_image(img_rgb) -> dict[str name, np.ndarray]` returning all 6.

---

## Task 4 — Balancing pass (can live inside `Augmentation.py`)

- Walk `root`, read `class_counts`, and augment **minority classes up to the largest class count**.
- Write the balanced result into `augmented_directory/` (structure `<class>/*.JPG`).
- This folder is what C trains on and what B zips for release.

---

## Entry (what you need to start)
- Just the raw dataset path, in the agreed layout `root/<class>/*.JPG`. Nothing from B or C.

## Exit (what you hand over)

| To | Deliverable | Format |
|---|---|---|
| B, C | `utils/dataset.py`, `utils/naming.py` | Python module, exact signatures above; images RGB `uint8 (H,W,3)` |
| C | `augmented_directory/` (balanced) | On-disk `<class>/*.JPG`, names `<base>_<Aug>.JPG` |
| B | Same balanced folder (for the release zip) | Same as above |

---

## Watch out
- RGB vs BGR: fix it once, in `load_image`/`save_image`. This is the most common silent bug in the whole project.
- `Distribution.py` must not hardcode `Apple` — pull names from the directory.
- Code must pass `flake8` and not crash on a bad/empty directory.
