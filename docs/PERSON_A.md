# Person A — Foundations + Distribution + Augmentation

You own the shared plumbing everyone else depends on, plus the two parts
that lean hardest on directory-walking.

**Your files:** `utils/dataset.py`, `utils/naming.py`, `Distribution.py`,
`Augmentation.py`, `requirements.txt`

**Status: all four tasks delivered.** What follows is the shipped state.
Function-by-function detail and runnable tests live in
[`guide.md`](guide.md) §3, §4, §6, §7.

---

## Task 1 — `utils/` (shipped first, it unblocked the team)

### `utils/dataset.py`
```python
list_images(root, include_root=False) -> list[tuple[str path, str label]]
load_image(path)                      -> np.ndarray   # RGB, uint8, (H, W, 3)
save_image(img, path)                 -> None         # image FIRST, path second
class_counts(root, include_root=False)-> dict[str label, int]
group_by_label(items)                 -> dict[str label, list]
split_dataset(root, val_ratio=0.2, seed=42) -> (train_list, val_list)
```
- **label** = leaf directory name (e.g. `Apple_healthy`). **plant type** =
  prefix before the first `_` (e.g. `Apple`), via `naming.plant_type`.
- BGR→RGB on load and RGB→BGR on save happen **here and nowhere else**.
- `split_dataset` is **stratified** (each class split by `val_ratio`) and
  **seeded**, and never empties a class from either side.
- `include_root=True` labels loose images in `root` with `root`'s own
  name — added so Parts 1 and 2 accept a single class directory. Training
  never uses it, so the split behind C's accuracy claim is untouched.

### `utils/naming.py`
```python
AUG_NAMES = ("Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion")
stem(path) / augmented_filename(source, aug, round_id=0) / augmented_path(...)
split_augmented(filename) / is_augmented(filename) / plant_type(label)
```
`augmented_filename` builds `<base>_<Aug>.JPG`; `round_id > 0` appends a
number so a class small enough to reuse its sources cannot overwrite its
own earlier copies. `split_augmented` is the exact inverse and is what the
leakage check in `guide.md` §11 uses — labels contain underscores of their
own (`Apple_Black_rot`), so a trailing segment counts only when it names
one of the six augmentations.

---

## Task 2 — `Distribution.py`

```bash
./Distribution.py leaves/images
./Distribution.py leaves/images --save-dir charts --no-display
./Distribution.py leaves/images/Apple_rust        # a single class works too
```

Pie chart + bar chart per plant type, every label pulled from a directory
name. Also prints the counts, so the analysis survives with no display.
Returns `1` with a one-line `error: ...` on a missing or image-free
directory.

Colours come from an eight-slot palette checked against colour-vision
deficiency simulation; the first four (what this dataset needs per plant)
stay distinguishable in every simulation. Every chart carries a written
label and a legend regardless, so identity never rests on colour.

---

## Task 3 — `Augmentation.py`, single image

```bash
./Augmentation.py "leaves/images/Apple_healthy/image (1).JPG"
```

Displays and saves the six augmentations beside the original as
`<base>_<Aug>.JPG`. Public API:

```python
flip / rotate / skew / shear / crop / distortion (img_rgb, rng=None)
AUGMENTATIONS                     # {title-case name: function}
augment_image(img_rgb, rng=None) -> dict[str, np.ndarray]   # all six
```

All geometric augmentations reflect at the border (`BORDER_REFLECT_101`)
rather than padding with black — a flat black wedge is a feature no real
leaf photo has, and the network would learn it as a shortcut to whichever
classes needed the most augmenting.

---

## Task 4 — Balancing pass (inside `Augmentation.py`)

```bash
./Augmentation.py -src leaves/images -dst augmented_directory
```

```python
build_augmented_dir(items, out_dir, seed=42) -> list[tuple[str, str]]
```

- Takes `[(path, label)]`, **not** a directory — that is what lets
  `train.py` balance its training split alone and leave the validation
  split untouched.
- Copies every image in, then augments the smaller classes up to the
  largest.
- Round numbering uses `math.lcm(len(paths), 6)`: a `(source,
  augmentation)` pair repeats every `lcm(n, 6)` steps, not `n × 6`. The
  wrong period lets a later file overwrite an earlier one, which
  under-fills precisely the rarest classes.
- Self-checking: counts the files actually on disk per class and raises
  `RuntimeError` if that disagrees with the target.

---

## Entry / Exit

**Entry:** the raw dataset path, layout `root/<class>/*.JPG`. Nothing from
B or C.

| To | Deliverable | Format |
|---|---|---|
| B, C | `utils/dataset.py`, `utils/naming.py` | Python module, signatures above; images RGB `uint8 (H,W,3)` |
| C | `build_augmented_dir` + the six augmentations | Imported by `train.py`, so training uses exactly this code |
| B | `augmented_directory/` (balanced) | On-disk `<class>/*.JPG`, names `<base>_<Aug>.JPG` |

---

## Watch out
- RGB vs BGR: fixed once, in `load_image`/`save_image`. This is the most
  common silent bug in the whole project — don't re-flip anywhere else.
- `save_image(img, path)` takes the **image first**. Calling it
  `(path, img)` is what broke Part 3's batch mode until it was fixed.
- `Distribution.py` must not hardcode `Apple` — names come from the
  directories.
- `flake8` on default settings (79 columns), and no crash on a bad or
  empty directory.
