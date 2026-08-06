# Person A — Understanding Your Code

A guide to *reading* what you own, so you can explain any line of it on
demand. Not a spec (that's [`PERSON_A.md`](PERSON_A.md)) and not a test
recipe (that's [`guide.md`](guide.md)) — this is the mental model.

**You own:** `utils/dataset.py`, `utils/naming.py`, `Distribution.py`,
`Augmentation.py`

**You will be asked to run:** Part 1 (Distribution) and Part 2
(Augmentation). You will also be asked about `utils/`, because B and C
both stand on it.

---

## 1. The one-paragraph summary

> Every image in this project enters memory through `utils/dataset.py` and
> leaves through it, always as RGB `uint8 (H, W, 3)`. The class label is
> the name of the folder the image sits in — nothing else records it.
> `Distribution.py` counts those folders and charts them.
> `Augmentation.py` fixes the fact that the folders are wildly unequal in
> size, by generating six kinds of variation until every class has as many
> images as the biggest one. `train.py` calls my balancing function
> directly, so the model trains on exactly the images my code produces.

If you can say that, you understand your part. The rest is detail.

---

## 2. Reading order

Read in this order — each file only depends on the ones above it.

1. `utils/naming.py` (86 lines) — pure string rules, no dependencies.
2. `utils/dataset.py` (121 lines) — I/O and the split.
3. `Distribution.py` (201 lines) — uses both, no other logic.
4. `Augmentation.py` (338 lines) — the six augmentations + balancing.

Total ~750 lines, half of it comments. An hour with this guide beside you
is enough.

---

## 3. `utils/naming.py` — conventions, not data

**The problem it solves:** nothing on disk records that
`image (1)_Flip.JPG` is a flipped copy of `image (1).JPG`, or that
`Apple_Black_rot` belongs to the plant `Apple`. Those are conventions.
If three files each spell the convention out separately, they drift.
So they live here once.

| Function | Rule |
|---|---|
| `stem(path)` | basename minus extension |
| `augmented_filename(source, aug, round_id=0)` | `image (1).JPG` + `Flip` → `image (1)_Flip.JPG` |
| `augmented_path(...)` | same, with a directory in front |
| `split_augmented(filename)` | the inverse: `('image (1)', 'Flip')` |
| `is_augmented(filename)` | did we make this file? |
| `plant_type(label)` | `Apple_Black_rot` → `Apple` |

**The subtle one is `split_augmented`.** Class labels contain underscores
of their own, so you cannot just split on `_` and take the last piece —
`Apple_Black_rot.JPG` would come back as augmentation `rot`. The code
only accepts the trailing segment when it actually appears in
`AUG_NAMES`. Test it and watch: `Apple_Black_rot.JPG` → `(…, None)`.

**Why `round_id` exists.** Balancing sometimes needs more copies than
`sources × 6`. `Apple_rust` has 220 training images and needs 1312, so
every source gets used many times over. Without a round number, the
second pass would write `image (1)_Flip.JPG` again, overwriting the
first. `round_id=1` makes it `image (1)_Flip1.JPG`. Round 0 keeps the
plain name the subject's `ls` example shows.

---

## 4. `utils/dataset.py` — the foundation everyone stands on

### The RGB rule
OpenCV reads and writes **BGR**. Everyone else in this project expects
**RGB**. So `load_image` converts on the way in and `save_image` converts
on the way out, and **no other file in the project touches channel
order**. If red and blue ever look swapped, the bug is a conversion
someone added elsewhere — not here.

Note the argument order: `save_image(img, path)`. Image first. Getting it
backwards raises `ValueError: save_image expects an (H, W, 3) ndarray`,
which is exactly how Part 3's batch mode bug was found.

### `list_images(root, include_root=False)`
Walks recursively, returns `[(path, label)]` **sorted by path**.

- The label is `os.path.basename` of the folder holding the file.
- **Why sorted?** Determinism. `os.walk` returns filesystem order, which
  differs between machines. `split_dataset` shuffles this list with a
  fixed seed — a shuffle of an unsorted list is not reproducible, so the
  sort is what makes "seed 42 gives the same split everywhere" true.
- **What is `include_root`?** Images lying loose in `root` have no class
  folder. Under a dataset root they're strays, so they're skipped. But if
  someone runs `Distribution.py leaves/images/Apple_rust`, *every* image
  is loose — with the default you'd count zero. `include_root=True`
  labels them with the folder's own name. Parts 1 and 2 fall back to it;
  **training never does**, so C's accuracy claim is unaffected.

### `split_dataset(root, val_ratio=0.2, seed=42)`
Three properties, and you should be able to name all three:

1. **Stratified** — each class is split separately, so every class
   contributes ~20% to validation. A global shuffle could hand you a
   validation set with no `Apple_rust` in it at all.
2. **Seeded** — `random.Random(seed)`, so the same images land on the
   same side every run. This is what makes C's accuracy claim checkable
   rather than a story.
3. **Never empties a class** — the held-out count is clamped to
   `[1, len-1]` for any class with ≥2 images.

Result on the real dataset: **5777 train / 1444 validation**.

### `group_by_label(items)`
Trivial bucketing helper, `{label: [items]}`. Shared so `split_dataset`
and anything else that needs per-class buckets agree.

---

## 5. `Distribution.py` — Part 1

### The shape of it
`main()` → `collect_counts` → `group_by_plant` → `report` (text) →
`draw_plant` per plant (charts).

### Four things to be able to explain

**The matplotlib backend block at the top.** On a machine with no
display, the default backend *raises* rather than drawing. So
`matplotlib.use("Agg")` is called **before** `pyplot` is imported — the
backend can't be changed afterwards. That ordering is also why the
imports below it carry `# noqa: E402` (flake8 wants imports at the top;
here they legitimately can't be).

**`collect_counts` has a fallback.** `class_counts(root) or
class_counts(root, include_root=True)` — an empty result means nothing
was nested, so try again counting the folder's own images. That single
line is what makes "must work on any subtree" true, right down to one
class directory.

**Names come from directories, never from code.** There is no list of
class names anywhere in this file. `plant_type(label)` splits on the
first underscore to group `Apple_*` charts together. Point at a folder of
tomatoes and it charts tomatoes.

**Colour is never the only signal.** `PALETTE` has eight slots, checked
against colour-vision-deficiency simulation; the first four (what this
dataset needs per plant) stay distinguishable in every simulation. But
every chart also carries a legend, and the bar chart is labelled and
annotated with exact counts — so a colour-blind evaluator loses nothing.

### The numbers you should know by heart
```
Apple: 3164 images / 4 classes    Apple_rust     275  (smallest)
Grape: 4057 images / 4 classes    Apple_healthy 1640  (largest)
Total: 7221 across 8 classes      → roughly 6× imbalance
```
That 6× is the entire reason Part 2 exists. Say it in that order and the
two parts connect.

---

## 6. `Augmentation.py` — Part 2

### Two modes, one code path
```bash
./Augmentation.py "<image>"                          # six variants, displayed + saved
./Augmentation.py -src <dir> -dst augmented_directory # balance a whole set
```
`main()` refuses both-at-once and neither-at-all, then dispatches to
`run_single` or `run_balance`.

### The six augmentations

Every one takes `(img, rng=None)` and returns the **same shape**. Two
helpers keep them short:

- `_rng(rng)` — falls back to `random.Random(SEED)` if no generator is
  passed, so each function works standalone *and* shares one stream when
  called in sequence.
- `_warp(img, matrix)` — a 2×3 matrix goes to `warpAffine`, a 3×3 to
  `warpPerspective`. Same call otherwise, so the matrix shape picks.

| | What it does | The parameter |
|---|---|---|
| `flip` | horizontal mirror | none — deterministic |
| `rotate` | turn about the centre | angle ±30° |
| `skew` | perspective tilt, top edge pinched | 8–20% of width |
| `shear` | horizontal slant, height preserved | factor ±0.25 |
| `crop` | random window, scaled back up | keeps 70–88% |
| `distortion` | barrel/pincushion lens bend | strength ±0.35 |

**The one design decision to defend: `BORDER_REFLECT_101`.** Rotating an
image leaves empty corners. Fill them with black and you've painted a
feature into the data that no real leaf photo has — and the classes that
got augmented most (the rare ones) would carry the most black. The
network would learn "black wedge ⇒ Apple_rust". Reflecting the image at
the border keeps the statistics plausible. This is the single best answer
you can give about Part 2.

**`distortion` is the only non-trivial maths.** Normalise coordinates to
`[-1, 1]` about the centre, then push each pixel outward (or inward) in
proportion to its **squared** distance from the centre:
`stretch = 1 + strength · (x² + y²)`. Squared distance is what bends
straight lines — a linear term would just scale the image. `cv2.remap`
then samples the source at those coordinates.

**`flip` ignores its `rng`.** That's intentional, not an oversight: a
mirror has no parameter. It still keeps the `(img, rng)` signature so
`AUGMENTATIONS` can be called uniformly.

### The balancing pass — the part that hides a real bug

`build_augmented_dir(items, out_dir, seed)`:

1. Wipe `out_dir` (runs are reproducible, not cumulative).
2. Bucket `items` by label; the target is the **largest** bucket.
3. Per class: copy the originals, then `_fill_class` tops it up.
4. Count the files actually on disk; **raise `RuntimeError`** if that
   doesn't equal the target.

Two things to be ready for:

**Why does it take `items`, not a directory?** Because `train.py` passes
it *only the training split*. If it took a directory it would balance
everything, including validation images, and augmented copies of
validation images would end up in training. Accuracy would look
brilliant and mean nothing. The signature is the safeguard.

**Why `math.lcm(len(paths), 6)`?** `_fill_class` walks sources and
augmentations in step:

```python
source = paths[step % len(paths)]
name   = AUG_NAMES[step % 6]
```

The **pair** `(source, name)` repeats when `step` is a multiple of *both*
periods — that is every `lcm(n, 6)` steps, not every `n × 6`. With 220
sources, `lcm(220, 6) = 660`, not 1320. Numbering rounds by `n × 6` would
mean step 660 reuses round 0's filename and silently overwrites it —
under-filling exactly the rarest class. The `RuntimeError` guard exists
because that bug is otherwise invisible: you just quietly get fewer
images than you asked for.

Result on the real training split: **8 classes × 1312 images**.

---

## 7. Questions you should expect

**"Show me it works on a different folder."**
`./Distribution.py leaves/images/Grape_Esca` — one class, charted. The
fallback in `collect_counts` handles it.

**"What happens if I point it at an empty folder?"**
`error: no images found under …`, exit 1. No traceback. Same for a
missing folder (`error: not a directory: …`).

**"Are your augmentations random? Then how is training reproducible?"**
Yes, random — from a seeded generator. `SEED = 42`, and `train.py`
imports that same constant. Same seed, same images, byte for byte.
(Demonstrable: balance twice into two folders and `diff -r` them.)

**"Why not just delete images from the big classes instead?"**
That's the other way to balance, and it throws away 4/5 of `Apple_healthy`.
Augmenting keeps every real image *and* adds plausible variation, which
also acts as regularisation.

**"Your rotate uses reflection — doesn't that create fake leaf tissue?"**
Yes, at the corners. The alternative is fake *black*, which is worse: it
correlates with class, and a network will take any shortcut it's offered.
Reflected tissue at least resembles the distribution the model must
handle.

**"Where is the class label stored?"**
In the directory name, nowhere else. No CSV, no filename encoding. That
is why `list_images` returns the folder basename as the label.

---

## 8. Live-demo cheatsheet

```bash
# Part 1 — whole set, then a subtree
.venv/bin/python Distribution.py leaves/images
.venv/bin/python Distribution.py leaves/images/Apple_rust

# Part 1 — headless / save figures
.venv/bin/python Distribution.py leaves/images --save-dir charts --no-display

# Part 2 — one image, six variants beside it
.venv/bin/python Augmentation.py "leaves/images/Apple_healthy/image (1).JPG"

# Part 2 — balance a whole set (~2 min on the full dataset)
.venv/bin/python Augmentation.py -src leaves/images -dst /tmp/balanced
for d in /tmp/balanced/*/; do echo -n "$(basename $d): "; ls "$d" | wc -l; done

# Error handling — all exit 1, no traceback
.venv/bin/python Distribution.py /nope
.venv/bin/python Augmentation.py /nope.JPG
```

See [`EVALUATION.md`](EVALUATION.md) for the full defense script.
