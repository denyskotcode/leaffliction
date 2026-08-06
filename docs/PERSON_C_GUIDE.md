# Person C — Understanding Your Code

A guide to *reading* what you own, so you can explain any line of it on
demand. Not a spec (that's [`PERSON_C.md`](PERSON_C.md)) and not a test
recipe (that's [`guide.md`](guide.md)) — this is the mental model.

**You own:** `utils/preprocess.py` (45 lines), `train.py` (292),
`predict.py` (185)

**You will be asked to run:** Part 4, and then to *defend the number*.
You reported 100% on 1444 held-out images. Expect that to be challenged —
sections 7 and 8 are the answer.

---

## 1. The one-paragraph summary

> Split the dataset before touching it, so 1444 images are set aside and
> never seen again until scoring. Balance only the training half, using
> Person A's augmentation code. Fine-tune a ResNet-18 that already knows
> what edges, textures and blobs look like from ImageNet — six epochs is
> enough because only the last layer starts from scratch. Keep the
> weights from the best epoch, score them on the untouched half, and ship
> the weights, the class order and the metrics in one zip. `predict.py`
> reads that zip back and must prepare images *exactly* as training did,
> which is why there is only one preprocessing function in the project.

---

## 2. Reading order

1. `utils/preprocess.py` — 45 lines, and the most correctness-critical
   file in the repo.
2. `train.py` top-to-bottom: `LeafDataset` → `build_model` → `evaluate`
   → `run_epoch` → `train_model` → `write_artifacts` → `main`.
3. `predict.py`: `locate_artifacts` → `load_model` → `predict` → `show`.

---

## 3. `utils/preprocess.py` — one function, two callers

```python
resize to (128, 128), INTER_AREA
scale to [0, 1]
normalise with ImageNet mean/std
transpose to channels-first -> float32 (3, 128, 128)
```

**Why this file exists at all.** If training resizes to 128 and
prediction resizes to 224, or training normalises and prediction doesn't,
accuracy collapses at prediction time *while training still reports
99%*. Nothing warns you. The structural fix is that there is exactly one
function and both programs import it — the mismatch is not merely
unlikely, it is unrepresentable.

**Why `INTER_AREA`?** It's the correct interpolation for *downscaling*
— it averages over the source region instead of point-sampling, so fine
lesion texture survives at 128×128 instead of aliasing.

**Why ImageNet mean/std?** The backbone was pretrained on ImageNet with
those statistics. Feeding it differently-scaled inputs means its learned
filters see a distribution they were never trained on.

**Why 128 and not 256?** The source images are 256×256. Halving the side
quarters the compute, which is what makes six epochs on a CPU feasible,
and lesions are still clearly visible at 128. It's a stated,
reversible trade — `IMAGE_SIZE` is one constant.

**Why does `PREPROCESS_NAME` exist?** It's written into `labels.json`.
`predict.py` compares it to its own and warns on stderr if they differ.
That converts the invisible failure above into a visible message.

**The output is not in `0..1`** — normalisation shifts it to roughly
`−2.12..2.64`. If you see output pinned to `0..1`, the normalisation step
was skipped.

---

## 4. `train.py` — the pipeline, in the order it runs

```
split_dataset()            5777 train / 1444 val   ← held out FIRST
build_augmented_dir()      8 × 1312 = 10496 balanced training images
build_model()              ResNet-18, ImageNet weights, new 8-way head
train_model()              6 epochs; evaluate after each; keep the best
write_artifacts()          learnings.zip
```

### The ordering *is* the design

The split happens **before** augmentation. If you balanced the whole
dataset first and split afterwards, augmented copies of validation images
would sit in the training set — the model would be tested on rotations of
pictures it had memorised. The score would be excellent and worthless.
This is literally the "results shouldn't look suspicious" failure the
subject warns about, and the ordering is your defence against it.

`build_augmented_dir(train_items, …)` takes a **list of items**, not a
directory, precisely so it cannot see the validation half.

### `LeafDataset`

A torch `Dataset` over `[(path, label)]` plus the index-ordered `classes`
list. `__getitem__` loads → preprocesses → returns `(tensor, index)`.

The `classes` list is the contract between the model's output neurons and
the label strings. **The same list** is written to `labels.json`, which
is why `predict.py`'s `classes[argmax]` is correct by construction rather
than by luck.

### `build_model` — transfer learning

```python
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)   # 512 -> 8
```

Everything except the final layer arrives already knowing edges,
textures and colour blobs from 1.2M ImageNet photos. Only the 8-way head
starts from random. That is why six epochs suffice — you are adapting a
feature extractor, not learning vision from scratch.

Why ResNet-18 and not something bigger? It is the smallest ResNet that
comfortably clears the bar, it trains on a CPU in minutes per epoch, and
the dataset is easy. A bigger model would cost more and prove nothing.

### `evaluate`

Returns `(accuracy, count, confusion)` under `torch.no_grad()` in
`eval()` mode. The confusion matrix is `confusion[true][pred]` — **rows
are truth, columns are predictions**. Accuracy is its trace over its sum,
so the reported number and the matrix cannot disagree with each other.

`eval()` matters: it switches BatchNorm to running statistics. Score a
model in `train()` mode and the batch statistics leak between validation
images.

### `train_model`

AdamW (`lr=3e-4`, `weight_decay=1e-4`), cosine LR schedule,
cross-entropy loss.

- **AdamW** — decoupled weight decay; the sane default for fine-tuning.
- **`3e-4`** — small, because most of the network is already trained.
  A large LR would destroy the pretrained features in the first steps.
- **Cosine schedule** — the LR decays smoothly to ~0 by the last epoch,
  which settles the weights instead of leaving them bouncing.
- **Best-epoch checkpointing** — after every epoch it evaluates and
  clones the weights if they beat the previous best; those are reloaded
  at the end. The saved model is the best seen, not merely the last.
  (Here the best *was* the last — `best_epoch: 6`.)

### `write_artifacts`

| File | Contents |
|---|---|
| `model.pt` | `state_dict`, `architecture`, `num_classes` |
| `labels.json` | `classes` (index-ordered), `input_size`, `preprocess` |
| `metrics.json` | `val_accuracy`, `val_count`, `best_epoch`, `classes`, `per_class` recall, `confusion` |

plus `augmented_directory/` — the modified images used for training,
which the subject requires the archive to contain.

> **Framework note you will be asked about.** The brief says
> `model.keras`. This machine runs Python 3.14, for which TensorFlow
> publishes no wheel, so Part 4 uses PyTorch and the artifact is
> `model.pt`. Everything else in the archive is unchanged. Say it in one
> sentence and move on — it's an environment constraint, not a shortcut.

---

## 5. `predict.py`

```
locate_artifacts()   model.pt + labels.json, from a zip OR a folder
load_model()         rebuild ResNet-18 with len(classes) outputs, load weights
predict()            preprocess -> batch of 1 -> softmax -> argmax
show()               original | transformed, titled with the class
```

Four things to point out:

**It reads only two files.** `metrics.json` and the augmented images are
training-time artifacts. A prediction needs weights and class names,
nothing else.

**The preprocessing warning.** If `labels.json` records a different
`preprocess` name than the current build, it warns to stderr. That is the
tripwire for the silent-failure mode described in §3.

**The display is optional by design.** `transformed()` wraps Person B's
`transformed_for_display` in `try/except`: if that module is missing or
throws, the prediction still prints and the original image is shown
instead. Classification must not depend on presentation.

**Headless behaviour.** With no `DISPLAY`, the `Agg` backend is selected
before `pyplot` is imported and the figure is written to
`prediction.png`. `--save-to` names the file explicitly; `--no-display`
skips the figure entirely.

Every error path returns `1` with a one-line `error: …` — missing image,
missing model, unreadable image, model that isn't a zip or folder.

---

## 6. The numbers you must know

| | |
|---|---|
| Total images | 7221, 8 classes |
| Train / validation | 5777 / **1444** (requirement: ≥100) |
| Balanced training set | 8 × 1312 = 10496 |
| Epochs / batch / input | 6 / 64 / 128×128 |
| Best epoch | 6 |
| **Validation accuracy** | **1.0 (100%)** |

---

## 7. Defending 100% — the part that matters

A perfect score *should* invite suspicion. Do not get defensive; have the
evidence ready. Four arguments, in this order:

**1. The split came first, and it's reproducible.**
`seed=42`, stratified. Re-run `split_dataset` in front of the evaluator
and get the same 1444 images. Then show train/val overlap is 0, and that
no validation image appears in `augmented_directory/` (the check is in
`guide.md` §11 and uses `split_augmented`, the same function that built
those filenames).

**2. The duplicates were already in the data, and removing them changes
nothing.** The raw PlantVillage set contains 3 byte-identical pairs and
17 near-duplicates (aHash) spanning train and val — 1.18% of validation.
Delete every one of them and the accuracy is still **100% (1427/1427)**.
So the score does not rest on them.

**3. The dataset is genuinely easy.** Lab-condition images: one centred
leaf, uniform grey background, consistent lighting, 256×256. An ImageNet
backbone fine-tuned on that is *expected* to sit near ceiling. This is
not CIFAR.

**4. The errors it does make are botanically sensible.** In an earlier
run the only two mistakes were `Grape_Esca → Grape_Black_rot`, two
diseases that look alike — not random noise, which is what leakage or a
label bug produces.

---

## 8. Questions you should expect

**"Prove the validation set was never trained on."**
Three commands: re-run the split (deterministic), show zero overlap, show
zero validation images inside `augmented_directory/`. All in
[`EVALUATION.md`](EVALUATION.md) §7.

**"What if I hand it an image from outside the dataset?"**
It classifies it into one of the 8 classes with a confidence — a softmax
always sums to 1. There is no "unknown" class; that's the assignment's
framing, not a bug. Say so plainly.

**"Why is `predict.py` reading a zip instead of a folder?"**
It accepts either. `--model learnings/` works, `--model learnings.zip`
works. The zip is what the subject asks to be delivered.

**"Retrain it now."**
~20 minutes on a 12-core CPU for the full 6 epochs. Offer the mini-set
version (~1 minute) if time is short — but note it rewrites `learnings/`
in the working directory, so run it from a scratch copy if the real
artifacts must survive.

**"Why AdamW and not SGD?"**
Fine-tuning with a small LR; AdamW converges in fewer epochs with less
tuning, which matters when the budget is 6 CPU epochs. SGD+momentum would
also work, with more schedule fiddling.

**"What is `weight_decay` doing?"**
L2-style regularisation, decoupled from the gradient step in AdamW. With
10496 balanced images and a pretrained backbone, overfitting pressure is
low, but it costs nothing.

**"Your training images include augmented copies — doesn't that inflate
the score?"** The score is computed on the validation split, which
contains **no** augmented images at all. `build_augmented_dir` only ever
receives the training items.

---

## 9. Live-demo cheatsheet

```bash
# predict one image
.venv/bin/python predict.py "leaves/images/Apple_healthy/image (1).JPG"

# predict headless / to a file
.venv/bin/python predict.py "<image>" --no-display
.venv/bin/python predict.py "<image>" --save-to /tmp/pred.png

# one image per class
for c in $(ls leaves/images); do
  f=$(ls "leaves/images/$c" | head -1)
  echo -n "$c -> "; .venv/bin/python predict.py "leaves/images/$c/$f" --no-display | grep "Class predicted"
done

# the proof
unzip -p learnings.zip metrics.json | .venv/bin/python -m json.tool | head -20
unzip -l learnings.zip | head -6

# error handling — exit 1, no traceback
.venv/bin/python predict.py /nope.JPG --no-display
.venv/bin/python predict.py "<image>" --model /nope.zip --no-display

# full retrain (~20 min)
.venv/bin/python -u train.py leaves/images --epochs 6 --batch-size 64 --workers 4
```

See [`EVALUATION.md`](EVALUATION.md) for the full defense script.
