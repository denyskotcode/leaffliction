# Person B — Understanding Your Code

A guide to *reading* what you own, so you can explain any line of it on
demand. Not a spec (that's [`PERSON_B.md`](PERSON_B.md)) and not a test
recipe (that's [`guide.md`](guide.md)) — this is the mental model.

**You own:** `Transformation.py` (323 lines), plus the release
(`dataset.zip` + `signature.txt`)

**You will be asked to run:** Part 3, in both modes, with flags and with
`-h`. You will also be asked to prove the signature matches — that check
is pass/fail for the whole group.

---

## 1. The one-paragraph summary

> Every transformation in this file starts from the same question: *where
> is the leaf?* Once you have a binary mask of the leaf and its outline,
> everything else is drawing. `leaf_outline` answers that question using
> saturation and Otsu thresholding; the five renderings on top of it are
> short. The file has two modes — show one image, or batch-save a
> directory — and `predict.py` borrows one function from it to put a
> picture next to each prediction.

---

## 2. Reading order

1. `leaf_outline` — the segmentation. Everything depends on it.
2. `_roi_objects`, `_analyze_object`, `_pseudolandmarks` — three short
   renderers, each guarded against "no leaf found".
3. `transform_image` — assembles the dict. Read it *after* the helpers
   and it is six obvious lines.
4. `color_histogram`, `transformed_for_display` — the two extras.
5. `requested`, `display_single`, `process_batch`, `main` — the CLI.

---

## 3. `leaf_outline` — the only hard part

```python
saturation = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
smoothed   = cv2.GaussianBlur(saturation, (5, 5), 0)
_, rough   = cv2.threshold(smoothed, 0, 255, THRESH_BINARY + THRESH_OTSU)
# morphological close, then open
# keep the largest contour, fill it
return mask, leaf
```

Four decisions, and you should be able to justify each:

**Why the saturation channel?** The leaves are coloured; the PlantVillage
background is a flat grey. Grey has *near-zero saturation* whatever its
brightness. Thresholding on saturation therefore separates leaf from
background without caring how bright the photo is. Using the green
channel instead would fail on a yellowing or brown-spotted leaf — exactly
the diseased ones this project must classify.

**Why Otsu instead of a fixed number?** Otsu picks the threshold that
best splits the histogram into two groups, per image. A hardcoded
constant would need retuning per class and would rot the moment the
dataset changed. (Note the `0` passed as the threshold value: with
`THRESH_OTSU` it is ignored — OpenCV computes it and returns it.)

**Why close, then open?** *Close* (dilate→erode) fills small holes — the
specular highlights on a glossy leaf come out as background otherwise.
*Open* (erode→dilate) removes small specks — dust and JPEG noise in the
background. Order matters: fill the leaf first, then clean around it.

**Why only the largest contour?** A leaf is one connected object.
Anything else the threshold caught is noise. `max(contours,
key=cv2.contourArea)` then `drawContours(..., FILLED)` gives a solid mask
with no interior holes at all.

**And if there is no contour?** It returns `(rough, None)`. Every
renderer takes `contour is None` as "return the image unchanged". The
program never crashes on a picture that isn't a leaf — that's an explicit
subject requirement.

---

## 4. The five renderings

`transform_image` returns a dict of six entries — `original` plus five
renderings — all RGB `uint8`, all the same shape as the input.

| Key | How it's made | What it shows |
|---|---|---|
| `gaussian_blur` | `GaussianBlur(img, (9,9), 0)` | Noise suppression; the classic first step of any CV pipeline. |
| `mask` | `cvtColor(mask, GRAY2RGB)` | The binary mask itself, promoted to 3 channels so it can share a display grid. |
| `roi_objects` | `bitwise_and` + red rectangle | The leaf cut out of its background, plus its bounding box. |
| `analyze_object` | contour + centroid + caption | Green outline, red cross at the centre of mass, `area=… perim=…`. |
| `pseudolandmarks` | 30 sampled outline points | Points spaced evenly along the outline, coloured by height band. |

Two details worth knowing:

**The centroid comes from image moments.** `cx = m10/m00`, `cy = m01/m00`
— the first moments over the zeroth (the area). The `if moments["m00"]`
guard avoids dividing by zero on a degenerate contour.

**Pseudolandmarks are sampled, not detected.** `np.linspace` over the
contour points picks 30 evenly spaced positions; `np.argsort` on their
y-coordinates, split into three, colours them top / middle / bottom. This
mirrors what PlantCV's pseudolandmarks convey — a coarse description of
shape — without the dependency.

---

## 5. `color_histogram`

**Nine curves, because Figure IV.7 names nine channels:** blue,
blue-yellow, green, green-magenta, hue, lightness, red, saturation,
value. They come from three colour spaces, and you should be able to say
why each earns its place:

- **RGB** (red, green, blue) — what the sensor actually recorded.
- **HSV** (hue, saturation, value) — separates *which* colour from *how
  much* of it and *how bright*, so a lesion stays recognisable under a
  different exposure.
- **LAB** (lightness, green-magenta, blue-yellow) — pulls brightness out
  of the colour information entirely; the two opponent axes are where
  green leaf tissue and brown rot separate most cleanly.

`histogram_channels()` returns the nine `(label, channel, colour)`
triples; `color_histogram()` just plots them. Splitting them out means the
channel list can be inspected — or reused — without building a figure.

**One range detail:** OpenCV stores 8-bit hue as 0..179 (degrees halved
to fit a byte), while every other channel spans 0..255. All nine are
binned over 0..255 anyway, so the hue curve simply stops at 179 rather
than being stretched into a range it never occupies. If an evaluator asks
why one curve ends early, that's the answer.

One more detail to defend:

```python
axis.plot(counts / counts.sum(), ...)
```

Counts are divided by their total, so the y-axis is a **proportion of
pixels**, not a raw count. Two images of different sizes then overlay
meaningfully. It's also why the y-axis numbers are small decimals rather
than thousands.

`transformed_for_display` returns the `analyze_object` view — the one
rendering that carries outline, centre and size at a glance. That's what
`predict.py` puts next to each prediction.

---

## 6. The CLI — two modes and one default

```bash
./Transformation.py <image>                            # display mode
./Transformation.py -src <dir> -dst <dir> [-mask …]    # batch mode
./Transformation.py -h
```

**`requested(args)` implements "no flag means everything".**
```python
chosen = [key for key in TRANSFORMS if getattr(args, key)]
if not chosen and not args.histogram:
    return list(TRANSFORMS), True
```
Each flag's `dest` is the same string as its dict key (`-mask` →
`mask`, `-object` → `analyze_object`), so the lookup is a `getattr` and
no flag→key table has to be maintained by hand.

**Batch mode forces `Agg`.** `matplotlib.use("Agg", force=True)` — nothing
is displayed, and on a headless machine matplotlib would otherwise refuse
to build a figure at all. (The file also has the standard headless guard
at the top, before `pyplot` is imported, like every other entry point in
this project.)

**Listing images reuses A's loader.** `list_images(src,
include_root=True)` — recursive, extension-aware, and `include_root`
makes `-src leaves/images/Apple_healthy` (a single class folder) work,
which is precisely the command the subject shows.

**Naming on save:** `<base>_<Transformation><ext>`, e.g.
`image (1)_Mask.JPG`; the histogram goes to `image (1)_Histogram.png`
(PNG because it's a plot, not a photo).

**An unreadable file is skipped, not fatal.** `process_batch` catches per
image and prints `skipping …`, so one corrupt JPEG can't kill a run over
thousands of files.

---

## 7. The bug that was in here — know this story

Batch mode used to call:

```python
save_image(os.path.join(dst, out_name), transforms[key])   # WRONG
```

but `utils.dataset.save_image` is `save_image(img, path)` — image first.
So every `-src/-dst` invocation crashed with
`ValueError: save_image expects an (H, W, 3) ndarray`.

It survived unnoticed because the file *also* carried a local fallback
definition (`def save_image(path, img_rgb)`) used when `utils` was
missing — with the opposite argument order. Two functions, one name, two
signatures.

**The fix:** correct the call, and delete the fallback. `utils.dataset` is
a hard dependency now, imported unconditionally like every other module
does. If an evaluator asks "how do you know your batch mode works?", the
honest answer is: it's tested now — 12 images in, 72 files out (five
renderings plus a histogram each), and that count is checked.

---

## 8. The release — your pass/fail responsibility

```bash
zip -r dataset.zip leaves/images learnings.zip
sha1sum dataset.zip | awk '{print $1}' > signature.txt
```

**What the evaluator does:** hashes your `dataset.zip` and compares it to
`signature.txt`. A mismatch is a **zero**.

Three ways to break it, all avoidable:
1. **Rebuilding the zip after writing the signature.** Zip files embed
   timestamps, so a rebuilt archive has a different hash even with
   identical contents. Rebuild → regenerate the signature, always in that
   order.
2. **Committing the dataset.** Also a zero. Verify with
   `git ls-files` — this repo tracks **23 files, none of them images or
   archives** (5 programs, 4 modules, 11 docs, 3 project files).
   `.gitignore` blocks `*.JPG`, `*.zip`,
   `augmented_directory/`, `leaves/`.
3. **Editing `signature.txt` by hand.** Regenerate it; never type it.

Current state: `signature.txt` =
`ee2c0426cb2f14ebc001554cb593813efe191e3c`, and `dataset.zip` (393 MB,
7231 entries) hashes to exactly that. The archive holds
`leaves/images/**` plus `learnings.zip`.

---

## 9. Questions you should expect

**"Why not PlantCV, which the subject recommends?"**
The pipeline is implemented in OpenCV/NumPy — the same operations PlantCV
wraps (saturation threshold, Otsu, morphology, contour analysis,
pseudolandmarks). Fewer dependencies, and every step is visible in the
file rather than behind a library call we'd have to explain anyway.

**"Show me a transformation on an image that isn't a leaf."**
Run it on anything. `leaf_outline` returns `contour=None`, the renderers
pass the image through, nothing crashes.

**"What does the mask do on a heavily diseased leaf?"**
Still works — saturation separates *coloured* from *grey*, and brown
lesions are coloured. It's specifically the green-channel approach that
would fail there.

**"Why is `original` in the dict if you never save it?"**
Display mode shows it as the first panel for comparison. Batch mode only
writes the five derived renderings — saving a copy of the input under a
new name would be pointless.

**"What if `-src` and `-dst` and a single image are all given?"**
`-src`/`-dst` wins (checked first). Give it no arguments at all and it
prints the help and exits 1.

---

## 10. Live-demo cheatsheet

```bash
# help
.venv/bin/python Transformation.py -h

# display mode — all six panels + histogram
.venv/bin/python Transformation.py "leaves/images/Apple_healthy/image (1).JPG"

# batch: everything
.venv/bin/python Transformation.py -src leaves/images/Apple_rust -dst /tmp/tr_all
ls /tmp/tr_all | head

# batch: just the mask (the subject's example command)
.venv/bin/python Transformation.py -src leaves/images/Apple_rust -dst /tmp/tr_mask -mask

# combine flags
.venv/bin/python Transformation.py -src leaves/images/Apple_rust -dst /tmp/tr_two -mask -histogram

# error handling — exit 1, no traceback
.venv/bin/python Transformation.py /nope.JPG
.venv/bin/python Transformation.py -src /nope -dst /tmp/x

# release check
sha1sum dataset.zip; cat signature.txt
git ls-files | wc -l          # 19, no images
```

See [`EVALUATION.md`](EVALUATION.md) for the full defense script.
