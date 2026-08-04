#!/usr/bin/env python3
"""
train.py <dir> -- Part 4 of Leaffliction: train the disease classifier.

Pipeline
--------
1. Split the raw dataset stratified and seeded (utils.dataset).
2. Balance *the training split only* by augmenting minority classes up
   to the largest class, writing the result to augmented_directory/.
3. Fine-tune an ImageNet-pretrained ResNet-18 on that balanced set.
4. Evaluate on the untouched validation split.
5. Pack model.pt + labels.json + metrics.json + augmented_directory/
   into learnings.zip.

The split happens *before* augmentation on purpose. Balancing the whole
dataset first and splitting afterwards would put augmented copies of
validation images into the training set: accuracy would look excellent
and would mean nothing.

Note: the team brief specifies model.keras, but this machine only has
Python 3.14, for which TensorFlow publishes no wheel. The artifact is a
PyTorch model.pt instead; the rest of learnings.zip is unchanged.
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from utils.dataset import load_image, save_image, split_dataset
from utils.preprocess import IMAGE_SIZE, PREPROCESS_NAME, preprocess

SEED = 42
AUG_NAMES = ("Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion")


# --------------------------------------------------------------------------
# Augmentation -- the six variants required by the subject.
# Each takes and returns RGB uint8 (H, W, 3).
# --------------------------------------------------------------------------
def _flip(img, rng):
    return cv2.flip(img, 1)


def _rotate(img, rng):
    height, width = img.shape[:2]
    angle = rng.uniform(-30.0, 30.0)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (width, height),
                          borderMode=cv2.BORDER_REFLECT_101)


def _skew(img, rng):
    height, width = img.shape[:2]
    shift = rng.uniform(0.08, 0.20) * width
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = np.float32([[shift, 0], [width - shift, 0],
                      [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height),
                               borderMode=cv2.BORDER_REFLECT_101)


def _shear(img, rng):
    height, width = img.shape[:2]
    factor = rng.uniform(-0.25, 0.25)
    matrix = np.float32([[1, factor, -factor * height / 2], [0, 1, 0]])
    return cv2.warpAffine(img, matrix, (width, height),
                          borderMode=cv2.BORDER_REFLECT_101)


def _crop(img, rng):
    height, width = img.shape[:2]
    keep = rng.uniform(0.70, 0.88)
    new_h, new_w = int(height * keep), int(width * keep)
    top = rng.randint(0, height - new_h)
    left = rng.randint(0, width - new_w)
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (width, height),
                      interpolation=cv2.INTER_LINEAR)


def _distortion(img, rng):
    """Barrel / pincushion lens distortion."""
    height, width = img.shape[:2]
    strength = rng.uniform(-0.35, 0.35)
    ys, xs = np.indices((height, width), dtype=np.float32)
    norm_x = (xs - width / 2) / (width / 2)
    norm_y = (ys - height / 2) / (height / 2)
    radius = norm_x ** 2 + norm_y ** 2
    scale = 1.0 + strength * radius
    map_x = (norm_x * scale) * (width / 2) + width / 2
    map_y = (norm_y * scale) * (height / 2) + height / 2
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


AUGMENTATIONS = {
    "Flip": _flip,
    "Rotate": _rotate,
    "Skew": _skew,
    "Shear": _shear,
    "Crop": _crop,
    "Distortion": _distortion,
}


def augment_image(img_rgb, rng=None):
    """Return {name: variant} for all six augmentation types."""
    rng = rng or random.Random(SEED)
    return {name: fn(img_rgb, rng) for name, fn in AUGMENTATIONS.items()}


# --------------------------------------------------------------------------
# Building the balanced training directory
# --------------------------------------------------------------------------
def build_augmented_dir(train_items, out_dir, seed=SEED):
    """
    Copy every training image into ``out_dir/<label>/`` and augment the
    minority classes until each class matches the largest one.

    Only images from the training split ever reach this directory.
    Returns [(path, label), ...] for the balanced set.
    """
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    by_label = {}
    for path, label in train_items:
        by_label.setdefault(label, []).append(path)

    target = max(len(paths) for paths in by_label.values())
    print(f"balancing {len(by_label)} classes up to {target} images each")

    balanced = []
    rng = random.Random(seed)
    for label in sorted(by_label):
        paths = sorted(by_label[label])
        class_dir = os.path.join(out_dir, label)
        os.makedirs(class_dir, exist_ok=True)

        for path in paths:
            dest = os.path.join(class_dir, os.path.basename(path))
            shutil.copyfile(path, dest)
            balanced.append((dest, label))

        needed = target - len(paths)
        # (source, augmentation) pairs repeat every lcm(n_paths, 6)
        # steps, not every n_paths * 6 steps. Numbering the rounds by
        # the wrong period lets a later file overwrite an earlier one,
        # which silently under-fills exactly the rarest classes.
        period = math.lcm(len(paths), len(AUG_NAMES))
        for index in range(needed):
            source = paths[index % len(paths)]
            name = AUG_NAMES[index % len(AUG_NAMES)]
            round_id = index // period
            base = os.path.splitext(os.path.basename(source))[0]
            suffix = f"_{name}.JPG" if round_id == 0 \
                else f"_{name}{round_id}.JPG"
            dest = os.path.join(class_dir, base + suffix)
            variant = AUGMENTATIONS[name](load_image(source), rng)
            save_image(variant, dest)
            balanced.append((dest, label))

        written = len(os.listdir(class_dir))
        if written != target:
            raise RuntimeError(
                f"{label}: wrote {written} images but expected {target}; "
                "augmented filenames are colliding")
        print(f"  {label}: {len(paths)} -> {written}")

    return balanced


# --------------------------------------------------------------------------
# Data / model
# --------------------------------------------------------------------------
class LeafDataset(Dataset):
    """Items are (path, label); labels are mapped through ``classes``."""

    def __init__(self, items, classes):
        self.items = items
        self.index_of = {label: i for i, label in enumerate(classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        tensor = torch.from_numpy(preprocess(load_image(path)))
        return tensor, self.index_of[label]


def build_model(num_classes):
    """ImageNet-pretrained ResNet-18 with a fresh classification head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, num_classes, device):
    """Return (accuracy, total, confusion_matrix)."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            predicted = model(inputs).argmax(dim=1).cpu()
            for true, pred in zip(targets.tolist(), predicted.tolist()):
                confusion[true][pred] += 1
            correct += int((predicted == targets).sum())
            total += targets.numel()
    accuracy = correct / total if total else 0.0
    return accuracy, total, confusion


def train_model(model, train_loader, val_loader, classes, device, epochs):
    """Fine-tune and keep the weights from the best validation epoch."""
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=epochs)
    best = {"accuracy": -1.0, "state": None, "confusion": None,
            "count": 0, "epoch": 0}

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        running = seen = 0.0
        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * targets.numel()
            seen += targets.numel()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / seen:.4f}", flush=True)
        scheduler.step()

        accuracy, count, confusion = evaluate(model, val_loader,
                                              len(classes), device)
        print(f"epoch {epoch}/{epochs}  train_loss {running / seen:.4f}  "
              f"val_accuracy {accuracy:.4f}  ({time.time() - started:.0f}s)",
              flush=True)

        if accuracy > best["accuracy"]:
            best = {
                "accuracy": accuracy,
                "state": {k: v.clone() for k, v in
                          model.state_dict().items()},
                "confusion": confusion,
                "count": count,
                "epoch": epoch,
            }

    model.load_state_dict(best["state"])
    return best


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------
def write_artifacts(model, classes, best, work_dir, aug_dir, zip_path):
    """Write model/labels/metrics and zip them with augmented_directory."""
    os.makedirs(work_dir, exist_ok=True)
    model_path = os.path.join(work_dir, "model.pt")
    labels_path = os.path.join(work_dir, "labels.json")
    metrics_path = os.path.join(work_dir, "metrics.json")

    torch.save({"state_dict": model.state_dict(),
                "architecture": "resnet18",
                "num_classes": len(classes)}, model_path)

    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump({"classes": list(classes),
                   "input_size": list(IMAGE_SIZE),
                   "preprocess": PREPROCESS_NAME}, handle, indent=2)

    confusion = best["confusion"]
    per_class = {}
    for index, label in enumerate(classes):
        support = int(confusion[index].sum())
        hits = int(confusion[index][index])
        per_class[label] = {
            "support": support,
            "correct": hits,
            "recall": round(hits / support, 4) if support else 0.0,
        }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"val_accuracy": round(best["accuracy"], 4),
                   "val_count": int(best["count"]),
                   "best_epoch": best["epoch"],
                   "classes": list(classes),
                   "per_class": per_class,
                   "confusion": confusion.tolist()}, handle, indent=2)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (model_path, labels_path, metrics_path):
            archive.write(path, os.path.basename(path))
        parent = os.path.dirname(os.path.abspath(aug_dir)) or "."
        for dirpath, _dirnames, filenames in os.walk(aug_dir):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                archive.write(full, os.path.relpath(full, parent))
    return model_path, labels_path, metrics_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the Leaffliction leaf-disease classifier.")
    parser.add_argument("directory",
                        help="dataset root: <dir>/<class>/*.JPG")
    parser.add_argument("--epochs", type=int, default=6,
                        help="training epochs (default: 6)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="mini-batch size (default: 64)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="validation fraction (default: 0.2)")
    parser.add_argument("--augmented-dir", default="augmented_directory",
                        help="where balanced training images are written")
    parser.add_argument("--out", default="learnings.zip",
                        help="output archive (default: learnings.zip)")
    parser.add_argument("--workers", type=int, default=4,
                        help="data-loading worker processes (default: 4)")
    return parser


def main():
    args = build_parser().parse_args()

    if not os.path.isdir(args.directory):
        print(f"error: not a directory: {args.directory}", file=sys.stderr)
        return 1

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    try:
        train_items, val_items = split_dataset(args.directory,
                                               val_ratio=args.val_ratio,
                                               seed=SEED)
    except (NotADirectoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not train_items or not val_items:
        print("error: no images found; expected <dir>/<class>/*.JPG",
              file=sys.stderr)
        return 1

    classes = sorted({label for _path, label in train_items})
    print(f"{len(train_items)} training images, {len(val_items)} held out, "
          f"{len(classes)} classes")
    if len(val_items) < 100:
        print(f"warning: only {len(val_items)} validation images; the "
              "subject requires at least 100", file=sys.stderr)

    balanced = build_augmented_dir(train_items, args.augmented_dir)

    train_loader = DataLoader(LeafDataset(balanced, classes),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(LeafDataset(val_items, classes),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    print(f"training resnet18 on {device} for {args.epochs} epochs "
          f"over {len(balanced)} images", flush=True)

    best = train_model(model, train_loader, val_loader, classes, device,
                       args.epochs)

    write_artifacts(model, classes, best, "learnings",
                    args.augmented_dir, args.out)

    accuracy = best["accuracy"]
    print(f"\nbest validation accuracy: {accuracy:.4f} "
          f"on {best['count']} held-out images (epoch {best['epoch']})")
    print(f"wrote {args.out}")
    if accuracy < 0.90:
        print("warning: below the 90% requirement", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
