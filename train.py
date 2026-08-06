#!/usr/bin/env python3
"""
train.py <dir> -- Part 4 of Leaffliction: train the disease classifier.

Pipeline
--------
1. Split the raw data set, stratified and seeded (utils.dataset).
2. Balance *the training split only*, augmenting minority classes up to
   the largest class, into augmented_directory/.
3. Fine-tune an ImageNet-pretrained ResNet-18 on that balanced set.
4. Score it on the untouched validation split.
5. Pack model.pt + labels.json + metrics.json + augmented_directory/
   into learnings.zip.

Splitting *before* augmenting is the point of step 1. Balance the whole
data set first and split afterwards, and augmented copies of validation
images end up in the training set: accuracy then looks excellent and
means nothing.

The six augmentations and the balancing pass live in Augmentation.py
(Part 2) and are imported from there, so this program trains on exactly
the images that program produces.

Note: the team brief names model.keras, but this machine runs Python
3.14, for which TensorFlow publishes no wheel. The artifact is a PyTorch
model.pt instead; the rest of learnings.zip is unchanged.
"""

import argparse
import json
import os
import random
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from Augmentation import SEED, build_augmented_dir
from utils.dataset import load_image, split_dataset
from utils.preprocess import IMAGE_SIZE, PREPROCESS_NAME, preprocess


# --------------------------------------------------------------------------
# Data / model
# --------------------------------------------------------------------------
class LeafDataset(Dataset):
    """Items are (path, label); labels are mapped through ``classes``."""

    def __init__(self, items, classes):
        self.items = items
        self.index_of = {label: index for index, label in enumerate(classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        return (torch.from_numpy(preprocess(load_image(path))),
                self.index_of[label])


def build_model(num_classes):
    """ImageNet-pretrained ResNet-18 with a fresh classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, num_classes, device):
    """Return (accuracy, images scored, confusion matrix)."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for inputs, targets in loader:
            guesses = model(inputs.to(device)).argmax(dim=1).cpu()
            for truth, guess in zip(targets.tolist(), guesses.tolist()):
                confusion[truth][guess] += 1

    total = int(confusion.sum())
    hits = int(np.trace(confusion))
    return (hits / total if total else 0.0), total, confusion


def run_epoch(model, loader, criterion, optimiser, device, epoch):
    """One pass over the training set; returns the mean loss."""
    model.train()
    total_loss = seen = 0.0
    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimiser.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimiser.step()

        total_loss += float(loss.detach()) * targets.numel()
        seen += targets.numel()
        if step % 20 == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} "
                  f"loss {total_loss / seen:.4f}", flush=True)
    return total_loss / seen


def train_model(model, train_loader, val_loader, classes, device, epochs):
    """Fine-tune, and keep the weights from the best validation epoch."""
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=epochs)
    best = {"accuracy": -1.0}

    for epoch in range(1, epochs + 1):
        started = time.time()
        loss = run_epoch(model, train_loader, criterion, optimiser,
                         device, epoch)
        scheduler.step()

        accuracy, count, confusion = evaluate(model, val_loader,
                                              len(classes), device)
        print(f"epoch {epoch}/{epochs}  train_loss {loss:.4f}  "
              f"val_accuracy {accuracy:.4f}  ({time.time() - started:.0f}s)",
              flush=True)

        if accuracy > best["accuracy"]:
            best = {
                "accuracy": accuracy,
                "state": {name: weights.clone() for name, weights
                          in model.state_dict().items()},
                "confusion": confusion,
                "count": count,
                "epoch": epoch,
            }

    # The last epoch is not always the best one; restore the one that was.
    model.load_state_dict(best["state"])
    return best


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------
def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def per_class_recall(classes, confusion):
    """{label: {support, correct, recall}} read off the confusion rows."""
    scores = {}
    for index, label in enumerate(classes):
        support = int(confusion[index].sum())
        correct = int(confusion[index][index])
        scores[label] = {
            "support": support,
            "correct": correct,
            "recall": round(correct / support, 4) if support else 0.0,
        }
    return scores


def write_artifacts(model, classes, best, work_dir, aug_dir, zip_path):
    """Write model/labels/metrics and zip them with augmented_directory."""
    os.makedirs(work_dir, exist_ok=True)
    model_path = os.path.join(work_dir, "model.pt")
    labels_path = os.path.join(work_dir, "labels.json")
    metrics_path = os.path.join(work_dir, "metrics.json")

    torch.save({"state_dict": model.state_dict(),
                "architecture": "resnet18",
                "num_classes": len(classes)}, model_path)

    _write_json(labels_path, {"classes": list(classes),
                              "input_size": list(IMAGE_SIZE),
                              "preprocess": PREPROCESS_NAME})

    confusion = best["confusion"]
    _write_json(metrics_path, {
        "val_accuracy": round(best["accuracy"], 4),
        "val_count": int(best["count"]),
        "best_epoch": best["epoch"],
        "classes": list(classes),
        "per_class": per_class_recall(classes, confusion),
        "confusion": confusion.tolist(),
    })

    # The augmented images keep their augmented_directory/ prefix inside
    # the archive, which is what the brief asks the zip to contain.
    parent = os.path.dirname(os.path.abspath(aug_dir)) or "."
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (model_path, labels_path, metrics_path):
            archive.write(path, os.path.basename(path))
        for folder, _subfolders, filenames in os.walk(aug_dir):
            for filename in filenames:
                full = os.path.join(folder, filename)
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


def seed_everything(seed):
    """One seed for every generator this program leans on."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = build_parser().parse_args()

    if not os.path.isdir(args.directory):
        print(f"error: not a directory: {args.directory}", file=sys.stderr)
        return 1

    seed_everything(SEED)

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
                              num_workers=args.workers)
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

    print(f"\nbest validation accuracy: {best['accuracy']:.4f} "
          f"on {best['count']} held-out images (epoch {best['epoch']})")
    print(f"wrote {args.out}")
    if best["accuracy"] < 0.90:
        print("warning: below the 90% requirement", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
