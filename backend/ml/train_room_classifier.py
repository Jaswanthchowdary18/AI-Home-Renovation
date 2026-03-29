"""
ARKEN — Room Classifier Fine-Tuning Script v3.0
=================================================
Production EfficientNet-B0 fine-tuning for room type classification.

v3.0 Changes over v2.0:
  - Scans BOTH dataset folders using os.walk:
      * interior_design_material_style/{room_type}/{style}/*.jpg
      * interior_design_images_metadata/{room_type}/{style}/*.jpg
    Room type inferred from folder depth-3 (parent of parent of image).
  - Data augmentation:
      Train: RandomHorizontalFlip, RandomRotation(15), ColorJitter(0.2,0.2,0.2),
             RandomResizedCrop(224, scale=(0.8,1.0))
      Val:   Resize(256) + CenterCrop(224) only
  - Class imbalance: CrossEntropyLoss with per-class inverse-frequency weights.
    Also uses WeightedRandomSampler so minority rooms appear equally in each batch.
  - AdamW optimizer, CosineAnnealingLR scheduler.
  - Per-epoch logging: train_loss, val_loss, val_accuracy, per_class_f1.
  - Saves best checkpoint (val_accuracy) to ml/weights/room_classifier.pt.
  - Saves training_report.json: best_val_accuracy, epochs_run, class_distribution,
    per_class_f1, dataset_size, label_to_idx.
  - Pre-flight: exits with clear error if < 50 valid images found.
  - argparse: --epochs (default 20), --batch-size (default 32),
              --weights-dir, --lr (default 1e-4).

Compatible with CVModelRegistry.EfficientNetRoomClassifier:
  - Saves state dict (not full model) to room_classifier.pt.
  - label_to_idx saved in training_report.json for inference-time label lookup.

Usage:
    cd backend
    python ml/train_room_classifier.py
    python ml/train_room_classifier.py --epochs 30 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arken.train_room")

# ── Add backend to sys.path ──────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

# ── Default paths ────────────────────────────────────────────────────────────
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))
_WEIGHTS_DIR = Path(os.getenv("ML_WEIGHTS_DIR", "/app/ml/weights"))

# For local dev fallback
if not _WEIGHTS_DIR.exists():
    _WEIGHTS_DIR = _BACKEND_DIR / "ml" / "weights"
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset roots to scan ────────────────────────────────────────────────────
_DATASET_ROOTS: List[Path] = []
for _root_candidate in [
    _APP_DIR  / "data" / "datasets" / "interior_design_material_style",
    _BACKEND_DIR / "data" / "datasets" / "interior_design_material_style",
    _APP_DIR  / "data" / "datasets" / "interior_design_images_metadata",
    _BACKEND_DIR / "data" / "datasets" / "interior_design_images_metadata",
]:
    if _root_candidate.exists():
        _DATASET_ROOTS.append(_root_candidate)

# ── Room labels supported by CVModelRegistry ─────────────────────────────────
ROOM_LABELS: List[str] = [
    "bedroom", "living_room", "kitchen", "bathroom",
    "dining_room", "study", "balcony", "hallway", "other",
]

# Known room-type folder name variants -> canonical label
_ROOM_FOLDER_ALIASES: Dict[str, str] = {
    "bedroom":     "bedroom",
    "bedrooms":    "bedroom",
    "living_room": "living_room",
    "livingroom":  "living_room",
    "lounge":      "living_room",
    "kitchen":     "kitchen",
    "kitchens":    "kitchen",
    "bathroom":    "bathroom",
    "bathrooms":   "bathroom",
    "dining_room": "dining_room",
    "dining":      "dining_room",
    "study":       "study",
    "office":      "study",
    "home_office": "study",
    "balcony":     "balcony",
    "terrace":     "balcony",
    "hallway":     "hallway",
    "corridor":    "hallway",
}

# ── Training hyperparameters ─────────────────────────────────────────────────
IMG_SIZE    = 224
TRAIN_SPLIT = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# Image discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_images(extra_roots: Optional[List[Path]] = None) -> List[Tuple[Path, str]]:
    """
    Walk all dataset roots and return (image_path, room_type) pairs.

    Folder structure expected:
        <dataset_root>/<room_type>/<style>/<image>.jpg
    Room type is inferred from the folder at depth -3 relative to the image.
    Unknown room types are mapped to "other".

    Returns:
        List of (Path, room_label) where room_label is in ROOM_LABELS.
    """
    roots = list(_DATASET_ROOTS)
    if extra_roots:
        roots.extend(extra_roots)

    records: List[Tuple[Path, str]] = []
    seen_paths: set = set()
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for root in roots:
        if not root.exists():
            logger.debug(f"[find_images] Root not found, skipping: {root}")
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden dirs and __pycache__
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]

            dir_path = Path(dirpath)
            parts    = dir_path.parts

            # We need at least 2 levels below root: <room>/<style>/
            # Infer room type from parent of style folder (depth = root_depth + 1)
            root_depth = len(root.parts)
            if len(parts) < root_depth + 2:
                # We're at root level or one below — no room type inferable here
                continue

            # Room folder is parts[root_depth], style folder is parts[root_depth + 1]
            room_folder_name = parts[root_depth].lower()
            room_label = _ROOM_FOLDER_ALIASES.get(room_folder_name, "other")
            if room_label not in ROOM_LABELS:
                room_label = "other"

            for fname in filenames:
                if Path(fname).suffix.lower() not in image_exts:
                    continue
                img_path = dir_path / fname
                # Deduplicate across dataset roots
                key = img_path.resolve()
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                records.append((img_path, room_label))

    return records


def dataset_health_report(records: List[Tuple[Path, str]]) -> Dict:
    """Print a summary of found images and their distribution."""
    counter = Counter(label for _, label in records)
    report  = {
        "total_images": len(records),
        "class_distribution": dict(counter),
        "classes_found": sorted(counter.keys()),
        "min_class_count": min(counter.values()) if counter else 0,
        "max_class_count": max(counter.values()) if counter else 0,
    }
    logger.info(
        f"[DatasetHealth] {report['total_images']} images across "
        f"{len(counter)} classes: {dict(counter)}"
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    records: List[Tuple[Path, str]],
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weights_dir: Path = _WEIGHTS_DIR,
) -> Dict:
    """
    Fine-tune EfficientNet-B0 on room type classification.

    Args:
        records:     List of (image_path, room_type) from find_images().
        epochs:      Number of training epochs.
        batch_size:  Mini-batch size.
        lr:          AdamW learning rate.
        weights_dir: Directory to save room_classifier.pt and training_report.json.

    Returns:
        training_report dict (also saved to weights_dir/training_report.json).
    """
    # ── Dependency check ─────────────────────────────────────────────────────
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        import torchvision.models as models
        import torchvision.transforms as T
        from PIL import Image
    except ImportError as e:
        logger.error(
            f"[train] Missing dependency: {e}. "
            "Run: pip install torch torchvision Pillow"
        )
        sys.exit(1)

    try:
        from sklearn.metrics import f1_score
        _HAS_SKLEARN = True
    except ImportError:
        _HAS_SKLEARN = False
        logger.warning("[train] scikit-learn not found — per-class F1 will not be logged")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[train] Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    # ── Label encoding ────────────────────────────────────────────────────────
    # Only use classes that actually appear in the dataset
    present_labels = sorted(set(label for _, label in records))
    label_to_idx   = {label: i for i, label in enumerate(present_labels)}
    idx_to_label   = {i: label for label, i in label_to_idx.items()}
    num_classes    = len(present_labels)
    logger.info(f"[train] Classes ({num_classes}): {present_labels}")

    # ── Dataset splits ────────────────────────────────────────────────────────
    # Stratified split: maintain class proportions in train/val
    from collections import defaultdict
    class_records: Dict[str, List] = defaultdict(list)
    for item in records:
        class_records[item[1]].append(item)

    train_records: List[Tuple[Path, str]] = []
    val_records:   List[Tuple[Path, str]] = []

    for cls_label, cls_recs in class_records.items():
        n_train = max(1, int(len(cls_recs) * TRAIN_SPLIT))
        train_records.extend(cls_recs[:n_train])
        val_records.extend(cls_recs[n_train:])

    logger.info(f"[train] Split: {len(train_records)} train | {len(val_records)} val")

    # ── PyTorch Dataset ───────────────────────────────────────────────────────
    train_transform = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class RoomDataset(Dataset):
        """
        PyTorch Dataset for room type images.
        Returns (tensor, label_idx) pairs.
        Corrupted images return a blank tensor rather than raising.
        """

        def __init__(self, records: List[Tuple[Path, str]], transform: T.Compose) -> None:
            self.records   = records
            self.transform = transform

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> Tuple:
            img_path, label = self.records[idx]
            label_idx = label_to_idx[label]
            try:
                img = Image.open(img_path).convert("RGB")
                return self.transform(img), label_idx
            except Exception as e:
                logger.debug(f"[RoomDataset] Corrupt image {img_path.name}: {e}")
                blank = torch.zeros(3, IMG_SIZE, IMG_SIZE)
                return blank, label_idx

    train_dataset = RoomDataset(train_records, train_transform)
    val_dataset   = RoomDataset(val_records,   val_transform)

    # ── Class-weighted sampler (fixes class imbalance) ────────────────────────
    train_labels_list = [label_to_idx[r[1]] for r in train_records]
    class_counts      = Counter(train_labels_list)
    class_weights     = {cls: 1.0 / max(cnt, 1) for cls, cnt in class_counts.items()}
    sample_weights    = [class_weights[lbl] for lbl in train_labels_list]

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_records),
        replacement = True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = 0,   # 0 = main process (avoids multiprocessing issues on all OS)
        pin_memory  = (device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace the classifier head for our room count
    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    logger.info(
        f"[train] EfficientNet-B0 loaded. "
        f"Replaced head: Linear({in_features} → {num_classes})"
    )

    # ── Loss with class weights ───────────────────────────────────────────────
    # Inverse-frequency class weights for CrossEntropyLoss
    total_samples = len(train_records)
    loss_weights  = torch.zeros(num_classes, device=device)
    for cls_idx, cls_label in idx_to_label.items():
        count = class_counts.get(cls_idx, 1)
        loss_weights[cls_idx] = total_samples / (num_classes * count)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # ── Optimiser and scheduler ───────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_epoch     = 0
    epoch_log      = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs   = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss    = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_imgs.size(0)

        train_loss /= max(len(train_dataset), 1)
        scheduler.step()

        # Validation
        model.eval()
        val_loss    = 0.0
        all_preds:  List[int] = []
        all_truths: List[int] = []

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs   = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                outputs      = model(batch_imgs)
                loss         = criterion(outputs, batch_labels)
                val_loss    += loss.item() * batch_imgs.size(0)
                preds        = outputs.argmax(dim=1).cpu().tolist()
                truths       = batch_labels.cpu().tolist()
                all_preds.extend(preds)
                all_truths.extend(truths)

        val_loss /= max(len(val_dataset), 1)
        n_correct = sum(p == t for p, t in zip(all_preds, all_truths))
        val_acc   = n_correct / max(len(all_preds), 1)

        # Per-class F1
        if _HAS_SKLEARN and all_preds:
            labels_range = list(range(num_classes))
            f1_scores    = f1_score(
                all_truths, all_preds,
                labels=labels_range, average=None, zero_division=0,
            )
            per_class_f1 = {
                idx_to_label[i]: round(float(f1_scores[i]), 4)
                for i in range(num_classes)
            }
            macro_f1 = round(float(f1_scores.mean()), 4)
        else:
            per_class_f1 = {}
            macro_f1     = round(val_acc, 4)

        epoch_entry = {
            "epoch":         epoch,
            "train_loss":    round(train_loss, 5),
            "val_loss":      round(val_loss,   5),
            "val_accuracy":  round(val_acc,    4),
            "macro_f1":      macro_f1,
            "per_class_f1":  per_class_f1,
        }
        epoch_log.append(epoch_entry)

        logger.info(
            f"  Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | "
            f"macro_f1={macro_f1:.3f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            pt_path      = weights_dir / "room_classifier.pt"
            torch.save(model.state_dict(), pt_path)
            logger.info(f"  [BEST] Saved checkpoint to {pt_path} (val_acc={val_acc:.4f})")

    logger.info(
        f"[train] Finished. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}."
    )

    # ── Save training report ──────────────────────────────────────────────────
    class_distribution = {
        label: sum(1 for _, l in records if l == label)
        for label in present_labels
    }
    report = {
        "best_val_accuracy":  round(best_val_acc, 4),
        "best_epoch":         best_epoch,
        "epochs_run":         epochs,
        "dataset_size":       len(records),
        "train_size":         len(train_records),
        "val_size":           len(val_records),
        "class_distribution": class_distribution,
        "per_class_f1":       epoch_log[-1]["per_class_f1"] if epoch_log else {},
        "macro_f1":           epoch_log[-1]["macro_f1"]     if epoch_log else 0.0,
        "label_to_idx":       label_to_idx,
        "num_classes":        num_classes,
        "model_architecture": "efficientnet_b0",
        "img_size":           IMG_SIZE,
        "training_date":      datetime.now(tz=timezone.utc).isoformat(),
        "device":             device,
        "epoch_log":          epoch_log,
    }

    report_path = weights_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"[train] Training report saved to {report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Command-line interface for room classifier training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B0 on ARKEN interior room images."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32). Reduce to 16 if OOM.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(_WEIGHTS_DIR),
        help=f"Directory to save model weights (default: {_WEIGHTS_DIR}).",
    )
    parser.add_argument(
        "--extra-root",
        type=str,
        default=None,
        help="Additional image root directory to scan (optional).",
    )
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    extra_roots = [Path(args.extra_root)] if args.extra_root else None

    # ── Discover images ───────────────────────────────────────────────────────
    logger.info("[main] Scanning dataset folders for room images …")
    records = find_images(extra_roots)

    if len(records) < 50:
        logger.error(
            f"[main] Pre-flight failed: only {len(records)} valid images found "
            f"(minimum 50 required). "
            f"Ensure images are present at:\n"
            + "\n".join(f"  {r}" for r in _DATASET_ROOTS)
        )
        sys.exit(1)

    # ── Health report ─────────────────────────────────────────────────────────
    health = dataset_health_report(records)
    logger.info(
        f"[main] Dataset ready: {health['total_images']} images | "
        f"{len(health['classes_found'])} classes | "
        f"min_class={health['min_class_count']}"
    )

    # ── Warn on severe imbalance ───────────────────────────────────────────────
    if health["max_class_count"] > health["min_class_count"] * 5:
        logger.warning(
            "[main] Severe class imbalance detected "
            f"(max={health['max_class_count']}, min={health['min_class_count']}). "
            "WeightedRandomSampler and weighted loss will compensate."
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    report = train(
        records     = records,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        weights_dir = weights_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ARKEN Room Classifier — Training Complete")
    print("=" * 60)
    print(f"  Best val accuracy : {report['best_val_accuracy']:.4f}  (epoch {report['best_epoch']})")
    print(f"  Macro F1          : {report['macro_f1']:.4f}")
    print(f"  Dataset size      : {report['dataset_size']} images")
    print(f"  Classes           : {list(report['label_to_idx'].keys())}")
    print(f"  Weights saved to  : {weights_dir / 'room_classifier.pt'}")
    print(f"  Report saved to   : {weights_dir / 'training_report.json'}")
    print("=" * 60)

    if report["best_val_accuracy"] < 0.70:
        print(
            "\n[WARNING] Val accuracy < 70%. Consider:\n"
            "  1. Adding more images per class (aim for 200+ per class)\n"
            "  2. Increasing epochs to 30+\n"
            "  3. Reducing batch size to 16 for better gradient signal\n"
        )
    elif report["best_val_accuracy"] >= 0.88:
        print(
            f"\n[SUCCESS] Excellent accuracy {report['best_val_accuracy']:.1%}. "
            "Model is production-ready."
        )
    else:
        print(
            f"\n[GOOD] Accuracy {report['best_val_accuracy']:.1%}. "
            "Reasonable for this dataset size. More images will push it higher."
        )


if __name__ == "__main__":
    main()