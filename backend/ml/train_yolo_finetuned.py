"""
ARKEN — YOLOv8 Fine-Tuning Pipeline v1.0
==========================================
Auto-annotates your real interior design images with pseudo-labels using
pretrained YOLOv8x, then fine-tunes YOLOv8n-seg for Indian room object detection.

Pipeline:
  Stage 1 — Auto-annotation (pseudo-labelling):
    • Loads pretrained YOLOv8x (highest accuracy for annotation quality).
    • Runs on ALL images in interior_design_images_metadata/{room}/{style}/*.jpg
      and interior_design_material_style/{room}/{style}/*.jpg.
    • Filters detections: confidence > 0.60, only INTERIOR_CLASS_IDS.
    • Saves YOLO-format .txt label files alongside images.
    • Skips images where < 1 high-confidence object detected (no label file).

  Stage 2 — Fine-tuning:
    • Generates data.yaml referencing the annotated image/label pairs.
    • Fine-tunes YOLOv8n-seg for 50 epochs (YOLOv8n = fast, suitable for CPU/GPU).
    • Saves best weights to ml/weights/yolo_indian_rooms.pt.
    • Saves training_report.json with mAP50, mAP50-95, epochs_run.

  Why YOLOv8x for annotation + YOLOv8n for training:
    • YOLOv8x has highest detection quality — better pseudo-labels.
    • YOLOv8n is lightweight for inference (6MB, fast on CPU) — correct
      for production use in CVModelRegistry.

Interior classes detected (COCO subset relevant to Indian apartments):
  chair, sofa/couch, bed, dining table, toilet, sink, refrigerator,
  tv/monitor, laptop, oven, microwave, potted plant, clock, vase,
  door (class 0 in COCO = person is excluded), wardrobe (not in COCO —
  detected via large vertical rectangle heuristic).

Dataset paths (searched in order):
  backend/data/datasets/interior_design_images_metadata/{room}/{style}/*.jpg
  backend/data/datasets/interior_design_material_style/{room}/{style}/*.jpg

Labels saved to:
  backend/data/datasets/interior_design_images_metadata/labels/{room}/{style}/*.txt
  (mirrors image folder structure)

Usage:
    cd backend
    python ml/train_yolo_finetuned.py                    # full pipeline
    python ml/train_yolo_finetuned.py --stage annotate   # annotation only
    python ml/train_yolo_finetuned.py --stage train      # training only (needs labels)
    python ml/train_yolo_finetuned.py --epochs 30 --conf-threshold 0.55

Requirements:
    pip install ultralytics==8.3.2 PyYAML Pillow
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arken.train_yolo")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))
_WEIGHTS_DIR = Path(os.getenv("ML_WEIGHTS_DIR", str(_APP_DIR / "ml" / "weights")))
if not _WEIGHTS_DIR.parent.exists():
    _WEIGHTS_DIR = _BACKEND_DIR / "ml" / "weights"
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset roots ─────────────────────────────────────────────────────────────
_DATASET_ROOTS: List[Path] = []
for _prefix in [_APP_DIR, _BACKEND_DIR]:
    for _ds_name in ["interior_design_images_metadata", "interior_design_material_style"]:
        _p = _prefix / "data" / "datasets" / _ds_name
        if _p.exists():
            _DATASET_ROOTS.append(_p)

# Primary label output root — always under interior_design_images_metadata
_LABEL_ROOT: Optional[Path] = None
for _prefix in [_APP_DIR, _BACKEND_DIR]:
    _p = _prefix / "data" / "datasets" / "interior_design_images_metadata"
    if _p.exists():
        _LABEL_ROOT = _p / "labels"
        break
if _LABEL_ROOT is None:
    _LABEL_ROOT = _BACKEND_DIR / "data" / "datasets" / "interior_design_images_metadata" / "labels"
_LABEL_ROOT.mkdir(parents=True, exist_ok=True)

# ── Room / style structure ─────────────────────────────────────────────────────
ROOM_LABELS  = ["bathroom", "bedroom", "kitchen", "living_room"]
STYLE_LABELS = ["boho", "industrial", "minimalist", "modern", "scandinavian"]

# ── COCO class IDs relevant to Indian interior spaces ─────────────────────────
# Key: COCO class_id → display name
INTERIOR_CLASS_IDS: Dict[int, str] = {
    56: "chair",
    57: "sofa",
    58: "potted_plant",
    59: "bed",
    60: "dining_table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    65: "remote",
    69: "oven",
    70: "microwave",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
}

# YOLO training class list (0-indexed, matches data.yaml)
YOLO_CLASSES = sorted(INTERIOR_CLASS_IDS.keys())
YOLO_CLASS_NAMES = [INTERIOR_CLASS_IDS[cid] for cid in YOLO_CLASSES]
COCO_TO_YOLO_IDX = {coco_id: idx for idx, coco_id in enumerate(YOLO_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Image discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_images() -> List[Tuple[Path, str, str]]:
    """
    Scan dataset roots for all JPG/PNG images.
    Returns list of (image_path, room_type, style).
    """
    found: List[Tuple[Path, str, str]] = []
    seen_stems: set = set()

    for root in _DATASET_ROOTS:
        for room in ROOM_LABELS:
            room_dir = root / room
            if not room_dir.is_dir():
                continue
            for style in STYLE_LABELS:
                style_dir = room_dir / style
                if not style_dir.is_dir():
                    continue
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"]:
                    for img_path in sorted(style_dir.glob(ext)):
                        # Deduplicate by stem (same file in both dataset roots)
                        stem_key = f"{room}_{style}_{img_path.stem}"
                        if stem_key not in seen_stems:
                            seen_stems.add(stem_key)
                            found.append((img_path, room, style))

    logger.info(
        f"[discover_images] Found {len(found)} unique images across "
        f"{len(_DATASET_ROOTS)} dataset root(s)"
    )
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Auto-annotation with YOLOv8x
# ─────────────────────────────────────────────────────────────────────────────

def auto_annotate(
    images: List[Tuple[Path, str, str]],
    confidence_threshold: float = 0.40,  # Lowered from 0.60 — interior images
    batch_size: int = 8,                  # often have objects at 0.40-0.59 conf
) -> Tuple[List[Path], List[Path], int]:
    """
    Use pretrained YOLOv8x to generate pseudo-labels on all images.

    Args:
        images:               List of (image_path, room_type, style).
        confidence_threshold: Minimum confidence to accept a detection.
        batch_size:           Images per YOLOv8 batch.

    Returns:
        (annotated_image_paths, label_paths, total_detections)
    """
    logger.info("=" * 60)
    logger.info(
        f"[AutoAnnotate] Starting pseudo-labelling with YOLOv8x "
        f"(conf_threshold={confidence_threshold}) on {len(images)} images …"
    )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics required: pip install ultralytics==8.3.2"
        )

    # Load YOLOv8x — highest quality for pseudo-label generation
    annotator_weights = _WEIGHTS_DIR / "yolov8x.pt"
    # Use GPU if available — annotating 4139 images on CPU takes ~90 min,
    # on GPU it takes ~8 min
    _ann_device = 0 if _cuda_available() else "cpu"
    try:
        annotator = YOLO(
            str(annotator_weights) if annotator_weights.exists() else "yolov8x.pt"
        )
        annotator.to(_ann_device if isinstance(_ann_device, str) else f"cuda:{_ann_device}")
        logger.info(
            f"[AutoAnnotate] YOLOv8x loaded "
            f"({'cached' if annotator_weights.exists() else 'auto-downloading'}) "
            f"on {'GPU' if _ann_device == 0 else 'CPU'}."
        )
    except Exception as e:
        raise RuntimeError(f"[AutoAnnotate] Failed to load YOLOv8x: {e}")

    annotated_image_paths: List[Path] = []
    label_paths: List[Path] = []
    total_detections = 0
    skipped_no_objects = 0
    errors = 0

    # Process in batches
    all_img_paths = [ip for ip, _, _ in images]

    for batch_start in range(0, len(images), batch_size):
        batch = images[batch_start: batch_start + batch_size]
        batch_img_paths = [str(ip) for ip, _, _ in batch]

        try:
            results = annotator(
                batch_img_paths,
                verbose=False,
                conf=confidence_threshold,
                iou=0.45,
                device=_ann_device,
                stream=False,
            )
        except Exception as e:
            logger.warning(f"[AutoAnnotate] Batch {batch_start}–{batch_start+batch_size} failed: {e}")
            errors += len(batch)
            continue

        for i, (result, (img_path, room, style)) in enumerate(zip(results, batch)):
            try:
                img_w = result.orig_shape[1]
                img_h = result.orig_shape[0]

                yolo_lines: List[str] = []
                n_dets = 0

                for box in result.boxes:
                    coco_cls = int(box.cls[0])
                    if coco_cls not in INTERIOR_CLASS_IDS:
                        continue
                    conf = float(box.conf[0])
                    if conf < confidence_threshold:
                        continue

                    yolo_cls_idx = COCO_TO_YOLO_IDX[coco_cls]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Convert to YOLO normalised format: class cx cy w h
                    cx = (x1 + x2) / 2.0 / img_w
                    cy = (y1 + y2) / 2.0 / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h

                    # Clamp to [0,1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.001, min(1.0, bw))
                    bh = max(0.001, min(1.0, bh))

                    yolo_lines.append(
                        f"{yolo_cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                    )
                    n_dets += 1

                if n_dets == 0:
                    skipped_no_objects += 1
                    continue

                # Save label file BESIDE the image file.
                # YOLO's DataLoader resolves labels by replacing the image
                # extension with .txt in the SAME directory. Saving to a
                # separate _LABEL_ROOT causes "No labels found" / 0 labels.
                label_path = img_path.with_suffix(".txt")

                with open(label_path, "w") as fh:
                    fh.write("\n".join(yolo_lines))

                annotated_image_paths.append(img_path)
                label_paths.append(label_path)
                total_detections += n_dets

            except Exception as e:
                logger.debug(f"[AutoAnnotate] Image {img_path.name} failed: {e}")
                errors += 1

        if (batch_start // batch_size) % 5 == 0:
            logger.info(
                f"  Progress: {min(batch_start + batch_size, len(images))}/{len(images)} images  "
                f"annotated={len(annotated_image_paths)}  "
                f"detections={total_detections}  "
                f"skipped_no_obj={skipped_no_objects}"
            )

    logger.info(
        f"[AutoAnnotate] Complete. "
        f"annotated={len(annotated_image_paths)}  "
        f"total_detections={total_detections}  "
        f"skipped_no_objects={skipped_no_objects}  "
        f"errors={errors}"
    )

    if len(annotated_image_paths) < 20:
        logger.warning(
            f"[AutoAnnotate] Only {len(annotated_image_paths)} images annotated. "
            "Check that images exist at dataset roots and YOLOv8x can detect "
            "interior objects. Common issue: images too small or heavily cropped."
        )

    return annotated_image_paths, label_paths, total_detections


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: data.yaml generation
# ─────────────────────────────────────────────────────────────────────────────

def create_data_yaml(
    annotated_image_paths: List[Path],
    train_ratio: float = 0.85,
) -> Path:
    """
    Create YOLO-format data.yaml pointing to annotated train/val image lists.

    Returns path to data.yaml.
    """
    import random

    # Shuffle for reproducibility
    random.seed(42)
    shuffled = list(annotated_image_paths)
    random.shuffle(shuffled)

    split_idx  = int(len(shuffled) * train_ratio)
    train_imgs = shuffled[:split_idx]
    val_imgs   = shuffled[split_idx:]

    # Write image list files
    yaml_dir = _WEIGHTS_DIR / "yolo_dataset"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    train_list_path = yaml_dir / "train_images.txt"
    val_list_path   = yaml_dir / "val_images.txt"

    with open(train_list_path, "w") as fh:
        fh.write("\n".join(str(p) for p in train_imgs))
    with open(val_list_path, "w") as fh:
        fh.write("\n".join(str(p) for p in val_imgs))

    # data.yaml
    # NOTE: NO 'path:' key here.
    # Ultralytics prepends 'path' to 'train'/'val' even when they are already
    # absolute, producing a doubled path like:
    #   /app/datasets/ml/weights/.../ml/weights/.../val_images.txt
    # Fix: use fully resolved absolute paths for train/val and omit 'path:'.
    data_yaml_content = f"""# ARKEN Indian Interior Rooms — YOLO Dataset
# Auto-generated by train_yolo_finetuned.py
# {len(train_imgs)} train images, {len(val_imgs)} val images
# Classes: {len(YOLO_CLASSES)} interior object types

train: {str(train_list_path.resolve())}
val:   {str(val_list_path.resolve())}

nc: {len(YOLO_CLASSES)}
names:
{chr(10).join(f'  - {name}' for name in YOLO_CLASS_NAMES)}
"""

    yaml_path = yaml_dir / "data.yaml"
    with open(yaml_path, "w") as fh:
        fh.write(data_yaml_content)

    logger.info(
        f"[DataYAML] Created at {yaml_path}  "
        f"train={len(train_imgs)}  val={len(val_imgs)}  "
        f"classes={len(YOLO_CLASSES)}"
    )
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: YOLOv8n-seg fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def finetune_yolo(
    data_yaml: Path,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    n_train_images: int = 0,
) -> Dict[str, Any]:
    """
    Fine-tune YOLOv8n-detect on the auto-annotated Indian room images.

    Uses YOLOv8n (nano, detection) — NOT seg — because our auto-annotator
    writes bounding-box labels only (class cx cy w h).  yolov8n-seg requires
    polygon segment masks in every label file; feeding it bbox-only labels
    causes: "segment dataset incorrectly formatted".

    Saves best weights to ml/weights/yolo_indian_rooms.pt.
    Returns training report dict.
    """
    logger.info("=" * 60)
    logger.info(
        f"[YOLO-Finetune] Starting YOLOv8n-seg fine-tuning  "
        f"epochs={epochs}  batch={batch_size}  img_size={img_size}"
    )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics required: pip install ultralytics==8.3.2")

    # Load pretrained YOLOv8n-detect as starting point.
    # We use the DETECT model (not seg) because our auto-annotator writes
    # bounding-box labels only (class cx cy w h).  The seg model requires
    # polygon segment masks which we don't have.
    model_weights = _WEIGHTS_DIR / "yolov8n.pt"
    model = YOLO(
        str(model_weights) if model_weights.exists() else "yolov8n.pt"
    )
    logger.info(
        f"[YOLO-Finetune] Base model: yolov8n-detect "
        f"({'cached' if model_weights.exists() else 'auto-downloading'})"
    )

    # Training output directory
    run_dir = _WEIGHTS_DIR / "yolo_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = "0" if _cuda_available() else "cpu"
    logger.info(f"[YOLO-Finetune] Device: {'GPU (cuda:0)' if device == '0' else 'CPU'}")

    # Auto-adjust batch size for CPU
    effective_batch = batch_size if device == "0" else min(batch_size, 8)
    if effective_batch != batch_size:
        logger.info(f"[YOLO-Finetune] CPU detected — reduced batch to {effective_batch}")

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=effective_batch,
            imgsz=img_size,
            device=device,
            workers=2,       # Reduced from 8 — prevents /dev/shm exhaustion on
                             # RTX 3060 Laptop (6 GB VRAM) inside Docker.
                             # Increase to 4 only if you add shm_size: 8gb
                             # to docker-compose.yml backend service.
            project=str(run_dir),
            name="arken_indian_rooms",
            exist_ok=True,
            # Augmentation — tuned for indoor images
            hsv_h=0.015,     # hue shift — helps with varied Indian paint colours
            hsv_s=0.5,       # saturation shift — handles different lighting
            hsv_v=0.3,       # brightness shift — indoor lighting variation
            degrees=5.0,     # slight rotation (room photos are mostly upright)
            translate=0.1,   # modest translation
            scale=0.4,       # scale variation
            shear=2.0,       # minor shear
            flipud=0.0,      # no vertical flip (rooms are right-way-up)
            fliplr=0.5,      # horizontal flip OK (mirror image of room is valid)
            mosaic=0.8,      # mosaic augmentation
            mixup=0.1,       # mixup
            # Training hyperparameters
            lr0=0.001,       # lower initial LR for fine-tuning
            lrf=0.01,        # final LR fraction
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            patience=15,     # early stopping patience
            save_period=10,  # save checkpoint every 10 epochs
            verbose=True,
        )

        # Locate best weights
        best_weights_src = (
            run_dir / "arken_indian_rooms" / "weights" / "best.pt"
        )
        if not best_weights_src.exists():
            # Fallback to last.pt
            best_weights_src = (
                run_dir / "arken_indian_rooms" / "weights" / "last.pt"
            )

        if best_weights_src.exists():
            dest = _WEIGHTS_DIR / "yolo_indian_rooms.pt"
            shutil.copy2(str(best_weights_src), str(dest))
            logger.info(f"[YOLO-Finetune] Best weights saved → {dest}")
        else:
            logger.warning(
                f"[YOLO-Finetune] Could not find best.pt at {best_weights_src}. "
                "Check training output in yolo_runs/arken_indian_rooms/weights/"
            )

        # Extract metrics from results
        try:
            metrics = results.results_dict if hasattr(results, "results_dict") else {}
        except Exception:
            metrics = {}

        report = {
            "training_date":     datetime.now(tz=timezone.utc).isoformat(),
            "model":             "YOLOv8n-detect (fine-tuned on Indian interior rooms)",
            "base_model":        "yolov8n.pt (pretrained COCO)",
            "dataset":           "interior_design_images_metadata (auto-annotated pseudo-labels)",
            "n_train_images":    n_train_images,
            "epochs_requested":  epochs,
            "batch_size":        effective_batch,
            "img_size":          img_size,
            "device":            device,
            "classes":           YOLO_CLASS_NAMES,
            "n_classes":         len(YOLO_CLASSES),
            "annotation_method": "YOLOv8x pseudo-labels (conf > 0.60)",
            "weights_file":      "ml/weights/yolo_indian_rooms.pt",
            "metrics":           {k: round(float(v), 4) for k, v in metrics.items()
                                  if isinstance(v, (int, float))},
            "mAP50":             round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50_95":          round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
        }

        report_path = _WEIGHTS_DIR / "yolo_training_report.json"
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info(
            f"[YOLO-Finetune] Done. "
            f"mAP50={report['mAP50']:.3f}  mAP50-95={report['mAP50_95']:.3f}  "
            f"report → {report_path}"
        )
        return report

    except Exception as e:
        logger.error(f"[YOLO-Finetune] Training failed: {e}")
        raise


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(
    n_images: int,
    n_annotated: int,
    total_detections: int,
    train_report: Optional[Dict],
) -> None:
    print("\n" + "=" * 70)
    print("ARKEN YOLOv8 Fine-Tuning Summary")
    print("=" * 70)
    print(f"\nAuto-annotation (YOLOv8x pseudo-labels):")
    print(f"  Total images discovered  : {n_images}")
    print(f"  Images annotated         : {n_annotated}")
    print(f"  Total detections saved   : {total_detections}")
    print(f"  Label output root        : {_LABEL_ROOT}")
    if train_report:
        print(f"\nYOLOv8n-detect Fine-Tuning:")
        print(f"  Classes                  : {', '.join(train_report.get('classes', []))}")
        print(f"  n_classes                : {train_report.get('n_classes', 0)}")
        print(f"  n_train_images           : {train_report.get('n_train_images', 0)}")
        print(f"  mAP50                    : {train_report.get('mAP50', 0):.3f}")
        print(f"  mAP50-95                 : {train_report.get('mAP50_95', 0):.3f}")
        print(f"  Weights saved            : {train_report.get('weights_file', 'N/A')}")
    print("\n" + "=" * 70)
    print(f"All outputs in: {_WEIGHTS_DIR}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ARKEN YOLOv8 Fine-Tuning Pipeline v1.0\n"
            "Stage 1: Auto-annotate interior images with YOLOv8x pseudo-labels.\n"
            "Stage 2: Fine-tune YOLOv8n-seg on the annotated dataset.\n"
        )
    )
    parser.add_argument(
        "--stage",
        choices=["annotate", "train", "all"],
        default="all",
        help="Which stage to run (default: all).",
    )
    parser.add_argument(
        "--epochs",        type=int,   default=50,
        help="Fine-tuning epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",    type=int,   default=16,
        help="Training batch size (default: 16; auto-reduced to 8 on CPU).",
    )
    parser.add_argument(
        "--img-size",      type=int,   default=640,
        help="YOLO input image size (default: 640).",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.40,
        help="Minimum confidence for pseudo-label acceptance (default: 0.40).",
    )
    parser.add_argument(
        "--annotation-batch-size", type=int, default=8,
        help="Images per annotation batch (default: 8).",
    )
    parser.add_argument(
        "--weights-dir", type=str, default=None,
        help="Override output weights directory.",
    )
    args = parser.parse_args()

    global _WEIGHTS_DIR, _LABEL_ROOT
    if args.weights_dir:
        _WEIGHTS_DIR = Path(args.weights_dir)
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        _LABEL_ROOT = _BACKEND_DIR / "data" / "datasets" / "interior_design_images_metadata" / "labels"
        _LABEL_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ARKEN YOLOv8 Fine-Tuning Pipeline v1.0")
    logger.info(f"Stage          : {args.stage}")
    logger.info(f"Weights dir    : {_WEIGHTS_DIR}")
    logger.info(f"Dataset roots  : {_DATASET_ROOTS}")
    logger.info(f"Label root     : {_LABEL_ROOT}")
    logger.info("=" * 60)

    if not _DATASET_ROOTS:
        logger.error(
            "No dataset roots found. Ensure images exist at:\n"
            "  backend/data/datasets/interior_design_images_metadata/{room}/{style}/\n"
            "  OR\n"
            "  backend/data/datasets/interior_design_material_style/{room}/{style}/"
        )
        sys.exit(1)

    n_images      = 0
    n_annotated   = 0
    total_dets    = 0
    train_report  = None
    data_yaml_path: Optional[Path] = None

    # ── Stage 1: Discover + Annotate ─────────────────────────────────────────
    if args.stage in ("annotate", "all"):
        images = discover_images()
        n_images = len(images)

        if n_images == 0:
            logger.error(
                "No images found. Check dataset folder structure:\n"
                "  {root}/{room_type}/{style}/*.jpg\n"
                "Example: interior_design_images_metadata/bathroom/scandinavian/bathroom_scandinavian_4.jpg"
            )
            sys.exit(1)

        annotated_paths, label_paths, total_dets = auto_annotate(
            images,
            confidence_threshold=args.conf_threshold,
            batch_size=args.annotation_batch_size,
        )
        n_annotated = len(annotated_paths)

        if n_annotated < 20:
            logger.error(
                f"Only {n_annotated} images were annotated (minimum 20 needed). "
                "Consider lowering --conf-threshold (e.g. 0.50) or checking images."
            )
            sys.exit(1)

        # Build data.yaml
        data_yaml_path = create_data_yaml(annotated_paths)

    # ── Stage 2: Fine-tune ───────────────────────────────────────────────────
    if args.stage in ("train", "all"):
        # If training-only, find existing data.yaml
        if data_yaml_path is None:
            yaml_candidate = _WEIGHTS_DIR / "yolo_dataset" / "data.yaml"
            if yaml_candidate.exists():
                data_yaml_path = yaml_candidate
                logger.info(f"[train] Using existing data.yaml at {data_yaml_path}")
            else:
                logger.error(
                    f"data.yaml not found at {yaml_candidate}. "
                    "Run --stage annotate first to generate labels and data.yaml."
                )
                sys.exit(1)

        train_report = finetune_yolo(
            data_yaml=data_yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            n_train_images=n_annotated,
        )

    _print_summary(n_images, n_annotated, total_dets, train_report)


if __name__ == "__main__":
    main()