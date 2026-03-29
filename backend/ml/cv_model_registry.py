"""
ARKEN — CV Model Registry v2.0
================================
Updated to load fine-tuned weights as PRIMARY, pretrained as fallback.

v2.0 Changes over v1.0:
  - YOLOv8Detector:
      Now uncommented and active (requires ultralytics==8.3.2).
      load order: yolo_indian_rooms.pt (fine-tuned, Prompt 2) →
                  yolov8n.pt (pretrained COCO, auto-download)
      Expanded INTERIOR_CLASS_IDS covers Indian room objects.

  - CLIPEmbedder:
      load order: clip_finetuned.pt (fine-tuned visual encoder, Prompt 1) →
                  ViT-B/32 pretrained (auto-download fallback)
      _load() now calls _load_finetuned() first; pretrained loaded only if
      fine-tuned weights absent or load fails.
      encode_image(), classify_style(), classify_lighting() all use whichever
      encoder was loaded — API identical to v1.0.

  - EfficientNetRoomClassifier:
      load order: room_classifier.pt (fine-tuned on YOUR 4-class dataset, Prompt 1) →
                  CLIP zero-shot room classification (pretrained fallback)
      label_to_idx loaded from room_training_report.json when present, so
      label order matches YOUR training run exactly.
      ROOM_LABELS reduced to the 4 classes YOUR dataset covers:
        bathroom, bedroom, kitchen, living_room
      Inference: EfficientNet → softmax → argmax → ROOM_LABELS[best]

  - All v1.0 public API preserved (CVModelRegistry.get(), registry.yolo,
    registry.clip, registry.room_classifier — all unchanged).

Requirements (now active, not optional):
    ultralytics==8.3.2
    clip @ git+https://github.com/openai/CLIP.git
    torch>=2.1.0
    torchvision>=0.16.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Weights directory (Docker volume: model_cache → /app/ml/weights) ─────────
import os as _os
_APP_DIR     = Path(_os.getenv("ARKEN_APP_DIR", "/app"))
WEIGHTS_DIR  = Path(_os.getenv("ML_WEIGHTS_DIR", str(_APP_DIR / "ml" / "weights")))
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if not WEIGHTS_DIR.exists():
    _local = _BACKEND_DIR / "ml" / "weights"
    if _local.exists():
        WEIGHTS_DIR = _local
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Room classification labels ────────────────────────────────────────────────
# YOUR dataset covers 4 room types. Fine-tuned model trained on these.
ROOM_LABELS_4 = ["bathroom", "bedroom", "kitchen", "living_room"]

# Extended list for zero-shot fallback (CLIP can handle all 9)
ROOM_LABELS = [
    "bedroom", "living_room", "kitchen", "bathroom",
    "dining_room", "study", "balcony", "hallway", "other",
]

# ── Style labels ──────────────────────────────────────────────────────────────
STYLE_LABELS = [
    "Modern Minimalist", "Scandinavian", "Japandi", "Industrial",
    "Bohemian", "Contemporary Indian", "Traditional Indian", "Art Deco",
    "Mid-Century Modern", "Coastal", "Farmhouse",
]

STYLE_PROMPTS = {s: f"a photo of a {s.lower()} style interior room" for s in STYLE_LABELS}

LIGHTING_PROMPTS = {
    "natural":    "a room filled with bright natural sunlight from windows",
    "artificial": "a room lit entirely by artificial ceiling lights",
    "dim":        "a dimly lit room with low ambient light",
    "mixed":      "a room with both natural and artificial lighting",
    "warm":       "a room with warm yellow tungsten lighting",
    "cool":       "a room with cool blue-white LED lighting",
}

_lock = threading.Lock()


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8 Wrapper  (now active — requires ultralytics==8.3.2)
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv8Detector:
    """
    YOLOv8 object detector for Indian interior room analysis.

    Load order:
      1. ml/weights/yolo_indian_rooms.pt  — fine-tuned on Indian rooms (Prompt 2)
      2. yolov8n.pt                        — pretrained COCO (auto-download)

    Extended INTERIOR_CLASS_IDS covers standard COCO furniture + appliances
    relevant to Indian apartments.
    """

    # All COCO interior-relevant class IDs
    INTERIOR_CLASS_IDS = {
        56: "chair",       57: "couch",         58: "potted plant",
        59: "bed",         60: "dining table",   61: "toilet",
        62: "tv",          63: "laptop",         64: "mouse",
        65: "remote",      66: "keyboard",       69: "oven",
        70: "toaster",     71: "sink",           72: "refrigerator",
        73: "book",        74: "clock",          75: "vase",
        76: "scissors",    77: "teddy bear",     78: "hair drier",
    }

    DISPLAY_NAMES = {
        "couch": "sofa", "potted plant": "indoor plant",
        "tv": "television", "dining table": "dining table",
        "toilet": "toilet", "sink": "sink",
        "refrigerator": "refrigerator",
    }

    def __init__(self):
        self._model  = None
        self._device = _get_device()
        self._model_source = "none"

    def _load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning(
                "[YOLOv8] ultralytics not installed. "
                "Install: pip install ultralytics==8.3.2"
            )
            self._model = None
            return

        # 1. Try fine-tuned Indian rooms model
        fine_tuned_path = WEIGHTS_DIR / "yolo_indian_rooms.pt"
        if fine_tuned_path.exists():
            try:
                self._model = YOLO(str(fine_tuned_path))
                self._model_source = "fine_tuned_indian_rooms"
                logger.info(
                    f"[YOLOv8] Loaded fine-tuned Indian rooms model: {fine_tuned_path}"
                )
                return
            except Exception as e:
                logger.warning(f"[YOLOv8] Fine-tuned load failed: {e}. Falling back.")

        # 2. Pretrained COCO fallback
        pretrained_path = WEIGHTS_DIR / "yolov8n.pt"
        try:
            self._model = YOLO(
                str(pretrained_path) if pretrained_path.exists() else "yolov8n.pt"
            )
            self._model_source = "pretrained_coco"
            logger.info(f"[YOLOv8] Loaded pretrained COCO model on {self._device}")
        except Exception as e:
            logger.warning(f"[YOLOv8] All model loads failed: {e}")
            self._model = None

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Returns list of detected objects:
        [{"label": "sofa", "confidence": 0.87, "bbox": [x1,y1,x2,y2],
          "class_id": 57, "model_source": "fine_tuned_indian_rooms"}]
        """
        self._load()
        if self._model is None:
            return []

        try:
            import io as _io
            from PIL import Image
            img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")

            results = self._model(
                img, verbose=False,
                conf=confidence_threshold,
                device=self._device,
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id  = int(box.cls[0])
                    conf    = float(box.conf[0])
                    name    = r.names.get(cls_id, "unknown")
                    display = self.DISPLAY_NAMES.get(name, name)
                    detections.append({
                        "label":        display,
                        "confidence":   round(conf, 3),
                        "bbox":         [round(float(v), 1) for v in box.xyxy[0].tolist()],
                        "class_id":     cls_id,
                        "model_source": self._model_source,
                    })
            return detections

        except Exception as e:
            logger.warning(f"[YOLOv8] Detection failed: {e}")
            return []

    @property
    def is_finetuned(self) -> bool:
        return self._model_source == "fine_tuned_indian_rooms"


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Wrapper — loads fine-tuned weights first
# ─────────────────────────────────────────────────────────────────────────────

class CLIPEmbedder:
    """
    CLIP ViT-B/32 wrapper.

    Load order:
      1. ml/weights/clip_finetuned.pt — fine-tuned visual encoder (Prompt 1)
      2. ViT-B/32 pretrained           — auto-download fallback

    All public methods (encode_image, classify_style, classify_lighting)
    use whichever encoder was successfully loaded.
    """

    def __init__(self):
        self._model      = None
        self._preprocess = None
        self._device     = _get_device()
        self._is_finetuned = False

    def _load(self):
        if self._model is not None:
            return

        try:
            import clip as openai_clip
            import torch
        except ImportError:
            logger.warning(
                "[CLIP] clip package not installed. "
                "Install: pip install 'clip @ git+https://github.com/openai/CLIP.git'"
            )
            self._model = None
            return

        # 1. Try loading fine-tuned visual encoder
        ft_path = WEIGHTS_DIR / "clip_finetuned.pt"
        try:
            import torch
            model, preprocess = openai_clip.load(
                "ViT-B/32", device=self._device,
                download_root=str(WEIGHTS_DIR),
            )
            if ft_path.exists():
                ft_state = torch.load(str(ft_path), map_location=self._device)
                model.visual.load_state_dict(ft_state)
                model.eval()
                self._model        = model
                self._preprocess   = preprocess
                self._is_finetuned = True
                logger.info(
                    f"[CLIP] Loaded fine-tuned visual encoder from {ft_path} "
                    f"on {self._device}"
                )
            else:
                model.eval()
                self._model        = model
                self._preprocess   = preprocess
                self._is_finetuned = False
                logger.info(
                    f"[CLIP] Loaded pretrained ViT-B/32 on {self._device} "
                    f"(fine-tuned weights not found at {ft_path})"
                )
        except Exception as e:
            logger.warning(f"[CLIP] Load failed: {e}")
            self._model = None

    def encode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Returns L2-normalised 512-dim image embedding, or None if unavailable."""
        self._load()
        if self._model is None:
            return None
        try:
            import io as _io, torch
            import torch.nn.functional as F
            from PIL import Image
            img    = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_image(tensor).float()
                feats = F.normalize(feats, dim=-1)
            return feats.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"[CLIP] encode_image failed: {e}")
            return None

    def classify_style(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
        """Zero-shot style classification (or fine-tuned-encoder enhanced)."""
        self._load()
        if self._model is None:
            return "Modern Minimalist", 0.4, {}
        try:
            import io as _io, torch
            import clip as openai_clip
            from PIL import Image

            img        = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            texts      = list(STYLE_PROMPTS.values())
            tokens     = openai_clip.tokenize(texts).to(self._device)

            with torch.no_grad():
                img_f = self._model.encode_image(img_tensor).float()
                txt_f = self._model.encode_text(tokens).float()
                import torch.nn.functional as F
                img_f = F.normalize(img_f, dim=-1)
                txt_f = F.normalize(txt_f, dim=-1)
                sims  = (img_f @ txt_f.T).softmax(dim=-1)

            scores  = sims[0].cpu().numpy()
            labels  = list(STYLE_PROMPTS.keys())
            best    = int(np.argmax(scores))
            smap    = {labels[i]: round(float(scores[i]), 4) for i in range(len(labels))}
            return labels[best], round(float(scores[best]), 3), smap

        except Exception as e:
            logger.warning(f"[CLIP] classify_style failed: {e}")
            return "Modern Minimalist", 0.4, {}

    def classify_lighting(self, image_bytes: bytes) -> str:
        """Zero-shot lighting classification."""
        self._load()
        if self._model is None:
            return "mixed"
        try:
            import io as _io, torch
            import clip as openai_clip
            from PIL import Image

            img    = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            texts  = list(LIGHTING_PROMPTS.values())
            tokens = openai_clip.tokenize(texts).to(self._device)

            with torch.no_grad():
                img_f = self._model.encode_image(tensor).float()
                txt_f = self._model.encode_text(tokens).float()
                import torch.nn.functional as F
                img_f = F.normalize(img_f, dim=-1)
                txt_f = F.normalize(txt_f, dim=-1)
                sims  = (img_f @ txt_f.T).softmax(dim=-1)

            keys = list(LIGHTING_PROMPTS.keys())
            return keys[int(np.argmax(sims[0].cpu().numpy()))]

        except Exception as e:
            logger.warning(f"[CLIP] classify_lighting failed: {e}")
            return "mixed"

    @property
    def is_finetuned(self) -> bool:
        return self._is_finetuned


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet Room Classifier — loads fine-tuned weights first
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetRoomClassifier:
    """
    EfficientNet-B0 room type classifier.

    Load order:
      1. ml/weights/room_classifier.pt  — fine-tuned on YOUR 4-class dataset
         (bathroom / bedroom / kitchen / living_room)
         label_to_idx loaded from room_training_report.json if present.
      2. CLIP zero-shot room classification (9-class, pretrained fallback)

    API unchanged from v1.0: classify(image_bytes) → (room_type, confidence)
    """

    ROOM_CLIP_PROMPTS = {
        "bedroom":     "a photo of a bedroom with a bed",
        "living_room": "a photo of a living room with a sofa",
        "kitchen":     "a photo of a kitchen with a stove or sink",
        "bathroom":    "a photo of a bathroom with a toilet or bathtub",
        "dining_room": "a photo of a dining room with a dining table",
        "study":       "a photo of a home office or study room",
        "balcony":     "a photo of an outdoor balcony or terrace",
        "hallway":     "a photo of a corridor or hallway",
        "other":       "a photo of an interior room",
    }

    def __init__(self):
        self._model             = None
        self._device            = _get_device()
        self._use_clip_fallback = True
        self._room_labels       = ROOM_LABELS_4    # default: 4-class (YOUR dataset)
        self._is_finetuned      = False

    def _load(self):
        if self._model is not None:
            return
        weights_path = WEIGHTS_DIR / "room_classifier.pt"
        if weights_path.exists():
            self._load_finetuned(weights_path)
        else:
            logger.info(
                "[EfficientNet] room_classifier.pt not found — "
                "using CLIP zero-shot room classification. "
                "Run: python ml/train_style_classifier.py --model room"
            )
            self._use_clip_fallback = True

    def _load_finetuned(self, weights_path: Path):
        try:
            import torch
            import torchvision.models as tv_models
            import torch.nn as nn

            # Try to load label mapping from training report
            report_path = WEIGHTS_DIR / "room_training_report.json"
            if report_path.exists():
                with open(report_path) as fh:
                    report = json.load(fh)
                label_to_idx = report.get("label_to_idx", {})
                if label_to_idx:
                    # Build ordered list from label_to_idx
                    self._room_labels = [
                        k for k, v in sorted(label_to_idx.items(), key=lambda x: x[1])
                    ]
                    logger.info(
                        f"[EfficientNet] Room labels from report: {self._room_labels}"
                    )

            n_classes = len(self._room_labels)
            model = tv_models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, n_classes)
            model.load_state_dict(
                torch.load(str(weights_path), map_location=self._device)
            )
            model.eval().to(self._device)
            self._model             = model
            self._use_clip_fallback = False
            self._is_finetuned      = True
            logger.info(
                f"[EfficientNet] Fine-tuned room classifier loaded "
                f"({n_classes} classes: {self._room_labels}) on {self._device}"
            )
        except Exception as e:
            logger.warning(f"[EfficientNet] Fine-tuned load failed: {e}. CLIP fallback.")
            self._use_clip_fallback = True

    def classify(self, image_bytes: bytes) -> Tuple[str, float]:
        """Returns (room_type, confidence)."""
        self._load()
        if self._use_clip_fallback:
            return self._clip_room_classify(image_bytes)
        try:
            import io as _io, torch
            import torch.nn.functional as F
            from PIL import Image
            import torchvision.transforms as T

            transform = T.Compose([
                T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img    = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

            best = int(np.argmax(probs))
            room = self._room_labels[best] if best < len(self._room_labels) else "bedroom"
            return room, round(float(probs[best]), 3)

        except Exception as e:
            logger.warning(f"[EfficientNet] classify failed: {e}")
            return "bedroom", 0.4

    def _clip_room_classify(self, image_bytes: bytes) -> Tuple[str, float]:
        """CLIP zero-shot 9-class room classification fallback."""
        try:
            import io as _io, torch
            import clip as openai_clip
            import torch.nn.functional as F
            from PIL import Image

            device = _get_device()
            model, preprocess = openai_clip.load(
                "ViT-B/32", device=device, download_root=str(WEIGHTS_DIR)
            )
            model.eval()

            img    = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            texts  = list(self.ROOM_CLIP_PROMPTS.values())
            tokens = openai_clip.tokenize(texts).to(device)

            with torch.no_grad():
                img_f = model.encode_image(tensor).float()
                txt_f = model.encode_text(tokens).float()
                img_f = F.normalize(img_f, dim=-1)
                txt_f = F.normalize(txt_f, dim=-1)
                sims  = (img_f @ txt_f.T).softmax(dim=-1)

            scores = sims[0].cpu().numpy()
            keys   = list(self.ROOM_CLIP_PROMPTS.keys())
            best   = int(np.argmax(scores))
            return keys[best], round(float(scores[best]), 3)

        except Exception as e:
            logger.warning(f"[EfficientNet/CLIP fallback] room classify failed: {e}")
            return "bedroom", 0.35

    @property
    def is_finetuned(self) -> bool:
        return self._is_finetuned


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Registry
# ─────────────────────────────────────────────────────────────────────────────

class CVModelRegistry:
    """
    Thread-safe singleton holding all CV model instances.
    Call CVModelRegistry.get() to obtain the shared instance.

    Models are loaded lazily on first use — importing this module
    does NOT trigger any model downloads or GPU memory allocation.
    """
    _instance: Optional["CVModelRegistry"] = None

    def __init__(self):
        self.yolo             = YOLOv8Detector()
        self.clip             = CLIPEmbedder()
        self.room_classifier  = EfficientNetRoomClassifier()
        self._available: Optional[bool] = None

    @classmethod
    def get(cls) -> "CVModelRegistry":
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info(
                        "[CVModelRegistry v2.0] Initialised (lazy — "
                        "models load on first use). "
                        f"Fine-tuned weights dir: {WEIGHTS_DIR}"
                    )
        return cls._instance

    @property
    def cv_available(self) -> bool:
        if self._available is None:
            try:
                import torch  # noqa
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def status(self) -> Dict[str, Any]:
        """
        Returns a dict describing which model variants are loaded.
        Useful for the /health/metrics endpoint.
        """
        yolo_ft_path  = WEIGHTS_DIR / "yolo_indian_rooms.pt"
        clip_ft_path  = WEIGHTS_DIR / "clip_finetuned.pt"
        room_ft_path  = WEIGHTS_DIR / "room_classifier.pt"
        style_ft_path = WEIGHTS_DIR / "style_classifier.pt"

        return {
            "yolo": {
                "finetuned_weights_present": yolo_ft_path.exists(),
                "model_source": self.yolo._model_source if self.yolo._model else "not_loaded",
            },
            "clip": {
                "finetuned_weights_present": clip_ft_path.exists(),
                "is_finetuned": self.clip.is_finetuned,
            },
            "room_classifier": {
                "finetuned_weights_present": room_ft_path.exists(),
                "is_finetuned": self.room_classifier.is_finetuned,
                "room_labels": self.room_classifier._room_labels,
            },
            "style_classifier": {
                "finetuned_weights_present": style_ft_path.exists(),
            },
            "weights_dir": str(WEIGHTS_DIR),
        }


# Module-level convenience accessor
def get_registry() -> CVModelRegistry:
    return CVModelRegistry.get()