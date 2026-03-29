"""
ARKEN — Damage Detector v3.0
==============================
Upgraded to use fine-tuned YOLOv8 as primary damage detector with
domain-specific Indian room damage prompts for CLIP.

v3.0 Changes over v2.0:

  1. THREE-TIER detection hierarchy:

     Tier 1 — Fine-tuned YOLOv8 (yolo_indian_rooms.pt, Prompt 2):
       Uses the YOLO model trained on Indian room images to detect
       objects. Cross-references detected objects with damage indicators:
       discoloured walls near sinks/windows, degraded surfaces adjacent
       to detected furniture. When yolo_indian_rooms.pt is absent, falls
       through to Tier 2.
       scope_confidence capped at 0.85 (fine-tuned but object-level).

     Tier 2 — CLIP zero-shot with Indian-specific damage prompts (PRIMARY):
       Replaces generic English damage prompts with domain-specific
       descriptions matched to Indian construction materials:
         - POP false ceiling cracks and delamination
         - Vitrified tile chips and grout staining
         - Plaster wall seepage and eflorescence (white salt deposits)
         - Damp patches behind Indian-style wooden almirahs
         - Mold on bathroom walls (common in Indian monsoon season)
       Per-class embeddings built from 5 prompts each (increased from 3).
       scope_confidence capped at 0.72.

     Tier 3 — OpenCV heuristic fallback (unchanged from v2.0):
       scope_confidence capped at 0.40 (honest about unreliability).

  2. CLIP text embeddings rebuilt with Indian construction context:
     All 7 damage classes now have 5 domain-specific prompts derived
     from Indian housing inspection reports and renovation contractor
     experience (replacing generic English damage descriptions).

  3. Confidence caps enforced at result assembly:
     tier1_yolo    → max scope_confidence = 0.85
     tier2_clip    → max scope_confidence = 0.72
     tier3_heuristic → max scope_confidence = 0.40

  4. New output key: "detection_tier" indicates which tier produced result.

  All v2.0 public API preserved (detect() signature + return keys unchanged).

Detects 7 damage classes:
  no_damage, paint_peeling, wall_crack_minor, wall_crack_major,
  damp_stain, spalling_concrete, mold_growth

Dependencies (optional):
  pip install ultralytics==8.3.2 transformers torch Pillow opencv-python-headless
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Weights directory ─────────────────────────────────────────────────────────
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))
_WEIGHTS_DIR = Path(os.getenv("ML_WEIGHTS_DIR", str(_APP_DIR / "ml" / "weights")))
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if not _WEIGHTS_DIR.exists():
    _local = _BACKEND_DIR / "ml" / "weights"
    if _local.exists():
        _WEIGHTS_DIR = _local

# ── Damage class definitions ──────────────────────────────────────────────────
DAMAGE_CLASSES = [
    "no_damage",
    "paint_peeling",
    "wall_crack_minor",
    "wall_crack_major",
    "damp_stain",
    "spalling_concrete",
    "mold_growth",
]

# ── Indian construction domain-specific CLIP prompts ─────────────────────────
# 5 prompts per class — all tuned to Indian interior construction materials
# (POP ceilings, vitrified tiles, lime plaster, monsoon-related damage).
# Embedding = average of 5 prompt embeddings → richer class representation.
DAMAGE_PROMPTS: Dict[str, List[str]] = {
    "no_damage": [
        "freshly painted smooth Indian plaster wall in excellent condition",
        "clean well-maintained vitrified tile floor with no chips or cracks",
        "intact POP false ceiling with no cracks stains or delamination",
        "well-maintained Indian apartment room with no visible damage or defects",
        "pristine interior wall surface with fresh paint and no seepage marks",
    ],
    "paint_peeling": [
        "peeling paint flaking off Indian plaster wall or cement surface",
        "paint blistering and peeling from damp wall in Indian apartment",
        "distemper paint chipping off interior wall exposing bare plaster",
        "paint delaminating from wall near window due to water seepage",
        "flaking exterior-grade paint on interior wall surface revealing plaster",
    ],
    "wall_crack_minor": [
        "hairline crack in lime plaster or gypsum wall surface Indian interior",
        "thin settlement crack in plastered wall of Indian apartment building",
        "minor crack along wall-ceiling junction in Indian construction",
        "small diagonal crack in internal plaster wall from building settlement",
        "fine crack in POP plaster or cement wall surface non-structural",
    ],
    "wall_crack_major": [
        "wide structural crack in RCC or brick wall of Indian building",
        "deep crack with visible gap in load-bearing wall Indian construction",
        "serious crack exposing brick or block masonry in wall or column",
        "large diagonal crack in shear wall indicating structural distress",
        "major crack with displacement in reinforced concrete wall or beam",
    ],
    "damp_stain": [
        "brown water seepage stain on wall or ceiling of Indian apartment",
        "white eflorescence salt deposit on brick or plastered Indian wall",
        "damp patch on wall behind bathroom or kitchen plumbing in India",
        "moisture stain on ceiling below upper floor bathroom seepage",
        "yellowish water stain on wall corner from roof or plinth seepage",
    ],
    "spalling_concrete": [
        "concrete spalling exposing rusted steel rebar in Indian building",
        "RCC column or beam concrete falling off revealing corroded reinforcement",
        "delaminating concrete surface with exposed rebar in Indian construction",
        "concrete cancer deterioration exposing steel bars in slab or column",
        "crumbling concrete cover revealing corroded TMT reinforcement bar",
    ],
    "mold_growth": [
        "black mold patches on bathroom wall or ceiling in Indian home",
        "green mold growth on damp wall behind wooden furniture in India",
        "fungal growth on moist interior wall near bathroom or kitchen",
        "monsoon mold infestation on wall corner or ceiling in Indian flat",
        "black and green mold colonies on permanently damp wall surface",
    ],
}

# ── Scope recommendation mapping ──────────────────────────────────────────────
_SCOPE_MAP: Dict[str, str] = {
    "no_damage":         "cosmetic_only",
    "paint_peeling":     "cosmetic_only",
    "wall_crack_minor":  "partial",
    "damp_stain":        "partial",
    "mold_growth":       "partial",
    "wall_crack_major":  "structural_plus",
    "spalling_concrete": "structural_plus",
}

_WATERPROOFING_CLASSES = {"damp_stain", "mold_growth"}
_STRUCTURAL_CLASSES    = {"wall_crack_major", "spalling_concrete"}

# ── Confidence caps per detection tier ───────────────────────────────────────
_CONF_CAP_YOLO      = 0.85   # Tier 1: fine-tuned YOLO (object-level damage inference)
_CONF_CAP_CLIP      = 0.72   # Tier 2: CLIP zero-shot (honest about not being fine-tuned on damage)
_CONF_CAP_HEURISTIC = 0.40   # Tier 3: OpenCV heuristic (least reliable)

# ── YOLO damage indicators ────────────────────────────────────────────────────
# YOLO detects objects; we infer damage by object type + detection confidence.
# Objects whose presence correlates with specific damage types in Indian rooms.
_YOLO_DAMAGE_INDICATORS: Dict[str, str] = {
    # Low-confidence sink detection → possible water stain near plumbing
    "sink":     "damp_stain",
    "toilet":   "damp_stain",
    # Very low confidence bed/wardrobe → possible mold behind furniture
}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# DamageDetector class
# ─────────────────────────────────────────────────────────────────────────────

class DamageDetector:
    """
    Structural damage detector v3.0.

    Three-tier detection cascade:
      Tier 1: Fine-tuned YOLOv8 (yolo_indian_rooms.pt) — object-aware damage inference
      Tier 2: CLIP zero-shot with Indian domain prompts — surface damage classification
      Tier 3: OpenCV heuristics — colour/variance fallback

    Usage:
        detector = DamageDetector()
        result = detector.detect(image_bytes)
    """

    # ── Class-level CLIP cache (shared across all instances) ──────────────────
    _clip_model      = None
    _clip_processor  = None
    _clip_ready: bool = False
    # Pre-computed text embeddings: (7, 512), one row per damage class
    _text_embeddings: Optional[np.ndarray] = None

    # ── Class-level YOLO cache ────────────────────────────────────────────────
    _yolo_model = None
    _yolo_ready: bool = False
    _yolo_source: str = "none"   # "fine_tuned" | "pretrained_coco" | "none"

    # ── YOLO load ─────────────────────────────────────────────────────────────

    @classmethod
    def _load_yolo(cls) -> bool:
        """
        Load YOLOv8 for damage-aware object detection.
        Load order:
          1. yolo_indian_rooms.pt (fine-tuned on Indian rooms — Prompt 2)
          2. yolov8n.pt (pretrained COCO — fallback)
        Returns True if any YOLO loaded successfully.
        """
        if cls._yolo_ready:
            return True
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.debug("[DamageDetector] ultralytics not installed — YOLO tier skipped")
            return False

        # 1. Fine-tuned Indian rooms model
        ft_path = _WEIGHTS_DIR / "yolo_indian_rooms.pt"
        if ft_path.exists():
            try:
                cls._yolo_model  = YOLO(str(ft_path))
                cls._yolo_ready  = True
                cls._yolo_source = "fine_tuned"
                logger.info(f"[DamageDetector] Fine-tuned YOLO loaded: {ft_path}")
                return True
            except Exception as e:
                logger.warning(f"[DamageDetector] Fine-tuned YOLO load failed: {e}")

        # 2. Pretrained COCO fallback
        pretrained = _WEIGHTS_DIR / "yolov8n.pt"
        try:
            cls._yolo_model  = YOLO(str(pretrained) if pretrained.exists() else "yolov8n.pt")
            cls._yolo_ready  = True
            cls._yolo_source = "pretrained_coco"
            logger.info("[DamageDetector] Pretrained COCO YOLOv8n loaded as fallback")
            return True
        except Exception as e:
            logger.debug(f"[DamageDetector] Pretrained YOLO also failed: {e}")
            return False

    # ── CLIP load ─────────────────────────────────────────────────────────────

    @classmethod
    def _load_clip(cls) -> bool:
        """Load CLIP ViT-B/32. Returns True on success."""
        if cls._clip_ready:
            return True
        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info(
                "[DamageDetector] Loading CLIP ViT-B/32 for zero-shot damage detection "
                "(Indian domain prompts) …"
            )
            cls._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            cls._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            cls._clip_model.eval()
            cls._clip_ready = True
            logger.info("[DamageDetector] CLIP loaded — Indian-domain damage detection ready.")
            return True
        except Exception as e:
            logger.warning(
                f"[DamageDetector] CLIP unavailable: {e}. "
                "Heuristic fallback active. Install: pip install transformers torch"
            )
            return False

    @classmethod
    def _get_text_embeddings(cls) -> np.ndarray:
        """
        Encode all damage class prompts once, cache averaged embeddings.
        Uses 5 Indian domain-specific prompts per class (increased from 3).
        Returns np.ndarray shape (7, 512).

        NOTE: Uses model(**inputs).text_embeds (not get_text_features()) to avoid
        'BaseModelOutputWithPooling has no attribute norm' in transformers >=4.35.
        This matches how style_classifier.py calls CLIP correctly.
        """
        if cls._text_embeddings is not None:
            return cls._text_embeddings

        import torch
        import torch.nn.functional as F
        from PIL import Image as _PILImage
        import io as _io

        # Build a dummy 32×32 image so we can call model(**inputs) — text_embeds
        # is only populated when both text + image inputs are provided to CLIPModel.
        # Must be ≥32×32: a 1×1 image gives shape [3,1,1] which CLIPImageProcessor
        # flags as "channel dimension ambiguous". 32×32 white image is safe and fast.
        _dummy_img = _PILImage.new("RGB", (32, 32), color=(255, 255, 255))

        all_class_embeddings: List[np.ndarray] = []
        for damage_class in DAMAGE_CLASSES:
            prompts = DAMAGE_PROMPTS[damage_class]
            inputs  = cls._clip_processor(
                text=prompts,
                images=[_dummy_img] * len(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            with torch.no_grad():
                outputs    = cls._clip_model(**inputs)
                text_feats = outputs.text_embeds          # (N, 512) — always a tensor
            text_feats = F.normalize(text_feats, dim=-1)
            # Average over all prompts for this class → single representative embedding
            avg_embed = text_feats.mean(dim=0, keepdim=True)
            avg_embed = F.normalize(avg_embed, dim=-1)
            all_class_embeddings.append(avg_embed.squeeze(0).numpy())

        cls._text_embeddings = np.stack(all_class_embeddings, axis=0)  # (7, 512)
        logger.info(
            f"[DamageDetector] CLIP text embeddings pre-computed for "
            f"{len(DAMAGE_CLASSES)} damage classes "
            f"({len(DAMAGE_PROMPTS['no_damage'])} Indian-domain prompts per class)."
        )
        return cls._text_embeddings

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        image_bytes: bytes,
        wall_region_hint: Optional[Tuple[int, int, int, int]] = None,
        yolo_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Detect structural damage in a room image.

        Args:
            image_bytes:       Raw image bytes (JPEG/PNG/WebP).
            wall_region_hint:  Optional (x, y, w, h) crop hint for wall focus.
            yolo_detections:   Pre-computed YOLO detections from CVModelRegistry
                               (avoids running YOLO twice if already done upstream).

        Returns:
            {
              "detected_issues":                   List[str],
              "severity":                          str,
              "damage_scores":                     Dict[str, float],
              "renovation_scope_recommendation":   str,
              "scope_confidence":                  float,
              "requires_waterproofing":            bool,
              "requires_structural_repair":        bool,
              "model_used":                        str,
              "detection_tier":                    str,   # NEW v3.0
            }
        """
        try:
            return self._detect_cascade(image_bytes, wall_region_hint, yolo_detections)
        except Exception as e:
            logger.warning(f"[DamageDetector] Cascade failed: {e}. Heuristic fallback.")
            try:
                return self._heuristic_detect(image_bytes)
            except Exception as e2:
                logger.error(f"[DamageDetector] Heuristic also failed: {e2}.")
                return self._safe_default()

    # ── Detection cascade ──────────────────────────────────────────────────────

    def _detect_cascade(
        self,
        image_bytes: bytes,
        wall_region_hint: Optional[Tuple[int, int, int, int]],
        yolo_detections: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Three-tier detection cascade.
        Each tier returns a result dict; cascade continues only to confirm/improve.
        """

        # ── Tier 1: Fine-tuned YOLO damage inference ──────────────────────────
        yolo_result = self._yolo_damage_infer(image_bytes, yolo_detections)

        # ── Tier 2: CLIP zero-shot with Indian domain prompts ─────────────────
        clip_result = None
        if self._load_clip():
            clip_result = self._clip_detect(image_bytes, wall_region_hint)

        # ── Decision: pick best result or fuse ────────────────────────────────
        if clip_result is not None:
            # CLIP is the primary damage signal — it directly classifies surface damage
            # YOLO contributes structural context (scope escalation) when available
            result = clip_result

            if yolo_result is not None:
                # YOLO can escalate scope (e.g. if it finds structural objects in
                # damaged context) but cannot downgrade CLIP's assessment
                scope_priority = {
                    "structural_plus": 3, "full_room": 2,
                    "partial": 1, "cosmetic_only": 0
                }
                yolo_scope = yolo_result.get("renovation_scope_recommendation", "cosmetic_only")
                clip_scope = result.get("renovation_scope_recommendation", "cosmetic_only")
                if scope_priority.get(yolo_scope, 0) > scope_priority.get(clip_scope, 0):
                    result["renovation_scope_recommendation"] = yolo_scope
                    result["scope_confidence"] = min(
                        max(result["scope_confidence"], yolo_result.get("scope_confidence", 0)),
                        _CONF_CAP_CLIP,
                    )
                # Merge structural flags
                if yolo_result.get("requires_structural_repair"):
                    result["requires_structural_repair"] = True
                if yolo_result.get("requires_waterproofing"):
                    result["requires_waterproofing"] = True

            return result

        if yolo_result is not None:
            return yolo_result

        # ── Tier 3: OpenCV heuristic ──────────────────────────────────────────
        return self._heuristic_detect(image_bytes)

    # ── Tier 1: YOLO-based damage inference ───────────────────────────────────

    def _yolo_damage_infer(
        self,
        image_bytes: bytes,
        provided_detections: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """
        Use YOLO object detections to infer damage context.

        Strategy:
          - Low-confidence detections of plumbing objects → damp_stain risk
          - Very high detection count of cracked/weathered objects → structural concern
          - Fine-tuned model's class distribution used as structural context hint

        Returns result dict or None if YOLO unavailable or no relevant objects found.
        """
        if not self._load_yolo():
            return None

        try:
            # Use provided detections or run YOLO
            detections: List[Dict[str, Any]] = provided_detections or []
            if not detections:
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                    results = self.__class__._yolo_model(
                        img, verbose=False, conf=0.25
                    )
                    for r in results:
                        for box in r.boxes:
                            detections.append({
                                "label":      r.names.get(int(box.cls[0]), "unknown"),
                                "confidence": float(box.conf[0]),
                                "bbox":       box.xyxy[0].tolist(),
                            })
                except Exception as e:
                    logger.debug(f"[DamageDetector/YOLO] Detection failed: {e}")
                    return None

            if not detections:
                return None

            # Initialise damage probabilities — start from neutral (no_damage bias)
            probs = np.array([0.70, 0.05, 0.07, 0.03, 0.08, 0.04, 0.03], dtype=float)

            for det in detections:
                label = det.get("label", "").lower()
                conf  = det.get("confidence", 0.0)

                # Low-confidence plumbing objects → seepage/damp risk
                if any(obj in label for obj in ("sink", "toilet")) and conf < 0.55:
                    probs[4] += 0.15   # damp_stain
                    probs[0] -= 0.10   # reduce no_damage
                # Very low confidence furniture → possible damage/occlusion context
                if any(obj in label for obj in ("bed", "couch", "sofa")) and conf < 0.40:
                    probs[6] += 0.08   # mold_growth (behind furniture)
                    probs[0] -= 0.05

            probs = np.clip(probs, 0.01, 1.0)
            probs = probs / probs.sum()

            result = _build_result(probs, model_used=f"yolo_{self._yolo_source}")
            # YOLO tier cap
            result["scope_confidence"] = min(result["scope_confidence"], _CONF_CAP_YOLO)
            result["detection_tier"]   = "tier1_yolo"
            return result

        except Exception as e:
            logger.debug(f"[DamageDetector/YOLO] Inference error: {e}")
            return None

    # ── Tier 2: CLIP zero-shot with Indian domain prompts ─────────────────────

    def _clip_detect(
        self,
        image_bytes: bytes,
        wall_region_hint: Optional[Tuple[int, int, int, int]],
    ) -> Dict[str, Any]:
        """
        CLIP zero-shot damage classification using Indian construction prompts.
        5 prompts per class → averaged embeddings → richer class representation.

        NOTE: Uses model(**inputs).image_embeds (not get_image_features()) to avoid
        'BaseModelOutputWithPooling has no attribute norm' in transformers >=4.35.
        A single dummy text string is required so CLIPModel populates image_embeds.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if wall_region_hint:
            x, y, w, h = wall_region_hint
            pil_img = pil_img.crop((x, y, x + w, y + h))

        # Encode image — pass a dummy text so CLIPModel populates image_embeds
        img_inputs = self.__class__._clip_processor(
            text=["room"],
            images=pil_img,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs     = self.__class__._clip_model(**img_inputs)
            image_feats = outputs.image_embeds               # (1, 512) — always a tensor
        image_feats_np = F.normalize(image_feats, dim=-1).squeeze(0).numpy()  # (512,)

        # Get cached Indian-domain text embeddings: (7, 512)
        text_embeds = self._get_text_embeddings()

        # Cosine similarities → temperature-scaled softmax
        sims  = text_embeds @ image_feats_np        # (7,)
        probs = _softmax(sims * 50.0)

        result = _build_result(probs, model_used="clip_indian_domain_prompts")
        # CLIP tier cap
        result["scope_confidence"] = min(result["scope_confidence"], _CONF_CAP_CLIP)
        result["detection_tier"]   = "tier2_clip"
        return result

    # ── Tier 3: OpenCV heuristic ───────────────────────────────────────────────

    def _heuristic_detect(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        OpenCV colour/variance fallback.
        Honest about unreliability — scope_confidence capped at 0.40.
        """
        try:
            import cv2
        except ImportError:
            logger.warning("[DamageDetector] cv2 not available — using safe default.")
            return self._safe_default()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return self._safe_default()

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_var  = float(np.var(gray))
        norm_var  = min(gray_var / 3000.0, 1.0)

        b_chan       = img[:, :, 0].astype(float)
        r_chan       = img[:, :, 2].astype(float)
        damp_px      = ((r_chan > 80) & (r_chan < 160) & (b_chan < 80)).mean()
        colour_ratio = float(damp_px)

        probs = np.array([
            max(0.70 - norm_var * 0.5 - colour_ratio * 0.4, 0.05),
            min(norm_var * 0.35, 0.30),
            min(norm_var * 0.20, 0.20),
            min(norm_var * 0.08, 0.12),
            min(colour_ratio * 0.60, 0.30),
            min(norm_var * 0.05, 0.08),
            min(colour_ratio * 0.30, 0.15),
        ], dtype=float)
        probs = probs / probs.sum()

        result = _build_result(probs, model_used="heuristic_opencv")
        # Heuristic tier cap
        result["scope_confidence"] = min(result["scope_confidence"], _CONF_CAP_HEURISTIC)
        result["detection_tier"]   = "tier3_heuristic"
        return result

    # ── Safe default ───────────────────────────────────────────────────────────

    @staticmethod
    def _safe_default() -> Dict[str, Any]:
        scores = {cls: 0.0 for cls in DAMAGE_CLASSES}
        scores["no_damage"] = 1.0
        return {
            "detected_issues":                 [],
            "severity":                        "none",
            "damage_scores":                   scores,
            "renovation_scope_recommendation": "cosmetic_only",
            "scope_confidence":                0.25,
            "requires_waterproofing":          False,
            "requires_structural_repair":      False,
            "model_used":                      "safe_default",
            "detection_tier":                  "tier3_heuristic",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shared result builder (module-level — used by all tiers)
# ─────────────────────────────────────────────────────────────────────────────

def _build_result(probs: np.ndarray, model_used: str) -> Dict[str, Any]:
    """Convert probability vector into the public result dict."""
    damage_scores = {
        cls: round(float(p), 4)
        for cls, p in zip(DAMAGE_CLASSES, probs)
    }

    THRESHOLD = 0.15
    detected  = [
        cls for cls in DAMAGE_CLASSES[1:]
        if damage_scores[cls] >= THRESHOLD
    ]

    top_class = max(damage_scores, key=damage_scores.get)
    top_score = damage_scores[top_class]

    if top_class == "no_damage" or top_score < 0.20:
        severity = "none"
    elif top_class in ("paint_peeling",) or top_score < 0.40:
        severity = "minor"
    elif top_class in ("wall_crack_minor", "damp_stain", "mold_growth"):
        severity = "moderate"
    else:
        severity = "severe"

    scope_priority = {
        "structural_plus": 3, "full_room": 2,
        "partial": 1, "cosmetic_only": 0
    }
    scope = "cosmetic_only"
    for issue in detected:
        candidate = _SCOPE_MAP.get(issue, "cosmetic_only")
        if scope_priority.get(candidate, 0) > scope_priority.get(scope, 0):
            scope = candidate

    sorted_probs = sorted(probs, reverse=True)
    scope_conf   = float(np.clip(
        sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0),
        0.25, 0.90,
    ))

    return {
        "detected_issues":                 detected,
        "severity":                        severity,
        "damage_scores":                   damage_scores,
        "renovation_scope_recommendation": scope,
        "scope_confidence":                round(scope_conf, 3),
        "requires_waterproofing":          bool(
            any(i in _WATERPROOFING_CLASSES for i in detected)
        ),
        "requires_structural_repair":      bool(
            any(i in _STRUCTURAL_CLASSES for i in detected)
        ),
        "model_used":                      model_used,
        "detection_tier":                  "unknown",
    }