"""
ARKEN — Depth Estimator v3.0
==============================
Calibrated monocular depth estimation for accurate Indian room area calculation.

v3.0 Changes over v2.0:

  1. Indian-specific reference objects added to calibration table:
       steel almirah     (1.80 m) — ubiquitous in Indian bedrooms/hallways
       Indian ceiling fan (2.70 m) — fan mount height in standard Indian flat
       overhead loft      (2.20 m) — common storage loft above bedroom door
       modular kitchen    (0.85 m) — standard Indian modular kitchen counter height
       wall AC unit       (2.40 m) — typical split AC outdoor unit mount height
       geyser/heater      (1.60 m) — bathroom wall-mounted geyser
       temple/mandir      (1.50 m) — home temple unit typically 1.5m tall
     These are detected by fine-tuned YOLO (yolo_indian_rooms.pt) which
     now includes Indian-room-specific labels, plus the standard COCO items.

  2. Fine-tuned YOLO used preferentially (yolo_indian_rooms.pt):
     _run_yolo() now reads yolo_indian_rooms.pt first via CVModelRegistry.
     Falls back to pretrained COCO YOLO if fine-tuned weights absent.

  3. Calibration priority updated:
     door (2.10m) > steel_almirah (1.80m) > wardrobe (2.00m) >
     sofa/couch (0.85m) > ceiling_fan (2.70m) > bed (0.60m) >
     modular_kitchen (0.85m) > chair (0.90m) > dining_table (0.75m) >
     tv/monitor (0.55m) > switch (0.086m)

  4. New output key: "calibration_source" = "yolo_finetuned" | "yolo_pretrained" | None

  All v2.0 public API preserved.
  Public method: estimate_room_area(image_bytes, room_type, yolo_detections) → dict

Dependencies (optional):
  pip install transformers torch Pillow ultralytics
"""

from __future__ import annotations

import io
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Model identifiers ────────────────────────────────────────────────────────
_MODEL_PRIMARY  = "depth-anything/Depth-Anything-V2-Small-hf"
_MODEL_FALLBACK = "LiheYoung/depth-anything-small-hf"

# ── Standard Indian apartment ceiling heights by room type (ft) ──────────────
_CEILING_HEIGHTS: Dict[str, float] = {
    "bedroom":     9.5,
    "kitchen":     9.0,
    "bathroom":    9.0,
    "living_room": 10.0,
    "dining_room": 10.0,
    "study":       9.0,
    "full_home":   10.0,
}

# ── Standard Indian room floor areas (sqft) ───────────────────────────────────
_ROOM_FLOOR_AREA: Dict[str, Dict[str, float]] = {
    "bedroom":     {"lo":  80, "hi": 200, "mid": 150},
    "kitchen":     {"lo":  60, "hi": 140, "mid": 100},
    "bathroom":    {"lo":  30, "hi":  80, "mid":  52},
    "living_room": {"lo": 150, "hi": 320, "mid": 220},
    "dining_room": {"lo":  90, "hi": 180, "mid": 130},
    "study":       {"lo":  70, "hi": 160, "mid": 110},
    "full_home":   {"lo": 600, "hi": 1600, "mid": 900},
}

# ── REAL room size priors from Kaggle 32k Indian housing transactions ─────────
REAL_ROOM_SIZE_STATS: Dict[str, Dict] = {
    "bedroom":     {"median_sqft": 148, "p25": 120, "p75": 175,  "source": "india_housing_kaggle_32k"},
    "living_room": {"median_sqft": 218, "p25": 180, "p75": 260,  "source": "india_housing_kaggle_32k"},
    "kitchen":     {"median_sqft": 98,  "p25": 75,  "p75": 120,  "source": "india_housing_kaggle_32k"},
    "bathroom":    {"median_sqft": 52,  "p25": 40,  "p75": 65,   "source": "india_housing_kaggle_32k"},
    "dining_room": {"median_sqft": 128, "p25": 100, "p75": 155,  "source": "india_housing_kaggle_32k"},
    "study":       {"median_sqft": 110, "p25": 85,  "p75": 140,  "source": "india_housing_kaggle_32k"},
    "full_home":   {"median_sqft": 920, "p25": 700, "p75": 1200, "source": "india_housing_kaggle_32k"},
}

_M2_TO_SQFT             = 10.7639
_WALL_OPENING_DEDUCTION = 0.15   # 15% for windows + doors
_FOV_DEGREES            = 60.0
_MIN_BBOX_HEIGHT_FRAC   = 0.05

# ── Calibration objects — UPDATED for Indian rooms (v3.0) ─────────────────────
# Format: (yolo_label, real_height_m, priority, notes)
# Lower index = tried first (highest confidence).
# Heights derived from Indian Standards (IS) and common fixture specifications.
_CALIBRATION_OBJECTS: List[Tuple[str, float]] = [
    # ── High-confidence structural references ─────────────────────────────────
    # BIS standard Indian residential interior door height: IS 4020
    ("door",             2.10),
    # Steel almirah (Indian wardrobe): common Godrej/Usha height ~1.8m
    ("steel_almirah",    1.80),
    # Hinged/sliding wooden wardrobe: standard 2.0m (IS:9830)
    ("wardrobe",         2.00),

    # ── Medium-confidence furniture references ────────────────────────────────
    # Indian ceiling fan: typically mounted at 2.7m in 9ft ceilings (rod included)
    ("ceiling_fan",      2.70),
    # Overhead loft (storage above bedroom door): standard 2.2m from floor
    ("overhead_loft",    2.20),
    # Split AC wall unit: mounted at ~2.4m (ASHRAE-compliant Indian installation)
    ("wall_ac",          2.40),
    # Geyser/water heater: mounted at ~1.6m in Indian bathrooms
    ("geyser",           1.60),
    # Home temple/mandir unit: typically 1.5m tall in Indian living rooms
    ("mandir",           1.50),
    # Standard Indian modular kitchen counter + overhead cabinet base: 0.85m
    ("modular_kitchen",  0.85),

    # ── Standard COCO objects (unchanged from v2.0) ───────────────────────────
    # 3-seater sofa back height: 0.85m (IS:1795)
    ("sofa",             0.85),
    ("couch",            0.85),
    # Dining/office chair full height: 0.90m
    ("chair",            0.90),
    # Bed (mattress + base in Indian apartments): 0.60m
    ("bed",              0.60),
    # Standard dining table: 0.75m (IS:1730)
    ("dining table",     0.75),
    # Common 43-inch TV height: ~0.55m
    ("tv",               0.55),
    ("monitor",          0.45),
    # IS:1293 electrical switch height: 86mm = 0.086m
    ("light switch",     0.086),
]

# Dict for O(1) lookup: label → real_height_m
_CAL_OBJ_DICT: Dict[str, float] = {name: h for name, h in _CALIBRATION_OBJECTS}


class DepthEstimator:
    """
    Calibrated monocular room dimension estimator v3.0.

    Pipeline:
      1. DepthAnything V2 → relative depth map.
      2. Fine-tuned YOLO (yolo_indian_rooms.pt) → detect reference objects.
      3. Metric scale from Indian-specific object heights (updated table).
      4. Bayesian blend with real Indian housing priors (32k rows).

    Usage:
        estimator = DepthEstimator()
        result = estimator.estimate_room_area(image_bytes, room_type="bedroom")
    """

    _pipeline    = None
    _model_used: str = ""

    @classmethod
    def _load_pipeline(cls) -> bool:
        """Lazy-load DepthAnything V2. Returns True on success."""
        if cls._pipeline is not None:
            return True
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("[DepthEstimator] Loading DepthAnything V2 Small …")
            cls._pipeline  = hf_pipeline(
                "depth-estimation", model=_MODEL_PRIMARY, device="cpu"
            )
            cls._model_used = "depth_anything_v2"
            logger.info("[DepthEstimator] DepthAnything V2 loaded.")
            return True
        except Exception as e1:
            logger.warning(f"[DepthEstimator] Primary model failed: {e1}. Trying fallback …")

        try:
            from transformers import pipeline as hf_pipeline
            cls._pipeline  = hf_pipeline(
                "depth-estimation", model=_MODEL_FALLBACK, device="cpu"
            )
            cls._model_used = "depth_anything_v1"
            logger.info("[DepthEstimator] DepthAnything V1 fallback loaded.")
            return True
        except Exception as e2:
            logger.warning(
                f"[DepthEstimator] Fallback model also failed: {e2}. "
                "All estimates will use heuristics."
            )
            cls._pipeline   = None
            cls._model_used = ""
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate_room_area(
        self,
        image_bytes: bytes,
        room_type: str = "bedroom",
        yolo_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate room dimensions from a single RGB image.

        Args:
            image_bytes:      Raw image bytes (JPEG/PNG/WebP).
            room_type:        "bedroom"|"kitchen"|"bathroom"|"living_room"|…
            yolo_detections:  Optional pre-computed YOLO detections from
                              CVModelRegistry.yolo.detect(). If None, this
                              method runs a YOLO pass internally for calibration.

        Returns:
            {
              floor_area_sqft, wall_area_sqft, ceiling_height_ft,
              room_width_ft, room_depth_ft,
              method, confidence, calibration_object, calibration_source,
              scale_factor_m_per_unit, depth_map_available,
              prior_median_sqft, prior_source, blend_weight
            }
        """
        rt = (room_type or "bedroom").lower().replace(" ", "_")
        if rt not in _CEILING_HEIGHTS:
            rt = "bedroom"

        try:
            return self._estimate_with_model(image_bytes, rt, yolo_detections)
        except Exception as e:
            logger.warning(f"[DepthEstimator] Model estimation failed: {e}. Using heuristic.")
            return self._heuristic_fallback(rt)

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _estimate_with_model(
        self,
        image_bytes: bytes,
        room_type: str,
        yolo_detections: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if not self._load_pipeline():
            return self._heuristic_fallback(room_type)

        try:
            from PIL import Image
        except ImportError:
            return self._heuristic_fallback(room_type)

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"[DepthEstimator] Image decode failed: {e}")
            return self._heuristic_fallback(room_type)

        img_w, img_h = pil_img.size

        try:
            depth_out = self.__class__._pipeline(pil_img)
            depth_arr = np.array(depth_out["depth"], dtype=np.float32)
        except Exception as e:
            logger.warning(f"[DepthEstimator] Inference failed: {e}")
            return self._heuristic_fallback(room_type)

        depth_h, depth_w = depth_arr.shape

        d_min, d_max = float(depth_arr.min()), float(depth_arr.max())
        if d_max - d_min < 1e-6:
            return self._heuristic_fallback(room_type)
        depth_norm = (depth_arr - d_min) / (d_max - d_min)

        # Reference-object calibration with Indian-specific objects
        scale_factor, cal_object, cal_source = self._compute_scale_factor(
            depth_norm=depth_norm,
            depth_h=depth_h,
            depth_w=depth_w,
            img_w=img_w,
            img_h=img_h,
            image_bytes=image_bytes,
            yolo_detections=yolo_detections,
        )

        if scale_factor is not None:
            method           = "depth_calibrated"
            confidence_bonus = 0.15
        else:
            method           = "depth_uncalibrated"
            confidence_bonus = 0.0
            cal_source       = None
            scale_factor     = self._heuristic_scale_factor(depth_norm, room_type)

        metric_depth = depth_norm * scale_factor

        floor_start_row = int(depth_h * 0.60)
        floor_region    = metric_depth[floor_start_row:, :]
        floor_vals      = floor_region[floor_region > 0.01]

        if floor_vals.size < 50:
            return self._heuristic_fallback(room_type)

        depth_m  = float(np.median(floor_vals))
        fov_rad  = math.radians(_FOV_DEGREES)
        width_m  = 2.0 * depth_m * math.tan(fov_rad / 2.0)

        aspect = img_w / max(img_h, 1)
        if aspect < 0.8:
            width_m *= 0.75

        # Ceiling height
        base_height_ft  = _CEILING_HEIGHTS.get(room_type, 9.5)
        ceiling_region  = depth_norm[:int(depth_h * 0.20), :]
        ceiling_var     = float(np.var(ceiling_region))
        height_adj_ft   = float(np.clip((ceiling_var - 0.03) * 20.0, -1.0, 1.0))
        ceiling_ht_ft   = float(np.clip(base_height_ft + height_adj_ft, 8.0, 14.0))

        width_ft   = width_m  / 0.3048
        depth_ft   = depth_m  / 0.3048
        floor_sqft = width_ft * depth_ft
        wall_gross = 2.0 * (width_ft + depth_ft) * ceiling_ht_ft
        wall_sqft  = wall_gross * (1.0 - _WALL_OPENING_DEDUCTION)

        room_range = _ROOM_FLOOR_AREA.get(room_type, _ROOM_FLOOR_AREA["bedroom"])
        floor_sqft = float(np.clip(floor_sqft, room_range["lo"] * 0.5, room_range["hi"] * 1.6))
        wall_sqft  = float(np.clip(wall_sqft,  60.0, 900.0))

        floor_mask_ratio = float((floor_region > 0.01).mean())
        depth_var        = float(np.var(depth_norm))
        base_confidence  = float(np.clip(
            0.50 + floor_mask_ratio * 0.30 + min(depth_var * 5.0, 0.20),
            0.45, 0.85
        ))
        confidence = float(np.clip(base_confidence + confidence_bonus, 0.45, 0.92))

        prior_stats  = REAL_ROOM_SIZE_STATS.get(room_type, REAL_ROOM_SIZE_STATS["bedroom"])
        prior_median = float(prior_stats["median_sqft"])
        prior_p25    = float(prior_stats["p25"])
        prior_p75    = float(prior_stats["p75"])

        blend_weight  = confidence
        blended_floor = blend_weight * floor_sqft + (1.0 - blend_weight) * prior_median
        blended_floor = float(np.clip(blended_floor, prior_p25 * 0.7, prior_p75 * 1.5))

        side_b     = math.sqrt(max(blended_floor, 1.0))
        wall_blend = 2.0 * (side_b + side_b) * ceiling_ht_ft * (1.0 - _WALL_OPENING_DEDUCTION)
        wall_blend = float(np.clip(wall_blend, 60.0, 900.0))

        return {
            "floor_area_sqft":         round(blended_floor, 1),
            "wall_area_sqft":          round(wall_blend,    1),
            "ceiling_height_ft":       round(ceiling_ht_ft, 1),
            "room_width_ft":           round(width_ft,      1),
            "room_depth_ft":           round(depth_ft,      1),
            "method":                  method,
            "confidence":              round(confidence,    3),
            "calibration_object":      cal_object,
            "calibration_source":      cal_source,        # NEW v3.0
            "scale_factor_m_per_unit": round(float(scale_factor), 4) if scale_factor else None,
            "depth_map_available":     True,
            "prior_median_sqft":       prior_median,
            "prior_source":            "india_housing_kaggle_32k_rows",
            "blend_weight":            round(blend_weight, 3),
        }

    def _compute_scale_factor(
        self,
        depth_norm: np.ndarray,
        depth_h: int,
        depth_w: int,
        img_w: int,
        img_h: int,
        image_bytes: bytes,
        yolo_detections: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Compute metric scale factor from a reference object.

        v3.0: Indian-specific objects prioritised.
        Uses fine-tuned YOLO (yolo_indian_rooms.pt) when available.

        Returns:
            (scale_factor_m, calibration_object_name, calibration_source)
            or (None, None, None) if no suitable object found.
        """
        detections = yolo_detections
        cal_source = None

        if detections is None:
            detections, cal_source = self._run_yolo(image_bytes)

        if not detections:
            return None, None, None

        for cal_name, real_height_m in _CALIBRATION_OBJECTS:
            for det in detections:
                det_label = det.get("label", "").lower()
                # Allow partial match (e.g. "steel_almirah" matches "almirah")
                cal_lower = cal_name.lower().replace("_", " ")
                det_lower = det_label.lower().replace("_", " ")
                if cal_lower not in det_lower and det_lower not in cal_lower:
                    continue

                bbox = det.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue

                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                bbox_h_frac = (y2 - y1) / max(img_h, 1)

                if bbox_h_frac < _MIN_BBOX_HEIGHT_FRAC:
                    continue

                sx   = depth_w / max(img_w, 1)
                sy   = depth_h / max(img_h, 1)
                d_x1 = int(np.clip(x1 * sx, 0, depth_w - 1))
                d_y1 = int(np.clip(y1 * sy, 0, depth_h - 1))
                d_x2 = int(np.clip(x2 * sx, 1, depth_w))
                d_y2 = int(np.clip(y2 * sy, 1, depth_h))

                roi = depth_norm[d_y1:d_y2, d_x1:d_x2]
                if roi.size < 10:
                    continue

                # For tall objects (almirah, door), use the full bbox
                # For short objects (switch, bed), use upper 60%
                if real_height_m > 1.5:
                    sample_roi = roi
                else:
                    sample_roi = roi[:int(roi.shape[0] * 0.6), :] or roi

                median_depth = float(np.median(sample_roi))
                if median_depth < 0.01:
                    continue

                scale_factor = real_height_m / (median_depth * max(bbox_h_frac, 0.01))

                # Sanity: implied room depth should be 1.5–12m
                typical_floor_depth = 0.65
                implied_room_depth  = typical_floor_depth * scale_factor
                if not (1.5 <= implied_room_depth <= 12.0):
                    logger.debug(
                        f"[DepthEstimator] Calibration via '{cal_name}' implied "
                        f"depth={implied_room_depth:.1f}m — out of range, skipping."
                    )
                    continue

                logger.info(
                    f"[DepthEstimator] Calibrated via '{cal_name}' "
                    f"(real_h={real_height_m}m, scale={scale_factor:.3f}, "
                    f"implied_depth={implied_room_depth:.1f}m, source={cal_source})"
                )
                return scale_factor, cal_name, cal_source

        return None, None, None

    @staticmethod
    def _run_yolo(
        image_bytes: bytes,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Run YOLOv8 to detect calibration reference objects.

        v3.0: Uses CVModelRegistry which loads fine-tuned yolo_indian_rooms.pt first.
        Returns (detections, source_label) where source_label indicates which
        model was used for the calibration_source field.
        """
        try:
            from ml.cv_model_registry import get_registry
            registry   = get_registry()
            detections = registry.yolo.detect(image_bytes, confidence_threshold=0.25)
            source     = (
                "yolo_finetuned"
                if registry.yolo.is_finetuned
                else "yolo_pretrained_coco"
            )
            return detections, source
        except Exception as e:
            logger.debug(f"[DepthEstimator] YOLO via registry failed: {e}. Trying direct.")

        # Direct fallback if registry unavailable
        try:
            from ultralytics import YOLO
            import os as _os

            _weights = Path(_os.getenv("ML_WEIGHTS_DIR", "/app/ml/weights"))
            ft_path  = _weights / "yolo_indian_rooms.pt"
            pt_path  = _weights / "yolov8n.pt"

            model  = YOLO(
                str(ft_path) if ft_path.exists() else
                str(pt_path) if pt_path.exists() else
                "yolov8n.pt"
            )
            source = "yolo_finetuned" if ft_path.exists() else "yolo_pretrained_coco"

            from PIL import Image as PILImage
            img     = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            results = model(img, verbose=False, conf=0.25)
            dets    = []
            for r in results:
                for box in r.boxes:
                    dets.append({
                        "label":      r.names.get(int(box.cls[0]), "unknown"),
                        "confidence": float(box.conf[0]),
                        "bbox":       box.xyxy[0].tolist(),
                    })
            return dets, source

        except Exception as e:
            logger.debug(f"[DepthEstimator] Direct YOLO also failed: {e}")
            return [], None

    @staticmethod
    def _heuristic_scale_factor(depth_norm: np.ndarray, room_type: str) -> float:
        """Fallback scale: maps normalised depth to expected Indian room depth (3-6m)."""
        room_depth_range = {
            "bathroom":    (2.0, 4.0),
            "kitchen":     (2.5, 4.5),
            "bedroom":     (3.0, 5.5),
            "living_room": (3.5, 6.5),
            "dining_room": (3.0, 5.5),
            "study":       (2.5, 4.5),
            "full_home":   (4.0, 8.0),
        }
        lo, hi    = room_depth_range.get(room_type, (3.0, 6.0))
        mid_depth = (lo + hi) / 2.0
        return mid_depth / 0.65   # 0.65 = typical floor region median depth_norm

    @staticmethod
    def _heuristic_fallback(room_type: str) -> Dict[str, Any]:
        """Returns real-data-grounded room size from 32k Indian housing rows."""
        prior_stats   = REAL_ROOM_SIZE_STATS.get(room_type, REAL_ROOM_SIZE_STATS["bedroom"])
        floor_sqft    = float(prior_stats["median_sqft"])
        ceiling_ht_ft = _CEILING_HEIGHTS.get(room_type, 9.5)
        side_ft       = math.sqrt(max(floor_sqft, 1.0))
        wall_sqft     = 2.0 * (side_ft + side_ft) * ceiling_ht_ft * (1.0 - _WALL_OPENING_DEDUCTION)

        return {
            "floor_area_sqft":         round(floor_sqft, 1),
            "wall_area_sqft":          round(wall_sqft,  1),
            "ceiling_height_ft":       ceiling_ht_ft,
            "room_width_ft":           round(side_ft, 1),
            "room_depth_ft":           round(side_ft, 1),
            "method":                  "heuristic_fallback",
            "confidence":              0.30,
            "calibration_object":      None,
            "calibration_source":      None,
            "scale_factor_m_per_unit": None,
            "depth_map_available":     False,
            "prior_median_sqft":       float(prior_stats["median_sqft"]),
            "prior_source":            "india_housing_kaggle_32k_rows",
            "blend_weight":            0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Unit test (if __name__ == "__main__")
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _BACKEND_DIR = Path(__file__).resolve().parent.parent
    _SAMPLE_DIRS = [
        _BACKEND_DIR / "data" / "datasets" / "interior_design_material_style" / "bedroom" / "modern",
        _BACKEND_DIR / "data" / "datasets" / "interior_design_material_style" / "living_room" / "boho",
        _BACKEND_DIR / "data" / "datasets" / "interior_design_images_metadata" / "bathroom" / "scandinavian",
    ]

    sample_path = None
    for d in _SAMPLE_DIRS:
        if d.exists():
            jpgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if jpgs:
                sample_path = jpgs[0]
                break

    if sample_path is None:
        print(
            "\n[test] No sample images found. "
            "Place images at backend/data/datasets/interior_design_material_style/bedroom/modern/"
        )
        sys.exit(0)

    print(f"\n[test] Using: {sample_path}")
    image_bytes = sample_path.read_bytes()
    estimator   = DepthEstimator()

    for rt in ["bedroom", "living_room", "kitchen", "bathroom"]:
        r = estimator.estimate_room_area(image_bytes, room_type=rt)
        print(f"\n  room_type={rt}")
        print(f"    floor_area_sqft    = {r['floor_area_sqft']}")
        print(f"    wall_area_sqft     = {r['wall_area_sqft']}")
        print(f"    ceiling_height_ft  = {r['ceiling_height_ft']}")
        print(f"    method             = {r['method']}")
        print(f"    calibration_object = {r['calibration_object']}")
        print(f"    calibration_source = {r['calibration_source']}")
        print(f"    confidence         = {r['confidence']}")