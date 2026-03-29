"""
ARKEN ML Package
================
Computer vision and property modelling modules.

All imports are **lazy** — no heavy DL dependencies are loaded at import time.
Modules degrade gracefully when optional packages (torch, ultralytics,
transformers) are absent: each class returns sensible fallback dicts and logs
a warning rather than raising ImportError.

Public API
----------
CV pipeline (vision analysis):
    from ml.depth_estimator     import DepthEstimator
    from ml.damage_detector     import DamageDetector
    from ml.style_classifier    import StyleClassifier
    from ml.cv_feature_extractor import CVFeatureExtractor, get_extractor
    from ml.cv_model_registry   import get_registry

Property / ROI modelling:
    from ml.housing_preprocessor import get_reno_preprocessor, RENO_COST_BENCHMARKS
    from ml.property_models      import ROIModel, RenovationCostModel, PropertyValueModel
    from ml.confidence_calibrator import ConfidenceCalibrator

Lazy import helpers (for top-level ``from ml import X`` style):
    DepthEstimator, DamageDetector, StyleClassifier,
    CVFeatureExtractor, get_extractor
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version & capability flags
# ---------------------------------------------------------------------------
__version__ = "3.0.0"

# Populated on first successful import of each heavy dep
_TORCH_AVAILABLE: bool | None = None
_ULTRALYTICS_AVAILABLE: bool | None = None
_TRANSFORMERS_AVAILABLE: bool | None = None


def _check_torch() -> bool:
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _check_ultralytics() -> bool:
    global _ULTRALYTICS_AVAILABLE
    if _ULTRALYTICS_AVAILABLE is None:
        try:
            import ultralytics  # noqa: F401
            _ULTRALYTICS_AVAILABLE = True
        except ImportError:
            _ULTRALYTICS_AVAILABLE = False
    return _ULTRALYTICS_AVAILABLE


def _check_transformers() -> bool:
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE is None:
        try:
            import transformers  # noqa: F401
            _TRANSFORMERS_AVAILABLE = True
        except ImportError:
            _TRANSFORMERS_AVAILABLE = False
    return _TRANSFORMERS_AVAILABLE


def capabilities() -> dict:
    """
    Return a dict describing which optional DL backends are available.

    Example::

        from ml import capabilities
        caps = capabilities()
        # {'torch': True, 'ultralytics': False, 'transformers': True,
        #  'yolo_available': False, 'clip_available': True, 'depth_available': True}
    """
    torch_ok        = _check_torch()
    ultralytics_ok  = _check_ultralytics()
    transformers_ok = _check_transformers()
    return {
        "torch":             torch_ok,
        "ultralytics":       ultralytics_ok,
        "transformers":      transformers_ok,
        "yolo_available":    ultralytics_ok,
        "clip_available":    transformers_ok and torch_ok,
        "depth_available":   transformers_ok,
        "resnet_available":  torch_ok,
    }


# ---------------------------------------------------------------------------
# Lazy module loader
# ---------------------------------------------------------------------------

def _lazy_import(module_path: str, attr: str) -> Any:
    """
    Import ``attr`` from ``module_path`` at call time (not at package import).

    Returns None and logs a warning if the import fails due to a missing
    optional dependency — callers should guard with ``if X is not None``.
    """
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    except ImportError as e:
        logger.warning(
            f"[ml] Could not import {attr} from {module_path}: {e}. "
            "Install optional dependencies: pip install torch torchvision "
            "transformers ultralytics"
        )
        return None
    except AttributeError as e:
        logger.error(f"[ml] {module_path} exists but is missing {attr}: {e}")
        return None


# ---------------------------------------------------------------------------
# Convenience top-level lazy attributes
# (``from ml import DepthEstimator`` works even if torch is not installed)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """
    Module-level ``__getattr__`` for lazy class resolution.

    Supports::

        from ml import DepthEstimator
        from ml import DamageDetector
        from ml import StyleClassifier
        from ml import CVFeatureExtractor
        from ml import get_extractor
        from ml import ROIModel
        from ml import RenovationCostModel

    Each import is deferred until first access; the module cache means
    subsequent accesses are free.
    """
    _LAZY_MAP: dict[str, tuple[str, str]] = {
        # CV pipeline
        "DepthEstimator":         ("ml.depth_estimator",      "DepthEstimator"),
        "DamageDetector":         ("ml.damage_detector",       "DamageDetector"),
        "StyleClassifier":        ("ml.style_classifier",      "StyleClassifier"),
        "CVFeatureExtractor":     ("ml.cv_feature_extractor",  "CVFeatureExtractor"),
        "CVFeatures":             ("ml.cv_feature_extractor",  "CVFeatures"),
        "get_extractor":          ("ml.cv_feature_extractor",  "get_extractor"),
        "get_registry":           ("ml.cv_model_registry",     "get_registry"),
        # Property / ROI
        "ROIModel":               ("ml.property_models",       "ROIModel"),
        "RenovationCostModel":    ("ml.property_models",       "RenovationCostModel"),
        "PropertyValueModel":     ("ml.property_models",       "PropertyValueModel"),
        "get_reno_preprocessor":  ("ml.housing_preprocessor",  "get_reno_preprocessor"),
        "get_preprocessor":       ("ml.housing_preprocessor",  "get_preprocessor"),
        "RENO_COST_BENCHMARKS":   ("ml.housing_preprocessor",  "RENO_COST_BENCHMARKS"),
        "ConfidenceCalibrator":   ("ml.confidence_calibrator", "ConfidenceCalibrator"),
    }

    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        obj = _lazy_import(module_path, attr)
        # Cache in module globals so next access is free
        if obj is not None:
            globals()[name] = obj
        return obj

    raise AttributeError(f"module 'ml' has no attribute {name!r}")
