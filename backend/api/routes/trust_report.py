"""
ARKEN — Trust Check API Router v1.0
=====================================
FastAPI router exposing GET /api/trust-check

Returns a real-time assessment of ARKEN deployment readiness:
  - Which ML models are loaded and operational
  - RAG knowledge base population status
  - Training dataset sizes
  - Overall readiness score and classification

Usage in main FastAPI app:
    from api.routes.trust_report import router as trust_router
    app.include_router(trust_router)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["trust"])

# ── Weights directory ─────────────────────────────────────────────────────────
_WEIGHTS_DIR = Path(os.getenv("ML_WEIGHTS_DIR", "/app/ml/weights"))
_CHROMA_DIR  = os.getenv("CHROMA_PERSIST_DIR", "/tmp/arken_chroma")


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks — each wrapped in try/except
# ─────────────────────────────────────────────────────────────────────────────

def _check_price_model() -> Dict[str, Any]:
    """Check PriceForecastAgent ML model status."""
    try:
        import json
        report_path = _WEIGHTS_DIR / "price_model_report.json"
        if not report_path.exists():
            # Check for XGB model file as proxy
            xgb_path = _WEIGHTS_DIR / "price_xgb.joblib"
            if xgb_path.exists():
                return {
                    "status":    "real_ml",
                    "model":     "xgboost_loaded",
                    "rows":      0,
                    "note":      "XGBoost model present; no model_report.json found",
                }
            return {"status": "seed_fallback", "model": "none", "rows": 0}
        with open(str(report_path), "r", encoding="utf-8") as fh:
            rpt = json.load(fh)
        model_used = rpt.get("ml_model_used", rpt.get("model_versions", {}) and "prophet+xgboost")
        rows       = int(rpt.get("training_rows", rpt.get("dataset_size", 0)))
        return {
            "status":   "real_ml" if rows > 0 else "seed_fallback",
            "model":    str(model_used or "unknown"),
            "rows":     rows,
            "mae":      rpt.get("mae"),
            "trained":  rpt.get("training_date", "unknown"),
        }
    except Exception as e:
        logger.debug(f"[trust_check] price_model check error: {e}")
        xgb_path = _WEIGHTS_DIR / "price_xgb.joblib"
        if xgb_path.exists():
            return {"status": "real_ml", "model": "xgboost_file_present", "rows": 0}
        return {"status": "seed_fallback", "model": "none", "rows": 0, "error": str(e)}


def _check_roi_model() -> Dict[str, Any]:
    """Check ROIModel ensemble status."""
    try:
        import json
        report_path = _WEIGHTS_DIR / "model_report.json"
        if not report_path.exists():
            # Check individual model files
            files_present = [
                (_WEIGHTS_DIR / "roi_xgb.joblib").exists(),
                (_WEIGHTS_DIR / "roi_rf.joblib").exists(),
                (_WEIGHTS_DIR / "roi_gbm.joblib").exists(),
            ]
            n_present = sum(files_present)
            if n_present == 3:
                return {"status": "ensemble", "models": 3, "rows": 0, "note": "No model_report.json"}
            elif n_present > 0:
                return {"status": "xgboost", "models": n_present, "rows": 0}
            return {"status": "heuristic", "models": 0, "rows": 0}
        with open(str(report_path), "r", encoding="utf-8") as fh:
            rpt = json.load(fh)
        n_models = int(rpt.get("n_models", rpt.get("n_models_in_ensemble", 0)))
        rows     = int(rpt.get("dataset_size", 0))
        status   = "ensemble" if n_models >= 3 else "xgboost" if n_models >= 1 else "heuristic"
        return {
            "status":  status,
            "models":  n_models,
            "rows":    rows,
            "mae":     rpt.get("mae", rpt.get("test_mae")),
            "r2":      rpt.get("r2", rpt.get("test_r2")),
            "trained": rpt.get("training_date", "unknown"),
        }
    except Exception as e:
        logger.debug(f"[trust_check] roi_model check error: {e}")
        return {"status": "heuristic", "models": 0, "rows": 0, "error": str(e)}


def _check_depth_model() -> Dict[str, Any]:
    """Check if DepthAnything model can be loaded."""
    try:
        from ml.depth_estimator import DepthEstimator
        de = DepthEstimator()
        # Check if pipeline is already loaded
        if de.__class__._pipeline is not None:
            return {"status": "loaded", "variant": de.__class__._model_used}
        # Try to load
        from transformers import pipeline as hf_pipeline
        return {"status": "available", "variant": "not_yet_loaded"}
    except ImportError:
        return {"status": "unavailable", "reason": "transformers not installed"}
    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}


def _check_damage_model() -> Dict[str, Any]:
    """Check if ResNet50 damage detector can be loaded."""
    try:
        import torch
        from ml.damage_detector import DamageDetector
        dd = DamageDetector()
        if dd.__class__._model_ready:
            return {"status": "loaded", "model": "resnet50_transfer"}
        return {"status": "available", "model": "resnet50_not_yet_loaded"}
    except ImportError:
        return {"status": "unavailable", "reason": "torch/torchvision not installed"}
    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}


def _check_style_model() -> Dict[str, Any]:
    """Check if CLIP style classifier is available."""
    try:
        from ml.style_classifier import StyleClassifier
        sc = StyleClassifier()
        if sc.__class__._clip_ready:
            return {"status": "clip_loaded"}
        from transformers import CLIPModel
        return {"status": "clip_available_not_loaded"}
    except ImportError:
        return {"status": "keyword_rules", "reason": "transformers/torch not installed"}
    except Exception as e:
        return {"status": "keyword_rules", "reason": str(e)}


def _check_rag_chunks() -> Dict[str, Any]:
    """Check ChromaDB knowledge chunk count."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=_CHROMA_DIR)
        try:
            collection = client.get_collection("arken_knowledge_v2")
            count = collection.count()
        except Exception:
            count = 0
        return {"count": count, "seeded": count >= 50, "path": _CHROMA_DIR}
    except ImportError:
        return {"count": 0, "seeded": False, "reason": "chromadb not installed"}
    except Exception as e:
        return {"count": 0, "seeded": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Readiness scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_readiness(
    price: Dict, roi: Dict, depth: Dict, damage: Dict, style: Dict, rag: Dict,
) -> Dict[str, Any]:
    score_parts: List[float] = []
    missing: List[str] = []

    # Price model (0.25 weight)
    if price.get("status") == "real_ml" and price.get("rows", 0) > 100:
        score_parts.append(1.0)
    elif price.get("status") == "real_ml":
        score_parts.append(0.70)
        missing.append("Price training data <100 rows — run train_price_models.py with real CSV")
    else:
        score_parts.append(0.20)
        missing.append("Price model using seed fallback — add india_material_prices_historical.csv")

    # ROI model (0.25 weight)
    if roi.get("status") == "ensemble" and roi.get("rows", 0) >= 1000:
        score_parts.append(1.0)
    elif roi.get("status") in ("ensemble", "xgboost") and roi.get("rows", 0) > 0:
        score_parts.append(0.80)
    elif roi.get("status") in ("ensemble", "xgboost"):
        score_parts.append(0.65)
        missing.append("ROI model lacks training rows — run train_roi_models.py with real CSV")
    else:
        score_parts.append(0.30)
        missing.append("ROI model using heuristic — run train_roi_models.py")

    # Depth model (0.15 weight)
    if depth.get("status") in ("loaded", "available", "clip_loaded", "clip_available_not_loaded"):
        score_parts.append(0.85)
    else:
        score_parts.append(0.30)
        missing.append("Depth model unavailable — pip install transformers torch torchvision")

    # Damage model (0.15 weight)
    if damage.get("status") in ("loaded", "available"):
        score_parts.append(0.85)
    else:
        score_parts.append(0.35)
        missing.append("Damage detector unavailable — pip install torch torchvision")

    # Style model (0.10 weight)
    if style.get("status") in ("clip_loaded", "clip_available_not_loaded"):
        score_parts.append(0.90)
    else:
        score_parts.append(0.40)
        missing.append("Style classifier using keyword rules — pip install transformers")

    # RAG (0.10 weight)
    if rag.get("count", 0) >= 300:
        score_parts.append(1.0)
    elif rag.get("count", 0) >= 50:
        score_parts.append(0.70)
        missing.append(f"RAG only has {rag['count']} chunks — re-run seed_knowledge.py for full 300+")
    else:
        score_parts.append(0.10)
        missing.append("RAG knowledge base empty — run python backend/data/rag_knowledge_base/seed_knowledge.py")

    weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
    readiness_score = sum(s * w for s, w in zip(score_parts, weights))

    if readiness_score >= 0.85:
        overall = "production_ready"
    elif readiness_score >= 0.55:
        overall = "partial"
    else:
        overall = "demo_only"

    return {
        "overall_readiness": overall,
        "readiness_score":   round(readiness_score, 3),
        "missing_for_production": missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Route
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/trust-check", summary="ARKEN Deployment Readiness Check")
async def trust_check() -> Dict[str, Any]:
    """
    Returns a real-time trust assessment of the current ARKEN deployment.

    Checks: price ML model, ROI ensemble, depth estimator, damage detector,
    style classifier, and RAG knowledge base chunk count.

    Every check is wrapped in try/except — no single failure blocks the response.
    """
    now = datetime.now(tz=timezone.utc).isoformat()

    price  = _check_price_model()
    roi    = _check_roi_model()
    depth  = _check_depth_model()
    damage = _check_damage_model()
    style  = _check_style_model()
    rag    = _check_rag_chunks()

    readiness = _compute_readiness(price, roi, depth, damage, style, rag)

    return {
        "deployment_status": {
            "price_model":          price.get("status", "unknown"),
            "price_model_detail":   price,
            "roi_model":            roi.get("status", "unknown"),
            "roi_model_detail":     roi,
            "depth_model":          depth.get("status", "unavailable"),
            "damage_model":         damage.get("status", "unavailable"),
            "style_model":          style.get("status", "keyword_rules"),
            "rag_chunks":           rag.get("count", 0),
            "rag_seeded":           rag.get("seeded", False),
            "price_training_rows":  price.get("rows", 0),
            "roi_training_rows":    roi.get("rows", 0),
        },
        "overall_readiness":        readiness["overall_readiness"],
        "readiness_score":          readiness["readiness_score"],
        "missing_for_production":   readiness["missing_for_production"],
        "last_model_check":         now,
    }
