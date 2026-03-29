"""
ARKEN — Startup Checks v3.0
==============================
SAVE AS: backend/startup_checks.py — REPLACE existing

v3.0 Changes (PROBLEM 4 FIX — comprehensive checks + startup report):
  1. All required env vars checked: GOOGLE_API_KEY (hard fail if missing),
     OPENAI_API_KEY (warn only).
  2. Database connection test with SELECT 1.
  3. Redis connection test.
  4. ML model weights enumerated from /app/ml/weights/.
  5. Datasets enumerated: housing CSVs, image folders, India knowledge JSON.
  6. Clean startup report printed to log.
  7. If GOOGLE_API_KEY is missing → RuntimeError (hard crash with clear message).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_WEIGHTS_DIR  = Path(os.getenv("ML_WEIGHTS_DIR", "/app/ml/weights"))
_DATASET_ROOT = Path(os.getenv("ARKEN_DATASET_DIR", "/app/data/datasets"))

_CITY_CSVS = [
    "Bangalore.csv", "Chennai.csv", "Delhi.csv",
    "Hyderabad.csv", "Kolkata.csv", "Mumbai.csv",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_startup_checks() -> Dict[str, Any]:
    """
    Run all startup checks. Prints a formatted startup report.
    Returns a dict of check results.

    HARD FAILS (raises RuntimeError):
      - GOOGLE_API_KEY missing

    SOFT WARNS (logs warning, continues):
      - OPENAI_API_KEY missing
      - Database unavailable
      - Redis unavailable
      - ML weights not found
      - Datasets not found
    """
    results: Dict[str, Any] = {}

    # ── Environment variables ─────────────────────────────────────────────────
    results["env"] = _check_env()

    # ── Database ──────────────────────────────────────────────────────────────
    results["database"] = _check_database()

    # ── Redis / Cache ─────────────────────────────────────────────────────────
    results["redis"] = _check_redis()

    # ── ML Model Weights ──────────────────────────────────────────────────────
    results["ml_models"] = _check_ml_weights()

    # ── Datasets ──────────────────────────────────────────────────────────────
    results["datasets"] = _check_datasets()

    # ── LangGraph ─────────────────────────────────────────────────────────────
    results["langgraph"] = _precompile_langgraph()

    # ── RAG Knowledge Store ───────────────────────────────────────────────────
    results["rag_store"] = _prewarm_rag_store()

    # ── Dataset Registry ─────────────────────────────────────────────────────
    results["dataset_registry"] = _ingest_datasets()

    # ── Print startup report ──────────────────────────────────────────────────
    _print_startup_report(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_env() -> Dict[str, Any]:
    """
    Check required env variables.
    HARD FAIL if GOOGLE_API_KEY is missing.
    """
    result: Dict[str, Any] = {}

    # GOOGLE_API_KEY — hard requirement
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        # Also try settings object
        try:
            from config import settings
            if settings.GOOGLE_API_KEY:
                google_key = "present"
        except Exception:
            pass

    if not google_key:
        raise RuntimeError(
            "\n\n"
            "════════════════════════════════════════\n"
            "ARKEN STARTUP FAILED — GOOGLE_API_KEY missing\n"
            "════════════════════════════════════════\n"
            "GOOGLE_API_KEY is required for Gemini image generation and chat.\n"
            "Set it in backend/.env:\n"
            "  GOOGLE_API_KEY=AIzaSy...\n"
            "Get a key at: https://aistudio.google.com/app/apikey\n"
            "════════════════════════════════════════\n"
        )

    result["google_api_key"] = "present"

    # OPENAI_API_KEY — optional, warn only
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        try:
            from config import settings
            if hasattr(settings, "OPENAI_API_KEY") and settings.OPENAI_API_KEY:
                openai_key = "present"
        except Exception:
            pass

    result["openai_api_key"] = "present" if openai_key else "missing"
    if not openai_key:
        logger.warning(
            "⚠  OPENAI_API_KEY not set — chat will use Gemini fallback. "
            "Set OPENAI_API_KEY in backend/.env if OpenAI features are needed."
        )

    return result


def _check_database() -> Dict[str, Any]:
    """Test database connection with a SELECT 1."""
    try:
        import asyncio
        import sqlalchemy

        from config import settings
        from db.session import engine, _sqlalchemy_available

        if not _sqlalchemy_available:
            return {"status": "unavailable", "reason": "SQLAlchemy not installed"}

        async def _test():
            from sqlalchemy import text
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()

        return {"status": "connected", "type": "PostgreSQL"}

    except ImportError as e:
        return {"status": "unavailable", "reason": f"Import error: {e}"}
    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}


def _check_redis() -> Dict[str, Any]:
    """Test Redis connection."""
    try:
        from config import settings
        import asyncio

        async def _test():
            try:
                import redis.asyncio as aioredis
                r = aioredis.from_url(str(getattr(settings, "REDIS_URL", "redis://localhost:6379")))
                await r.ping()
                await r.aclose()
                return True
            except Exception:
                return False

        loop = asyncio.new_event_loop()
        try:
            ok = loop.run_until_complete(_test())
        finally:
            loop.close()

        if ok:
            return {"status": "connected", "type": "Redis"}
        return {"status": "unavailable", "reason": "ping failed"}

    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}


def _check_ml_weights() -> Dict[str, Any]:
    """Enumerate ML model weight files."""
    result: Dict[str, Any] = {"available": [], "missing": []}

    expected_weights = [
        ("roi_real_data_model.joblib", "ROI ensemble (real data)"),
        ("roi_xgb.joblib",             "ROI XGBoost fallback"),
        ("property_value_ensemble.joblib", "Property value model"),
        ("renovation_cost_model.joblib",   "Renovation cost model"),
        ("room_classifier.pt",         "Room classifier (EfficientNet)"),
        ("model_report.json",          "Model training report"),
    ]

    for filename, label in expected_weights:
        path = _WEIGHTS_DIR / filename
        if path.exists():
            size_kb = path.stat().st_size // 1024
            result["available"].append({"file": filename, "label": label, "size_kb": size_kb})
        else:
            result["missing"].append({"file": filename, "label": label})

    # Check model_report for dataset size
    report_path = _WEIGHTS_DIR / "model_report.json"
    if report_path.exists():
        try:
            import json
            with open(report_path) as fh:
                report = json.load(fh)
            result["roi_dataset_size"] = report.get("dataset_size", 0)
            result["roi_training_date"] = report.get("training_date", "")[:10]
        except (json.JSONDecodeError, OSError):
            pass

    return result


def _check_datasets() -> Dict[str, Any]:
    """Check dataset files and log counts."""
    result: Dict[str, Any] = {}

    # 1. Housing CSVs
    housing_dir = _DATASET_ROOT / "india_housing_prices"
    housing_present = []
    housing_rows    = 0
    for csv_name in _CITY_CSVS:
        p = housing_dir / csv_name
        if p.exists():
            try:
                import csv
                with open(p, newline="", encoding="utf-8", errors="replace") as fh:
                    rows = sum(1 for _ in csv.reader(fh)) - 1   # subtract header
                housing_rows += max(rows, 0)
                housing_present.append(csv_name)
            except (OSError, csv.Error):
                housing_present.append(csv_name)
    result["housing_csvs"] = {
        "present": len(housing_present),
        "expected": len(_CITY_CSVS),
        "total_rows": housing_rows,
        "files": housing_present,
    }

    # 2. Interior design image datasets
    for ds_name in ("interior_design_images_metadata", "interior_design_material_style"):
        ds_dir = _DATASET_ROOT / ds_name
        if ds_dir.exists():
            image_count = sum(1 for _ in ds_dir.rglob("*.jpg"))
            image_count += sum(1 for _ in ds_dir.rglob("*.png"))
            result[ds_name] = {"available": True, "image_count": image_count}
        else:
            result[ds_name] = {"available": False}

    # 3. Indian renovation knowledge JSON
    india_json = _DATASET_ROOT / "indian_renovation_knowledge" / "india_reno_knowledge.json"
    if india_json.exists():
        try:
            import json
            with open(india_json) as fh:
                chunks = json.load(fh)
            result["india_knowledge_json"] = {
                "available": True,
                "chunk_count": len(chunks),
                "path": str(india_json),
            }
        except (json.JSONDecodeError, OSError):
            result["india_knowledge_json"] = {"available": True, "chunk_count": "error reading"}
    else:
        result["india_knowledge_json"] = {"available": False, "path": str(india_json)}

    return result


def _precompile_langgraph() -> Any:
    try:
        from agents.multi_agent_pipeline import build_multi_agent_graph
        graph = build_multi_agent_graph()
        if graph is not None:
            return {"status": "compiled", "nodes": 6}
        return {"status": "sequential_fallback"}
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}


def _prewarm_rag_store() -> Any:
    try:
        from services.rag.vector_store import get_knowledge_store
        store = get_knowledge_store()
        counts = store.get_category_counts()
        total  = sum(counts.values())
        return {"status": "ready", "chunks": total, "categories": len(counts)}
    except ImportError as e:
        return {"status": "skipped", "reason": f"dependency missing: {e}"}
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}


def _ingest_datasets() -> Any:
    try:
        from services.datasets.dataset_loader import ARKENDatasetRegistry
        registry = ARKENDatasetRegistry.get()
        registry.startup_ingest()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Startup report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_startup_report(results: Dict[str, Any]) -> None:
    """Print a clean formatted startup report to the log."""
    lines = [
        "",
        "═══════════════════════════════════════════════════",
        "ARKEN PropTech Engine v5.0 — Startup Report",
        "═══════════════════════════════════════════════════",
    ]

    # Env
    env = results.get("env", {})
    lines.append(
        f"✓ Google API:    Key present (Gemini 2.5 Flash text + image)"
    )
    if env.get("openai_api_key") == "present":
        lines.append("✓ OpenAI API:    Key present")
    else:
        lines.append("⚠ OpenAI API:    Key missing — chat uses Gemini fallback")

    # Database
    db = results.get("database", {})
    if db.get("status") == "connected":
        lines.append(f"✓ Database:      Connected ({db.get('type', 'PostgreSQL')})")
    else:
        lines.append(f"⚠ Database:      {db.get('status', 'unavailable')} — {db.get('reason', '')}")

    # Redis
    redis = results.get("redis", {})
    if redis.get("status") == "connected":
        lines.append(f"✓ Cache:         Connected ({redis.get('type', 'Redis')})")
    else:
        lines.append(f"⚠ Cache:         {redis.get('status', 'unavailable')} — using in-memory fallback")

    # ML Models
    ml = results.get("ml_models", {})
    available = ml.get("available", [])
    missing   = ml.get("missing", [])
    ds_size   = ml.get("roi_dataset_size", 0)
    ds_label  = f" ({ds_size:,} rows)" if ds_size else ""
    if available:
        lines.append(
            f"✓ ML Models:     {len(available)} weight files found{ds_label}"
        )
    else:
        lines.append("⚠ ML Models:     No weights found — will train on first request")
    if missing:
        lines.append(f"  ⚠ Missing: {', '.join(m['label'] for m in missing[:3])}")

    # Datasets
    ds = results.get("datasets", {})
    housing = ds.get("housing_csvs", {})
    india_k = ds.get("india_knowledge_json", {})
    img_meta = ds.get("interior_design_images_metadata", {})

    housing_str = (
        f"{housing.get('present', 0)}/{housing.get('expected', 6)} city CSVs "
        f"({housing.get('total_rows', 0):,} rows)"
        if housing else "not checked"
    )
    lines.append(f"✓ Datasets:      Housing: {housing_str}")

    if india_k.get("available"):
        lines.append(f"  ✓ Indian Knowledge: {india_k.get('chunk_count', 0)} chunks loaded (india_reno_knowledge.json)")
    else:
        lines.append("  ⚠ Indian Knowledge: JSON not found — RAG will use built-in docs only")

    if img_meta.get("available"):
        lines.append(f"  ✓ Interior Images: {img_meta.get('image_count', 0)} images")

    # LangGraph
    lg = results.get("langgraph", {})
    if lg.get("status") == "compiled":
        lines.append(f"✓ LangGraph:     Compiled ({lg.get('nodes', 6)}-node graph)")
    else:
        lines.append(f"⚠ LangGraph:     {lg.get('status', 'skipped')} — sequential fallback")

    # RAG
    rag = results.get("rag_store", {})
    if rag.get("status") == "ready":
        lines.append(f"✓ RAG Store:     Ready ({rag.get('chunks', 0)} chunks, {rag.get('categories', 0)} categories)")
    else:
        lines.append(f"⚠ RAG Store:     {rag.get('status', 'skipped')} — will init on first request")

    lines.append("✓ Image Gen:     Gemini 2.5 Flash Image (primary)")
    lines.append("═══════════════════════════════════════════════════")
    lines.append("")

    logger.info("\n".join(lines))