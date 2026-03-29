"""
ARKEN — Data Freshness Checker v1.0
=====================================
Monitors the health and freshness of all data sources that ML models
and the RAG system depend on. Consumed by GET /health/data.

Checks performed:
  1. Material prices CSV   — last date, row count, staleness
  2. Property transactions CSV — same schema
  3. ML model .joblib files — file modification time, retraining recommendation
  4. ChromaDB corpus        — chunk count, domain coverage
  5. Overall health score (0–100)

Design:
  - All file paths derived from environment variables or backend-relative paths.
  - No external network calls — purely local filesystem and SQLite inspection.
  - Safe to import at module level; expensive checks are lazy.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Paths (all relative to backend root, with /app/ override for Docker) ──────
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # services/monitoring/../../ = backend/
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))

# Prefer /app/ layout (Docker); fall back to local dev layout
def _resolve(app_rel: str, local_rel: str) -> Path:
    """Try /app/<app_rel> first, then <backend_dir>/<local_rel>."""
    app_path   = _APP_DIR / app_rel
    local_path = _BACKEND_DIR / local_rel
    return app_path if app_path.exists() else local_path

_MATERIAL_CSV = _resolve(
    "data/datasets/material_prices/india_material_prices_historical.csv",
    "data/datasets/material_prices/india_material_prices_historical.csv",
)
_PROPERTY_CSV = _resolve(
    "data/datasets/property_transactions/india_property_transactions.csv",
    "data/datasets/property_transactions/india_property_transactions.csv",
)
_WEIGHTS_DIR  = _resolve("ml/weights", "ml/weights")
_MODEL_REPORT = _WEIGHTS_DIR / "model_report.json"
_CHROMA_DIR   = Path(os.getenv("CHROMA_PERSIST_DIR", str(_resolve("data/chroma", "data/chroma"))))

# Staleness thresholds (days)
_FRESH_DAYS    = int(os.getenv("DATA_FRESHNESS_ALERT_DAYS", "45"))
_STALE_DAYS    = _FRESH_DAYS * 2
_MODEL_WARN    = 30
_MODEL_CRIT    = 60

# Model files to check
_MODEL_FILES: List[str] = [
    "price_xgb.joblib",
    "roi_xgb.joblib",
    "roi_rf.joblib",
    "roi_gbm.joblib",
    "renovation_cost_model.joblib",
]


def _file_age_days(path: Path) -> Optional[float]:
    """Return age in days, or None if file doesn't exist."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 86400
    except OSError:
        return None


def _staleness_status(days_old: Optional[float]) -> str:
    if days_old is None:
        return "missing"
    if days_old <= _FRESH_DAYS:
        return "fresh"
    if days_old <= _STALE_DAYS:
        return "stale"
    return "very_stale"


class DataFreshnessChecker:
    """
    Checks freshness of all ARKEN data sources and returns structured reports.
    Stateless — every method re-reads from filesystem/DB on each call.
    Safe to instantiate multiple times (lightweight).
    """

    # ── Material prices CSV ───────────────────────────────────────────────────

    def check_material_prices(self) -> Dict[str, Any]:
        """
        Returns:
            status:            "fresh" | "stale" | "very_stale" | "missing"
            last_date:         str (ISO date of most recent row, or "N/A")
            days_old:          int (days since last_date, or -1 if missing)
            rows:              int
            materials_covered: int (distinct material_key values)
            cities_covered:    int (distinct city values)
            alert_message:     str
        """
        return self._check_csv(
            path=_MATERIAL_CSV,
            date_col="date",
            name="Material prices",
            extra_cols=["material_key", "city"],
        )

    # ── Property transactions CSV ─────────────────────────────────────────────

    def check_property_data(self) -> Dict[str, Any]:
        """Same schema as check_material_prices()."""
        return self._check_csv(
            path=_PROPERTY_CSV,
            date_col=None,  # no date col in this CSV; use file mtime
            name="Property transactions",
            extra_cols=["city", "room_renovated"],
        )

    # ── Shared CSV checker ────────────────────────────────────────────────────

    def _check_csv(
        self,
        path: Path,
        date_col: Optional[str],
        name: str,
        extra_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status":            "missing",
            "last_date":         "N/A",
            "days_old":          -1,
            "rows":              0,
            "materials_covered": 0,
            "cities_covered":    0,
            "alert_message":     f"{name} CSV not found.",
            "path":              str(path),
        }

        if not path.exists():
            return result

        try:
            import pandas as pd

            df = pd.read_csv(str(path), nrows=None, low_memory=False)
            rows = len(df)
            result["rows"] = rows

            # Date-based freshness
            if date_col and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                last_date    = df[date_col].max()
                if pd.notna(last_date):
                    days_old = (datetime.now(tz=timezone.utc) - last_date.tz_localize("UTC")).days
                    result["last_date"] = last_date.date().isoformat()
                    result["days_old"]  = days_old
                else:
                    days_old = None
            else:
                # Fall back to file mtime
                age = _file_age_days(path)
                days_old = int(age) if age is not None else None
                if age is not None:
                    last_date_dt = datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    )
                    result["last_date"] = last_date_dt.date().isoformat()
                    result["days_old"]  = int(age)

            # Coverage counts
            if extra_cols:
                for col in extra_cols:
                    if col in df.columns:
                        count = df[col].nunique()
                        if "material" in col:
                            result["materials_covered"] = count
                        elif "city" in col:
                            result["cities_covered"] = count
                        else:
                            result[f"{col}_covered"] = count

            status = _staleness_status(days_old)
            result["status"] = status

            if status == "fresh":
                result["alert_message"] = (
                    f"{name}: {rows:,} rows, up to date ({result['last_date']})."
                )
            elif status == "stale":
                result["alert_message"] = (
                    f"⚠️ {name}: last updated {result['days_old']} days ago. "
                    f"Consider refreshing (threshold: {_FRESH_DAYS} days)."
                )
            else:
                result["alert_message"] = (
                    f"🚨 {name}: critically stale — {result['days_old']} days old. "
                    "Re-generate dataset immediately."
                )

        except ImportError:
            # pandas not available; just check file existence and size
            size = path.stat().st_size
            age  = _file_age_days(path)
            result.update({
                "status":        _staleness_status(age),
                "days_old":      int(age) if age is not None else -1,
                "rows":          -1,  # unknown without pandas
                "alert_message": f"{name}: {size:,} bytes (pandas unavailable for detailed check).",
            })
        except Exception as exc:
            logger.warning(f"[DataFreshnessChecker] {name} check failed: {exc}")
            result["alert_message"] = f"{name}: check error — {exc}"

        return result

    # ── ML model files ────────────────────────────────────────────────────────

    def check_ml_models(self) -> Dict[str, Any]:
        """
        Returns per-model freshness and overall retraining recommendation.

        Keys:
            models:                  {model_name: {exists, age_days, status, recommend_retrain}}
            any_critical:            bool
            any_warning:             bool
            retrain_recommended:     List[str]
            retrain_critical:        List[str]
            model_report_available:  bool
            model_report_summary:    dict | None
        """
        model_statuses: Dict[str, Any] = {}
        retrain_recommended: List[str] = []
        retrain_critical:    List[str] = []

        for fname in _MODEL_FILES:
            fpath = _WEIGHTS_DIR / fname
            age   = _file_age_days(fpath)
            exists = fpath.exists()

            if not exists:
                status = "missing"
                recommend = True
                critical  = True
            elif age is None:
                status = "missing"
                recommend = True
                critical  = True
            elif age > _MODEL_CRIT:
                status    = "critical"
                recommend = True
                critical  = True
            elif age > _MODEL_WARN:
                status    = "warning"
                recommend = True
                critical  = False
            else:
                status    = "ok"
                recommend = False
                critical  = False

            model_statuses[fname] = {
                "exists":           exists,
                "age_days":         round(age, 1) if age is not None else None,
                "status":           status,
                "recommend_retrain": recommend,
                "path":             str(fpath),
            }

            if critical:
                retrain_critical.append(fname)
            elif recommend:
                retrain_recommended.append(fname)

        # Read model_report.json if available
        report_summary = None
        if _MODEL_REPORT.exists():
            try:
                with open(_MODEL_REPORT) as fh:
                    rpt = json.load(fh)
                report_summary = {
                    "training_date":  rpt.get("training_date"),
                    "dataset_size":   rpt.get("dataset_size"),
                    "mae":            rpt.get("mae"),
                    "rmse":           rpt.get("rmse"),
                    "r2":             rpt.get("r2"),
                    "model_versions": list(rpt.get("model_versions", {}).keys()),
                }
            except Exception as exc:
                logger.debug(f"[DataFreshnessChecker] model_report read failed: {exc}")

        return {
            "models":                 model_statuses,
            "any_critical":           bool(retrain_critical),
            "any_warning":            bool(retrain_recommended),
            "retrain_recommended":    retrain_recommended,
            "retrain_critical":       retrain_critical,
            "model_report_available": _MODEL_REPORT.exists(),
            "model_report_summary":   report_summary,
        }

    # ── ChromaDB RAG corpus ───────────────────────────────────────────────────

    def check_rag_corpus(self) -> Dict[str, Any]:
        """
        Returns:
            status:          "healthy" | "sparse" | "empty" | "unavailable"
            chunks:          int
            domains_covered: list[str]
            last_seeded:     str | None  (ISO timestamp if detectable)
            chroma_dir:      str
            alert_message:   str
        """
        result: Dict[str, Any] = {
            "status":          "unavailable",
            "chunks":          0,
            "domains_covered": [],
            "last_seeded":     None,
            "chroma_dir":      str(_CHROMA_DIR),
            "alert_message":   "ChromaDB unavailable.",
        }

        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
            try:
                coll = client.get_collection("arken_knowledge_v2")
            except Exception:
                result["alert_message"] = (
                    "ChromaDB exists but 'arken_knowledge_v2' collection not found. "
                    "Run: python scripts/build_rag_corpus.py"
                )
                return result

            count = coll.count()
            result["chunks"] = count

            # Sample metadata to find domain coverage
            if count > 0:
                try:
                    sample = coll.get(limit=min(count, 500), include=["metadatas"])
                    domains = list({
                        m.get("domain", "unknown")
                        for m in (sample.get("metadatas") or [])
                        if m
                    })
                    result["domains_covered"] = sorted(domains)
                except Exception:
                    result["domains_covered"] = []

            # Health scoring
            if count >= 1000:
                result["status"] = "healthy"
                result["alert_message"] = (
                    f"RAG corpus: {count:,} chunks across "
                    f"{len(result['domains_covered'])} domains."
                )
            elif count >= 100:
                result["status"] = "sparse"
                result["alert_message"] = (
                    f"⚠️ RAG corpus sparse: only {count} chunks. "
                    "Run: python scripts/build_rag_corpus.py"
                )
            else:
                result["status"] = "empty"
                result["alert_message"] = (
                    f"🚨 RAG corpus critically empty: {count} chunks. "
                    "Run: python scripts/build_rag_corpus.py"
                )

            # Approximate last-seeded from ChromaDB dir mtime
            try:
                chroma_age = _file_age_days(_CHROMA_DIR)
                if chroma_age is not None:
                    from datetime import timedelta
                    seeded_dt = datetime.now(tz=timezone.utc) - timedelta(days=chroma_age)
                    result["last_seeded"] = seeded_dt.date().isoformat()
            except Exception:
                pass

        except ImportError:
            result["alert_message"] = "ChromaDB not installed. Run: pip install chromadb"
        except Exception as exc:
            logger.warning(f"[DataFreshnessChecker] RAG corpus check failed: {exc}")
            result["alert_message"] = f"ChromaDB check error: {exc}"

        return result

    # ── Full health report ────────────────────────────────────────────────────

    def get_full_health_report(self) -> Dict[str, Any]:
        """
        Run all 4 checks, compute overall_health_score (0–100), return
        structured report plus a human-readable summary string.
        """
        ts = datetime.now(tz=timezone.utc).isoformat()

        prices   = self.check_material_prices()
        property_ = self.check_property_data()
        models   = self.check_ml_models()
        rag      = self.check_rag_corpus()

        # ── Score computation (0–100) ────────────────────────────────────────
        # Each component contributes 25 points max
        score = 0

        # Material prices: 25 pts
        pstat = prices["status"]
        if pstat == "fresh":
            score += 25
        elif pstat == "stale":
            score += 15
        elif pstat == "very_stale":
            score += 5
        # missing = 0

        # Property data: 25 pts
        pstat2 = property_["status"]
        if pstat2 == "fresh":
            score += 25
        elif pstat2 == "stale":
            score += 15
        elif pstat2 == "very_stale":
            score += 5

        # ML models: 25 pts
        if not models["any_critical"] and not models["any_warning"]:
            score += 25
        elif not models["any_critical"]:
            score += 15
        elif len(models["retrain_critical"]) <= 1:
            score += 5
        # all critical = 0

        # RAG corpus: 25 pts
        rstat = rag["status"]
        if rstat == "healthy":
            score += 25
        elif rstat == "sparse":
            score += 12
        # empty/unavailable = 0

        # ── Human-readable summary ────────────────────────────────────────────
        issues = []
        if pstat != "fresh":
            issues.append(f"material prices {pstat}")
        if pstat2 != "fresh":
            issues.append(f"property data {pstat2}")
        if models["retrain_critical"]:
            issues.append(f"{len(models['retrain_critical'])} model(s) critical")
        if models["retrain_recommended"]:
            issues.append(f"{len(models['retrain_recommended'])} model(s) need retraining")
        if rstat != "healthy":
            issues.append(f"RAG corpus {rstat}")

        if score >= 90:
            summary = "✅ All systems healthy."
        elif score >= 60:
            summary = f"⚠️ Degraded: {'; '.join(issues)}."
        else:
            summary = f"🚨 Critical issues: {'; '.join(issues)}."

        return {
            "overall_health_score": score,
            "summary":              summary,
            "generated_at":         ts,
            "components": {
                "material_prices":     prices,
                "property_data":       property_,
                "ml_models":           models,
                "rag_corpus":          rag,
            },
            "thresholds": {
                "data_freshness_alert_days": _FRESH_DAYS,
                "model_retrain_warn_days":   _MODEL_WARN,
                "model_retrain_crit_days":   _MODEL_CRIT,
            },
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_checker_instance: Optional[DataFreshnessChecker] = None


def get_freshness_checker() -> DataFreshnessChecker:
    """Return singleton DataFreshnessChecker."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = DataFreshnessChecker()
    return _checker_instance
