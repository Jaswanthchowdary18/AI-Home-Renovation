"""
ARKEN — Prediction Logger v1.0
================================
Non-blocking, thread-safe SQLite logger for all ML predictions and
pipeline runs. Used for:
  - Performance monitoring (drift detection, confidence tracking)
  - Pseudo-label export for incremental model retraining
  - Pipeline health dashboards via /health/pipeline

Storage: SQLite at DATA_DIR/monitoring/predictions.db
         (falls back to /tmp/arken_monitoring/predictions.db if DATA_DIR unwritable)

Design:
  - Singleton with a background ThreadPoolExecutor for fire-and-forget logging.
  - All log_*() methods return immediately (submit to executor, never block pipeline).
  - get_model_performance_summary() and export_for_retraining() are synchronous
    (called from health endpoints, not hot pipeline path).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Storage path ──────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_DEFAULT_DB_DIR = Path(os.getenv("ARKEN_DATA_DIR", str(_BACKEND_DIR / "data"))) / "monitoring"
_FALLBACK_DB_DIR = Path("/tmp/arken_monitoring")

# ── Schema ────────────────────────────────────────────────────────────────────
_CREATE_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS price_forecasts (
        log_id          TEXT PRIMARY KEY,
        ts              TEXT NOT NULL,
        material_key    TEXT NOT NULL,
        city            TEXT NOT NULL,
        predicted_price REAL NOT NULL,
        confidence      REAL NOT NULL,
        data_quality    TEXT,
        model_used      TEXT,
        is_high_conf    INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roi_predictions (
        log_id          TEXT PRIMARY KEY,
        ts              TEXT NOT NULL,
        city            TEXT NOT NULL,
        room_type       TEXT NOT NULL,
        renovation_cost REAL NOT NULL,
        predicted_roi   REAL NOT NULL,
        confidence      REAL NOT NULL,
        model_type      TEXT,
        is_high_conf    INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        log_id              TEXT PRIMARY KEY,
        ts                  TEXT NOT NULL,
        project_id          TEXT,
        duration_seconds    REAL NOT NULL,
        agents_completed    INTEGER NOT NULL DEFAULT 0,
        errors              TEXT,
        success             INTEGER NOT NULL DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pf_ts       ON price_forecasts(ts)",
    "CREATE INDEX IF NOT EXISTS idx_pf_conf     ON price_forecasts(confidence)",
    "CREATE INDEX IF NOT EXISTS idx_roi_ts      ON roi_predictions(ts)",
    "CREATE INDEX IF NOT EXISTS idx_pipe_ts     ON pipeline_runs(ts)",
]

# Confidence threshold above which a prediction is considered "high confidence"
_HIGH_CONF_THRESHOLD = float(os.getenv("PRICE_PREDICTION_CONFIDENCE_THRESHOLD", "0.65"))


class PredictionLogger:
    """
    Thread-safe singleton logger for ARKEN ML predictions and pipeline runs.

    All log_*() calls are non-blocking (fire-and-forget via background executor).
    Query methods (get_model_performance_summary, export_for_retraining) are
    synchronous and safe to call from FastAPI endpoints.
    """

    _instance: Optional["PredictionLogger"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "PredictionLogger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialised = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            self._db_path = self._resolve_db_path()
            self._ensure_schema()
            self._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="arken_pred_logger"
            )
            self._initialised = True
            logger.info(f"[PredictionLogger] Initialised. DB: {self._db_path}")

    # ── Path resolution ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_db_path() -> Path:
        for candidate in (_DEFAULT_DB_DIR, _FALLBACK_DB_DIR):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                test = candidate / ".write_test"
                test.touch()
                test.unlink()
                return candidate / "predictions.db"
            except OSError:
                continue
        raise RuntimeError("PredictionLogger: Cannot find writable directory for SQLite DB.")

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            for sql in _CREATE_TABLES_SQL:
                conn.execute(sql)
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── Internal write helpers (run in executor thread) ───────────────────────

    def _write_price_forecast(
        self,
        log_id: str,
        material_key: str,
        city: str,
        predicted_price: float,
        confidence: float,
        data_quality: str,
        model_used: str,
    ) -> None:
        try:
            ts = datetime.now(tz=timezone.utc).isoformat()
            is_hc = int(confidence >= _HIGH_CONF_THRESHOLD)
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO price_forecasts
                        (log_id, ts, material_key, city, predicted_price,
                         confidence, data_quality, model_used, is_high_conf)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (log_id, ts, material_key, city, predicted_price,
                     confidence, data_quality, model_used, is_hc),
                )
                conn.commit()
        except Exception as exc:
            logger.debug(f"[PredictionLogger] price_forecast write failed: {exc}")

    def _write_roi_prediction(
        self,
        log_id: str,
        city: str,
        room_type: str,
        renovation_cost: float,
        predicted_roi: float,
        confidence: float,
        model_type: str,
    ) -> None:
        try:
            ts = datetime.now(tz=timezone.utc).isoformat()
            is_hc = int(confidence >= _HIGH_CONF_THRESHOLD)
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO roi_predictions
                        (log_id, ts, city, room_type, renovation_cost,
                         predicted_roi, confidence, model_type, is_high_conf)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (log_id, ts, city, room_type, renovation_cost,
                     predicted_roi, confidence, model_type, is_hc),
                )
                conn.commit()
        except Exception as exc:
            logger.debug(f"[PredictionLogger] roi_prediction write failed: {exc}")

    def _write_pipeline_run(
        self,
        log_id: str,
        project_id: str,
        duration_seconds: float,
        agents_completed: int,
        errors: List[str],
    ) -> None:
        try:
            ts = datetime.now(tz=timezone.utc).isoformat()
            success = int(not errors)
            errors_json = json.dumps(errors) if errors else "[]"
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO pipeline_runs
                        (log_id, ts, project_id, duration_seconds,
                         agents_completed, errors, success)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (log_id, ts, project_id, duration_seconds,
                     agents_completed, errors_json, success),
                )
                conn.commit()
        except Exception as exc:
            logger.debug(f"[PredictionLogger] pipeline_run write failed: {exc}")

    # ── Public fire-and-forget API ─────────────────────────────────────────────

    def log_price_forecast(
        self,
        material_key: str,
        city: str,
        predicted_price: float,
        confidence: float,
        data_quality: str = "unknown",
        model_used: str = "unknown",
    ) -> str:
        """
        Log a price forecast prediction.
        Non-blocking — returns log_id immediately.

        Args:
            material_key:    e.g. "cement_opc53_per_bag_50kg"
            city:            e.g. "Hyderabad"
            predicted_price: forecasted price in INR
            confidence:      0.0–1.0 model confidence
            data_quality:    "real_csv" | "seed_extrapolated" | "heuristic"
            model_used:      "xgboost" | "prophet" | "hybrid" | "seed_fallback"

        Returns:
            log_id (UUID string)
        """
        log_id = str(uuid.uuid4())
        self._executor.submit(
            self._write_price_forecast,
            log_id, material_key, city, float(predicted_price),
            float(confidence), str(data_quality), str(model_used),
        )
        return log_id

    def log_roi_prediction(
        self,
        city: str,
        room_type: str,
        renovation_cost: float,
        predicted_roi: float,
        confidence: float,
        model_type: str = "unknown",
    ) -> str:
        """
        Log an ROI prediction.
        Non-blocking — returns log_id immediately.

        Returns:
            log_id (UUID string)
        """
        log_id = str(uuid.uuid4())
        self._executor.submit(
            self._write_roi_prediction,
            log_id, city, room_type, float(renovation_cost),
            float(predicted_roi), float(confidence), str(model_type),
        )
        return log_id

    def log_pipeline_run(
        self,
        project_id: str,
        duration_seconds: float,
        agents_completed: int,
        errors: Optional[List[str]] = None,
    ) -> str:
        """
        Log a completed pipeline run.
        Non-blocking — returns log_id immediately.

        Returns:
            log_id (UUID string)
        """
        log_id = str(uuid.uuid4())
        self._executor.submit(
            self._write_pipeline_run,
            log_id, str(project_id), float(duration_seconds),
            int(agents_completed), list(errors or []),
        )
        return log_id

    # ── Synchronous query API (health endpoints) ───────────────────────────────

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Returns aggregate performance metrics from logged predictions.

        Keys:
            price_forecast_avg_confidence   float
            roi_avg_confidence              float
            pipeline_success_rate_7d        float  (0.0–1.0)
            most_common_fallback_reason     str
            total_price_forecasts           int
            total_roi_predictions           int
            total_pipeline_runs_7d          int
        """
        cutoff_7d = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()

        try:
            with self._conn() as conn:
                # Average confidence across all price forecasts
                row = conn.execute(
                    "SELECT AVG(confidence), COUNT(*) FROM price_forecasts"
                ).fetchone()
                pf_avg_conf = round(float(row[0] or 0.0), 4)
                pf_total    = int(row[1] or 0)

                # Average confidence across all ROI predictions
                row = conn.execute(
                    "SELECT AVG(confidence), COUNT(*) FROM roi_predictions"
                ).fetchone()
                roi_avg_conf = round(float(row[0] or 0.0), 4)
                roi_total    = int(row[1] or 0)

                # Pipeline success rate (last 7 days)
                row = conn.execute(
                    "SELECT COUNT(*), SUM(success) FROM pipeline_runs WHERE ts >= ?",
                    (cutoff_7d,),
                ).fetchone()
                pipe_total   = int(row[0] or 0)
                pipe_success = int(row[1] or 0)
                success_rate = round(pipe_success / pipe_total, 4) if pipe_total else 0.0

                # Most common fallback model (last 7 days of price forecasts)
                fallback_rows = conn.execute(
                    """
                    SELECT model_used, COUNT(*) as cnt
                    FROM price_forecasts
                    WHERE model_used LIKE '%fallback%' OR model_used LIKE '%seed%'
                      OR model_used LIKE '%heuristic%'
                    GROUP BY model_used ORDER BY cnt DESC LIMIT 1
                    """
                ).fetchone()
                fallback_reason = fallback_rows["model_used"] if fallback_rows else "none"

                return {
                    "price_forecast_avg_confidence": pf_avg_conf,
                    "roi_avg_confidence":            roi_avg_conf,
                    "pipeline_success_rate_7d":      success_rate,
                    "most_common_fallback_reason":   fallback_reason,
                    "total_price_forecasts":         pf_total,
                    "total_roi_predictions":         roi_total,
                    "total_pipeline_runs_7d":        pipe_total,
                }

        except Exception as exc:
            logger.warning(f"[PredictionLogger] get_model_performance_summary failed: {exc}")
            return {
                "price_forecast_avg_confidence": 0.0,
                "roi_avg_confidence":            0.0,
                "pipeline_success_rate_7d":      0.0,
                "most_common_fallback_reason":   "unavailable",
                "total_price_forecasts":         0,
                "total_roi_predictions":         0,
                "total_pipeline_runs_7d":        0,
            }

    def get_pipeline_stats_7d(self) -> List[Dict[str, Any]]:
        """Return per-run stats for the last 7 days (for /health/pipeline)."""
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT log_id, ts, project_id, duration_seconds,
                           agents_completed, errors, success
                    FROM pipeline_runs
                    WHERE ts >= ?
                    ORDER BY ts DESC
                    LIMIT 200
                    """,
                    (cutoff,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"[PredictionLogger] get_pipeline_stats_7d failed: {exc}")
            return []

    def export_for_retraining(self, days_back: int = 30) -> "pd.DataFrame":
        """
        Export high-confidence predictions from the last N days as a DataFrame.
        Used as pseudo-labels for incremental model improvement.

        Returns:
            pd.DataFrame with columns from both price_forecasts and roi_predictions,
            merged as separate source tables, unified schema.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("[PredictionLogger] pandas required for export_for_retraining")
            return None  # type: ignore

        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).isoformat()

        try:
            with self._conn() as conn:
                # High-confidence price forecasts
                pf_df = pd.read_sql_query(
                    """
                    SELECT log_id, ts, material_key, city, predicted_price,
                           confidence, data_quality, model_used,
                           'price_forecast' AS prediction_type
                    FROM price_forecasts
                    WHERE ts >= ? AND is_high_conf = 1
                    ORDER BY ts DESC
                    """,
                    conn,
                    params=(cutoff,),
                )

                # High-confidence ROI predictions
                roi_df = pd.read_sql_query(
                    """
                    SELECT log_id, ts, city, room_type, renovation_cost,
                           predicted_roi, confidence, model_type AS model_used,
                           'roi_prediction' AS prediction_type
                    FROM roi_predictions
                    WHERE ts >= ? AND is_high_conf = 1
                    ORDER BY ts DESC
                    """,
                    conn,
                    params=(cutoff,),
                )

            # Return a combined frame — callers filter by prediction_type
            combined = pd.concat([pf_df, roi_df], ignore_index=True, sort=False)
            logger.info(
                f"[PredictionLogger] export_for_retraining: "
                f"{len(pf_df)} price + {len(roi_df)} ROI = {len(combined)} rows "
                f"(last {days_back}d, high-confidence only)"
            )
            return combined

        except Exception as exc:
            logger.warning(f"[PredictionLogger] export_for_retraining failed: {exc}")
            try:
                import pandas as pd
                return pd.DataFrame()
            except Exception:
                return None  # type: ignore


# ── Module-level singleton accessor ──────────────────────────────────────────

_logger_instance: Optional[PredictionLogger] = None


def get_prediction_logger() -> PredictionLogger:
    """Return the singleton PredictionLogger. Safe to call from any thread."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PredictionLogger()
    return _logger_instance
