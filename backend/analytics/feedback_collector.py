"""
ARKEN — Feedback Collector v1.0
==================================
Collects user corrections and renovation completion data to create
ground-truth labels for incremental model retraining.

Data collected:
  - Style corrections    → pseudo-labels for style classifier retraining
  - BOQ corrections      → price/quantity calibration signals
  - Renovation actuals   → gold labels for ROI model retraining
  - ROI actuals          → property appraisal ground truth

Uses the SAME SQLite database as PredictionLogger (same DB path resolution).
Tables are created on first use via CREATE TABLE IF NOT EXISTS.

Thread-safety: all writes go through a single threading.Lock.

Usage:
    from analytics.feedback_collector import get_feedback_collector
    fc = get_feedback_collector()
    fc.log_style_correction(project_id, "Bohemian eclectic", "Industrial loft", "bedroom", 0.42)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Shared DB path (must match PredictionLogger exactly) ──────────────────────
_BACKEND_DIR     = Path(__file__).resolve().parent.parent
_DEFAULT_DB_DIR  = Path(os.getenv("ARKEN_DATA_DIR", str(_BACKEND_DIR / "data"))) / "monitoring"
_FALLBACK_DB_DIR = Path("/tmp/arken_monitoring")

# Minimum samples needed before export_for_retraining() is allowed
_MIN_STYLE_SAMPLES    = 50
_MIN_ROI_SAMPLES      = 20


class InsufficientDataError(ValueError):
    """Raised when there are fewer samples than required for retraining export."""


class FeedbackCollector:
    """
    Thread-safe singleton for collecting and exporting user feedback.

    All log_*() methods are synchronous and return immediately after the write.
    Tables are auto-created on first write.

    Singleton — use get_feedback_collector() to obtain the shared instance.
    """

    _instance: Optional["FeedbackCollector"] = None
    _write_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FeedbackCollector":
        if cls._instance is None:
            with cls._write_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._db_path    = cls._resolve_db_path()
                    inst._initialised = False
                    cls._instance   = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        with self._write_lock:
            if self._initialised:
                return
            self._ensure_schema()
            self._initialised = True
            logger.info(f"[FeedbackCollector] Initialised. DB: {self._db_path}")

    # ── Path resolution ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_db_path() -> Path:
        """Resolve the same SQLite path as PredictionLogger."""
        # First check if an existing DB is already present
        for candidate in (_DEFAULT_DB_DIR, _FALLBACK_DB_DIR):
            db = candidate / "predictions.db"
            if db.exists():
                return db
        # Create in default location
        for candidate in (_DEFAULT_DB_DIR, _FALLBACK_DB_DIR):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                test = candidate / ".write_test"
                test.touch()
                test.unlink()
                return candidate / "predictions.db"
            except OSError:
                continue
        raise RuntimeError("FeedbackCollector: Cannot find writable directory for SQLite DB.")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Create all feedback tables if they don't exist."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS style_corrections (
                log_id                  TEXT PRIMARY KEY,
                ts                      TEXT NOT NULL,
                project_id              TEXT NOT NULL,
                original_prediction     TEXT NOT NULL,
                user_corrected_to       TEXT NOT NULL,
                room_type               TEXT NOT NULL,
                confidence_at_prediction REAL,
                is_disagreement         INTEGER NOT NULL DEFAULT 1
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_sc_ts ON style_corrections(ts)",
            "CREATE INDEX IF NOT EXISTS idx_sc_pid ON style_corrections(project_id)",
            """
            CREATE TABLE IF NOT EXISTS boq_corrections (
                log_id              TEXT PRIMARY KEY,
                ts                  TEXT NOT NULL,
                project_id          TEXT NOT NULL,
                material_key        TEXT NOT NULL,
                original_qty        REAL,
                user_qty            REAL,
                original_price_inr  REAL,
                user_price_inr      REAL,
                pct_qty_change      REAL,
                pct_price_change    REAL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_boq_ts  ON boq_corrections(ts)",
            "CREATE INDEX IF NOT EXISTS idx_boq_mat ON boq_corrections(material_key)",
            """
            CREATE TABLE IF NOT EXISTS renovation_actuals (
                log_id              TEXT PRIMARY KEY,
                ts                  TEXT NOT NULL,
                project_id          TEXT NOT NULL,
                actual_total_inr    REAL NOT NULL,
                completion_date     TEXT,
                city                TEXT NOT NULL,
                room_type           TEXT NOT NULL,
                budget_tier         TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_ra_ts  ON renovation_actuals(ts)",
            "CREATE INDEX IF NOT EXISTS idx_ra_pid ON renovation_actuals(project_id)",
            """
            CREATE TABLE IF NOT EXISTS roi_actuals (
                log_id                  TEXT PRIMARY KEY,
                ts                      TEXT NOT NULL,
                project_id              TEXT NOT NULL,
                actual_roi_pct          REAL NOT NULL,
                appraisal_date          TEXT,
                property_value_before   REAL,
                property_value_after    REAL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_roia_ts  ON roi_actuals(ts)",
            "CREATE INDEX IF NOT EXISTS idx_roia_pid ON roi_actuals(project_id)",
            # Add actual_roi_pct to roi_predictions if it doesn't exist yet
            # (used by DriftMonitor for MAE computation)
            """
            CREATE TABLE IF NOT EXISTS price_actuals (
                log_id          TEXT PRIMARY KEY,
                ts              TEXT NOT NULL,
                material_key    TEXT NOT NULL,
                city            TEXT NOT NULL,
                actual_price    REAL NOT NULL,
                source          TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_pa_ts  ON price_actuals(ts)",
            "CREATE INDEX IF NOT EXISTS idx_pa_mat ON price_actuals(material_key)",
        ]

        with self._conn() as conn:
            for sql in tables:
                try:
                    conn.execute(sql)
                except Exception as e:
                    logger.debug(f"[FeedbackCollector] Schema SQL failed (non-fatal): {e}")
            conn.commit()

    # ─────────────────────────────────────────────────────────────────────────
    # Public logging methods
    # ─────────────────────────────────────────────────────────────────────────

    def log_style_correction(
        self,
        project_id: str,
        original_prediction: str,
        user_corrected_to: str,
        room_type: str,
        confidence_at_prediction: float = 0.5,
    ) -> str:
        """
        Log a user style correction (model was wrong, user picked a different style).

        Args:
            project_id:                UUID of the project.
            original_prediction:       Style the model predicted (e.g. "Bohemian eclectic").
            user_corrected_to:         Style the user selected (e.g. "Industrial loft").
            room_type:                 Room type at time of prediction.
            confidence_at_prediction:  Model confidence score at time of prediction.

        Returns:
            log_id (UUID string)
        """
        log_id         = str(uuid.uuid4())
        ts             = datetime.now(tz=timezone.utc).isoformat()
        is_disagreement = int(original_prediction.lower() != user_corrected_to.lower())

        try:
            with self._write_lock:
                with self._conn() as conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO style_corrections
                            (log_id, ts, project_id, original_prediction, user_corrected_to,
                             room_type, confidence_at_prediction, is_disagreement)
                        VALUES (?,?,?,?,?,?,?,?)
                        """,
                        (log_id, ts, str(project_id), str(original_prediction),
                         str(user_corrected_to), str(room_type),
                         float(confidence_at_prediction), is_disagreement),
                    )
                    conn.commit()
            logger.debug(
                f"[FeedbackCollector] Style correction logged: "
                f"{original_prediction} → {user_corrected_to} (project={project_id})"
            )
        except Exception as e:
            logger.warning(f"[FeedbackCollector] log_style_correction failed: {e}")

        return log_id

    def log_boq_correction(
        self,
        project_id: str,
        material_key: str,
        original_qty: Optional[float],
        user_qty: Optional[float],
        original_price_inr: Optional[float],
        user_price_inr: Optional[float],
    ) -> str:
        """
        Log a user BOQ correction (edited quantity or price for a material).

        Args:
            project_id:         UUID of the project.
            material_key:       Material identifier (e.g. "asian_paints_premium_per_litre").
            original_qty:       Quantity the BOQ estimated.
            user_qty:           Quantity the user entered.
            original_price_inr: Unit price the BOQ used.
            user_price_inr:     Unit price the user entered.

        Returns:
            log_id (UUID string)
        """
        log_id = str(uuid.uuid4())
        ts     = datetime.now(tz=timezone.utc).isoformat()

        # Compute percentage changes
        pct_qty_change = None
        if original_qty and user_qty and original_qty != 0:
            pct_qty_change = round((user_qty - original_qty) / original_qty * 100, 2)

        pct_price_change = None
        if original_price_inr and user_price_inr and original_price_inr != 0:
            pct_price_change = round(
                (user_price_inr - original_price_inr) / original_price_inr * 100, 2
            )

        try:
            with self._write_lock:
                with self._conn() as conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO boq_corrections
                            (log_id, ts, project_id, material_key, original_qty, user_qty,
                             original_price_inr, user_price_inr, pct_qty_change, pct_price_change)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                        """,
                        (log_id, ts, str(project_id), str(material_key),
                         original_qty, user_qty, original_price_inr, user_price_inr,
                         pct_qty_change, pct_price_change),
                    )
                    conn.commit()
            logger.debug(
                f"[FeedbackCollector] BOQ correction: {material_key} qty_chg={pct_qty_change}% "
                f"price_chg={pct_price_change}%"
            )
        except Exception as e:
            logger.warning(f"[FeedbackCollector] log_boq_correction failed: {e}")

        return log_id

    def log_actual_renovation_cost(
        self,
        project_id: str,
        actual_total_inr: float,
        completion_date: str,
        city: str,
        room_type: str,
        budget_tier: str,
    ) -> str:
        """
        Log actual renovation cost after project completion.
        This is a gold label for ROI model retraining.

        Args:
            project_id:       UUID of the project.
            actual_total_inr: Actual total spent (INR).
            completion_date:  ISO date string when renovation completed.
            city:             City name.
            room_type:        Room type.
            budget_tier:      "basic" | "mid" | "premium".

        Returns:
            log_id (UUID string)
        """
        log_id = str(uuid.uuid4())
        ts     = datetime.now(tz=timezone.utc).isoformat()

        try:
            with self._write_lock:
                with self._conn() as conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO renovation_actuals
                            (log_id, ts, project_id, actual_total_inr, completion_date,
                             city, room_type, budget_tier)
                        VALUES (?,?,?,?,?,?,?,?)
                        """,
                        (log_id, ts, str(project_id), float(actual_total_inr),
                         str(completion_date), str(city), str(room_type), str(budget_tier)),
                    )
                    conn.commit()
            logger.info(
                f"[FeedbackCollector] Renovation actual logged: ₹{actual_total_inr:,.0f} "
                f"for {room_type} in {city} (project={project_id})"
            )
        except Exception as e:
            logger.warning(f"[FeedbackCollector] log_actual_renovation_cost failed: {e}")

        return log_id

    def log_actual_roi(
        self,
        project_id: str,
        actual_roi_pct: float,
        appraisal_date: str,
        property_value_before: Optional[float] = None,
        property_value_after: Optional[float] = None,
    ) -> str:
        """
        Log actual ROI from post-renovation property appraisal.
        True ground truth for ROI model — the most valuable training signal.

        Args:
            project_id:             UUID of the project.
            actual_roi_pct:         Observed ROI percentage.
            appraisal_date:         ISO date string of the appraisal.
            property_value_before:  Property value before renovation (INR).
            property_value_after:   Property value after renovation (INR).

        Returns:
            log_id (UUID string)
        """
        log_id = str(uuid.uuid4())
        ts     = datetime.now(tz=timezone.utc).isoformat()

        try:
            with self._write_lock:
                with self._conn() as conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO roi_actuals
                            (log_id, ts, project_id, actual_roi_pct, appraisal_date,
                             property_value_before, property_value_after)
                        VALUES (?,?,?,?,?,?,?)
                        """,
                        (log_id, ts, str(project_id), float(actual_roi_pct),
                         str(appraisal_date), property_value_before, property_value_after),
                    )
                    conn.commit()
            logger.info(
                f"[FeedbackCollector] ROI actual logged: {actual_roi_pct:.2f}% "
                f"(project={project_id})"
            )
        except Exception as e:
            logger.warning(f"[FeedbackCollector] log_actual_roi failed: {e}")

        return log_id

    # ─────────────────────────────────────────────────────────────────────────
    # Export and summary methods
    # ─────────────────────────────────────────────────────────────────────────

    def export_for_retraining(
        self,
        model_name: str,
        min_samples: int = 50,
    ) -> "pd.DataFrame":
        """
        Export feedback data as a DataFrame ready for incremental model training.

        Args:
            model_name:   "style_classifier" | "roi" | "price" | "boq"
            min_samples:  Minimum rows required — raises InsufficientDataError if below.

        Returns:
            pd.DataFrame with columns specific to the model type.

        Raises:
            InsufficientDataError: if fewer than min_samples rows available.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")

        if not self._db_exists():
            raise InsufficientDataError(
                f"No feedback data available — predictions.db not yet created. "
                f"Users must submit feedback via the dashboard first."
            )

        try:
            with self._conn() as conn:
                if model_name == "style_classifier":
                    df = pd.read_sql_query(
                        """
                        SELECT
                            sc.log_id, sc.ts, sc.project_id,
                            sc.original_prediction  AS predicted_style,
                            sc.user_corrected_to    AS true_style,
                            sc.room_type,
                            sc.confidence_at_prediction,
                            sc.is_disagreement
                        FROM style_corrections sc
                        WHERE sc.is_disagreement = 1
                        ORDER BY sc.ts DESC
                        """,
                        conn,
                    )

                elif model_name == "roi":
                    df = pd.read_sql_query(
                        """
                        SELECT
                            ra.log_id, ra.ts, ra.project_id,
                            ra.actual_roi_pct,
                            ra.appraisal_date,
                            ra.property_value_before,
                            ra.property_value_after,
                            ren.actual_total_inr AS renovation_cost_inr,
                            ren.city,
                            ren.room_type,
                            ren.budget_tier
                        FROM roi_actuals ra
                        LEFT JOIN renovation_actuals ren ON ra.project_id = ren.project_id
                        ORDER BY ra.ts DESC
                        """,
                        conn,
                    )

                elif model_name == "price":
                    df = pd.read_sql_query(
                        """
                        SELECT
                            bc.log_id, bc.ts, bc.project_id,
                            bc.material_key,
                            bc.user_price_inr   AS actual_price_inr,
                            bc.original_price_inr AS predicted_price_inr,
                            bc.pct_price_change
                        FROM boq_corrections bc
                        WHERE bc.user_price_inr IS NOT NULL
                          AND bc.pct_price_change IS NOT NULL
                        ORDER BY bc.ts DESC
                        """,
                        conn,
                    )

                elif model_name == "boq":
                    df = pd.read_sql_query(
                        """
                        SELECT *
                        FROM boq_corrections
                        ORDER BY ts DESC
                        """,
                        conn,
                    )

                else:
                    raise ValueError(
                        f"Unknown model_name: '{model_name}'. "
                        "Valid options: style_classifier | roi | price | boq"
                    )

        except InsufficientDataError:
            raise
        except Exception as e:
            raise RuntimeError(f"[FeedbackCollector] export_for_retraining failed: {e}") from e

        if len(df) < min_samples:
            raise InsufficientDataError(
                f"Only {len(df)} samples available for '{model_name}' "
                f"(minimum {min_samples} required). "
                f"Collect more user feedback before retraining."
            )

        logger.info(
            f"[FeedbackCollector] Exported {len(df)} rows for {model_name} retraining"
        )
        return df

    def get_correction_summary(self) -> Dict[str, Any]:
        """
        Return counts of all feedback types collected.

        Returns:
            {
              style_corrections_count: int,
              boq_corrections_count: int,
              actuals_count: int,
              roi_actuals_count: int,
              data_ready_for_retraining: bool,
              db_path: str,
              db_exists: bool,
            }
        """
        if not self._db_exists():
            return {
                "style_corrections_count":  0,
                "boq_corrections_count":    0,
                "actuals_count":            0,
                "roi_actuals_count":        0,
                "data_ready_for_retraining": False,
                "db_path":                  str(self._db_path),
                "db_exists":                False,
            }

        try:
            with self._conn() as conn:
                sc_count = self._count_table(conn, "style_corrections")
                bq_count = self._count_table(conn, "boq_corrections")
                ra_count = self._count_table(conn, "renovation_actuals")
                ri_count = self._count_table(conn, "roi_actuals")

            data_ready = (
                sc_count >= _MIN_STYLE_SAMPLES or ri_count >= _MIN_ROI_SAMPLES
            )

            return {
                "style_corrections_count":   sc_count,
                "boq_corrections_count":     bq_count,
                "actuals_count":             ra_count,
                "roi_actuals_count":         ri_count,
                "data_ready_for_retraining": data_ready,
                "min_style_samples_needed":  _MIN_STYLE_SAMPLES,
                "min_roi_samples_needed":    _MIN_ROI_SAMPLES,
                "db_path":                   str(self._db_path),
                "db_exists":                 True,
            }

        except Exception as e:
            logger.warning(f"[FeedbackCollector] get_correction_summary failed: {e}")
            return {
                "style_corrections_count":   0,
                "boq_corrections_count":     0,
                "actuals_count":             0,
                "roi_actuals_count":         0,
                "data_ready_for_retraining": False,
                "error":                     str(e),
            }

    def _db_exists(self) -> bool:
        return self._db_path.exists()

    @staticmethod
    def _count_table(conn: sqlite3.Connection, table: str) -> int:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0


# ── Singleton accessor ─────────────────────────────────────────────────────────

_collector_instance: Optional[FeedbackCollector] = None


def get_feedback_collector() -> FeedbackCollector:
    """Return the singleton FeedbackCollector. Safe to call from any thread."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = FeedbackCollector()
    return _collector_instance
