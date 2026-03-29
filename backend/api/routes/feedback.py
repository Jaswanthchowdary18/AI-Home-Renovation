"""
ARKEN — Feedback Accuracy Router v1.0
========================================
POST /api/v1/feedback/accuracy

Accepts user accuracy feedback on ROI, price, and style predictions and
persists it to the PredictionLogger SQLite database so drift_monitor.py
has real user-correction data to analyse.

Endpoint contract:
    POST /api/v1/feedback/accuracy
    {
        "project_id":  str,          required
        "feature":     "roi" | "price" | "style",   required
        "was_accurate": bool,        required
        "actual_value": float | null,  optional — user-reported ground truth
        "comment":     str,          optional — free-form feedback
    }

    Response 200:
    {
        "feedback_id": str,          UUID for the recorded feedback
        "status":      "recorded",
        "feature":     str,
        "was_accurate": bool,
        "message":     str
    }

Storage:
    Written to the SAME SQLite DB as PredictionLogger via a new
    `user_feedback` table.  drift_monitor.py is extended to read this
    table for style accuracy drift computation.

The endpoint is intentionally lightweight — no auth required so the
frontend FeedbackPanel can call it without a token (feedback is not
sensitive data).  Rate-limiting is handled at the nginx/ingress layer.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class FeedbackFeature:
    ROI   = "roi"
    PRICE = "price"
    STYLE = "style"
    ALL   = {"roi", "price", "style"}


class AccuracyFeedbackRequest(BaseModel):
    project_id:   str   = Field(..., min_length=1, max_length=128,
                                description="Project UUID from ARKEN analysis run")
    feature:      str   = Field(..., description="roi | price | style")
    was_accurate: bool  = Field(..., description="True if prediction matched reality")
    actual_value: Optional[float] = Field(
        None,
        description="User-reported ground truth (e.g. actual ROI%, actual price ₹)"
    )
    comment:      str   = Field("", max_length=500,
                                description="Optional free-text feedback")

    @validator("feature")
    def feature_must_be_valid(cls, v: str) -> str:  # noqa: N805
        v = v.lower().strip()
        if v not in FeedbackFeature.ALL:
            raise ValueError(f"feature must be one of: {sorted(FeedbackFeature.ALL)}")
        return v

    @validator("comment", pre=True, always=True)
    def sanitise_comment(cls, v) -> str:  # noqa: N805
        return str(v or "")[:500]


class AccuracyFeedbackResponse(BaseModel):
    feedback_id:  str
    status:       str
    feature:      str
    was_accurate: bool
    message:      str


# ── SQLite feedback table ─────────────────────────────────────────────────────

_CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id   TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    project_id    TEXT NOT NULL,
    feature       TEXT NOT NULL,
    was_accurate  INTEGER NOT NULL,
    actual_value  REAL,
    comment       TEXT,
    processed     INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_fb_ts      ON user_feedback(ts)",
    "CREATE INDEX IF NOT EXISTS idx_fb_feature ON user_feedback(feature)",
    "CREATE INDEX IF NOT EXISTS idx_fb_project ON user_feedback(project_id)",
]

_db_init_lock = threading.Lock()
_db_initialized = False


def _get_db_path() -> Path:
    """Reuse the same DB path as PredictionLogger (DRY — single SQLite file)."""
    try:
        from services.monitoring.prediction_logger import _DEFAULT_DB_DIR, _FALLBACK_DB_DIR
        for candidate in (_DEFAULT_DB_DIR, _FALLBACK_DB_DIR):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                test = candidate / ".write_test"
                test.touch()
                test.unlink()
                return candidate / "predictions.db"
            except OSError:
                continue
    except ImportError:
        pass

    # Absolute fallback
    fallback = Path("/tmp/arken_monitoring")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback / "predictions.db"


_DB_PATH = _get_db_path()


def _ensure_feedback_table() -> None:
    """Idempotently create user_feedback table in the shared SQLite DB."""
    global _db_initialized
    if _db_initialized:
        return
    with _db_init_lock:
        if _db_initialized:
            return
        try:
            with _db_connection() as conn:
                conn.execute(_CREATE_FEEDBACK_TABLE)
                for idx_sql in _CREATE_INDEXES:
                    conn.execute(idx_sql)
                conn.commit()
            _db_initialized = True
            logger.info(
                f"[FeedbackRouter] user_feedback table ensured in {_DB_PATH}"
            )
        except Exception as e:
            logger.error(f"[FeedbackRouter] Failed to create feedback table: {e}")


@contextmanager
def _db_connection():
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _write_feedback(
    feedback_id: str,
    project_id:  str,
    feature:     str,
    was_accurate: bool,
    actual_value: Optional[float],
    comment:     str,
) -> None:
    """Write feedback row to SQLite.  Runs in a background thread."""
    try:
        ts = datetime.now(tz=timezone.utc).isoformat()
        with _db_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO user_feedback
                    (feedback_id, ts, project_id, feature,
                     was_accurate, actual_value, comment, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    feedback_id,
                    ts,
                    project_id,
                    feature,
                    int(was_accurate),
                    actual_value,
                    comment or None,
                ),
            )
            conn.commit()
        logger.debug(
            f"[FeedbackRouter] Wrote feedback {feedback_id}: "
            f"project={project_id} feature={feature} accurate={was_accurate}"
        )

        # Also update the PredictionLogger's style accuracy tracking
        # so drift_monitor can compute rolling correction rates
        if feature == FeedbackFeature.STYLE:
            _notify_style_drift_monitor(feedback_id, project_id, was_accurate)

    except Exception as e:
        logger.error(f"[FeedbackRouter] Failed to write feedback {feedback_id}: {e}")


def _notify_style_drift_monitor(
    feedback_id: str,
    project_id:  str,
    was_accurate: bool,
) -> None:
    """
    Best-effort: write a style correction event so drift_monitor can compute
    rolling style accuracy from real user corrections (not just held-out eval).
    This writes to the same predictions.db under a style_corrections table.
    """
    _STYLE_CORRECTIONS_DDL = """
    CREATE TABLE IF NOT EXISTS style_corrections (
        log_id      TEXT PRIMARY KEY,
        ts          TEXT NOT NULL,
        project_id  TEXT NOT NULL,
        feedback_id TEXT NOT NULL,
        was_correct INTEGER NOT NULL
    )
    """
    try:
        with _db_connection() as conn:
            conn.execute(_STYLE_CORRECTIONS_DDL)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sc_ts
                ON style_corrections(ts)
                """,
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO style_corrections
                    (log_id, ts, project_id, feedback_id, was_correct)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    datetime.now(tz=timezone.utc).isoformat(),
                    project_id,
                    feedback_id,
                    int(was_accurate),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"[FeedbackRouter] style_corrections write skipped: {e}")


# ── Route ─────────────────────────────────────────────────────────────────────

@router.on_event("startup")
async def _startup_init() -> None:
    """Ensure DB table exists on router startup."""
    _ensure_feedback_table()


@router.post(
    "/accuracy",
    response_model=AccuracyFeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit prediction accuracy feedback",
    description=(
        "Record whether an ARKEN prediction (ROI / price / style) was accurate. "
        "Results feed the drift monitoring system and are used for model "
        "improvement tracking.  No authentication required."
    ),
    tags=["feedback"],
)
async def post_accuracy_feedback(
    body: AccuracyFeedbackRequest,
) -> AccuracyFeedbackResponse:
    """
    Accept and persist a user accuracy feedback event.

    - **project_id**: UUID from the ARKEN analysis run
    - **feature**: `roi` | `price` | `style`
    - **was_accurate**: `true` if model prediction matched reality
    - **actual_value**: ground truth value (optional but improves drift detection)
    - **comment**: free-text user comment (optional)
    """
    _ensure_feedback_table()

    feedback_id = str(uuid.uuid4())

    # Fire-and-forget to SQLite (non-blocking for the API response)
    import threading as _threading
    t = _threading.Thread(
        target=_write_feedback,
        args=(
            feedback_id,
            body.project_id,
            body.feature,
            body.was_accurate,
            body.actual_value,
            body.comment,
        ),
        daemon=True,
    )
    t.start()

    # Also fire to PredictionLogger for ROI / price features
    try:
        if body.feature == FeedbackFeature.ROI and body.actual_value is not None:
            from services.monitoring.prediction_logger import get_prediction_logger
            pl = get_prediction_logger()
            # Log actual ROI as a cross-reference prediction for drift detection
            pl.log_roi_prediction(
                city="unknown",          # project_id carries the identity
                room_type="unknown",
                renovation_cost=0.0,
                predicted_roi=body.actual_value,
                confidence=1.0,          # ground truth → confidence = 1.0
                model_type="user_ground_truth",
            )
        elif body.feature == FeedbackFeature.PRICE and body.actual_value is not None:
            from services.monitoring.prediction_logger import get_prediction_logger
            pl = get_prediction_logger()
            pl.log_price_forecast(
                material_key="user_reported",
                city="unknown",
                predicted_price=body.actual_value,
                confidence=1.0,
                data_quality="user_ground_truth",
                model_used="user_ground_truth",
            )
    except Exception as e:
        logger.debug(f"[FeedbackRouter] PredictionLogger update skipped: {e}")

    feature_labels = {
        "roi":   "ROI prediction",
        "price": "price forecast",
        "style": "style classification",
    }
    accuracy_word = "accurate" if body.was_accurate else "inaccurate"
    msg = (
        f"Your feedback on the {feature_labels.get(body.feature, body.feature)} "
        f"has been recorded as {accuracy_word}. Thank you — this helps improve ARKEN."
    )

    logger.info(
        f"[FeedbackRouter] Feedback received: id={feedback_id} "
        f"project={body.project_id} feature={body.feature} "
        f"accurate={body.was_accurate}"
    )

    return AccuracyFeedbackResponse(
        feedback_id=feedback_id,
        status="recorded",
        feature=body.feature,
        was_accurate=body.was_accurate,
        message=msg,
    )


@router.get(
    "/accuracy/summary",
    summary="Get feedback accuracy summary",
    description="Return aggregated accuracy counts by feature (for health dashboard).",
    tags=["feedback"],
)
async def get_accuracy_summary() -> dict:
    """
    Return aggregated accuracy feedback counts by feature.
    Used by the health dashboard to show user satisfaction metrics.
    """
    _ensure_feedback_table()
    try:
        with _db_connection() as conn:
            rows = conn.execute(
                """
                SELECT feature,
                       COUNT(*)                                    AS total,
                       SUM(was_accurate)                          AS accurate_count,
                       ROUND(1.0 * SUM(was_accurate) / COUNT(*), 4) AS accuracy_rate
                FROM user_feedback
                GROUP BY feature
                """
            ).fetchall()

            summary = {}
            for row in rows:
                summary[row["feature"]] = {
                    "total_feedback":   int(row["total"]),
                    "accurate_count":   int(row["accurate_count"] or 0),
                    "accuracy_rate":    float(row["accuracy_rate"] or 0.0),
                }

            total_row = conn.execute(
                "SELECT COUNT(*) AS total FROM user_feedback"
            ).fetchone()

            return {
                "by_feature":     summary,
                "total_feedback": int(total_row["total"] if total_row else 0),
                "db_path":        str(_DB_PATH),
            }
    except Exception as e:
        logger.error(f"[FeedbackRouter] Summary query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve feedback summary: {e}",
        )
