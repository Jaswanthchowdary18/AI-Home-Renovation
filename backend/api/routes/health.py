"""
ARKEN — Health & MLOps API Router v2.0
=========================================
Extends v1.0 with new analytics endpoints:
  GET  /health/metrics                    — held-out eval metrics (24h cached)
  GET  /health/drift                      — real-time drift detection report
  POST /health/feedback/style-correction  — log style prediction correction
  POST /health/feedback/actual-cost       — log actual renovation cost
  GET  /health/feedback/summary           — feedback collection counts
  POST /health/retrain/{model_name}       — trigger model retraining (admin)

Preserved from v1.0 (unchanged):
  GET  /health/data              → DataFreshnessChecker.get_full_health_report()
  GET  /health/models            → ML model ages + retraining recommendations
  GET  /health/pipeline          → Last 7 days of pipeline run stats
  POST /health/retrain           → Trigger async retraining (all models), returns job_id
  GET  /health/retrain/{job_id}  → Poll retraining job status
  GET  /health/retrain           → List all retraining jobs

Auth:
  GET /health/metrics             requires pro or admin JWT
  POST /health/retrain/{model}    requires admin JWT
  All other endpoints: no auth required (monitoring)
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Auth helpers ──────────────────────────────────────────────────────────────
_oauth2_scheme      = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)
_oauth2_scheme_req  = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=True)


def _decode_token(token: str) -> Dict[str, Any]:
    """Decode JWT and return payload. Raises HTTPException on failure."""
    try:
        import jwt
        from config import settings
        payload = jwt.decode(
            token,
            settings.SECRET_KEY.get_secret_value(),
            algorithms=[settings.ALGORITHM],
        )
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e}")


async def _require_pro_or_admin(token: str = Depends(_oauth2_scheme_req)) -> Dict:
    """Dependency: requires pro or admin role JWT."""
    payload = _decode_token(token)
    role = payload.get("role", "user")
    if role not in ("pro", "enterprise", "admin"):
        raise HTTPException(status_code=403, detail="Pro or admin role required.")
    return payload


async def _require_admin(token: str = Depends(_oauth2_scheme_req)) -> Dict:
    """Dependency: requires admin role JWT."""
    payload = _decode_token(token)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return payload


# ── In-memory job registry (v1.0 — preserved) ─────────────────────────────────
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _ok(data: Any) -> Dict[str, Any]:
    return {
        "status":       "ok",
        "data":         data,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _error(message: str, code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={
            "status":       "error",
            "error":        message,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRESERVED v1.0 ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/data", summary="Data freshness report for all ML data sources")
async def health_data() -> Dict[str, Any]:
    """
    Returns a full health report covering:
    - Material prices CSV freshness
    - Property transactions CSV freshness
    - ML model file ages and retraining recommendations
    - RAG corpus chunk count and domain coverage
    - Overall health score (0–100)
    """
    try:
        from services.monitoring.data_freshness_checker import get_freshness_checker
        checker = get_freshness_checker()
        report  = checker.get_full_health_report()
        return _ok(report)
    except Exception as exc:
        logger.error(f"[health/data] {exc}", exc_info=True)
        return _error(f"Health check failed: {exc}")


@router.get("/models", summary="ML model ages and retraining status")
async def health_models() -> Dict[str, Any]:
    """
    Returns:
    - Per-model file age in days
    - Last training date from model_report.json
    - Retraining recommendations (warn if >30 days, critical if >60 days)
    - Model performance metrics (MAE, RMSE, R² from model_report.json)
    """
    try:
        from services.monitoring.data_freshness_checker import get_freshness_checker
        from services.monitoring.prediction_logger import get_prediction_logger

        checker = get_freshness_checker()
        models  = checker.check_ml_models()
        perf    = get_prediction_logger().get_model_performance_summary()

        return _ok({
            "model_files":            models["models"],
            "any_critical":           models["any_critical"],
            "any_warning":            models["any_warning"],
            "retrain_recommended":    models["retrain_recommended"],
            "retrain_critical":       models["retrain_critical"],
            "model_report":           models.get("model_report_summary"),
            "prediction_performance": perf,
        })
    except Exception as exc:
        logger.error(f"[health/models] {exc}", exc_info=True)
        return _error(f"Model health check failed: {exc}")


@router.get("/pipeline", summary="Pipeline run stats for the last 7 days")
async def health_pipeline() -> Dict[str, Any]:
    """
    Returns aggregated and per-run stats from the last 7 days of pipeline executions.
    """
    try:
        from services.monitoring.prediction_logger import get_prediction_logger
        pl = get_prediction_logger()

        summary = pl.get_model_performance_summary()
        runs    = pl.get_pipeline_stats_7d()

        total       = len(runs)
        successful  = sum(1 for r in runs if r.get("success", 0))
        avg_duration = (
            round(sum(r.get("duration_seconds", 0) for r in runs) / total, 2)
            if total else 0.0
        )

        return _ok({
            "summary": {
                "total_runs_7d":            total,
                "successful_runs_7d":       successful,
                "failed_runs_7d":           total - successful,
                "success_rate_7d":          round(successful / total, 4) if total else 0.0,
                "avg_duration_s":           avg_duration,
                "pipeline_success_rate_7d": summary.get("pipeline_success_rate_7d", 0.0),
            },
            "recent_runs": runs[:50],
        })
    except Exception as exc:
        logger.error(f"[health/pipeline] {exc}", exc_info=True)
        return _error(f"Pipeline stats failed: {exc}")


@router.post("/retrain", summary="Trigger async model retraining (all models)")
async def trigger_retrain(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Triggers async retraining of both price XGBoost and ROI ensemble models.
    Runs in background thread. Models replaced only if MAE improves.
    Returns: {job_id: str}
    """
    with _jobs_lock:
        running = [j for j in _jobs.values() if j["status"] == "running"]
        if running:
            existing_id = running[0]["job_id"]
            return _ok({
                "job_id":  existing_id,
                "message": f"Retraining already in progress (job {existing_id}).",
                "queued":  False,
            })

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id":     job_id,
            "status":     "queued",
            "model":      "all",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "started_at": None,
            "ended_at":   None,
            "result":     None,
            "error":      None,
        }

    background_tasks.add_task(_run_retrain_job, job_id, "all")
    return _ok({
        "job_id":  job_id,
        "status":  "queued",
        "message": "Retraining job queued. Poll GET /health/retrain/{job_id} for status.",
    })


@router.get("/retrain/{job_id}", summary="Poll retraining job status")
async def retrain_status(job_id: str) -> Dict[str, Any]:
    """Returns current status of a retraining job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _ok(dict(job))


@router.get("/retrain", summary="List all retraining jobs")
async def list_retrain_jobs() -> Dict[str, Any]:
    """List all retraining jobs (most recent 20)."""
    with _jobs_lock:
        jobs = sorted(
            _jobs.values(), key=lambda j: j.get("created_at", ""), reverse=True
        )[:20]
    return _ok({"jobs": list(jobs), "total": len(_jobs)})


# ─────────────────────────────────────────────────────────────────────────────
# NEW v2.0 ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# ── GET /health/metrics ───────────────────────────────────────────────────────

@router.get("/metrics", summary="ML model evaluation metrics (requires pro/admin)")
async def health_metrics(
    _user: Dict = Depends(_require_pro_or_admin),
) -> Dict[str, Any]:
    """
    Returns held-out evaluation metrics for all three ML model families:
    - Style classifier: accuracy, F1, confusion matrix
    - ROI ensemble: MAE, RMSE, R², CI coverage
    - Price forecast: MAPE, MAE INR, directional accuracy

    Results are cached for 24 hours. First call may take 30-60 seconds.
    Requires pro or admin JWT.
    """
    try:
        from analytics.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        metrics   = evaluator.get_all_metrics()
        return _ok(metrics)
    except Exception as exc:
        logger.error(f"[health/metrics] {exc}", exc_info=True)
        return _error(f"Metrics evaluation failed: {exc}")


# ── GET /health/drift ─────────────────────────────────────────────────────────

@router.get("/drift", summary="Real-time model drift detection report")
async def health_drift() -> Dict[str, Any]:
    """
    Checks for model degradation signals across all three model families.
    No authentication required — this is a monitoring endpoint.

    Returns:
      - overall_health: "good" | "degraded" | "critical"
      - style_accuracy_30d: rolling 30-day style correction rate
      - roi_mae_90d: rolling 90-day ROI MAE vs baseline
      - price_mape: per-material MAPE with flagged pairs
      - drift_alerts: list of active alerts
      - recommended_actions: specific remediation steps
    """
    try:
        from analytics.drift_monitor import get_drift_monitor
        monitor = get_drift_monitor()
        report  = monitor.get_full_drift_report()
        return _ok(report)
    except Exception as exc:
        logger.error(f"[health/drift] {exc}", exc_info=True)
        return _error(f"Drift check failed: {exc}")


# ── POST /health/feedback/style-correction ────────────────────────────────────

class StyleCorrectionRequest(BaseModel):
    project_id:        str
    original_style:    str
    corrected_style:   str
    room_type:         str
    confidence:        Optional[float] = 0.5


@router.post("/feedback/style-correction", summary="Log style prediction correction")
async def feedback_style_correction(body: StyleCorrectionRequest) -> Dict[str, Any]:
    """
    Called by the dashboard when a user corrects the detected style.
    Logged to SQLite for drift detection and style classifier retraining.

    Body: {project_id, original_style, corrected_style, room_type, confidence?}
    Returns: {saved: true, log_id: str}
    """
    try:
        from analytics.feedback_collector import get_feedback_collector
        fc     = get_feedback_collector()
        log_id = fc.log_style_correction(
            project_id               = body.project_id,
            original_prediction      = body.original_style,
            user_corrected_to        = body.corrected_style,
            room_type                = body.room_type,
            confidence_at_prediction = body.confidence or 0.5,
        )
        logger.info(
            f"[health/feedback/style] {body.original_style} → {body.corrected_style} "
            f"(project={body.project_id})"
        )
        return _ok({"saved": True, "log_id": log_id})
    except Exception as exc:
        logger.error(f"[health/feedback/style] {exc}", exc_info=True)
        return _error(f"Style correction logging failed: {exc}")


# ── POST /health/feedback/actual-cost ─────────────────────────────────────────

class ActualCostRequest(BaseModel):
    project_id:       str
    actual_total_inr: float
    completion_date:  str
    city:             str
    room_type:        str
    budget_tier:      str


@router.post("/feedback/actual-cost", summary="Log actual renovation cost after completion")
async def feedback_actual_cost(body: ActualCostRequest) -> Dict[str, Any]:
    """
    Called when a user submits their actual renovation cost after completion.
    Creates a gold label for ROI model retraining.

    Body: {project_id, actual_total_inr, completion_date, city, room_type, budget_tier}
    Returns: {saved: true, log_id: str}
    """
    try:
        from analytics.feedback_collector import get_feedback_collector
        fc     = get_feedback_collector()
        log_id = fc.log_actual_renovation_cost(
            project_id       = body.project_id,
            actual_total_inr = body.actual_total_inr,
            completion_date  = body.completion_date,
            city             = body.city,
            room_type        = body.room_type,
            budget_tier      = body.budget_tier,
        )
        logger.info(
            f"[health/feedback/actual-cost] ₹{body.actual_total_inr:,.0f} "
            f"for {body.room_type} in {body.city} (project={body.project_id})"
        )
        return _ok({"saved": True, "log_id": log_id})
    except Exception as exc:
        logger.error(f"[health/feedback/actual-cost] {exc}", exc_info=True)
        return _error(f"Actual cost logging failed: {exc}")


# ── GET /health/feedback/summary ──────────────────────────────────────────────

@router.get("/feedback/summary", summary="Feedback collection counts")
async def feedback_summary() -> Dict[str, Any]:
    """
    Returns counts of all feedback collected so far.

    Returns:
      - style_corrections_count: int
      - boq_corrections_count: int
      - actuals_count: int
      - roi_actuals_count: int
      - data_ready_for_retraining: bool
    """
    try:
        from analytics.feedback_collector import get_feedback_collector
        fc      = get_feedback_collector()
        summary = fc.get_correction_summary()
        return _ok(summary)
    except Exception as exc:
        logger.error(f"[health/feedback/summary] {exc}", exc_info=True)
        return _error(f"Feedback summary failed: {exc}")


# ── POST /health/retrain/{model_name} ─────────────────────────────────────────

@router.post("/retrain/{model_name}", summary="Trigger model retraining (admin only)")
async def trigger_model_retrain(
    model_name: str,
    background_tasks: BackgroundTasks,
    _user: Dict = Depends(_require_admin),
) -> Dict[str, Any]:
    """
    Trigger retraining of a specific model. Admin role required.

    model_name: "roi" | "price" | "all"

    Returns: {triggered: true, model: str, job_id: str}
    """
    valid_models = {"roi", "price", "all"}
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name '{model_name}'. Valid: {', '.join(sorted(valid_models))}",
        )

    # Check if already running
    with _jobs_lock:
        running = [j for j in _jobs.values()
                   if j["status"] == "running" and j.get("model") == model_name]
        if running:
            return _ok({
                "triggered": False,
                "model":     model_name,
                "job_id":    running[0]["job_id"],
                "message":   f"Retraining for '{model_name}' already running.",
            })

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id":     job_id,
            "status":     "queued",
            "model":      model_name,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "started_at": None,
            "ended_at":   None,
            "result":     None,
            "error":      None,
        }

    background_tasks.add_task(_run_retrain_job, job_id, model_name)
    logger.info(f"[health/retrain/{model_name}] Job {job_id} queued by admin")

    return _ok({
        "triggered": True,
        "model":     model_name,
        "job_id":    job_id,
        "message":   f"Retraining for '{model_name}' queued. Poll GET /health/retrain/{job_id}",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Background task runner (shared by all retrain triggers)
# ─────────────────────────────────────────────────────────────────────────────

def _run_retrain_job(job_id: str, model_name: str = "all") -> None:
    """Background task: run retraining and update job registry."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"]     = "running"
            _jobs[job_id]["started_at"] = datetime.now(tz=timezone.utc).isoformat()

    try:
        from services.monitoring.model_retrainer import get_model_retrainer
        retrainer = get_model_retrainer()

        if model_name == "roi":
            result = retrainer.retrain_roi_model()
        elif model_name == "price":
            result = retrainer.retrain_price_model()
        else:
            result = retrainer.retrain_all()

        has_error = bool(result.get("error") or result.get("any_error"))
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update({
                    "status":   "failed" if has_error else "completed",
                    "ended_at": datetime.now(tz=timezone.utc).isoformat(),
                    "result":   result,
                    "error":    result.get("error") or result.get("any_error"),
                })

        logger.info(
            f"[health/retrain] Job {job_id} ({model_name}) completed. "
            f"improved={result.get('any_improved')}"
        )

    except Exception as exc:
        logger.error(f"[health/retrain] Job {job_id} failed: {exc}", exc_info=True)
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update({
                    "status":   "failed",
                    "ended_at": datetime.now(tz=timezone.utc).isoformat(),
                    "error":    str(exc),
                })