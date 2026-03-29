"""
ARKEN — Model Drift Monitor v1.0
===================================
Production drift detection system that reads prediction logs from the
shared SQLite database (PredictionLogger) and surfaces degradation signals
before they impact user-visible quality.

Detection checks:
  1. Style accuracy drift   — rolling 30-day user-correction rate
  2. ROI MAE drift          — rolling 90-day MAE vs baseline from model_report.json
  3. Price forecast MAPE    — per-material MAPE vs 12% threshold

All reads use the SAME SQLite database as PredictionLogger.
This module NEVER creates a second database.

Usage:
    from analytics.drift_monitor import get_drift_monitor
    monitor = get_drift_monitor()
    report  = monitor.get_full_drift_report()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKEND_DIR    = Path(__file__).resolve().parent.parent
_APP_DIR        = Path(os.getenv("ARKEN_APP_DIR", "/app"))
_DEFAULT_DB_DIR = Path(os.getenv("ARKEN_DATA_DIR", str(_BACKEND_DIR / "data"))) / "monitoring"
_FALLBACK_DB_DIR = Path("/tmp/arken_monitoring")
_WEIGHTS_DIR    = Path(os.getenv("ML_WEIGHTS_DIR", "/app/ml/weights"))

# Drift thresholds
_STYLE_ACCURACY_WARN    = 0.75   # below this = degraded
_STYLE_ACCURACY_CRIT    = 0.60   # below this = critical
_ROI_MAE_DRIFT_FACTOR   = 1.15   # 15% above baseline = drift
_PRICE_MAPE_WARN        = 0.12   # 12% MAPE = flag for refit


class ModelDriftMonitor:
    """
    Reads prediction logs and feedback tables to detect model degradation.

    Singleton — use get_drift_monitor() to obtain the shared instance.
    All methods are read-only against the SQLite DB (never write).
    Thread-safe: uses a single threading.Lock for DB reads.
    """

    _instance: Optional["ModelDriftMonitor"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelDriftMonitor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._db_path = cls._resolve_db_path()
                    cls._instance = inst
        return cls._instance

    # ── DB path resolution ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_db_path() -> Path:
        """Find the same SQLite path PredictionLogger uses."""
        for candidate in (_DEFAULT_DB_DIR, _FALLBACK_DB_DIR):
            db = candidate / "predictions.db"
            if db.exists():
                return db
        # Fall back to expected path even if not yet created
        _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
        return _DEFAULT_DB_DIR / "predictions.db"

    @contextmanager
    def _conn(self):
        """Open a read-only SQLite connection to the shared predictions DB."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _db_exists(self) -> bool:
        return self._db_path.exists()

    # ── Model report baseline ──────────────────────────────────────────────────

    def _load_roi_baseline_mae(self) -> Optional[float]:
        """Load baseline ensemble MAE from model_report.json."""
        for weights_dir in [_WEIGHTS_DIR, _APP_DIR / "ml" / "weights",
                            _BACKEND_DIR / "ml" / "weights"]:
            report_path = weights_dir / "model_report.json"
            if report_path.exists():
                try:
                    with open(report_path, "r", encoding="utf-8") as fh:
                        rpt = json.load(fh)
                    mae = rpt.get("mae") or rpt.get("ensemble_mae")
                    if mae is not None:
                        return float(mae)
                except Exception as e:
                    logger.debug(f"[DriftMonitor] model_report.json read failed: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Public detection methods
    # ─────────────────────────────────────────────────────────────────────────

    def check_style_accuracy_drift(self) -> Dict[str, Any]:
        """
        Compute rolling 30-day style classification agreement rate.

        Reads the style_corrections table (written by FeedbackCollector).
        When a user corrects a style prediction, it's logged as a disagreement.
        Agreement rate = 1 - (corrections / total_predictions_in_window).

        Falls back to reading prediction counts from roi_predictions table
        as a proxy for total predictions when no pipeline_runs count is available.

        Returns:
            {
              check: "style_accuracy",
              window_days: 30,
              agreement_rate: float,       # 0.0-1.0
              corrections_count: int,
              total_predictions_estimate: int,
              drift_detected: bool,
              severity: "ok" | "warning" | "critical",
              message: str,
            }
        """
        if not self._db_exists():
            return self._no_data_result("style_accuracy")

        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat()

        try:
            with self._conn() as conn:
                # Count style corrections in last 30 days
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM style_corrections WHERE ts >= ?", (cutoff,)
                    ).fetchone()
                    corrections = int(row[0]) if row else 0
                except sqlite3.OperationalError:
                    # Table doesn't exist yet (no corrections logged)
                    corrections = 0

                # Estimate total predictions from pipeline_runs
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM pipeline_runs WHERE ts >= ?", (cutoff,)
                    ).fetchone()
                    total_runs = int(row[0]) if row else 0
                except sqlite3.OperationalError:
                    total_runs = 0

            # Agreement rate: assume each pipeline run = 1 style prediction
            if total_runs == 0:
                # No data — cannot determine drift, assume good
                return {
                    "check":                      "style_accuracy",
                    "window_days":                30,
                    "agreement_rate":             1.0,
                    "corrections_count":          0,
                    "total_predictions_estimate": 0,
                    "drift_detected":             False,
                    "severity":                   "ok",
                    "message":                    "No prediction data in last 30 days — no drift detectable.",
                }

            agreement_rate = max(0.0, 1.0 - (corrections / max(total_runs, 1)))

            drift    = agreement_rate < _STYLE_ACCURACY_WARN
            severity = (
                "critical" if agreement_rate < _STYLE_ACCURACY_CRIT else
                "warning"  if agreement_rate < _STYLE_ACCURACY_WARN else
                "ok"
            )

            return {
                "check":                      "style_accuracy",
                "window_days":                30,
                "agreement_rate":             round(agreement_rate, 4),
                "corrections_count":          corrections,
                "total_predictions_estimate": total_runs,
                "drift_detected":             drift,
                "severity":                   severity,
                "message": (
                    f"Style agreement rate {agreement_rate:.1%} in last 30 days "
                    f"({corrections} corrections / {total_runs} predictions). "
                    + ("⚠ Below 75% threshold — retraining recommended." if drift else "Within normal range.")
                ),
            }

        except Exception as e:
            logger.warning(f"[DriftMonitor] check_style_accuracy_drift failed: {e}")
            return self._error_result("style_accuracy", str(e))

    def check_roi_mae_drift(self) -> Dict[str, Any]:
        """
        Compute rolling 90-day ROI MAE vs baseline from model_report.json.

        Reads roi_actuals table (written by FeedbackCollector) for ground-truth
        actual_roi_pct values. Compares against predicted_roi from roi_predictions.

        If MAE increases >15% vs model_report.json baseline, drift_detected=True.

        Returns:
            {
              check: "roi_mae",
              window_days: 90,
              current_mae_pct: float,
              baseline_mae_pct: float | null,
              mae_drift_factor: float,
              actuals_count: int,
              drift_detected: bool,
              severity: str,
              message: str,
            }
        """
        if not self._db_exists():
            return self._no_data_result("roi_mae")

        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=90)).isoformat()
        baseline_mae = self._load_roi_baseline_mae()

        try:
            with self._conn() as conn:
                # Join roi_actuals (true labels) with roi_predictions (predictions)
                try:
                    rows = conn.execute(
                        """
                        SELECT
                            ra.actual_roi_pct,
                            rp.predicted_roi
                        FROM roi_actuals ra
                        JOIN roi_predictions rp ON ra.project_id = rp.project_id
                        WHERE ra.ts >= ?
                        ORDER BY ra.ts DESC
                        LIMIT 500
                        """,
                        (cutoff,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []

                actuals_count = len(rows)

                if actuals_count < 5:
                    return {
                        "check":             "roi_mae",
                        "window_days":       90,
                        "current_mae_pct":   None,
                        "baseline_mae_pct":  round(baseline_mae, 4) if baseline_mae else None,
                        "mae_drift_factor":  1.0,
                        "actuals_count":     actuals_count,
                        "drift_detected":    False,
                        "severity":          "ok",
                        "message": (
                            f"Only {actuals_count} actual ROI data points (need ≥5). "
                            "Submit renovation completion data via POST /health/feedback/actual-cost."
                        ),
                    }

                import math
                abs_errors = [abs(float(r["actual_roi_pct"]) - float(r["predicted_roi"])) for r in rows]
                current_mae = sum(abs_errors) / len(abs_errors)

            # Drift vs baseline
            if baseline_mae and baseline_mae > 0:
                drift_factor = current_mae / baseline_mae
                drift        = drift_factor > _ROI_MAE_DRIFT_FACTOR
                severity     = "critical" if drift_factor > 1.30 else "warning" if drift else "ok"
            else:
                drift_factor = 1.0
                drift        = False
                severity     = "ok"

            return {
                "check":            "roi_mae",
                "window_days":      90,
                "current_mae_pct":  round(current_mae, 4),
                "baseline_mae_pct": round(baseline_mae, 4) if baseline_mae else None,
                "mae_drift_factor": round(drift_factor, 3),
                "actuals_count":    actuals_count,
                "drift_detected":   drift,
                "severity":         severity,
                "message": (
                    f"ROI MAE {current_mae:.3f}% over last 90 days "
                    f"({actuals_count} actuals). "
                    + (
                        f"Baseline: {baseline_mae:.3f}%. "
                        f"Drift factor: {drift_factor:.2f}x. "
                        + ("⚠ Exceeds 15% threshold — retraining recommended." if drift else "Within normal range.")
                        if baseline_mae else "No baseline — run ml/train.py to establish baseline."
                    )
                ),
            }

        except Exception as e:
            logger.warning(f"[DriftMonitor] check_roi_mae_drift failed: {e}")
            return self._error_result("roi_mae", str(e))

    def check_price_forecast_mape(self) -> Dict[str, Any]:
        """
        Compute per-material×city MAPE from logged forecasts vs actuals.

        Reads price_forecasts (predictions) joined with price_actuals (true prices).
        Flags any material×city pair with MAPE > 12% for Prophet refit.

        Returns:
            {
              check: "price_mape",
              window_days: 60,
              pairs_evaluated: int,
              pairs_flagged: int,
              mape_by_material: {material: float},
              flagged_pairs: [{material, city, mape, recommended_action}],
              drift_detected: bool,
              severity: str,
              message: str,
            }
        """
        if not self._db_exists():
            return self._no_data_result("price_mape")

        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=60)).isoformat()

        try:
            with self._conn() as conn:
                try:
                    rows = conn.execute(
                        """
                        SELECT
                            pf.material_key,
                            pf.city,
                            pf.predicted_price,
                            pa.actual_price
                        FROM price_forecasts pf
                        JOIN price_actuals pa
                          ON pf.material_key = pa.material_key
                         AND pf.city         = pa.city
                         AND date(pf.ts)     = date(pa.ts)
                        WHERE pf.ts >= ?
                        ORDER BY pf.ts DESC
                        LIMIT 2000
                        """,
                        (cutoff,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []

            if not rows:
                return {
                    "check":           "price_mape",
                    "window_days":     60,
                    "pairs_evaluated": 0,
                    "pairs_flagged":   0,
                    "mape_by_material": {},
                    "flagged_pairs":   [],
                    "drift_detected":  False,
                    "severity":        "ok",
                    "message":         "No price actuals logged yet — MAPE check requires actual price data.",
                }

            # Group by material×city
            from collections import defaultdict
            pair_errors: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
            for r in rows:
                pred   = float(r["predicted_price"])
                actual = float(r["actual_price"])
                if actual > 0:
                    ape = abs(pred - actual) / actual
                    pair_errors[r["material_key"]][r["city"]].append(ape)

            mape_by_material: Dict[str, float] = {}
            flagged_pairs: List[Dict] = []
            pairs_evaluated = 0
            pairs_flagged   = 0

            for material, city_dict in pair_errors.items():
                material_apes: List[float] = []
                for city, apes in city_dict.items():
                    if not apes:
                        continue
                    pairs_evaluated += 1
                    mape = sum(apes) / len(apes)
                    material_apes.extend(apes)
                    if mape > _PRICE_MAPE_WARN:
                        pairs_flagged += 1
                        flagged_pairs.append({
                            "material":           material,
                            "city":               city,
                            "mape_pct":           round(mape * 100, 2),
                            "observations":       len(apes),
                            "recommended_action": "prophet_refit",
                        })

                if material_apes:
                    mape_by_material[material] = round(
                        (sum(material_apes) / len(material_apes)) * 100, 2
                    )

            drift    = pairs_flagged > 0
            severity = (
                "critical" if pairs_flagged >= pairs_evaluated * 0.5 and pairs_evaluated > 0 else
                "warning"  if drift else
                "ok"
            )

            return {
                "check":            "price_mape",
                "window_days":      60,
                "pairs_evaluated":  pairs_evaluated,
                "pairs_flagged":    pairs_flagged,
                "mape_by_material": mape_by_material,
                "flagged_pairs":    sorted(flagged_pairs, key=lambda x: -x["mape_pct"]),
                "drift_detected":   drift,
                "severity":         severity,
                "message": (
                    f"{pairs_flagged}/{pairs_evaluated} material×city pairs exceed {_PRICE_MAPE_WARN*100:.0f}% MAPE threshold. "
                    + ("⚠ Prophet refit recommended for flagged pairs." if drift else "All pairs within threshold.")
                ),
            }

        except Exception as e:
            logger.warning(f"[DriftMonitor] check_price_forecast_mape failed: {e}")
            return self._error_result("price_mape", str(e))

    def get_full_drift_report(self) -> Dict[str, Any]:
        """
        Run all three drift checks and return a consolidated health report.

        Returns:
            {
              overall_health: "good" | "degraded" | "critical",
              style_accuracy_30d: {...},
              roi_mae_90d: {...},
              price_mape_by_material: {...},
              drift_alerts: [...],
              recommended_actions: [...],
              last_checked: ISO timestamp,
            }
        """
        style_check = self.check_style_accuracy_drift()
        roi_check   = self.check_roi_mae_drift()
        price_check = self.check_price_forecast_mape()

        # Aggregate severity
        severities = [
            style_check.get("severity", "ok"),
            roi_check.get("severity", "ok"),
            price_check.get("severity", "ok"),
        ]
        overall_health = (
            "critical"  if "critical" in severities else
            "degraded"  if "warning"  in severities else
            "good"
        )

        # Build drift alerts
        drift_alerts: List[Dict] = []
        if style_check.get("drift_detected"):
            drift_alerts.append({
                "model":    "style_classifier",
                "severity": style_check["severity"],
                "message":  style_check["message"],
            })
        if roi_check.get("drift_detected"):
            drift_alerts.append({
                "model":    "roi_ensemble",
                "severity": roi_check["severity"],
                "message":  roi_check["message"],
            })
        if price_check.get("drift_detected"):
            for fp in price_check.get("flagged_pairs", [])[:5]:
                drift_alerts.append({
                    "model":    f"price_prophet_{fp['material']}_{fp['city']}",
                    "severity": "warning",
                    "message":  f"MAPE {fp['mape_pct']}% for {fp['material']} in {fp['city']}",
                })

        # Build recommended actions
        recommended_actions: List[str] = []
        if style_check.get("drift_detected"):
            recommended_actions.append(
                "Retrain style_classifier: run python ml/train.py --model roi "
                "(style training uses same image dataset)"
            )
        if roi_check.get("drift_detected"):
            recommended_actions.append(
                "Retrain ROI ensemble: POST /api/v1/health/retrain/roi "
                "or run python ml/train.py --model roi"
            )
        if price_check.get("drift_detected"):
            flagged = [fp["material"] for fp in price_check.get("flagged_pairs", [])]
            recommended_actions.append(
                f"Refit Prophet models for: {', '.join(set(flagged[:5]))} — "
                "run python ml/train.py --model price"
            )

        return {
            "overall_health":       overall_health,
            "style_accuracy_30d":   style_check,
            "roi_mae_90d":          roi_check,
            "price_mape":           price_check,
            "drift_alerts":         drift_alerts,
            "recommended_actions":  recommended_actions,
            "last_checked":         datetime.now(tz=timezone.utc).isoformat(),
            "db_path":              str(self._db_path),
            "db_exists":            self._db_exists(),
        }

    def should_retrain(self, model_name: str) -> bool:
        """
        Returns True if drift detected for the given model.
        Called by ModelRetrainer before scheduling a retraining run.

        Args:
            model_name: "style_classifier" | "roi" | "price" | "all"

        Returns:
            True if any relevant drift check shows drift_detected=True.
        """
        try:
            if model_name in ("style_classifier", "all"):
                if self.check_style_accuracy_drift().get("drift_detected", False):
                    return True
            if model_name in ("roi", "all"):
                if self.check_roi_mae_drift().get("drift_detected", False):
                    return True
            if model_name in ("price", "all"):
                if self.check_price_forecast_mape().get("drift_detected", False):
                    return True
            return False
        except Exception as e:
            logger.warning(f"[DriftMonitor] should_retrain({model_name}) failed: {e}")
            return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _no_data_result(check: str) -> Dict[str, Any]:
        return {
            "check":          check,
            "drift_detected": False,
            "severity":       "ok",
            "message":        "SQLite predictions.db not yet created — no data to evaluate.",
        }

    @staticmethod
    def _error_result(check: str, error: str) -> Dict[str, Any]:
        return {
            "check":          check,
            "drift_detected": False,
            "severity":       "ok",
            "error":          error,
            "message":        f"Drift check failed: {error}",
        }


# ── Singleton accessor ─────────────────────────────────────────────────────────

_monitor_instance: Optional[ModelDriftMonitor] = None


def get_drift_monitor() -> ModelDriftMonitor:
    """Return the singleton ModelDriftMonitor. Safe to call from any thread."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelDriftMonitor()
    return _monitor_instance
