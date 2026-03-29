# INSTRUCTIONS: APPEND this block to the BOTTOM of your existing analytics/__init__.py
# Do NOT replace the whole file — only add these lines at the end.

# Analytics v2.0 exports — drift detection, feedback, model evaluation
# ─────────────────────────────────────────────────────────────────────────────

from analytics.drift_monitor import ModelDriftMonitor, get_drift_monitor
from analytics.feedback_collector import (
    FeedbackCollector,
    InsufficientDataError,
    get_feedback_collector,
)
from analytics.model_evaluator import ModelEvaluator

__all__ = [
    # v1.0 analytics
    "InsightDeriver",
    "DecisionScorer",
    "BudgetOptimiser",
    "MarketBenchmarker",
    "InsightFormatter",
    "TrustScoreEngine",
    # v2.0 analytics
    "ModelDriftMonitor",
    "get_drift_monitor",
    "FeedbackCollector",
    "InsufficientDataError",
    "get_feedback_collector",
    "ModelEvaluator",
]
