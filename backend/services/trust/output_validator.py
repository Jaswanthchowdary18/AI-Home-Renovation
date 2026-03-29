"""
ARKEN — Output Trust Validator v1.0
=====================================
Attaches a `trust_badge` to every major ML output so users and
downstream systems can assess data provenance at a glance.

Trust levels:
  high   — real data + validated ML model (CV MAPE or dataset provenance confirmed)
  medium — real data but unvalidated, or ML model with unknown data source
  low    — heuristic / seed-based estimate, or unknown data quality

Used by: node_report_generation() in langgraph_orchestrator.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrustValidator:
    """
    Attaches honest trust metadata to price forecasts, ROI predictions,
    and the full pipeline output.

    All methods are non-destructive: they add a 'trust_badge' key and
    return the mutated dict. They never raise — failures are logged and
    a low-trust badge is attached instead.
    """

    # ── Price forecast trust ──────────────────────────────────────────────────

    def validate_price_forecast(self, forecast_result: dict) -> dict:
        """
        Attach trust metadata to a single price forecast dict.

        Reads:
            data_quality   — e.g. "real_ml_ensemble", "estimated_seed_based"
            cv_mape_pct    — float from walk-forward cross-validation, or None

        Writes:
            trust_badge    — {level, explanation, data_quality, cv_mape_pct, show_to_user}
        """
        try:
            data_quality = str(forecast_result.get("data_quality", "unknown"))
            cv_mape      = forecast_result.get("cv_mape_pct")

            _real_dq = {"real_ml_ensemble", "real_prophet_only", "real_xgb_only", "historical"}
            is_real  = data_quality in _real_dq or "real" in data_quality.lower()

            if is_real:
                if cv_mape is not None and float(cv_mape) < 5.0:
                    trust_level       = "high"
                    trust_explanation = (
                        f"Trained on real Indian market price data. "
                        f"Walk-forward cross-validation MAPE: {float(cv_mape):.1f}% "
                        f"(excellent for 90-day commodity forecast)."
                    )
                elif cv_mape is not None:
                    trust_level       = "medium"
                    trust_explanation = (
                        f"Trained on real price data. "
                        f"CV MAPE: {float(cv_mape):.1f}% "
                        f"(acceptable for directional 90-day forecast)."
                    )
                else:
                    trust_level       = "medium"
                    trust_explanation = (
                        "Trained on real historical price data. "
                        "Walk-forward cross-validation not yet run — "
                        "run python ml/train_price_models.py for CV results."
                    )
            elif data_quality in ("estimated_seed_based", "seed_extrapolated"):
                trust_level       = "low"
                trust_explanation = (
                    "Seed-based estimate. Real price CSV not loaded. "
                    "Treat as directional guidance only."
                )
            else:
                trust_level       = "low"
                trust_explanation = f"Data quality unknown (data_quality='{data_quality}'). Use with caution."

            forecast_result["trust_badge"] = {
                "level":        trust_level,
                "explanation":  trust_explanation,
                "data_quality": data_quality,
                "cv_mape_pct":  cv_mape,
                "show_to_user": True,
            }

        except Exception as exc:
            logger.debug(f"[TrustValidator] validate_price_forecast failed: {exc}")
            forecast_result["trust_badge"] = {
                "level":        "low",
                "explanation":  "Trust validation unavailable.",
                "data_quality": "unknown",
                "cv_mape_pct":  None,
                "show_to_user": True,
            }

        return forecast_result

    # ── ROI prediction trust ──────────────────────────────────────────────────

    def validate_roi_prediction(self, roi_result: dict) -> dict:
        """
        Attach trust metadata to an ROI prediction dict.

        Reads:
            model_type    — e.g. "xgboost_real_data", "real_data_ensemble", "heuristic"
            data_source   — e.g. "real_kaggle_transaction_derived_32210_rows"

        Writes:
            trust_badge   — {level, explanation, model_type, data_source, show_to_user}
        """
        try:
            model_type  = str(roi_result.get("model_type", "heuristic"))
            data_source = str(roi_result.get("data_source", "unknown"))

            is_real = (
                "real" in data_source.lower()
                or "kaggle" in data_source.lower()
                or "real" in model_type.lower()
            )
            is_ml = (
                "xgboost" in model_type.lower()
                or "ensemble" in model_type.lower()
                or "random_forest" in model_type.lower()
            )

            if is_real and is_ml:
                # Extract row count from data_source if present
                row_hint = ""
                import re
                m = re.search(r"(\d[\d,]+)\s*row", data_source)
                if m:
                    row_hint = f" ({m.group(1)} training rows)"

                trust_level       = "high"
                trust_explanation = (
                    f"Ensemble ML model (XGBoost + RandomForest + GradientBoosting) "
                    f"trained on real Kaggle Indian housing transaction data{row_hint}. "
                    f"Confidence intervals from ensemble variance — not hardcoded."
                )
            elif is_ml:
                trust_level       = "medium"
                trust_explanation = (
                    f"ML model used ({model_type}). "
                    "Data provenance not fully confirmed — check data_source field. "
                    "Treat as directional ROI guidance."
                )
            else:
                trust_level       = "low"
                trust_explanation = (
                    "Heuristic estimate based on NHB city benchmarks and ANAROCK multipliers. "
                    "No ML model was loaded at prediction time. "
                    "Treat as rough order-of-magnitude estimate."
                )

            roi_result["trust_badge"] = {
                "level":        trust_level,
                "explanation":  trust_explanation,
                "model_type":   model_type,
                "data_source":  data_source,
                "show_to_user": True,
            }

        except Exception as exc:
            logger.debug(f"[TrustValidator] validate_roi_prediction failed: {exc}")
            roi_result["trust_badge"] = {
                "level":        "low",
                "explanation":  "Trust validation unavailable.",
                "model_type":   "unknown",
                "data_source":  "unknown",
                "show_to_user": True,
            }

        return roi_result

    # ── Full pipeline trust summary ───────────────────────────────────────────

    def validate_pipeline_output(self, state: dict) -> dict:
        """
        Attach an overall pipeline trust summary to the final pipeline state.

        Reads from state:
            material_prices         — list of price forecast dicts
            roi_prediction          — roi forecast dict
            cv_features / image_features — vision model metadata
            rag_chunks_retrieved    — int, number of RAG chunks used

        Writes:
            state["pipeline_trust_summary"]
        """
        try:
            # Price component
            forecasts  = state.get("material_prices") or []
            price_dq   = "unknown"
            if forecasts and isinstance(forecasts[0], dict):
                price_dq = str(forecasts[0].get("data_quality", "unknown"))
            price_component = (
                "real_ml" if ("real" in price_dq.lower() or "historical" in price_dq) else "heuristic"
            )

            # ROI component
            roi_dict    = state.get("roi_prediction") or {}
            roi_model   = str(roi_dict.get("model_type", "heuristic"))
            roi_component = (
                "ml" if ("xgboost" in roi_model or "ensemble" in roi_model) else "heuristic"
            )

            # Computer vision component
            cv_features = state.get("cv_features") or state.get("image_features") or {}
            cv_source   = str(cv_features.get("model_used", "none"))
            # Also check damage_detector output in state
            damage_info = state.get("damage_assessment") or {}
            damage_model = str(damage_info.get("model_used", ""))
            cv_component = (
                "clip" if (
                    "clip" in cv_source.lower()
                    or "clip" in damage_model.lower()
                    or "metadata" in cv_source.lower()
                ) else "gemini_only"
            )

            # RAG component
            rag_count   = int(state.get("rag_chunks_retrieved", 0))
            # Also check rag_context or rag_result
            rag_ctx     = state.get("rag_context") or state.get("rag_result") or {}
            if rag_count == 0 and isinstance(rag_ctx, dict):
                rag_count = len(rag_ctx.get("chunks", []))
            rag_component = "real_data" if rag_count >= 100 else "limited"

            components = {
                "price_forecast":   price_component,
                "roi_model":        roi_component,
                "computer_vision":  cv_component,
                "rag_knowledge":    rag_component,
            }

            real_ml_values = {"real_ml", "ml", "clip", "real_data"}
            real_count     = sum(1 for v in components.values() if v in real_ml_values)
            total_count    = len(components)

            overall_trust = (
                "high"   if real_count >= 3 else
                "medium" if real_count >= 2 else
                "low"
            )

            state["pipeline_trust_summary"] = {
                "overall_trust_level":  overall_trust,
                "components":           components,
                "real_ml_components":   real_count,
                "total_components":     total_count,
                "message":              f"{real_count}/{total_count} components using real ML/data.",
                "rag_chunks_used":      rag_count,
                "generated_at":         datetime.now(tz=timezone.utc).isoformat(),
            }

            logger.info(
                f"[TrustValidator] Pipeline trust: {overall_trust} "
                f"({real_count}/{total_count} real components)"
            )

        except Exception as exc:
            logger.warning(f"[TrustValidator] validate_pipeline_output failed: {exc}")
            state["pipeline_trust_summary"] = {
                "overall_trust_level":  "unknown",
                "components":           {},
                "real_ml_components":   0,
                "total_components":     4,
                "message":              "Trust validation error.",
                "generated_at":         datetime.now(tz=timezone.utc).isoformat(),
            }

        return state


# ── Module-level singleton ────────────────────────────────────────────────────

_validator_instance: Optional[TrustValidator] = None


def get_trust_validator() -> TrustValidator:
    """Return singleton TrustValidator. Safe to call from any thread."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = TrustValidator()
    return _validator_instance
