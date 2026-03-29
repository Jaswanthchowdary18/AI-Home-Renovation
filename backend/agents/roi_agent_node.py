"""
ARKEN — ROIAgent (LangGraph node) v2.0
=======================================
Agent 5 in the LangGraph multi-agent pipeline.

v2.0 fixes (2026-03-23):
  FIX 1 — Property value bug: floor_area_sqft from CV is the ROOM area (~84 sqft),
           not the flat area. Passing it raw to predict() → prop_val = 84×6×5000 = ₹2.5L.
           Fix: pass current_property_value_inr derived from flat_area × city_psf instead.

  FIX 2 — Stripped roi_output dict: the old roi_output only copied 12 fields from the
           full roi dict, silently dropping rent_before/after, rent_uplift_pct,
           monthly_rental_increase_inr, rupee_breakdown, effective_yield_premium,
           comparable_context etc. — causing ROIPanel to show ₹0 rent.
           Fix: roi_output IS the full roi dict, with a thin compatibility overlay.

  FIX 3 — renovation_scope "not_assessed" sentinel: when the vision pipeline hasn't
           run yet, renovation_scope = "not_assessed" which is not in the multiplier
           table → scope_mult falls back to 1.0 silently. Fix: map sentinel to "partial".

Responsibilities (unchanged):
  - Predicts property value increase from renovation using XGBoost/ensemble model
  - Computes equity gain, rental yield improvement, payback period
  - Adjusts prediction based on user goals from UserGoalAgent
  - Cross-references budget_estimate from BudgetEstimatorAgent
  - Returns structured ROIOutput

Input state keys:  budget_inr, city, room_type, budget_tier, floor_area_sqft,
                   image_features, budget_estimate, user_goals, past_renovation_goals
Output state keys: roi_output, roi_prediction, payback_months, equity_gain_inr
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Flat area estimates by BHK / room type used for property value calculation.
# CRITICAL: floor_area_sqft from CV vision is the single ROOM area (e.g. 84 sqft).
# The property value must be based on the entire FLAT area, not the room area.
# These are conservative 2BHK/3BHK medians from Indian housing datasets.
_FLAT_AREA_BY_ROOM_TYPE: Dict[str, int] = {
    "bedroom":     900,   # 2BHK/3BHK flat — room is one of 2-3 bedrooms
    "kitchen":     850,   # kitchen in a 2BHK/3BHK flat
    "bathroom":    800,   # bathroom in a 2BHK/3BHK flat
    "living_room": 1000,  # living room implies at least a 2BHK
    "full_home":   0,     # use actual floor_area_sqft (user renovating whole flat)
    "dining_room": 900,
    "study":       800,
}

# City PSF (price per sqft) for flat value estimation
# Source: NHB Residex 2024, ANAROCK Q4 2024
_CITY_PSF: Dict[str, int] = {
    "Mumbai": 10323, "Delhi NCR": 5926, "Bangalore": 5387,
    "Chennai": 5383, "Hyderabad": 5000, "Kolkata": 4380,
    "Pune": 6200, "Ahmedabad": 4100, "Surat": 3600,
    "Jaipur": 3800, "Lucknow": 3400, "Chandigarh": 5100,
    "Nagpur": 3200, "Indore": 3500, "Bhopal": 3000,
}

# Valid renovation scope values (anything else → "partial")
_VALID_SCOPES = {"cosmetic_only", "partial", "full_room", "structural_plus"}


class ROIAgentNode:
    """
    LangGraph node wrapping ROIForecastAgent with personalised goal adjustment.
    v2.0: fixes property value calculation and full roi dict passthrough.
    """

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        name = "roi_agent"
        logger.info(f"[{state.get('project_id', '')}] ROIAgent starting")

        try:
            updates = self._predict(state)
        except Exception as e:
            logger.error(f"[roi_agent] Error: {e}", exc_info=True)
            updates = {
                "roi_output": {
                    "roi_pct": 12.0, "equity_gain_inr": 0, "payback_months": 36,
                    "model_type": "fallback", "model_confidence": 0.5,
                    # Ensure rent fields are never missing — zeros are visible/debuggable
                    "rent_before_inr_per_month": 0, "rent_after_inr_per_month": 0,
                    "rent_uplift_pct": 0.0, "monthly_rental_increase_inr": 0,
                    "rental_yield_base_pct": 3.0, "rental_yield_post_pct": 3.0,
                    "pre_reno_value_inr": 0, "post_reno_value_inr": 0,
                },
                "roi_prediction": {
                    "roi_pct": 12.0, "model_type": "fallback", "payback_months": 36,
                    "rent_before_inr_per_month": 0, "rent_after_inr_per_month": 0,
                    "rent_uplift_pct": 0.0, "monthly_rental_increase_inr": 0,
                },
                "payback_months": 36,
                "equity_gain_inr": 0,
                "errors": (state.get("errors") or []) + [f"roi_agent: {e}"],
            }

        elapsed = round(time.perf_counter() - t0, 3)
        timings = dict(state.get("agent_timings") or {})
        timings[name] = elapsed
        updates["agent_timings"] = timings

        completed = list(state.get("completed_agents") or [])
        if name not in completed:
            completed.append(name)
        updates["completed_agents"] = completed

        roi_pred = updates.get("roi_prediction") or {}
        logger.info(
            f"[roi_agent] done in {elapsed}s — "
            f"ROI={roi_pred.get('roi_pct', 0):.1f}% "
            f"rent_uplift={roi_pred.get('rent_uplift_pct', 0):.1f}% "
            f"rent_before=₹{roi_pred.get('rent_before_inr_per_month', 0):,} "
            f"rent_after=₹{roi_pred.get('rent_after_inr_per_month', 0):,} "
            f"equity=₹{updates.get('equity_gain_inr', 0):,}"
        )
        return updates

    def _predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.roi_forecast import ROIForecastAgent

        city      = state.get("city", "Hyderabad")
        room_type = state.get("room_type", "bedroom")
        budget_tier = state.get("budget_tier", "mid")

        # ── FIX 1: Compute property value from FLAT area, not room area ───────
        # floor_area_sqft from the CV pipeline is the detected ROOM area (e.g. 84 sqft
        # for a bedroom). Using it raw → prop_val = 84×6×PSF ≈ ₹2.5–4L — completely wrong.
        # We pass current_property_value_inr directly so _build_report() uses it as-is
        # instead of recomputing from area_sqft.
        dims = state.get("room_dimensions") or {}
        room_area_sqft = float(
            state.get("floor_area_sqft")
            or dims.get("floor_area_sqft", 120)
        )

        # Estimate flat area: full_home uses actual area; single rooms use lookup table
        if room_type == "full_home":
            flat_area_sqft = max(int(room_area_sqft), 800)
        else:
            # Minimum flat area from table; also ensure at least 6× the room area
            table_area = _FLAT_AREA_BY_ROOM_TYPE.get(room_type, 900)
            flat_area_sqft = max(table_area, int(room_area_sqft * 6), 500)

        city_psf = _CITY_PSF.get(city, 5000)
        # Use 10th-percentile area (conservative) to avoid overestimating
        current_property_value_inr = int(flat_area_sqft * city_psf)
        logger.debug(
            f"[roi_agent] Property value: flat_area={flat_area_sqft} sqft "
            f"× PSF={city_psf} = ₹{current_property_value_inr:,} "
            f"(room_area={room_area_sqft} sqft, city={city})"
        )

        # ── Use actual renovation cost from BudgetEstimatorAgent if available ─
        budget_estimate = state.get("budget_estimate") or {}
        if isinstance(budget_estimate, dict) and budget_estimate.get("total_cost_inr"):
            reno_cost = int(budget_estimate["total_cost_inr"])
        else:
            reno_cost = state.get("budget_inr", 750_000)

        # ── Map room_condition from image_features ────────────────────────────
        features = state.get("image_features") or state.get("vision_features") or {}
        if isinstance(features, dict):
            cond_raw = features.get("room_condition", "fair")
        else:
            cond_raw = "fair"
        condition_map = {"new": "good", "good": "good", "fair": "average", "poor": "poor"}
        existing_condition = condition_map.get(cond_raw, "average")

        # ── FIX 3: Map "not_assessed" renovation_scope sentinel to "partial" ──
        raw_scope = (
            state.get("renovation_scope")
            or (features.get("renovation_scope_needed") if isinstance(features, dict) else None)
            or "partial"
        )
        renovation_scope = raw_scope if raw_scope in _VALID_SCOPES else "partial"

        # ── Extract detected materials from CV/vision ─────────────────────────
        cv_features = state.get("cv_features") or {}
        detected_materials = state.get("material_types") or cv_features.get("materials") or []

        agent = ROIForecastAgent()
        roi = agent.predict(
            renovation_cost_inr=reno_cost,
            area_sqft=flat_area_sqft,          # flat area for PSF calc inside agent
            city=city,
            room_type=room_type,
            budget_tier=budget_tier,
            current_property_value_inr=current_property_value_inr,  # FIX 1: bypass area×PSF
            existing_condition=existing_condition,
            renovation_scope=renovation_scope,  # FIX 3: clean scope value
            materials=detected_materials,
            cv_features=cv_features,
        )

        # ── Apply goal-based adjustment ───────────────────────────────────────
        user_goals = state.get("user_goals") or {}
        if isinstance(user_goals, dict):
            goal = user_goals.get("primary_goal", "personal_comfort")
            if goal == "maximise_resale_value":
                roi["roi_pct"] = round(roi["roi_pct"] * 1.05, 2)
            elif goal == "investment_optimisation":
                roi["roi_pct"] = round(roi["roi_pct"] * 1.03, 2)

        # ── FIX 2: roi_output IS the full roi dict ────────────────────────────
        # Old code built a stripped 12-key subset, dropping all the rent fields
        # that ROIPanel depends on (rent_before, rent_after, rent_uplift_pct,
        # monthly_rental_increase_inr, rupee_breakdown, effective_yield_premium).
        # Fix: pass the full dict through and add a thin compat overlay.
        roi_output = {
            **roi,  # full dict — all fields from _build_report() preserved
            # Backward-compat explicit keys (some consumers read roi_output directly)
            "roi_pct":               roi.get("roi_pct", 0),
            "equity_gain_inr":       roi.get("equity_gain_inr", roi.get("net_gain_inr", 0)),
            "payback_months":        roi.get("payback_months", 36),
            "pre_reno_value_inr":    roi.get("pre_reno_value_inr", 0),
            "post_reno_value_inr":   roi.get("post_reno_value_inr", 0),
            "rental_yield_base_pct": roi.get("rental_yield_base_pct", 3.0),
            "rental_yield_post_pct": roi.get("rental_yield_post_pct", 3.0),
            "rental_yield_delta":    roi.get("rental_yield_delta", 0),
            "model_type":            roi.get("model_type", "heuristic"),
            "model_confidence":      roi.get("model_confidence", 0.65),
            "city":                  city,
            "city_tier":             roi.get("city_tier", 2),
            # New rent fields — explicitly surfaced for ROIPanel hero card
            "rent_before_inr_per_month":   roi.get("rent_before_inr_per_month", 0),
            "rent_after_inr_per_month":    roi.get("rent_after_inr_per_month", 0),
            "monthly_rental_increase_inr": roi.get("monthly_rental_increase_inr", 0),
            "rent_uplift_pct":             roi.get("rent_uplift_pct", 0.0),
            "effective_yield_premium":     roi.get("effective_yield_premium", 1.0),
        }

        return {
            "roi_output":    roi_output,
            "roi_prediction": roi_output,   # both point to the same full dict
            "payback_months": roi.get("payback_months", 36),
            "equity_gain_inr": roi.get("equity_gain_inr", roi.get("net_gain_inr", 0)),
        }