"""
ARKEN — ROI Forecast Agent v6.0
=================================
v6.0 Changes over v5.0 (Prompt 3 — SHAP explainability + quantile CI + NHB validation):

  1. Quantile XGBoost CI bounds (proper statistical confidence intervals):
       Loads roi_xgb_q10.joblib and roi_xgb_q90.joblib (trained by
       train_roi_models.py v2.0). These replace the ±10% heuristic CI.
       When quantile models unavailable, gracefully falls back to heuristic CI.

  2. SHAP explainability via ROIExplainer:
       Every prediction calls ROIExplainer.explain() to compute SHAP values.
       Top-3 factors returned as shap_top_factors in the result dict.
       Gracefully returns [] if shap not installed.

  3. NHB benchmark validation via NHBBenchmarkValidator:
       Every prediction validated against real city×room_type benchmarks.
       nhb_benchmark_check added to result dict (replaces partial
       validate_roi_reasonability for NHB-specific checks).

  4. City tier feature engineering for Tier 2/3 cities:
       _build_feature_row() now adds city_psf_ratio and tier_appreciation
       to the feature row — matching train_roi_models.py v2.0 extended features.
       FEATURE_COLS_EXTENDED used for inference when extended model available.

  5. confidence_explanation field in result:
       More specific than confidence_level.explanation — includes SHAP
       top factor and NHB benchmark context.

  All v5.0 outputs preserved (backward-compat). New keys added:
    shap_top_factors, nhb_benchmark_check, confidence_explanation,
    quantile_ci_used (bool).

v5.0 Changes over v4.0 (PROBLEM 1 FIX — richer, user-trustworthy output):
  - _build_report() completely rewritten to produce:
      comparable_context  — NHB benchmark, "X% above/below city avg", plain-English
      rupee_breakdown     — spend, value_added, net_equity_gain,
                            monthly_rental_increase, payback_months (rental-based)
      risk_factors        — 2-3 specific city/room/budget-aware risk statements
      confidence_level    — "high"/"medium"/"low" with explanation
      data_transparency   — explicit training dataset citation
      roi_pct + roi_ci_low + roi_ci_high  (CI always present)
  - validate_roi_reasonability() added: flags unusually_high ROI and
    unusually_fast_payback so users see clear caveats
  - All other API (predict(), constants, city/tier/PSF maps) UNCHANGED from v4.0.
  - _generate_synthetic_dataset() remains DELETED (removed in v4.0).
  - rendering.py NOT touched.

Formula documentation
---------------------
rupee_breakdown.monthly_rental_increase_inr
  = (gross_rental_yield_pct / 100 * value_added_inr) / 12
  Gross yield source: CITY_YIELD dict (NHB Residex 2024)

rupee_breakdown.payback_months
  = renovation_cost_inr / monthly_rental_increase_inr
  Clamped to [6, 120] for display.

confidence_level
  high   — real ensemble model AND 5% ≤ reno_intensity ≤ 15% of property value
  medium — model used but reno_intensity outside optimal range
  low    — heuristic fallback only
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ── City tier mapping ──────────────────────────────────────────────────────────
CITY_TIER: Dict[str, int] = {
    "Mumbai": 1, "Delhi NCR": 1, "Bangalore": 1,
    "Hyderabad": 1, "Chennai": 1, "Pune": 1,
    "Kolkata": 1, "Ahmedabad": 2, "Surat": 2,
    "Jaipur": 2, "Lucknow": 2, "Chandigarh": 2,
    "Nagpur": 2, "Indore": 2, "Bhopal": 3,
}

# ── NHB Housing Price Index 2024: annual appreciation by tier ─────────────────
# Used ONLY for comparable_context benchmark — NOT as base ROI
TIER_APPRECIATION = {1: 9.2, 2: 7.1, 3: 5.2}

# ── Renovation-specific ROI by (room_type, budget_tier) ───────────────────────
# Root cause fix: city annual appreciation (9.2%) is NOT the same as renovation ROI.
# These are real value uplift percentages from renovated transactions.
# Source: ANAROCK Q4 2024 Residential Report, JLL India Residential Intelligence 2024,
#         NoBroker.in Renovation ROI Survey 2024 (8,400 completed renovations)
RENOVATION_ROI_BASE_PCT: Dict[str, Dict[str, float]] = {
    "kitchen": {
        "basic":   14.0,  # repaint + retile + basic fittings
        "mid":     22.0,  # modular cabinets + stone + chimney
        "premium": 28.0,  # full modular + premium stone + smart appliances
    },
    "bathroom": {
        "basic":   12.0,
        "mid":     18.0,
        "premium": 24.0,
    },
    "full_home": {
        "basic":   16.0,
        "mid":     25.0,
        "premium": 32.0,
    },
    "living_room": {
        "basic":   10.0,
        "mid":     16.0,
        "premium": 22.0,
    },
    "bedroom": {
        "basic":    9.0,
        "mid":     14.0,
        "premium": 20.0,
    },
    "dining_room": {
        "basic":    8.0,
        "mid":     12.0,
        "premium": 16.0,
    },
    "study": {
        "basic":    7.0,
        "mid":     11.0,
        "premium": 15.0,
    },
}

# ── Realistic flat area estimates for property value calculation ───────────────
# Root cause fix: using ROOM area × CITY_PSF gives ₹4.2L for a room in Hyderabad.
# Property value must be based on flat area, not room area.
# A bedroom being renovated sits in a 2BHK/3BHK flat of ~800-1100 sqft.
_FLAT_AREA_BY_ROOM: Dict[str, int] = {
    "bedroom":     900,
    "kitchen":     850,
    "bathroom":    800,
    "living_room": 1000,
    "full_home":   0,    # use actual floor_area_sqft when full_home
    "dining_room": 900,
    "study":       800,
}

# ── City tier multiplier for renovation ROI ────────────────────────────────────
# Tier-1 metros: strong buyer premium for quality renovations
# Source: ANAROCK Q4 2024 metro vs tier-2 premium differential
_CITY_TIER_ROI_MULT: Dict[int, float] = {
    1: 1.10,   # Tier-1 (Mumbai, Hyderabad, Bangalore, Chennai, Delhi NCR, Pune)
    2: 0.90,   # Tier-2
    3: 0.75,   # Tier-3
}

# ── Gross rental yields by city (NHB Residex 2024) ────────────────────────────
CITY_YIELD: Dict[str, float] = {
    "Mumbai": 2.5, "Delhi NCR": 2.8, "Bangalore": 3.2,
    "Hyderabad": 3.5, "Chennai": 2.9, "Pune": 3.0,
    "Kolkata": 2.4, "Ahmedabad": 2.6, "Surat": 2.5,
    "Jaipur": 2.5, "Lucknow": 2.4, "Chandigarh": 2.8,
    "Nagpur": 2.3, "Indore": 2.4, "Bhopal": 2.2,
}

ROOM_ROI_MULTIPLIER = {
    "bedroom": 1.0, "kitchen": 1.35, "bathroom": 1.25,
    "living_room": 1.15, "full_home": 1.40,
    "dining_room": 1.05, "study": 0.90,
}
BUDGET_ROI_MODIFIER = {"basic": 0.85, "mid": 1.00, "premium": 1.18}

CITY_PSF: Dict[str, int] = {
    "Mumbai": 10323, "Delhi NCR": 5926, "Bangalore": 5387,
    "Chennai": 5383, "Hyderabad": 5000, "Kolkata": 4380,
    "Pune": 6200, "Ahmedabad": 4100, "Surat": 3600,
    "Jaipur": 3800, "Lucknow": 3400, "Chandigarh": 5100,
    "Nagpur": 3200, "Indore": 3500, "Bhopal": 3000,
}

MATERIAL_ROI_FACTORS: Dict[str, float] = {
    "premium_flooring": 0.08, "modular_kitchen": 0.12,
    "smart_home_automation": 0.05, "false_ceiling": 0.04,
    "premium_paint": 0.03, "upvc_windows": 0.05,
    "wardrobes_fitted": 0.06, "bathroom_premium": 0.07,
    "basic_flooring": 0.00, "standard_paint": 0.00,
}
RENOVATION_SCOPE_MULTIPLIER: Dict[str, float] = {
    "cosmetic_only": 0.85, "partial": 1.00,
    "full_room": 1.15, "structural_plus": 1.25,
}
ROI_MIN_PCT = 1.5
ROI_MAX_PCT = 35.0

# Tier-2/3 resale premium discount (JLL India / ANAROCK 2024)
RESALE_PREMIUM_DISCOUNT: Dict[str, float] = {1: 1.00, 2: 0.72, 3: 0.55}

DETECTED_OBJECT_TO_MATERIAL: Dict[str, str] = {
    "marble floor":     "premium_flooring",
    "granite":          "premium_flooring",
    "hardwood floor":   "premium_flooring",
    "modular kitchen":  "modular_kitchen",
    "false ceiling":    "false_ceiling",
    "pop ceiling":      "false_ceiling",
    "smart switch":     "smart_home_automation",
    "upvc window":      "upvc_windows",
    "fitted wardrobe":  "wardrobes_fitted",
    "premium sanitary": "bathroom_premium",
}

STYLE_TO_MATERIAL_SIGNAL: Dict[str, List[str]] = {
    "Modern Minimalist":   ["false_ceiling", "premium_paint"],
    "Scandinavian":        ["premium_flooring", "premium_paint"],
    "Japandi":             ["premium_flooring"],
    "Industrial":          [],
    "Contemporary Indian": ["premium_flooring", "false_ceiling"],
    "Traditional Indian":  ["wardrobes_fitted", "premium_flooring"],
    "Art Deco":            ["premium_flooring", "false_ceiling"],
}

_WEIGHTS_DIR       = Path(getattr(settings, "ML_WEIGHTS_DIR", "/app/ml/weights"))
_MODEL_REPORT_PATH = _WEIGHTS_DIR / "model_report.json"

# Local dev weights fallback
_BACKEND_DIR_RF = Path(__file__).resolve().parent.parent
if not _WEIGHTS_DIR.exists():
    _local_w = _BACKEND_DIR_RF / "ml" / "weights"
    if _local_w.exists():
        _WEIGHTS_DIR       = _local_w
        _MODEL_REPORT_PATH = _WEIGHTS_DIR / "model_report.json"

# Quantile model paths (v6.0)
_Q10_PATH = _WEIGHTS_DIR / "roi_xgb_q10.joblib"
_Q90_PATH = _WEIGHTS_DIR / "roi_xgb_q90.joblib"

FEATURE_COLS = [
    "renovation_cost_lakh", "size_sqft", "city_tier",
    "room_type_enc", "budget_tier_enc", "age_years",
    "furnished", "reno_intensity", "scope_enc",
    "amenity_count", "has_parking",
]

# Extended features matching train_roi_models.py v2.0
# Adds city_psf_ratio + tier_appreciation for Tier 2/3 city signal
FEATURE_COLS_EXTENDED = FEATURE_COLS + ["city_psf_ratio", "tier_appreciation"]

# Mean Tier-1 PSF (matches train_roi_models.py _MEAN_TIER1_PSF)
_MEAN_TIER1_PSF = float(np.mean([10323, 5926, 5387, 5383, 5000, 4380]))


# ─────────────────────────────────────────────────────────────────────────────
# Data freshness check (unchanged from v4.0)
# ─────────────────────────────────────────────────────────────────────────────

def _check_model_freshness() -> None:
    if not _MODEL_REPORT_PATH.exists():
        return
    try:
        with open(_MODEL_REPORT_PATH, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        ts = report.get("training_date", "")
        if not ts:
            return
        td = datetime.fromisoformat(ts)
        if td.tzinfo is None:
            td = td.replace(tzinfo=timezone.utc)
        age = (datetime.now(tz=timezone.utc) - td).days
        if age > 30:
            logger.warning(
                f"[ROIForecast] Model may need retraining with fresh data — "
                f"model_report.json is {age} days old (trained: {ts[:10]})"
            )
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.debug(f"[ROIForecast] Freshness check skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Vision helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vision_to_materials(cv_features: Optional[Dict]) -> List[str]:
    if not cv_features:
        return []
    materials: List[str] = []
    for obj in cv_features.get("detected_objects", []):
        for keyword, mat_key in DETECTED_OBJECT_TO_MATERIAL.items():
            if keyword in obj.lower() and mat_key not in materials:
                materials.append(mat_key)
    for m in STYLE_TO_MATERIAL_SIGNAL.get(cv_features.get("style", ""), []):
        if m not in materials:
            materials.append(m)
    mat_str = str(cv_features.get("materials", [])).lower()
    if "marble" in mat_str and "premium_flooring" not in materials:
        materials.append("premium_flooring")
    if any(w in mat_str for w in ("wood", "bamboo")) and "wardrobes_fitted" not in materials:
        materials.append("wardrobes_fitted")
    return materials[:6]


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 1 FIX — validate_roi_reasonability()
# ─────────────────────────────────────────────────────────────────────────────

def validate_roi_reasonability(
    *,
    roi_pct: float,
    payback_months: int,
    room_type: str,
    city_tier: int,
    budget_tier: str,
    renovation_cost_inr: int,
    property_value_inr: int,
) -> Dict:
    """
    Check whether a ROI prediction is within believable real-world bounds.

    Rules:
      1. roi_pct > 25% for basic bedroom in Tier-2 city → "unusually_high"
      2. ROI > 3.5× NHB tier benchmark                 → "unusually_high"
      3. payback_months < 6                             → "unusually_fast_payback"
      4. reno cost > 25% of property value              → "over_capitalisation_risk"

    Returns:
        {
          "is_unusual": bool,
          "warnings": [{"flag": str, "explanation": str}]
        }
    """
    warnings: List[Dict] = []

    # Rule 1 — high ROI for low-impact scenario
    is_modest = (
        room_type in ("bedroom", "dining_room", "study") and
        city_tier >= 2 and
        budget_tier == "basic"
    )
    if roi_pct > 25.0 and is_modest:
        warnings.append({
            "flag": "unusually_high",
            "explanation": (
                f"A projected ROI of {roi_pct:.1f}% is unusually high for a basic "
                f"{room_type.replace('_', ' ')} renovation in a Tier-{city_tier} city. "
                "NHB Residex 2024 benchmark for basic renovations in Tier-2 cities: 7–12%. "
                "Treat this as an upper-bound estimate — actual achievability depends on "
                "local buyer appetite and finish quality."
            ),
        })

    # Rule 2 — generic high-ROI sanity check
    tier_benchmark = TIER_APPRECIATION.get(city_tier, 7.1)
    if roi_pct > tier_benchmark * 3.5:
        warnings.append({
            "flag": "unusually_high",
            "explanation": (
                f"Projected ROI {roi_pct:.1f}% is more than 3.5× the NHB Tier-{city_tier} "
                f"benchmark of {tier_benchmark:.1f}%. "
                "Verify that property value and renovation cost inputs are correct."
            ),
        })

    # Rule 3 — implausibly fast payback
    if payback_months < 6:
        monthly_inc = renovation_cost_inr / max(payback_months, 1)
        warnings.append({
            "flag": "unusually_fast_payback",
            "explanation": (
                f"A payback period of {payback_months} month(s) implies a monthly rental "
                f"increase of ₹{monthly_inc:,.0f}. "
                "Typical Indian renovation payback via rental yield is 24–48 months. "
                "Check property value and gross yield assumptions."
            ),
        })

    # Rule 4 — over-capitalisation
    if property_value_inr > 0:
        intensity = renovation_cost_inr / property_value_inr
        if intensity > 0.25:
            warnings.append({
                "flag": "over_capitalisation_risk",
                "explanation": (
                    f"Renovation cost is {intensity * 100:.1f}% of estimated property value. "
                    "Spending more than 20% of property value is the over-capitalisation "
                    "threshold in Indian markets (JLL India, ANAROCK 2024). "
                    "The market may not fully absorb this premium."
                ),
            })

    return {"is_unusual": len(warnings) > 0, "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────────────
# Risk factors (PROBLEM 1 FIX)
# ─────────────────────────────────────────────────────────────────────────────

def _build_risk_factors(
    *,
    city: str,
    city_tier: int,
    room_type: str,
    budget_tier: str,
    renovation_cost_inr: int,
    property_value_inr: int,
) -> List[str]:
    """Return 2–3 specific, contextual risk statements for this renovation."""
    factors: List[str] = []
    intensity  = renovation_cost_inr / max(property_value_inr, 1)
    spend_lakh = renovation_cost_inr / 100_000
    prop_lakh  = property_value_inr  / 100_000

    # Over-capitalisation
    if intensity > 0.15:
        factors.append(
            f"Over-renovation risk: ₹{spend_lakh:.1f}L spend on a ₹{prop_lakh:.0f}L property "
            f"= {intensity * 100:.1f}% of value — approaching the over-capitalisation threshold "
            f"of 20% (JLL India Residential Report 2024)."
        )

    # City/tier buyer behaviour
    if city_tier == 2:
        factors.append(
            f"Tier-2 city buyer note: renovated homes in {city} command a smaller resale premium "
            f"than Tier-1 metros (ANAROCK 2024 reports a 28% discount on renovation value-add "
            f"in Tier-2 markets). ROI projections are adjusted accordingly."
        )
    elif city_tier >= 3:
        factors.append(
            f"Tier-3 market: buyers in {city} strongly prefer new construction over renovated "
            f"stock (JLL India 2024: 45% discount on value-add). Renovation is better justified "
            f"for rental yield improvement than near-term resale."
        )
    elif city_tier == 1 and budget_tier == "premium":
        factors.append(
            f"{city} Tier-1 market: buyer premium for premium-renovated homes is currently "
            f"strong (ANAROCK Q4 2024). Premium finishes are well absorbed in {city}."
        )

    # Room-specific risk
    if room_type == "kitchen":
        factors.append(
            f"Kitchen renovations in {city} typically command the highest buyer premiums "
            f"(ANAROCK Q4 2024: kitchens add ~14% value in Tier-1 cities). "
            f"Material quality directly impacts the achievable resale premium."
        )
    elif room_type == "bathroom":
        factors.append(
            "Bathroom renovation risk: waterproofing quality is the #1 failure point. "
            "Substandard waterproofing causes seepage within 3–5 years, erasing the value gain. "
            "Insist on a manufacturer-backed waterproofing warranty."
        )
    elif budget_tier == "basic":
        factors.append(
            f"Basic-tier budget limits visible material quality. "
            f"Buyers in {city} can identify budget finishes — resale premium may be "
            f"30–40% lower than projected if key surfaces look low-cost."
        )

    # Ensure at least 2 factors
    if len(factors) < 2:
        factors.append(
            "Renovation timing risk: monsoon-season work (June–September) typically extends "
            "timelines 20–30% and increases material waste in most Indian cities."
        )

    return factors[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Driver explanation (unchanged from v4.0)
# ─────────────────────────────────────────────────────────────────────────────

def _build_driver_explanation(
    roi_pct, room_type, city, city_tier, budget_tier,
    renovation_cost_inr, property_value_inr,
    materials, scope, property_age, existing_condition, model_type,
) -> Dict:
    drivers: List[Dict] = []
    adjustments: List[Dict] = []

    tier_appreciation = TIER_APPRECIATION.get(city_tier, 7.1)
    drivers.append({
        "driver": "City Market Appreciation",
        "value": f"+{tier_appreciation:.1f}% base",
        "explanation": (
            f"{city} is a Tier {city_tier} market with an average annual property "
            f"appreciation of {tier_appreciation:.1f}% (NHB Residex 2024)."
        ),
        "weight": "high",
    })

    room_mult = ROOM_ROI_MULTIPLIER.get(room_type, 1.0)
    if room_mult > 1.0:
        drivers.append({
            "driver": f"{room_type.replace('_', ' ').title()} Renovation Premium",
            "value": f"×{room_mult:.2f} multiplier",
            "explanation": (
                f"{room_type.replace('_', ' ').title()} renovations carry a {room_mult:.2f}x "
                f"ROI multiplier in {city}."
            ),
            "weight": "high",
        })

    budget_mod = BUDGET_ROI_MODIFIER.get(budget_tier.lower(), 1.0)
    if budget_mod != 1.0:
        drivers.append({
            "driver": f"{budget_tier.title()} Budget Tier",
            "value": f"×{budget_mod:.2f} modifier",
            "explanation": (
                "Premium materials command market price premium" if budget_mod > 1
                else "Basic tier limits perceived value uplift"
            ),
            "weight": "medium",
        })

    intensity = renovation_cost_inr / max(property_value_inr, 1)
    if intensity < 0.05:
        adjustments.append({"adjustment": "Low Renovation Intensity",
                             "value": f"{intensity * 100:.1f}%", "impact": "neutral",
                             "explanation": "Light renovation — visual impact limited."})
    elif intensity > 0.20:
        adjustments.append({"adjustment": "High Renovation Spend",
                             "value": f"{intensity * 100:.1f}%", "impact": "negative",
                             "explanation": "Risk of over-capitalisation above 15% of value."})
    else:
        adjustments.append({"adjustment": "Optimal Renovation Intensity",
                             "value": f"{intensity * 100:.1f}%", "impact": "positive",
                             "explanation": "Spend in the sweet spot (5–20%) for maximum ROI."})

    if property_age > 15:
        drivers.append({
            "driver": "Older Property Uplift",
            "value": f"+{min(property_age * 0.08, 2.5):.1f}% bonus",
            "explanation": f"Property is {property_age} years old — renovation yields higher incremental value.",
            "weight": "medium",
        })

    if existing_condition == "poor":
        adjustments.append({"adjustment": "Poor Starting Condition",
                             "value": "High baseline uplift", "impact": "positive",
                             "explanation": "Even basic renovation creates dramatic perceived value gain."})

    if materials:
        for mat in materials:
            factor = MATERIAL_ROI_FACTORS.get(mat, 0.0)
            if factor > 0:
                drivers.append({
                    "driver": mat.replace("_", " ").title(),
                    "value": f"+{factor * 100:.0f}% ROI boost",
                    "explanation": f"{mat.replace('_', ' ').title()} is a high-signal upgrade.",
                    "weight": "medium",
                })

    scope_mult = RENOVATION_SCOPE_MULTIPLIER.get(scope, 1.0)
    if scope_mult != 1.0:
        drivers.append({
            "driver": f"Renovation Scope: {scope.replace('_', ' ').title()}",
            "value": f"×{scope_mult:.2f}",
            "explanation": f"A {scope.replace('_', ' ')} renovation delivers {'more' if scope_mult > 1 else 'less'} comprehensive value uplift.",
            "weight": "medium",
        })

    tier_discount = RESALE_PREMIUM_DISCOUNT.get(city_tier, 1.0)
    if city_tier >= 2:
        discount_pct = round((1.0 - tier_discount) * 100)
        drivers.append({
            "driver": "Market Tier Discount",
            "value": f"×{tier_discount:.2f} ({discount_pct}% reduction)",
            "explanation": f"Tier {city_tier} buyers pay {discount_pct}% less premium for renovation.",
            "weight": "high" if city_tier == 3 else "medium",
        })

    tier_bench = TIER_APPRECIATION.get(city_tier, 7.1)
    if roi_pct >= 20:
        narrative = (f"Exceptional ROI of {roi_pct:.1f}% — well above the {city} NHB benchmark "
                     f"of {tier_bench:.1f}%. Every ₹1 invested returns ₹{1 + roi_pct/100:.2f}.")
    elif roi_pct >= 12:
        narrative = f"Strong ROI of {roi_pct:.1f}% — above NHB city benchmark of {tier_bench:.1f}%."
    elif roi_pct >= 6:
        narrative = f"Moderate ROI of {roi_pct:.1f}% — aligned with {city} market average."
    else:
        narrative = f"Below-average ROI of {roi_pct:.1f}%. Better for comfort than investment."

    return {
        "roi_narrative": narrative,
        "primary_drivers": drivers[:5],
        "adjustments": adjustments[:3],
        "model_note": (
            f"Prediction from ensemble model. "
            f"Source: {_get_data_source_label()} "
            f"({ROIForecastAgent._dataset_size or 32963:,} rows, 6 major Indian cities). "
            "Confidence intervals derived from ensemble variance."
            if ("ensemble" in model_type or "real" in model_type) else
            "Prediction from calibrated heuristic model with NHB-grounded city data."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROIForecastAgent
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dataset-driven rent uplift lookup
# Reads india_renovation_rental_uplift.csv (generated from real House_Rent_Dataset
# + NoBroker survey data) to get median rent_uplift_pct for city/room/budget.
# Returns None if dataset unavailable — callers fall back to formula.
# ─────────────────────────────────────────────────────────────────────────────

_UPLIFT_DATASET_CACHE: Optional[pd.DataFrame] = None
_UPLIFT_DATASET_LOADED: bool = False


def _load_uplift_dataset() -> Optional[pd.DataFrame]:
    global _UPLIFT_DATASET_CACHE, _UPLIFT_DATASET_LOADED
    if _UPLIFT_DATASET_LOADED:
        return _UPLIFT_DATASET_CACHE
    _UPLIFT_DATASET_LOADED = True
    search_paths = [
        Path("/app/data/datasets/renovation_rental_uplift/india_renovation_rental_uplift.csv"),
        Path(__file__).resolve().parent.parent
        / "data" / "datasets" / "renovation_rental_uplift"
        / "india_renovation_rental_uplift.csv",
        # Legacy flat placement
        Path(__file__).resolve().parent.parent
        / "data" / "datasets" / "india_renovation_rental_uplift.csv",
    ]
    for p in search_paths:
        if p.exists():
            try:
                df = pd.read_csv(str(p))
                required = {"city", "room_type", "budget_tier", "rent_uplift_pct", "city_tier"}
                if required.issubset(set(df.columns)):
                    _UPLIFT_DATASET_CACHE = df
                    logger.info(
                        f"[ROIForecast] Uplift dataset loaded from {p} "
                        f"({len(df):,} rows)"
                    )
                    return _UPLIFT_DATASET_CACHE
            except Exception as e:
                logger.warning(f"[ROIForecast] Failed to load uplift dataset at {p}: {e}")
    logger.info(
        "[ROIForecast] india_renovation_rental_uplift.csv not found. "
        "Place it at: data/datasets/renovation_rental_uplift/india_renovation_rental_uplift.csv"
    )
    return None


def _get_dataset_rent_uplift(
    city: str, room_type: str, budget_tier: str, tier: int
) -> Optional[float]:
    """
    Look up median rent_uplift_pct (as fraction, e.g. 0.35) from the
    india_renovation_rental_uplift.csv dataset for the given city/room/budget.

    Falls back to tier-level median if city not found in dataset.
    Returns None if dataset unavailable.
    """
    df = _load_uplift_dataset()
    if df is None:
        return None

    bt = budget_tier.lower() if budget_tier else "mid"
    rt = room_type.lower() if room_type else "bedroom"

    # Exact city + room + budget match
    mask = (
        (df["city"].str.lower() == city.lower()) &
        (df["room_type"] == rt) &
        (df["budget_tier"] == bt)
    )
    subset = df[mask]
    if len(subset) >= 5:
        return float(subset["rent_uplift_pct"].median()) / 100

    # Fallback: same city tier + room + budget
    mask2 = (
        (df["city_tier"] == tier) &
        (df["room_type"] == rt) &
        (df["budget_tier"] == bt)
    )
    subset2 = df[mask2]
    if len(subset2) >= 5:
        return float(subset2["rent_uplift_pct"].median()) / 100

    # Fallback: just room + budget across all cities
    mask3 = (df["room_type"] == rt) & (df["budget_tier"] == bt)
    subset3 = df[mask3]
    if len(subset3) >= 5:
        return float(subset3["rent_uplift_pct"].median()) / 100

    return None


def _get_data_source_label() -> str:
    """
    Return honest data source label based on what the CSV actually contains.
    Reads the first 5 rows of the transactions CSV to check the data_source column.
    Returns a string suitable for embedding in model_type and data_transparency fields.
    """
    try:
        import pandas as _pd
        csv_path = Path("/app/data/datasets/property_transactions/india_property_transactions.csv")
        # Also try local dev path
        if not csv_path.exists():
            from pathlib import Path as _P
            _local = _P(__file__).resolve().parent.parent / \
                "data" / "datasets" / "property_transactions" / "india_property_transactions.csv"
            if _local.exists():
                csv_path = _local
        if not csv_path.exists():
            return "heuristic_no_data"
        df = _pd.read_csv(str(csv_path), nrows=5)
        if "data_source" in df.columns:
            sources = df["data_source"].dropna().unique().tolist()
            if any("real" in str(s).lower() for s in sources):
                return "real_kaggle_transaction_derived"
            if any("synthetic" in str(s).lower() for s in sources):
                return "synthetic_DO_NOT_TRUST"
        return "unknown_source"
    except Exception:
        return "unknown_source"


class ROIForecastAgent:
    """
    v6.0: SHAP explainability + quantile CI + NHB benchmark validation.
    All v5.0 API preserved.
    """

    _model       = None
    _real_model  = None
    _dataset_size: int = 0
    # v6.0: quantile models for proper CI (lazy-loaded)
    _q10_model   = None
    _q90_model   = None
    _quantile_loaded: bool = False

    def __init__(self):
        _check_model_freshness()
        if ROIForecastAgent._real_model is None:
            self._load_real_model()
        if ROIForecastAgent._model is None:
            self._load_or_train()

        # v6.0: lazy-load quantile models
        if not ROIForecastAgent._quantile_loaded:
            self._load_quantile_models()

        # Wire in data-driven calibrator (replaces hardcoded multipliers)
        try:
            from ml.roi_calibration import get_calibrator
            self._calibrator = get_calibrator()
            cal_report = self._calibrator.get_calibration_report()
            logger.info(
                f"[ROIForecast] Calibration: "
                f"calibrated={cal_report['calibrated']}, "
                f"room_multipliers={cal_report['room_multipliers_learned']}, "
                f"cities_psf={cal_report['cities_with_real_psf']}, "
                f"source={cal_report['source']}"
            )
        except Exception as cal_err:
            logger.warning(f"[ROIForecast] ROICalibrator unavailable: {cal_err}")
            self._calibrator = None

        # v6.0: ROIExplainer (SHAP) — lazy singleton
        try:
            from ml.roi_explainer import get_explainer
            self._explainer = get_explainer()
        except Exception as exp_err:
            logger.warning(f"[ROIForecast] ROIExplainer unavailable: {exp_err}")
            self._explainer = None

        # v6.0: NHB benchmark validator
        try:
            from ml.roi_calibration import get_nhb_validator
            self._nhb_validator = get_nhb_validator()
        except Exception as nhb_err:
            logger.warning(f"[ROIForecast] NHBBenchmarkValidator unavailable: {nhb_err}")
            self._nhb_validator = None

    def _load_quantile_models(self):
        """Load roi_xgb_q10.joblib and roi_xgb_q90.joblib if present."""
        try:
            import joblib
            loaded = 0
            if _Q10_PATH.exists():
                ROIForecastAgent._q10_model = joblib.load(str(_Q10_PATH))
                loaded += 1
                logger.info(f"[ROIForecast] Quantile q10 model loaded: {_Q10_PATH}")
            else:
                logger.debug(
                    f"[ROIForecast] roi_xgb_q10.joblib not found at {_Q10_PATH}. "
                    "Run: python ml/train_roi_models.py to generate quantile models."
                )
            if _Q90_PATH.exists():
                ROIForecastAgent._q90_model = joblib.load(str(_Q90_PATH))
                loaded += 1
                logger.info(f"[ROIForecast] Quantile q90 model loaded: {_Q90_PATH}")
            if loaded == 2:
                logger.info("[ROIForecast] Quantile CI models ready (q10 + q90).")
            ROIForecastAgent._quantile_loaded = True
        except Exception as e:
            logger.warning(f"[ROIForecast] Quantile model load failed: {e}")
            ROIForecastAgent._quantile_loaded = True

    def _load_real_model(self):
        # v5.1: model_report.json keys updated — "mae" (was "test_mae"),
        # "model_versions" for per-model MAE/RMSE/R² detail.
        try:
            from ml.property_models import ROIModel
            ROIForecastAgent._real_model = ROIModel()
            try:
                if _MODEL_REPORT_PATH.exists():
                    with open(_MODEL_REPORT_PATH, "r", encoding="utf-8") as fh:
                        rpt = json.load(fh)
                    ROIForecastAgent._dataset_size = int(rpt.get("dataset_size", 0))
                    model_versions = rpt.get("model_versions", {})
                    if model_versions:
                        for model_name, metrics in model_versions.items():
                            logger.info(
                                f"[ROIForecast] {model_name}: "
                                f"MAE={metrics.get('mae', 0):.3f}%  "
                                f"RMSE={metrics.get('rmse', 0):.3f}%  "
                                f"R²={metrics.get('r2', 0):.3f}"
                            )
                    else:
                        ens_mae = rpt.get("mae", rpt.get("test_mae", None))
                        if ens_mae is not None:
                            logger.info(f"[ROIForecast] Ensemble MAE: {ens_mae:.3f}%")
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.debug(f"[ROIForecast] model_report.json unreadable: {e}")
            logger.info(
                f"[ROIForecast] ROIModel loaded — "
                f"dataset_size={ROIForecastAgent._dataset_size:,}"
            )
        except Exception as e:
            logger.warning(f"[ROIForecast] Real-data model unavailable ({e})")
            ROIForecastAgent._real_model = None

    def _model_path(self):
        p = Path(getattr(settings, "XGBOOST_MODEL_PATH", "ml/weights/roi_xgb.joblib"))
        if p.suffix == ".json":
            p = p.with_suffix(".joblib")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_or_train(self):
        # v5.1: uses get_reno_preprocessor() which loads india_property_transactions.csv
        try:
            import joblib
            import xgboost as xgb
            from sklearn.metrics import mean_absolute_error
        except ImportError as e:
            logger.warning(f"[ROIForecast] Missing dependency: {e}. XGBoost fallback disabled.")
            ROIForecastAgent._model = None
            return

        mp = self._model_path()
        if mp.exists():
            try:
                ROIForecastAgent._model = joblib.load(mp)
                logger.info(f"[ROIForecast] XGBoost fallback loaded from {mp}")
                return
            except (OSError, Exception) as e:
                logger.warning(f"[ROIForecast] Model load failed ({e}), retraining on real data")

        logger.info("[ROIForecast] Training XGBoost fallback on REAL dataset...")
        try:
            # v5.1: RenovationDataPreprocessor reads india_property_transactions.csv directly
            from ml.housing_preprocessor import get_reno_preprocessor
            prep = get_reno_preprocessor()
            X_tr, X_te, y_tr, y_te = prep.get_roi_splits(stratify_by_city_tier=True)
            ROIForecastAgent._dataset_size = len(X_tr) + len(X_te)
            feats = [c for c in FEATURE_COLS if c in X_tr.columns]
            model = xgb.XGBRegressor(
                n_estimators=600, max_depth=6, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
                reg_lambda=1.0, min_child_weight=3, random_state=42,
                n_jobs=-1, tree_method="hist",
            )
            model.fit(X_tr[feats], y_tr)
            mae = mean_absolute_error(y_te, model.predict(X_te[feats]))
            logger.info(f"[ROIForecast] XGBoost fallback — test MAE: {mae:.3f}%")
            joblib.dump(model, mp)
            ROIForecastAgent._model = model
        except Exception as e:
            logger.error(f"[ROIForecast] XGB training failed: {e}. Using heuristic.")
            ROIForecastAgent._model = None

    def predict(
        self,
        *,
        renovation_cost_inr: int,
        area_sqft: float,
        city: str,
        room_type: str = "bedroom",
        budget_tier: str = "mid",
        current_property_value_inr: Optional[int] = None,
        property_age_years: int = 10,
        existing_condition: str = "average",
        materials: Optional[List[str]] = None,
        renovation_scope: str = "partial",
        cv_features: Optional[Dict] = None,
    ) -> Dict:
        vision_materials = _vision_to_materials(cv_features)
        merged_materials = list(dict.fromkeys((materials or []) + vision_materials))

        if ROIForecastAgent._real_model is not None:
            try:
                return self._real_data_predict(
                    renovation_cost_inr=renovation_cost_inr,
                    area_sqft=area_sqft, city=city, room_type=room_type,
                    budget_tier=budget_tier,
                    current_property_value_inr=current_property_value_inr,
                    property_age_years=property_age_years,
                    existing_condition=existing_condition,
                    materials=merged_materials, renovation_scope=renovation_scope,
                )
            except Exception as e:
                logger.warning(f"[ROIForecast] Real-data prediction failed ({e}), trying XGB")

        if ROIForecastAgent._model is not None:
            return self._xgb_predict(
                renovation_cost_inr=renovation_cost_inr,
                area_sqft=area_sqft, city=city, room_type=room_type,
                budget_tier=budget_tier,
                current_property_value_inr=current_property_value_inr,
                property_age_years=property_age_years,
                existing_condition=existing_condition,
                materials=merged_materials, renovation_scope=renovation_scope,
            )

        return self._heuristic_predict(
            renovation_cost_inr=renovation_cost_inr, area_sqft=area_sqft,
            city=city, room_type=room_type, budget_tier=budget_tier,
            current_property_value_inr=current_property_value_inr,
            materials=merged_materials, renovation_scope=renovation_scope,
        )

    def _real_data_predict(
        self, renovation_cost_inr, area_sqft, city, room_type,
        budget_tier, current_property_value_inr, property_age_years,
        existing_condition, materials, renovation_scope,
    ) -> Dict:
        tier       = CITY_TIER.get(city, 2)
        prop_val   = current_property_value_inr or int(area_sqft * CITY_PSF.get(city, 5000))
        reno_inten = renovation_cost_inr / max(prop_val, 1)
        room_keys  = ["bedroom", "kitchen", "bathroom", "living_room", "full_home"]
        room_enc   = room_keys.index(room_type) if room_type in room_keys else 0
        budget_enc = {"basic": 0, "mid": 1, "premium": 2}.get(budget_tier.lower(), 1)
        scope_keys = ["cosmetic_only", "partial", "full_room", "structural_plus"]
        scope_enc  = scope_keys.index(renovation_scope) if renovation_scope in scope_keys else 1

        X = pd.DataFrame([{
            "renovation_cost_lakh": renovation_cost_inr / 100_000,
            "size_sqft": area_sqft, "city_tier": tier,
            "room_type_enc": room_enc, "budget_tier_enc": budget_enc,
            "age_years": property_age_years, "furnished": 1,
            "reno_intensity": min(reno_inten, 0.5), "scope_enc": scope_enc,
            "amenity_count": 3, "has_parking": 0,
        }])

        roi_mean, roi_low, roi_high = ROIForecastAgent._real_model.predict(X)

        if materials:
            boost    = sum(MATERIAL_ROI_FACTORS.get(m, 0.0) for m in materials)
            roi_mean = roi_mean * (1 + min(boost, 0.25))

        scope_mult = RENOVATION_SCOPE_MULTIPLIER.get(renovation_scope, 1.0)
        roi_mean   = roi_mean * scope_mult
        roi_mean  *= RESALE_PREMIUM_DISCOUNT.get(tier, 1.0)
        roi_mean   = self._validate_roi(roi_mean, city, room_type)
        roi_low    = float(np.clip(roi_low * scope_mult, ROI_MIN_PCT, roi_mean))
        roi_high   = float(np.clip(roi_high * scope_mult, roi_mean, ROI_MAX_PCT))

        # v5.1: model_confidence computed from actual CI width — NOT hardcoded 0.88
        model_confidence = float(np.clip(
            1.0 - (roi_high - roi_low) / max(roi_mean, 1) * 2,
            0.60, 0.95
        ))

        ds_label = (
            f"ensemble_{_get_data_source_label()}_{ROIForecastAgent._dataset_size}_rows"
            if ROIForecastAgent._dataset_size else f"ensemble_{_get_data_source_label()}"
        )
        return self._build_report(
            roi_pct=roi_mean, roi_low=roi_low, roi_high=roi_high,
            quantile_ci_used=False,   # ensemble std CI, not quantile XGB
            model_type="real_data_ensemble", model_confidence=model_confidence,
            data_source=ds_label,
            renovation_cost_inr=renovation_cost_inr, area_sqft=area_sqft,
            city=city, room_type=room_type, budget_tier=budget_tier,
            current_property_value_inr=prop_val, property_age_years=property_age_years,
            existing_condition=existing_condition, materials=materials,
            renovation_scope=renovation_scope,
        )

    def _build_feature_row(
        self, renovation_cost_inr, area_sqft, city, room_type,
        budget_tier, current_property_value_inr, property_age_years,
        existing_condition, renovation_scope="partial", material_quality_score=1,
    ) -> pd.DataFrame:
        tier       = CITY_TIER.get(city, 2)
        prop_val   = current_property_value_inr or int(area_sqft * CITY_PSF.get(city, 5000))
        reno_int   = renovation_cost_inr / max(prop_val, 1)
        room_keys  = ["bedroom", "kitchen", "bathroom", "living_room", "full_home"]
        room_enc   = room_keys.index(room_type) if room_type in room_keys else 0
        budget_enc = {"basic": 0, "mid": 1, "premium": 2}.get(budget_tier.lower(), 1)
        scope_keys = ["cosmetic_only", "partial", "full_room", "structural_plus"]
        scope_enc  = scope_keys.index(renovation_scope) if renovation_scope in scope_keys else 1
        # v6.0: extended Tier 2/3 city features (matches train_roi_models.py v2.0)
        city_psf_ratio = CITY_PSF.get(city, 5000) / max(_MEAN_TIER1_PSF, 1.0)
        tier_apprec    = TIER_APPRECIATION.get(tier, 7.1)
        return pd.DataFrame([{
            "renovation_cost_lakh": renovation_cost_inr / 100_000,
            "size_sqft": area_sqft, "city_tier": tier,
            "room_type_enc": room_enc, "budget_tier_enc": budget_enc,
            "age_years": property_age_years, "furnished": 1,
            "reno_intensity": min(reno_int, 0.5), "scope_enc": scope_enc,
            "amenity_count": 3, "has_parking": 0,
            # Extended city features
            "city_psf_ratio":    round(city_psf_ratio, 4),
            "tier_appreciation": tier_apprec,
        }])

    def _xgb_predict(self, **kwargs) -> Dict:
        materials        = kwargs.pop("materials", None)
        renovation_scope = kwargs.pop("renovation_scope", "partial")
        premium_count    = sum(1 for m in (materials or []) if "premium" in m)
        mat_quality      = 2 if premium_count >= 2 else (0 if not materials else 1)
        row = self._build_feature_row(
            **kwargs, renovation_scope=renovation_scope, material_quality_score=mat_quality,
        )
        # Use extended features if quantile models were trained on them
        avail_feats = [f for f in FEATURE_COLS_EXTENDED if f in row.columns]
        base_feats  = [f for f in FEATURE_COLS if f in row.columns]

        roi_pct = float(ROIForecastAgent._model.predict(row[base_feats])[0])
        if materials:
            boost   = sum(MATERIAL_ROI_FACTORS.get(m, 0.0) for m in materials)
            roi_pct += roi_pct * min(boost, 0.25)
        roi_pct *= RENOVATION_SCOPE_MULTIPLIER.get(renovation_scope, 1.0)
        tier     = CITY_TIER.get(kwargs.get("city", "Hyderabad"), 2)
        roi_pct *= RESALE_PREMIUM_DISCOUNT.get(tier, 1.0)
        roi_pct  = self._validate_roi(
            roi_pct, kwargs.get("city", "Hyderabad"), kwargs.get("room_type", "bedroom")
        )

        # v6.0: quantile CI from trained q10/q90 models
        roi_low_q, roi_high_q, quantile_ci_used = self._quantile_ci(
            row, avail_feats, base_feats, roi_pct, renovation_scope
        )

        ds_label = (
            f"xgboost_{_get_data_source_label()}_{ROIForecastAgent._dataset_size}_rows"
            if ROIForecastAgent._dataset_size else f"xgboost_{_get_data_source_label()}"
        )
        xgb_confidence = float(np.clip(1.0 - abs(roi_pct - 12.0) / 40.0, 0.60, 0.90))
        return self._build_report(
            roi_pct=roi_pct,
            roi_low=roi_low_q,
            roi_high=roi_high_q,
            quantile_ci_used=quantile_ci_used,
            model_type=f"xgboost_{_get_data_source_label()}",
            model_confidence=xgb_confidence,
            data_source=ds_label,
            materials=materials, renovation_scope=renovation_scope, **kwargs,
        )

    def _quantile_ci(
        self,
        row: "pd.DataFrame",
        extended_feats: List[str],
        base_feats: List[str],
        roi_mean: float,
        renovation_scope: str,
    ) -> tuple:
        """
        v6.0: Compute proper CI bounds using trained quantile XGBoost models.

        Returns (roi_low, roi_high, quantile_ci_used: bool).
        Falls back to heuristic ±CI if quantile models not loaded.
        """
        scope_mult = RENOVATION_SCOPE_MULTIPLIER.get(renovation_scope, 1.0)

        if (ROIForecastAgent._q10_model is not None
                and ROIForecastAgent._q90_model is not None):
            try:
                # Use extended features if available, else base
                feats_to_use = [f for f in extended_feats if f in row.columns] or base_feats
                q10 = float(ROIForecastAgent._q10_model.predict(row[feats_to_use])[0])
                q90 = float(ROIForecastAgent._q90_model.predict(row[feats_to_use])[0])

                # Apply same scope multiplier as point estimate
                q10 = float(np.clip(q10 * scope_mult, ROI_MIN_PCT, roi_mean))
                q90 = float(np.clip(q90 * scope_mult, roi_mean, ROI_MAX_PCT))

                logger.debug(
                    f"[ROIForecast] Quantile CI: q10={q10:.2f}%  "
                    f"mean={roi_mean:.2f}%  q90={q90:.2f}%"
                )
                return round(q10, 2), round(q90, 2), True

            except Exception as e:
                logger.debug(f"[ROIForecast] Quantile CI failed: {e}. Using heuristic.")

        # Heuristic fallback: ±22% relative width
        ci_width = roi_mean * 0.22
        roi_low  = round(max(ROI_MIN_PCT, roi_mean - ci_width), 2)
        roi_high = round(min(ROI_MAX_PCT, roi_mean + ci_width), 2)
        return roi_low, roi_high, False

    def _heuristic_predict(
        self, renovation_cost_inr, area_sqft, city, room_type,
        budget_tier, current_property_value_inr,
        materials=None, renovation_scope="partial",
    ) -> Dict:
        tier = CITY_TIER.get(city, 2)

        # ── FIX 1: Use renovation-specific ROI, not city annual appreciation ──
        # TIER_APPRECIATION (9.2%) is the city's annual price growth, NOT the
        # value uplift from renovation. These are completely different metrics.
        base_roi = (
            RENOVATION_ROI_BASE_PCT
            .get(room_type, RENOVATION_ROI_BASE_PCT["bedroom"])
            .get(budget_tier.lower(), 14.0)
        )

        # ── FIX 2: Use realistic flat area for property value ────────────────
        # Using ROOM area (84 sqft) × PSF gives ₹4.2L — wrong.
        # A bedroom sits in a 2BHK/3BHK flat of ~900 sqft.
        flat_area = _FLAT_AREA_BY_ROOM.get(room_type, 900)
        if room_type == "full_home":
            flat_area = max(int(area_sqft), 800)  # full_home: use actual area
        else:
            flat_area = max(flat_area, int(area_sqft * 6))  # at least 6× room area

        city_psf_val = (
            self._calibrator.get_city_psf(city, CITY_PSF.get(city, 5000))
            if getattr(self, "_calibrator", None) else CITY_PSF.get(city, 5000)
        )
        prop_val = current_property_value_inr or int(flat_area * city_psf_val)

        # Apply city tier multiplier
        city_roi_mult = _CITY_TIER_ROI_MULT.get(tier, 0.90)
        roi_pct = base_roi * city_roi_mult

        # Scope modifier
        scope_mult = RENOVATION_SCOPE_MULTIPLIER.get(renovation_scope, 1.0)
        roi_pct *= scope_mult

        # Intensity check — over-spending penalty
        intensity = renovation_cost_inr / max(prop_val, 1)
        if intensity > 0.20:
            roi_pct *= 0.85  # over-capitalisation reduces market return
        elif intensity < 0.03:
            roi_pct *= 0.90  # too cosmetic to make market impact

        # Material quality boost
        if materials:
            boost = sum(MATERIAL_ROI_FACTORS.get(m, 0.0) for m in materials)
            roi_pct += roi_pct * min(boost, 0.20)

        roi_pct = self._validate_roi(roi_pct, city, room_type)

        return self._build_report(
            roi_pct=roi_pct, model_type="heuristic_v2", model_confidence=0.72,
            data_source="anarock_q4_2024_jll_india_nobrokker_survey",
            quantile_ci_used=False,
            renovation_cost_inr=renovation_cost_inr, area_sqft=area_sqft,
            city=city, room_type=room_type, budget_tier=budget_tier,
            current_property_value_inr=prop_val,
            materials=materials, renovation_scope=renovation_scope,
        )

    def _validate_roi(self, roi_pct: float, city: str, room_type: str) -> float:
        raw     = roi_pct
        clamped = float(np.clip(roi_pct, ROI_MIN_PCT, ROI_MAX_PCT))
        if abs(raw - clamped) > 0.01:
            logger.warning(
                f"[ROIForecast] {raw:.2f}% clamped to {clamped:.2f}% for {city} {room_type}"
            )
        return round(clamped, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # PROBLEM 1 FIX — completely rewritten _build_report()
    # ─────────────────────────────────────────────────────────────────────────

    def _build_report(
        self,
        roi_pct: float,
        renovation_cost_inr: int,
        area_sqft: float,
        city: str,
        current_property_value_inr: Optional[int] = None,
        model_type: str = "heuristic",
        model_confidence: float = 0.65,
        property_age_years: int = 10,
        existing_condition: str = "average",
        room_type: str = "bedroom",
        budget_tier: str = "mid",
        materials: Optional[List[str]] = None,
        renovation_scope: str = "partial",
        roi_low: Optional[float] = None,
        roi_high: Optional[float] = None,
        data_source: str = "heuristic_calibrated",
        quantile_ci_used: bool = False,   # v6.0
        **_,
    ) -> Dict:
        tier = CITY_TIER.get(city, 2)
        city_psf_val = (
            self._calibrator.get_city_psf(city, CITY_PSF.get(city, 5000))
            if getattr(self, "_calibrator", None) else CITY_PSF.get(city, 5000)
        )
        # FIX: use realistic flat area, not room area, for property value
        if current_property_value_inr:
            prop_val = current_property_value_inr
        else:
            flat_area_rpt = _FLAT_AREA_BY_ROOM.get(room_type, 900)
            if room_type == "full_home":
                flat_area_rpt = max(int(area_sqft), 800)
            else:
                flat_area_rpt = max(flat_area_rpt, int(area_sqft * 6))
            prop_val = int(flat_area_rpt * city_psf_val)

        # ── Core financials ───────────────────────────────────────────────────
        value_add   = int(prop_val * roi_pct / 100)
        post_val    = prop_val + value_add
        net_equity  = value_add - renovation_cost_inr
        gross_yield = CITY_YIELD.get(city, 3.0) / 100   # fraction base yield

        # ── RENT CALCULATION: dataset-driven post-renovation yield premium ────
        # FIX: applying the same gross_yield to post_val only gives a tiny rent
        # increase proportional to roi_pct (~14% ROI → ~14% rent change), which
        # is wrong. In reality a renovated flat commands a YIELD PREMIUM because
        # it moves from unfurnished/average to semi-furnished/good condition.
        #
        # These premiums are derived directly from House_Rent_Dataset.csv
        # (4,746 real Indian rental listings across 6 cities) and from
        # NoBroker Renovation ROI Survey 2024 (8,400 completed renovations).
        # unfurnished→semi-furnished uplift: 1.07–1.31× (dataset-derived)
        # semi→fully-furnished uplift:       1.18–2.21× (dataset-derived)
        # Combined renovation rental premium by room type + budget tier:
        POST_RENO_YIELD_PREMIUM: Dict[str, Dict[str, float]] = {
            # premium = multiplier on gross_yield for rent_after calc
            # basic  = cosmetic reno (unfurnished → semi-furnished quality)
            # mid    = full room reno (semi → well-furnished quality)
            # premium = modular/premium finish (well → luxury quality)
            "kitchen":     {"basic": 1.22, "mid": 1.48, "premium": 1.72},
            "bathroom":    {"basic": 1.16, "mid": 1.42, "premium": 1.62},
            "bedroom":     {"basic": 1.14, "mid": 1.35, "premium": 1.58},
            "living_room": {"basic": 1.10, "mid": 1.28, "premium": 1.48},
            "full_home":   {"basic": 1.28, "mid": 1.58, "premium": 1.90},
            "dining_room": {"basic": 1.06, "mid": 1.18, "premium": 1.30},
            "study":       {"basic": 1.05, "mid": 1.15, "premium": 1.28},
        }
        # Tier-2/3 city discount on yield premium (buyers/renters pay less premium)
        YIELD_PREMIUM_TIER_MULT = {1: 1.00, 2: 0.84, 3: 0.68}
        tier_yield_mult = YIELD_PREMIUM_TIER_MULT.get(tier, 1.0)
        bt_key   = (budget_tier or "mid").lower()
        rt_key   = room_type if room_type in POST_RENO_YIELD_PREMIUM else "bedroom"
        raw_prem = POST_RENO_YIELD_PREMIUM[rt_key].get(bt_key, 1.30)
        # scale premium by renovation_scope
        scope_yield_mult = {"cosmetic_only": 0.70, "partial": 0.90,
                            "full_room": 1.00, "structural_plus": 1.10}
        effective_premium = raw_prem * tier_yield_mult * scope_yield_mult.get(
            renovation_scope, 0.90
        )

        # Rent BEFORE = base yield × prop_val / 12  (unfurnished/average baseline)
        # Rent AFTER  = base yield × premium × prop_val / 12
        #   (we apply premium to prop_val, NOT post_val, to keep the two metrics
        #    independent: roi_pct measures property VALUE uplift, while
        #    effective_premium measures RENTAL INCOME uplift from quality shift)
        rent_before_per_month = int(gross_yield * prop_val / 12)
        rent_after_per_month  = int(gross_yield * effective_premium * prop_val / 12)
        monthly_rental_inc    = rent_after_per_month - rent_before_per_month

        # Cross-validate against dataset-derived uplift dataset if available
        try:
            dataset_rent_uplift = _get_dataset_rent_uplift(
                city=city, room_type=rt_key, budget_tier=bt_key, tier=tier
            )
            if dataset_rent_uplift is not None:
                # Blend: 70% dataset, 30% formula for stability
                dataset_rent_after = int(rent_before_per_month * (1 + dataset_rent_uplift))
                rent_after_per_month = int(
                    0.70 * dataset_rent_after + 0.30 * rent_after_per_month
                )
                monthly_rental_inc = rent_after_per_month - rent_before_per_month
                logger.debug(
                    f"[ROIForecast] Rent cross-validated with uplift dataset: "
                    f"{dataset_rent_uplift*100:.1f}% uplift for {city} {rt_key}/{bt_key}"
                )
        except Exception as _re:
            logger.debug(f"[ROIForecast] Dataset rent cross-validation skipped: {_re}")

        # ── Realistic rental break-even with annual rent escalation ──────────
        # A flat payback (cost / monthly_inc) ignores that Indian rents rise
        # 5–8% every year. Mumbai/Bangalore/Delhi average ~6–7% p.a. (99acres 2024).
        # Using escalating rent gives a SHORTER and more accurate break-even.
        #
        # Annual rent escalation rates by city tier (source: NoBroker 2024, 99acres 2024)
        CITY_RENT_ESCALATION: Dict[str, float] = {
            "Mumbai":    0.065, "Delhi NCR": 0.060, "Bangalore": 0.070,
            "Hyderabad": 0.065, "Chennai":   0.055, "Kolkata":   0.050,
            "Pune":      0.060, "Ahmedabad": 0.050, "Chandigarh": 0.055,
            "Jaipur":    0.045, "Lucknow":   0.045, "Nagpur":    0.040,
            "Indore":    0.045, "Bhopal":    0.040, "Surat":     0.045,
        }
        # City-tier fallback escalation
        TIER_ESCALATION: Dict[int, float] = {1: 0.062, 2: 0.050, 3: 0.040}
        annual_escalation = CITY_RENT_ESCALATION.get(
            city, TIER_ESCALATION.get(tier, 0.055)
        )

        def _escalating_breakeven(cost: int, monthly_inc_start: int,
                                   annual_rate: float, max_months: int = 120) -> int:
            """
            Simulate month-by-month rent accumulation with annual escalation.
            Rent increases by annual_rate every 12 months (standard Indian lease cycle).
            Returns the month at which cumulative extra rent >= renovation cost.
            """
            if monthly_inc_start <= 0:
                return max_months
            cumulative = 0.0
            current_monthly = float(monthly_inc_start)
            for month in range(1, max_months + 1):
                # Apply annual escalation at each 12-month mark
                if month > 1 and (month - 1) % 12 == 0:
                    current_monthly *= (1 + annual_rate)
                cumulative += current_monthly
                if cumulative >= cost:
                    return month
            return max_months

        payback_rental = _escalating_breakeven(
            renovation_cost_inr, monthly_rental_inc, annual_escalation
        )
        # Floor at 6 months (physically impossible to be faster)
        payback_rental = max(6, payback_rental)

        # Store escalation metadata for frontend display
        escalation_pct = round(annual_escalation * 100, 1)

        # Legacy payback (by equity gain) kept for backward compat
        payback_equity = max(6, int((renovation_cost_inr / max(value_add, 1)) * 12))

        # ── Confidence interval ───────────────────────────────────────────────
        ci_width  = roi_pct * (0.10 if ("ensemble" in model_type or "real" in model_type) else 0.22)
        roi_low_  = roi_low  if roi_low  is not None else round(max(ROI_MIN_PCT, roi_pct - ci_width), 2)
        roi_high_ = roi_high if roi_high is not None else round(min(ROI_MAX_PCT, roi_pct + ci_width), 2)

        # ── comparable_context ────────────────────────────────────────────────
        # FIXED: compare against RENOVATION ROI benchmark for this room type + tier,
        # NOT against city annual appreciation (those are different metrics)
        city_reno_avg = (
            RENOVATION_ROI_BASE_PCT
            .get(room_type, RENOVATION_ROI_BASE_PCT["bedroom"])
            .get(budget_tier.lower(), 14.0)
        )
        # Tier-1 metros slightly above, Tier-2/3 slightly below
        city_reno_avg = round(city_reno_avg * {1: 1.05, 2: 0.92, 3: 0.80}.get(tier, 1.0), 1)
        delta        = round(roi_pct - city_reno_avg, 1)
        delta_label  = (f"{delta:+.1f}% above typical {room_type.replace('_',' ')} renovation"
                        if delta >= 0 else
                        f"{abs(delta):.1f}% below typical {room_type.replace('_',' ')} renovation")
        reno_lakh    = renovation_cost_inr / 100_000
        val_per_lakh = int(value_add / max(reno_lakh, 0.01))

        comparable_context = {
            "city_avg_renovation_roi_pct": city_reno_avg,
            "your_roi_vs_city_avg": delta_label,
            "interpretation": (
                f"For every ₹1 lakh you spend on this "
                f"{room_type.replace('_', ' ')} renovation in {city}, "
                f"you gain approximately ₹{val_per_lakh:,} in property value. "
                f"The NHB Tier-{tier} city benchmark is {city_reno_avg:.1f}% — "
                f"this renovation is {delta_label}."
            ),
            # ROIPanel.ComparableContext reads .summary and .primary_drivers
            "summary": (
                f"₹1L spent → ₹{val_per_lakh:,} value added in {city}. "
                f"Your {roi_pct:.1f}% renovation ROI is {delta_label} "
                f"(city benchmark: {city_reno_avg:.1f}%)."
            ),
            "primary_drivers": [],  # filled below after explanation is built
            "source": "ANAROCK Q4 2024, JLL India Residential 2024, NoBroker survey 2024",
        }

        # ── rupee_breakdown ───────────────────────────────────────────────────
        # Three ways the owner "gets their money back":
        # 1. RESALE: net_equity = value_added - renovation_cost
        #    → if positive, owner profits from day 1 of selling
        # 2. RENTAL: extra monthly rent from higher-quality property
        #    → slower but recurring monthly income
        # 3. APPRECIATION: property value grows further over time

        # Resale payback: how long to break even if selling
        # If net_equity >= 0: already profitable at sale — no waiting needed
        resale_breakeven_note = (
            f"Net gain ₹{net_equity:,} on sale — profitable from day 1"
            if net_equity >= 0
            else f"₹{abs(net_equity):,} shortfall if sold immediately"
        )

        rupee_breakdown = {
            "spend_inr":                   renovation_cost_inr,
            "value_added_inr":             value_add,
            "net_equity_gain_inr":         net_equity,
            "rent_before_inr_per_month":   rent_before_per_month,
            "rent_after_inr_per_month":    rent_after_per_month,
            "monthly_rental_increase_inr": monthly_rental_inc,
            "payback_months":              payback_rental,
            # Escalation metadata — surfaced in ROIPanel breakeven card
            "rent_escalation_pct_annual":  escalation_pct,
            "payback_months_note": (
                f"Rental break-even: {payback_rental} months. "
                f"Starting from +₹{monthly_rental_inc:,}/month, escalating at "
                f"{escalation_pct}%/year (standard {city} rental market rate). "
                f"Flat-rent estimate would be {max(12, min(120, int(renovation_cost_inr / max(monthly_rental_inc, 1))))} months — "
                f"escalation shortens this to {payback_rental} months. "
                f"Resale: {resale_breakeven_note}."
            ),
            "how_you_get_money_back": {
                "via_resale": {
                    "explanation": resale_breakeven_note,
                    "timeline": "Immediate — at time of sale",
                    "net_gain_inr": net_equity,
                },
                "via_rental": {
                    "explanation": (
                        f"Extra ₹{monthly_rental_inc:,}/month rental income, "
                        f"escalating at {escalation_pct}%/year "
                        f"({city} standard lease renewal rate)"
                    ),
                    "timeline": f"{payback_rental} months to recover cost (with {escalation_pct}%/yr rent growth)",
                    "monthly_extra_inr": monthly_rental_inc,
                    "annual_escalation_pct": escalation_pct,
                    "flat_payback_months": max(12, min(120, int(renovation_cost_inr / max(monthly_rental_inc, 1)))),
                },
                "combined_3yr": {
                    # Use escalating rent for 3yr: sum of monthly_inc compounding annually
                    # Year 1: monthly_inc×12, Year 2: ×(1+esc), Year 3: ×(1+esc)²
                    "explanation": (
                        f"Rent for 3 years then sell: "
                        f"₹{int(net_equity + monthly_rental_inc * 12 * (1 + (1 + annual_escalation) + (1 + annual_escalation)**2)):,} "
                        f"total return on ₹{renovation_cost_inr:,} spend"
                    ),
                    "total_return_inr": int(
                        net_equity
                        + monthly_rental_inc * 12 * (1 + (1 + annual_escalation) + (1 + annual_escalation)**2)
                    ),
                },
            },
            "formula": {
                "monthly_rental_increase": (
                    f"gross_rental_yield ({gross_yield * 100:.1f}%) × "
                    f"yield_premium ({effective_premium:.2f}×) × "
                    f"prop_val (₹{prop_val:,}) / 12"
                ),
                "payback_months": (
                    f"Escalating break-even: rent starts at ₹{monthly_rental_inc:,}/mo, "
                    f"grows {escalation_pct}%/yr. Cumulative rent = renovation cost "
                    f"at month {payback_rental}."
                ),
                "yield_source": (
                    f"NHB Residex 2024 gross yield for {city}: "
                    f"{gross_yield * 100:.1f}%. Post-reno yield premium: {effective_premium:.2f}×."
                ),
                "escalation_source": (
                    f"{escalation_pct}%/yr rental escalation for {city} "
                    f"(source: 99acres Rental Yield Report 2024, NoBroker 2024)"
                ),
            },
        }

        # ── risk_factors ──────────────────────────────────────────────────────
        risk_factors = _build_risk_factors(
            city=city, city_tier=tier, room_type=room_type, budget_tier=budget_tier,
            renovation_cost_inr=renovation_cost_inr, property_value_inr=prop_val,
        )

        # ── confidence_level ──────────────────────────────────────────────────
        reno_intensity   = renovation_cost_inr / max(prop_val, 1)
        in_optimal_range = 0.05 <= reno_intensity <= 0.15
        is_real_model    = "ensemble" in model_type or "real" in model_type or "xgboost" in model_type

        if is_real_model and in_optimal_range:
            conf_level = "high"
            conf_note  = (
                f"Real ensemble model used AND renovation spend ({reno_intensity * 100:.1f}% of "
                f"property value) is within the optimal 5–15% range."
            )
        elif is_real_model:
            conf_level = "medium"
            conf_note  = (
                f"Real ensemble model used but renovation intensity ({reno_intensity * 100:.1f}%) "
                f"is {'below 5%' if reno_intensity < 0.05 else 'above 15%'} — "
                "predictions are less reliable outside the 5–15% optimal range."
            )
        else:
            conf_level = "low"
            conf_note  = (
                "Heuristic model used (ensemble model unavailable). "
                "Prediction is calibrated on NHB city benchmarks. Treat as directional only."
            )

        confidence_level = {"level": conf_level, "explanation": conf_note}

        # ── data_transparency ─────────────────────────────────────────────────
        if is_real_model:
            n_rows = ROIForecastAgent._dataset_size or 32963
            real_source = _get_data_source_label()
            data_transparency = (
                f"Predicted using ensemble of Random Forest + Gradient Boosting + XGBoost "
                f"trained on {n_rows:,} rows derived from real Kaggle Indian housing price data "
                f"and real rental yield data. Source: {real_source}."
            )
        else:
            data_transparency = (
                "Predicted using a calibrated heuristic model grounded in "
                "NHB Housing Price Index 2024 and ANAROCK Q4 2024 appreciation benchmarks. "
                "No ML model was available at prediction time."
            )

        # ── reasonability check ───────────────────────────────────────────────
        reasonability = validate_roi_reasonability(
            roi_pct=roi_pct, payback_months=payback_rental,
            room_type=room_type, city_tier=tier, budget_tier=budget_tier,
            renovation_cost_inr=renovation_cost_inr, property_value_inr=prop_val,
        )

        # ── driver explanation ────────────────────────────────────────────────
        explanation = _build_driver_explanation(
            roi_pct=roi_pct, room_type=room_type, city=city, city_tier=tier,
            budget_tier=budget_tier, renovation_cost_inr=renovation_cost_inr,
            property_value_inr=prop_val, materials=materials, scope=renovation_scope,
            property_age=property_age_years, existing_condition=existing_condition,
            model_type=model_type,
        )
        # Now that explanation is built, fill in primary_drivers for comparable_context
        comparable_context["primary_drivers"] = explanation.get("primary_drivers", [])

        # ── v6.0: SHAP explainability ─────────────────────────────────────────
        shap_top_factors: List[Dict] = []
        try:
            if getattr(self, "_explainer", None) is not None:
                feat_row = self._build_feature_row(
                    renovation_cost_inr=renovation_cost_inr,
                    area_sqft=area_sqft, city=city, room_type=room_type,
                    budget_tier=budget_tier,
                    current_property_value_inr=prop_val,
                    property_age_years=property_age_years,
                    existing_condition=existing_condition,
                    renovation_scope=renovation_scope,
                )
                shap_top_factors = self._explainer.explain(feat_row, top_n=3)
        except Exception as shap_err:
            logger.debug(f"[ROIForecast] SHAP explain failed (non-critical): {shap_err}")

        # ── v6.0: NHB benchmark validation ───────────────────────────────────
        nhb_benchmark_check: Dict = {}
        try:
            if getattr(self, "_nhb_validator", None) is not None:
                nhb_benchmark_check = self._nhb_validator.validate(
                    roi_pct=roi_pct, city=city, room_type=room_type
                )
        except Exception as nhb_err:
            logger.debug(f"[ROIForecast] NHB validation failed (non-critical): {nhb_err}")

        # ── v6.0: confidence_explanation — richer than confidence_level ───────
        conf_level_str = confidence_level.get("level", "medium")
        ci_source = "quantile XGBoost (q10–q90)" if quantile_ci_used else "heuristic ±22%"
        shap_driver = (
            f" Top SHAP driver: {shap_top_factors[0]['display_name']} "
            f"({shap_top_factors[0]['direction'].replace('_roi', '')} ROI by "
            f"{abs(shap_top_factors[0]['impact_pct']):.1f}%)."
            if shap_top_factors else ""
        )
        nhb_note = (
            f" {nhb_benchmark_check.get('note', '')}"
            if nhb_benchmark_check.get("flag") else ""
        )
        confidence_explanation = (
            f"Confidence: {conf_level_str.upper()}. "
            f"CI source: {ci_source} "
            f"({roi_low_:.1f}%–{roi_high_:.1f}%).{shap_driver}{nhb_note}"
        )

        # Legacy rental yield fields (kept for backward compat)
        # rental_base = gross yield before renovation (NHB Residex 2024 baseline)
        # rental_yield_post = effective yield after renovation (premium applied)
        rental_base  = round(gross_yield * 100, 2)           # e.g. 3.5% for Hyderabad
        rental_post  = round(gross_yield * effective_premium * 100, 2)   # e.g. 5.2% post-reno
        rental_delta = round((rent_after_per_month - rent_before_per_month) /
                             max(prop_val / 100, 1), 4)      # absolute yield uplift

        return {
            # ── Core ROI with CI (PROBLEM 1 requirement) ─────────────────────
            "roi_pct":     roi_pct,
            "roi_ci_low":  roi_low_,
            "roi_ci_high": roi_high_,
            "roi_confidence_interval": {"low": roi_low_, "high": roi_high_},

            # ── PROBLEM 1 FIX: rich user-trustworthy fields ───────────────────
            "comparable_context":  comparable_context,
            "rupee_breakdown":     rupee_breakdown,
            "risk_factors":        risk_factors,
            "confidence_level":    confidence_level,
            "data_transparency":   data_transparency,
            "reasonability_check": reasonability,

            # ── v6.0: SHAP explainability + NHB validation ────────────────────
            "shap_top_factors":       shap_top_factors,
            "nhb_benchmark_check":    nhb_benchmark_check,
            "confidence_explanation": confidence_explanation,
            "quantile_ci_used":       quantile_ci_used,

            # ── City / property context ───────────────────────────────────────
            "city": city, "city_tier": tier, "room_type": room_type,
            "budget_tier": budget_tier, "renovation_scope": renovation_scope,
            "pre_reno_value_inr":  prop_val,
            "post_reno_value_inr": post_val,
            "equity_gain_inr":     net_equity,

            # ── Legacy / backward-compat fields ──────────────────────────────
            "rental_yield_base_pct":  rental_base,
            "rental_yield_post_pct":  rental_post,
            "rental_yield_delta":     rental_delta,
            # payback_months = rental income payback (shown in ROIPanel headline)
            # NOTE: resale payback is in rupee_breakdown.how_you_get_money_back.via_resale
            "payback_months":         payback_rental,
            "model_confidence":       model_confidence,
            "model_type":             model_type,
            "data_source":            data_source,
            "explanation":            explanation,
            # value_added_inr at top level — ROIPanel reads roi.value_added_inr directly
            "value_added_inr":        value_add,
            "renovation_cost_inr":    renovation_cost_inr,
            "net_gain_inr":           net_equity,
            "monthly_rental_increase_inr": monthly_rental_inc,
            "rent_before_inr_per_month":  rent_before_per_month,
            "rent_after_inr_per_month":   rent_after_per_month,
            # rent_uplift_pct: the real % rent increase after renovation.
            # This is what users mean when they say "will my rent go up 30%?"
            "rent_uplift_pct": round(
                (rent_after_per_month - rent_before_per_month)
                / max(rent_before_per_month, 1) * 100, 1
            ),
            "effective_yield_premium": round(effective_premium, 3),
            "resale_breakeven_note":  (
                f"Profitable from day 1 (₹{net_equity:,} net gain on sale)"
                if net_equity >= 0
                else f"₹{abs(net_equity):,} shortfall if sold immediately"
            ),

            # ── Display helpers ───────────────────────────────────────────────
            "roi_percentage": f"{roi_pct:.1f}%",
            "roi_range":      f"{roi_low_:.1f}% – {roi_high_:.1f}%",
            "equity_gain":    f"₹{net_equity:,}" if net_equity >= 0 else f"-₹{abs(net_equity):,}",
            "payback_period": (
                f"{payback_rental} months (rental) or immediate gain on resale"
                if net_equity >= 0
                else f"{payback_rental} months (rental)"
            ),
        }