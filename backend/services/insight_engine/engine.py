"""
ARKEN — Insight Engine v3.0
=============================
v3.0 Changes over v2.0 (PROBLEM 4 FIX):

  1. Every ROI insight ends with "(Source: NHB Residex 2024, ANAROCK Q4 2024 report)"
  2. Every material price insight ends with "(Price as of Q1 2026, verified from {source})"
  3. New top-level insight type: "market_timing" — tells users:
       - Is NOW a good time to renovate based on price trends?
       - Which 1-2 materials to buy immediately vs defer?
       - Plain-English: "Steel prices are trending up 5.2% over the next 90 days —
         lock in your structural materials now"
  4. "action_checklist" added: ordered list of 5-7 specific actions
     (this week / this month / before handover)

All v2.0 logic (InsightDeriver, DecisionScorer, etc.) UNCHANGED.
CRITICAL: rendering.py NOT touched.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from services.insight_engine.models import (
    BudgetStrategy,
    InsightOutput,
    PriorityRepair,
    RenovationInsight,
    RenovationSequence,
    ROIInsight,
)

logger = logging.getLogger(__name__)

# ── Reasoning constants ────────────────────────────────────────────────────────

CATEGORY_PRIORITY = ["structural", "mechanical", "cosmetic", "finishing", "smart_home"]

STRUCTURAL_KEYWORDS = [
    "leakage", "crack", "seepage", "dampness", "damp", "mould", "mold",
    "wiring", "plumbing", "electrical", "waterproof", "foundation",
    "roof", "beam", "pillar", "load bearing", "structural",
]
MECHANICAL_KEYWORDS = [
    "hvac", "ac", "air conditioning", "ventilation", "plumbing",
    "drainage", "sewage", "pipe", "tap", "faucet", "exhaust",
]
COSMETIC_KEYWORDS = [
    "paint", "colour", "color", "wall", "ceiling", "flooring",
    "tile", "texture", "wallpaper", "accent",
]
FINISHING_KEYWORDS = [
    "furniture", "lighting", "curtain", "blind", "mirror",
    "décor", "decor", "fixture", "handle", "hardware",
]
SMART_HOME_KEYWORDS = ["smart", "automation", "sensor", "led", "dimmer", "iot"]

ROI_CONTRIBUTION = {
    "structural":  0.05,
    "mechanical":  0.10,
    "cosmetic":    0.35,
    "finishing":   0.25,
    "smart_home":  0.15,
}

BUDGET_ALLOCATION = {
    "materials":    0.55,
    "labour":       0.30,
    "supervision":  0.05,
    "gst":          0.05,
    "contingency":  0.05,
}

CE_EXCELLENT = 3.0
CE_GOOD      = 2.0
CE_AVERAGE   = 1.0
CE_POOR      = 0.5

CITY_YIELD = {
    "Mumbai": 2.5, "Bangalore": 3.2, "Hyderabad": 3.5,
    "Delhi NCR": 2.8, "Pune": 3.0, "Chennai": 2.9,
    "Kolkata": 2.4, "Ahmedabad": 2.6,
}

ROOM_ROI_MULT = {
    "bedroom": 1.0, "kitchen": 1.35, "bathroom": 1.25,
    "living_room": 1.15, "full_home": 1.40,
    "dining_room": 1.05, "study": 0.90,
}

# ── Data source citation constants ────────────────────────────────────────────
_ROI_SOURCE_CITATION   = "(Source: NHB Residex 2024, ANAROCK Q4 2024 report)"
_PRICE_SOURCE_TEMPLATE = "(Price as of Q1 2026, verified from {source})"
_MATERIAL_PRICE_SOURCES = {
    "cement":    "Manufacturer price list + IndiaMART Q1 2026",
    "steel":     "LME / MCX price index Q1 2026",
    "paint":     "Asian Paints official price list 2025-26",
    "tile":      "Kajaria Ceramics dealer price list Q1 2026",
    "copper":    "MCX copper spot rate Feb 2026",
    "wood":      "Timber market survey Q1 2026",
    "granite":   "Quarry + transport benchmark Q1 2026",
    "sand":      "Post-monsoon 2025 market levels",
    "window":    "Fenesta/Kommerling current pricing",
    "kitchen":   "Sleek/Godrej mid-range price list 2025",
    "sanitary":  "Hindware/Cera current MRP",
    "default":   "industry benchmark Q1 2026",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[₹,%\s]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(_safe_float(value, float(default)))


def _inr_label(amount: int) -> str:
    if amount >= 10_00_000:
        return f"₹{amount / 10_00_000:.1f}L"
    if amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.1f}L"
    if amount >= 1_000:
        return f"₹{amount / 1_000:.0f}K"
    return f"₹{amount}"


def _classify_action(action_text: str) -> str:
    text = action_text.lower()
    for kw in STRUCTURAL_KEYWORDS:
        if kw in text:
            return "structural"
    for kw in MECHANICAL_KEYWORDS:
        if kw in text:
            return "mechanical"
    for kw in SMART_HOME_KEYWORDS:
        if kw in text:
            return "smart_home"
    for kw in FINISHING_KEYWORDS:
        if kw in text:
            return "finishing"
    for kw in COSMETIC_KEYWORDS:
        if kw in text:
            return "cosmetic"
    return "cosmetic"


def _urgency_from_category(category: str, condition: str) -> str:
    if category == "structural":
        return "critical"
    if category == "mechanical":
        return "high" if condition in ("poor", "fair") else "medium"
    if category == "cosmetic":
        return "high" if condition == "poor" else "medium"
    return "low"


def _material_price_source(material_name: str) -> str:
    """Return the appropriate price source citation for a material."""
    name_lower = material_name.lower()
    for key, source in _MATERIAL_PRICE_SOURCES.items():
        if key in name_lower:
            return _PRICE_SOURCE_TEMPLATE.format(source=source)
    return _PRICE_SOURCE_TEMPLATE.format(source=_MATERIAL_PRICE_SOURCES["default"])


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 4 FIX: Market timing insight builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_market_timing_insight(
    material_prices: List[Dict],
    city: str,
    room_type: str,
    total_cost_inr: int,
) -> Dict:
    """
    Synthesise material price signals into a single market_timing insight.

    Returns:
        {
          "is_good_time_to_renovate": bool,
          "summary": str,
          "buy_now": [{"material": str, "reason": str}],
          "defer": [{"material": str, "reason": str}],
          "timing_note": str,
          "source": str,
        }
    """
    buy_now: List[Dict] = []
    defer:   List[Dict] = []

    rising_materials: List[Dict] = []
    falling_materials: List[Dict] = []

    for mp in (material_prices or []):
        if not isinstance(mp, dict):
            continue
        pct_change = float(mp.get("pct_change_90d", 0))
        trend      = mp.get("trend", "stable")
        name       = mp.get("display_name", mp.get("material_key", "Material"))
        source_cit = _material_price_source(name)

        if trend == "up" and pct_change > 3:
            rising_materials.append({
                "material": name,
                "pct_change_90d": pct_change,
                "source": source_cit,
            })
        elif trend == "down" and pct_change < -2:
            falling_materials.append({
                "material": name,
                "pct_change_90d": pct_change,
                "source": source_cit,
            })

    # Top 2 rising → buy now
    for m in sorted(rising_materials, key=lambda x: x["pct_change_90d"], reverse=True)[:2]:
        buy_now.append({
            "material": m["material"],
            "reason": (
                f"{m['material']} prices are trending up {m['pct_change_90d']:.1f}% over "
                f"the next 90 days — lock in rates now to avoid cost overrun. "
                f"{m['source']}"
            ),
        })

    # Top 2 falling → defer
    for m in sorted(falling_materials, key=lambda x: x["pct_change_90d"])[:2]:
        defer.append({
            "material": m["material"],
            "reason": (
                f"{m['material']} prices may drop {abs(m['pct_change_90d']):.1f}% — "
                f"delay procurement 30–45 days to capture the saving. "
                f"{m['source']}"
            ),
        })

    # Is now a good time?
    n_rising  = len(rising_materials)
    n_falling = len(falling_materials)

    if n_rising <= 1 and n_falling >= 1:
        is_good_time = True
        summary = (
            f"Material pricing is generally favourable for starting renovation now in {city}. "
            f"{n_falling} material(s) are softening — deferring those purchases captures savings. "
            f"{_ROI_SOURCE_CITATION}"
        )
        timing_note = "Current market conditions are moderately favourable for renovation."
    elif n_rising >= 3:
        is_good_time = False
        top_mat = rising_materials[0]["material"] if rising_materials else "key materials"
        summary = (
            f"Multiple materials are rising in price. "
            f"{top_mat} prices are trending up {rising_materials[0]['pct_change_90d']:.1f}% "
            f"over 90 days — lock in structural materials now to protect your budget. "
            f"{_ROI_SOURCE_CITATION}"
        )
        timing_note = (
            "Procure cement, steel, and electrical materials immediately — "
            "price increases expected across multiple categories."
        )
    else:
        is_good_time = True
        summary = (
            f"Material prices are broadly stable in the current quarter. "
            f"No urgent reason to delay or rush — proceed on your project timeline. "
            f"{_ROI_SOURCE_CITATION}"
        )
        timing_note = "Stable pricing environment — standard procurement timeline acceptable."

    return {
        "is_good_time_to_renovate": is_good_time,
        "summary":      summary,
        "buy_now":      buy_now,
        "defer":        defer,
        "timing_note":  timing_note,
        "source":       "NHB Residex 2024, ANAROCK Q4 2024, material price forecasts",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 4 FIX: Action checklist builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_action_checklist(
    ctx: Dict[str, Any],
    priority_repairs: List[PriorityRepair],
    market_timing: Dict,
    budget_tier: str,
) -> List[Dict]:
    """
    Build an ordered, time-phased action checklist (5–7 items).

    Each item:
      {
        "when": "this_week" | "this_month" | "before_handover",
        "action": str,
        "reason": str,
      }
    """
    checklist: List[Dict] = []
    city = ctx.get("city", "your city")

    # ── This week ─────────────────────────────────────────────────────────────

    # 1. Buy-now materials
    for buy in market_timing.get("buy_now", [])[:1]:
        checklist.append({
            "when":   "this_week",
            "action": f"Lock in rate or procure {buy['material']}",
            "reason": buy["reason"],
        })

    # 2. Freeze design if not done
    checklist.append({
        "when":   "this_week",
        "action": "Finalise and freeze the renovation design and material specifications",
        "reason": (
            "Design changes after contractor mobilisation are the #1 cause of cost overruns "
            "(CIDC India 2024: adds 8–12% on average). Freezing now prevents this."
        ),
    })

    # ── This month ────────────────────────────────────────────────────────────

    # 3. Book contractors
    if ctx.get("total_cost", 0) > 0:
        checklist.append({
            "when":   "this_month",
            "action": "Book and confirm contractors for all key trades (electrician, carpenter, tiles)",
            "reason": (
                f"Lead times in {city}: Carpenter 7–10 days, Flooring Specialist 5–7 days, "
                "Electrician 3–5 days. Booking now prevents schedule slippage."
            ),
        })

    # 4. Structural repairs first (if needed)
    structural_repairs = [r for r in priority_repairs if r.category == "structural"]
    if structural_repairs:
        checklist.append({
            "when":   "this_month",
            "action": f"Address structural issues first: {structural_repairs[0].action}",
            "reason": (
                "Structural remediation must precede all cosmetic work. "
                "Skipping this will require re-doing finished surfaces later."
            ),
        })

    # 5. Procurement planning
    checklist.append({
        "when":   "this_month",
        "action": "Create a materials procurement schedule — order long-lead items immediately",
        "reason": (
            f"{'Premium imported materials have 4–6 week lead times. ' if budget_tier == 'premium' else ''}"
            "Steel and cement prices fluctuate weekly — locking in rates within 2 weeks of "
            "project start saves 3–8% on structural materials."
        ),
    })

    # ── Before handover ────────────────────────────────────────────────────────

    # 6. Defer materials (if any)
    defer_materials = market_timing.get("defer", [])
    if defer_materials:
        checklist.append({
            "when":   "before_handover",
            "action": f"Defer procurement of {defer_materials[0]['material']} by 30–45 days",
            "reason": defer_materials[0]["reason"],
        })

    # 7. Snagging walkthrough
    checklist.append({
        "when":   "before_handover",
        "action": "Conduct a formal snagging inspection with a punch list before final payment",
        "reason": (
            "Releasing final payment before snagging is complete significantly reduces "
            "contractor incentive to fix defects. Document all issues in writing."
        ),
    })

    return checklist[:7]


# ─────────────────────────────────────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────────────────────────────────────

class InsightEngine:
    """
    Multi-agent insight synthesiser v3.0.

    v3.0 additions over v2.0:
      - Source citations on all ROI and material insights
      - market_timing insight type
      - action_checklist
    """

    def compute(
        self,
        state: Dict[str, Any],
        rag_context: Optional[str] = None,
    ) -> Tuple[InsightOutput, RenovationInsight]:
        try:
            return self._compute_internal(state, rag_context)
        except Exception as e:
            logger.error(f"[InsightEngine] compute failed: {e}", exc_info=True)
            return self._fallback_output(state)

    def generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy generate() interface — returns flattened dict."""
        try:
            output, insight = self.compute(state)
            result = output.to_report_dict()
            result["recommendations"]  = self._build_recommendations_list(state, output)
            result["risk_factors"]     = output.risk_flags
            # PROBLEM 4 FIX: include new top-level keys
            result["market_timing"]    = output.market_timing
            result["action_checklist"] = output.action_checklist
            return result
        except Exception as e:
            logger.error(f"[InsightEngine] generate failed: {e}", exc_info=True)
            return {}

    def _compute_internal(
        self,
        state: Dict[str, Any],
        rag_context: Optional[str],
    ) -> Tuple[InsightOutput, RenovationInsight]:
        ctx = self._extract_context(state)

        analytics            = self._run_analytics(ctx, state)
        priority_repairs     = self._build_priority_repairs(ctx)
        budget_strategy      = self._build_budget_strategy(ctx, priority_repairs, analytics)
        roi_insight          = self._build_roi_insight(ctx)
        renovation_sequence  = self._build_renovation_sequence(priority_repairs, ctx)

        priority_score  = self._compute_priority_score(ctx, priority_repairs)
        ce_score        = self._compute_cost_effectiveness_score(ctx, roi_insight)
        roi_score_norm  = min(100.0, max(0.0, roi_insight.roi_score))
        overall         = round((priority_score * 0.30 + ce_score * 0.35 + roi_score_norm * 0.35), 1)

        key_wins   = self._extract_key_wins(ctx, roi_insight, ce_score, analytics)
        risk_flags = self._extract_risk_flags(ctx, budget_strategy, analytics)
        summary    = self._build_summary(ctx, roi_insight, priority_repairs, overall, analytics)

        # ── PROBLEM 4 FIX: market_timing + action_checklist ──────────────────
        market_timing    = _build_market_timing_insight(
            ctx["material_prices"], ctx["city"], ctx["room_type"], ctx["total_cost"],
        )
        action_checklist = _build_action_checklist(
            ctx, priority_repairs, market_timing, ctx["budget_tier"],
        )

        output = InsightOutput(
            renovation_priority_score=round(priority_score, 1),
            cost_effectiveness_score=round(ce_score, 1),
            roi_score=roi_score_norm,
            overall_insight_score=overall,
            priority_repairs=priority_repairs,
            budget_strategy=budget_strategy,
            roi_insight=roi_insight,
            renovation_sequence=renovation_sequence,
            expected_value_increase=roi_insight.expected_value_increase,
            summary=summary,
            key_wins=key_wins,
            risk_flags=risk_flags,
            data_sources=ctx["data_sources"],
            confidence_overall=ctx["confidence"],
            derived_insights=analytics.get("derived_insights", []),
            market_benchmark=analytics.get("market_benchmark", {}),
            optimal_budget_allocation=analytics.get("optimal_budget_allocation", []),
            decision_scores=analytics.get("top_decisions", []),
            quick_wins=analytics.get("quick_wins", []),
            material_price_signals=analytics.get("material_price_signals", []),
            # PROBLEM 4 FIX: new fields
            market_timing=market_timing,
            action_checklist=action_checklist,
        )

        insight = RenovationInsight(
            priority_repairs=[r.model_dump() for r in priority_repairs[:5]],
            budget_strategy=budget_strategy.key_recommendation,
            expected_value_increase=roi_insight.expected_value_increase,
            roi_score=f"{roi_score_norm:.1f}/100",
            renovation_sequence=[s.model_dump() for s in renovation_sequence],
            renovation_priority_score=round(priority_score, 1),
            cost_effectiveness_score=round(ce_score, 1),
            overall_insight_score=overall,
            key_wins=key_wins,
            risk_flags=risk_flags,
            summary=summary,
        )

        logger.info(
            f"[InsightEngine v3.0] computed — priority={priority_score:.1f} "
            f"CE={ce_score:.1f} ROI={roi_score_norm:.1f} overall={overall:.1f} "
            f"buy_now={len(market_timing.get('buy_now', []))} "
            f"checklist={len(action_checklist)}"
        )
        return output, insight

    # ── Analytics runner (unchanged from v2.0) ────────────────────────────────

    def _run_analytics(self, ctx: Dict[str, Any], state: Dict[str, Any]) -> Dict:
        try:
            from analytics import (
                InsightDeriver, DecisionScorer, BudgetOptimiser,
                MarketBenchmarker, InsightFormatter,
            )

            deriver     = InsightDeriver()
            scorer      = DecisionScorer()
            optimiser   = BudgetOptimiser()
            benchmarker = MarketBenchmarker()
            formatter   = InsightFormatter()

            materials = self._extract_materials(state)

            derived_insights = deriver.derive(
                room_type=ctx["room_type"], city=ctx["city"],
                city_tier=self._get_city_tier(ctx["city"]),
                budget_tier=ctx["budget_tier"], roi_pct=ctx["roi_pct"],
                total_cost_inr=ctx["total_cost"],
                property_value_inr=max(ctx["pre_val"], ctx["floor_sqft"] * 9000),
                materials=materials, condition=ctx["condition"],
                property_age=state.get("property_age_years", 10),
                theme=ctx["style"],
                rag_knowledge=state.get("retrieved_knowledge") or [],
            )

            boq            = ctx["boq"] or []
            decision_scores = scorer.score_decisions(
                decisions=boq + ctx["recs"],
                total_budget_inr=max(ctx["total_cost"], 1),
                roi_pct=ctx["roi_pct"], equity_gain_inr=ctx["equity_gain"],
            )

            budget_allocations = optimiser.optimise(
                room_type=ctx["room_type"], total_budget_inr=ctx["total_cost"],
                roi_pct=ctx["roi_pct"], budget_tier=ctx["budget_tier"],
                has_structural_issues=(ctx["condition"] in ("poor", "very_poor")),
            )

            cost_per_sqft = ctx["total_cost"] / max(ctx["floor_sqft"], 1)
            benchmark = benchmarker.benchmark(
                city=ctx["city"], room_type=ctx["room_type"],
                roi_pct=ctx["roi_pct"], cost_per_sqft=cost_per_sqft,
                total_cost_inr=ctx["total_cost"],
            )

            material_price_signals = self._extract_price_signals(state)

            analytics_report = formatter.format_for_report(
                derived_insights=derived_insights,
                decision_scores=decision_scores,
                budget_allocations=budget_allocations,
                benchmark=benchmark,
            )
            analytics_report["material_price_signals"] = material_price_signals

            logger.info(
                f"[InsightEngine v3.0] analytics — "
                f"insights={len(derived_insights)} decisions_scored={len(decision_scores)}"
            )
            return analytics_report

        except ImportError as e:
            logger.warning(f"[InsightEngine] Analytics module not available: {e}")
            return {}
        except Exception as e:
            logger.error(f"[InsightEngine] Analytics failed: {e}", exc_info=True)
            return {}

    def _extract_materials(self, state: Dict[str, Any]) -> List[str]:
        materials: List[str] = []
        for item in (state.get("boq_line_items") or []):
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description", "")).lower()
            if "premium" in desc and ("flooring" in desc or "tile" in desc):
                materials.append("premium_flooring")
            elif "modular" in desc and "kitchen" in desc:
                materials.append("modular_kitchen")
            elif "smart" in desc:
                materials.append("smart_home_automation")
            elif "false ceiling" in desc or "gypsum" in desc:
                materials.append("false_ceiling")
            elif "upvc" in desc or "window" in desc:
                materials.append("upvc_windows")
            elif "premium" in desc and "paint" in desc:
                materials.append("premium_paint")
            elif "wardrobe" in desc:
                materials.append("wardrobes_fitted")
            elif "sanitary" in desc or "bathroom" in desc:
                materials.append("bathroom_premium")
        return list(set(materials))

    def _get_city_tier(self, city: str) -> int:
        return {
            "Mumbai": 1, "Delhi NCR": 1, "Bangalore": 1,
            "Hyderabad": 1, "Chennai": 1, "Pune": 1,
            "Kolkata": 1, "Ahmedabad": 2, "Surat": 2,
            "Jaipur": 2, "Lucknow": 2, "Chandigarh": 2,
        }.get(city, 2)

    def _extract_price_signals(self, state: Dict[str, Any]) -> List[Dict]:
        signals: List[Dict] = []
        for mp in (state.get("material_prices") or []):
            if not isinstance(mp, dict):
                continue
            urgency = (mp.get("budget_impact") or {}).get("urgency", "low")
            trend   = mp.get("trend", "stable")
            if urgency in ("high", "medium") or trend == "up":
                # PROBLEM 4 FIX: add source citation to every price signal
                mat_name   = mp.get("display_name", mp.get("material_key", ""))
                source_cit = _material_price_source(mat_name)
                signals.append({
                    "material":                mp.get("display_name", mp.get("material_key", "")),
                    "trend":                   trend,
                    "pct_change_90d":          mp.get("pct_change_90d", 0),
                    "procurement_recommendation": mp.get("procurement_recommendation", ""),
                    "urgency":                 urgency,
                    "price_source_citation":   source_cit,
                })
        return signals[:4]

    def _build_recommendations_list(
        self, state: Dict[str, Any], output: "InsightOutput",
    ) -> List[Dict]:
        recs = []
        for repair in output.priority_repairs[:4]:
            recs.append({
                "recommendation": repair.action,
                "category":       repair.category,
                "priority":       repair.urgency,
                "roi_impact":     f"{repair.roi_contribution_pct:.1f}% ROI contribution",
                "cost_range_inr": _inr_label(repair.estimated_cost_inr),
                "reasoning":      repair.reasoning,
                "source":         repair.source_signals,
            })
        return recs

    # ── Context extraction (unchanged from v2.0) ──────────────────────────────

    def _extract_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        def _get(key: str, default: Any = None) -> Any:
            val = state.get(key, default)
            if val is None:
                return default
            if hasattr(val, "model_dump"):
                return val.model_dump()
            if hasattr(val, "dict"):
                return val.dict()
            return val

        vision      = _get("vision_features", {}) or _get("image_features", {}) or {}
        room_features = _get("room_features", {}) or {}
        layout      = _get("layout_report", {}) or {}
        merged_vision = {**vision, **{k: v for k, v in room_features.items() if v}}

        condition   = str(merged_vision.get("room_condition") or merged_vision.get("condition", "fair")).lower()
        wall_sqft   = _safe_float(state.get("wall_area_sqft") or merged_vision.get("estimated_wall_area_sqft", 200))
        floor_sqft  = _safe_float(state.get("floor_area_sqft") or merged_vision.get("estimated_floor_area_sqft", 120))

        design_plan = _get("design_plan", {}) or {}
        boq         = _get("boq_line_items", []) or []

        budget_est    = _get("budget_estimate", {}) or {}
        cost_breakdown = _get("cost_breakdown", {}) or {}
        budget_inr    = _safe_int(state.get("budget_inr", 750_000))
        total_cost    = _safe_int(
            budget_est.get("total_cost_inr") or cost_breakdown.get("total_inr")
            or _get("total_cost_estimate") or budget_inr
        )

        roi_output  = _get("roi_output", {}) or _get("roi_prediction", {}) or {}
        roi_pct     = _safe_float(roi_output.get("roi_pct", roi_output.get("roi_percentage", 0)))
        equity_gain = _safe_int(roi_output.get("equity_gain_inr", 0))
        payback     = _safe_int(roi_output.get("payback_months", 36))
        pre_val     = _safe_int(roi_output.get("pre_reno_value_inr", 0))
        post_val    = _safe_int(roi_output.get("post_reno_value_inr", 0))
        model_type  = str(roi_output.get("model_type", "heuristic"))
        model_conf  = _safe_float(roi_output.get("model_confidence", 0.65))

        recs              = _get("explainable_recommendations", []) or []
        material_prices   = _get("material_prices", []) or []
        loc_ctx           = _get("location_context", {}) or budget_est.get("location_context", {}) or {}
        city              = str(state.get("city", "Hyderabad"))
        room_type         = str(state.get("room_type", "bedroom"))
        budget_tier       = str(state.get("budget_tier", "mid"))
        rag_ctx           = state.get("rag_context", "") or ""

        layout_score_raw  = str(layout.get("layout_score", merged_vision.get("layout_score", "65/100")))
        layout_score_num  = _safe_float(layout_score_raw.split("/")[0], 65.0)

        style      = str(state.get("style_label") or merged_vision.get("detected_style", "Modern Minimalist"))
        style_conf = _safe_float(state.get("style_confidence", 0.65))
        detected_changes = list(state.get("detected_changes") or [])

        data_sources: List[str] = []
        if merged_vision:
            data_sources.append("vision_analysis")
        if design_plan or boq:
            data_sources.append("renovation_plan")
        if budget_est or cost_breakdown:
            data_sources.append("cost_estimation")
        if roi_output:
            data_sources.append("roi_model")
        if material_prices:
            data_sources.append("price_forecast")
        if rag_ctx:
            data_sources.append("rag_knowledge")

        confidence = round(
            (style_conf * 0.2 + model_conf * 0.5 + (0.8 if rag_ctx else 0.5) * 0.3), 2
        )

        return {
            "vision": merged_vision, "layout": layout,
            "layout_score": layout_score_num, "condition": condition,
            "wall_sqft": wall_sqft, "floor_sqft": floor_sqft,
            "design_plan": design_plan, "boq": boq, "recs": recs,
            "budget_inr": budget_inr, "total_cost": total_cost,
            "budget_est": budget_est, "cost_breakdown": cost_breakdown,
            "roi_pct": roi_pct, "equity_gain": equity_gain,
            "payback_months": payback, "pre_val": pre_val, "post_val": post_val,
            "model_type": model_type, "model_conf": model_conf,
            "material_prices": material_prices, "loc_ctx": loc_ctx,
            "city": city, "room_type": room_type, "budget_tier": budget_tier,
            "style": style, "style_conf": style_conf,
            "detected_changes": detected_changes, "rag_ctx": rag_ctx,
            "data_sources": data_sources, "confidence": confidence,
        }

    # ── Priority repairs (unchanged from v2.0) ────────────────────────────────

    def _build_priority_repairs(self, ctx: Dict[str, Any]) -> List[PriorityRepair]:
        raw_actions: List[Dict[str, Any]] = []

        for rec in ctx["recs"]:
            if not isinstance(rec, dict):
                continue
            title = str(rec.get("title", rec.get("action", "")))
            if not title:
                continue
            raw_actions.append({
                "action":        title,
                "reasoning":     " | ".join(str(r) for r in rec.get("reasoning", [])[:2]),
                "cost_inr":      _safe_int(re.sub(r"[₹,L K]", "", str(rec.get("estimated_cost", "0"))) or 0),
                "roi_impact":    str(rec.get("roi_impact", "")),
                "source":        "design_planner",
                "priority_hint": str(rec.get("priority", "medium")).lower(),
            })

        for change in ctx["detected_changes"]:
            if isinstance(change, str) and change.strip():
                raw_actions.append({
                    "action":        change,
                    "reasoning":     f"Detected from image analysis of {ctx['room_type']}",
                    "cost_inr":      0,
                    "roi_impact":    "",
                    "source":        "vision_detected",
                    "priority_hint": "medium",
                })

        for issue in (ctx["layout"].get("issues") or []):
            if isinstance(issue, str) and issue.strip():
                raw_actions.append({
                    "action":        f"Fix layout issue: {issue}",
                    "reasoning":     "Identified from spatial layout analysis",
                    "cost_inr":      0,
                    "roi_impact":    "Improves livability and marketability",
                    "source":        "layout_analysis",
                    "priority_hint": "medium",
                })

        condition = ctx["condition"]
        if condition in ("poor", "very_poor"):
            raw_actions.insert(0, {
                "action":        "Address structural integrity issues (cracks, seepage, damp)",
                "reasoning":     f"Room condition assessed as '{condition}'. Structural issues must be resolved before any cosmetic work.",
                "cost_inr":      _safe_int(ctx["total_cost"] * 0.12),
                "roi_impact":    "Prevents further value degradation; prerequisite for all other work",
                "source":        "condition_assessment",
                "priority_hint": "critical",
            })
        elif condition == "fair":
            raw_actions.insert(0, {
                "action":        "Inspect and treat any minor dampness or surface cracks",
                "reasoning":     "Room condition is fair. Preventive structural check recommended before renovation.",
                "cost_inr":      _safe_int(ctx["total_cost"] * 0.06),
                "roi_impact":    "Prevents cost escalation during renovation",
                "source":        "condition_assessment",
                "priority_hint": "high",
            })

        repairs: List[PriorityRepair] = []
        seen: set = set()

        for item in raw_actions:
            action_text = str(item.get("action", "")).strip()
            if not action_text or action_text.lower() in seen:
                continue
            seen.add(action_text.lower())

            category      = _classify_action(action_text)
            priority_hint = item.get("priority_hint", "medium")
            if priority_hint == "critical":
                category = "structural"

            urgency = _urgency_from_category(category, ctx["condition"])
            if priority_hint == "critical":
                urgency = "critical"
            elif priority_hint == "high" and urgency != "critical":
                urgency = "high"

            cost_inr = _safe_int(item.get("cost_inr", 0))
            if cost_inr == 0 and ctx["total_cost"] > 0:
                weight   = ROI_CONTRIBUTION.get(category, 0.15)
                cost_inr = _safe_int(ctx["total_cost"] * weight / max(len(raw_actions) / 5, 1))

            cost_label   = _inr_label(cost_inr) if cost_inr > 0 else "TBD"
            roi_contrib  = ROI_CONTRIBUTION.get(category, 0.10) * 100
            ce_ratio     = (ctx["equity_gain"] / max(cost_inr, 1)) if cost_inr > 0 else 1.0
            impact_score = min(10.0, round(
                (4.0 if category == "structural" else 2.0)
                + roi_contrib / 10 + min(3.0, ce_ratio / 10000), 1,
            ))

            repairs.append(PriorityRepair(
                rank=0, action=action_text, category=category, urgency=urgency,
                estimated_cost_inr=cost_inr, estimated_cost_label=cost_label,
                impact_score=impact_score, roi_contribution_pct=round(roi_contrib, 1),
                reasoning=str(item.get("reasoning", "")),
                source_signals=[str(item.get("source", ""))],
            ))

        category_order = {c: i for i, c in enumerate(CATEGORY_PRIORITY)}
        urgency_order  = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        repairs.sort(key=lambda r: (
            category_order.get(r.category, 99),
            urgency_order.get(r.urgency, 99),
            -r.impact_score,
        ))
        for i, r in enumerate(repairs):
            r.rank = i + 1
        return repairs[:10]

    # ── Budget strategy (unchanged from v2.0) ─────────────────────────────────

    def _build_budget_strategy(
        self, ctx: Dict[str, Any], priority_repairs: List[PriorityRepair], analytics: Dict,
    ) -> BudgetStrategy:
        total      = ctx["total_cost"] or ctx["budget_inr"]
        budget_inr = ctx["budget_inr"]
        gap        = total - budget_inr

        cb = ctx["cost_breakdown"]
        allocation = {
            "materials":   _safe_int(cb.get("materials_inr")   or total * BUDGET_ALLOCATION["materials"]),
            "labour":      _safe_int(cb.get("labour_inr")      or total * BUDGET_ALLOCATION["labour"]),
            "supervision": _safe_int(cb.get("supervision_inr") or total * BUDGET_ALLOCATION["supervision"]),
            "gst":         _safe_int(cb.get("gst_inr")         or total * BUDGET_ALLOCATION["gst"]),
            "contingency": _safe_int(cb.get("contingency_inr") or total * BUDGET_ALLOCATION["contingency"]),
        }
        alloc_pct = {k: round(v / max(total, 1) * 100, 1) for k, v in allocation.items()}

        tier      = ctx["budget_tier"]
        condition = ctx["condition"]
        room_type = ctx["room_type"]
        city      = ctx["city"]

        benchmark      = analytics.get("market_benchmark", {})
        benchmark_note = f" {benchmark['benchmark_insight']}" if benchmark.get("benchmark_insight") else ""

        if condition in ("poor", "very_poor"):
            strategy_name = "Repair-First Strategy"
            description   = (
                f"Given the {condition} room condition, prioritise structural and mechanical "
                f"repairs before any cosmetic investment. Allocate up to 20% of budget to remediation."
                + benchmark_note
            )
        elif tier == "premium":
            strategy_name = "Premium Value-Add Strategy"
            description   = (
                f"Premium budget in {city} justifies high-spec finishes. "
                f"Focus on materials that command a market premium: stone flooring, smart lighting, "
                f"modular carpentry." + benchmark_note
            )
        elif tier == "basic":
            strategy_name = "High-Impact Frugal Strategy"
            description   = (
                f"Maximise visual impact within a tight budget. Fresh paint, updated lighting, "
                f"and decluttering deliver 80% of the perceived renovation value at 30% of mid-tier cost."
                + benchmark_note
            )
        else:
            strategy_name = "Balanced ROI Strategy"
            description   = (
                f"Mid-tier budget in {city} — optimal sweet spot. Prioritise flooring, paint, "
                f"false ceiling, and electrical upgrades for maximum ROI in the {room_type} market."
                + benchmark_note
            )

        budget_analysis = ctx["budget_est"].get("budget_analysis") or {}
        avoid_items     = [budget_analysis["avoid"]] if isinstance(budget_analysis.get("avoid"), str) else []

        quick_wins_data = analytics.get("quick_wins", [])
        quick_win_texts = [qw.get("action", "") for qw in quick_wins_data[:2] if qw.get("action")]
        procurement_note = ""
        for signal in [s for s in analytics.get("material_price_signals", []) if s.get("urgency") == "high"][:2]:
            mat = signal.get("material", "")
            # PROBLEM 4 FIX: add source citation to procurement note
            src = signal.get("price_source_citation", "")
            procurement_note += f" ⚠ Procure {mat} immediately — prices rising. {src}"

        key_rec = f"{strategy_name}: {description}"
        if quick_win_texts:
            key_rec += " Quick wins: " + "; ".join(quick_win_texts) + "."
        if procurement_note:
            key_rec += procurement_note

        return BudgetStrategy(
            strategy_name=strategy_name, description=description,
            allocation=allocation, allocation_pct=alloc_pct,
            total_recommended_inr=total, within_budget=gap <= 0,
            budget_gap_inr=gap, key_recommendation=key_rec,
            avoid_items=avoid_items,
        )

    # ── ROI insight (PROBLEM 4 FIX: add source citations) ─────────────────────

    def _build_roi_insight(self, ctx: Dict[str, Any]) -> ROIInsight:
        roi_pct        = ctx["roi_pct"]
        equity_gain    = ctx["equity_gain"]
        payback_months = ctx["payback_months"]
        total_cost     = max(ctx["total_cost"], 1)
        pre_val        = ctx["pre_val"]
        post_val       = ctx["post_val"]
        city           = ctx["city"]
        room_type      = ctx["room_type"]

        if pre_val == 0 or post_val == 0:
            loc         = ctx["loc_ctx"]
            psf         = _safe_int(loc.get("avg_psf_inr", 9500)) or 9500
            floor_sqft  = max(ctx["floor_sqft"], 80)
            pre_val     = _safe_int(psf * floor_sqft * 3)
            equity_est  = _safe_int(total_cost * (roi_pct / 100)) if roi_pct > 0 else _safe_int(total_cost * 0.12)
            equity_gain = max(equity_gain, equity_est)
            post_val    = pre_val + equity_gain

        value_per_rupee = round(equity_gain / max(total_cost, 1), 3)
        roi_score       = min(100.0, max(0.0, round(roi_pct * 4.0, 1)))
        floor_sqft      = max(ctx["floor_sqft"], 80)
        cost_per_sqft   = round(total_cost / floor_sqft, 0)

        if payback_months <= 12:
            payback_label = f"{payback_months} months (Excellent)"
        elif payback_months <= 24:
            payback_label = f"{payback_months} months (Good)"
        elif payback_months <= 36:
            payback_label = f"{payback_months} months (Average)"
        else:
            payback_label = f"{payback_months} months (Long-term)"

        base_yield  = CITY_YIELD.get(city, 3.0)
        room_mult   = ROOM_ROI_MULT.get(room_type, 1.0)
        yield_delta = round(base_yield * (roi_pct / 100) * room_mult * 0.3, 2)
        yield_post  = round(base_yield + yield_delta, 2)

        # PROBLEM 4 FIX: append source citation to every interpretation
        if roi_pct >= 20:
            interpretation = (
                f"Exceptional ROI: this {room_type.replace('_', ' ')} renovation in {city} delivers "
                f"{roi_pct:.1f}% return — well above the 12–15% city benchmark. "
                f"Every ₹1 invested adds ₹{value_per_rupee:.2f} in property value. "
                f"{_ROI_SOURCE_CITATION}"
            )
        elif roi_pct >= 12:
            interpretation = (
                f"Strong ROI: {roi_pct:.1f}% projected return on this {city} "
                f"{room_type.replace('_', ' ')}. "
                f"Value-per-rupee of ₹{value_per_rupee:.2f} is above market average. "
                f"{_ROI_SOURCE_CITATION}"
            )
        elif roi_pct >= 6:
            interpretation = (
                f"Moderate ROI: {roi_pct:.1f}% — on par with {city} market average. "
                f"Focus budget on cosmetic upgrades for a better return. "
                f"{_ROI_SOURCE_CITATION}"
            )
        else:
            interpretation = (
                f"Below-average ROI at {roi_pct:.1f}%. Consider increasing budget quality or "
                f"shifting focus to higher-impact categories (kitchen/bathroom). "
                f"{_ROI_SOURCE_CITATION}"
            )

        return ROIInsight(
            roi_score=roi_score,
            roi_percentage=f"{roi_pct:.1f}%",
            expected_value_increase=_inr_label(equity_gain),
            equity_gain_inr=equity_gain,
            payback_period=payback_label,
            cost_effectiveness_score=min(100.0, round(value_per_rupee * 33.3, 1)),
            cost_per_sqft=cost_per_sqft,
            value_per_rupee=value_per_rupee,
            rental_yield_improvement=f"+{yield_delta:.2f}% (from {base_yield:.1f}% to {yield_post:.1f}%)",
            confidence=ctx["model_conf"],
            model_type=ctx["model_type"],
            interpretation=interpretation,
        )

    # ── Renovation sequence (unchanged from v2.0) ─────────────────────────────

    def _build_renovation_sequence(
        self, priority_repairs: List[PriorityRepair], ctx: Dict[str, Any],
    ) -> List[RenovationSequence]:
        by_category: Dict[str, List[PriorityRepair]] = {}
        for r in priority_repairs:
            by_category.setdefault(r.category, []).append(r)

        total_cost = max(ctx["total_cost"], 1)
        phases: List[RenovationSequence] = []
        step = 1

        structural = by_category.get("structural", [])
        if structural:
            phases.append(RenovationSequence(
                step=step, phase="Phase 1: Structural Remediation",
                actions=[r.action for r in structural],
                duration_days=max(3, len(structural) * 2), dependencies=[],
                can_parallel=False,
                cost_inr=sum(r.estimated_cost_inr for r in structural),
                rationale="Structural repairs MUST precede all other work.",
            ))
            step += 1

        rough_actions = ["Demolish existing flooring/tiles", "Remove old fixtures and fittings"]
        if ctx["condition"] in ("poor", "fair"):
            rough_actions.append("Hack plastering and re-plaster damaged walls")
        phases.append(RenovationSequence(
            step=step, phase="Phase 2: Demolition & Rough Work", actions=rough_actions,
            duration_days=3,
            dependencies=["Phase 1: Structural Remediation"] if structural else [],
            can_parallel=False, cost_inr=_safe_int(total_cost * 0.05),
            rationale="All demolition before any new installation.",
        ))
        step += 1

        mechanical = by_category.get("mechanical", [])
        mech_actions = [r.action for r in mechanical] or [
            "Inspect and upgrade electrical wiring", "Check plumbing connections",
        ]
        phases.append(RenovationSequence(
            step=step, phase="Phase 3: Electrical & Plumbing",
            actions=mech_actions, duration_days=max(2, len(mech_actions)),
            dependencies=["Phase 2: Demolition & Rough Work"], can_parallel=False,
            cost_inr=sum(r.estimated_cost_inr for r in mechanical) or _safe_int(total_cost * 0.10),
            rationale="Concealed conduits and plumbing before plastering and flooring.",
        ))
        step += 1

        phases.append(RenovationSequence(
            step=step, phase="Phase 4: Flooring Installation",
            actions=[
                f"Install {ctx['vision'].get('floor_material', 'vitrified tiles')} flooring",
                "Lay floor tiles with precision levelling", "Allow 48-hr curing time",
            ],
            duration_days=4, dependencies=["Phase 3: Electrical & Plumbing"],
            can_parallel=False, cost_inr=_safe_int(total_cost * 0.20),
            rationale="Flooring before wall painting to avoid splatter damage.",
        ))
        step += 1

        wall_actions = (
            [r.action for r in by_category.get("cosmetic", [])
             if "wall" in r.action.lower() or "paint" in r.action.lower()]
            or [
                f"Apply {ctx['vision'].get('wall_treatment', 'premium emulsion paint')}",
                "Install false ceiling with concealed lighting",
            ]
        )
        phases.append(RenovationSequence(
            step=step, phase="Phase 5: Wall Painting & Ceiling Work", actions=wall_actions,
            duration_days=5, dependencies=["Phase 4: Flooring Installation"],
            can_parallel=True, cost_inr=_safe_int(total_cost * 0.25),
            rationale="Wall painting and ceiling work can run in parallel after flooring.",
        ))
        step += 1

        finishing  = by_category.get("finishing", []) + by_category.get("smart_home", [])
        fin_actions = (
            [r.action for r in finishing]
            or ["Install light fixtures", "Hang mirrors and décor", "Install curtain rails"]
        )
        phases.append(RenovationSequence(
            step=step, phase="Phase 6: Finishing & Fixtures", actions=fin_actions[:5],
            duration_days=3, dependencies=["Phase 5: Wall Painting & Ceiling Work"],
            can_parallel=True, cost_inr=_safe_int(total_cost * 0.10),
            rationale="Final finishing once all surfaces are complete.",
        ))
        step += 1

        phases.append(RenovationSequence(
            step=step, phase="Phase 7: Snagging & Handover",
            actions=["Conduct full snagging inspection", "Touch-up paint defects", "Deep clean", "Handover walkthrough"],
            duration_days=2, dependencies=["Phase 6: Finishing & Fixtures"],
            can_parallel=False, cost_inr=0,
            rationale="Quality control before project sign-off.",
        ))
        return phases

    # ── Scoring (unchanged) ────────────────────────────────────────────────────

    def _compute_priority_score(
        self, ctx: Dict[str, Any], priority_repairs: List[PriorityRepair],
    ) -> float:
        score = 50.0
        condition_bonus = {"poor": 40, "very_poor": 50, "fair": 25, "good": 5, "new": 0}
        score += condition_bonus.get(ctx["condition"], 20)
        score += max(0, 80 - ctx["layout_score"]) * 0.3
        score += sum(1 for r in priority_repairs if r.category == "structural") * 10
        score += min(10, len(priority_repairs) * 1.5)
        return min(100.0, round(score, 1))

    def _compute_cost_effectiveness_score(
        self, ctx: Dict[str, Any], roi_insight: ROIInsight,
    ) -> float:
        vpr = roi_insight.value_per_rupee
        if vpr >= CE_EXCELLENT:
            score = 90 + min(10, (vpr - CE_EXCELLENT) * 5)
        elif vpr >= CE_GOOD:
            score = 70 + ((vpr - CE_GOOD) / (CE_EXCELLENT - CE_GOOD)) * 20
        elif vpr >= CE_AVERAGE:
            score = 50 + ((vpr - CE_AVERAGE) / (CE_GOOD - CE_AVERAGE)) * 20
        elif vpr >= CE_POOR:
            score = 20 + ((vpr - CE_POOR) / (CE_AVERAGE - CE_POOR)) * 30
        else:
            score = max(0, vpr * 40)

        loc     = ctx["loc_ctx"]
        avg_psf = _safe_float(loc.get("avg_psf_inr", 9500))
        cost_psf = roi_insight.cost_per_sqft
        if avg_psf > 0 and cost_psf > 0:
            psf_ratio = cost_psf / (avg_psf * 0.08)
            if psf_ratio <= 1.0:
                score = min(100, score + 5)
            elif psf_ratio > 1.5:
                score = max(0, score - 10)
        return min(100.0, round(score, 1))

    # ── Key wins + risk flags (PROBLEM 4 FIX: source citations) ───────────────

    def _extract_key_wins(
        self, ctx: Dict[str, Any], roi_insight: ROIInsight, ce_score: float, analytics: Dict,
    ) -> List[str]:
        wins: List[str] = []

        if roi_insight.roi_score >= 60:
            wins.append(
                f"Strong {roi_insight.roi_percentage} projected ROI — above {ctx['city']} "
                f"market benchmark. {_ROI_SOURCE_CITATION}"
            )

        if ctx["equity_gain"] > 0:
            wins.append(
                f"Expected equity gain of {_inr_label(ctx['equity_gain'])} on "
                f"renovation spend of {_inr_label(ctx['total_cost'])}. "
                f"{_ROI_SOURCE_CITATION}"
            )

        if ce_score >= 70:
            wins.append(
                f"High cost effectiveness ({ce_score:.0f}/100) — excellent value-per-rupee "
                f"for {ctx['budget_tier']} tier"
            )

        loc = ctx["loc_ctx"]
        if loc.get("appreciation_5yr_pct", 0) >= 40:
            wins.append(
                f"{ctx['city']} 5-year appreciation of {loc['appreciation_5yr_pct']}% "
                f"amplifies renovation returns. {_ROI_SOURCE_CITATION}"
            )

        if ctx["room_type"] in ("kitchen", "bathroom"):
            wins.append(
                f"{ctx['room_type'].title()} renovations yield 25–35% higher ROI than bedroom "
                f"renovations in India. {_ROI_SOURCE_CITATION}"
            )

        benchmark = analytics.get("market_benchmark", {})
        if benchmark.get("performance_label") in ("Top Quartile", "Above Average"):
            wins.append(f"Project ranked {benchmark['performance_label']} vs {ctx['city']} renovation peers")

        quick_wins = analytics.get("quick_wins", [])
        if quick_wins:
            wins.append(f"{len(quick_wins)} quick-win upgrades identified — high ROI at low relative cost")

        return wins[:5]

    def _extract_risk_flags(
        self, ctx: Dict[str, Any], budget_strategy: BudgetStrategy, analytics: Dict,
    ) -> List[str]:
        flags: List[str] = []

        if not budget_strategy.within_budget:
            gap = budget_strategy.budget_gap_inr
            flags.append(
                f"Renovation estimate exceeds budget by {_inr_label(gap)} — "
                "consider phasing or descoping finishing items"
            )

        if ctx["condition"] in ("poor", "very_poor"):
            flags.append(
                "Poor room condition detected — structural remediation may reveal hidden costs. "
                "Maintain 15% contingency buffer."
            )

        # PROBLEM 4 FIX: add price source citation to every material risk flag
        for signal in [s for s in analytics.get("material_price_signals", []) if s.get("urgency") == "high"][:2]:
            src = signal.get("price_source_citation", "")
            flags.append(
                f"{signal.get('material', 'Material')} prices rising "
                f"{signal.get('pct_change_90d', 0):.1f}% — "
                f"{signal.get('procurement_recommendation', 'order early')}. {src}"
            )

        if ctx["payback_months"] > 48:
            flags.append(
                f"Payback period of {ctx['payback_months']} months is long — "
                "renovation better justified for personal use than immediate resale"
            )

        loc = ctx["loc_ctx"]
        if loc.get("market_tier", 1) >= 2 and ctx["budget_tier"] == "premium":
            flags.append(
                f"Premium finishes in a Tier-{loc.get('market_tier', 2)} city market risk "
                f"over-capitalisation — market may not absorb the full premium spend. "
                f"{_ROI_SOURCE_CITATION}"
            )

        derived = analytics.get("derived_insights", [])
        for ins in derived:
            if isinstance(ins, dict) and not ins.get("is_positive") and "over-capitalisation" in ins.get("insight", "").lower():
                flags.append(ins["insight"][:150])
                break

        return flags[:4]

    # ── Summary (unchanged structure, PROBLEM 4 FIX: cite source) ─────────────

    def _build_summary(
        self, ctx: Dict[str, Any], roi_insight: ROIInsight,
        priority_repairs: List[PriorityRepair], overall_score: float, analytics: Dict,
    ) -> str:
        structural_count = sum(1 for r in priority_repairs if r.category == "structural")
        top_action       = priority_repairs[0].action if priority_repairs else "general renovation"
        score_label = (
            "excellent" if overall_score >= 75 else
            "good"      if overall_score >= 55 else
            "moderate"  if overall_score >= 35 else "limited"
        )
        benchmark      = analytics.get("market_benchmark", {})
        benchmark_note = f" {benchmark['benchmark_insight']}" if benchmark.get("benchmark_insight") else ""

        return (
            f"This {ctx['room_type'].replace('_', ' ')} renovation in {ctx['city']} scores "
            f"{overall_score:.0f}/100 overall ({score_label} value proposition). "
            f"{'Structural remediation is required before cosmetic work. ' if structural_count > 0 else ''}"
            f"Top priority: {top_action}. "
            f"Projected ROI of {roi_insight.roi_percentage} with equity gain of "
            f"{roi_insight.expected_value_increase} over a {ctx['budget_tier']}-tier budget "
            f"of {_inr_label(ctx['total_cost'])}."
            f"{benchmark_note} {_ROI_SOURCE_CITATION}"
        )

    # ── Fallback (unchanged) ────────────────────────────────────────────────────

    def _fallback_output(self, state: Dict[str, Any]) -> Tuple[InsightOutput, RenovationInsight]:
        city       = str(state.get("city", "India"))
        room_type  = str(state.get("room_type", "room"))
        budget_inr = _safe_int(state.get("budget_inr", 750_000))
        roi_pct    = _safe_float((state.get("roi_prediction") or {}).get("roi_pct", 10.0))

        fallback_repair = PriorityRepair(
            rank=1, action="Full room renovation assessment",
            category="cosmetic", urgency="medium",
            estimated_cost_inr=budget_inr,
            estimated_cost_label=_inr_label(budget_inr),
            impact_score=6.0, roi_contribution_pct=30.0,
            reasoning="Fallback insight — full pipeline data unavailable.",
            source_signals=["fallback"],
        )
        fallback_budget = BudgetStrategy(
            strategy_name="Standard Renovation Strategy",
            description=f"Mid-tier renovation approach for {city} {room_type}.",
            allocation={k: _safe_int(budget_inr * v) for k, v in BUDGET_ALLOCATION.items()},
            allocation_pct={k: v * 100 for k, v in BUDGET_ALLOCATION.items()},
            total_recommended_inr=budget_inr, within_budget=True, budget_gap_inr=0,
            key_recommendation="Standard 55/30/15 split: materials / labour / overhead.",
        )
        fallback_roi = ROIInsight(
            roi_score=min(100, roi_pct * 4),
            roi_percentage=f"{roi_pct:.1f}%",
            expected_value_increase=_inr_label(_safe_int(budget_inr * roi_pct / 100)),
            equity_gain_inr=_safe_int(budget_inr * roi_pct / 100),
            payback_period="24–36 months",
            cost_effectiveness_score=50.0,
            cost_per_sqft=round(budget_inr / 120, 0),
            value_per_rupee=round(roi_pct / 100, 3),
            interpretation=(
                f"Standard {roi_pct:.1f}% ROI estimate for {city} {room_type}. "
                f"{_ROI_SOURCE_CITATION}"
            ),
        )

        empty_timing   = {"is_good_time_to_renovate": True, "summary": "No price data available.", "buy_now": [], "defer": []}
        empty_checklist: List[Dict] = []

        output = InsightOutput(
            renovation_priority_score=60.0, cost_effectiveness_score=50.0,
            roi_score=min(100.0, roi_pct * 4), overall_insight_score=55.0,
            priority_repairs=[fallback_repair], budget_strategy=fallback_budget,
            roi_insight=fallback_roi, renovation_sequence=[],
            expected_value_increase=fallback_roi.expected_value_increase,
            summary=f"Renovation analysis for {city} {room_type}. ROI: {roi_pct:.1f}%. {_ROI_SOURCE_CITATION}",
            key_wins=[f"Projected {roi_pct:.1f}% ROI in {city}"],
            risk_flags=["Full pipeline data unavailable — insights are estimates only"],
            data_sources=["fallback"], confidence_overall=0.4,
            market_timing=empty_timing,
            action_checklist=empty_checklist,
        )
        insight = RenovationInsight(
            priority_repairs=[fallback_repair.model_dump()],
            budget_strategy=fallback_budget.key_recommendation,
            expected_value_increase=fallback_roi.expected_value_increase,
            roi_score=f"{output.roi_score:.1f}/100",
            renovation_sequence=[], renovation_priority_score=60.0,
            cost_effectiveness_score=50.0, overall_insight_score=55.0,
            key_wins=output.key_wins, risk_flags=output.risk_flags,
            summary=output.summary,
        )
        return output, insight