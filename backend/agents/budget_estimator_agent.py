"""
ARKEN — BudgetEstimatorAgent (LangGraph node) v2.1
====================================================
Agent 4 in the LangGraph multi-agent pipeline.

v2.1 Changes (BUG 3 FIX):
  - _SANITY_BENCHMARKS is now a per-tier dict instead of flat scalars.
    Previously the benchmarks only checked mid-tier underestimation and
    basic-tier overestimation — premium had no bounds, so wildly wrong
    totals slipped through unchecked.
  - sanity_check() updated to use per-tier min/max benchmarks.
  - Benchmarks (CIDC India Construction Cost Index 2024):
      Basic:   ₹150–₹800/sqft   (cosmetic paint + standard tiles)
      Mid:     ₹500–₹1,800/sqft (sheen walls + large tiles + POP ceiling)
      Premium: ₹1,200–₹5,000/sqft (luxury stone + smart home + gypsum)

v2.0 Changes (preserved):
  1. sanity_check() method attached to budget_estimate["sanity_check"]
  2. GST breakdown per line item
  3. price_validity_date field
  4. alternatives section on each BOQ line item
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── City market data ──────────────────────────────────────────────────────────

CITY_MARKET_DATA = {
    "Mumbai":    {"rental_yield_pct": 2.5, "appreciation_5yr_pct": 42, "labour_premium_pct": 40, "market_tier": 1, "avg_psf_inr": 28000, "trend": "Supply constrained premium market. Renovation adds 18-22% rental premium."},
    "Bangalore": {"rental_yield_pct": 3.2, "appreciation_5yr_pct": 48, "labour_premium_pct": 25, "market_tier": 1, "avg_psf_inr": 13000, "trend": "Tech hub demand strong. Modern/Japandi styles command highest premiums."},
    "Hyderabad": {"rental_yield_pct": 3.5, "appreciation_5yr_pct": 55, "labour_premium_pct": 10, "market_tier": 1, "avg_psf_inr": 9500,  "trend": "Fastest appreciating metro 2023-25. Renovation ROI 3x vs non-renovated units."},
    "Delhi NCR": {"rental_yield_pct": 2.8, "appreciation_5yr_pct": 35, "labour_premium_pct": 30, "market_tier": 1, "avg_psf_inr": 16000, "trend": "Luxury segment outperforms. Premium kitchens/bathrooms highest ROI."},
    "Pune":      {"rental_yield_pct": 3.0, "appreciation_5yr_pct": 38, "labour_premium_pct": 18, "market_tier": 2, "avg_psf_inr": 10000, "trend": "IT/education hub. Mid-budget renovations see strongest rental demand."},
    "Chennai":   {"rental_yield_pct": 2.9, "appreciation_5yr_pct": 32, "labour_premium_pct": 15, "market_tier": 2, "avg_psf_inr": 9800,  "trend": "Steady market. Traditional + contemporary fusion most popular."},
    "Kolkata":   {"rental_yield_pct": 2.4, "appreciation_5yr_pct": 25, "labour_premium_pct": 12, "market_tier": 2, "avg_psf_inr": 7500,  "trend": "Stable market. Mid-tier renovations optimal for rental yield improvement."},
    "Ahmedabad": {"rental_yield_pct": 2.6, "appreciation_5yr_pct": 28, "labour_premium_pct": 8,  "market_tier": 2, "avg_psf_inr": 7000,  "trend": "Industrial/commercial growth driving residential demand."},
}

BUDGET_TIER_ANALYSIS = {
    "basic":   {"budget_range": "₹3–5L",  "what_it_covers": "Paint, basic flooring, lighting upgrades, minor carpentry",                       "roi_potential": "8–12%",  "best_for": "Rental yield improvement.",           "avoid": "Structural changes, modular kitchen at this budget",         "recommended_brands": ["Asian Paints Apcolite", "Kajaria GVT Basic", "Havells Efficiencia"]},
    "mid":     {"budget_range": "₹5–10L", "what_it_covers": "Full paint, premium flooring, false ceiling, modular carpentry, electrical upgrade", "roi_potential": "12–18%", "best_for": "Maximum ROI sweet spot.",             "avoid": "Over-specification in non-premium localities",               "recommended_brands": ["Asian Paints Royale Sheen", "Kajaria Endura", "Greenply Club Prime", "Legrand Myrius"]},
    "premium": {"budget_range": "₹10L+",  "what_it_covers": "Premium stone flooring, smart home integration, Italian-finish carpentry, lighting", "roi_potential": "14–22%", "best_for": "Tier-1 city luxury market.",          "avoid": "Over-capitalisation in Tier-2/3 markets",                   "recommended_brands": ["Dulux Velvet Touch", "Simpolo GVT Slab", "Greenply Marine", "Schneider AvatarOn"]},
}

LABOR_RATES: Dict[str, Dict[str, int]] = {
    "painting":     {"Mumbai": 18, "Delhi NCR": 16, "Bangalore": 15, "Hyderabad": 12, "Pune": 13, "Chennai": 12, "Kolkata": 10},
    "tiling":       {"Mumbai": 45, "Delhi NCR": 42, "Bangalore": 38, "Hyderabad": 30, "Pune": 32, "Chennai": 30, "Kolkata": 28},
    "false_ceiling":{"Mumbai": 95, "Delhi NCR": 90, "Bangalore": 80, "Hyderabad": 65, "Pune": 70, "Chennai": 65, "Kolkata": 55},
    "electrical":   {"Mumbai": 85, "Delhi NCR": 80, "Bangalore": 70, "Hyderabad": 60, "Pune": 62, "Chennai": 60, "Kolkata": 50},
    "plumbing":     {"Mumbai": 120,"Delhi NCR": 110,"Bangalore": 100,"Hyderabad": 80, "Pune": 85, "Chennai": 80, "Kolkata": 70},
    "carpentry":    {"Mumbai": 550,"Delhi NCR": 520,"Bangalore": 480,"Hyderabad": 380,"Pune": 400,"Chennai": 380,"Kolkata": 350},
    "waterproofing":{"Mumbai": 65, "Delhi NCR": 60, "Bangalore": 55, "Hyderabad": 45, "Pune": 48, "Chennai": 45, "Kolkata": 40},
}

_LABOR_FALLBACK_CITY = "Hyderabad"

_ROOM_WORK_TYPES: Dict[str, List[str]] = {
    "kitchen":     ["painting", "tiling", "electrical", "plumbing", "carpentry"],
    "bathroom":    ["tiling", "plumbing", "waterproofing", "electrical"],
    "bedroom":     ["painting", "electrical", "carpentry"],
    "living_room": ["painting", "tiling", "electrical", "false_ceiling"],
    "study":       ["painting", "electrical", "carpentry"],
    "full_home":   ["painting", "tiling", "electrical", "plumbing", "carpentry",
                    "false_ceiling", "waterproofing"],
}

# ── BUG 3 FIX: Per-tier sanity check benchmarks ───────────────────────────────
# Source: CIDC India Construction Cost Index 2024, NIC India 2024,
#         HomeLane / Livspace / NoBroker renovation pricing data Q1 2026.
#
# Basic  (Rs.3-5L):  ₹150–800/sqft  — cosmetic work: paint, 600×600 tiles, basic electrical
# Mid    (Rs.5-10L): ₹500–1,800/sqft — sheen paint, 800×800, POP ceiling, plywood carpentry
# Premium (Rs.10L+): ₹1,200–5,000/sqft — stone/slab tiles, gypsum, smart home, luxury fittings
_SANITY_BENCHMARKS: Dict[str, Dict[str, int]] = {
    "basic":   {"min_cost_per_sqft": 150,  "max_cost_per_sqft": 800},
    "mid":     {"min_cost_per_sqft": 500,  "max_cost_per_sqft": 1800},
    "premium": {"min_cost_per_sqft": 1200, "max_cost_per_sqft": 5000},
}

# ── Material alternatives database ────────────────────────────────────────────
_MATERIAL_ALTERNATIVES: Dict[str, Tuple[str, str, str, str]] = {
    "paint":     ("Berger Easy Clean",    "₹280/litre",  "Asian Paints Royale Aspira",       "₹380/litre"),
    "primer":    ("Asian Paints Primer",  "₹110/litre",  "Dulux Weathershield Primer",       "₹160/litre"),
    "putty":     ("Birla White Putty",    "₹18/kg",      "JK Wall Putty Premium",            "₹26/kg"),
    "tile":      ("Somany Vitro Basic",   "₹55/sqft",    "Simpolo GVT Slab",                 "₹140/sqft"),
    "flooring":  ("Kajaria GVT 600×600",  "₹65/sqft",    "Simpolo Nano GVT",                 "₹150/sqft"),
    "granite":   ("Kotda Black Granite",  "₹85/sqft",    "Black Galaxy Granite Premium",     "₹220/sqft"),
    "plywood":   ("Century Ply Club",     "₹85/sqft",    "Greenply Marine Ply",              "₹160/sqft"),
    "electrical":("Havells Crabtree",     "₹45/point",   "Schneider AvatarOn",               "₹120/point"),
    "switch":    ("Havells Crabtree",     "₹45/point",   "Legrand Myrius",                   "₹95/point"),
    "fan":       ("Havells Efficiencia",  "₹2,200/unit", "Orient Electric Aeroquiet",        "₹4,500/unit"),
    "sanitary":  ("Hindware Basic",       "₹9,000/set",  "Kohler Serif",                     "₹28,000/set"),
    "faucet":    ("Jaquar Solo",          "₹2,500/unit", "Grohe BauEdge",                    "₹6,800/unit"),
    "wardrobe":  ("Durian Basic Laminates","₹900/sqft",  "Hafele Premium Acrylic Finish",    "₹2,200/sqft"),
    "kitchen":   ("Sleek Basic Laminate", "₹1,000/sqft", "Godrej Interio Premium Acrylic",  "₹2,800/sqft"),
    "ceiling":   ("Armstrong Basic Grid", "₹55/sqft",    "Saint-Gobain Gyproc Aerocon",      "₹130/sqft"),
    "waterproof":("Dr. Fixit Pidifin",    "₹220/litre",  "Fosroc Sika Waterproofing System", "₹480/litre"),
    "window":    ("RK Windows UPVC",      "₹650/sqft",   "Fenesta Premium UPVC Triple-Seal", "₹1,100/sqft"),
    "door":      ("Merino Laminates Flush","₹8,000/unit","Greenply Marine Solid Wood",       "₹22,000/unit"),
}

GST_RATE = 0.18   # 18% GST on materials (CGST + SGST)
PRICE_VALIDITY_NOTE = (
    "Prices valid as of Q1 2026. "
    "Material costs typically increase 5–8% annually in India "
    "(Source: CIDC India Construction Cost Index 2024)."
)


class BudgetEstimatorAgent:
    """
    Computes detailed renovation budget breakdown.

    v2.1: Per-tier sanity benchmarks — basic/mid/premium each have
    their own realistic min/max ₹/sqft bounds.
    """

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0   = time.perf_counter()
        name = "budget_estimator_agent"
        logger.info(f"[{state.get('project_id', '')}] BudgetEstimatorAgent v2.1 starting")

        try:
            updates = self._estimate(state)
        except Exception as e:
            logger.error(f"[budget_estimator_agent] Error: {e}", exc_info=True)
            budget_inr = state.get("budget_inr", 750_000)
            city       = state.get("city", "Hyderabad")
            updates = {
                "budget_estimate":  self._fallback_estimate(budget_inr, city),
                "location_context": CITY_MARKET_DATA.get(city, CITY_MARKET_DATA["Hyderabad"]),
                "budget_analysis":  BUDGET_TIER_ANALYSIS.get(state.get("budget_tier", "mid"), {}),
                "material_prices":  [],
                "cost_breakdown":   {"total_inr": budget_inr},
                "errors": (state.get("errors") or []) + [f"budget_estimator_agent: {e}"],
            }

        elapsed = round(time.perf_counter() - t0, 3)
        timings = dict(state.get("agent_timings") or {})
        timings[name] = elapsed
        updates["agent_timings"] = timings

        completed = list(state.get("completed_agents") or [])
        if name not in completed:
            completed.append(name)
        updates["completed_agents"] = completed

        logger.info(
            f"[budget_estimator_agent] done in {elapsed}s — "
            f"total=₹{updates.get('budget_estimate', {}).get('total_cost_inr', 0):,}"
        )
        return updates

    # ── BUG 3 FIX: per-tier sanity_check ─────────────────────────────────────

    def sanity_check(
        self,
        total_cost_inr: int,
        floor_area_sqft: float,
        budget_tier: str,
    ) -> Dict:
        """
        Validate BOQ total against per-tier known cost-per-sqft benchmarks.

        Benchmarks (CIDC India Construction Cost Index 2024):
          Basic:   ₹150–800/sqft
          Mid:     ₹500–1,800/sqft
          Premium: ₹1,200–5,000/sqft

        Returns:
          {
            "status": "ok" | "possibly_underestimated" | "possibly_overestimated",
            "cost_per_sqft": float,
            "warning": str | None,
            "recommendation": str | None,
            "benchmark_source": str,
          }
        """
        cost_per_sqft = total_cost_inr / max(floor_area_sqft, 1)
        tier          = budget_tier.lower() if budget_tier else "mid"
        benchmarks    = _SANITY_BENCHMARKS.get(tier, _SANITY_BENCHMARKS["mid"])
        min_bench     = benchmarks["min_cost_per_sqft"]
        max_bench     = benchmarks["max_cost_per_sqft"]

        status        = "ok"
        warning: Optional[str] = None
        recommendation: Optional[str] = None

        if cost_per_sqft < min_bench:
            status = "possibly_underestimated"
            warning = (
                f"⚠ BOQ total of ₹{cost_per_sqft:.0f}/sqft is below the minimum benchmark of "
                f"₹{min_bench}/sqft for {tier}-tier renovation (CIDC India 2024). "
                f"The estimate may be missing labour, GST, or finishing items."
            )
            recommendation = (
                f"Review the BOQ for missing line items: supervision, contingency, "
                f"GST@18%, and snagging. Typical {tier}-tier projects run "
                f"₹{min_bench}–₹{max_bench}/sqft all-in."
            )
            logger.warning(f"[BudgetEstimator] Sanity check: {warning}")

        elif cost_per_sqft > max_bench:
            status = "possibly_overestimated"
            warning = (
                f"⚠ BOQ total of ₹{cost_per_sqft:.0f}/sqft exceeds the maximum benchmark of "
                f"₹{max_bench}/sqft for {tier}-tier renovation (CIDC India 2024). "
                f"Either the scope has items beyond this tier, or the estimate is inflated."
            )
            recommendation = (
                f"Review line items for materials inconsistent with {tier} tier. "
                f"Consider downgrading brand selections to match the intended budget."
            )
            logger.warning(f"[BudgetEstimator] Sanity check: {warning}")

        return {
            "status":          status,
            "cost_per_sqft":   round(cost_per_sqft, 1),
            "min_benchmark":   min_bench,
            "max_benchmark":   max_bench,
            "warning":         warning,
            "recommendation":  recommendation,
            "benchmark_source": "CIDC India Construction Cost Index 2024",
        }

    # ── GST breakdown per line item ───────────────────────────────────────────

    @staticmethod
    def add_gst_to_line_items(boq_line_items: List[Dict]) -> List[Dict]:
        """
        Add GST breakdown to each BOQ line item in-place.
        Adds: base_price_inr, gst_18_pct_inr, total_with_gst
        """
        enriched: List[Dict] = []
        for item in boq_line_items:
            item = dict(item)
            unit_price = (
                item.get("unit_price_inr") or
                item.get("amount_inr") or
                item.get("total_inr") or
                item.get("cost_inr") or 0
            )
            try:
                base_price = float(unit_price)
            except (TypeError, ValueError):
                base_price = 0.0

            gst_amount             = round(base_price * GST_RATE)
            item["base_price_inr"] = int(base_price)
            item["gst_18_pct_inr"] = gst_amount
            item["total_with_gst"] = int(base_price) + gst_amount
            enriched.append(item)
        return enriched

    # ── Alternatives lookup ───────────────────────────────────────────────────

    @staticmethod
    def _find_alternatives(item: Dict) -> Dict:
        desc = (
            str(item.get("description", "")) + " " +
            str(item.get("category", "")) + " " +
            str(item.get("name", ""))
        ).lower()

        for keyword, (b_brand, b_price, p_brand, p_price) in _MATERIAL_ALTERNATIVES.items():
            if keyword in desc:
                return {
                    "budget_alternative":  {"brand": b_brand, "price": b_price},
                    "premium_alternative": {"brand": p_brand, "price": p_price},
                }
        return {}

    # ── Enrich BOQ (GST + alternatives) ──────────────────────────────────────

    def enrich_boq(self, boq_line_items: List[Dict]) -> List[Dict]:
        with_gst = self.add_gst_to_line_items(boq_line_items)
        return [
            {**item, "alternatives": self._find_alternatives(item)}
            for item in with_gst
        ]

    # ── Labour computation (unchanged from v2.0) ──────────────────────────────

    def _compute_real_labour(
        self,
        city: str,
        room_type: str,
        floor_area: float,
        boq_line_items: List[Dict],
    ) -> Tuple[int, Dict[str, int]]:
        work_types = _ROOM_WORK_TYPES.get(room_type, _ROOM_WORK_TYPES["bedroom"])

        if boq_line_items:
            boq_categories = {item.get("category", "").lower() for item in boq_line_items}
            inferred: List[str] = []
            if any(c in boq_categories for c in ("paint", "primer", "putty", "wall preparation")):
                inferred.append("painting")
            if any(c in boq_categories for c in ("flooring tiles", "tile", "tiling")):
                inferred.append("tiling")
            if any(c in boq_categories for c in ("fan", "switches", "lighting", "electrical")):
                inferred.append("electrical")
            if any(c in boq_categories for c in ("wc", "basin", "shower", "faucet", "plumbing")):
                inferred.append("plumbing")
            if any(c in boq_categories for c in ("wardrobe", "carpentry", "plywood", "chimney", "sink")):
                inferred.append("carpentry")
            if any(c in boq_categories for c in ("false ceiling",)):
                inferred.append("false_ceiling")
            if any(c in boq_categories for c in ("waterproofing",)):
                inferred.append("waterproofing")
            if inferred:
                work_types = list(set(work_types) | set(inferred))

        wall_area  = floor_area * 2.5
        carpet_rft = (floor_area ** 0.5) * 4 * 2

        def _rate(wtype: str) -> int:
            city_rates = LABOR_RATES.get(wtype, {})
            return city_rates.get(city, city_rates.get(_LABOR_FALLBACK_CITY, 0))

        breakdown: Dict[str, int] = {}
        for wt in work_types:
            rate = _rate(wt)
            if rate <= 0:
                continue
            if wt == "painting":
                breakdown["painting"]      = int(rate * (wall_area + floor_area))
            elif wt == "tiling":
                breakdown["tiling"]        = int(rate * floor_area)
            elif wt == "false_ceiling":
                breakdown["false_ceiling"] = int(rate * floor_area)
            elif wt == "electrical":
                breakdown["electrical"]    = int(rate * floor_area)
            elif wt == "plumbing":
                breakdown["plumbing"]      = int(rate * floor_area)
            elif wt == "carpentry":
                breakdown["carpentry"]     = int(rate * carpet_rft)
            elif wt == "waterproofing":
                breakdown["waterproofing"] = int(rate * floor_area * 1.5)

        total = sum(breakdown.values())
        return total, breakdown

    # ── Core estimate ──────────────────────────────────────────────────────────

    def _estimate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        budget_inr   = state.get("budget_inr", 750_000)
        budget_tier  = state.get("budget_tier", "mid")
        city         = state.get("city", "Hyderabad")
        room_type    = state.get("room_type", "bedroom")
        floor_area   = float(state.get("floor_area_sqft") or 120.0)

        city_data    = CITY_MARKET_DATA.get(city, CITY_MARKET_DATA["Hyderabad"])
        budget_data  = BUDGET_TIER_ANALYSIS.get(budget_tier, BUDGET_TIER_ANALYSIS["mid"])

        design_plan     = state.get("design_plan") or {}
        boq_line_items: List[Dict] = []
        if isinstance(design_plan, dict):
            boq_line_items = (
                design_plan.get("line_items") or
                design_plan.get("boq_line_items") or []
            )

        if isinstance(design_plan, dict) and design_plan.get("total_inr"):
            total_est = design_plan["total_inr"]
            # material_inr from design_plan already excludes labour (v3 computes correctly)
            mat       = design_plan.get("material_inr", int(total_est * 0.55))
            cont      = design_plan.get("contingency_inr", int(total_est * 0.05))
        elif boq_line_items:
            # Derive from line items directly
            mat = sum(
                i.get("total_inr", 0) for i in boq_line_items
                if not str(i.get("category", "")).startswith("Labour")
            )
            total_est = mat + lab
            cont      = int(total_est * 0.10)
        else:
            total_est = state.get("total_cost_estimate") or budget_inr
            mat       = int(total_est * 0.55)
            cont      = int(total_est * 0.05)

        # Labour — sum the "Labour -" category items already in the BOQ line items.
        # design_planner v3 puts every trade (painting, tiling, electrical, etc.)
        # as visible line items so we never need to recompute from scratch.
        lab = sum(
            i.get("total_inr", 0) for i in boq_line_items
            if str(i.get("category", "")).startswith("Labour")
        )
        labour_breakdown = {}
        for i in boq_line_items:
            cat = str(i.get("category", ""))
            if cat.startswith("Labour"):
                trade = cat.replace("Labour - ", "").lower().replace(" ", "_")
                labour_breakdown[trade] = labour_breakdown.get(trade, 0) + i.get("total_inr", 0)
        # Fallback if design_planner didn't produce any labour items
        if lab <= 0:
            lab = int(total_est * 0.30)
            labour_breakdown = {"fallback_flat_30pct": lab}

        # If design_plan already has a correct total (v3 computes grand_total including
        # GST and contingency), use it directly to avoid double-counting.
        if isinstance(design_plan, dict) and design_plan.get("total_inr"):
            total            = design_plan["total_inr"]
            gst_on_materials = design_plan.get("gst_inr", int(mat * 0.18))
            gst_on_labour    = 0  # already included in design_plan gst_inr
            gst              = gst_on_materials
            cont             = design_plan.get("contingency_inr", cont)
            subtotal         = mat + lab
        else:
            gst_on_materials = int(mat * 0.18)
            gst_on_labour    = int(lab * 0.12)
            gst              = gst_on_materials + gst_on_labour
            subtotal         = mat + lab
            total            = subtotal + gst + cont
        supervision      = int(total * 0.03)
        within_budget    = total <= budget_inr * 1.10
        cost_per_sqft    = round(total / max(floor_area, 1))

        # Enrich BOQ (GST + alternatives)
        enriched_boq = self.enrich_boq(boq_line_items) if boq_line_items else []

        # BUG 3 FIX: per-tier sanity check
        sanity = self.sanity_check(total, floor_area, budget_tier)

        # Material price forecasts
        material_prices: List[Dict] = []
        try:
            from agents.price_forecast import PriceForecastAgent
            material_prices = PriceForecastAgent().forecast_all(horizon_days=90)
        except Exception as fe:
            logger.warning(f"[budget_estimator] Price forecast failed: {fe}")

        cost_breakdown = {
            "materials_inr":       mat,
            "labour_inr":          lab,
            "labour_breakdown":    labour_breakdown,
            "gst_on_materials_inr": gst_on_materials,
            "gst_on_labour_inr":   gst_on_labour,
            "gst_inr":             gst,
            "contingency_inr":     cont,
            "supervision_inr":     supervision,
            "misc_contingency_inr": cont,
            "total_inr":           total,
            "city_multiplier":     1.0 + city_data.get("labour_premium_pct", 10) / 100,
        }

        budget_estimate = {
            "total_cost_inr":       total,
            "materials_inr":        mat,
            "labour_inr":           lab,
            "labour_breakdown":     labour_breakdown,
            "supervision_inr":      supervision,
            "contingency_inr":      cont,
            "gst_on_materials_inr": gst_on_materials,
            "gst_on_labour_inr":    gst_on_labour,
            "gst_inr":              gst,
            "location_context":     city_data,
            "budget_analysis":      budget_data,
            "material_prices":      material_prices,
            "cost_per_sqft":        cost_per_sqft,
            "within_budget":        within_budget,
            "enriched_boq":        enriched_boq,
            "sanity_check":        sanity,
            "price_validity_date": PRICE_VALIDITY_NOTE,
        }

        return {
            "budget_estimate":  budget_estimate,
            "location_context": city_data,
            "budget_analysis":  budget_data,
            "material_prices":  material_prices,
            "cost_breakdown":   cost_breakdown,
        }

    def _fallback_estimate(self, budget_inr: int, city: str) -> Dict[str, Any]:
        city_data = CITY_MARKET_DATA.get(city, CITY_MARKET_DATA["Hyderabad"])
        return {
            "total_cost_inr":  budget_inr,
            "materials_inr":   int(budget_inr * 0.55),
            "labour_inr":      int(budget_inr * 0.30),
            "supervision_inr": int(budget_inr * 0.03),
            "contingency_inr": int(budget_inr * 0.07),
            "gst_inr":         int(budget_inr * 0.05),
            "location_context": city_data,
            "budget_analysis": {},
            "material_prices": [],
            "cost_per_sqft":   0,
            "within_budget":   True,
            "price_validity_date": PRICE_VALIDITY_NOTE,
            "sanity_check":    {"status": "unknown", "warning": None},
        }
