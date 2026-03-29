"""
ARKEN — Multi-Agent Pipeline v4.0
===================================
Fix v4.0:
  - REMOVED broken LangGraph parallel graph.
    Root cause: LangGraph raises "Can receive only one value per step"
    when parallel branches all write to shared keys (project_id, errors,
    agent_timings, completed_agents). This caused a crash + fallback that
    ran image_analysis TWICE (20s wasted) with the old code.
  - Now uses clean sequential execution only. Same 5 agents, same state.
  - Added design_plan + schedule to PipelineState TypedDict.
  - agent_planning now calls ProjectCoordinatorAgent for CPM schedule.
  - agent_image_analysis uses robust JSON extraction (regex + trailing comma fix).
"""
from __future__ import annotations

import base64
import json
import logging
import re as _re
import time
from typing import Any, Callable, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared pipeline state
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    # Inputs
    project_id: str
    original_image_b64: str
    original_image_mime: str
    renovated_image_b64: str
    renovated_image_mime: str
    theme: str
    city: str
    budget_tier: str
    budget_inr: int
    room_type: str

    # Agent 1 — Image Analysis
    image_features: Dict
    detected_changes: List[str]
    room_dimensions: Dict
    visual_style: List[str]

    # Agent 2 — Planning (BOQ + CPM Schedule)
    material_plan: Dict
    boq_line_items: List[Dict]
    labour_estimate: int
    total_cost_estimate: int
    design_plan: Dict        # full plan dict: total_inr, material_inr, labour_inr, gst_inr, contingency_inr, line_items
    schedule: Dict           # CPM schedule: total_days, critical_path_days, tasks[], risks[]

    # Agent 3 — ROI
    roi_prediction: Dict
    payback_months: int
    equity_gain_inr: int

    # Agent 4 — Budget & Location
    location_context: Dict
    budget_analysis: Dict
    material_prices: List[Dict]

    # Agent 5 — Insights
    insights: Dict
    chat_context: str

    # Metadata
    errors: List[str]
    agent_timings: Dict[str, float]
    completed_agents: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1: Image Analysis
# ─────────────────────────────────────────────────────────────────────────────


def _safe_parse_json(raw: str) -> dict:
    """
    Robustly parse a JSON string that may be:
      - wrapped in markdown fences
      - truncated mid-string or mid-key (unterminated string)
      - containing trailing commas

    Handles the exact patterns observed in production logs:
      "Unterminated string starting at: line 1 column 365 (char 364)"
      "Unterminated string starting at: line 1 column 296 (char 295)"
    """
    import json as _json, re as _r

    # 1. Strip markdown fences
    cleaned = _r.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # 2. Extract the outermost {...} block
    m = _r.search(r"\{[\s\S]*\}", cleaned)
    if m:
        cleaned = m.group(0)

    # 3. Fix trailing commas before } or ]
    cleaned = _r.sub(r",\s*([}\]])", r"\1", cleaned)

    # 4. Try direct parse first (fast path)
    try:
        return _json.loads(cleaned)
    except _json.JSONDecodeError:
        pass

    # 5. Truncation repair: handles mid-string and mid-key cuts
    repaired = _repair_truncated_json(cleaned)
    try:
        return _json.loads(repaired)
    except _json.JSONDecodeError:
        pass

    # 6. Last resort: regex key-value extraction so pipeline never crashes
    return _extract_kv_fallback(raw)


def _repair_truncated_json(s: str) -> str:
    """
    Repair a JSON string truncated by a token limit.

    Strategy:
      1. Walk the string tracking in_string state and last safe boundary.
      2. If truncated mid-string (or mid-key), cut back to the last complete
         key:value pair boundary (after a comma or closing bracket).
      3. Close any still-open arrays and objects.

    This correctly handles both:
      - Truncation mid-value: {...,"room_condition":"go   <- cut here
      - Truncation mid-key:   {...,"room_co              <- cut here
    """
    import re as _r

    in_string = False
    escape_next = False
    last_safe_pos = 0   # index after last , or } or ] while not in a string

    i = 0
    while i < len(s):
        ch = s[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if ch == "\\":
            escape_next = True
            i += 1
            continue
        if ch == '"' and not in_string:
            in_string = True
        elif ch == '"' and in_string:
            in_string = False
        elif not in_string:
            if ch in ('}', ']', ','):
                last_safe_pos = i + 1
        i += 1

    # If we ended inside a string, roll back to the last safe boundary
    if in_string:
        s = s[:last_safe_pos].rstrip().rstrip(',')

    # Re-count open depths on the trimmed string
    in_s2 = False
    esc2 = False
    d_obj = 0
    d_arr = 0
    for ch in s:
        if esc2:
            esc2 = False
            continue
        if ch == "\\":
            esc2 = True
            continue
        if ch == '"' and not in_s2:
            in_s2 = True
        elif ch == '"' and in_s2:
            in_s2 = False
        elif not in_s2:
            if ch == '{':
                d_obj += 1
            elif ch == '}':
                d_obj = max(0, d_obj - 1)
            elif ch == '[':
                d_arr += 1
            elif ch == ']':
                d_arr = max(0, d_arr - 1)

    suffix = "]" * d_arr + "}" * d_obj
    repaired = s + suffix
    repaired = _r.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _extract_kv_fallback(raw: str) -> dict:
    """
    Last-resort extraction: pull out key:value pairs that look valid
    using regex so the pipeline never hard-crashes.
    """
    import re as _r
    result: dict = {}
    # string values
    for m in _r.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', raw):
        result[m.group(1)] = m.group(2)
    # numeric values
    for m in _r.finditer(r'"(\w+)"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw):
        key, val = m.group(1), m.group(2)
        if key not in result:
            result[key] = float(val) if '.' in val else int(val)
    # array values  (simplified — captures first array per key)
    for m in _r.finditer(r'"(\w+)"\s*:\s*\[([^\]]*)\]', raw):
        key = m.group(1)
        if key not in result:
            items = [i.strip().strip('"') for i in m.group(2).split(',') if i.strip()]
            result[key] = items
    return result


def agent_image_analysis(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    agent_name = "image_analysis"
    logger.info(f"[{state.get('project_id','')}] Agent: {agent_name}")

    try:
        from config import settings
        from google import genai
        from google.genai import types

        img_b64  = state.get("renovated_image_b64") or state.get("original_image_b64", "")
        img_mime = state.get("renovated_image_mime") or state.get("original_image_mime", "image/jpeg")
        theme       = state.get("theme", "Modern Minimalist")
        city        = state.get("city", "Hyderabad")
        budget_tier = state.get("budget_tier", "mid")
        room_type   = state.get("room_type", "bedroom")

        if not img_b64 or not settings.GOOGLE_API_KEY:
            raise ValueError("No image or API key available")

        client = genai.Client(api_key=settings.GOOGLE_API_KEY.get_secret_value())

        # Compact prompt prevents JSON truncation
        analysis_prompt = (
            f"Analyse this {room_type} image for a {theme} renovation in {city} at {budget_tier} budget.\n"
            "Respond with ONLY a valid JSON object, no markdown, no explanation:\n"
            '{"wall_treatment":"<desc>","floor_material":"<desc>","ceiling_treatment":"<desc>",'
            '"furniture_items":["item1","item2"],"lighting_type":"<desc>",'
            '"colour_palette":["c1","c2","c3"],'
            f'"detected_style":"{theme}","quality_tier":"{budget_tier}",'
            '"specific_changes":["Change 1","Change 2","Change 3","Change 4","Change 5"],'
            '"estimated_wall_area_sqft":200,"estimated_floor_area_sqft":120,"room_condition":"good"}'
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[
                types.Part(inline_data=types.Blob(mime_type=img_mime, data=base64.b64decode(img_b64))),
                types.Part(text=analysis_prompt),
            ])],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048),
        )

        raw = response.text.strip()
        features = _safe_parse_json(raw)

        room_dimensions = {
            "wall_area_sqft":      features.get("estimated_wall_area_sqft", 200),
            "floor_area_sqft":     features.get("estimated_floor_area_sqft", 120),
            "estimated_length_ft": 14.0,
            "estimated_width_ft":  12.0,
            "estimated_height_ft": 9.0,
        }
        updates: PipelineState = {
            "image_features":  features,
            "detected_changes": features.get("specific_changes", []),
            "room_dimensions": room_dimensions,
            "visual_style":    [features.get("detected_style", theme)],
        }

    except Exception as e:
        logger.error(f"[image_analysis] Error: {e}")
        updates: PipelineState = {
            "image_features": {
                "wall_treatment": f"{state.get('theme','Modern')} style walls",
                "floor_material": "vitrified tiles",
                "ceiling_treatment": "POP false ceiling with LED lighting",
                "colour_palette": ["white", "grey", "accent"],
                "detected_style": state.get("theme", "Modern Minimalist"),
                "quality_tier": state.get("budget_tier", "mid"),
                "specific_changes": [
                    "Applied new wall paint/texture treatment",
                    "Installed new flooring material",
                    "Upgraded ceiling with POP/false ceiling",
                    "Updated lighting fixtures",
                    "Refreshed overall aesthetic",
                ],
                "estimated_wall_area_sqft": 200,
                "estimated_floor_area_sqft": 120,
            },
            "detected_changes": ["Wall treatment applied", "Flooring updated", "Ceiling work done"],
            "room_dimensions": {
                "wall_area_sqft": 200, "floor_area_sqft": 120,
                "estimated_length_ft": 14.0, "estimated_width_ft": 12.0, "estimated_height_ft": 9.0,
            },
            "visual_style": [state.get("theme", "Modern Minimalist")],
            "errors": (state.get("errors") or []) + [f"image_analysis: {e}"],
        }

    _record_timing(updates, agent_name, t0, state)
    _mark_completed(updates, agent_name, state)
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2: Planning — BOQ + CPM Schedule
# ─────────────────────────────────────────────────────────────────────────────

def agent_planning(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    agent_name = "planning"
    logger.info(f"[{state.get('project_id','')}] Agent: {agent_name}")

    try:
        from agents.design_planner import DesignPlannerAgent

        dims        = state.get("room_dimensions", {})
        budget_tier = state.get("budget_tier", "mid")
        budget_inr  = state.get("budget_inr", 750000)
        theme       = state.get("theme", "Modern Minimalist")
        city        = state.get("city", "Hyderabad")
        room_type   = state.get("room_type", "bedroom")
        area_sqft   = float(dims.get("floor_area_sqft", 120))

        quantities = {
            "paint_liters":     dims.get("wall_area_sqft", 200) * 0.037 * 2,
            "floor_tiles_sqft": area_sqft * 1.1,
            "plywood_sqft":     area_sqft * 0.3,
        }

        planner = DesignPlannerAgent()
        plan = planner.plan(
            theme=theme,
            budget_inr=budget_inr,
            budget_tier=budget_tier,
            area_sqft=area_sqft,
            room_type=room_type,
            city=city,
            quantities=quantities,
        )

        # CPM Schedule via ProjectCoordinatorAgent
        schedule = {}
        try:
            from agents.coordinator import ProjectCoordinatorAgent
            from datetime import date, timedelta
            coordinator = ProjectCoordinatorAgent()
            schedule = coordinator.generate_schedule(
                area_sqft=area_sqft,
                budget_inr=budget_inr,
                room_type=room_type,
                city=city,
                start_date=date.today() + timedelta(days=7),
            )
        except Exception as se:
            logger.warning(f"[planning] Schedule failed: {se}")
            schedule = {}

        updates: PipelineState = {
            "material_plan":       plan.get("recommendations", {}),
            "boq_line_items":      plan.get("line_items", []),
            "labour_estimate":     plan.get("labour_inr", 0),
            "total_cost_estimate": plan.get("total_inr", budget_inr),
            "design_plan":         plan,      # full plan with total_inr, gst_inr etc.
            "schedule":            schedule,  # CPM tasks + risks
        }

    except Exception as e:
        logger.error(f"[planning] Error: {e}")
        updates: PipelineState = {
            "material_plan": {}, "boq_line_items": [],
            "labour_estimate": 0, "total_cost_estimate": state.get("budget_inr", 750000),
            "design_plan": {}, "schedule": {},
            "errors": (state.get("errors") or []) + [f"planning: {e}"],
        }

    _record_timing(updates, agent_name, t0, state)
    _mark_completed(updates, agent_name, state)
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3: ROI Prediction
# ─────────────────────────────────────────────────────────────────────────────

def agent_roi_prediction(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    agent_name = "roi_prediction"
    logger.info(f"[{state.get('project_id','')}] Agent: {agent_name}")

    try:
        from agents.roi_forecast import ROIForecastAgent

        dims     = state.get("room_dimensions", {})
        features = state.get("image_features", {})
        cmap     = {"new": "good", "good": "good", "fair": "average", "poor": "poor"}
        condition = cmap.get(features.get("room_condition", "fair"), "average")

        agent = ROIForecastAgent()
        roi = agent.predict(
            renovation_cost_inr=state.get("budget_inr", 750000),
            area_sqft=dims.get("floor_area_sqft", 120),
            city=state.get("city", "Hyderabad"),
            room_type=state.get("room_type", "bedroom"),
            budget_tier=state.get("budget_tier", "mid"),
            existing_condition=condition,
        )
        updates: PipelineState = {
            "roi_prediction":  roi,
            "payback_months":  roi.get("payback_months", 36),
            "equity_gain_inr": roi.get("equity_gain_inr", 0),
        }

    except Exception as e:
        logger.error(f"[roi_prediction] Error: {e}")
        updates: PipelineState = {
            "roi_prediction": {"roi_pct": 12.0, "model_type": "fallback"},
            "payback_months": 36, "equity_gain_inr": 0,
            "errors": (state.get("errors") or []) + [f"roi_prediction: {e}"],
        }

    _record_timing(updates, agent_name, t0, state)
    _mark_completed(updates, agent_name, state)
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4: Budget & Location
# ─────────────────────────────────────────────────────────────────────────────

CITY_MARKET_DATA = {
    "Mumbai":    {"rental_yield_pct": 2.5, "appreciation_5yr_pct": 42, "labour_premium_pct": 40, "market_tier": 1, "avg_psf_inr": 28000, "trend": "Supply constrained premium market. Renovation adds 18-22% rental premium."},
    "Bangalore": {"rental_yield_pct": 3.2, "appreciation_5yr_pct": 48, "labour_premium_pct": 25, "market_tier": 1, "avg_psf_inr": 13000, "trend": "Tech hub demand strong. Modern/Japandi styles command highest premiums."},
    "Hyderabad": {"rental_yield_pct": 3.5, "appreciation_5yr_pct": 55, "labour_premium_pct": 10, "market_tier": 1, "avg_psf_inr": 9500,  "trend": "Fastest appreciating metro 2023-25. Renovation ROI 3x vs non-renovated units."},
    "Delhi NCR": {"rental_yield_pct": 2.8, "appreciation_5yr_pct": 35, "labour_premium_pct": 30, "market_tier": 1, "avg_psf_inr": 16000, "trend": "Luxury segment outperforms. Premium kitchens/bathrooms highest ROI."},
    "Pune":      {"rental_yield_pct": 3.0, "appreciation_5yr_pct": 38, "labour_premium_pct": 18, "market_tier": 2, "avg_psf_inr": 10000, "trend": "IT/education hub. Mid-budget renovations see strongest rental demand."},
    "Chennai":   {"rental_yield_pct": 2.9, "appreciation_5yr_pct": 32, "labour_premium_pct": 15, "market_tier": 2, "avg_psf_inr": 9800,  "trend": "Steady market. Traditional + contemporary fusion most popular."},
}

BUDGET_TIER_ANALYSIS = {
    "basic":   {"budget_range": "₹3–5L",  "what_it_covers": "Paint, basic flooring, lighting upgrades, minor carpentry",                       "roi_potential": "8–12%",  "best_for": "Rental yield improvement.",           "avoid": "Structural changes, modular kitchen at this budget",         "recommended_brands": ["Asian Paints Apcolite", "Kajaria GVT Basic", "Havells Efficiencia"]},
    "mid":     {"budget_range": "₹5–10L", "what_it_covers": "Full paint, premium flooring, false ceiling, modular carpentry, electrical upgrade", "roi_potential": "12–18%", "best_for": "Maximum ROI sweet spot.",             "avoid": "Over-specification in non-premium localities",               "recommended_brands": ["Asian Paints Royale Sheen", "Kajaria Endura", "Greenply Club Prime", "Legrand Myrius"]},
    "premium": {"budget_range": "₹10L+",  "what_it_covers": "Premium stone flooring, smart home integration, Italian-finish carpentry, lighting", "roi_potential": "14–22%", "best_for": "Tier-1 city luxury market.",          "avoid": "Over-capitalisation in Tier-2/3 markets",                   "recommended_brands": ["Dulux Velvet Touch", "Simpolo GVT Slab", "Greenply Marine", "Schneider AvatarOn"]},
}


def agent_budget_location(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    agent_name = "budget_location"
    logger.info(f"[{state.get('project_id','')}] Agent: {agent_name}")

    try:
        from agents.price_forecast import PriceForecastAgent

        city        = state.get("city", "Hyderabad")
        budget_tier = state.get("budget_tier", "mid")
        city_data   = CITY_MARKET_DATA.get(city, CITY_MARKET_DATA["Hyderabad"])
        budget_data = BUDGET_TIER_ANALYSIS.get(budget_tier, BUDGET_TIER_ANALYSIS["mid"])

        try:
            forecasts = PriceForecastAgent().forecast_all(horizon_days=90)
        except Exception as fe:
            logger.warning(f"Price forecast failed: {fe}")
            forecasts = []

        updates: PipelineState = {
            "location_context": city_data,
            "budget_analysis":  budget_data,
            "material_prices":  forecasts,
        }

    except Exception as e:
        logger.error(f"[budget_location] Error: {e}")
        updates: PipelineState = {
            "location_context": CITY_MARKET_DATA.get(state.get("city", "Hyderabad"), {}),
            "budget_analysis":  BUDGET_TIER_ANALYSIS.get(state.get("budget_tier", "mid"), {}),
            "material_prices":  [],
            "errors": (state.get("errors") or []) + [f"budget_location: {e}"],
        }

    _record_timing(updates, agent_name, t0, state)
    _mark_completed(updates, agent_name, state)
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5: Insight Generation
# ─────────────────────────────────────────────────────────────────────────────

def agent_insight_generation(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()
    agent_name = "insight_generation"
    logger.info(f"[{state.get('project_id','')}] Agent: {agent_name}")

    try:
        features        = state.get("image_features", {})
        roi             = state.get("roi_prediction", {})
        loc             = state.get("location_context", {})
        budget_analysis = state.get("budget_analysis", {})
        changes         = state.get("detected_changes", [])
        boq             = state.get("boq_line_items", [])
        city            = state.get("city", "Hyderabad")
        theme           = state.get("theme", "Modern Minimalist")
        budget_tier     = state.get("budget_tier", "mid")
        budget_inr      = state.get("budget_inr", 750000)
        room_type       = state.get("room_type", "bedroom")

        def fmt(n):
            if n >= 10_000_000: return f"₹{n/10_000_000:.1f}Cr"
            if n >= 100_000:    return f"₹{n/100_000:.1f}L"
            return f"₹{n:,.0f}"

        roi_pct      = roi.get("roi_pct", 0)
        equity       = roi.get("equity_gain_inr", 0)
        payback      = roi.get("payback_months", 36)
        rental_delta = roi.get("rental_yield_delta", 0)
        model_conf   = roi.get("model_confidence", 0.65)

        financial = {
            "renovation_cost":          fmt(budget_inr),
            "projected_roi":            f"{roi_pct:.1f}%",
            "equity_gain":              fmt(equity),
            "payback_period":           f"{payback} months",
            "rental_yield_improvement": f"+{rental_delta:.2f}%",
            "model_confidence":         f"{model_conf*100:.0f}%",
            "model_type":               roi.get("model_type", "heuristic"),
        }
        market = {
            "city":                city,
            "market_trend":        loc.get("trend", f"{city} real estate market stable."),
            "avg_appreciation_5yr": f"{loc.get('appreciation_5yr_pct', 30)}%",
            "rental_yield":        f"{loc.get('rental_yield_pct', 3.0)}%",
            "labour_premium":      f"+{loc.get('labour_premium_pct', 10)}% vs national avg",
            "market_tier":         f"Tier {loc.get('market_tier', 2)}",
        }
        budget = {
            "tier":              budget_tier,
            "range":             budget_analysis.get("budget_range", fmt(budget_inr)),
            "covers":            budget_analysis.get("what_it_covers", "Renovation work"),
            "roi_potential":     budget_analysis.get("roi_potential", f"{roi_pct:.0f}%"),
            "best_for":          budget_analysis.get("best_for", "Value addition"),
            "cautions":          budget_analysis.get("avoid", "None"),
            "recommended_brands": budget_analysis.get("recommended_brands", []),
        }
        top_materials = [
            f"{i.get('brand','?')} {i.get('product','?')} — {fmt(i.get('total_inr',0))}"
            for i in boq[:4]
        ]
        visual_summary = {
            "style_detected":           features.get("detected_style", theme),
            "wall_treatment":           features.get("wall_treatment", "Updated wall finish"),
            "floor_material":           features.get("floor_material", "New flooring installed"),
            "ceiling":                  features.get("ceiling_treatment", "Ceiling updated"),
            "colour_palette":           features.get("colour_palette", ["neutral"]),
            "quality_observed":         features.get("quality_tier", budget_tier),
            "specific_changes_detected": (changes[:5] if changes else [f"Applied {theme} style treatment"]),
        }

        insights = {
            "visual_analysis":    visual_summary,
            "financial_outlook":  financial,
            "market_intelligence": market,
            "budget_assessment":  budget,
            "top_materials":      top_materials,
            "recommendations":    _generate_recommendations(features, roi, loc, budget_tier, city, theme, room_type, budget_inr),
            "risk_factors":       _generate_risks(city, budget_tier, roi_pct),
            "summary_headline":   _generate_headline(theme, city, roi_pct, equity, budget_tier),
        }
        chat_context = _build_chat_context(state, insights)

        updates: PipelineState = {"insights": insights, "chat_context": chat_context}

    except Exception as e:
        logger.error(f"[insight_generation] Error: {e}")
        updates: PipelineState = {
            "insights": {
                "summary_headline": f"{state.get('theme','Modern')} renovation analysis",
                "financial_outlook": {}, "visual_analysis": {},
                "market_intelligence": {}, "budget_assessment": {},
                "recommendations": [], "risk_factors": [], "top_materials": [],
            },
            "chat_context": f"Renovation: {state.get('theme')} in {state.get('city')}",
            "errors": (state.get("errors") or []) + [f"insight_generation: {e}"],
        }

    _record_timing(updates, agent_name, t0, state)
    _mark_completed(updates, agent_name, state)
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_headline(theme, city, roi_pct, equity, budget_tier):
    f = lambda n: f"₹{n/100_000:.1f}L" if n >= 100_000 else f"₹{n:,.0f}"
    if equity > 0:
        return f"{theme} renovation in {city} projects {roi_pct:.1f}% ROI with {f(equity)} equity gain at {budget_tier} budget"
    return f"{theme} renovation complete — {roi_pct:.1f}% projected ROI for {city} market"


def _generate_recommendations(features, roi, loc, budget_tier, city, theme, room_type, budget_inr):
    recs, roi_pct = [], roi.get("roi_pct", 0)
    wall  = features.get("wall_treatment", "")
    floor = features.get("floor_material", "")
    recs.append(f"Priority action: {room_type.title()} renovation in {city} at {budget_tier} tier projects {roi_pct:.1f}% ROI — proceed with contractor procurement")
    if loc.get("market_tier") == 1:
        recs.append(f"Market timing: {city} Tier-1 market shows {loc.get('appreciation_5yr_pct',30)}% 5-year appreciation — renovation now maximises value")
    else:
        recs.append(f"{city} market: Focus on rental yield improvement ({loc.get('rental_yield_pct',3.0):.1f}% → {loc.get('rental_yield_pct',3.0)+roi.get('rental_yield_delta',0.5):.1f}%)")
    if "paint" in wall.lower() or "emulsion" in wall.lower():
        recs.append("Wall finish: Asian Paints Royale Sheen or Dulux Velvet Touch recommended — ₹380–680/litre range")
    if "tile" in floor.lower() or "vitrified" in floor.lower():
        recs.append("Flooring: Kajaria or Simpolo GVT tiles — confirm grout sealing within 30 days of installation")
    if budget_tier == "premium" and loc.get("market_tier", 2) > 1:
        recs.append(f"Budget alert: Premium budget in Tier-{loc.get('market_tier',2)} market risks over-capitalisation — verify comparables")
    elif budget_tier == "basic":
        recs.append("Budget tip: Prioritise paint + lighting over flooring for maximum visual impact per rupee")
    return recs[:5]


def _generate_risks(city, budget_tier, roi_pct):
    risks = []
    if city in ["Mumbai", "Delhi NCR"]:
        risks.append({"factor": "Labour cost overrun", "probability": "High", "detail": f"{city} labour premium 30–40% above national average", "mitigation": "Lock labour rate in contract before start"})
    if budget_tier == "premium" and roi_pct < 15:
        risks.append({"factor": "ROI compression", "probability": "Medium", "detail": "Premium spend at this ROI may not justify vs mid-tier", "mitigation": "Compare mid-tier scenario"})
    risks.append({"factor": "Material price volatility", "probability": "Medium", "detail": "Cement +8%, Steel +6% annual inflation projected", "mitigation": "Lock procurement contract in week 1"})
    return risks


def _build_chat_context(state: PipelineState, insights: dict) -> str:
    features    = state.get("image_features", {})
    roi         = state.get("roi_prediction", {})
    city        = state.get("city", "Hyderabad")
    theme       = state.get("theme", "Modern Minimalist")
    budget_tier = state.get("budget_tier", "mid")
    budget_inr  = state.get("budget_inr", 750000)
    room_type   = state.get("room_type", "bedroom")
    f = lambda n: f"\u20b9{n/100_000:.1f}L" if n >= 100_000 else f"\u20b9{n:,.0f}"
    changes_text = "; ".join(state.get("detected_changes", [])[:4]) or "renovation applied"
    base_ctx = (
        f"RENOVATION CONTEXT:\nTheme: {theme} | City: {city} | Budget: {budget_tier} ({f(budget_inr)}) | Room: {room_type}\n\n"
        f"CHANGES DETECTED:\n{changes_text}\n\n"
        f"VISUAL: walls={features.get('wall_treatment','?')} | floor={features.get('floor_material','?')} | style={features.get('detected_style',theme)}\n\n"
        f"FINANCIAL: ROI={roi.get('roi_pct',0):.1f}% | Equity={f(roi.get('equity_gain_inr',0))} | Payback={roi.get('payback_months',36)}mo\n\n"
        f"HEADLINE: {insights.get('summary_headline','')}"
    )
    # Append RAG knowledge context if available (passed through state from graph_pipeline.py)
    rag_ctx = state.get("rag_context", "")
    if rag_ctx:
        base_ctx += f"\n\n{rag_ctx[:2000]}"
    return base_ctx


def run_pipeline(
    initial_state: PipelineState,
    on_progress: Optional[Callable] = None,
) -> PipelineState:
    """
    Sequential 5-agent pipeline. LangGraph parallel graph is intentionally
    removed — it raised 'Can receive only one value per step' due to all
    parallel branches writing to shared keys (errors, agent_timings, etc.)
    and caused image_analysis to run twice.

    v4.1: ProductSuggesterAgent runs after all 5 agents complete.
    Uses rendered image (renovated_image_b64) + design recommendations
    to build shop_this_look links. Never blocks pipeline on failure.
    """
    state = dict(initial_state)

    for fn in [
        agent_image_analysis,
        agent_planning,
        agent_roi_prediction,
        agent_budget_location,
        agent_insight_generation,
    ]:
        try:
            state = fn(state)
            if on_progress:
                try:
                    on_progress(state)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Agent {fn.__name__} failed: {e}", exc_info=True)
            errs = list(state.get("errors") or [])
            errs.append(f"{fn.__name__}: {e}")
            state["errors"] = errs

    # ── Product Suggestions (post-pipeline, non-blocking) ──────────────────
    # Runs after all 5 agents so it can use the rendered image +
    # design_recommendations produced by the planning agent.
    try:
        import asyncio as _asyncio
        from agents.pipeline_product_integration import enrich_with_product_suggestions

        # enrich_with_product_suggestions is async — run it from sync context
        try:
            loop = _asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context (shouldn't happen here, but safe)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _asyncio.run, enrich_with_product_suggestions(state)
                    )
                    state = future.result(timeout=30)
            else:
                state = loop.run_until_complete(enrich_with_product_suggestions(state))
        except RuntimeError:
            # No event loop — create a new one
            state = _asyncio.run(enrich_with_product_suggestions(state))

        items = (state.get("product_suggestions") or {}).get("items_detected", 0)
        logger.info(f"[pipeline] ProductSuggesterAgent: {items} items detected")
    except Exception as e:
        logger.warning(f"[pipeline] ProductSuggesterAgent skipped: {e}")
        state.setdefault("product_suggestions", None)

    return state


def _record_timing(updates: dict, agent_name: str, t0: float, state: PipelineState):
    elapsed = round(time.perf_counter() - t0, 3)
    timings = dict(state.get("agent_timings") or {})
    timings[agent_name] = elapsed
    updates["agent_timings"] = timings


def _mark_completed(updates: dict, agent_name: str, state: PipelineState):
    completed = list(state.get("completed_agents") or [])
    if agent_name not in completed:
        completed.append(agent_name)
    updates["completed_agents"] = completed