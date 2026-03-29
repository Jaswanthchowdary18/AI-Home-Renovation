"""
ARKEN — Forecast API Routes v2.0
==================================
Exposes PriceForecastAgent and ROIForecastAgent via FastAPI.

v2.0 upgrades:
  - /forecast/materials/project: project-aware material forecast with portfolio summary
  - /forecast/roi: accepts materials list and renovation_scope for richer prediction
  - /forecast/roi/explain: returns full explainability breakdown
  - All responses include validation metadata and procurement recommendations
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from agents.price_forecast import PriceForecastAgent, SEED_DATA
from agents.roi_forecast import ROIForecastAgent
from pydantic import BaseModel, Field

router = APIRouter()


# ── Request models ─────────────────────────────────────────────────────────────

class ROIRequest(BaseModel):
    renovation_cost_inr: int
    area_sqft: float
    city: str
    room_type: str = "bedroom"
    budget_tier: str = "mid"
    current_property_value_inr: Optional[int] = None
    property_age_years: int = 10
    existing_condition: str = "average"
    # v2.0 new fields
    materials: Optional[List[str]] = Field(
        default=None,
        description="List of material keys from MATERIAL_ROI_FACTORS (e.g. 'premium_flooring', 'modular_kitchen')"
    )
    renovation_scope: str = Field(
        default="partial",
        description="cosmetic_only | partial | full_room | structural_plus"
    )


class ProjectForecastRequest(BaseModel):
    city: str
    room_type: str
    area_sqft: float
    horizon_days: int = 90
    materials_override: Optional[List[str]] = None


# ── Material forecast endpoints ────────────────────────────────────────────────

@router.get("/materials")
async def get_material_forecast(
    horizon_days: int = Query(90, ge=30, le=180),
    city: Optional[str] = None,
    area_sqft: Optional[float] = None,
    room_type: Optional[str] = None,
):
    """
    Returns 90-day price forecast for all tracked Indian construction materials.
    Optionally accepts city/area_sqft/room_type for project-adjusted pricing.
    """
    agent = PriceForecastAgent()
    return {
        "forecasts": agent.forecast_all(
            horizon_days=horizon_days,
            city=city,
            area_sqft=area_sqft,
            room_type=room_type,
        )
    }


@router.get("/materials/{material_key}")
async def get_single_material_forecast(
    material_key: str,
    horizon_days: int = 90,
    city: Optional[str] = None,
    area_sqft: Optional[float] = None,
    room_type: Optional[str] = None,
):
    """
    Returns 90-day forecast for a single material.
    Includes procurement recommendation and budget impact estimate.
    """
    agent = PriceForecastAgent()
    result = agent.forecast_material(
        material_key, horizon_days,
        city=city, area_sqft=area_sqft, room_type=room_type,
    )
    if not result:
        raise HTTPException(404, f"Material '{material_key}' not in catalog. "
                                 f"Available: {list(SEED_DATA.keys())}")
    return result


@router.get("/materials/catalog")
async def get_material_catalog():
    """Returns full material catalog with display names and categories."""
    return {
        "materials": [
            {
                "key": k,
                "display_name": v.get("display_name", k),
                "category": v.get("category", ""),
                "unit": v.get("unit", ""),
                "current_price_inr": v.get("current_inr", 0),
            }
            for k, v in SEED_DATA.items()
        ]
    }


@router.post("/materials/project")
async def get_project_material_forecast(req: ProjectForecastRequest):
    """
    Project-aware material forecast.
    Returns only materials relevant to the room_type with portfolio summary.
    Includes total budget exposure from price movements.
    """
    agent = PriceForecastAgent()
    return agent.forecast_for_project(
        room_type=req.room_type,
        area_sqft=req.area_sqft,
        city=req.city,
        horizon_days=req.horizon_days,
        materials_override=req.materials_override,
    )


# ── ROI forecast endpoints ─────────────────────────────────────────────────────

@router.post("/roi")
async def predict_roi(req: ROIRequest):
    """
    Predict property ROI for a renovation project.
    v2.0: accepts materials list and renovation_scope for richer, validated predictions.
    Returns confidence interval and explainability breakdown.
    """
    agent = ROIForecastAgent()
    return agent.predict(
        renovation_cost_inr=req.renovation_cost_inr,
        area_sqft=req.area_sqft,
        city=req.city,
        room_type=req.room_type,
        budget_tier=req.budget_tier,
        current_property_value_inr=req.current_property_value_inr,
        property_age_years=req.property_age_years,
        existing_condition=req.existing_condition,
        materials=req.materials,
        renovation_scope=req.renovation_scope,
    )


@router.post("/roi/explain")
async def explain_roi(req: ROIRequest):
    """
    Returns full ROI explanation with driver breakdown.
    Same as /roi but response includes the full explanation.explanation dict.
    """
    agent = ROIForecastAgent()
    result = agent.predict(
        renovation_cost_inr=req.renovation_cost_inr,
        area_sqft=req.area_sqft,
        city=req.city,
        room_type=req.room_type,
        budget_tier=req.budget_tier,
        current_property_value_inr=req.current_property_value_inr,
        property_age_years=req.property_age_years,
        existing_condition=req.existing_condition,
        materials=req.materials,
        renovation_scope=req.renovation_scope,
    )
    # Ensure explanation is always present in this endpoint
    if "explanation" not in result:
        result["explanation"] = {
            "roi_narrative": result.get("explanation", {}).get("roi_narrative", ""),
            "primary_drivers": [],
            "adjustments": [],
        }
    return result


@router.get("/roi/materials")
async def get_material_roi_factors():
    """Returns all material ROI contribution factors used in the prediction model."""
    from agents.roi_forecast import MATERIAL_ROI_FACTORS, RENOVATION_SCOPE_MULTIPLIER
    return {
        "material_roi_factors": {
            k: {"boost_pct": round(v * 100, 1), "description": k.replace("_", " ").title()}
            for k, v in MATERIAL_ROI_FACTORS.items()
        },
        "renovation_scope_multipliers": {
            k: {"multiplier": v, "description": k.replace("_", " ").title()}
            for k, v in RENOVATION_SCOPE_MULTIPLIER.items()
        },
    }