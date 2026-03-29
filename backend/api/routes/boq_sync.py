"""
ARKEN — /api/v1/boq/sync route (NEW FILE)
==========================================
SAVE AS: backend/api/routes/boq_sync.py

BUG 2 FIX — BOQ generated before the renovated image exists.

Root cause: The main pipeline runs BOQ generation (node_design_planning)
before the rendering step. The rendering step is called separately by the
frontend through /api/v1/render. So by the time the renovated image exists,
the BOQ is already finalised against the ORIGINAL room's features.

This endpoint fixes that by re-generating the BOQ specifically against
the rendered/renovated image:

  1. Accept the rendered image (base64) from the frontend.
  2. Call Gemini Vision to detect the actual materials, finishes, and
     fixtures that are PRESENT in the renovated image.
  3. Map detected materials back to design_planner.py catalogs.
  4. Re-run DesignPlannerAgent.plan() with the detected data.
  5. Re-run BudgetEstimatorAgent with the new plan.
  6. Return updated boq_line_items, budget_estimate, and total_cost_inr.

The frontend should call this AFTER /api/v1/render returns an image,
passing the rendered image back. The response replaces the original BOQ.

Usage:
  POST /api/v1/boq/sync
  {
    "project_id": "...",
    "rendered_image_b64": "<base64 PNG/JPEG>",
    "rendered_image_mime": "image/png",
    "budget_tier": "mid",
    "room_type": "bedroom",
    "city": "Hyderabad",
    "floor_area_sqft": 120.0,
    "theme": "Modern Minimalist"
  }

Response:
  {
    "project_id": "...",
    "boq_line_items": [...],
    "budget_estimate": {...},
    "total_cost_inr": 650000,
    "materials_detected": [...],
    "sync_source": "rendered_image_gemini_vision",
    "note": "..."
  }
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Gemini Vision prompt for material detection ───────────────────────────────
_MATERIAL_DETECT_PROMPT = (
    "Analyse this renovated interior room image. "
    "For every surface, finish, and fixture you can see, identify: "
    "(1) item type (e.g. 'floor tiles', 'wall paint', 'ceiling', 'fan', 'wardrobe'), "
    "(2) material description (e.g. 'large format polished porcelain', 'matte emulsion', "
    "'POP false ceiling with LED cove'), "
    "(3) estimated brand tier: 'economy' | 'mid-range' | 'premium', "
    "(4) approximate quantity or area if estimable (e.g. '120 sqft', '4 units'). "
    "Also report: wall_condition ('good'|'fair'|'poor'), floor_condition ('good'|'fair'|'poor'). "
    "Return ONLY valid JSON with this schema: "
    '{"materials_detected": [{"item": str, "description": str, "tier": str, "quantity_hint": str}], '
    '"wall_condition": str, "floor_condition": str}'
)

# ── Map detected material tier names → design_planner tier keys ───────────────
_TIER_MAP = {
    "economy":   "basic",
    "basic":     "basic",
    "mid-range": "mid",
    "mid":       "mid",
    "premium":   "premium",
    "luxury":    "premium",
}

# ── Map detected item names → design_planner quantity keys ────────────────────
_ITEM_TO_QTY_KEY = {
    "floor":        "floor_tiles_sqft",
    "tile":         "floor_tiles_sqft",
    "wall":         "wall_area_sqft",
    "wall tile":    "wall_tiles_sqft",
    "plywood":      "plywood_sqft",
    "paint":        "paint_liters",
}


class BOQSyncRequest(BaseModel):
    project_id:           str
    rendered_image_b64:   str
    rendered_image_mime:  str = "image/png"
    budget_tier:          str = "mid"
    room_type:            str = "bedroom"
    city:                 str = "Hyderabad"
    floor_area_sqft:      float = 120.0
    theme:                str = "Modern Minimalist"
    budget_inr:           int = 750_000


class BOQSyncResponse(BaseModel):
    project_id:        str
    boq_line_items:    List[Dict[str, Any]]
    budget_estimate:   Dict[str, Any]
    total_cost_inr:    int
    materials_detected: List[Dict[str, Any]]
    sync_source:       str
    boq_scope_tier:    str
    note:              str


@router.post("/sync", response_model=BOQSyncResponse)
async def sync_boq_with_render(req: BOQSyncRequest):
    """
    Re-generate the BOQ and cost estimate based on the actual rendered image.

    Call this AFTER /api/v1/render returns an image. Pass the rendered
    image b64 string. The response BOQ replaces the pre-render BOQ.
    """
    logger.info(
        f"[boq_sync] project={req.project_id} tier={req.budget_tier} "
        f"room={req.room_type} area={req.floor_area_sqft}sqft"
    )

    # ── Decode rendered image ─────────────────────────────────────────────────
    try:
        img_b64 = req.rendered_image_b64
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(img_b64)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 rendered image: {e}")

    # ── Step 1: Gemini Vision — detect materials in rendered image ────────────
    materials_detected: List[Dict] = []
    wall_condition   = "fair"
    floor_condition  = "fair"
    sync_source      = "rendered_image_gemini_vision"

    try:
        from services.llm import _client
        from google.genai import types as _types

        client = _client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                _types.Content(
                    role="user",
                    parts=[
                        _types.Part(inline_data=_types.Blob(
                            mime_type=req.rendered_image_mime,
                            data=image_bytes,
                        )),
                        _types.Part(text=_MATERIAL_DETECT_PROMPT),
                    ],
                )
            ],
            config=_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        raw_text = response.text.strip()
        # Clean markdown fences
        raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"\n?```$", "", raw_text, flags=re.MULTILINE).strip()

        parsed = json.loads(raw_text)
        materials_detected = parsed.get("materials_detected", [])
        wall_condition     = parsed.get("wall_condition", "fair")
        floor_condition    = parsed.get("floor_condition", "fair")

        logger.info(
            f"[boq_sync] Gemini detected {len(materials_detected)} materials "
            f"wall={wall_condition} floor={floor_condition}"
        )

    except Exception as e:
        logger.warning(f"[boq_sync] Gemini material detection failed: {e} — using defaults")
        sync_source = "defaults_gemini_unavailable"

    # ── Step 2: Map detected materials → quantities ───────────────────────────
    quantities: Dict[str, float] = {
        "floor_tiles_sqft": req.floor_area_sqft,
        "wall_area_sqft":   req.floor_area_sqft * 3.5,
        "paint_liters":     req.floor_area_sqft * 3.5 / 12,   # coverage ~12sqft/litre
        "plywood_sqft":     req.floor_area_sqft * 0.25 if req.budget_tier != "basic" else 0.0,
        "wall_tiles_sqft":  req.floor_area_sqft * 0.6 if req.room_type in ("bathroom", "kitchen") else 0.0,
    }

    # Override quantities with any specific measurements Gemini extracted
    for mat in materials_detected:
        quantity_hint = str(mat.get("quantity_hint", ""))
        item_name     = str(mat.get("item", "")).lower()
        # Extract numeric value from hint like "120 sqft" or "4 units"
        nums = re.findall(r"\d+\.?\d*", quantity_hint)
        if nums:
            val = float(nums[0])
            for key_kw, qty_key in _ITEM_TO_QTY_KEY.items():
                if key_kw in item_name:
                    quantities[qty_key] = val
                    break

    # ── Step 3: Determine effective tier from detected materials ──────────────
    # If Gemini sees "premium" materials in a "basic" render, trust the render.
    detected_tiers = [
        _TIER_MAP.get(str(mat.get("tier", "")).lower(), req.budget_tier.lower())
        for mat in materials_detected
    ]
    if detected_tiers:
        tier_votes = {t: detected_tiers.count(t) for t in set(detected_tiers)}
        effective_tier = max(tier_votes, key=tier_votes.get)
        if effective_tier != req.budget_tier.lower():
            logger.info(
                f"[boq_sync] Detected tier '{effective_tier}' differs from "
                f"requested '{req.budget_tier}' — using detected tier for BOQ"
            )
    else:
        effective_tier = req.budget_tier.lower()

    # ── Step 4: Re-run DesignPlannerAgent with image-derived data ────────────
    plan: Dict[str, Any] = {}
    try:
        from agents.design_planner import DesignPlannerAgent

        planner = DesignPlannerAgent()
        plan = planner.plan(
            theme=req.theme,
            budget_inr=req.budget_inr,
            budget_tier=effective_tier,
            area_sqft=req.floor_area_sqft,
            room_type=req.room_type,
            city=req.city,
            quantities=quantities,
            wall_condition=wall_condition,
            floor_condition=floor_condition,
            issues_detected=[],
            renovation_scope="partial",
            high_value_upgrades=[
                mat.get("item", "") for mat in materials_detected
                if _TIER_MAP.get(str(mat.get("tier", "")).lower()) == "premium"
            ],
            condition_score=75,   # post-renovation — condition is good
        )
        logger.info(
            f"[boq_sync] DesignPlanner done — "
            f"{len(plan.get('line_items', []))} line items, "
            f"total=₹{plan.get('total_inr', 0):,}"
        )
    except Exception as e:
        logger.error(f"[boq_sync] DesignPlannerAgent failed: {e}", exc_info=True)
        raise HTTPException(500, f"BOQ generation failed: {e}")

    boq_line_items = plan.get("line_items", [])

    # ── Step 5: Re-run BudgetEstimatorAgent ──────────────────────────────────
    budget_estimate: Dict[str, Any] = {}
    try:
        from agents.budget_estimator_agent import BudgetEstimatorAgent

        estimator = BudgetEstimatorAgent()
        mock_state = {
            "budget_inr":      req.budget_inr,
            "budget_tier":     effective_tier,
            "city":            req.city,
            "room_type":       req.room_type,
            "floor_area_sqft": req.floor_area_sqft,
            "design_plan":     plan,
        }
        result = estimator._estimate(mock_state)
        budget_estimate = result.get("budget_estimate", {})
        logger.info(
            f"[boq_sync] BudgetEstimator done — "
            f"total=₹{budget_estimate.get('total_cost_inr', 0):,} "
            f"sanity={budget_estimate.get('sanity_check', {}).get('status', '?')}"
        )
    except Exception as e:
        logger.warning(f"[boq_sync] BudgetEstimatorAgent failed: {e} — using plan total")
        budget_estimate = {
            "total_cost_inr": plan.get("total_inr", 0),
            "materials_inr":  plan.get("material_inr", 0),
            "labour_inr":     plan.get("labour_inr", 0),
            "gst_inr":        plan.get("gst_inr", 0),
            "contingency_inr": plan.get("contingency_inr", 0),
        }

    total_cost_inr = budget_estimate.get("total_cost_inr", plan.get("total_inr", 0))

    note = (
        f"BOQ re-generated against the AI-renovated image. "
        f"Gemini Vision detected {len(materials_detected)} material(s). "
        f"Effective tier: {effective_tier}. "
        f"This BOQ reflects what is actually shown in the rendered room."
    )
    if sync_source != "rendered_image_gemini_vision":
        note = (
            "BOQ generated using defaults (Gemini Vision unavailable). "
            "For best accuracy, ensure GOOGLE_API_KEY is configured."
        )

    return BOQSyncResponse(
        project_id=req.project_id,
        boq_line_items=boq_line_items,
        budget_estimate=budget_estimate,
        total_cost_inr=total_cost_inr,
        materials_detected=materials_detected,
        sync_source=sync_source,
        boq_scope_tier=effective_tier,
        note=note,
    )
