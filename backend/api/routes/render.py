"""
ARKEN — /api/v1/render route
Uses gemini-2.5-flash-image for renovation image generation.
Returns base64 PNG for direct frontend use.
"""

import base64
import logging
import uuid
from typing import Optional, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from agents.rendering import RenderingAgent

router = APIRouter()
logger = logging.getLogger(__name__)
rendering_agent = RenderingAgent()


class RenderRequest(BaseModel):
    project_id: str
    original_image_b64: str
    original_mime: str = "image/jpeg"
    version: int = 1
    theme: str
    city: str
    budget_tier: str        # "basic" | "mid" | "premium"
    room_type: str = "bedroom"
    custom_instructions: str = ""
    material_overrides: Optional[Dict] = None
    # CV analysis data — populated by vision pipeline, optional for direct API callers
    detected_objects: Optional[list] = None
    wall_color: str = "neutral"
    floor_type: str = "tiles"
    window_count: int = 1
    door_count: int = 1
    room_dimensions_hint: str = ""


class RenderResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    render_id: str
    image_b64: str          # use as: data:image/png;base64,{image_b64}
    image_mime: str = "image/png"
    model_used: str
    version: int
    cdn_url: Optional[str] = None
    generation_time_ms: int


class ReRenderRequest(BaseModel):
    project_id: str
    original_image_b64: str
    original_mime: str = "image/jpeg"
    version: int = 2
    theme: str
    city: str
    budget_tier: str
    room_type: str = "bedroom"
    new_instructions: str = ""
    material_overrides: Optional[Dict] = None
    # CV analysis data — optional, passed through from prior vision analysis
    detected_objects: Optional[list] = None
    wall_color: str = "neutral"
    floor_type: str = "tiles"
    window_count: int = 1
    door_count: int = 1
    room_dimensions_hint: str = ""


@router.post("/", response_model=RenderResponse)
async def create_render(req: RenderRequest):
    """Renovate a room using gemini-2.5-flash-image."""
    if not settings.GOOGLE_API_KEY:
        raise HTTPException(503, detail="GOOGLE_API_KEY not configured in backend/.env")

    render_id = str(uuid.uuid4())
    logger.info(f"[{render_id}] Render — theme={req.theme}, city={req.city}, room={req.room_type}")

    try:
        image_bytes = base64.b64decode(req.original_image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data")

    try:
        result = await rendering_agent.render(
            original_image_bytes=image_bytes,
            project_id=req.project_id,
            version=req.version,
            theme=req.theme,
            city=req.city,
            budget_tier=req.budget_tier,
            room_type=req.room_type,
            custom_instructions=req.custom_instructions,
            material_overrides=req.material_overrides,
            detected_objects=req.detected_objects,
            wall_color=req.wall_color,
            floor_type=req.floor_type,
            window_count=req.window_count,
            door_count=req.door_count,
            room_dimensions_hint=req.room_dimensions_hint,
        )
    except RuntimeError as e:
        logger.error(f"[{render_id}] {e}")
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        logger.error(f"[{render_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Render failed: {str(e)}")

    if not result.get("image_b64"):
        raise HTTPException(500, "Render completed but no image was returned")

    return RenderResponse(
        render_id=render_id,
        image_b64=result["image_b64"],
        image_mime=result.get("image_mime", "image/png"),
        model_used=result["model_used"],
        version=req.version,
        cdn_url=result.get("cdn_url"),
        generation_time_ms=result.get("generation_time_ms", 0),
    )


@router.post("/rerender", response_model=RenderResponse)
async def rerender(req: ReRenderRequest):
    """Re-render with updated instructions."""
    if not settings.GOOGLE_API_KEY:
        raise HTTPException(503, "GOOGLE_API_KEY not configured")

    render_id = str(uuid.uuid4())

    try:
        image_bytes = base64.b64decode(req.original_image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data")

    try:
        result = await rendering_agent.render(
            original_image_bytes=image_bytes,
            project_id=req.project_id,
            version=req.version,
            theme=req.theme,
            city=req.city,
            budget_tier=req.budget_tier,
            room_type=req.room_type,
            custom_instructions=req.new_instructions,
            material_overrides=req.material_overrides,
            detected_objects=req.detected_objects,
            wall_color=req.wall_color,
            floor_type=req.floor_type,
            window_count=req.window_count,
            door_count=req.door_count,
            room_dimensions_hint=req.room_dimensions_hint,
        )
    except RuntimeError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Re-render failed: {str(e)}")

    return RenderResponse(
        render_id=render_id,
        image_b64=result["image_b64"],
        image_mime=result.get("image_mime", "image/png"),
        model_used=result["model_used"],
        version=req.version,
        cdn_url=result.get("cdn_url"),
        generation_time_ms=result.get("generation_time_ms", 0),
    )