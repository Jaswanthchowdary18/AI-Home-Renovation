"""
ARKEN — Products API Routes v1.0
===================================
SAVE AS: backend/api/routes/products.py  — NEW FILE

POST /api/v1/products/suggest
  Accepts a project_id + rendered image (base64 or S3 URL) + metadata.
  Calls ProductSuggesterAgent and returns shop_this_look.
  Result cached in Redis/in-memory for 24 hours keyed by project_id.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from agents.product_suggester_agent import ProductSuggesterAgent
from services.cache import cache_service

logger  = logging.getLogger(__name__)
router  = APIRouter()

_CACHE_TTL_SECONDS = 86_400   # 24 hours


# ── Request / Response models ─────────────────────────────────────────────────

class ProductSuggestRequest(BaseModel):
    project_id: str
    # Caller supplies ONE of: rendered_image_base64 OR rendered_image_url
    rendered_image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded JPEG/PNG of the rendered room image",
    )
    rendered_image_url: Optional[str] = Field(
        default=None,
        description="Publicly accessible URL (e.g. S3 pre-signed URL) for the rendered image",
    )
    room_type:    str = "bedroom"
    style_label:  str = "Modern Minimalist"
    budget_tier:  str = "mid"
    city:         str = "Hyderabad"
    # Optional: design recommendations from visual_assessor for fallback
    design_recommendations: Optional[list] = Field(
        default=None,
        description="Output of design_planner / visual_assessor (used as fallback when image analysis fails)",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_key(project_id: str) -> str:
    return f"arken:product_suggestions:{project_id}"


async def _fetch_image_bytes(url: str) -> bytes:
    """Download image from a URL (S3 pre-signed or CDN)."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def _decode_base64(b64: str) -> bytes:
    """Decode a base64 string, stripping data URI prefix if present."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/suggest", summary="Generate shop-this-look suggestions for a rendered room")
async def suggest_products(req: ProductSuggestRequest):
    """
    Detect furniture and decor items in the rendered room image and return
    e-commerce links for each item from Amazon India, Flipkart, Pepperfry, etc.

    Results are cached for 24 hours per project_id to avoid redundant Gemini calls.
    """
    cache_key = _cache_key(req.project_id)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    cached = await cache_service.get(cache_key)
    if cached:
        try:
            logger.info(f"[Products] Cache hit for project {req.project_id}")
            return json.loads(cached)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[Products] Cache parse failed for {req.project_id}: {e}")

    # ── Resolve image bytes ───────────────────────────────────────────────────
    image_bytes: Optional[bytes] = None
    mime_type   = "image/jpeg"

    if req.rendered_image_base64:
        try:
            image_bytes = _decode_base64(req.rendered_image_base64)
        except Exception as e:
            logger.warning(f"[Products] base64 decode failed: {e}")

    elif req.rendered_image_url:
        try:
            image_bytes = await _fetch_image_bytes(req.rendered_image_url)
        except httpx.HTTPStatusError as e:
            logger.warning(f"[Products] Image URL fetch failed (HTTP {e.response.status_code}): {e}")
        except httpx.RequestError as e:
            logger.warning(f"[Products] Image URL request error: {e}")

    # ── Call agent ────────────────────────────────────────────────────────────
    try:
        agent  = ProductSuggesterAgent()
        result = agent.suggest(
            rendered_image_bytes=image_bytes,
            room_type=req.room_type,
            style_label=req.style_label,
            budget_tier=req.budget_tier,
            city=req.city,
            design_recommendations=req.design_recommendations,
            mime_type=mime_type,
        )
    except Exception as e:
        logger.error(f"[Products] ProductSuggesterAgent failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Product suggestion failed: {e}",
        )

    # ── Cache result ──────────────────────────────────────────────────────────
    try:
        await cache_service.set(cache_key, json.dumps(result), ttl=_CACHE_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"[Products] Cache write failed: {e}")

    return result


@router.delete("/suggest/{project_id}", summary="Invalidate cached product suggestions")
async def invalidate_suggestions(project_id: str):
    """Force-expire the cached shop_this_look result for a project (e.g. after re-render)."""
    cache_key = _cache_key(project_id)
    try:
        await cache_service.delete(cache_key)
    except Exception as e:
        logger.warning(f"[Products] Cache delete failed for {project_id}: {e}")
    return {"deleted": True, "project_id": project_id}
