"""
ARKEN — /api/v1/chat route v4.0
================================
SAVE AS: backend/api/routes/chat.py — REPLACE existing

v4.0 Changes over v3.0 (PROBLEM 2 FIX — project-grounded responses):
  1. build_project_context_prompt() injected into EVERY chat request
     when project_id or pipeline_context is present.
  2. System prompt updated to GROUNDED_SYSTEM_PROMPT from llm.py —
     instructs model to cite exact ROI/cost numbers and Indian standards.
  3. pipeline_context field enriched from project cache when only
     project_id is supplied (no need for frontend to re-send full context).
  4. max_output_tokens = 8192 preserved from v3.0.

CRITICAL: rendering.py and IMAGE_MODELS are NOT touched.
"""

import base64
import json
import logging
import re
import uuid
from typing import AsyncGenerator, Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

from config import settings
from services.cache import cache_service
from services.llm import build_project_context_prompt, GROUNDED_SYSTEM_PROMPT

router = APIRouter()
logger = logging.getLogger(__name__)

GEMINI_TEXT_MODEL = "gemini-2.5-flash"

# ── Renovation system prompt (extends GROUNDED_SYSTEM_PROMPT from llm.py) ─────
RENOVATION_SYSTEM_PROMPT = f"""{GROUNDED_SYSTEM_PROMPT}

You have been given TWO room images:
1. ORIGINAL ROOM — the user's actual room photo before renovation
2. RENOVATED ROOM — the AI-generated renovation in the chosen style

ADDITIONAL RULES:
- ONLY reference what you can actually see in both images + what is in the context
- Give specific Indian brand + product names (Asian Paints, Kajaria, Greenply, Havells, \
Jaquar, Hindware, Legrand, Somany, Nitco, Century Ply)
- Always quote prices in INR with ranges (e.g., ₹45-65 per sqft)
- Be comprehensive and complete — do NOT cut responses short
- Finish every answer fully before stopping
- Format lists clearly using plain dashes, no markdown bold

WHEN USER WANTS A CHANGE (darker walls, different tiles, add greenery, etc.):
At the end of your response, include this exact block:
<action>
{{"custom_instructions": "describe the visual change clearly", "material_overrides": {{}}, "theme_change": null}}
</action>

Only include <action> when the user explicitly wants a visual re-render change."""


def _get_client() -> genai.Client:
    if not settings.GOOGLE_API_KEY:
        raise HTTPException(503, "GOOGLE_API_KEY not configured in backend/.env")
    return genai.Client(api_key=settings.GOOGLE_API_KEY.get_secret_value())


# ── Request / Response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    project_id: str
    session_id: Optional[str] = None
    messages: List[ChatMessage]
    original_image_b64: Optional[str] = None
    original_mime: str = "image/jpeg"
    renovated_image_b64: Optional[str] = None
    renovated_mime: str = "image/png"
    theme: str = "Modern Minimalist"
    city: str = "Hyderabad"
    budget_tier: str = "mid"
    room_type: str = "bedroom"
    pipeline_context: Optional[str] = None   # pre-built context string from pipeline
    pipeline_result: Optional[Dict[str, Any]] = None  # full pipeline result dict
    stream: bool = False


class InsightsRequest(BaseModel):
    project_id: str
    original_image_b64: Optional[str] = None
    original_image_mime: str = "image/jpeg"
    renovated_image_b64: Optional[str] = None
    renovated_image_mime: str = "image/png"
    theme: str = "Modern Minimalist"
    city: str = "Hyderabad"
    budget_tier: str = "mid"
    budget_inr: int = 750000
    room_type: str = "bedroom"


class ChatResponse(BaseModel):
    session_id: str
    message: str
    action: Optional[dict] = None
    triggers_rerender: bool = False


class InsightsResponse(BaseModel):
    insights: Dict[str, Any]
    pipeline_summary: Dict[str, Any]
    agent_timings: Dict[str, float]
    completed_agents: List[str]
    errors: List[str]


# ── PROBLEM 2 FIX: Project context injection ──────────────────────────────────

async def _get_project_context(req: ChatRequest) -> str:
    """
    Build the project context string to inject into the system prompt.

    Priority:
      1. If pipeline_result is provided directly, build from it.
      2. If pipeline_context (pre-built string) is provided, use it.
      3. Try to load cached pipeline result by project_id.
      4. Fall back to basic context from request fields.
    """
    # Option 1: Full pipeline result provided
    if req.pipeline_result:
        try:
            return build_project_context_prompt(req.pipeline_result)
        except Exception as e:
            logger.warning(f"[chat] build_project_context_prompt failed: {e}")

    # Option 2: Pre-built context string
    if req.pipeline_context:
        return req.pipeline_context

    # Option 3: Try cache
    cache_key = f"pipeline_result:{req.project_id}"
    try:
        cached = await cache_service.get(cache_key)
        if cached and isinstance(cached, dict):
            return build_project_context_prompt(cached)
    except Exception as e:
        logger.debug(f"[chat] Cache lookup for project context failed: {e}")

    # Option 4: Minimal fallback from request fields
    budget_label = {"basic": "₹3–5L", "mid": "₹5–10L", "premium": "₹10L+"}.get(
        req.budget_tier, "₹5–10L"
    )
    return (
        f"CURRENT PROJECT CONTEXT:\n"
        f"City: {req.city}\n"
        f"Room: {req.room_type.replace('_', ' ').title()}\n"
        f"Budget: {budget_label} ({req.budget_tier} tier)\n"
        f"Style: {req.theme}\n"
        f"Note: Full pipeline data not yet available — run analysis first for detailed ROI and BOQ."
    )


# ── Vision contents builder ────────────────────────────────────────────────────

def _build_vision_contents(req: ChatRequest, project_context: str) -> list:
    # Build context text
    context_text = project_context

    contents = []
    first_parts = [types.Part(text=context_text)]

    # Attach original image
    if req.original_image_b64:
        try:
            first_parts.append(types.Part(inline_data=types.Blob(
                mime_type=req.original_mime,
                data=base64.b64decode(req.original_image_b64),
            )))
            first_parts.append(types.Part(text="[IMAGE 1: ORIGINAL ROOM — before renovation]"))
        except Exception as e:
            logger.warning(f"[chat] Could not attach original image: {e}")

    # Attach renovated image
    if req.renovated_image_b64:
        try:
            first_parts.append(types.Part(inline_data=types.Blob(
                mime_type=req.renovated_mime,
                data=base64.b64decode(req.renovated_image_b64),
            )))
            first_parts.append(types.Part(text="[IMAGE 2: RENOVATED ROOM — after AI renovation]"))
        except Exception as e:
            logger.warning(f"[chat] Could not attach renovated image: {e}")

    contents.append(types.Content(role="user", parts=first_parts))
    contents.append(types.Content(role="model", parts=[
        types.Part(text=(
            "I can clearly see both the original and renovated room images, "
            "and I have the full project context including ROI, budget, and schedule data. "
            "I will answer using the specific numbers from your project."
        ))
    ]))

    # Add conversation history (all but last message)
    for msg in req.messages[:-1]:
        role = "user" if msg.role == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

    # Add final user message with renovated image for reference
    if req.messages:
        final_parts = [types.Part(text=req.messages[-1].content)]
        if req.renovated_image_b64:
            try:
                final_parts.append(types.Part(inline_data=types.Blob(
                    mime_type=req.renovated_mime,
                    data=base64.b64decode(req.renovated_image_b64),
                )))
            except Exception:
                pass
        contents.append(types.Content(role="user", parts=final_parts))

    return contents


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    client     = _get_client()

    # PROBLEM 2 FIX: always inject project context
    project_context = await _get_project_context(req)
    contents        = _build_vision_contents(req, project_context)

    try:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=RENOVATION_SYSTEM_PROMPT,
                temperature=0.35,
                max_output_tokens=8192,
            ),
        )
        response_text = response.text or "I could not generate a response. Please try again."
    except Exception as e:
        logger.error(f"[chat] Gemini error: {e}", exc_info=True)
        raise HTTPException(500, f"Chat error: {str(e)}")

    action = None
    triggers_rerender = False
    clean_response = response_text

    if "<action>" in response_text:
        match = re.search(r"<action>(.*?)</action>", response_text, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group(1).strip())
                triggers_rerender = bool(
                    action.get("material_overrides") or
                    action.get("theme_change") or
                    action.get("custom_instructions")
                )
            except json.JSONDecodeError:
                pass
        clean_response = re.sub(r"<action>.*?</action>", "", response_text, flags=re.DOTALL).strip()

    try:
        await cache_service.set(
            f"chat_session:{session_id}",
            {"project_id": req.project_id, "theme": req.theme, "city": req.city},
            ttl=7200,
        )
    except Exception as e:
        logger.debug(f"[chat] Session cache write failed: {e}")

    return ChatResponse(
        session_id=session_id,
        message=clean_response,
        action=action,
        triggers_rerender=triggers_rerender,
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    client = _get_client()
    project_context = await _get_project_context(req)
    contents        = _build_vision_contents(req, project_context)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            for chunk in client.models.generate_content_stream(
                model=GEMINI_TEXT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=RENOVATION_SYSTEM_PROMPT,
                    temperature=0.35,
                    max_output_tokens=8192,
                ),
            ):
                if chunk.text:
                    yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/insights", response_model=InsightsResponse)
async def run_insights(req: InsightsRequest):
    """
    Run the full multi-agent pipeline on the (renovated) image.
    Caches the result by project_id for 2 hours so chat can retrieve it.
    """
    from agents.multi_agent_pipeline import run_multi_agent_pipeline, build_initial_state

    # build_initial_state uses image_b64/image_mime for the primary (renovated) image.
    # The original image is passed separately into the state after construction.
    initial_state = build_initial_state(
        project_id=req.project_id,
        image_b64=req.renovated_image_b64 or req.original_image_b64 or "",
        image_mime=req.renovated_image_mime or req.original_image_mime or "image/jpeg",
        theme=req.theme,
        city=req.city,
        budget_tier=req.budget_tier,
        budget_inr=req.budget_inr or 0,
        room_type=req.room_type,
    )
    # Preserve both image fields so downstream agents can access each separately
    initial_state["original_image_b64"]  = req.original_image_b64 or ""
    initial_state["original_image_mime"] = req.original_image_mime or "image/jpeg"
    initial_state["renovated_image_b64"]  = req.renovated_image_b64 or ""
    initial_state["renovated_image_mime"] = req.renovated_image_mime or "image/jpeg"
    initial_state["errors"] = []
    initial_state["agent_timings"] = {}
    initial_state["completed_agents"] = []

    try:
        final_state = run_multi_agent_pipeline(initial_state)
    except Exception as e:
        logger.error(f"[chat/insights] Pipeline failed for {req.project_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Pipeline failed: {str(e)}")

    # ── Enrich with product suggestions (post-pipeline step) ─────────────────
    try:
        from agents.pipeline_product_integration import enrich_with_product_suggestions
        final_state = await enrich_with_product_suggestions(final_state)
    except Exception as e:
        logger.warning(f"[chat/insights] Product enrichment failed: {e}")

    # PROBLEM 2 FIX: cache full pipeline result for chat grounding
    try:
        await cache_service.set(
            f"pipeline_result:{req.project_id}",
            final_state,
            ttl=7200,
        )
    except Exception as e:
        logger.debug(f"[chat/insights] Pipeline result cache write failed: {e}")

    # Extract contractor_list from schedule if present
    _schedule = final_state.get("schedule", {})
    _contractor_list = _schedule.get("contractor_list", []) if isinstance(_schedule, dict) else []

    # Compute labour total from Labour- category items so the frontend
    # BOQTable shows the correct Labour stat card value.
    _boq_items = final_state.get("boq_line_items", [])
    _labour_total = sum(
        i.get("total_inr", 0) for i in _boq_items
        if str(i.get("category", "")).startswith("Labour")
    )
    # Fall back to state.labour_estimate if BOQ has no labour items
    _labour_estimate = _labour_total or final_state.get("labour_estimate", 0)

    pipeline_summary = {
        "chat_context":       final_state.get("chat_context", ""),
        "image_features":     final_state.get("image_features", {}),
        # FIX: roi_output is now the full dict (roi_agent_node v2.0).
        # Send both keys so both old and new dashboard code picks it up.
        "roi":                final_state.get("roi_output") or final_state.get("roi_prediction", {}),
        "roi_prediction":     final_state.get("roi_output") or final_state.get("roi_prediction", {}),
        "location_context":   final_state.get("location_context", {}),
        "budget_analysis":    final_state.get("budget_analysis", {}),
        "boq_line_items":     _boq_items,
        "labour_estimate":    _labour_estimate,
        "material_plan":      final_state.get("material_plan", {}),
        "detected_changes":   final_state.get("detected_changes", []),
        "design_plan":        final_state.get("design_plan", {}),
        "schedule":           _schedule,
        "material_prices":    final_state.get("material_prices", []),
        # ── AUTHORITATIVE COST: cost_estimate is set by BudgetEstimatorAgent ─
        # total_inr = city-adjusted final cost (materials + labour + GST + contingency)
        # This is the single source of truth — both headline and cost accuracy card
        # must read from this, NOT from design_plan.total_inr (pre-city-adjustment).
        "cost_estimate":      final_state.get("cost_estimate", {}),
        # ── FEATURE: Product Recommender ─────────────────────────────────────
        "product_suggestions": final_state.get("product_suggestions", None),
        # ── FEATURE: Contractor Network ───────────────────────────────────────
        "contractor_list":    _contractor_list,
    }

    return InsightsResponse(
        insights=final_state.get("insights", {}),
        pipeline_summary=pipeline_summary,
        agent_timings=final_state.get("agent_timings", {}),
        completed_agents=final_state.get("completed_agents", []),
        errors=final_state.get("errors", []),
    )