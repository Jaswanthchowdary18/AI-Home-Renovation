"""
ARKEN — LLM Service v2.0
==========================
SAVE AS: backend/services/llm.py — REPLACE existing

v2.0 Changes over v1.0 (PROBLEM 2 FIX — project-grounded chat):
  1. build_project_context_prompt() added: converts the full pipeline
     result into a structured, numbers-first context string the model
     reads before answering any question.
  2. GROUNDED_SYSTEM_PROMPT added: replaces the generic system prompt
     with one that explicitly instructs the model to use project data,
     cite Indian standards, and quote exact numbers.
  3. LLMService.chat() and chat_stream() accept an optional
     project_context kwarg — prepended to the system instruction when
     present.

CRITICAL: rendering.py and IMAGE_MODELS are NOT touched.
"""

import logging
from typing import AsyncGenerator, Dict, Any, List, Optional

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

GEMINI_TEXT_MODEL = "gemini-2.5-flash"


def _client() -> genai.Client:
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in backend/.env")
    return genai.Client(api_key=settings.GOOGLE_API_KEY.get_secret_value())


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 2 FIX: Grounded system prompt
# ─────────────────────────────────────────────────────────────────────────────

GROUNDED_SYSTEM_PROMPT = """You are ARKEN, an AI renovation advisor for Indian homes. \
You MUST use the project context data provided to give specific, numbers-backed answers. \
Never give generic advice when specific project data is available.

MANDATORY RULES:
1. When asked about ROI, always quote the exact predicted ROI% and equity gain from the \
context, then explain what factors drive it.
2. When asked about budget, reference the exact BOQ line items and costs from the context.
3. Always end financial advice with: "This estimate is based on {data_source}. \
For a binding quote, get 3 contractor quotes from UrbanCompany, Sulekha, or JustDial."
4. All prices must be in INR (₹). All standards referenced must be Indian \
(BIS/IS codes, not American NEC/NFPA).
5. When recommending brands, only recommend brands available in India \
(Asian Paints, Kajaria, Jaquar, Havells, Legrand, Pidilite, Astral, etc.).
6. If the user asks about timeline, quote the CPM schedule data from context \
(best case, realistic, worst case dates if available).
7. When discussing materials, reference the relevant BIS/IS standard where applicable.
8. Complete every answer fully — do not cut responses short."""


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 2 FIX: build_project_context_prompt()
# ─────────────────────────────────────────────────────────────────────────────

def build_project_context_prompt(pipeline_result: Dict[str, Any]) -> str:
    """
    Convert the full pipeline result into a structured context string.

    This function is called before every chat message. The returned string
    is prepended to the system instruction so the model answers with
    project-specific data rather than generic advice.

    Args:
        pipeline_result: The final state dict from run_multi_agent_pipeline()
                         or the pipeline_summary from /chat/insights.

    Returns:
        A structured context string for injection into the system prompt.
    """
    if not pipeline_result:
        return ""

    # ── Extract fields defensively ────────────────────────────────────────────
    city         = str(pipeline_result.get("city", ""))
    room_type    = str(pipeline_result.get("room_type", "")).replace("_", " ").title()
    budget_tier  = str(pipeline_result.get("budget_tier", "mid"))
    budget_inr   = int(pipeline_result.get("budget_inr", 0))
    style_label  = str(pipeline_result.get("theme", pipeline_result.get("style_label", "")))
    area_sqft    = pipeline_result.get("floor_area_sqft") or pipeline_result.get("area_sqft") or 0

    # City tier
    _CITY_TIER = {
        "Mumbai": 1, "Delhi NCR": 1, "Bangalore": 1, "Hyderabad": 1,
        "Chennai": 1, "Pune": 1, "Kolkata": 1, "Ahmedabad": 2,
    }
    tier = _CITY_TIER.get(city, 2)

    # ROI data
    roi_raw     = pipeline_result.get("roi_prediction") or pipeline_result.get("roi") or {}
    roi_pct     = roi_raw.get("roi_pct", 0)
    confidence  = roi_raw.get("model_confidence", 0)
    pre_value   = roi_raw.get("pre_reno_value_inr", 0)
    post_value  = roi_raw.get("post_reno_value_inr", 0)
    equity_gain = roi_raw.get("equity_gain_inr", 0)
    payback     = roi_raw.get("payback_months", 0)
    roi_ci_low  = roi_raw.get("roi_ci_low", 0)
    roi_ci_high = roi_raw.get("roi_ci_high", 0)
    model_type  = roi_raw.get("model_type", "heuristic")
    data_source = roi_raw.get("data_source", roi_raw.get("data_transparency", "ARKEN model"))

    # Confidence label
    conf_pct = int(confidence * 100) if confidence else 0
    conf_level = "High" if confidence >= 0.80 else "Medium" if confidence >= 0.60 else "Low"

    # Budget / BOQ data
    budget_raw   = pipeline_result.get("budget_estimate") or pipeline_result.get("design") or {}
    reno_cost    = budget_raw.get("total_cost_inr") or budget_raw.get("total_inr") or budget_inr
    mat_cost     = budget_raw.get("materials_inr", 0)
    labour_cost  = budget_raw.get("labour_inr", 0)
    gst_cost     = budget_raw.get("gst_inr", 0)
    contingency  = budget_raw.get("contingency_inr", 0)

    # Schedule data
    schedule_raw  = pipeline_result.get("schedule") or {}
    total_days    = schedule_raw.get("total_days", 0)
    start_date    = schedule_raw.get("start_date", "")
    best_end      = schedule_raw.get("best_case_end", schedule_raw.get("projected_end", ""))
    realistic_end = schedule_raw.get("realistic_end_date", best_end)
    worst_end     = schedule_raw.get("worst_case_end", "")
    cp_days       = schedule_raw.get("critical_path_days", total_days)

    # Top priorities from vision analysis
    recs = pipeline_result.get("explainable_recommendations") or []
    priorities_text = "; ".join(
        str(r.get("title") or r.get("action", ""))
        for r in recs[:3]
        if isinstance(r, dict) and (r.get("title") or r.get("action"))
    ) or "See BOQ for full list"

    # Key risks
    risks_raw = schedule_raw.get("risks") or pipeline_result.get("risk_flags") or []
    risks_text = "; ".join(
        str(r.get("factor") or r) for r in risks_raw[:3]
        if isinstance(r, (dict, str))
    ) or "Standard renovation risks apply"

    # Material price trend
    prices = pipeline_result.get("material_prices") or []
    urgent = [
        p.get("display_name", p.get("material_key", ""))
        for p in prices
        if isinstance(p, dict) and p.get("pct_change_90d", 0) > 4
    ][:3]
    price_trend_text = (
        f"{', '.join(urgent)} trending up >4% in 90 days — consider early procurement"
        if urgent else "Material prices broadly stable"
    )

    # ── Format the context string ─────────────────────────────────────────────
    lines = ["CURRENT PROJECT CONTEXT (use this data to answer questions):"]

    if city:
        lines.append(f"City: {city} (Tier {tier} market)")
    if room_type:
        area_str = f", estimated {area_sqft:.0f} sqft" if area_sqft else ""
        lines.append(f"Room: {room_type}{area_str}")
    if budget_inr:
        lines.append(f"Budget: ₹{budget_inr:,} ({budget_tier} tier)")
    if style_label:
        lines.append(f"Style: {style_label}")
    if reno_cost:
        lines.append(f"Estimated renovation cost: ₹{reno_cost:,} (±15%)")
    if roi_pct:
        ci_str = f" | Range: {roi_ci_low:.1f}% – {roi_ci_high:.1f}%" if roi_ci_low and roi_ci_high else ""
        lines.append(f"Predicted ROI: {roi_pct:.1f}%{ci_str} (confidence: {conf_level} {conf_pct}%)")
    if pre_value:
        lines.append(f"Property value before renovation: ₹{pre_value:,}")
    if post_value:
        lines.append(f"Estimated value after renovation: ₹{post_value:,}")
    if equity_gain:
        lines.append(f"Net equity gain: ₹{equity_gain:,}")
    if payback:
        lines.append(f"Payback period: {payback} months")

    # BOQ breakdown
    if mat_cost or labour_cost:
        lines.append(
            f"BOQ breakdown: Materials ₹{mat_cost:,} | Labour ₹{labour_cost:,} | "
            f"GST ₹{gst_cost:,} | Contingency ₹{contingency:,}"
        )

    # Schedule
    if total_days:
        sched_str = f"CPM schedule: {total_days} working days"
        if start_date and best_end:
            sched_str += f" ({start_date} → best case {best_end}"
            if realistic_end and realistic_end != best_end:
                sched_str += f", expected {realistic_end}"
            if worst_end:
                sched_str += f", worst case {worst_end}"
            sched_str += ")"
        lines.append(sched_str)
        if cp_days and cp_days != total_days:
            lines.append(f"Critical path: {cp_days} days minimum (no float)")

    lines.append(f"Top 3 renovation priorities from image analysis: {priorities_text}")
    lines.append(f"Key risks: {risks_text}")
    lines.append(f"Material price trend: {price_trend_text}")

    lines.append(
        f"DATA SOURCES: Housing data: 32,963 real Indian property transactions. "
        f"ROI model: {model_type}. Prices: Verified Q1 2026 Indian market."
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _build_history(messages: List[dict]) -> tuple:
    """Convert OpenAI-style messages → (system_prompt, history, last_user_msg)."""
    system, history, last = "", [], ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            system = content
        elif role == "user":
            history.append(types.Content(role="user", parts=[types.Part(text=content)]))
            last = content
        elif role == "assistant":
            history.append(types.Content(role="model", parts=[types.Part(text=content)]))
    return system, history[:-1] if len(history) > 1 else [], last


# ─────────────────────────────────────────────────────────────────────────────
# LLMService
# ─────────────────────────────────────────────────────────────────────────────

class LLMService:
    """
    gemini-2.5-flash — text chat, streaming, and vision Q&A.

    v2.0: accepts optional project_context kwarg which is injected
    into the system instruction to ground responses in project data.
    """

    async def chat(
        self,
        messages: List[dict],
        project_context: Optional[str] = None,
    ) -> str:
        client = _client()
        system, history, last = _build_history(messages)

        # PROBLEM 2 FIX: inject project context into system instruction
        full_system = GROUNDED_SYSTEM_PROMPT
        if project_context:
            full_system = f"{GROUNDED_SYSTEM_PROMPT}\n\n{project_context}"
        if system:
            full_system = f"{full_system}\n\n{system}"

        session = client.chats.create(
            model=GEMINI_TEXT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=full_system or None,
                temperature=0.4,
                max_output_tokens=1500,
            ),
            history=history,
        )
        response = session.send_message(last)
        return response.text

    async def chat_stream(
        self,
        messages: List[dict],
        project_context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        client = _client()
        system, history, last = _build_history(messages)

        full_system = GROUNDED_SYSTEM_PROMPT
        if project_context:
            full_system = f"{GROUNDED_SYSTEM_PROMPT}\n\n{project_context}"
        if system:
            full_system = f"{full_system}\n\n{system}"

        session = client.chats.create(
            model=GEMINI_TEXT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=full_system or None,
                temperature=0.4,
                max_output_tokens=1500,
            ),
            history=history,
        )
        for chunk in session.send_message_stream(last):
            if chunk.text:
                yield chunk.text

    async def vision_chat(
        self,
        system_prompt: str,
        image_b64: str,
        image_mime: str,
        user_text: str,
        history: Optional[list] = None,
        project_context: Optional[str] = None,
    ) -> str:
        import base64
        client = _client()
        parts = [
            types.Part(inline_data=types.Blob(
                mime_type=image_mime,
                data=base64.b64decode(image_b64),
            )),
        ]
        if history:
            ctx = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
            parts.append(types.Part(text=f"Previous conversation:\n{ctx}\n\nCurrent question:"))
        parts.append(types.Part(text=user_text))

        full_system = GROUNDED_SYSTEM_PROMPT
        if project_context:
            full_system = f"{GROUNDED_SYSTEM_PROMPT}\n\n{project_context}"
        if system_prompt:
            full_system = f"{full_system}\n\n{system_prompt}"

        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=full_system or None,
                temperature=0.4,
                max_output_tokens=1000,
            ),
        )
        return response.text


llm_service = LLMService()