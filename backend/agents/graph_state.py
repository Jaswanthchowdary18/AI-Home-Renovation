"""
ARKEN — Canonical Graph State Schema v2.1
==========================================
CHANGES from v2.0:
  - Added cv_features: Dict  — raw structured CV pipeline output
  - Added detected_style_grounded: str — style from CV (may differ from user theme)
  - Added image_specific_actions: List[Dict] — per-object/per-style actions
  - Added diy_renovation_tips: List[Dict] — from DIY dataset
  - Added dataset_grounded: bool — True when dataset was used
  All other fields and factory helpers are IDENTICAL to v2.0.

Pipeline execution order (enforced by graph edges):
  [START]
    ↓
  node_user_goal          — parse intent, load memory
    ↓
  node_vision_analysis    — CV pipeline (YOLO+CLIP+EfficientNet) + Gemini Vision
    ↓
  node_rag_retrieval      — vector-store knowledge enrichment
    ↓
  node_design_planning    — image-grounded BOQ, material plan, CPM schedule
    ↓
  node_budget_estimation  — city-adjusted cost breakdown
    ↓
  node_roi_forecasting    — XGBoost equity + rental yield model
    ↓
  node_insight_generation — image-grounded structured reasoning
    ↓
  node_report_generation  — final RenovationReport assembly
    ↓
  [END]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ARKENGraphState(TypedDict, total=False):
    """
    Shared state flowing through every node in the ARKEN LangGraph pipeline.
    """

    # ── ① Pipeline Inputs ─────────────────────────────────────────────────────
    project_id: str
    user_id: str
    original_image_b64: str
    original_image_mime: str
    original_image_bytes: bytes
    renovated_image_b64: str
    renovated_image_mime: str
    theme: str
    city: str
    budget_tier: str
    budget_inr: int
    room_type: str
    user_intent: str
    project_name: str

    # ── ② node_user_goal ──────────────────────────────────────────────────────
    user_goal: str
    user_goals: Dict[str, Any]
    parsed_intent: Dict[str, Any]
    memory_context: str
    past_budget_constraints: List[int]
    past_design_preferences: List[str]
    past_renovation_goals: List[str]

    # ── ③ node_vision_analysis ────────────────────────────────────────────────
    # CV pipeline structured output
    cv_features: Dict[str, Any]          # {room_type, detected_objects, style, materials, lighting, style_confidence, room_type_confidence, extraction_source}

    # Image-grounded fields (populated by CV + design planner)
    detected_style_grounded: str         # style from CV (may differ from user theme)
    image_specific_actions: List[Dict[str, Any]]   # per-object/style renovation actions
    diy_renovation_tips: List[Dict[str, Any]]      # from DIY dataset, matched to image
    dataset_grounded: bool               # True when interior design datasets were used

    # NEW v3.0: Task 1 — Gemini condition extraction fields
    condition_score: int                 # 0–100: 90+=new, 70-89=good, 50-69=fair, 30-49=poor, <30=very poor
    wall_condition: str                  # new|good|fair|poor|very poor
    floor_condition: str                 # new|good|fair|poor|very poor
    ceiling_condition: str               # new|good|fair|poor
    issues_detected: List[str]           # specific issues from Gemini image analysis
    renovation_scope: str                # cosmetic_only|partial|full_room|structural_plus
    high_value_upgrades: List[str]       # 2-3 high-ROI upgrades detected by Gemini
    floor_quality: str                   # basic|mid|premium
    furniture_quality: str               # basic|mid|premium

    # Canonical vision fields
    detected_features: Dict[str, Any]
    detected_style: str
    detected_style_confidence: float
    detected_objects: List[str]
    detected_changes: List[str]
    room_condition: str
    damage_assessment: Dict[str, Any]

    # Legacy-compat
    vision_features: Dict[str, Any]
    image_features: Dict[str, Any]
    room_features: Dict[str, Any]
    room_dimensions: Dict[str, Any]
    visual_style: List[str]
    wall_area_sqft: float
    floor_area_sqft: float
    layout_report: Dict[str, Any]
    style_label: str
    style_confidence: float
    explainable_recommendations: List[Dict[str, Any]]
    material_quantities: Dict[str, Any]
    material_types: List[str]

    # ── ④ node_rag_retrieval ──────────────────────────────────────────────────
    retrieved_knowledge: List[Dict[str, Any]]
    rag_context: str
    rag_budget_context: str
    rag_design_context: str
    rag_roi_context: str
    rag_sources: List[str]
    rag_retrieval_stats: Dict[str, Any]

    # ── ⑤ node_design_planning ───────────────────────────────────────────────
    design_plan: Dict[str, Any]          # now includes image_grounded, detected_style, image_specific_actions, dataset_style_examples, diy_guidance
    material_plan: Dict[str, Any]
    boq_line_items: List[Dict[str, Any]] # now includes style_grounded flag per item
    labour_estimate: int
    total_cost_estimate: int
    schedule: Dict[str, Any]
    layout_suggestions: List[str]

    # ── ⑥ node_budget_estimation ─────────────────────────────────────────────
    cost_estimate: Dict[str, Any]
    budget_estimate: Dict[str, Any]
    location_context: Dict[str, Any]
    budget_analysis: Dict[str, Any]
    material_prices: List[Dict[str, Any]]
    cost_breakdown: Dict[str, Any]
    cost_per_sqft: float
    within_budget: bool

    # ── ⑦ node_roi_forecasting ───────────────────────────────────────────────
    roi_prediction: Dict[str, Any]
    roi_output: Dict[str, Any]
    roi_pct: float
    equity_gain_inr: int
    payback_months: int
    pre_reno_value_inr: int
    post_reno_value_inr: int
    rental_yield_post_pct: float
    model_confidence: float

    # ── ⑧ node_insight_generation ────────────────────────────────────────────
    insights: Dict[str, Any]             # now includes image_grounded, dataset_grounded, visual_analysis with full CV fields
    insight_engine_output: Dict[str, Any]
    renovation_sequence: List[Dict[str, Any]]  # now starts with image-specific steps
    priority_repairs: List[Dict[str, Any]]
    budget_strategy: Dict[str, Any]

    # ── ⑨ node_report_generation ─────────────────────────────────────────────
    final_report: Dict[str, Any]
    renovation_report: Dict[str, Any]
    chat_context: str
    final_recommendations: List[Dict[str, Any]]

    # ── Rendering pass-through ────────────────────────────────────────────────
    render_url: str
    render_prompt: str
    renovated_image_url: str
    timeline: Dict[str, Any]

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    errors: List[str]
    error_details: Dict[str, Any]          # per-node structured error records for debugging
    agent_timings: Dict[str, float]
    completed_agents: List[str]
    pipeline_version: str
    inference_metrics: Dict[str, Any]


# ── State factory helpers ──────────────────────────────────────────────────────

def make_initial_state(
    *,
    project_id: str,
    user_id: str = "",
    image_bytes: bytes = b"",
    image_b64: str = "",
    image_mime: str = "image/jpeg",
    budget_inr: int,
    city: str,
    theme: str = "Modern Minimalist",
    budget_tier: str = "mid",
    room_type: str = "bedroom",
    user_intent: str = "",
    project_name: str = "Untitled Project",
) -> ARKENGraphState:
    """
    Construct the initial state dict passed to graph.invoke().
    Handles b64 ↔ bytes interop automatically.
    Identical signature to v2.0 — fully backward compatible.
    """
    import base64

    if not project_id:
        raise ValueError("project_id is required")
    if budget_inr <= 0:
        raise ValueError("budget_inr must be a positive integer")
    if not city:
        raise ValueError("city is required")

    if image_bytes and not image_b64:
        image_b64 = base64.b64encode(image_bytes).decode()
    elif image_b64 and not image_bytes:
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            image_bytes = b""

    return ARKENGraphState(
        # Inputs
        project_id=project_id,
        user_id=user_id,
        original_image_bytes=image_bytes,
        original_image_b64=image_b64,
        original_image_mime=image_mime,
        renovated_image_b64="",
        renovated_image_mime=image_mime,
        theme=theme,
        city=city,
        budget_tier=budget_tier,
        budget_inr=budget_inr,
        room_type=room_type,
        user_intent=user_intent,
        project_name=project_name,
        # Pipeline metadata
        errors=[],
        error_details={},
        agent_timings={},
        completed_agents=[],
        # Contract state — pre-initialised as empty
        user_goal="",
        detected_features={},
        retrieved_knowledge=[],
        rag_context="",
        rag_sources=[],
        rag_retrieval_stats={},
        design_plan={},
        cost_estimate={},
        roi_prediction={},
        insights={},
        final_report={},
        final_recommendations=[],
        damage_assessment={},
        within_budget=True,
        # Task 1 condition fields — populated ONLY by node_vision_analysis.
        # Defaults are None / "not_assessed" so downstream consumers know
        # no real assessment has been performed yet (not fake "fair"/"65").
        condition_score=None,
        wall_condition="not_assessed",
        floor_condition="not_assessed",
        ceiling_condition="not_assessed",
        issues_detected=[],
        renovation_scope="not_assessed",
        high_value_upgrades=[],
        floor_quality="mid",
        furniture_quality="mid",
        pipeline_version="3.0",
    )


def extract_contract_state(state: ARKENGraphState) -> Dict[str, Any]:
    """
    Return only the 8 canonical contract keys consumed by the API response serialiser.
    Identical to v2.0 — fully backward compatible.
    """
    return {
        "user_goal":           state.get("user_goal", ""),
        "uploaded_images":     [state.get("original_image_b64", "")],
        "detected_features":   state.get("detected_features", {}),
        "retrieved_knowledge": state.get("retrieved_knowledge", []),
        "design_plan":         state.get("design_plan", {}),
        "cost_estimate":       state.get("cost_estimate", {}),
        "roi_prediction":      state.get("roi_prediction", {}),
        "insights":            state.get("insights", {}),
        "final_report":        state.get("final_report", state.get("renovation_report", {})),
    }