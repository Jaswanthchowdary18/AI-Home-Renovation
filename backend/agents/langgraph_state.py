"""
ARKEN — LangGraph Multi-Agent Shared State v2.0
================================================
Pydantic-backed typed state that flows through every agent node.
All agents READ from this state and return partial updates merged by LangGraph.

v2.0 additions (additive — no breaking changes):
  - user_goal        : canonical parsed goal string (NEW)
  - detected_features: canonical vision output dict (NEW)
  - retrieved_knowledge: RAG results list (NEW)
  - rag_context      : formatted RAG string (NEW)
  - rag_sources      : provenance list (NEW)
  - rag_retrieval_stats: retrieval telemetry (NEW)
  - renovation_sequence: ordered renovation steps (NEW)
  - priority_repairs : must-do structural repairs (NEW)
  - budget_strategy  : phasing + allocation guidance (NEW)
  - final_report     : canonical report key (NEW, alias of renovation_report)
  - pipeline_version : "2.0" (NEW)

Compatible with:
  - pipeline.py, graph_pipeline.py, orchestrator.py (v1 + v2)
  - All existing API routes (analyze, forecast, chat, render)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Pydantic sub-models ───────────────────────────────────────────────────────

class UserGoals(BaseModel):
    """Parsed user renovation objectives from UserGoalAgent."""
    primary_goal: str = "personal_comfort"
    priority: str = "aesthetics"          # "roi" | "aesthetics" | "rental"
    style_preference: str = "Modern Minimalist"
    room_type: str = "bedroom"
    budget_tier: str = "mid"
    constraints: List[str] = Field(default_factory=list)
    extracted_keywords: List[str] = Field(default_factory=list)
    confidence: float = 0.7

    class Config:
        extra = "allow"


class VisionFeatures(BaseModel):
    """Structured room analysis produced by VisionAnalyzerAgent."""
    wall_treatment: str = ""
    floor_material: str = ""
    ceiling_treatment: str = ""
    furniture_items: List[str] = Field(default_factory=list)
    lighting_type: str = ""
    colour_palette: List[str] = Field(default_factory=list)
    detected_style: str = ""
    quality_tier: str = "mid"
    specific_changes: List[str] = Field(default_factory=list)
    estimated_wall_area_sqft: float = 200.0
    estimated_floor_area_sqft: float = 120.0
    room_condition: str = "fair"
    layout_score: str = "65/100"
    walkable_space: str = "45%"
    natural_light_quality: str = "moderate"
    extraction_source: str = "fallback"

    class Config:
        extra = "allow"


class DesignPlan(BaseModel):
    """Output from DesignPlannerAgent."""
    total_inr: int = 0
    material_inr: int = 0
    labour_inr: int = 0
    gst_inr: int = 0
    contingency_inr: int = 0
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: Dict[str, Any] = Field(default_factory=dict)
    supplier_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    schedule: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class BudgetEstimate(BaseModel):
    """Output from BudgetEstimatorAgent."""
    total_cost_inr: int = 0
    materials_inr: int = 0
    labour_inr: int = 0
    supervision_inr: int = 0
    contingency_inr: int = 0
    gst_inr: int = 0
    location_context: Dict[str, Any] = Field(default_factory=dict)
    budget_analysis: Dict[str, Any] = Field(default_factory=dict)
    material_prices: List[Dict[str, Any]] = Field(default_factory=list)
    cost_per_sqft: float = 0.0
    within_budget: bool = True

    class Config:
        extra = "allow"


class ROIOutput(BaseModel):
    """Output from ROIAgent."""
    roi_pct: float = 0.0
    equity_gain_inr: int = 0
    payback_months: int = 36
    pre_reno_value_inr: int = 0
    post_reno_value_inr: int = 0
    rental_yield_base_pct: float = 3.0
    rental_yield_delta: float = 0.0
    rental_yield_post_pct: float = 3.0
    model_type: str = "heuristic"
    model_confidence: float = 0.65
    city: str = ""
    city_tier: int = 2

    class Config:
        extra = "allow"
        protected_namespaces = ()


class RenovationReport(BaseModel):
    """Final structured report from ReportAgent."""
    report_version: str = "2.0"
    project_id: str = ""
    summary_headline: str = ""
    room_analysis: Dict[str, Any] = Field(default_factory=dict)
    style_analysis: Dict[str, Any] = Field(default_factory=dict)
    layout_analysis: Dict[str, Any] = Field(default_factory=dict)
    design_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    cost_estimate: Dict[str, Any] = Field(default_factory=dict)
    boq_summary: List[Dict[str, Any]] = Field(default_factory=list)
    roi_forecast: Dict[str, Any] = Field(default_factory=dict)
    renovation_timeline: Dict[str, Any] = Field(default_factory=dict)
    market_intelligence: Dict[str, Any] = Field(default_factory=dict)
    financial_outlook: Dict[str, Any] = Field(default_factory=dict)
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    render_url: str = ""
    chat_context: str = ""
    insights: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


# ── Main shared LangGraph State ───────────────────────────────────────────────

class ARKENState(BaseModel):
    """
    Single shared state object flowing through all LangGraph agent nodes.
    Each agent receives the full state and returns a partial dict of updates.
    LangGraph merges updates into this state automatically.

    v2.0: New canonical fields added (additive — all existing code unchanged).
    """

    # ── Inputs ─────────────────────────────────────────────────────────────
    project_id: str = ""
    user_id: str = ""
    original_image_b64: str = ""
    original_image_mime: str = "image/jpeg"
    original_image_bytes: bytes = b""
    renovated_image_b64: str = ""
    renovated_image_mime: str = "image/jpeg"
    theme: str = "Modern Minimalist"
    city: str = "Hyderabad"
    budget_tier: str = "mid"
    budget_inr: int = 750_000
    room_type: str = "bedroom"
    user_intent: str = ""
    project_name: str = "Untitled Project"

    # ── Agent 1: UserGoalAgent ─────────────────────────────────────────────
    user_goal: str = ""                       # v2.0 canonical goal string
    user_goals: Optional[UserGoals] = None
    parsed_intent: Dict[str, Any] = Field(default_factory=dict)

    # ── Agent 2: VisionAnalyzerAgent ──────────────────────────────────────
    detected_features: Dict[str, Any] = Field(default_factory=dict)   # v2.0 canonical
    damage_assessment: Dict[str, Any] = Field(default_factory=dict)   # v2.0 canonical
    vision_features: Optional[VisionFeatures] = None
    image_features: Dict[str, Any] = Field(default_factory=dict)      # legacy compat
    room_features: Dict[str, Any] = Field(default_factory=dict)       # rich schema
    room_dimensions: Dict[str, Any] = Field(default_factory=dict)
    detected_changes: List[str] = Field(default_factory=list)
    visual_style: List[str] = Field(default_factory=list)
    wall_area_sqft: float = 200.0
    floor_area_sqft: float = 120.0
    detected_objects: List[str] = Field(default_factory=list)
    material_quantities: Dict[str, Any] = Field(default_factory=dict)
    layout_report: Dict[str, Any] = Field(default_factory=dict)
    style_label: str = ""
    style_confidence: float = 0.0
    explainable_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    material_types: List[str] = Field(default_factory=list)

    # ── Agent 3: RAGRetrievalAgent (NEW in v2.0) ──────────────────────────
    retrieved_knowledge: List[Dict[str, Any]] = Field(default_factory=list)
    rag_context: str = ""
    rag_sources: List[str] = Field(default_factory=list)
    rag_retrieval_stats: Dict[str, Any] = Field(default_factory=dict)

    # ── Agent 4: DesignPlannerAgent ───────────────────────────────────────
    design_plan: Optional[DesignPlan] = None
    material_plan: Dict[str, Any] = Field(default_factory=dict)
    boq_line_items: List[Dict[str, Any]] = Field(default_factory=list)
    labour_estimate: int = 0
    total_cost_estimate: int = 0
    schedule: Dict[str, Any] = Field(default_factory=dict)
    layout_suggestions: List[str] = Field(default_factory=list)       # v2.0

    # ── Agent 5: BudgetEstimatorAgent ─────────────────────────────────────
    cost_estimate: Dict[str, Any] = Field(default_factory=dict)       # v2.0 canonical
    budget_estimate: Optional[BudgetEstimate] = None
    location_context: Dict[str, Any] = Field(default_factory=dict)
    budget_analysis: Dict[str, Any] = Field(default_factory=dict)
    material_prices: List[Dict[str, Any]] = Field(default_factory=list)
    cost_breakdown: Dict[str, Any] = Field(default_factory=dict)
    cost_per_sqft: float = 0.0
    within_budget: bool = True

    # ── Agent 6: ROIAgent ─────────────────────────────────────────────────
    roi_output: Optional[ROIOutput] = None
    roi_prediction: Dict[str, Any] = Field(default_factory=dict)
    roi_pct: float = 0.0
    equity_gain_inr: int = 0
    payback_months: int = 36
    pre_reno_value_inr: int = 0
    post_reno_value_inr: int = 0
    rental_yield_post_pct: float = 3.0
    model_confidence: float = 0.65

    # ── Agent 7: InsightGenerationAgent (NEW in v2.0) ─────────────────────
    renovation_sequence: List[Dict[str, Any]] = Field(default_factory=list)
    priority_repairs: List[Dict[str, Any]] = Field(default_factory=list)
    budget_strategy: Dict[str, Any] = Field(default_factory=dict)

    # ── Agent 8: ReportAgent ──────────────────────────────────────────────
    final_report: Dict[str, Any] = Field(default_factory=dict)        # v2.0 canonical
    renovation_report: Optional[RenovationReport] = None
    final_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    insights: Dict[str, Any] = Field(default_factory=dict)
    chat_context: str = ""
    insight_engine_output: Dict[str, Any] = Field(default_factory=dict)

    # ── Memory ────────────────────────────────────────────────────────────
    memory_context: str = ""
    past_budget_constraints: List[int] = Field(default_factory=list)
    past_design_preferences: List[str] = Field(default_factory=list)
    past_renovation_goals: List[str] = Field(default_factory=list)

    # ── Pipeline metadata ─────────────────────────────────────────────────
    errors: List[str] = Field(default_factory=list)
    agent_timings: Dict[str, float] = Field(default_factory=dict)
    completed_agents: List[str] = Field(default_factory=list)
    inference_metrics: Dict[str, Any] = Field(default_factory=dict)
    pipeline_version: str = "2.0"                                     # v2.0

    # Rendering (pass-through — NEVER modified by pipeline agents)
    render_url: str = ""
    render_prompt: str = ""
    renovated_image_url: str = ""
    timeline: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def to_legacy_pipeline_state(self) -> Dict[str, Any]:
        """Convert to the flat PipelineState TypedDict used by pipeline.py."""
        return {
            "project_id": self.project_id,
            "original_image_b64": self.original_image_b64,
            "original_image_mime": self.original_image_mime,
            "renovated_image_b64": self.renovated_image_b64,
            "renovated_image_mime": self.renovated_image_mime,
            "theme": self.theme,
            "city": self.city,
            "budget_tier": self.budget_tier,
            "budget_inr": self.budget_inr,
            "room_type": self.room_type,
            "image_features": self.image_features,
            "room_features": self.room_features,
            "detected_changes": self.detected_changes,
            "room_dimensions": self.room_dimensions,
            "visual_style": self.visual_style,
            "material_plan": self.material_plan,
            "boq_line_items": self.boq_line_items,
            "labour_estimate": self.labour_estimate,
            "total_cost_estimate": self.total_cost_estimate,
            "design_plan": self.design_plan.model_dump() if self.design_plan else {},
            "schedule": self.schedule,
            "roi_prediction": self.roi_prediction,
            "payback_months": self.payback_months,
            "equity_gain_inr": self.equity_gain_inr,
            "location_context": self.location_context,
            "budget_analysis": self.budget_analysis,
            "material_prices": self.material_prices,
            "insights": self.insights,
            "chat_context": self.chat_context,
            "insight_engine_output": self.insight_engine_output,
            "errors": self.errors,
            "agent_timings": self.agent_timings,
            "completed_agents": self.completed_agents,
            # v2.0 additions
            "user_goal": self.user_goal,
            "detected_features": self.detected_features,
            "retrieved_knowledge": self.retrieved_knowledge,
            "rag_context": self.rag_context,
            "cost_estimate": self.cost_estimate,
            "renovation_sequence": self.renovation_sequence,
            "priority_repairs": self.priority_repairs,
            "budget_strategy": self.budget_strategy,
            "final_report": self.final_report,
            "final_recommendations": self.final_recommendations,
        }

    def merge_legacy(self, legacy: Dict[str, Any]) -> "ARKENState":
        """Return new state merged with flat dict from legacy agents."""
        data = self.model_dump()
        for k, v in legacy.items():
            if k in data and v is not None:
                data[k] = v
        return ARKENState(**data)