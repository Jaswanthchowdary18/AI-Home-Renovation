"""
ARKEN — Insight Engine Data Models v2.0
=========================================
Typed models for the structured insight output produced by InsightEngine.

v2.0 additions:
  - InsightOutput carries analytics extensions (derived_insights, market_benchmark, etc.)
  - to_report_dict() v2 includes all analytics fields for PDF generator
  - All fields have safe defaults — no field is ever None or missing
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PriorityRepair(BaseModel):
    """A single repair or renovation action, ordered by urgency."""
    rank: int = 0
    action: str = ""
    category: str = ""          # structural | cosmetic | mechanical | finishing
    urgency: str = "medium"     # critical | high | medium | low
    estimated_cost_inr: int = 0
    estimated_cost_label: str = ""
    impact_score: float = 0.0   # 0–10: combined ROI + livability impact
    roi_contribution_pct: float = 0.0
    reasoning: str = ""
    source_signals: List[str] = Field(default_factory=list)


class BudgetStrategy(BaseModel):
    """Structured budget allocation recommendation."""
    strategy_name: str = ""
    description: str = ""
    allocation: Dict[str, int] = Field(default_factory=dict)
    allocation_pct: Dict[str, float] = Field(default_factory=dict)
    total_recommended_inr: int = 0
    within_budget: bool = True
    budget_gap_inr: int = 0
    key_recommendation: str = ""
    avoid_items: List[str] = Field(default_factory=list)


class ROIInsight(BaseModel):
    """Computed ROI and value-increase projections."""
    roi_score: float = 0.0
    roi_percentage: str = "0%"
    expected_value_increase: str = "₹0"
    equity_gain_inr: int = 0
    payback_period: str = ""
    cost_effectiveness_score: float = 0.0
    cost_per_sqft: float = 0.0
    value_per_rupee: float = 0.0
    rental_yield_improvement: str = ""
    confidence: float = 0.65
    model_type: str = "heuristic"
    interpretation: str = ""


class RenovationSequence(BaseModel):
    """An ordered step in the renovation execution plan."""
    step: int = 0
    phase: str = ""
    actions: List[str] = Field(default_factory=list)
    duration_days: int = 0
    dependencies: List[str] = Field(default_factory=list)
    can_parallel: bool = False
    cost_inr: int = 0
    rationale: str = ""


class InsightOutput(BaseModel):
    """
    Complete structured output from the Insight Engine v2.0.
    Backward compatible with v1.0 — all new fields are optional with safe defaults.
    """
    # Core scores (v1.0)
    renovation_priority_score: float = 0.0
    cost_effectiveness_score: float = 0.0
    roi_score: float = 0.0
    overall_insight_score: float = 0.0

    # Structured outputs (v1.0)
    priority_repairs: List[PriorityRepair] = Field(default_factory=list)
    budget_strategy: BudgetStrategy = Field(default_factory=BudgetStrategy)
    roi_insight: ROIInsight = Field(default_factory=ROIInsight)
    renovation_sequence: List[RenovationSequence] = Field(default_factory=list)

    # Summaries (v1.0)
    expected_value_increase: str = "₹0"
    summary: str = ""
    key_wins: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)

    # Metadata (v1.0)
    insight_version: str = "2.0"
    data_sources: List[str] = Field(default_factory=list)
    confidence_overall: float = 0.65

    # ── v2.0 Analytics extensions ──────────────────────────────────────────
    # All optional for backward compatibility with v1.0 callers

    # Data-backed, source-attributed facts from RENOVATION_FACTS KB
    derived_insights: List[Dict[str, Any]] = Field(default_factory=list)

    # City/room benchmark comparison
    market_benchmark: Dict[str, Any] = Field(default_factory=dict)

    # Optimal budget allocation per category
    optimal_budget_allocation: List[Dict[str, Any]] = Field(default_factory=list)

    # Scored BOQ decisions
    decision_scores: List[Dict[str, Any]] = Field(default_factory=list)

    # Quick-win opportunities
    quick_wins: List[Dict[str, Any]] = Field(default_factory=list)

    # Material price signals from PriceForecastAgent
    material_price_signals: List[Dict[str, Any]] = Field(default_factory=list)

    def to_report_dict(self) -> Dict[str, Any]:
        """Flatten to the dict format expected by report_generator.py."""
        base = {
            "priority_repairs": [r.model_dump() for r in self.priority_repairs],
            "budget_strategy": self.budget_strategy.model_dump(),
            "expected_value_increase": self.expected_value_increase,
            "roi_score": f"{self.roi_score:.1f}/100",
            "renovation_sequence": [s.model_dump() for s in self.renovation_sequence],
            "renovation_priority_score": round(self.renovation_priority_score, 1),
            "cost_effectiveness_score": round(self.cost_effectiveness_score, 1),
            "overall_insight_score": round(self.overall_insight_score, 1),
            "roi_insight": self.roi_insight.model_dump(),
            "summary": self.summary,
            "key_wins": self.key_wins,
            "risk_flags": self.risk_flags,
            "confidence_overall": self.confidence_overall,
            "insight_version": self.insight_version,
        }
        # v2.0 analytics fields (included when available)
        if self.derived_insights:
            base["derived_insights"] = self.derived_insights
        if self.market_benchmark:
            base["market_benchmark"] = self.market_benchmark
        if self.optimal_budget_allocation:
            base["optimal_budget_allocation"] = self.optimal_budget_allocation
        if self.decision_scores:
            base["decision_scores"] = self.decision_scores
        if self.quick_wins:
            base["quick_wins"] = self.quick_wins
        if self.material_price_signals:
            base["material_price_signals"] = self.material_price_signals
        return base

    class Config:
        extra = "allow"


class RenovationInsight(BaseModel):
    """
    Lightweight facade returned by InsightEngine.compute().
    Mirrors the JSON contract described in the spec.
    """
    priority_repairs: List[Dict[str, Any]] = Field(default_factory=list)
    budget_strategy: str = ""
    expected_value_increase: str = "₹0"
    roi_score: str = "0/100"
    renovation_sequence: List[Dict[str, Any]] = Field(default_factory=list)

    renovation_priority_score: float = 0.0
    cost_effectiveness_score: float = 0.0
    overall_insight_score: float = 0.0
    key_wins: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    summary: str = ""

    class Config:
        extra = "allow"