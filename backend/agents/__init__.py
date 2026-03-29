"""
ARKEN — Agents Package v3.0
============================
Exports all agent classes, the LangGraph multi-agent pipeline,
and the new structured LangGraph orchestrator.

New in v3.0:
  - agents/orchestrator/langgraph_orchestrator.py — structured 5-node
    LangGraph pipeline with typed contract state
    (vision_output, damage_assessment, renovation_plan,
     cost_estimate, roi_prediction, final_recommendations)
  - LangGraphOrchestrator class with run() / run_sync() / get_contract_state()
  - build_orchestrator_state() helper

Unchanged from v2.0:
  - LangGraph multi-agent pipeline (multi_agent_pipeline.py)
  - 6 specialised agent nodes (user_goal, vision_analyzer, design_planner,
    budget_estimator, roi, report)
  - Shared Pydantic state model (langgraph_state.py)
  - Memory system integration

Legacy agents (unchanged):
  - DesignPlannerAgent    (design_planner.py)
  - ROIForecastAgent      (roi_forecast.py)
  - PriceForecastAgent    (price_forecast.py)
  - ProjectCoordinatorAgent (coordinator.py)
  - VisualAssessorAgent   (visual_assessor.py)
  - RenderingAgent        (rendering.py)  <- FROZEN: image generation pipeline
"""

# ── Legacy agents (unchanged) ─────────────────────────────────────────────────
from agents.design_planner import DesignPlannerAgent
from agents.roi_forecast import ROIForecastAgent
from agents.price_forecast import PriceForecastAgent
from agents.coordinator import ProjectCoordinatorAgent

# ── New LangGraph agent nodes ─────────────────────────────────────────────────
from agents.user_goal_agent import UserGoalAgent
from agents.vision_analyzer_agent import VisionAnalyzerAgent
from agents.design_planner_node import DesignPlannerAgentNode
from agents.budget_estimator_agent import BudgetEstimatorAgent
from agents.roi_agent_node import ROIAgentNode
from agents.report_agent_node import ReportAgentNode

# ── LangGraph pipeline ────────────────────────────────────────────────────────
from agents.multi_agent_pipeline import (
    run_multi_agent_pipeline,
    run_multi_agent_pipeline_async,
    build_initial_state,
    build_multi_agent_graph,
)

# ── Structured LangGraph Orchestrator (NEW in v3.0) ───────────────────────────
from agents.orchestrator.langgraph_orchestrator import (
    LangGraphOrchestrator,
    OrchestratorState,
    build_orchestrator_graph,
    build_orchestrator_state,
    get_orchestrator,
)

# ── Shared state ──────────────────────────────────────────────────────────────
from agents.langgraph_state import (
    ARKENState,
    UserGoals,
    VisionFeatures,
    DesignPlan,
    BudgetEstimate,
    ROIOutput,
    RenovationReport,
)

__all__ = [
    # Legacy
    "DesignPlannerAgent",
    "ROIForecastAgent",
    "PriceForecastAgent",
    "ProjectCoordinatorAgent",
    # New agent nodes
    "UserGoalAgent",
    "VisionAnalyzerAgent",
    "DesignPlannerAgentNode",
    "BudgetEstimatorAgent",
    "ROIAgentNode",
    "ReportAgentNode",
    # Pipeline
    "run_multi_agent_pipeline",
    "run_multi_agent_pipeline_async",
    "build_initial_state",
    "build_multi_agent_graph",
    # Orchestrator (new)
    "LangGraphOrchestrator",
    "OrchestratorState",
    "build_orchestrator_graph",
    "build_orchestrator_state",
    "get_orchestrator",
    # State models
    "ARKENState",
    "UserGoals",
    "VisionFeatures",
    "DesignPlan",
    "BudgetEstimate",
    "ROIOutput",
    "RenovationReport",
]