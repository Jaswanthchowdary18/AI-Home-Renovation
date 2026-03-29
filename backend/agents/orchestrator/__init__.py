"""
ARKEN — Orchestrator Package v2.0
====================================
LangGraph-based 8-node deterministic orchestration layer.

Exports the primary orchestrator interface and all node functions.
All underlying agent implementations remain unchanged.

Pipeline nodes (in execution order):
  1. node_user_goal
  2. node_vision_analysis
  3. node_rag_retrieval          ← NEW in v2.0
  4. node_design_planning
  5. node_budget_estimation
  6. node_roi_forecasting
  7. node_insight_generation     ← NEW in v2.0
  8. node_report_generation
"""

from agents.orchestrator.langgraph_orchestrator import (
    LangGraphOrchestrator,
    OrchestratorState,           # backward-compat alias for ARKENGraphState
    build_orchestrator_graph,
    build_orchestrator_state,
    get_orchestrator,
    # Node functions (for direct use or testing)
    node_user_goal,
    node_vision_analysis,
    node_rag_retrieval,          # NEW
    node_design_planning,
    node_budget_estimation,
    node_roi_forecasting,
    node_insight_generation,     # NEW
    node_report_generation,
    # v1.0 node aliases for backward compatibility
    node_renovation_planning,
    node_cost_estimation,
    node_roi_prediction,
)

__all__ = [
    # Class + factories
    "LangGraphOrchestrator",
    "OrchestratorState",
    "build_orchestrator_graph",
    "build_orchestrator_state",
    "get_orchestrator",
    # v2.0 nodes
    "node_user_goal",
    "node_vision_analysis",
    "node_rag_retrieval",
    "node_design_planning",
    "node_budget_estimation",
    "node_roi_forecasting",
    "node_insight_generation",
    "node_report_generation",
    # v1.0 compat aliases
    "node_renovation_planning",
    "node_cost_estimation",
    "node_roi_prediction",
]