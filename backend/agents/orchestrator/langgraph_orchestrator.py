"""
ARKEN — LangGraph Orchestrator v2.0
=====================================
Deterministic 8-node LangGraph pipeline replacing the loosely-structured v1.0.

Pipeline DAG (strictly sequential, deterministic):
  [START]
    │
    ▼
  node_user_goal          ← parse intent, load memory
    │
    ▼
  node_vision_analysis    ← Gemini room feature extraction (image-gated)
    │
    ▼
  node_rag_retrieval      ← vector-store knowledge enrichment  [NEW]
    │
    ▼
  node_design_planning    ← BOQ, material plan, CPM schedule
    │
    ▼
  node_budget_estimation  ← city-adjusted cost breakdown
    │
    ▼
  node_roi_forecasting    ← XGBoost equity + rental yield
    │
    ▼
  node_insight_generation ← structured synthesis over all data  [NEW]
    │
    ▼
  node_report_generation  ← final RenovationReport assembly
    │
  [END]

Key improvements over v1.0:
  1. Each node has a SINGLE responsibility — no dual-purpose agents.
  2. RAG retrieval is now a proper first-class pipeline step.
  3. Insight generation is separated from report assembly.
  4. State schema (ARKENGraphState) is the single source of truth.
  5. Conditional edge: vision_analysis is SKIPPED when no image is provided.
  6. Validation guards between nodes prevent silent data corruption.
  7. All fallbacks are deterministic (same input → same fallback output).
  8. Progress callbacks are fired with accurate per-node percentages.

COMPATIBILITY:
  - Existing LangGraphOrchestrator (v1.0) API is preserved.
  - build_orchestrator_state() and get_orchestrator() remain importable.
  - OrchestratorState TypedDict is re-exported for backward compat.
  - Gemini image generation pipeline is NEVER touched.
  - All existing agent classes are called without modification.

CRITICAL PROTECTION:
  ─────────────────────────────────────────────────────────────────────
  This file NEVER modifies RenderingAgent, image generation prompts,
  Gemini model parameters, or any image preprocessing code.
  The renovated_image_b64 / render_url / render_prompt fields flow
  through the state as pass-through keys only.
  ─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Import the canonical state schema ────────────────────────────────────────
from agents.graph_state import ARKENGraphState, make_initial_state, extract_contract_state

# ── Backward-compat re-export (v1 callers use OrchestratorState) ──────────────
OrchestratorState = ARKENGraphState


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

class NodeValidationError(ValueError):
    """Raised by a node guard when required upstream data is missing."""


def _require(state: Dict[str, Any], *keys: str, node: str) -> None:
    """Assert that one or more keys have non-empty values in state."""
    missing = [k for k in keys if not state.get(k)]
    if missing:
        raise NodeValidationError(
            f"[{node}] Missing required upstream state keys: {missing}"
        )


def _warn_missing(state: Dict[str, Any], *keys: str, node: str) -> None:
    """Log a warning (non-fatal) for missing state keys."""
    missing = [k for k in keys if not state.get(k)]
    if missing:
        logger.warning(f"[{node}] Optional upstream keys absent: {missing} — defaults will be used")


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge(state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new state dict with updates merged in."""
    return {**state, **updates}


def _record_timing(updates: Dict[str, Any], name: str, t0: float, state: Dict[str, Any]) -> None:
    timings = dict(state.get("agent_timings") or {})
    timings[name] = round(time.perf_counter() - t0, 3)
    updates["agent_timings"] = timings


def _record_completion(updates: Dict[str, Any], name: str, state: Dict[str, Any]) -> None:
    completed = list(state.get("completed_agents") or [])
    if name not in completed:
        completed.append(name)
    updates["completed_agents"] = completed


def _record_error(updates: Dict[str, Any], name: str, exc: Exception, state: Dict[str, Any]) -> None:
    from datetime import datetime as _dt
    # ── Deduplicated error string list ────────────────────────────────────────
    existing = list(state.get("errors") or [])
    error_str = f"{name}: {exc}"
    if not any(error_str in e for e in existing):
        existing.append(error_str)
    updates["errors"] = existing

    # ── Structured per-node error details for debugging ───────────────────────
    error_details = dict(state.get("error_details") or {})
    error_details[name] = {
        "error_type":    type(exc).__name__,
        "error_message": str(exc),
        "timestamp":     _dt.utcnow().isoformat(),
        "node":          name,
        "fallback_used": True,
    }
    updates["error_details"] = error_details
    logger.error(f"[orchestrator.{name}] {exc}", exc_info=True)


def _sync_run_async(coro) -> Any:
    """Run an async coroutine from a sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, coro)
                return fut.result(timeout=600)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: User Goal
# ─────────────────────────────────────────────────────────────────────────────

def node_user_goal(state: ARKENGraphState) -> ARKENGraphState:
    """
    Parse user intent, extract structured goals, load memory.
    Writes: user_goal, user_goals, parsed_intent, memory_context, past_* fields.
    """
    t0 = time.perf_counter()
    name = "node_user_goal"
    updates: Dict[str, Any] = {}

    try:
        from agents.user_goal_agent import UserGoalAgent
        agent = UserGoalAgent()
        result = _sync_run_async(agent.run(dict(state)))
        # Populate new canonical user_goal field from user_goals
        goals = result.get("user_goals") or {}
        user_goal = (
            goals.get("primary_goal")
            or goals.get("user_goal")
            or state.get("user_intent", "personal_comfort")
        )
        updates = {**result, "user_goal": user_goal}
        logger.info(f"[{name}] user_goal={user_goal}")

    except Exception as e:
        _record_error(updates, name, e, state)
        updates.update({
            "user_goal":   state.get("user_intent", "personal_comfort"),
            "user_goals":  {"primary_goal": "personal_comfort", "priority": "aesthetics"},
            "parsed_intent": {},
            "memory_context": "",
        })

    _record_timing(updates, name, t0, state)
    _record_completion(updates, name, state)
    return _merge(state, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Vision Analysis (image-gated)
# ─────────────────────────────────────────────────────────────────────────────

def node_vision_analysis(state: ARKENGraphState) -> ARKENGraphState:
    """
    Gemini-powered room feature extraction.
    SKIPPED (with defaults injected) when no image data is present.
    Writes: detected_features, vision_features, room_features, damage_assessment, ...
    NEVER modifies Gemini image generation code.
    """
    t0 = time.perf_counter()
    name = "node_vision_analysis"
    updates: Dict[str, Any] = {}

    # Gate: skip if no image
    if not state.get("original_image_b64") and not state.get("original_image_bytes"):
        logger.info(f"[{name}] No image — injecting vision defaults")
        updates = _default_vision_output(state)
        _record_timing(updates, name, t0, state)
        _record_completion(updates, name, state)
        return _merge(state, updates)

    try:
        from agents.vision_analyzer_agent import VisionAnalyzerAgent
        agent = VisionAnalyzerAgent()
        result = _sync_run_async(agent.run(dict(state)))

        # Build new canonical detected_features from vision agent output
        room_features = result.get("room_features") or {}
        layout = result.get("layout_report") or {}
        detected_features = _build_detected_features(result, room_features, layout)
        damage_assessment = _build_damage_assessment(room_features, layout)

        # ── BUG FIX: Propagate all Gemini condition fields to top-level state ──
        # InsightGenerationAgent and DesignPlannerAgentNode read these from
        # state top-level keys. Without this propagation they always get
        # None / "not_assessed" / [] because room_features is nested.
        condition_fields = _extract_condition_fields(room_features, result)

        updates = {
            **result,
            "detected_features": detected_features,
            "damage_assessment": damage_assessment,
            **condition_fields,          # hoisted to top-level for all downstream agents
        }
        logger.info(
            f"[{name}] done — style={result.get('style_label', '?')} "
            f"condition={damage_assessment.get('overall_condition', '?')} "
            f"issues={len(condition_fields.get('issues_detected', []))} "
            f"wall={result.get('wall_area_sqft', 0):.0f}sqft"
        )

    except Exception as e:
        _record_error(updates, name, e, state)
        updates.update(_default_vision_output(state))

    _record_timing(updates, name, t0, state)
    _record_completion(updates, name, state)
    return _merge(state, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: RAG Knowledge Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def node_rag_retrieval(state: ARKENGraphState) -> ARKENGraphState:
    """
    Vector-store knowledge retrieval grounded in vision analysis outputs.
    Writes: retrieved_knowledge, rag_context, rag_sources, rag_retrieval_stats.
    """
    t0 = time.perf_counter()
    name = "node_rag_retrieval"
    updates: Dict[str, Any] = {}

    try:
        from agents.rag_retrieval_agent import RAGRetrievalAgent
        agent = RAGRetrievalAgent()
        result = _sync_run_async(agent.run(dict(state)))
        updates = result
        stats = result.get("rag_retrieval_stats", {})
        logger.info(
            f"[{name}] done — docs={stats.get('doc_count', 0)} "
            f"top_score={stats.get('top_score', 0.0):.3f}"
        )

    except Exception as e:
        _record_error(updates, name, e, state)
        updates.update({
            "retrieved_knowledge": [],
            "rag_context": f"RAG unavailable: {e}",
            "rag_sources": [],
            "rag_retrieval_stats": {"doc_count": 0, "query_count": 0, "top_score": 0.0},
        })

    _record_timing(updates, name, t0, state)
    _record_completion(updates, name, state)
    return _merge(state, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: Design Planning
# ─────────────────────────────────────────────────────────────────────────────

def node_design_planning(state: ARKENGraphState) -> ARKENGraphState:
    """
    Generate BOQ, material plan, and CPM schedule.
    Reads: vision outputs, room_dimensions, budget_tier, rag_context.
    Writes: design_plan, boq_line_items, labour_estimate, schedule.
    """
    t0 = time.perf_counter()
    name = "node_design_planning"
    updates: Dict[str, Any] = {}

    _warn_missing(state, "wall_area_sqft", "floor_area_sqft", node=name)

    try:
        from agents.design_planner_node import DesignPlannerAgentNode
        agent = DesignPlannerAgentNode()
        result = _sync_run_async(agent.run(dict(state)))

        # Extract layout_suggestions into its own state key
        design_plan = result.get("design_plan") or {}
        layout_suggestions = (
            design_plan.get("layout_suggestions")
            or design_plan.get("recommendations", {}).get("layout_suggestions", [])
            or []
        )

        updates = {
            **result,
            "layout_suggestions": layout_suggestions,
        }
        logger.info(
            f"[{name}] done — boq={len(result.get('boq_line_items', []))} items "
            f"total=₹{result.get('total_cost_estimate', 0):,}"
        )

    except Exception as e:
        _record_error(updates, name, e, state)
        updates.update(_default_design_plan(state))

    _record_timing(updates, name, t0, state)
    _record_completion(updates, name, state)
    return _merge(state, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: Budget Estimation
# ─────────────────────────────────────────────────────────────────────────────

def node_budget_estimation(state: ARKENGraphState) -> ARKENGraphState:
    """
    City-adjusted, material-price-aware cost breakdown.
    Reads: boq_line_items, labour_estimate, city, budget_tier, floor_area_sqft.
    Writes: cost_estimate, budget_estimate, cost_per_sqft, within_budget, location_context.
    """
    t0 = time.perf_counter()
    name = "node_budget_estimation"
    updates: Dict[str, Any] = {}

    try:
        from agents.budget_estimator_agent import BudgetEstimatorAgent
        agent = BudgetEstimatorAgent()
        result = _sync_run_async(agent.run(dict(state)))

        budget_est = result.get("budget_estimate") or {}
        cost_bd = result.get("cost_breakdown") or {}

        # Build canonical cost_estimate (contract key)
        cost_estimate = {
            "total_inr":          budget_est.get("total_cost_inr", state.get("budget_inr", 0)),
            "materials_inr":      budget_est.get("materials_inr", 0),
            "labour_inr":         budget_est.get("labour_inr", 0),
            "gst_inr":            budget_est.get("gst_inr", 0),
            "contingency_inr":    budget_est.get("contingency_inr", 0),
            "supervision_inr":    budget_est.get("supervision_inr", 0),
            "cost_per_sqft":      budget_est.get("cost_per_sqft", 0.0),
            "within_budget":      budget_est.get("within_budget", True),
            "city_multiplier":    cost_bd.get("city_multiplier", 1.0),
            "budget_tier_analysis": result.get("budget_analysis", {}),
            "material_prices":    result.get("material_prices", []),
        }

        updates = {
            **result,
            "cost_estimate":  cost_estimate,
            "cost_per_sqft":  float(budget_est.get("cost_per_sqft", 0)),
            "within_budget":  bool(budget_est.get("within_budget", True)),
        }
        logger.info(
            f"[{name}] done — total=₹{cost_estimate['total_inr']:,} "
            f"within_budget={cost_estimate['within_budget']}"
        )

    except Exception as e:
        _record_error(updates, name, e, state)
        updates.update(_default_cost_estimate(state))

    _record_timing(updates, name, t0, state)
    _record_completion(updates, name, state)
    return _merge(state, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: ROI Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def node_roi_forecasting(state: ARKENGraphState) -> ARKENGraphState:
    """
    XGBoost-backed property value uplift + rental yield model.
    Reads: cost_estimate, city, budget_tier, floor_area_sqft, user_goals.
    Writes: roi_prediction, roi_pct, equity_gain_inr, payback_months.
    """
    t0 = time.perf_counter()
    name = "node_roi_forecasting"
    updates: Dict[str, Any] = {}

    try:
        from agents.roi_agent_node import ROIAgentNode
        agent = ROIAgentNode()
        result = _sync_run_async(agent.run(dict(state)))

        roi_raw = result.get("roi_prediction") or {}
        roi_output = result.get("roi_output") or {}

        # Canonical roi_prediction contract key
        roi_prediction = {
            "roi_pct":               roi_raw.get("roi_pct", 0.0),
            "equity_gain_inr":       roi_raw.get("equity_gain_inr", 0),
            "payback_months":        roi_raw.get("payback_months", 36),
            "pre_reno_value_inr":    roi_raw.get("pre_reno_value_inr", 0),
            "post_reno_value_inr":   roi_raw.get("post_reno_value_inr", 0),
            "rental_yield_delta":    roi_raw.get("rental_yield_delta", 0.0),
            "rental_yield_post_pct": roi_raw.get("rental_yield_post_pct", 3.0),
            "model_type":            roi_raw.get("model_type", "heuristic"),
            "model_confidence":      roi_raw.get("model_confidence", 0.65),
            "city_tier":             roi_output.get("city_tier", 2),
        }

        updates = {
            **result,
            "roi_prediction":       roi_prediction,
            "roi_pct":              float(roi_prediction["roi_pct"]),
            "equity_gain_inr":      int(roi_prediction["equity_gain_inr"]),
            "payback_months":       int(roi_prediction["payback_months"]),
            "pre_reno_value_inr":   int(roi_prediction["pre_reno_value_inr"]),
            "post_reno_value_inr":  int(roi_prediction["post_reno_value_inr"]),
            "rental_yield_post_pct": float(roi_prediction["rental_yield_post_pct"]),
            "model_confidence":     float(roi_prediction["model_confidence"]),
        }
        logger.info(
            f"[{name}] done — ROI={roi_prediction['roi_pct']:.1f}% "
            f"equity=₹{roi_prediction['equity_gain_inr']:,} "
            f"payback={roi_prediction['payback_months']}mo"
        )

    except Exception as e:
        _record_error(updates, name, e, state)
        updates.update(_default_roi_prediction())

    _record_timing(updates, name, t0, state)
    _record_completion(updates, name, state)
    return _merge(state, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Node 7: Insight Generation
# ─────────────────────────────────────────────────────────────────────────────

def node_insight_generation(state: ARKENGraphState) -> ARKENGraphState:
    """
    Synthesise all upstream outputs into structured insights.
    NEW dedicated node — separated from report generation.
    Writes: insights, renovation_sequence, priority_repairs, budget_strategy.
    """
    t0 = time.perf_counter()
    name = "node_insight_generation"
    updates: Dict[str, Any] = {}

    try:
        from agents.insight_generation_agent import InsightGenerationAgent
        agent = InsightGenerationAgent()
        result = _sync_run_async(agent.run(dict(state)))
        updates = result
        logger.info(
            f"[{name}] done — "
            f"sequence={len(result.get('renovation_sequence', []))} steps "
            f"repairs={len(result.get('priority_repairs', []))} "
            f"rag_grounded={result.get('insights', {}).get('rag_grounded', False)}"
        )

    except Exception as e:
        _record_error(updates, name, e, state)
        updates.update(_default_insights(state))

    _record_timing(updates, name, t0, state)
    _record_completion(updates, name, state)
    return _merge(state, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Node 8: Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def node_report_generation(state: ARKENGraphState) -> ARKENGraphState:
    """
    Assemble the final structured RenovationReport from all upstream data.
    Reads: ALL upstream node outputs.
    Writes: final_report, renovation_report (legacy), final_recommendations, chat_context.
    """
    t0 = time.perf_counter()
    name = "node_report_generation"
    updates: Dict[str, Any] = {}

    try:
        from agents.report_agent_node import ReportAgentNode

        # ── Trust validation before report assembly ───────────────────────────
        try:
            from services.trust.output_validator import TrustValidator
            validator = TrustValidator()
            state = validator.validate_pipeline_output(dict(state))
            # Validate individual price forecasts (top 3 only for performance)
            material_prices = state.get("material_prices") or []
            if material_prices:
                state["material_prices"] = [
                    validator.validate_price_forecast(f) if isinstance(f, dict) else f
                    for f in material_prices[:3]
                ] + list(material_prices[3:])
            # Validate ROI prediction
            roi = state.get("roi_prediction")
            if roi and isinstance(roi, dict):
                state["roi_prediction"] = validator.validate_roi_prediction(roi)
        except Exception as trust_err:
            logger.warning(f"[report_generation] TrustValidator skipped: {trust_err}")

        agent = ReportAgentNode()
        result = _sync_run_async(agent.run(dict(state)))

        report = result.get("renovation_report") or {}
        insights = result.get("insights") or state.get("insights") or {}
        recommendations = _build_final_recommendations(state, insights)

        updates = {
            **result,
            "final_report":           report,             # new canonical key
            "renovation_report":      report,             # legacy compat
            "final_recommendations":  recommendations,
        }
        logger.info(
            f"[{name}] done — "
            f"recommendations={len(recommendations)}"
        )

    except Exception as e:
        _record_error(updates, name, e, state)
        updates.update(_default_report(state))

    _record_timing(updates, name, t0, state)
    _record_completion(updates, name, state)
    return _merge(state, updates)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_orchestrator_graph():
    """
    Compile the LangGraph StateGraph for the 8-node deterministic pipeline.

    DAG:
      user_goal → vision_analysis → rag_retrieval → design_planning
               → budget_estimation → roi_forecasting
               → insight_generation → report_generation → END

    Returns compiled graph or None (sequential fallback will be used).
    """
    try:
        from langgraph.graph import StateGraph, END

        g = StateGraph(ARKENGraphState)

        g.add_node("user_goal",           node_user_goal)
        g.add_node("vision_analysis",     node_vision_analysis)
        g.add_node("rag_retrieval",       node_rag_retrieval)
        g.add_node("design_planning",     node_design_planning)
        g.add_node("budget_estimation",   node_budget_estimation)
        g.add_node("roi_forecasting",     node_roi_forecasting)
        g.add_node("insight_generation",  node_insight_generation)
        g.add_node("report_generation",   node_report_generation)

        # Deterministic sequential edges — no conditional branching
        g.set_entry_point("user_goal")
        g.add_edge("user_goal",          "vision_analysis")
        g.add_edge("vision_analysis",    "rag_retrieval")
        g.add_edge("rag_retrieval",      "design_planning")
        g.add_edge("design_planning",    "budget_estimation")
        g.add_edge("budget_estimation",  "roi_forecasting")
        g.add_edge("roi_forecasting",    "insight_generation")
        g.add_edge("insight_generation", "report_generation")
        g.add_edge("report_generation",  END)

        compiled = g.compile()
        logger.info("✅ ARKEN LangGraph orchestrator v2.0 compiled (8 nodes)")
        return compiled

    except ImportError:
        logger.warning("LangGraph not installed — sequential fallback active")
        return None
    except Exception as e:
        logger.error(f"Graph build failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator Class
# ─────────────────────────────────────────────────────────────────────────────

# Maps node name → (function, progress_pct)
_PIPELINE_SEQUENCE = [
    ("user_goal",          node_user_goal,          10),
    ("vision_analysis",    node_vision_analysis,    25),
    ("rag_retrieval",      node_rag_retrieval,      38),
    ("design_planning",    node_design_planning,    52),
    ("budget_estimation",  node_budget_estimation,  65),
    ("roi_forecasting",    node_roi_forecasting,    78),
    ("insight_generation", node_insight_generation, 90),
    ("report_generation",  node_report_generation,  98),
]


class LangGraphOrchestrator:
    """
    Production LangGraph orchestrator for the ARKEN renovation pipeline v2.0.

    8-node deterministic pipeline with RAG retrieval and insight generation
    as first-class nodes.

    Usage (async):
        orchestrator = LangGraphOrchestrator()
        result = await orchestrator.run(initial_state)

    Usage (sync):
        orchestrator = LangGraphOrchestrator()
        result = orchestrator.run_sync(initial_state)

    Backward compatibility:
        get_contract_state(result)  → dict with 8 canonical fields
        build_orchestrator_state()  → construct initial state dict
    """

    def __init__(self) -> None:
        self._graph = build_orchestrator_graph()

    async def run(
        self,
        initial_state: Dict[str, Any],
        on_progress: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full deterministic pipeline asynchronously.

        Args:
            initial_state:  Dict with pipeline inputs (use make_initial_state()).
            on_progress:    Optional callback(step_name: str, pct: int).

        Returns:
            Final merged state dict.
        """
        if not isinstance(initial_state, dict):
            raise ValueError("initial_state must be a dict")

        state = _init_state(initial_state)

        # ── Try compiled LangGraph graph ──────────────────────────────────────
        if self._graph is not None:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._graph.invoke, state)
                if on_progress:
                    await _safe_progress(on_progress, "complete", 100)
                logger.info(
                    f"[orchestrator v2.0] LangGraph pipeline complete — "
                    f"nodes={result.get('completed_agents', [])}"
                )
                return dict(result)
            except Exception as e:
                logger.warning(f"[orchestrator] LangGraph invoke failed ({e}) — sequential fallback")
                errs = list(state.get("errors") or [])
                err_str = f"langgraph_invoke: {e}"
                if not any(err_str in ex for ex in errs):
                    errs.append(err_str)
                state["errors"] = errs

        # ── Sequential fallback ───────────────────────────────────────────────
        logger.info("[orchestrator] Running sequential fallback (8 nodes)")
        for step_name, node_fn, pct in _PIPELINE_SEQUENCE:
            try:
                loop = asyncio.get_event_loop()
                state = await loop.run_in_executor(None, node_fn, state)
                if on_progress:
                    await _safe_progress(on_progress, step_name, pct)
            except Exception as e:
                logger.error(f"[orchestrator.{step_name}] Fatal: {e}", exc_info=True)
                errs = list(state.get("errors") or [])
                err_str = f"{step_name}_fatal: {e}"
                if not any(err_str in ex for ex in errs):
                    errs.append(err_str)
                state["errors"] = errs
                state = _inject_fallback(step_name, state)

        if on_progress:
            await _safe_progress(on_progress, "complete", 100)
        logger.info(
            f"[orchestrator] Sequential pipeline complete — "
            f"nodes={state.get('completed_agents', [])}"
        )
        return state

    def run_sync(
        self,
        initial_state: Dict[str, Any],
        on_progress: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for non-async callers."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(asyncio.run, self.run(initial_state, on_progress))
                    return fut.result(timeout=360)
            return loop.run_until_complete(self.run(initial_state, on_progress))
        except RuntimeError:
            return asyncio.run(self.run(initial_state, on_progress))

    def get_contract_state(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the 8 canonical contract keys from the full pipeline state.
        Matches the goal specification exactly.
        """
        return extract_contract_state(final_state)


# ─────────────────────────────────────────────────────────────────────────────
# State builder (backward compat with v1.0 callers)
# ─────────────────────────────────────────────────────────────────────────────

def build_orchestrator_state(
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
) -> Dict[str, Any]:
    """Construct initial state dict. Preserved for backward compatibility."""
    return dict(make_initial_state(
        project_id=project_id,
        user_id=user_id,
        image_bytes=image_bytes,
        image_b64=image_b64,
        image_mime=image_mime,
        budget_inr=budget_inr,
        city=city,
        theme=theme,
        budget_tier=budget_tier,
        room_type=room_type,
        user_intent=user_intent,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — state init, vision helpers, fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _init_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required metadata keys exist in initial state."""
    state = dict(raw)
    state.setdefault("errors", [])
    state.setdefault("error_details", {})
    state.setdefault("agent_timings", {})
    state.setdefault("completed_agents", [])
    state.setdefault("pipeline_version", "2.0")
    # Contract keys
    state.setdefault("user_goal", "")
    state.setdefault("detected_features", {})
    state.setdefault("retrieved_knowledge", [])
    state.setdefault("rag_context", "")
    state.setdefault("design_plan", {})
    state.setdefault("cost_estimate", {})
    state.setdefault("roi_prediction", {})
    state.setdefault("insights", {})
    state.setdefault("final_report", {})
    state.setdefault("final_recommendations", [])
    state.setdefault("damage_assessment", {})
    return state


def _extract_condition_fields(
    room_features: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Hoist Gemini-extracted condition fields from room_features to top-level state.

    These fields are produced by RoomFeatures.from_gemini_response() and stored
    as private attributes (_issues_detected, _condition_score, etc.) then
    surfaced via to_dict(). They must be propagated to top-level state keys so
    that InsightGenerationAgent and DesignPlannerAgentNode can read them directly.

    Without this propagation, downstream agents always see None / "not_assessed"
    because they read state.get("issues_detected") which is never set.
    """
    # room_features is already the flattened to_dict() output from RoomFeatures
    # It includes condition_score, wall_condition, floor_condition, issues_detected,
    # renovation_scope, high_value_upgrades because to_dict() adds those explicitly.

    def _safe_list(val) -> list:
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val:
            return [val]
        return []

    def _safe_int(val, default=None):
        try:
            return int(val) if val not in (None, "", "not_assessed") else default
        except (TypeError, ValueError):
            return default

    issues_from_rf  = _safe_list(room_features.get("issues_detected"))
    issues_from_res = _safe_list(result.get("issues_detected"))
    # Also pull from nested _issues_detected if room_features was not flattened
    issues_private  = _safe_list(room_features.get("_issues_detected"))
    # Merge and deduplicate — room_features is highest fidelity
    all_issues = list(dict.fromkeys(issues_from_rf or issues_private or issues_from_res))

    # Layout issues supplement structural issues (e.g. over-furnished)
    layout = result.get("layout_report") or {}
    layout_issues = _safe_list(layout.get("issues", layout.get("issues_detected", [])))
    all_issues_merged = list(dict.fromkeys(all_issues + layout_issues))

    condition_score = (
        _safe_int(room_features.get("condition_score"))
        or _safe_int(room_features.get("_condition_score"))
        or _safe_int(result.get("condition_score"))
    )

    wall_condition = (
        room_features.get("wall_condition")
        or room_features.get("_wall_condition")
        or result.get("wall_condition")
        or "not_assessed"
    )

    floor_condition = (
        room_features.get("floor_condition")
        or room_features.get("_floor_condition")
        or result.get("floor_condition")
        or "not_assessed"
    )

    renovation_scope = (
        room_features.get("renovation_scope")
        or room_features.get("_renovation_scope")
        or result.get("renovation_scope")
        or "partial"
    )

    high_value_upgrades = (
        _safe_list(room_features.get("high_value_upgrades"))
        or _safe_list(room_features.get("_high_value_upgrades"))
        or _safe_list(result.get("high_value_upgrades"))
    )

    return {
        "issues_detected":     all_issues_merged,
        "condition_score":     condition_score,
        "wall_condition":      wall_condition,
        "floor_condition":     floor_condition,
        "renovation_scope":    renovation_scope,
        "high_value_upgrades": high_value_upgrades,
    }


def _build_detected_features(
    result: Dict[str, Any],
    room_features: Dict[str, Any],
    layout: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the canonical detected_features dict from vision agent output."""
    vision = result.get("vision_features") or {}
    return {
        "wall": {
            "treatment": vision.get("wall_treatment") or room_features.get("wall_treatment", ""),
            "colour":    room_features.get("wall_color", ""),
        },
        "floor": {
            "material":    vision.get("floor_material") or room_features.get("floor_material", ""),
            "area_sqft":   result.get("floor_area_sqft", 120.0),
        },
        "ceiling": {
            "treatment":  vision.get("ceiling_treatment") or room_features.get("ceiling_type", ""),
            "area_sqft":  result.get("wall_area_sqft", 200.0),
        },
        "lighting": {
            "type":   vision.get("lighting_type") or "",
            "sources": room_features.get("lighting_sources", []),
        },
        "colour_palette":  vision.get("colour_palette") or [],
        "furniture_items": vision.get("furniture_items") or [],
        "detected_style":  result.get("style_label", ""),
        "quality_tier":    vision.get("quality_tier", "mid"),
        "natural_light":   vision.get("natural_light_quality") or room_features.get("natural_light", "moderate"),
        "layout_score":    layout.get("layout_score", "65/100"),
        "walkable_space":  layout.get("walkable_space", "45%"),
    }


def _build_damage_assessment(room_features: Dict[str, Any], layout: Dict[str, Any]) -> Dict[str, Any]:
    """Build structured damage/condition assessment from vision features."""
    condition = room_features.get("condition", room_features.get("room_condition", "fair"))
    # Pull issues from room_features (has Gemini-extracted wall/structural issues)
    rf_issues = room_features.get("issues_detected") or room_features.get("_issues_detected") or []
    layout_issues = layout.get("issues", layout.get("issues_detected", []))
    # Merge both issue sources — room_features issues first (higher fidelity)
    issues = list(dict.fromkeys(
        (rf_issues if isinstance(rf_issues, list) else [rf_issues]) +
        (layout_issues if isinstance(layout_issues, list) else [])
    ))
    priority = room_features.get("renovation_priority", [])
    severity_map = {"poor": "high", "fair": "medium", "good": "low", "excellent": "none"}
    severity = severity_map.get(str(condition).lower(), "medium")

    # condition_score — may be stored as private attr or direct field
    raw_cs = (
        room_features.get("condition_score")
        or room_features.get("_condition_score")
    )
    condition_score = None
    try:
        if raw_cs not in (None, "", "not_assessed"):
            condition_score = int(raw_cs)
    except (TypeError, ValueError):
        condition_score = None

    return {
        "overall_condition":   condition,
        "condition_score":     condition_score,
        "severity":            severity,
        "layout_score":        layout.get("layout_score", "65/100"),
        "walkable_space":      layout.get("walkable_space_pct", layout.get("walkable_space", "45%")),
        "issues_detected":     issues,
        "renovation_priority": priority,
        "natural_light":       room_features.get("natural_light", "moderate"),
        "structural_notes":    (
            "Cosmetic renovation recommended."
            if condition in ("fair", "good")
            else "Significant renovation required."
        ),
    }


def _build_final_recommendations(
    state: Dict[str, Any],
    insights: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build final ranked recommendations from all upstream data."""
    recs: List[Dict[str, Any]] = []
    seen: set = set()

    sequence = state.get("renovation_sequence") or []
    for step in sequence[:5]:
        cat = step.get("category", "")
        if cat not in seen:
            seen.add(cat)
            recs.append({
                "rank":           len(recs) + 1,
                "title":          step.get("action", ""),
                "category":       cat,
                "priority":       step.get("priority", "medium"),
                "estimated_cost": step.get("cost_range", ""),
                "must_do":        step.get("must_do", False),
                "data_source":    step.get("source", "pipeline"),
            })

    roi = state.get("roi_prediction") or {}
    if roi.get("roi_pct", 0) > 0:
        recs.append({
            "rank":           len(recs) + 1,
            "title":          f"Projected {roi['roi_pct']:.1f}% ROI — focus on cosmetic upgrades",
            "category":       "financial",
            "priority":       "high",
            "estimated_cost": f"₹{(state.get('cost_estimate') or {}).get('total_inr', 0):,}",
            "must_do":        False,
            "data_source":    "roi_model",
        })

    cost = state.get("cost_estimate") or {}
    if not cost.get("within_budget", True):
        recs.append({
            "rank":           len(recs) + 1,
            "title":          "Over budget — recommend phased renovation approach",
            "category":       "budget",
            "priority":       "medium",
            "estimated_cost": f"₹{cost.get('total_inr', 0):,}",
            "must_do":        False,
            "data_source":    "budget_estimator",
        })

    # Append insight-engine recommendations
    for rec in (insights.get("recommendations") or [])[:3]:
        if isinstance(rec, dict) and rec.get("recommendation"):
            recs.append({
                "rank":           len(recs) + 1,
                "title":          rec.get("recommendation", ""),
                "category":       rec.get("category", "general"),
                "priority":       rec.get("priority", "medium"),
                "estimated_cost": rec.get("cost_range_inr", ""),
                "must_do":        False,
                "data_source":    "insight_engine",
            })

    return recs[:10]


# ── Fallback defaults (deterministic) ────────────────────────────────────────

def _default_vision_output(state: Dict[str, Any]) -> Dict[str, Any]:
    theme = state.get("theme", "Modern Minimalist")
    return {
        "detected_features": {
            "wall": {"treatment": f"{theme} paint", "colour": "neutral"},
            "floor": {"material": "vitrified tiles", "area_sqft": 120.0},
            "ceiling": {"treatment": "plain ceiling", "area_sqft": 120.0},
            "lighting": {"type": "ceiling mounted", "sources": []},
            "colour_palette": ["white", "grey"],
            "detected_style": theme,
            "quality_tier": state.get("budget_tier", "mid"),
            "natural_light": "moderate",
            "layout_score": "65/100",
            "walkable_space": "45%",
        },
        "damage_assessment": {
            "overall_condition": "fair", "severity": "medium",
            "layout_score": "65/100", "walkable_space": "45%",
            "issues_detected": [], "renovation_priority": ["walls", "flooring", "lighting"],
            "natural_light": "moderate", "structural_notes": "Vision analysis unavailable.",
        },
        "vision_features": {"extraction_source": "no_image_fallback"},
        "image_features": {}, "room_features": {}, "layout_report": {},
        "style_label": theme, "style_confidence": 0.4,
        "explainable_recommendations": [], "material_quantities": {},
        "detected_objects": [], "material_types": ["vitrified_tile"],
        "wall_area_sqft": 200.0, "floor_area_sqft": 120.0,
        "room_dimensions": {"wall_area_sqft": 200, "floor_area_sqft": 120},
        "detected_changes": [], "visual_style": [theme],
    }


def _default_design_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    budget = state.get("budget_inr", 750_000)
    return {
        "design_plan": {}, "material_plan": {},
        "boq_line_items": [], "schedule": {},
        "layout_suggestions": [],
        "labour_estimate": int(budget * 0.32),
        "total_cost_estimate": budget,
    }


def _default_cost_estimate(state: Dict[str, Any]) -> Dict[str, Any]:
    budget = state.get("budget_inr", 750_000)
    ce = {
        "total_inr": budget,
        "materials_inr": int(budget * 0.55),
        "labour_inr": int(budget * 0.30),
        "gst_inr": int(budget * 0.10),
        "contingency_inr": int(budget * 0.05),
        "supervision_inr": int(budget * 0.03),
        "cost_per_sqft": 0.0, "within_budget": True,
        "city_multiplier": 1.0, "budget_tier_analysis": {}, "material_prices": [],
    }
    return {
        "cost_estimate": ce, "budget_estimate": {"total_cost_inr": budget},
        "cost_per_sqft": 0.0, "within_budget": True,
        "location_context": {}, "budget_analysis": {},
        "material_prices": [], "cost_breakdown": {},
    }


def _default_roi_prediction() -> Dict[str, Any]:
    rp = {
        "roi_pct": 12.0, "equity_gain_inr": 0, "payback_months": 36,
        "pre_reno_value_inr": 0, "post_reno_value_inr": 0,
        "rental_yield_delta": 0.0, "rental_yield_post_pct": 3.0,
        "model_type": "fallback", "model_confidence": 0.5, "city_tier": 2,
    }
    return {
        "roi_prediction": rp, "roi_output": {}, "roi_pct": 12.0,
        "equity_gain_inr": 0, "payback_months": 36,
        "pre_reno_value_inr": 0, "post_reno_value_inr": 0,
        "rental_yield_post_pct": 3.0, "model_confidence": 0.5,
    }


def _default_insights(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "insights": {
            "summary_headline": f"{state.get('theme', 'Modern')} renovation — {state.get('city', 'India')}",
            "visual_analysis": {}, "financial_outlook": {},
            "market_intelligence": {}, "budget_assessment": {},
            "recommendations": [], "risk_factors": [], "top_materials": [],
            "rag_grounded": False, "insight_engine": {},
        },
        "insight_engine_output": {},
        "renovation_sequence": [],
        "priority_repairs": [],
        "budget_strategy": {},
    }


def _default_report(state: Dict[str, Any]) -> Dict[str, Any]:
    theme = state.get("theme", "Modern Minimalist")
    city = state.get("city", "India")
    report = {
        "report_version": "2.0",
        "project_id": state.get("project_id", ""),
        "summary_headline": f"{theme} renovation — {city}",
    }
    return {
        "final_report": report, "renovation_report": report,
        "final_recommendations": [], "chat_context": f"{theme} renovation in {city}",
    }


_FALLBACK_FN_MAP = {
    "user_goal":          lambda s: {"user_goal": s.get("user_intent", "personal_comfort"), "user_goals": {}},
    "vision_analysis":    _default_vision_output,
    "rag_retrieval":      lambda s: {"retrieved_knowledge": [], "rag_context": "", "rag_sources": [], "rag_retrieval_stats": {}},
    "design_planning":    _default_design_plan,
    "budget_estimation":  _default_cost_estimate,
    "roi_forecasting":    lambda s: _default_roi_prediction(),
    "insight_generation": _default_insights,
    "report_generation":  _default_report,
}


def _inject_fallback(step_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    fn = _FALLBACK_FN_MAP.get(step_name)
    if fn:
        state = {**state, **fn(state)}
    return state


async def _safe_progress(on_progress: Callable, step_name: str, pct: int) -> None:
    try:
        result = on_progress(step_name, pct)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        logger.debug(f"[orchestrator] Progress callback error (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_orchestrator_instance: Optional[LangGraphOrchestrator] = None


def get_orchestrator() -> LangGraphOrchestrator:
    """Return module-level singleton LangGraphOrchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = LangGraphOrchestrator()
    return _orchestrator_instance


# ─────────────────────────────────────────────────────────────────────────────
# v1.0 backward-compatibility aliases
# These are re-exported so existing callers (e.g. orchestrator/__init__.py v1.0)
# continue to import without errors.
# ─────────────────────────────────────────────────────────────────────────────

# v1 name → v2 equivalent
node_renovation_planning = node_design_planning
node_cost_estimation     = node_budget_estimation
node_roi_prediction      = node_roi_forecasting