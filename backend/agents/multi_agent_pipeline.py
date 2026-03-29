"""
ARKEN — LangGraph Multi-Agent Pipeline v3.0
=============================================
8-node deterministic pipeline using the upgraded orchestrator.

Pipeline order (strictly deterministic):
  node_user_goal → node_vision_analysis → node_rag_retrieval
  → node_design_planning → node_budget_estimation → node_roi_forecasting
  → node_insight_generation → node_report_generation → END

Architecture:
  - ARKENGraphState (TypedDict) is the single shared state schema.
  - Each node has ONE responsibility and writes to non-overlapping state keys.
  - Nodes communicate ONLY through state — no direct agent-to-agent calls.
  - RAG retrieval and Insight Generation are now first-class pipeline steps.
  - LangGraph compiled graph is tried first; sequential fallback on failure.
  - All existing services (forecasting, vector store, routes) are untouched.

Backward compatibility:
  - run_multi_agent_pipeline() signature unchanged.
  - run_multi_agent_pipeline_async() signature unchanged.
  - build_initial_state() signature unchanged.
  - All response keys from v2.0 are preserved.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional

# Use canonical state schema
from agents.graph_state import ARKENGraphState, make_initial_state

# Import all 8 node functions from the upgraded orchestrator
from agents.orchestrator.langgraph_orchestrator import (
    node_user_goal,
    node_vision_analysis,
    node_rag_retrieval,
    node_design_planning,
    node_budget_estimation,
    node_roi_forecasting,
    node_insight_generation,
    node_report_generation,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Per-agent timeout budgets (seconds)
# Pipeline MUST complete even if individual agents time out.
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TIMEOUTS: Dict[str, int] = {
    "user_goal_agent":          10,
    "vision_analyzer_agent":    45,   # Gemini vision can be slow
    "rag_retrieval_agent":      15,
    "design_planner_agent":     20,
    "budget_estimator_agent":   30,   # calls price_forecast (ML)
    "roi_agent":                20,   # XGBoost trains on first run
    "insight_generation_agent": 15,
    "report_agent":             10,
}


# ─────────────────────────────────────────────────────────────────────────────
# Ordered pipeline sequence with progress percentages
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SEQUENCE = [
    ("user_goal_agent",          node_user_goal,           10),
    ("vision_analyzer_agent",    node_vision_analysis,     25),
    ("rag_retrieval_agent",      node_rag_retrieval,       38),
    ("design_planner_agent",     node_design_planning,     52),
    ("budget_estimator_agent",   node_budget_estimation,   65),
    ("roi_agent",                node_roi_forecasting,     78),
    ("insight_generation_agent", node_insight_generation,  90),
    ("report_agent",             node_report_generation,   98),
]

STEP_PROGRESS = {name: pct for name, _, pct in AGENT_SEQUENCE}


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_multi_agent_graph():
    """
    Build the LangGraph StateGraph with 8 specialised agent nodes.

    Pipeline:
      user_goal → vision_analysis → rag_retrieval → design_planning
               → budget_estimation → roi_forecasting
               → insight_generation → report_generation → END
    """
    try:
        from langgraph.graph import StateGraph, END

        g = StateGraph(ARKENGraphState)

        for node_name, node_fn, _ in AGENT_SEQUENCE:
            g.add_node(node_name, node_fn)

        g.set_entry_point("user_goal_agent")
        g.add_edge("user_goal_agent",          "vision_analyzer_agent")
        g.add_edge("vision_analyzer_agent",    "rag_retrieval_agent")
        g.add_edge("rag_retrieval_agent",      "design_planner_agent")
        g.add_edge("design_planner_agent",     "budget_estimator_agent")
        g.add_edge("budget_estimator_agent",   "roi_agent")
        g.add_edge("roi_agent",                "insight_generation_agent")
        g.add_edge("insight_generation_agent", "report_agent")
        g.add_edge("report_agent",             END)

        compiled = g.compile()
        logger.info("✅ LangGraph multi-agent graph v3.0 compiled (8 nodes)")
        return compiled

    except ImportError as e:
        logger.warning(f"LangGraph not installed ({e}) — sequential fallback active")
        return None
    except Exception as e:
        logger.error(f"Graph build failed: {e} — sequential fallback active")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (sync)
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_agent_pipeline(
    initial_state: Dict[str, Any],
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Execute the 8-agent deterministic LangGraph pipeline.

    Tries LangGraph compiled graph first.
    Falls back to sequential node execution if LangGraph fails.

    Args:
        initial_state:  Dict matching ARKENGraphState fields.
        on_progress:    Optional sync callback(step_name, pct) for SSE progress.

    Returns:
        Final merged state dict with all agent outputs.
    """
    start = time.perf_counter()
    state = dict(initial_state)

    # Ensure required metadata fields
    state.setdefault("errors", [])
    state.setdefault("agent_timings", {})
    state.setdefault("completed_agents", [])
    state.setdefault("pipeline_version", "3.0")

    # ── Try compiled LangGraph graph ──────────────────────────────────────────
    graph = build_multi_agent_graph()
    if graph is not None:
        try:
            logger.info("▶ Running LangGraph multi-agent pipeline v3.0 (8 nodes)")
            result = graph.invoke(state)
            elapsed = round(time.perf_counter() - start, 2)
            logger.info(f"✅ LangGraph pipeline completed in {elapsed}s")
            if on_progress:
                try:
                    on_progress("report_agent", 100)
                except Exception:
                    pass
            return dict(result)
        except Exception as e:
            logger.warning(f"LangGraph invoke failed ({e}) — falling back to sequential")
            state["errors"] = list(state.get("errors") or []) + [f"langgraph_invoke: {e}"]

    # ── Sequential fallback ───────────────────────────────────────────────────
    logger.info("▶ Running sequential multi-agent fallback (8 nodes)")
    for agent_name, node_fn, pct in AGENT_SEQUENCE:
        timeout_sec = AGENT_TIMEOUTS.get(agent_name, 30)
        try:
            # Run the synchronous node inside a thread with a real-clock timeout.
            # asyncio.wait_for needs a coroutine — wrap with run_in_executor.
            loop = asyncio.new_event_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(node_fn, state)
                try:
                    state = future.result(timeout=timeout_sec)
                except concurrent.futures.TimeoutError:
                    raise asyncio.TimeoutError()
            if on_progress:
                try:
                    on_progress(agent_name, pct)
                except Exception:
                    pass
        except asyncio.TimeoutError:
            msg = f"{agent_name}: timed out after {timeout_sec}s"
            logger.warning(f"[{agent_name}] ⏱ {msg} — pipeline continues with partial state")
            errs = list(state.get("errors") or [])
            if not any(msg in e for e in errs):
                errs.append(msg)
            state["errors"] = errs
        except Exception as e:
            logger.error(f"[{agent_name}] Fatal error in sequential fallback: {e}", exc_info=True)
            errs = list(state.get("errors") or [])
            err_str = f"{agent_name}: {e}"
            if not any(err_str in ex for ex in errs):
                errs.append(err_str)
            state["errors"] = errs

    elapsed = round(time.perf_counter() - start, 2)
    logger.info(f"✅ Sequential multi-agent pipeline completed in {elapsed}s")
    if on_progress:
        try:
            on_progress("report_agent", 100)
        except Exception:
            pass
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Async wrapper
# ─────────────────────────────────────────────────────────────────────────────

async def run_multi_agent_pipeline_async(
    initial_state: Dict[str, Any],
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Async wrapper — runs pipeline in thread executor to avoid blocking event loop.
    Signature unchanged from v2.0 for backward compatibility.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(run_multi_agent_pipeline, initial_state, on_progress),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Initial state builder (signature preserved from v2.0)
# ─────────────────────────────────────────────────────────────────────────────

def build_initial_state(
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
) -> Dict[str, Any]:
    """
    Construct the initial pipeline state dict from raw API parameters.
    Wraps make_initial_state() from graph_state module.
    Signature preserved from v2.0 for full backward compatibility.
    """
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
        project_name=project_name,
    ))