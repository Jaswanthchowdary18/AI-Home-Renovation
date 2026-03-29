"""
ARKEN — /api/v1/analyze routes v4.0 (LangGraph Multi-Agent)
============================================================
Upgraded to use the new LangGraph multi-agent pipeline while keeping
full backward compatibility with the existing response schema.

Changes from v3.0:
  - _run_pipeline now calls multi_agent_pipeline.run_multi_agent_pipeline_async
  - Falls back to ARKENOrchestrator if multi-agent pipeline fails
  - Added /agent-status endpoint for per-agent progress details
  - Added /memory/{user_id} endpoint for user memory context

Unchanged:
  - All request/response Pydantic models
  - /status/{task_id} polling endpoint
  - Magic-byte image validation
  - Cache TTL and structure
"""

import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from db.session import get_db
from services.cache import cache_service

router = APIRouter()

MAX_IMAGE_SIZE = 20 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/heic", "image/webp"}
MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"GIF8": "image/gif",
    b"RIFF": "image/webp",
}


def _is_valid_image(data: bytes) -> bool:
    for magic in MAGIC_BYTES:
        if data[:len(magic)] == magic:
            return True
    return len(data) > 1024


class AnalyzeResponse(BaseModel):
    project_id: str
    task_id: str
    status: str
    message: str


class PipelineStatusResponse(BaseModel):
    task_id: str
    project_id: str
    status: str
    progress_pct: int
    current_step: str
    result: Optional[dict] = None
    error: Optional[str] = None


class AgentStatusResponse(BaseModel):
    task_id: str
    project_id: str
    status: str
    completed_agents: list
    agent_timings: dict
    progress_pct: int
    errors: list


@router.post("/", response_model=AnalyzeResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Room image (JPG/PNG/HEIC/WebP)"),
    budget_inr: int = Form(..., ge=100_000, le=5_000_000),
    city: str = Form(...),
    theme: str = Form("Modern Minimalist"),
    budget_tier: str = Form("mid"),
    room_type: str = Form("bedroom"),
    project_name: str = Form("Untitled Project"),
    user_intent: str = Form(""),
    user_id: str = Form(""),
    db=Depends(get_db),
):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPG, PNG, HEIC or WebP.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(400, "Image too large. Maximum 20 MB.")
    if len(image_bytes) < 1024:
        raise HTTPException(400, "Image too small or corrupted.")
    if not _is_valid_image(image_bytes):
        raise HTTPException(400, "File does not appear to be a valid image.")

    project_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    # DB record — non-fatal
    try:
        from db.models import Project, ProjectStatus
        project = Project(
            id=uuid.UUID(project_id),
            name=project_name,
            owner_id=uuid.UUID(user_id) if user_id else uuid.uuid4(),
            status=ProjectStatus.ANALYZING,
            city=city,
            budget_inr=budget_inr,
            budget_tier=budget_tier,
            theme=theme,
            room_type=room_type,
        )
        db.add(project)
        await db.flush()
    except Exception:
        pass

    # Cache initial status
    await cache_service.set(
        f"task:{task_id}",
        {
            "status": "queued",
            "progress_pct": 0,
            "current_step": "Queued",
            "project_id": project_id,
            "completed_agents": [],
            "agent_timings": {},
            "errors": [],
        },
        ttl=7200,
    )

    background_tasks.add_task(
        _run_pipeline,
        task_id=task_id,
        project_id=project_id,
        user_id=user_id,
        image_bytes=image_bytes,
        budget_inr=budget_inr,
        city=city,
        theme=theme,
        budget_tier=budget_tier,
        room_type=room_type,
        user_intent=user_intent,
        project_name=project_name,
    )

    return AnalyzeResponse(
        project_id=project_id,
        task_id=task_id,
        status="queued",
        message="Multi-agent analysis pipeline started. Poll /status/{task_id} for updates.",
    )


@router.get("/status/{task_id}", response_model=PipelineStatusResponse)
async def get_status(task_id: str):
    data = await cache_service.get(f"task:{task_id}")
    if data is None:
        raise HTTPException(404, "Task not found or expired.")
    return PipelineStatusResponse(task_id=task_id, **{
        k: v for k, v in data.items()
        if k in PipelineStatusResponse.model_fields
    })


@router.get("/agent-status/{task_id}", response_model=AgentStatusResponse)
async def get_agent_status(task_id: str):
    """
    Returns per-agent progress detail for frontend display.
    Shows which agents have completed and their individual timings.
    """
    data = await cache_service.get(f"task:{task_id}")
    if data is None:
        raise HTTPException(404, "Task not found or expired.")

    return AgentStatusResponse(
        task_id=task_id,
        project_id=data.get("project_id", ""),
        status=data.get("status", "unknown"),
        completed_agents=data.get("completed_agents", []),
        agent_timings=data.get("agent_timings", {}),
        progress_pct=data.get("progress_pct", 0),
        errors=data.get("errors", []),
    )


@router.get("/memory/{user_id}")
async def get_user_memory(user_id: str):
    """
    Returns the memory context for a user — past renovation sessions.
    Used by frontend to personalise the next session.
    """
    try:
        from memory.agent_memory import agent_memory
        ctx = await agent_memory.recall(user_id)
        return {"user_id": user_id, **ctx}
    except Exception as e:
        raise HTTPException(500, f"Memory retrieval failed: {e}")


# ── Agent display names for progress updates ──────────────────────────────────

AGENT_DISPLAY = {
    "user_goal_agent":         "Understanding Goals",
    "vision_analyzer_agent":   "Analysing Room Vision",
    "design_planner_agent":    "Planning Design & BOQ",
    "budget_estimator_agent":  "Estimating Budget",
    "roi_agent":               "Predicting ROI",
    "report_agent":            "Generating Report",
    # Legacy orchestrator steps (kept for fallback compatibility)
    "Visual Assessor":         "Analysing Room Vision",
    "Design Planner":          "Planning Design & BOQ",
    "ROI Forecast":            "Predicting ROI",
    "Price Oracle":            "Fetching Market Prices",
    "Project Coordinator":     "Building Schedule",
    "Rendering Engine":        "Rendering Design",
}

AGENT_PROGRESS = {
    "user_goal_agent":         10,
    "vision_analyzer_agent":   30,
    "design_planner_agent":    50,
    "budget_estimator_agent":  65,
    "roi_agent":               80,
    "report_agent":            95,
    "Visual Assessor":         25,
    "Design Planner":          45,
    "ROI Forecast":            60,
    "Price Oracle":            65,
    "Project Coordinator":     75,
    "Rendering Engine":        90,
}


async def _run_pipeline(
    task_id: str,
    project_id: str,
    user_id: str,
    image_bytes: bytes,
    budget_inr: int,
    city: str,
    theme: str,
    budget_tier: str,
    room_type: str,
    user_intent: str = "",
    project_name: str = "Untitled Project",
):
    """
    Main background task: runs multi-agent pipeline node-by-node,
    writing cache after EVERY agent so /status/{task_id} shows real progress.
    Falls back to ARKENOrchestrator if the multi-agent pipeline fails entirely.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    async def _write_progress(
        step_name: str,
        pct: int,
        state: dict,
        status: str = "processing",
    ) -> None:
        """Write per-agent progress to cache immediately after each node."""
        current = await cache_service.get(f"task:{task_id}") or {}
        await cache_service.set(
            f"task:{task_id}",
            {
                **current,
                "status": status,
                "progress_pct": min(pct, 98),
                "current_step": step_name,
                "project_id": project_id,
                "completed_agents": state.get("completed_agents", []),
                "agent_timings": state.get("agent_timings", {}),
                "error_count": len(state.get("errors", [])),
                "errors": state.get("errors", []),
            },
            ttl=3600,
        )

    # Legacy on_progress callback kept for the fallback orchestrator path
    async def on_progress(step_name: str, pct_or_status):
        if isinstance(pct_or_status, int):
            pct = pct_or_status
        else:
            base_pct = AGENT_PROGRESS.get(step_name, 50)
            pct = base_pct + 5 if pct_or_status == "complete" else base_pct
        display = AGENT_DISPLAY.get(step_name, step_name)
        current = await cache_service.get(f"task:{task_id}") or {}
        await cache_service.set(
            f"task:{task_id}",
            {
                **current,
                "status": "running",
                "progress_pct": min(pct, 95),
                "current_step": display,
                "project_id": project_id,
            },
            ttl=7200,
        )

    try:
        from agents.multi_agent_pipeline import (
            AGENT_SEQUENCE,
            STEP_PROGRESS,
            build_initial_state,
        )
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

        state = build_initial_state(
            project_id=project_id,
            user_id=user_id,
            image_bytes=image_bytes,
            image_mime="image/jpeg",
            budget_inr=budget_inr,
            city=city,
            theme=theme,
            budget_tier=budget_tier,
            room_type=room_type,
            user_intent=user_intent,
            project_name=project_name,
        )

        # ── Step-aware sequential execution ──────────────────────────────────
        # Run each node individually so we can write progress after every step.
        # This is the same execution order as AGENT_SEQUENCE but gives the
        # /status endpoint real incremental updates instead of 0 → 100.
        for agent_name, node_fn, pct in AGENT_SEQUENCE:
            display = AGENT_DISPLAY.get(agent_name, agent_name)
            try:
                # Signal "starting" before the node runs
                await _write_progress(f"Starting: {display}", max(pct - 10, 1), state)

                # Execute node synchronously in thread executor
                import asyncio as _asyncio
                loop = _asyncio.get_event_loop()
                state = await loop.run_in_executor(None, node_fn, state)

                # Write progress after successful completion
                await _write_progress(display, pct, state)
                _log.info(f"[analyze] ✅ {agent_name} ({pct}%)")

            except Exception as node_err:
                _log.error(f"[analyze] {agent_name} failed: {node_err}", exc_info=True)
                errs = list(state.get("errors") or [])
                err_str = f"{agent_name}: {node_err}"
                if not any(err_str in e for e in errs):
                    errs.append(err_str)
                state["errors"] = errs
                await _write_progress(f"⚠ {display} (error)", pct, state)
                # Continue — pipeline must always finish

        # ── Product Suggestions (post-pipeline, non-blocking) ─────────────────
        try:
            from agents.pipeline_product_integration import enrich_with_product_suggestions
            await _write_progress("Detecting furniture for Shop This Look...", 99, state)
            state = await enrich_with_product_suggestions(state)
            items = (state.get("product_suggestions") or {}).get("items_detected", 0)
            _log.info(f"[analyze] ProductSuggesterAgent: {items} items detected")
        except Exception as ps_err:
            _log.warning(f"[analyze] ProductSuggesterAgent skipped: {ps_err}")
            state.setdefault("product_suggestions", None)

        # ── Build summary from final state ────────────────────────────────────
        report = state.get("renovation_report") or {}
        if hasattr(report, "model_dump"):
            report = report.model_dump()
        elif hasattr(report, "dict"):
            report = report.dict()

        summary = {
            "project_id":      project_id,
            "status":          "complete" if not state.get("errors") else "partial",
            "errors":          state.get("errors", []),
            "error_details":   state.get("error_details", {}),
            "renovation_report": report,
            "insights":        state.get("insights") or {},
            # FIX: roi_output is now the full dict (roi_agent_node v2.0).
            # Prefer roi_output (explicit, always full) over roi_prediction for
            # backward-compat with older nodes that may have set roi_prediction
            # to the stripped 12-key subset.
            "roi":             state.get("roi_output") or state.get("roi_prediction") or {},
            "design_plan":     state.get("design_plan") or {},
            "boq_line_items":  state.get("boq_line_items") or [],
            "schedule":        state.get("schedule") or {},
            "budget_estimate": state.get("budget_estimate") or {},
            "location_context": state.get("location_context") or {},
            "agent_timings":   state.get("agent_timings") or {},
            "completed_agents": state.get("completed_agents") or [],
            "visual":          state.get("room_features") or state.get("image_features") or {},
            "design": {
                "total_inr":    state.get("total_cost_estimate", 0),
                "line_items":   state.get("boq_line_items") or [],
                "material_plan": state.get("material_plan") or {},
            },
            "price_forecast":  state.get("material_prices") or [],
            "chat_context":    state.get("chat_context") or "",
            # Product suggestions for Shop This Look
            "product_suggestions": state.get("product_suggestions"),
            # BUG 4 FIX: Products injected into BOQ — cost included in budget_estimate total
            "products_in_boq":       state.get("products_in_boq", []),
            "products_subtotal_inr": state.get("products_subtotal_inr", 0),
            # Convenience breakdown: renovation cost vs furniture cost vs grand total
            "cost_summary": {
                "renovation_materials_and_labour_inr": (
                    state.get("budget_estimate") or {}
                ).get("renovation_cost_inr") or (
                    state.get("budget_estimate") or {}
                ).get("total_cost_inr", 0),
                "furniture_and_fixtures_inr": state.get("products_subtotal_inr", 0),
                "grand_total_inr": (state.get("budget_estimate") or {}).get("total_cost_inr", 0),
                "products_included": bool(state.get("products_in_boq")),
            },
        }

    except Exception as e:
        # ── Hard fallback to legacy ARKENOrchestrator ─────────────────────────
        _log.error(
            f"Multi-agent pipeline failed: {e} — falling back to ARKENOrchestrator",
            exc_info=True,
        )
        try:
            from agents.orchestrator import ARKENOrchestrator, PipelineContext

            ctx = PipelineContext(
                project_id=project_id,
                user_id=user_id,
                image_bytes=image_bytes,
                budget_inr=budget_inr,
                city=city,
                theme=theme,
                budget_tier=budget_tier,
                room_type=room_type,
                custom_instructions=user_intent,
            )

            orchestrator = ARKENOrchestrator()
            ctx = await orchestrator.run(ctx, on_progress=on_progress)
            summary = ctx.to_summary()
            summary["pipeline_mode"] = "legacy_orchestrator_fallback"
        except Exception as e2:
            await cache_service.set(
                f"task:{task_id}",
                {
                    "status": "failed",
                    "progress_pct": 0,
                    "current_step": "Error",
                    "project_id": project_id,
                    "completed_agents": [],
                    "agent_timings": {},
                    "errors": [str(e), str(e2)],
                },
                ttl=7200,
            )
            return

    # ── Cache full result ─────────────────────────────────────────────────────
    await cache_service.set(
        f"project_report:{project_id}",
        summary,
        ttl=7200,
    )

    await cache_service.set(
        f"task:{task_id}",
        {
            "status": "complete",
            "progress_pct": 100,
            "current_step": "Done",
            "project_id": project_id,
            "completed_agents": summary.get("completed_agents", []),
            "agent_timings": summary.get("agent_timings", {}),
            "error_count": len(summary.get("errors", [])),
            "errors": summary.get("errors", []),
            "result": summary,
        },
        ttl=7200,
    )