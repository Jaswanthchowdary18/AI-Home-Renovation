"""
ARKEN — CrewAI-style Multi-Agent Orchestrator
Coordinates the full renovation pipeline:
  VisualAssessor → DesignPlanner → ROIForecast → PriceForecast
  → Coordinator → Rendering → ChatSupervisor
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from agents.coordinator import ProjectCoordinatorAgent
from agents.design_planner import DesignPlannerAgent
from agents.price_forecast import PriceForecastAgent
from agents.rendering import RenderingAgent, build_material_spec
from agents.roi_forecast import ROIForecastAgent
from agents.visual_assessor import VisualAssessorAgent

logger = logging.getLogger(__name__)


# ── Pipeline context ──────────────────────────────────────────────────────────

@dataclass
class PipelineContext:
    """Shared state passed between agents in the pipeline."""
    project_id: str
    user_id: str
    image_bytes: bytes
    budget_inr: int
    city: str
    theme: str
    budget_tier: str
    room_type: str = "bedroom"
    area_sqft: float = 120.0
    custom_instructions: str = ""

    # Populated by agents
    visual_result: Dict = field(default_factory=dict)
    design_result: Dict = field(default_factory=dict)
    roi_result: Dict = field(default_factory=dict)
    price_result: List[Dict] = field(default_factory=list)
    schedule_result: Dict = field(default_factory=dict)
    render_result: Dict = field(default_factory=dict)

    # Runtime
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def to_summary(self) -> Dict:
        visual = self.visual_result or {}
        # Attach new structured fields directly to visual for downstream use
        visual_out = {
            **visual,
            "room_features": visual.get("room_features", {}),
            "layout_report": visual.get("layout_report", {}),
            "design_recommendations": visual.get("design_recommendations", []),
            "style_label": visual.get("style_label", ""),
            "style_confidence": visual.get("style_confidence", 0),
        }
        return {
            "project_id": self.project_id,
            "status": "complete" if not self.errors else "partial",
            "errors": self.errors,
            "visual": visual_out,
            "design": self.design_result,
            "roi": self.roi_result,
            "price_forecast": self.price_result,
            "schedule": self.schedule_result,
            "render": self.render_result,
            "duration_seconds": (
                (self.completed_at or datetime.utcnow()) - self.started_at
            ).total_seconds(),
        }


# ── Agent wrappers ────────────────────────────────────────────────────────────

class AgentStep:
    """
    Wraps an agent call with error isolation, logging, and progress callbacks.
    """

    def __init__(self, name: str, fn: Callable, critical: bool = True):
        self.name = name
        self.fn = fn
        self.critical = critical

    async def run(self, ctx: PipelineContext, on_progress: Optional[Callable] = None) -> bool:
        logger.info(f"[Pipeline] Starting: {self.name}")
        if on_progress:
            await on_progress(self.name, "started")
        try:
            await self.fn(ctx)
            logger.info(f"[Pipeline] Completed: {self.name}")
            if on_progress:
                await on_progress(self.name, "complete")
            return True
        except Exception as e:
            msg = f"{self.name} failed: {str(e)}"
            logger.error(msg, exc_info=True)
            ctx.errors.append(msg)
            if on_progress:
                await on_progress(self.name, "error")
            if self.critical:
                raise
            return False


# ── Step implementations ──────────────────────────────────────────────────────

async def _step_visual(ctx: PipelineContext):
    agent = VisualAssessorAgent()
    ctx.visual_result = await agent.analyze(ctx.image_bytes, ctx.project_id, room_type=ctx.room_type)
    # Update area estimate from vision
    dims = ctx.visual_result.get("spatial_map", {}).get("dimensions", {})
    ctx.area_sqft = dims.get("floor_area_sqft", ctx.area_sqft)


async def _step_design(ctx: PipelineContext):
    agent = DesignPlannerAgent()
    ctx.design_result = agent.plan(
        theme=ctx.theme,
        budget_inr=ctx.budget_inr,
        budget_tier=ctx.budget_tier,
        area_sqft=ctx.area_sqft,
        room_type=ctx.room_type,
        city=ctx.city,
        quantities=ctx.visual_result.get("material_quantities", {}),
    )


async def _step_roi(ctx: PipelineContext):
    agent = ROIForecastAgent()
    ctx.roi_result = agent.predict(
        renovation_cost_inr=ctx.budget_inr,
        area_sqft=ctx.area_sqft,
        city=ctx.city,
        room_type=ctx.room_type,
        budget_tier=ctx.budget_tier,
    )


async def _step_price(ctx: PipelineContext):
    agent = PriceForecastAgent()
    ctx.price_result = agent.forecast_all(horizon_days=90)


async def _step_schedule(ctx: PipelineContext):
    agent = ProjectCoordinatorAgent()
    ctx.schedule_result = agent.generate_schedule(
        area_sqft=ctx.area_sqft,
        budget_inr=ctx.budget_inr,
        room_type=ctx.room_type,
        city=ctx.city,
    )


async def _step_render(ctx: PipelineContext):
    agent = RenderingAgent()
    masks = ctx.visual_result.get("spatial_map", {}).get("masks_s3", {})
    combined_mask_key = masks.get("combined_reno", "")

    # Load mask bytes from S3 (or use blank fallback)
    from services.storage import s3_service
    from config import settings

    mask_bytes = b""
    if combined_mask_key:
        try:
            mask_bytes = await s3_service.download_bytes(
                combined_mask_key, bucket=settings.S3_BUCKET_UPLOADS
            )
        except Exception:
            logger.warning("Could not load mask from S3, using blank mask.")

    material_spec = build_material_spec(ctx.theme, ctx.budget_tier)

    # Extract CV analysis data from visual_result for structural fidelity in prompt
    detected_objects = ctx.visual_result.get("detected_objects", [])
    room_features   = ctx.visual_result.get("room_features", {})
    wall_color      = room_features.get("wall_color", "neutral")
    floor_type      = room_features.get("floor_type", "tiles")
    room_dims       = ctx.visual_result.get("spatial_map", {}).get("dimensions", {})
    room_dimensions_hint = (
        f"{room_dims.get('floor_area_sqft', '')} sqft".strip()
        if room_dims else ""
    )
    # Count windows and doors from detected objects list
    objects_lower = [str(o).lower() for o in detected_objects]
    window_count  = max(1, sum(1 for o in objects_lower if "window" in o))
    door_count    = max(1, sum(1 for o in objects_lower if "door" in o))

    ctx.render_result = await agent.render(
        original_image_bytes=ctx.image_bytes,
        project_id=ctx.project_id,
        version=1,
        theme=ctx.theme,
        city=ctx.city,
        budget_tier=ctx.budget_tier,
        custom_instructions=ctx.custom_instructions,
        detected_objects=detected_objects,
        wall_color=wall_color,
        floor_type=floor_type,
        window_count=window_count,
        door_count=door_count,
        room_dimensions_hint=room_dimensions_hint,
    )


# ── Main Orchestrator ─────────────────────────────────────────────────────────

class ARKENOrchestrator:
    """
    Coordinates all agents in the ARKEN pipeline.
    Supports parallel execution of independent steps.
    """

    PIPELINE: List[AgentStep] = [
        AgentStep("Visual Assessor", _step_visual, critical=True),
        # These two can run in parallel after visual
        AgentStep("Design Planner", _step_design, critical=True),
        # ROI + Price run concurrently
        AgentStep("ROI Forecast", _step_roi, critical=False),
        AgentStep("Price Oracle", _step_price, critical=False),
        AgentStep("Project Coordinator", _step_schedule, critical=False),
        AgentStep("Rendering Engine", _step_render, critical=True),
    ]

    async def run(
        self,
        ctx: PipelineContext,
        on_progress: Optional[Callable] = None,
    ) -> PipelineContext:
        """
        Execute the full renovation pipeline.
        Parallel execution where dependencies allow.
        """
        # Step 1: Visual assessment (blocking — all depend on it)
        await self.PIPELINE[0].run(ctx, on_progress)

        # Step 2: Design planner (needs visual output)
        await self.PIPELINE[1].run(ctx, on_progress)

        # Steps 3-5: ROI, Price, Schedule — run in parallel
        await asyncio.gather(
            self.PIPELINE[2].run(ctx, on_progress),
            self.PIPELINE[3].run(ctx, on_progress),
            self.PIPELINE[4].run(ctx, on_progress),
            return_exceptions=True,
        )

        # Step 6: Render (needs visual mask + design spec)
        await self.PIPELINE[5].run(ctx, on_progress)

        ctx.completed_at = datetime.utcnow()
        return ctx

    async def rerun_render(
        self,
        ctx: PipelineContext,
        new_instructions: str,
        version: int,
        material_overrides: Optional[Dict] = None,
    ) -> Dict:
        """Re-trigger only the rendering step with updated instructions."""
        ctx.custom_instructions = new_instructions
        agent = RenderingAgent()
        from agents.rendering import build_material_spec

        # Re-use CV data already captured in visual_result
        detected_objects = ctx.visual_result.get("detected_objects", [])
        room_features   = ctx.visual_result.get("room_features", {})
        wall_color      = room_features.get("wall_color", "neutral")
        floor_type      = room_features.get("floor_type", "tiles")
        room_dims       = ctx.visual_result.get("spatial_map", {}).get("dimensions", {})
        room_dimensions_hint = (
            f"{room_dims.get('floor_area_sqft', '')} sqft".strip()
            if room_dims else ""
        )
        objects_lower = [str(o).lower() for o in detected_objects]
        window_count  = max(1, sum(1 for o in objects_lower if "window" in o))
        door_count    = max(1, sum(1 for o in objects_lower if "door" in o))

        return await agent.render(
            original_image_bytes=ctx.image_bytes,
            project_id=ctx.project_id,
            version=version,
            theme=ctx.theme,
            city=ctx.city,
            budget_tier=ctx.budget_tier,
            custom_instructions=new_instructions,
            material_overrides=material_overrides,
            detected_objects=detected_objects,
            wall_color=wall_color,
            floor_type=floor_type,
            window_count=window_count,
            door_count=door_count,
            room_dimensions_hint=room_dimensions_hint,
        )