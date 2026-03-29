"""
ARKEN — ReportAgent (LangGraph node) v2.0
==========================================
Agent 6 (final) in the LangGraph multi-agent pipeline.

Responsibilities:
  - Assembles final structured RenovationReport from all upstream agents
  - Runs InsightEngine to synthesise priority scores, ROI, and strategies
  - Generates insights and chat context (delegates to pipeline insight agent)
  - Saves memory for the user session
  - Ensures backward compatibility with existing report_generator.py

Input state keys:  All outputs from agents 1-5 (vision, design, budget, ROI)
Output state keys: renovation_report, insights, chat_context, insight_engine_output
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ReportAgentNode:
    """
    LangGraph node that assembles the final structured renovation report.
    Integrates the InsightEngine for model-driven, structured insights.
    Saves user session to memory for future personalisation.
    """

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        name = "report_agent"
        logger.info(f"[{state.get('project_id', '')}] ReportAgent v2.0 starting")

        try:
            updates = await self._generate(state)
        except Exception as e:
            logger.error(f"[report_agent] Error: {e}", exc_info=True)
            updates = {
                "renovation_report": {
                    "report_version": "2.0",
                    "project_id": state.get("project_id", ""),
                    "summary_headline": f"{state.get('theme', 'Modern')} renovation report",
                    "error": str(e),
                },
                "insights": {
                    "summary_headline": f"{state.get('theme', 'Modern')} renovation — {state.get('city', 'India')}",
                    "financial_outlook": {},
                    "visual_analysis": {},
                    "market_intelligence": {},
                    "budget_assessment": {},
                    "recommendations": [],
                    "risk_factors": [],
                    "top_materials": [],
                    "insight_engine": {},
                },
                "chat_context": f"Renovation: {state.get('theme')} in {state.get('city')}",
                "insight_engine_output": {},
                "errors": (state.get("errors") or []) + [f"report_agent: {e}"],
            }

        elapsed = round(time.perf_counter() - t0, 3)
        timings = dict(state.get("agent_timings") or {})
        timings[name] = elapsed
        updates["agent_timings"] = timings

        completed = list(state.get("completed_agents") or [])
        if name not in completed:
            completed.append(name)
        updates["completed_agents"] = completed

        logger.info(f"[report_agent] done in {elapsed}s")
        return updates

    async def _generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Run InsightEngine (model-driven structured insights)
        insight_output, renovation_insight = self._run_insight_engine(state)

        # 2. Generate legacy insights via existing pipeline insight agent
        insights, chat_context = self._build_insights(state, insight_output)

        # 3. Build structured RenovationReport
        report = self._build_report(state, insights, insight_output)

        # 4. Save session to memory
        await self._save_memory(state, insights)

        try:
            insight_engine_dict = insight_output.to_report_dict()
        except Exception:
            insight_engine_dict = renovation_insight.model_dump()

        return {
            "renovation_report": report,
            "insights": insights,
            "chat_context": chat_context,
            "insight_engine_output": insight_engine_dict,
        }

    def _run_insight_engine(self, state: Dict[str, Any]):
        try:
            from services.insight_engine.engine import InsightEngine

            rag_context = state.get("rag_context") or state.get("knowledge_context") or ""
            engine = InsightEngine()
            output, insight = engine.compute(state, rag_context=rag_context)

            logger.info(
                f"[report_agent] InsightEngine — "
                f"priority={output.renovation_priority_score:.1f} "
                f"CE={output.cost_effectiveness_score:.1f} "
                f"ROI={output.roi_score:.1f} "
                f"overall={output.overall_insight_score:.1f}"
            )
            return output, insight

        except Exception as e:
            logger.warning(f"[report_agent] InsightEngine failed: {e}")
            from services.insight_engine.models import InsightOutput, RenovationInsight
            return InsightOutput(), RenovationInsight()

    def _build_insights(self, state: Dict[str, Any], insight_output) -> tuple:
        try:
            # Read insights already populated by node_insight_generation (the dedicated pipeline step).
            # Do NOT call node_insight_generation here — it has already run upstream and calling it
            # again causes an infinite loop: node_report_generation → _build_insights →
            # node_insight_generation → node_report_generation → ...
            insights = dict(state.get("insights") or {})
            chat_context = state.get("chat_context", "")

            if state.get("layout_report"):
                insights["layout_analysis"] = state.get("layout_report", {})
            if state.get("explainable_recommendations"):
                insights["explainable_design_recommendations"] = state.get("explainable_recommendations", [])
            if state.get("user_goals"):
                goals = state["user_goals"]
                if isinstance(goals, dict):
                    insights["user_goal_summary"] = goals.get("primary_goal", "personal_comfort")
            if state.get("memory_context"):
                insights["memory_context"] = state["memory_context"]

            # Inject InsightEngine output
            try:
                ie_dict = insight_output.to_report_dict()
                insights["insight_engine"] = ie_dict
                insights["priority_repairs"] = ie_dict.get("priority_repairs", [])
                insights["budget_strategy"] = ie_dict.get("budget_strategy", {})
                insights["renovation_sequence"] = ie_dict.get("renovation_sequence", [])
                insights["expected_value_increase"] = ie_dict.get("expected_value_increase", "")
                insights["roi_score"] = ie_dict.get("roi_score", "")
                insights["renovation_priority_score"] = ie_dict.get("renovation_priority_score", 0)
                insights["cost_effectiveness_score"] = ie_dict.get("cost_effectiveness_score", 0)
                insights["overall_insight_score"] = ie_dict.get("overall_insight_score", 0)
                insights["key_wins"] = ie_dict.get("key_wins", [])
                insights["risk_flags"] = ie_dict.get("risk_flags", [])
                insights["insight_summary"] = ie_dict.get("summary", "")
            except Exception as ie_err:
                logger.warning(f"[report_agent] InsightEngine dict injection failed: {ie_err}")

            return insights, chat_context

        except Exception as e:
            logger.warning(f"[report_agent] Insight generation via pipeline failed: {e}")
            return self._static_insights(state, insight_output), ""

    def _build_report(self, state: Dict[str, Any], insights: Dict, insight_output) -> Dict:
        # Build the report directly from state — do NOT call node_report_generation here.
        # Calling it would cause an infinite loop:
        #   node_report_generation → ReportAgentNode._build_report → node_report_generation → ...
        report = dict(state.get("renovation_report") or state.get("final_report") or {})
        if not report:
            report = {
                "report_version": "2.0",
                "project_id": state.get("project_id", ""),
                "summary_headline": (
                    f"{state.get('theme', 'Modern')} renovation — {state.get('city', 'India')}"
                ),
            }

        # Inject InsightEngine data into the report
        try:
            ie_dict = insight_output.to_report_dict()
            report["insight_engine"] = ie_dict
            report["priority_repairs"] = ie_dict.get("priority_repairs", [])
            report["budget_strategy_insight"] = ie_dict.get("budget_strategy", {})
            report["renovation_sequence"] = ie_dict.get("renovation_sequence", [])
            report["expected_value_increase"] = ie_dict.get("expected_value_increase", "")
            report["roi_score"] = ie_dict.get("roi_score", "")
            report["renovation_priority_score"] = ie_dict.get("renovation_priority_score", 0)
            report["cost_effectiveness_score"] = ie_dict.get("cost_effectiveness_score", 0)
            report["overall_insight_score"] = ie_dict.get("overall_insight_score", 0)
            report["insight_summary"] = ie_dict.get("summary", "")
            report["key_wins"] = ie_dict.get("key_wins", [])
            report["risk_flags"] = ie_dict.get("risk_flags", [])

            # Enrich roi_forecast with insight data
            if isinstance(report.get("roi_forecast"), dict):
                roi_ins = ie_dict.get("roi_insight", {})
                report["roi_forecast"]["expected_value_increase"] = roi_ins.get("expected_value_increase", "")
                report["roi_forecast"]["value_per_rupee"] = roi_ins.get("value_per_rupee", 0)
                report["roi_forecast"]["cost_effectiveness_score"] = roi_ins.get("cost_effectiveness_score", 0)
                report["roi_forecast"]["roi_interpretation"] = roi_ins.get("interpretation", "")
        except Exception as inject_err:
            logger.warning(f"[report_agent] InsightEngine injection into report failed: {inject_err}")

        report["pipeline_summary"] = {
            "completed_agents": state.get("completed_agents", []),
            "agent_timings": state.get("agent_timings", {}),
            "errors": state.get("errors", []),
            "user_goal": (state.get("user_goals") or {}).get("primary_goal", "personal_comfort")
            if isinstance(state.get("user_goals"), dict) else "personal_comfort",
            "memory_used": bool(state.get("memory_context")),
            "insight_engine_version": "1.0",
        }

        # ── Indian homeowner sections ─────────────────────────────────────────
        report["procurement_calendar"] = self._build_procurement_calendar(state)
        report["contractor_checklist"] = self._build_contractor_checklist(state)
        report["seasonal_advice"]      = self._build_seasonal_advice()

        # ── Trust assessment (v2.1 addition — one line) ───────────────────────
        try:
            from analytics import TrustScoreEngine
            report["trust_assessment"] = TrustScoreEngine().compute(state)
        except Exception as _te:
            logger.debug(f"[report_agent] TrustScoreEngine failed (non-critical): {_te}")

        return report

    # ── Indian homeowner helper methods ──────────────────────────────────────

    def _build_procurement_calendar(self, state: Dict[str, Any]) -> Dict[str, str]:
        """
        Returns material procurement timing guidance relevant to the project.
        Entries are filtered to categories present in boq_line_items or room_type.
        """
        room_type: str = state.get("room_type", "bedroom")
        boq: list = state.get("boq_line_items") or []
        boq_categories = {
            str(item.get("category", "")).lower() for item in boq if isinstance(item, dict)
        }

        def _has(*keywords) -> bool:
            return (
                any(kw in room_type for kw in keywords)
                or any(kw in cat for kw in keywords for cat in boq_categories)
            )

        calendar: Dict[str, str] = {}

        # Paint — almost universal
        if _has("paint", "wall", "bedroom", "living", "kitchen", "bathroom", "full"):
            calendar["paint"] = (
                "Buy paint 2 weeks before painting starts. "
                "Avoid buying during festive season (Oct–Nov) when prices spike 5–8%."
            )

        # Tiles — relevant for tiling rooms
        if _has("tile", "flooring", "kitchen", "bathroom", "living", "full"):
            calendar["tiles"] = (
                "Order tiles 3 weeks before laying. "
                "Always order 10% extra for cuts and future repairs."
            )

        # Electrical — universal
        calendar["electrical"] = (
            "Buy branded switches (Legrand/Havells) early — "
            "local shops stock limited branded stock."
        )

        # Plumbing — wet rooms
        if _has("plumb", "bathroom", "kitchen", "full"):
            city: str = state.get("city", "")
            tier_note = (
                "delivery takes 3–5 days in Tier-1"
                if city in ("Mumbai", "Delhi NCR", "Bangalore", "Hyderabad", "Chennai", "Pune")
                else "delivery takes 7–10 days in Tier-2/3"
            )
            calendar["plumbing"] = (
                f"Buy Jaquar/Hindware fittings before plumber starts — {tier_note}."
            )

        return calendar

    def _build_contractor_checklist(self, state: Dict[str, Any]) -> list:
        """
        Returns 4–5 most relevant pre-start checklist items based on scope and room.
        """
        renovation_scope: str = (
            state.get("renovation_scope")
            or (state.get("room_features") or {}).get("renovation_scope")
            or "partial"
        )
        room_type: str = state.get("room_type", "bedroom")

        # Full checklist ordered by universal importance
        full_checklist = [
            "Get 3 quotes minimum — middle quote is usually most reliable.",
            "Contractor must provide written scope of work before starting.",
            "Never pay more than 30% advance. "
            "Structure as: 30% start + 40% mid + 30% completion.",
            "Insist on ISI-marked materials (cement, tiles, electrical wire).",
            "Check GST registration of contractor — unregistered contractors cannot provide GST invoice.",
            "Specify brands in contract (e.g. Asian Paints Royale, not just 'good paint').",
            "Get 1-year workmanship warranty in writing.",
            "Take timestamped photos before, during, and after each stage.",
        ]

        # Always include the first 3 (universal). Add context-specific ones.
        selected = full_checklist[:3]

        if renovation_scope in ("full_room", "structural_plus") or room_type == "full_home":
            # ISI materials + GST for major renovations
            selected += [full_checklist[3], full_checklist[4]]
        elif room_type in ("kitchen", "bathroom"):
            # Brand spec matters most for wet rooms
            selected += [full_checklist[5], full_checklist[6]]
        else:
            # Photo documentation + warranty for cosmetic work
            selected += [full_checklist[7], full_checklist[6]]

        return selected[:5]  # cap at 5

    def _build_seasonal_advice(self) -> str:
        """
        Returns seasonal renovation advice based on current calendar month.
        """
        import datetime
        month = datetime.datetime.now().month

        if month in (1, 2):
            return (
                "Good time to start — contractor availability is high and "
                "sand prices are stable post-monsoon."
            )
        elif month in (3, 4, 5):
            return (
                "Ideal season — dry weather helps paint and tile adhesive cure properly. "
                "Best window for outdoor and waterproofing work."
            )
        elif month in (6, 7, 8, 9):
            return (
                "Monsoon season — avoid waterproofing and outdoor work. "
                "Good time for indoor carpentry and painting."
            )
        elif month in (10, 11):
            return (
                "Festive season — material prices are higher and contractors are busy. "
                "Book early or wait for December for better rates."
            )
        else:  # December
            return (
                "Good availability and year-end discounts from suppliers. "
                "Ideal time to lock in contractor rates for a January start."
            )

    async def _save_memory(self, state: Dict[str, Any], insights: Dict) -> None:
        try:
            from memory.agent_memory import agent_memory

            user_goals = state.get("user_goals") or {}
            goals_list = []
            if isinstance(user_goals, dict):
                goal = user_goals.get("primary_goal", "")
                if goal:
                    goals_list = [goal]

            roi = state.get("roi_prediction") or state.get("roi_output") or {}
            if isinstance(roi, dict):
                roi_pct = roi.get("roi_pct", 0)
            else:
                roi_pct = 0

            memory_data = {
                "budget_inr": state.get("budget_inr", 0),
                "city": state.get("city", ""),
                "theme": state.get("theme", ""),
                "budget_tier": state.get("budget_tier", "mid"),
                "room_type": state.get("room_type", "bedroom"),
                "goals": goals_list,
                "roi_pct": roi_pct,
                "design_preferences": state.get("past_design_preferences", []),
                "style_label": state.get("style_label", ""),
                "insight_score": insights.get("insight_engine", {}).get("overall_insight_score", 0),
            }

            user_id = state.get("user_id", "anonymous")
            project_id = state.get("project_id", "")
            await agent_memory.save(user_id, project_id, memory_data)
        except Exception as me:
            logger.warning(f"[report_agent] Memory save failed: {me}")

    def _static_insights(self, state: Dict[str, Any], insight_output) -> Dict:
        theme = state.get("theme", "Modern Minimalist")
        city = state.get("city", "India")
        roi = state.get("roi_prediction") or {}
        roi_pct = roi.get("roi_pct", 0) if isinstance(roi, dict) else 0

        base = {
            "summary_headline": f"{theme} renovation in {city} — {roi_pct:.1f}% projected ROI",
            "financial_outlook": {},
            "visual_analysis": {},
            "market_intelligence": {},
            "budget_assessment": {},
            "recommendations": [],
            "risk_factors": [],
            "top_materials": [],
        }

        try:
            ie_dict = insight_output.to_report_dict()
            base["insight_engine"] = ie_dict
            base["priority_repairs"] = ie_dict.get("priority_repairs", [])
            base["budget_strategy"] = ie_dict.get("budget_strategy", {})
            base["renovation_sequence"] = ie_dict.get("renovation_sequence", [])
            base["expected_value_increase"] = ie_dict.get("expected_value_increase", "")
            base["roi_score"] = ie_dict.get("roi_score", "")
        except Exception:
            pass

        return base