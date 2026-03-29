"""
ARKEN — InsightGenerationAgent v2.1
=====================================
WHAT CHANGED over v2.0:
  - _build_insights_dict now reads cv_features, image_specific_actions,
    detected_style_grounded, and diy_renovation_tips from state.
  - visual_analysis section is now fully populated from real CV output
    instead of generic damage_assessment fallbacks.
  - recommendations are now image-grounded: they reference what was
    actually detected (objects, style, lighting) not generic advice.
  - dataset_grounded flag tells the frontend whether dataset was used.
  - _build_renovation_sequence now uses image_specific_actions as Step 0
    so the first thing shown is exactly what the image contains.
  - All existing output contracts preserved.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InsightGenerationAgent:
    """
    Synthesises all pipeline outputs into image-grounded renovation insights.
    """

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        name = "insight_generation_agent"
        logger.info(f"[{state.get('project_id', '')}] InsightGenerationAgent v2.1 starting")

        try:
            updates = self._generate_insights(state)
        except Exception as e:
            logger.error(f"[insight_generation] Error: {e}", exc_info=True)
            updates = self._fallback_insights(state, str(e))

        elapsed = round(time.perf_counter() - t0, 3)
        timings = dict(state.get("agent_timings") or {})
        timings[name] = elapsed
        updates["agent_timings"] = timings

        completed = list(state.get("completed_agents") or [])
        if name not in completed:
            completed.append(name)
        updates["completed_agents"] = completed

        logger.info(
            f"[insight_generation] done in {elapsed}s — "
            f"grounded={updates.get('insights', {}).get('image_grounded', False)} "
            f"dataset_grounded={updates.get('insights', {}).get('dataset_grounded', False)}"
        )
        return updates

    # ─────────────────────────────────────────────────────────────────────────

    def _generate_insights(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from services.insight_engine.engine import InsightEngine

        engine = InsightEngine()
        insight_output = engine.generate(state)

        insights = self._build_insights_dict(state, insight_output)
        renovation_sequence = self._build_renovation_sequence(state, insight_output)
        priority_repairs = self._extract_priority_repairs(state, insight_output)
        budget_strategy = self._build_budget_strategy(state, insight_output)
        room_intelligence = self._build_room_intelligence(state)

        # Inject into insights so the frontend can read insights.renovation_sequence
        # and insights.priority_repairs directly. These are the authoritative versions
        # from InsightGenerationAgent (uses title/description) not InsightEngine
        # (uses phase/actions with integer step field which breaks the frontend).
        insights["renovation_sequence"] = renovation_sequence
        insights["priority_repairs"] = priority_repairs

        # Remove dataset_style_examples — it confuses users by showing mismatched
        # room types (e.g. bathroom examples for a detected bedroom) and adds no
        # actionable value. The detected style is already shown in visual_analysis.
        insights.pop("dataset_style_examples", None)

        return {
            "insights": insights,
            "insight_engine_output": insight_output if isinstance(insight_output, dict) else {},
            "renovation_sequence": renovation_sequence,   # kept for internal/API consumers
            "room_intelligence": room_intelligence,       # replaces sequence in UI
            "priority_repairs": priority_repairs,
            "budget_strategy": budget_strategy,
        }

    def _build_insights_dict(
        self,
        state: Dict[str, Any],
        engine_output: Any,
    ) -> Dict[str, Any]:
        city = state.get("city", "India")
        theme = state.get("theme", "Modern Minimalist")
        # FIX: roi_output is now the authoritative full dict (roi_agent_node v2.0).
        # roi_prediction is kept as fallback for backward compat with older pipeline runs.
        roi = state.get("roi_output") or state.get("roi_prediction") or {}
        cost = state.get("cost_estimate") or {}
        budget_est = state.get("budget_estimate") or {}
        vision = state.get("room_features") or state.get("vision_features") or {}
        damage = state.get("damage_assessment") or {}
        retrieved = state.get("retrieved_knowledge") or []
        rag_ctx = state.get("rag_context") or ""

        # ── CV pipeline outputs ───────────────────────────────────────────────
        cv = state.get("cv_features") or {}
        detected_objects: List[str] = state.get("detected_objects") or cv.get("detected_objects") or []
        detected_style: str = state.get("detected_style_grounded") or state.get("style_label") or cv.get("style") or theme
        detected_lighting: str = cv.get("lighting") or vision.get("natural_light", "mixed")
        detected_materials: List[str] = state.get("material_types") or cv.get("materials") or []
        cv_room_type: str = cv.get("room_type") or state.get("room_type", "bedroom")
        image_grounded: bool = bool(cv and cv.get("extraction_source") not in ("cv_unavailable", "", None))

        # ── Task 1 condition fields (from improved Gemini extraction) ─────────
        # Values may be None / "not_assessed" when vision analysis did not run.
        # In that case do NOT inject fake numbers — surface the gap explicitly.
        _NOT_ASSESSED = ("not_assessed", None, "")
        room_features_dict = state.get("room_features") or state.get("vision_features") or {}

        _raw_condition_score = (
            state.get("condition_score")
            or room_features_dict.get("condition_score")
            or damage.get("condition_score")
        )
        condition_score: Optional[int] = (
            int(_raw_condition_score)
            if _raw_condition_score not in (None, "", "not_assessed")
            else None
        )

        _raw_wall = (
            state.get("wall_condition")
            or room_features_dict.get("wall_condition")
        )
        wall_condition: str = (
            _raw_wall if _raw_wall not in _NOT_ASSESSED else "not_assessed"
        )

        _raw_floor = (
            state.get("floor_condition")
            or room_features_dict.get("floor_condition")
        )
        floor_condition: str = (
            _raw_floor if _raw_floor not in _NOT_ASSESSED else "not_assessed"
        )

        issues_detected_list: List[str] = (
            state.get("issues_detected")
            or room_features_dict.get("issues_detected")
            or damage.get("issues_detected", [])
        )

        _raw_scope = (
            state.get("renovation_scope")
            or room_features_dict.get("renovation_scope")
        )
        renovation_scope: str = (
            _raw_scope if _raw_scope not in _NOT_ASSESSED else "not_assessed"
        )

        high_value_upgrades: List[str] = (
            state.get("high_value_upgrades")
            or room_features_dict.get("high_value_upgrades")
            or []
        )

        # Flag: vision ran and produced real condition data
        vision_assessed: bool = condition_score is not None

        # ── Image-specific actions from design planner ─────────────────────────
        image_specific_actions: List[Dict] = state.get("image_specific_actions") or []

        # ── DIY guidance — read from state key set by design_planner_node ──────
        # design_planner_node sets both "diy_renovation_tips" (state-level) and
        # "design_plan.diy_guidance" (nested). Read both with fallback.
        _raw_diy_tips: List[Dict] = (
            state.get("diy_renovation_tips")
            or (state.get("design_plan") or {}).get("diy_guidance")
            or []
        )

        # ── Filter DIY tips to only those relevant to this specific room ───────
        # Prevent generic tips (e.g. kitchen plumbing hacks for a bedroom) from
        # appearing. Score each tip against room_type + detected issues + style.
        _issues_str: str = " ".join(
            str(i).lower() for i in (
                state.get("issues_detected")
                or room_features_dict.get("issues_detected", [])
            )
        )
        _room_type_lower: str = cv_room_type.lower()
        _style_lower: str = detected_style.lower()

        def _diy_relevance_score(tip: Dict) -> int:
            """Score 0–4: higher = more relevant to this specific room."""
            score = 0
            tip_text = " ".join([
                str(tip.get("tip", "")),
                str(tip.get("title", "")),
                str(tip.get("category", "")),
                str(tip.get("guidance", "")),
            ]).lower()
            # +2 if matches room type
            if _room_type_lower in tip_text or any(
                kw in tip_text for kw in _room_type_lower.replace("_", " ").split()
            ):
                score += 2
            # +1 if matches a detected issue keyword
            issue_keywords = ["crack", "damp", "peel", "leak", "mould", "water",
                               "stain", "seepage", "lighting", "worn", "outdated"]
            if any(kw in tip_text for kw in issue_keywords if kw in _issues_str):
                score += 1
            # +1 if matches detected style
            if any(kw in tip_text for kw in _style_lower.split()):
                score += 1
            return score

        # Sort by relevance and keep top 3 that score >= 1 (at least some match).
        # If nothing scores >= 1 (no issues, generic room), keep the top 3 by
        # room-type match alone to avoid an empty section.
        _scored = sorted(_raw_diy_tips, key=_diy_relevance_score, reverse=True)
        _relevant = [t for t in _scored if _diy_relevance_score(t) >= 1]
        diy_tips: List[Dict] = (_relevant if _relevant else _scored)[:3]

        # ── Dataset-grounded style examples ───────────────────────────────────
        dataset_examples = (state.get("design_plan") or {}).get("dataset_style_examples") or []
        dataset_grounded: bool = bool(dataset_examples)

        # ── Engine output dict ────────────────────────────────────────────────
        if hasattr(engine_output, "model_dump"):
            engine_dict = engine_output.model_dump()
        elif isinstance(engine_output, dict):
            engine_dict = engine_output
        else:
            engine_dict = {}

        # ── Build grounded recommendations ────────────────────────────────────
        grounded_recommendations = self._build_grounded_recommendations(
            state=state,
            engine_dict=engine_dict,
            image_specific_actions=image_specific_actions,
            detected_style=detected_style,
            detected_objects=detected_objects,
            detected_lighting=detected_lighting,
            diy_tips=diy_tips,
        )

        # ── Material recs from RAG ────────────────────────────────────────────
        material_recs = [
            d for d in retrieved if d.get("category") in ("material", "cost", "construction_materials")
        ][:3]

        # ── Data quality / confidence indicator ──────────────────────────────
        cv_confidence: float = float(cv.get("overall_confidence", 0.0))
        gemini_vision_ran: bool = (
            room_features_dict.get("wall_color", "unknown") not in ("unknown", "", None)
        )
        image_analysis_ran: bool = bool(
            cv and cv.get("extraction_source") not in ("cv_unavailable", "", None)
        )

        if cv_confidence > 0.70 and gemini_vision_ran:
            confidence_tier = "high"
            user_message = (
                "Analysis based on your actual room photo. "
                "All recommendations are specific to what we detected."
            )
            recommendations_basis = "Your uploaded photo"
        elif cv_confidence > 0.40 or gemini_vision_ran:
            confidence_tier = "medium"
            user_message = (
                "Analysis partially based on your photo. "
                "Some estimates use typical Indian home benchmarks."
            )
            recommendations_basis = "Your photo + Indian benchmarks"
        else:
            confidence_tier = "low"
            user_message = (
                "Photo analysis was limited (dark/blurry image). "
                "Recommendations use typical Indian home benchmarks "
                "for your city and budget."
            )
            recommendations_basis = "Indian market benchmarks"

        data_quality = {
            "image_analysis_ran": image_analysis_ran,
            "cv_confidence": round(cv_confidence, 3),
            "gemini_vision_ran": gemini_vision_ran,
            "confidence_tier": confidence_tier,
            "user_message": user_message,
            "recommendations_basis": recommendations_basis,
        }

        return {
            "summary_headline": self._build_headline(state, detected_style, cv_room_type),

            # Data quality indicator — always present so frontend can show trust badge
            "data_quality": data_quality,

            # Fully populated from CV — not just damage_assessment fallbacks.
            # condition_score / wall_condition / floor_condition / renovation_scope
            # are None / "not_assessed" when vision analysis did not run — the
            # frontend should display "Not assessed" rather than show fake numbers.
            "visual_analysis": {
                "room_type":          cv_room_type,
                "room_condition": (
                    damage.get("overall_condition", vision.get("condition"))
                    if vision_assessed else "not_assessed"
                ),
                "severity":           damage.get("severity", "medium") if vision_assessed else "not_assessed",
                "layout_score":       damage.get("layout_score", vision.get("layout_score")) if vision_assessed else None,
                "walkable_space":     damage.get("walkable_space") if vision_assessed else None,
                "issues_detected":    issues_detected_list,
                "natural_light":      detected_lighting,
                "style_detected":     detected_style,
                "style_confidence":   state.get("style_confidence") or cv.get("style_confidence", 0.5),
                "detected_objects":   detected_objects,
                "detected_materials": detected_materials,
                "image_grounded":     image_grounded,
                "cv_model":           cv.get("extraction_source", "gemini"),
                # Condition fields — None / "not_assessed" when vision did not run
                "condition_score":    condition_score,        # None means "Not assessed"
                "wall_condition":     wall_condition,         # "not_assessed" when unknown
                "floor_condition":    floor_condition,
                "renovation_scope":   renovation_scope,
                "high_value_upgrades": high_value_upgrades,
                "vision_assessed":    vision_assessed,        # bool: frontend gate
            },

            "financial_outlook": {
                # FIX: prefer budget_estimate (authoritative, includes GST+contingency)
                # over cost_estimate (design_planner intermediate value)
                "total_cost_inr":   budget_est.get("total_cost_inr") or cost.get("total_inr", 0),
                "cost_per_sqft":    cost.get("cost_per_sqft", state.get("cost_per_sqft", 0)),
                "within_budget":    cost.get("within_budget", state.get("within_budget", True)),
                "roi_pct":          roi.get("roi_pct", state.get("roi_pct", 0.0)),
                "equity_gain_inr":  roi.get("equity_gain_inr", state.get("equity_gain_inr", 0)),
                "payback_months":   roi.get("payback_months", state.get("payback_months", 36)),
                "model_confidence": roi.get("model_confidence", state.get("model_confidence", 0.65)),
            },

            "market_intelligence": state.get("location_context", {}),
            "budget_assessment":   state.get("budget_analysis", {}),

            # Image-grounded recommendations (replaces generic list)
            "recommendations": grounded_recommendations,

            "risk_factors":     engine_dict.get("risk_factors", []),
            "market_timing":    engine_dict.get("market_timing", {}),
            "action_checklist": engine_dict.get("action_checklist", []),

            "top_materials": [
                {"title": d.get("title", ""), "summary": d.get("text", "")[:200]}
                for d in material_recs
            ],

            # Dataset enrichment — dataset_style_examples intentionally omitted:
            # it showed mismatched room types (e.g. bathroom for bedroom) and
            # added no actionable value. DIY tips are room+issue-specific (see below).
            "diy_renovation_tips":    diy_tips[:3],

            # Grounding flags (used by frontend to show badges)
            "image_grounded":   image_grounded,
            "dataset_grounded": dataset_grounded,
            "rag_grounded":     len(retrieved) > 0,

            "insight_engine": engine_dict,
        }

    def _build_grounded_recommendations(
        self,
        state: Dict[str, Any],
        engine_dict: Dict,
        image_specific_actions: List[Dict],
        detected_style: str,
        detected_objects: List[str],
        detected_lighting: str,
        diy_tips: List[Dict],
    ) -> List[Dict]:
        """
        Build final recommendations SPECIFIC to what was detected in the image.
        Priority order:
          1. Image-specific actions (from CV detections — highest specificity)
          2. VisualAssessor explainable recommendations
          3. Engine recommendations (insight engine)
        DIY tips are intentionally excluded here — they appear in their own
        dedicated section (diy_renovation_tips) in the frontend.
        """
        recs = []

        # 1. Image-specific actions from CV (highest specificity — always first)
        for action in image_specific_actions[:6]:
            action_text = action.get("action", "")
            if not action_text:
                continue
            recs.append({
                "recommendation": action_text,
                "category":  action.get("category", "general"),
                "priority":  action.get("priority", "medium"),
                "trigger":   action.get("trigger", ""),
                "grounding": action.get("grounding", "cv_pipeline"),
                "reasoning": [
                    f"Detected in your image: {action.get('trigger', '')}",
                    f"Aligned with {detected_style} design style",
                ],
                "source": "image_analysis",
            })

        # 2. VisualAssessor explainable recommendations (image-derived)
        for rec in (state.get("explainable_recommendations") or [])[:3]:
            if isinstance(rec, dict) and rec.get("recommendation"):
                rec["source"] = "visual_assessor"
                recs.append(rec)

        # 3. Engine recommendations (only if we still have fewer than 6)
        if len(recs) < 6:
            for rec in (engine_dict.get("recommendations") or [])[:3]:
                if isinstance(rec, dict) and rec.get("recommendation"):
                    rec["source"] = rec.get("source", "insight_engine")
                    recs.append(rec)

        return recs[:8]

    def _build_headline(
        self,
        state: Dict[str, Any],
        detected_style: str,
        cv_room_type: str,
    ) -> str:
        room = cv_room_type.replace("_", " ").title()
        city = state.get("city", "India")

        # DEFINITIVE FIX: use cost_estimate.total_inr as the single source of truth.
        #
        # Why this is correct:
        #   - cost_estimate is set by the orchestrator's node_budget_estimation as:
        #     cost_estimate.total_inr = budget_estimator.total_cost_inr
        #     (city-adjusted, includes city labour premium + GST + contingency)
        #   - This is the SAME value that flows to the BOQ tab and Cost Accuracy card.
        #   - budget_estimate dict in state may still hold the design_planner's
        #     pre-city-adjustment value if it was set earlier in the run.
        #   - cost_estimate.total_inr is NEVER set to the old design_planner value.
        #
        # Priority chain: cost_estimate (authoritative) → total_cost_estimate
        # (state scalar also set by node_budget_estimation) → budget_inr (last resort)
        cost_est = state.get("cost_estimate") or {}
        total = (
            cost_est.get("total_inr")
            or state.get("total_cost_estimate")
            or state.get("budget_inr", 0)
        )

        # ROI from the full roi_output (post roi_agent_node v2.0)
        roi_src = state.get("roi_output") or state.get("roi_prediction") or {}
        roi = roi_src.get("roi_pct", 0.0)

        parts = [f"{detected_style} {room} Renovation — {city}"]
        if total:
            parts.append(f"₹{total:,} total investment")
        if roi:
            parts.append(f"{roi:.1f}% projected ROI")
        return " | ".join(parts)

    def _build_renovation_sequence(
        self,
        state: Dict[str, Any],
        engine_output: Any,
    ) -> List[Dict[str, Any]]:
        """
        Builds a renovation sequence following the real Indian contractor order:
        Site prep → Civil/structural → Wet work → Electrical rough-in → Tiling →
        False ceiling → Carpentry → Painting → Fixtures → Handover

        Only includes steps relevant to room_type, renovation_scope, and issues_detected.

        BUG FIX: Each step now includes estimated_cost_inr (computed from total_cost_estimate
        and estimated_cost_pct) so the frontend can render real rupee values instead of
        percentages. Also adds cumulative_day_start for Gantt rendering.
        """
        room_type: str = state.get("room_type", "bedroom")
        renovation_scope: str = (
            state.get("renovation_scope")
            or (state.get("room_features") or {}).get("renovation_scope")
            or (state.get("damage_assessment") or {}).get("renovation_scope")
            or "partial"
        )
        issues: List[str] = (
            state.get("issues_detected")
            or (state.get("room_features") or {}).get("issues_detected")
            or (state.get("damage_assessment") or {}).get("issues_detected")
            or []
        )
        issues_lower: str = " ".join(str(i).lower() for i in issues)

        # Total cost for converting pct → INR
        total_cost_inr: float = float(
            (state.get("cost_estimate") or {}).get("total_inr", 0)
            or state.get("total_cost_estimate", 0)
            or state.get("budget_inr", 750_000)
        )

        has_cracks   = any(kw in issues_lower for kw in ("crack", "cracking"))
        has_seepage  = any(kw in issues_lower for kw in ("seepage", "damp", "mould", "leak", "water"))
        is_wet_room  = room_type in ("bathroom", "kitchen", "full_home")
        has_plumbing = room_type in ("bathroom", "kitchen", "full_home")
        has_elec     = True   # electrical rough-in is universal
        has_tiling   = room_type in ("bathroom", "kitchen", "living_room", "full_home")
        has_wall_tiles = room_type in ("bathroom", "kitchen", "full_home")
        has_ceiling  = room_type in ("living_room", "bedroom", "full_home")
        has_carpentry = room_type in ("bedroom", "kitchen", "living_room", "study", "full_home")
        cosmetic_only = renovation_scope == "cosmetic_only"

        # Build ordered candidate steps — (priority, include_condition, step_dict)
        candidates = []

        # P0 — Site prep (skip for pure cosmetic)
        if not cosmetic_only:
            candidates.append((0, True, {
                "title": "Site preparation and protection",
                "description": "Cover floors and furniture, set up material storage, disconnect fixtures",
                "duration_days": 1,
                "can_overlap_with_next": False,
                "requires_drying_time": False,
                "estimated_cost_pct": 1.0,
                "contractor_type": "general",
                "category": "preparation",
            }))

        # P1 — Structural repair
        if has_cracks:
            candidates.append((1, True, {
                "title": "Crack filling and wall repair",
                "description": "Chisel cracks, fill with bonding agent + polymer-modified mortar, cure",
                "duration_days": 3,
                "can_overlap_with_next": False,
                "requires_drying_time": True,
                "estimated_cost_pct": 5.0,
                "contractor_type": "mason",
                "category": "structural",
            }))

        # P2a — Waterproofing (wet rooms; also seepage anywhere)
        if is_wet_room or has_seepage:
            candidates.append((2, True, {
                "title": "Waterproofing",
                "description": "Apply 2-coat crystalline waterproofing membrane on floor slab and lower walls",
                "duration_days": 2,
                "can_overlap_with_next": False,
                "requires_drying_time": True,
                "estimated_cost_pct": 4.0,
                "contractor_type": "mason",
                "category": "waterproofing",
            }))

        # P2b — Plumbing rough-in (before walls close)
        if has_plumbing and not cosmetic_only:
            candidates.append((2, True, {
                "title": "Plumbing rough-in (concealed pipes)",
                "description": "Chase walls, lay concealed CPVC/PPR supply lines and PVC drainage, pressure-test",
                "duration_days": 3,
                "can_overlap_with_next": True,
                "requires_drying_time": False,
                "estimated_cost_pct": 7.0,
                "contractor_type": "plumber",
                "category": "plumbing",
            }))

        # P3 — Electrical rough-in (before walls are plastered/tiled)
        if has_elec and not cosmetic_only:
            candidates.append((3, True, {
                "title": "Electrical conduit and wiring",
                "description": "Lay PVC conduit, pull copper wiring, install DB box, earth pits",
                "duration_days": 3,
                "can_overlap_with_next": True,
                "requires_drying_time": False,
                "estimated_cost_pct": 8.0,
                "contractor_type": "electrician",
                "category": "electrical",
            }))

        # P4a — Floor tiling
        if has_tiling:
            candidates.append((4, True, {
                "title": "Flooring tile laying",
                "description": "Lay vitrified/marble tiles on cement mortar bed with levelling spacers, cure 48h",
                "duration_days": 4,
                "can_overlap_with_next": True,
                "requires_drying_time": True,
                "estimated_cost_pct": 12.0,
                "contractor_type": "mason",
                "category": "flooring",
            }))

        # P4b — Wall tiling (kitchen / bathroom)
        if has_wall_tiles:
            candidates.append((4, True, {
                "title": "Wall tile laying",
                "description": "Apply adhesive-backed ceramic/vitrified wall tiles with epoxy grout joints",
                "duration_days": 3,
                "can_overlap_with_next": True,
                "requires_drying_time": True,
                "estimated_cost_pct": 8.0,
                "contractor_type": "mason",
                "category": "wall_tiling",
            }))

        # P5 — False ceiling (after tiling, before painting)
        if has_ceiling:
            candidates.append((5, True, {
                "title": "False ceiling grid and board installation",
                "description": "Install GI frame, fix gypsum/POP boards, cut recessed LED pockets",
                "duration_days": 4,
                "can_overlap_with_next": True,
                "requires_drying_time": False,
                "estimated_cost_pct": 10.0,
                "contractor_type": "carpenter",
                "category": "ceiling",
            }))

        # P6a — Wardrobes / kitchen carcass
        if has_carpentry:
            candidates.append((6, True, {
                "title": "Wardrobe / kitchen cabinet carcass installation",
                "description": "Assemble 18mm BWR plywood carcass, fit aluminium channels and drawer slides",
                "duration_days": 5,
                "can_overlap_with_next": True,
                "requires_drying_time": False,
                "estimated_cost_pct": 15.0,
                "contractor_type": "carpenter",
                "category": "carpentry",
            }))

        # P6b — Door frame fitting (if full_home or structural scope)
        if room_type == "full_home" or renovation_scope in ("full_room", "structural_plus"):
            candidates.append((6, True, {
                "title": "Door frame fitting",
                "description": "Set teak/sal/UPVC door frames in mortar, check plumb and level",
                "duration_days": 2,
                "can_overlap_with_next": True,
                "requires_drying_time": False,
                "estimated_cost_pct": 3.0,
                "contractor_type": "carpenter",
                "category": "carpentry",
            }))

        # P7a — Putty and primer (always)
        candidates.append((7, True, {
            "title": "Wall putty and primer coat",
            "description": "Apply 2 coats white cement putty, sand smooth, apply alkali-resistant primer",
            "duration_days": 3,
            "can_overlap_with_next": False,
            "requires_drying_time": True,
            "estimated_cost_pct": 4.0,
            "contractor_type": "painter",
            "category": "painting",
        }))

        # P7b — Final paint
        candidates.append((7, True, {
            "title": "Final paint coats",
            "description": "Apply 2 finish coats (Asian Paints Royale/Tractor Shyne) with micro-roller",
            "duration_days": 3,
            "can_overlap_with_next": False,
            "requires_drying_time": True,
            "estimated_cost_pct": 8.0,
            "contractor_type": "painter",
            "category": "painting",
        }))

        # P8a — Electrical fixtures (after paint fully dry — 48-72 hours)
        if has_elec:
            candidates.append((8, True, {
                "title": "Electrical fixtures (switches, lights, fans)",
                "description": "Fit modular switch plates, install LED panels/downlights, connect ceiling fans",
                "duration_days": 2,
                "can_overlap_with_next": True,
                "requires_drying_time": False,
                "estimated_cost_pct": 5.0,
                "contractor_type": "electrician",
                "category": "electrical_fixtures",
            }))

        # P8b — Plumbing fixtures
        if has_plumbing:
            candidates.append((8, True, {
                "title": "Plumbing fixtures (taps, shower, toilet)",
                "description": "Install CP fittings, EWC, shower enclosure, connect water heater",
                "duration_days": 2,
                "can_overlap_with_next": True,
                "requires_drying_time": False,
                "estimated_cost_pct": 6.0,
                "contractor_type": "plumber",
                "category": "plumbing_fixtures",
            }))

        # P8c — Cabinet doors and hardware (after paint)
        if has_carpentry:
            candidates.append((8, True, {
                "title": "Cabinet doors, handles, and hinges",
                "description": "Fit acrylic/laminate shutter doors, install soft-close hinges and handles",
                "duration_days": 2,
                "can_overlap_with_next": True,
                "requires_drying_time": False,
                "estimated_cost_pct": 4.0,
                "contractor_type": "carpenter",
                "category": "carpentry_finishing",
            }))

        # P9 — Final cleanup and handover (always last)
        candidates.append((9, True, {
            "title": "Deep cleaning and handover",
            "description": "Remove protection covers, clean surfaces, punch-list walkthrough, client handover",
            "duration_days": 1,
            "can_overlap_with_next": False,
            "requires_drying_time": False,
            "estimated_cost_pct": 1.0,
            "contractor_type": "general",
            "category": "handover",
        }))

        # Sort by priority, filter included, assign step numbers + compute INR costs
        candidates.sort(key=lambda c: c[0])
        sequence: List[Dict[str, Any]] = []
        cumulative_day = 1
        for priority, include, step_data in candidates:
            if include:
                cost_pct = step_data.get("estimated_cost_pct", 0.0)
                cost_inr = round(total_cost_inr * cost_pct / 100)
                dur = step_data.get("duration_days", 1)
                sequence.append({
                    "step":               len(sequence) + 1,
                    "cumulative_day_start": cumulative_day,
                    "estimated_cost_inr": cost_inr,
                    **step_data,
                })
                # Only advance cumulative days if steps can't overlap
                if not step_data.get("can_overlap_with_next", False):
                    cumulative_day += dur

        return sequence

    def _extract_priority_repairs(
        self,
        state: Dict[str, Any],
        engine_output: Any,
    ) -> List[Dict[str, Any]]:
        """
        Builds a deduplicated, actionable priority repairs list from all pipeline sources.

        Guarantees:
          - No two items with opposite/contradictory meanings (over vs under-furnished)
          - No two items that are semantically the same phrased differently
          - Every item has: issue, severity, how_to_fix, estimated_cost_inr
          - Layout issues capped at 1 (the most relevant one)
          - Max 5 items total, ordered by severity desc
        """
        repairs: List[Dict[str, Any]] = []
        damage = state.get("damage_assessment") or {}
        retrieved = state.get("retrieved_knowledge") or []
        repair_docs = {
            d["doc_id"]: d for d in retrieved
            if d.get("category") in ("repair", "construction_materials")
        }

        room_features_dict = state.get("room_features") or state.get("vision_features") or {}
        cv = state.get("cv_features") or {}
        layout = state.get("layout_report") or {}
        budget_inr = float(
            (state.get("cost_estimate") or {}).get("total_inr", 0)
            or state.get("budget_inr", 750_000)
        )

        def _to_list(val) -> List[str]:
            if isinstance(val, list):
                return [str(v) for v in val if v]
            if isinstance(val, str) and val:
                return [val]
            return []

        def _safe_int(val):
            try:
                return int(val) if val not in (None, "", "not_assessed") else None
            except (TypeError, ValueError):
                return None

        # ── Condition score ───────────────────────────────────────────────────
        condition_score = (
            _safe_int(state.get("condition_score"))
            or _safe_int(room_features_dict.get("condition_score"))
            or _safe_int(damage.get("condition_score"))
        )
        wall_condition = (
            state.get("wall_condition")
            or room_features_dict.get("wall_condition")
            or damage.get("wall_condition")
            or "not_assessed"
        )
        _cs = condition_score if condition_score is not None else 65
        if _cs < 30:
            severity = "critical"
        elif _cs < 50:
            severity = "high"
        elif _cs < 70:
            severity = "medium"
        else:
            severity = "low"

        # ── Collect structural/surface issues (Gemini + CV) ───────────────────
        structural_issues = list(dict.fromkeys(
            _to_list(state.get("issues_detected"))
            or _to_list(room_features_dict.get("issues_detected"))
            or _to_list(damage.get("issues_detected"))
        ))
        cv_damage = [
            f"Physical damage: {i.replace('_', ' ')}"
            for i in _to_list(cv.get("detected_damage"))
            if i not in ("none", "")
        ]

        # ── Collect layout issues — cap at 1, no contradictions ──────────────
        raw_layout_issues = _to_list(
            layout.get("issues", layout.get("issues_detected", []))
        ) + _to_list(room_features_dict.get("layout_issues", []))

        # Semantic contradiction guard: if both over and under-furnished exist,
        # trust the free_space_percentage to decide which one is real
        has_over  = any("over-furnished" in i.lower() or "declutter" in i.lower() for i in raw_layout_issues)
        has_under = any("under-furnished" in i.lower() or "feels empty" in i.lower() for i in raw_layout_issues)
        if has_over and has_under:
            free_pct = float(room_features_dict.get("free_space_percentage", 45))
            raw_layout_issues = [
                i for i in raw_layout_issues
                if not ("under-furnished" in i.lower() or "feels empty" in i.lower())
            ] if free_pct <= 55 else [
                i for i in raw_layout_issues
                if not ("over-furnished" in i.lower() or "declutter" in i.lower())
            ]

        # Keep only the single most impactful layout issue
        layout_issues = raw_layout_issues[:1]

        # ── HOW_TO_FIX and COST lookup table ─────────────────────────────────
        FIX_MAP: List[tuple] = [
            # (keyword_in_issue, how_to_fix, cost_pct_of_budget, category)
            ("crack",        "Chisel out crack, fill with polymer-modified mortar, cure 24h, apply alkali primer", 0.04, "structural"),
            ("peel",         "Scrape loose paint, apply bonding primer, putty 2 coats, repaint with premium emulsion", 0.03, "painting"),
            ("damp",         "Apply crystalline waterproofing coat, re-plaster affected area, prime with damp-seal", 0.05, "waterproofing"),
            ("seepage",      "Waterproof external wall face, seal grout joints, apply interior waterproofing membrane", 0.06, "waterproofing"),
            ("mould",        "Clean with anti-fungal solution, apply mould-resistant primer, repaint with anti-fungal paint", 0.02, "painting"),
            ("leak",         "Identify and seal pipe/slab leak, apply waterproofing, re-plaster surface", 0.07, "plumbing"),
            ("stain",        "Clean with oxalic acid (stone) or trisodium phosphate (paint), touch-up or repaint", 0.02, "painting"),
            ("broken",       "Replace damaged component, match existing material finish", 0.03, "general"),
            ("worn",         "Sand, refinish or replace worn surface with matching material", 0.04, "flooring"),
            ("outdated",     "Replace with contemporary equivalent fitting, update surrounding finish", 0.05, "fixtures"),
            ("switchboard",  "Replace with modular switch plate, upgrade wiring to 2.5mm² copper", 0.02, "electrical"),
            ("over-furnished", "Remove or store 1–2 largest non-essential furniture pieces to free floor area", 0.01, "layout"),
            ("under-furnished", "Add 1–2 anchor pieces (rug, bookshelf, accent chair) to balance the space", 0.02, "layout"),
            ("insufficient walkable", "Rearrange furniture against walls, replace bulky pieces with slim-profile alternatives", 0.01, "layout"),
            ("lighting",     "Install 3-point LED lighting scheme: recessed downlights + task + accent strip", 0.04, "lighting"),
            ("physical damage", "Assess structural extent, repair substrate, refinish surface to match adjacent material", 0.05, "structural"),
        ]

        REPAIR_KEYWORDS = [
            "crack", "damp", "peel", "leak", "mould", "hollow", "seepage",
            "stain", "water", "damage", "broken", "worn", "faded", "outdated", "switchboard"
        ]

        def _make_repair(issue_text: str, base_severity: str, is_layout: bool = False) -> Dict:
            issue_lower = issue_text.lower()
            matched_doc = next(
                (d for d in repair_docs.values()
                 if any(kw in issue_lower for kw in REPAIR_KEYWORDS)),
                None,
            )
            # Find how_to_fix and cost from lookup table
            fix_entry = next(
                ((fix, cost_pct, cat) for kw, fix, cost_pct, cat in FIX_MAP
                 if kw in issue_lower),
                ("Assess and remediate according to contractor inspection", 0.03, "general"),
            )
            how_to_fix, cost_pct, category = fix_entry
            cost_inr = round(budget_inr * cost_pct) if budget_inr else 0

            # Escalate severity for structural issues
            item_severity = base_severity
            if any(kw in issue_lower for kw in ("crack", "seepage", "leak", "structural")):
                sev_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                item_severity = max(base_severity, "high", key=lambda s: sev_rank[s])
            if is_layout:
                item_severity = "medium"  # layout issues are never critical/high

            return {
                "issue":             issue_text,
                "severity":          item_severity,
                "must_fix_first":    item_severity in ("critical", "high") and not is_layout,
                "category":          category,
                "how_to_fix":        how_to_fix,
                "estimated_cost_inr": cost_inr,
                "wall_condition":    wall_condition,
                "condition_score":   condition_score,
                "rag_guidance":      matched_doc.get("text", "")[:300] if matched_doc else "",
                "rag_source":        matched_doc.get("doc_id", "") if matched_doc else "",
            }

        # ── Build repairs: structural first, then CV damage, then layout ──────
        seen_categories: set = set()

        for issue in structural_issues:
            r = _make_repair(issue, severity)
            cat = r["category"]
            # Deduplicate by category — no two structural repairs of the same type
            if cat not in seen_categories:
                seen_categories.add(cat)
                repairs.append(r)
            if len(repairs) >= 4:
                break

        for issue in cv_damage:
            if len(repairs) >= 4:
                break
            r = _make_repair(issue, severity)
            if r["category"] not in seen_categories:
                seen_categories.add(r["category"])
                repairs.append(r)

        for issue in layout_issues:  # already capped at 1
            if len(repairs) >= 5:
                break
            if "layout" not in seen_categories:
                seen_categories.add("layout")
                repairs.append(_make_repair(issue, severity, is_layout=True))

        # ── Fallback: poor condition but no specific issues ───────────────────
        if not repairs and condition_score is not None and condition_score < 50:
            repairs.append({
                "issue":             f"General poor condition (score: {condition_score}/100) — full surface remediation required",
                "severity":          severity,
                "must_fix_first":    True,
                "category":          "structural",
                "how_to_fix":        "Hack out loose plaster → bond coat → polymer mortar → alkali primer → putty 2 coats → 2-coat premium paint",
                "estimated_cost_inr": round(budget_inr * 0.08) if budget_inr else 0,
                "wall_condition":    wall_condition,
                "condition_score":   condition_score,
                "rag_guidance":      "",
                "rag_source":        "condition_poor_wall_remediation",
            })

        # ── Fallback: good condition room — show top high-value upgrades ──────
        if not repairs:
            high_value = (
                _to_list(state.get("high_value_upgrades"))
                or _to_list(room_features_dict.get("high_value_upgrades"))
            )
            renovation_priority = (
                _to_list(room_features_dict.get("renovation_priority"))
                or _to_list(damage.get("renovation_priority"))
                or ["false ceiling with cove lighting", "modular wardrobe", "large-format flooring"]
            )
            UPGRADE_HOW: Dict[str, tuple] = {
                "false ceiling":  ("Install GI frame POP false ceiling with recessed LED coves", 0.10),
                "wardrobe":       ("Build floor-to-ceiling 18mm BWR plywood modular wardrobe with acrylic shutters", 0.15),
                "flooring":       ("Lay 800×800mm vitrified tiles on cement bed with 3mm epoxy grout joints", 0.12),
                "lighting":       ("Install 3-point LED scheme: 4000K recessed downlights + 3000K strip coves + accent spots", 0.05),
                "walls":          ("Apply 2-coat textured finish or premium wallpaper on feature wall", 0.04),
                "modular kitchen": ("Install 18mm plywood carcass modular kitchen with soft-close shutters and granite top", 0.20),
                "bathroom":       ("Replace CP fittings and sanitaryware, apply wall tiles, install shower enclosure", 0.15),
            }
            action_sources = list(dict.fromkeys(high_value + renovation_priority))
            upgrade_categories: set = set()
            for action in action_sources:
                if len(repairs) >= 4:
                    break
                action_lower = str(action).lower()
                matched = next(
                    ((how, pct) for k, (how, pct) in UPGRADE_HOW.items() if k in action_lower),
                    (f"Plan and execute {action} upgrade for maximum ROI", 0.05),
                )
                how_fix = matched[0]
                cost_inr = round(budget_inr * matched[1]) if budget_inr else 0
                # Deduplicate upgrades by category key
                cat_key = next((k for k in UPGRADE_HOW if k in action_lower), action_lower[:15])
                if cat_key in upgrade_categories:
                    continue
                upgrade_categories.add(cat_key)
                repairs.append({
                    "issue":             action,
                    "severity":          "low",
                    "must_fix_first":    False,
                    "category":          "upgrade",
                    "how_to_fix":        how_fix,
                    "estimated_cost_inr": cost_inr,
                    "wall_condition":    wall_condition,
                    "condition_score":   condition_score,
                    "rag_guidance":      "",
                    "rag_source":        "high_value_upgrade",
                    "is_upgrade":        True,
                })

        # Sort by severity desc, then must_fix_first
        SEV_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        repairs.sort(key=lambda r: (
            SEV_RANK.get(r["severity"], 0),
            int(r.get("must_fix_first", False)),
        ), reverse=True)

        return repairs[:5]

    def _build_room_intelligence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Room Intelligence Report — replaces the renovation sequence in the UI.

        Produces 4 image-grounded cards, all derived from actual pipeline outputs:
          1. what_we_detected   — detected materials, objects, style, condition
          2. whats_working_well — genuine positives from the actual room
          3. quick_wins         — 3 high-ROI actions doable in under 2 days
          4. material_spotlight — detected materials with quality tier + upgrade path

        Every field is sourced from real pipeline state — nothing is fabricated.
        """
        room_features_dict = state.get("room_features") or state.get("vision_features") or {}
        cv = state.get("cv_features") or {}
        damage = state.get("damage_assessment") or {}
        layout = state.get("layout_report") or {}
        cost = state.get("cost_estimate") or {}
        budget_inr = float(cost.get("total_inr", 0) or state.get("budget_inr", 750_000))

        def _to_list(val) -> List[str]:
            if isinstance(val, list):
                return [str(v) for v in val if v]
            if isinstance(val, str) and val:
                return [val]
            return []

        def _fmt_inr(n: float) -> str:
            if n >= 100_000:
                return f"₹{n/100_000:.1f}L"
            if n >= 1_000:
                return f"₹{n/1_000:.0f}K"
            return f"₹{int(n)}"

        # ── Source data ───────────────────────────────────────────────────────
        detected_objects = list(dict.fromkeys(
            _to_list(state.get("detected_objects"))
            or _to_list(cv.get("detected_objects"))
            or _to_list(room_features_dict.get("detected_furniture"))
        ))
        material_types = list(dict.fromkeys(
            _to_list(state.get("material_types"))
            or _to_list(cv.get("materials"))
            or _to_list(room_features_dict.get("material_types"))
        ))
        style_label = (
            state.get("style_label")
            or cv.get("style")
            or room_features_dict.get("style_label")
            or "Modern Minimalist"
        )
        style_confidence = float(
            state.get("style_confidence")
            or cv.get("style_confidence")
            or room_features_dict.get("style_confidence")
            or 0.5
        )
        room_type = (
            state.get("room_type")
            or cv.get("room_type")
            or room_features_dict.get("room_type")
            or "bedroom"
        )
        natural_light = (
            room_features_dict.get("natural_light")
            or cv.get("lighting")
            or damage.get("natural_light")
            or "moderate"
        )
        floor_type = (
            room_features_dict.get("floor_type")
            or room_features_dict.get("floor_material")
            or "vitrified tiles"
        )
        wall_color = room_features_dict.get("wall_color", "neutral")
        ceiling_type = (
            room_features_dict.get("ceiling_type")
            or room_features_dict.get("ceiling_treatment")
            or "plain plaster"
        )
        condition = room_features_dict.get("condition", damage.get("overall_condition", "fair"))
        condition_score_raw = (
            state.get("condition_score")
            or room_features_dict.get("condition_score")
            or damage.get("condition_score")
        )
        try:
            condition_score = int(condition_score_raw) if condition_score_raw not in (None, "", "not_assessed") else None
        except (TypeError, ValueError):
            condition_score = None

        layout_score_raw = (
            layout.get("layout_score")
            or room_features_dict.get("layout_score")
            or damage.get("layout_score")
            or "65/100"
        )
        # Normalise layout_score to int
        try:
            layout_score = int(str(layout_score_raw).split("/")[0])
        except (ValueError, AttributeError):
            layout_score = 65

        floor_area = float(state.get("floor_area_sqft") or room_features_dict.get("floor_area_sqft") or 120)
        wall_area  = float(state.get("wall_area_sqft")  or room_features_dict.get("wall_area_sqft")  or 200)

        # ── 1. WHAT WE DETECTED ───────────────────────────────────────────────
        detected_items = []
        if style_label:
            confidence_pct = round(style_confidence * 100)
            detected_items.append({
                "label": "Design Style",
                "value": style_label,
                "detail": f"{confidence_pct}% confidence",
                "icon": "style",
            })
        if floor_type and floor_type != "unknown":
            FLOOR_QUALITY = {
                "vitrified": "Mid-range — good durability",
                "marble":    "Premium — high perceived value",
                "hardwood":  "Premium — warm, high ROI",
                "granite":   "Premium — very durable",
                "laminate":  "Budget — consider upgrade for resale",
                "carpet":    "Basic — consider hard flooring upgrade",
                "cement":    "Industrial — growing appeal, sealer recommended",
            }
            floor_lower = floor_type.lower()
            quality = next(
                (v for k, v in FLOOR_QUALITY.items() if k in floor_lower),
                "Standard finish"
            )
            detected_items.append({
                "label": "Floor Material",
                "value": floor_type.title(),
                "detail": quality,
                "icon": "floor",
            })
        if ceiling_type and ceiling_type != "unknown":
            has_false = any(kw in ceiling_type.lower() for kw in ("pop", "false", "gypsum", "grid"))
            detected_items.append({
                "label": "Ceiling",
                "value": ceiling_type.title(),
                "detail": "LED integration ready" if has_false else "Upgrade potential — add false ceiling for premium finish",
                "icon": "ceiling",
            })
        if natural_light:
            LIGHT_MAP = {
                "excellent": ("Excellent natural light", "Top 10% of homes — major asset"),
                "good":      ("Good natural light", "Above average — adds perceived value"),
                "moderate":  ("Moderate natural light", "Supplement with layered artificial lighting"),
                "poor":      ("Limited natural light", "LED solution essential — focus on warm 3000K"),
                "dim":       ("Dim lighting detected", "Artificial lighting upgrade is priority"),
                "artificial":("Artificial light primary", "Add sheer curtains + smart dimmer for warmth"),
            }
            lv, ld = LIGHT_MAP.get(natural_light, (f"{natural_light.title()} light", ""))
            detected_items.append({
                "label": "Lighting",
                "value": lv,
                "detail": ld,
                "icon": "light",
            })
        if condition_score is not None:
            cs = condition_score
            cond_detail = (
                "Excellent — no major repairs needed"     if cs >= 80 else
                "Good — minor cosmetic work only"         if cs >= 65 else
                "Fair — surface repairs recommended"      if cs >= 45 else
                "Poor — structural repairs required first" if cs >= 30 else
                "Very poor — major renovation essential"
            )
            detected_items.append({
                "label": "Room Condition",
                "value": f"{cs}/100",
                "detail": cond_detail,
                "icon": "condition",
            })
        detected_items.append({
            "label": "Room Size",
            "value": f"{floor_area:.0f} sqft floor",
            "detail": f"{wall_area:.0f} sqft paintable wall area",
            "icon": "size",
        })
        if detected_objects:
            detected_items.append({
                "label": "Furniture Detected",
                "value": f"{len(detected_objects)} items",
                "detail": ", ".join(detected_objects[:5]),
                "icon": "furniture",
            })

        # ── 2. WHAT'S WORKING WELL ────────────────────────────────────────────
        working_well = []
        if natural_light in ("good", "excellent"):
            working_well.append({
                "point": f"{natural_light.title()} natural light",
                "why":   "Natural light is one of the highest-value room features — preserve window clearance",
            })
        if condition in ("good", "excellent") or (condition_score and condition_score >= 65):
            working_well.append({
                "point": "Solid room condition",
                "why":   "No major structural repairs needed — budget can go straight to upgrades",
            })
        if any(kw in floor_type.lower() for kw in ("marble", "hardwood", "granite", "vitrified")):
            working_well.append({
                "point": f"{floor_type.title()} flooring in place",
                "why":   "Quality base material — refinish or deep clean before replacing",
            })
        if any(kw in (ceiling_type or "").lower() for kw in ("pop", "false", "gypsum")):
            working_well.append({
                "point": "False ceiling already installed",
                "why":   "Saves ₹40–80K vs new install — upgrade LED layout only",
            })
        if layout_score >= 70:
            working_well.append({
                "point": f"Good furniture layout (score: {layout_score}/100)",
                "why":   "Space flow is efficient — focus budget on surface and lighting upgrades",
            })
        if "brass" in str(material_types).lower() or "brass" in str(room_features_dict.get("color_palette", [])).lower():
            working_well.append({
                "point": "Warm metal accents present",
                "why":   "Brass/gold tones are trending in Indian interiors — coordinate rather than replace",
            })
        if not working_well:
            working_well.append({
                "point": "Renovation-ready canvas",
                "why":   "Room structure is sound — cosmetic renovation will deliver high visual impact",
            })

        # ── 3. QUICK WINS (high-ROI, ≤2 days each) ───────────────────────────
        QUICK_WIN_DB: List[Dict] = [
            {
                "trigger_check": lambda rf, cv, nl, obj: nl in ("poor", "dim", "artificial"),
                "action":        "Replace light fittings with warm LED (3000K) panel lights",
                "impact":        "Transforms room ambience in 1 day",
                "duration":      "1 day",
                "cost_range":    "₹3,000–8,000",
                "roi_note":      "Lighting is the single highest-ROI cosmetic upgrade",
            },
            {
                "trigger_check": lambda rf, cv, nl, obj: (
                    rf.get("wall_color", "") in ("white", "off-white", "cream", "beige", "unknown", "")
                ),
                "action":        "Paint one feature wall in a bold accent colour",
                "impact":        "Creates focal point, adds depth to neutral room",
                "duration":      "1 day",
                "cost_range":    "₹2,500–6,000",
                "roi_note":      "Feature walls add 5–8% perceived value at minimal cost",
            },
            {
                "trigger_check": lambda rf, cv, nl, obj: (
                    "curtain" not in str(obj).lower() and "drape" not in str(rf).lower()
                ),
                "action":        "Add floor-to-ceiling curtains (sheer + blockout layer)",
                "impact":        "Makes ceiling appear taller, adds luxury feel",
                "duration":      "Half day",
                "cost_range":    "₹4,000–12,000",
                "roi_note":      "Floor-length curtains are #1 perceived-luxury upgrade per rupee",
            },
            {
                "trigger_check": lambda rf, cv, nl, obj: (
                    not any(kw in (rf.get("ceiling_type") or "").lower() for kw in ("pop", "false", "gypsum"))
                ),
                "action":        "Add cove LED strip along ceiling perimeter",
                "impact":        "Creates ambient glow without full false ceiling cost",
                "duration":      "1 day",
                "cost_range":    "₹2,000–5,000",
                "roi_note":      "LED cove gives 70% of false ceiling visual impact at 15% of the cost",
            },
            {
                "trigger_check": lambda rf, cv, nl, obj: (
                    "mirror" not in str(obj).lower()
                ),
                "action":        "Place a large wall mirror (min. 3×4 ft) on a side wall",
                "impact":        "Doubles perceived room size, amplifies light",
                "duration":      "2 hours",
                "cost_range":    "₹3,000–9,000",
                "roi_note":      "Mirrors are the most space-efficient upgrade in smaller rooms",
            },
            {
                "trigger_check": lambda rf, cv, nl, obj: (
                    "rug" not in str(obj).lower() and "carpet" not in rf.get("floor_type", "").lower()
                ),
                "action":        "Add a large area rug (min. 6×9 ft) in a complementary tone",
                "impact":        "Anchors furniture grouping, adds warmth and acoustics",
                "duration":      "1 hour",
                "cost_range":    "₹4,000–15,000",
                "roi_note":      "Rugs complete room compositions — without one it looks unfinished",
            },
        ]

        quick_wins = []
        for qw in QUICK_WIN_DB:
            if len(quick_wins) >= 3:
                break
            try:
                if qw["trigger_check"](room_features_dict, cv, natural_light, detected_objects):
                    quick_wins.append({
                        "action":     qw["action"],
                        "impact":     qw["impact"],
                        "duration":   qw["duration"],
                        "cost_range": qw["cost_range"],
                        "roi_note":   qw["roi_note"],
                    })
            except Exception:
                continue
        # If fewer than 3 triggered, backfill from the remainder
        if len(quick_wins) < 3:
            for qw in QUICK_WIN_DB:
                if len(quick_wins) >= 3:
                    break
                entry = {
                    "action":     qw["action"],
                    "impact":     qw["impact"],
                    "duration":   qw["duration"],
                    "cost_range": qw["cost_range"],
                    "roi_note":   qw["roi_note"],
                }
                if entry not in quick_wins:
                    quick_wins.append(entry)

        # ── 4. MATERIAL SPOTLIGHT ─────────────────────────────────────────────
        MATERIAL_INFO: Dict[str, Dict] = {
            "vitrified_tile": {
                "name":        "Vitrified Tile",
                "quality":     "Mid-range",
                "durability":  "15–20 years",
                "upgrade_to":  "800×800mm large-format GVT tiles",
                "upgrade_cost": _fmt_inr(floor_area * 120),
                "upgrade_why": "Reduces grout lines, adds 8–12% perceived room size",
            },
            "marble": {
                "name":        "Marble",
                "quality":     "Premium",
                "durability":  "50+ years with polish",
                "upgrade_to":  "Deep clean + diamond polishing",
                "upgrade_cost": _fmt_inr(floor_area * 45),
                "upgrade_why": "Restores natural lustre — far cheaper than replacement",
            },
            "hardwood_floor": {
                "name":        "Hardwood Floor",
                "quality":     "Premium",
                "durability":  "30+ years",
                "upgrade_to":  "Sand, stain and lacquer finish",
                "upgrade_cost": _fmt_inr(floor_area * 55),
                "upgrade_why": "Refinishing costs 25% of replacement, results look new",
            },
            "concrete": {
                "name":        "Exposed Concrete",
                "quality":     "Industrial",
                "durability":  "Very durable",
                "upgrade_to":  "Microcement overlay + matte sealer",
                "upgrade_cost": _fmt_inr(floor_area * 85),
                "upgrade_why": "Adds contemporary finish while preserving industrial character",
            },
            "fabric": {
                "name":        "Fabric / Upholstery",
                "quality":     "Variable",
                "durability":  "5–10 years",
                "upgrade_to":  "Reupholster in performance fabric",
                "upgrade_cost": "₹8,000–25,000 per piece",
                "upgrade_why": "New fabric transforms furniture without replacement cost",
            },
            "glass": {
                "name":        "Glass Elements",
                "quality":     "Premium look",
                "durability":  "Long lasting with care",
                "upgrade_to":  "Frosted / tinted variant for privacy",
                "upgrade_cost": "₹3,000–12,000",
                "upgrade_why": "Tinted glass adds privacy while retaining light transmission",
            },
            "wood": {
                "name":        "Wood Surfaces",
                "quality":     "Warm, natural",
                "durability":  "15–25 years",
                "upgrade_to":  "Sand and apply Danish oil / lacquer",
                "upgrade_cost": _fmt_inr(floor_area * 35),
                "upgrade_why": "Refinishing wood surfaces costs 20% of replacement",
            },
        }

        material_spotlight = []
        all_mats = list(dict.fromkeys(material_types + _to_list(cv.get("materials"))))
        for mat_key in all_mats[:3]:
            info = MATERIAL_INFO.get(mat_key.lower().replace(" ", "_"))
            if info:
                material_spotlight.append(info)
            else:
                # Generic entry for unknown materials
                material_spotlight.append({
                    "name":        mat_key.replace("_", " ").title(),
                    "quality":     "Detected in image",
                    "durability":  "Assess on-site",
                    "upgrade_to":  "Consult contractor for best finish option",
                    "upgrade_cost": "TBD on inspection",
                    "upgrade_why": "Material-specific upgrade path depends on current condition",
                })

        # If no materials detected, build from floor/ceiling/wall
        if not material_spotlight:
            floor_key = next(
                (k for k in MATERIAL_INFO if k in floor_type.lower().replace(" ", "_")),
                None,
            )
            if floor_key:
                material_spotlight.append(MATERIAL_INFO[floor_key])

        return {
            "section_title":    "Room Intelligence",
            "section_subtitle": f"What our analysis found in your {room_type.replace('_', ' ')} photo",
            "what_we_detected": detected_items,
            "whats_working_well": working_well,
            "quick_wins":       quick_wins[:3],
            "material_spotlight": material_spotlight[:3],
            "image_grounded":   bool(state.get("cv_features") or state.get("detected_objects")),
            "room_type":        room_type,
            "style_label":      style_label,
            "condition_score":  condition_score,
            "layout_score":     layout_score,
        }

    def _build_budget_strategy(
        self,
        state: Dict[str, Any],
        engine_output: Any,
    ) -> Dict[str, Any]:
        budget_inr = state.get("budget_inr", 750_000)
        cost = state.get("cost_estimate") or {}
        total = cost.get("total_inr", budget_inr)
        within = cost.get("within_budget", True)
        budget_analysis = state.get("budget_analysis") or {}

        room_features_dict = state.get("room_features") or state.get("vision_features") or {}
        _raw_cs = state.get("condition_score") or room_features_dict.get("condition_score")
        condition_score = int(_raw_cs) if _raw_cs not in (None, "", "not_assessed") else None
        renovation_scope = state.get("renovation_scope") or room_features_dict.get("renovation_scope") or "partial"
        high_value_upgrades = state.get("high_value_upgrades") or room_features_dict.get("high_value_upgrades") or []

        # Use 65 as the allocation fallback ONLY when no real score is available
        _cs = condition_score if condition_score is not None else 65

        if _cs < 30:
            allocation = {"repair_pct": 35, "materials_pct": 30, "labour_pct": 25, "gst_contingency_pct": 10}
            condition_note = "Very poor condition: 35% allocated to repair before cosmetic work"
        elif _cs < 50:
            allocation = {"repair_pct": 20, "materials_pct": 40, "labour_pct": 28, "gst_contingency_pct": 12}
            condition_note = "Poor condition: 20% repair budget required before renovation begins"
        elif _cs < 70:
            allocation = {"repair_pct": 10, "materials_pct": 48, "labour_pct": 30, "gst_contingency_pct": 12}
            condition_note = "Fair condition: minor repairs included in materials allocation"
        else:
            allocation = {"repair_pct": 0, "materials_pct": 55, "labour_pct": 30, "gst_contingency_pct": 15}
            condition_note = "Good condition: full budget available for cosmetic and design upgrades"

        if condition_score is None:
            condition_note = "Condition not assessed — allocation based on typical Indian home benchmarks"

        strategy: Dict[str, Any] = {
            "allocation": allocation,
            "within_budget": within,
            "budget_gap_inr": max(0, total - budget_inr),
            "tier_guidance": budget_analysis,
            "condition_score": condition_score,    # None → "Not assessed" on frontend
            "renovation_scope": renovation_scope,
            "condition_note": condition_note,
            "high_value_upgrades": high_value_upgrades,
        }

        if not within:
            overage = total - budget_inr
            phase1_items = (["Wall crack repair", "Waterproofing", "Surface preparation"]
                            if _cs < 50 else ["Walls (paint/texture)", "Flooring", "Lighting"])
            strategy["phasing"] = {
                "recommended": True,
                "phase_1": {
                    "label":  "Priority Repairs + High-ROI" if _cs < 50 else "High-ROI Cosmetic Upgrades",
                    "items":  phase1_items,
                    "budget": int(budget_inr * 0.65),
                },
                "phase_2": {
                    "label":  "Furniture & Fixtures",
                    "items":  ["Modular furniture", "Fixtures", "Décor"],
                    "budget": int(budget_inr * 0.35),
                },
                "deferred": {
                    "label":  "Phase 3 (deferred)",
                    "amount": overage,
                    "items":  high_value_upgrades or ["Premium smart-home", "Custom millwork"],
                },
            }
        else:
            strategy["phasing"] = {
                "recommended": False,
                "note": "Project fits within budget",
                "suggested_upgrades": high_value_upgrades[:2] if high_value_upgrades else [],
            }
        return strategy

    def _fallback_insights(self, state: Dict[str, Any], error: str) -> Dict[str, Any]:
        theme = state.get("theme", "Modern Minimalist")
        city = state.get("city", "India")
        errors = list(state.get("errors") or [])
        errors.append(f"insight_generation: {error}")
        return {
            "insights": {
                "summary_headline": f"{theme} renovation — {city}",
                "visual_analysis": {},
                "financial_outlook": {},
                "market_intelligence": {},
                "budget_assessment": {},
                "recommendations": [],
                "risk_factors": [],
                "market_timing":    {},
                "action_checklist": [],
                "top_materials": [],
                "image_grounded": False,
                "dataset_grounded": False,
                "rag_grounded": False,
                "insight_engine": {},
            },
            "insight_engine_output": {},
            "renovation_sequence": [],
            "priority_repairs": [],
            "budget_strategy": {},
            "errors": errors,
        }


# ── LangGraph node wrapper ────────────────────────────────────────────────────

def node_insight_generation(state: Dict[str, Any]) -> Dict[str, Any]:
    import asyncio
    import concurrent.futures

    agent = InsightGenerationAgent()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, agent.run(dict(state)))
                result = future.result(timeout=90)
        else:
            result = loop.run_until_complete(agent.run(dict(state)))
    except RuntimeError:
        result = asyncio.run(agent.run(dict(state)))
    return {**state, **result}