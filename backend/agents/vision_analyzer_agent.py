"""
ARKEN — VisionAnalyzerAgent v2.0
==================================
Agent 2 in the LangGraph multi-agent pipeline.

v2.0 Changes:
  - Integrates new CVFeatureExtractor (YOLOv8 + CLIP + EfficientNet)
  - CV pipeline runs FIRST and enriches Gemini output
  - Structured output matches downstream agent contracts
  - GPU-aware (uses CVModelRegistry device selection)
  - Redis-cached inference results

Execution order:
  1. CVFeatureExtractor (YOLOv8 + CLIP + EfficientNet) — fast local inference
  2. VisualAssessorAgent (Gemini Vision) — semantic depth
  3. Fusion: CV features enrich/override Gemini where CV confidence is higher
  4. Fallback to pipeline.py if both fail

Input state keys:  original_image_b64, original_image_mime, original_image_bytes,
                   theme, city, budget_tier, room_type, project_id
Output state keys: vision_features, image_features, room_features, room_dimensions,
                   detected_changes, visual_style, wall_area_sqft, floor_area_sqft,
                   layout_report, style_label, style_confidence, explainable_recommendations,
                   material_quantities, detected_objects, material_types,
                   cv_features (new: raw structured CV output)
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class VisionAnalyzerAgent:
    """
    Fused CV + Gemini room analyser.
    Runs local CV models first, then enriches with Gemini Vision.
    """

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        name = "vision_analyzer_agent"
        logger.info(f"[{state.get('project_id', '')}] VisionAnalyzerAgent v2.0 starting")

        updates: Dict[str, Any] = {}
        image_bytes = self._resolve_image(state)

        # ── Stage A: CV Pipeline (local, cached, fast) ────────────────────────
        cv_features_dict: Dict[str, Any] = {}
        if image_bytes:
            cv_features_dict = await self._run_cv_pipeline(
                image_bytes,
                hint_room_type=state.get("room_type"),
                state=state,
            )
            updates["cv_features"] = cv_features_dict

        # ── Stage B: Gemini Vision (semantic depth) ───────────────────────────
        try:
            gemini_updates = await self._run_visual_assessor(state, image_bytes)
            updates.update(gemini_updates)
        except Exception as e:
            logger.warning(f"[vision_analyzer] Gemini failed ({e}), using pipeline fallback")
            try:
                fallback = self._run_pipeline_fallback(state)
                updates.update(fallback)
            except Exception as e2:
                logger.error(f"[vision_analyzer] Pipeline fallback also failed: {e2}")
                updates.update(self._static_fallback(state))
                updates["errors"] = (state.get("errors") or []) + [
                    f"vision_analyzer (all paths failed): {e2}"
                ]

        # ── Stage C: Fuse CV into Gemini output ───────────────────────────────
        if cv_features_dict:
            updates = self._fuse_cv_into_state(updates, cv_features_dict)

        # ── Timing + bookkeeping ──────────────────────────────────────────────
        elapsed = round(time.perf_counter() - t0, 3)
        timings = dict(state.get("agent_timings") or {})
        timings[name] = elapsed
        updates["agent_timings"] = timings

        completed = list(state.get("completed_agents") or [])
        if name not in completed:
            completed.append(name)
        updates["completed_agents"] = completed

        logger.info(
            f"[vision_analyzer] done in {elapsed}s — "
            f"room={updates.get('cv_features', {}).get('room_type', '?')} "
            f"style={updates.get('style_label', '?')} "
            f"objects={len(updates.get('detected_objects', []))} "
            f"wall={updates.get('wall_area_sqft', 0):.0f}sqft"
        )
        return updates

    # ─────────────────────────────────────────────────────────────────────────
    # CV Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_cv_pipeline(
        self,
        image_bytes: bytes,
        hint_room_type: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        v3.0: Run YOLOv8 + CLIP + EfficientNet (existing) PLUS:
          - DepthEstimator   → measured room area
          - DamageDetector   → structural damage + scope recommendation
          - StyleClassifier  → reliable CLIP-based style

        All new modules degrade gracefully — their failures never break
        the existing YOLOv8/CLIP pipeline result.
        Returns empty dict only if ALL paths fail.
        """
        import asyncio

        cv_features_dict: Dict[str, Any] = {}

        # ── A. Existing YOLOv8 + CLIP + EfficientNet (UNCHANGED) ─────────────
        try:
            from ml.cv_feature_extractor import get_extractor
            extractor   = get_extractor()
            cv_features = await extractor.extract(
                image_bytes,
                use_cache=True,
                hint_room_type=hint_room_type,
            )
            if cv_features.cv_available:
                cv_features_dict = cv_features.to_vision_agent_format()
        except Exception as e:
            logger.warning(f"[vision_analyzer] YOLOv8/CLIP pipeline failed (non-critical): {e}")

        # Room type from YOLO result or hint
        room_type = cv_features_dict.get("room_type") or hint_room_type or "bedroom"

        # ── B. DepthEstimator — measured room area ────────────────────────────
        try:
            from ml.depth_estimator import DepthEstimator
            depth_est   = DepthEstimator()
            depth_result = await asyncio.to_thread(
                depth_est.estimate_room_area, image_bytes, room_type
            )
            cv_features_dict["measured_floor_area_sqft"]       = depth_result["floor_area_sqft"]
            cv_features_dict["measured_wall_area_sqft"]        = depth_result["wall_area_sqft"]
            cv_features_dict["measured_ceiling_height_ft"]     = depth_result["ceiling_height_ft"]
            cv_features_dict["area_measurement_method"]        = depth_result["method"]
            cv_features_dict["area_measurement_confidence"]    = depth_result["confidence"]
            cv_features_dict["depth_map_available"]            = depth_result["depth_map_available"]
            logger.info(
                f"[vision_analyzer] DepthEstimator: "
                f"floor={depth_result['floor_area_sqft']:.0f}sqft  "
                f"wall={depth_result['wall_area_sqft']:.0f}sqft  "
                f"method={depth_result['method']}  "
                f"conf={depth_result['confidence']:.2f}"
            )
        except Exception as e:
            logger.warning(f"[vision_analyzer] DepthEstimator failed (non-critical): {e}")

        # ── C. DamageDetector — structural damage + scope ─────────────────────
        try:
            from ml.damage_detector import DamageDetector
            dmg_detector = DamageDetector()
            damage_result = await asyncio.to_thread(
                dmg_detector.detect, image_bytes
            )
            cv_features_dict["detected_damage"]              = damage_result["detected_issues"]
            cv_features_dict["damage_severity"]              = damage_result["severity"]
            cv_features_dict["damage_scores"]                = damage_result["damage_scores"]
            cv_features_dict["renovation_scope_from_damage"] = damage_result["renovation_scope_recommendation"]
            cv_features_dict["requires_waterproofing"]       = damage_result["requires_waterproofing"]
            cv_features_dict["requires_structural_repair"]   = damage_result["requires_structural_repair"]
            cv_features_dict["damage_model_used"]            = damage_result["model_used"]
            logger.info(
                f"[vision_analyzer] DamageDetector: "
                f"issues={damage_result['detected_issues']}  "
                f"severity={damage_result['severity']}  "
                f"scope={damage_result['renovation_scope_recommendation']}"
            )
        except Exception as e:
            logger.warning(f"[vision_analyzer] DamageDetector failed (non-critical): {e}")

        # ── D. StyleClassifier — CLIP-based style (replaces keyword matching) ─
        try:
            from ml.style_classifier import StyleClassifier
            gemini_hint  = (state or {}).get("theme", "") if state else ""
            style_clf    = StyleClassifier()
            style_result = await asyncio.to_thread(
                style_clf.classify, image_bytes, gemini_hint
            )
            cv_features_dict["clip_style_label"]      = style_result["style_label"]
            cv_features_dict["clip_style_confidence"] = style_result["style_confidence"]
            cv_features_dict["clip_style_top3"]        = style_result["top_3_styles"]
            cv_features_dict["style_model_used"]       = style_result["model_used"]
            cv_features_dict["style_gemini_agreement"] = style_result["gemini_agreement"]
            logger.info(
                f"[vision_analyzer] StyleClassifier: "
                f"style={style_result['style_label']}  "
                f"conf={style_result['style_confidence']:.2f}  "
                f"model={style_result['model_used']}  "
                f"gemini_agree={style_result['gemini_agreement']}"
            )
        except Exception as e:
            logger.warning(f"[vision_analyzer] StyleClassifier failed (non-critical): {e}")

        return cv_features_dict

    # ─────────────────────────────────────────────────────────────────────────
    # Gemini Visual Assessor (unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_visual_assessor(
        self,
        state: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        from agents.visual_assessor import VisualAssessorAgent

        agent = VisualAssessorAgent()

        if not image_bytes:
            image_bytes = self._resolve_image(state)
        if not image_bytes:
            raise ValueError("No image data in state")

        result = await agent.analyze(
            image_bytes,
            state.get("project_id", "unknown"),
            room_type=state.get("room_type", "bedroom"),
            theme=state.get("theme", "Modern Minimalist"),
            budget_tier=state.get("budget_tier", "mid"),
            city=state.get("city", "Hyderabad"),
        )

        features = result.get("features", {})
        wall_sqft = float(features.get("estimated_wall_area_sqft", 200.0))
        floor_sqft = float(features.get("estimated_floor_area_sqft", 120.0))

        vision_features_dict = {
            "wall_treatment": features.get("wall_treatment", features.get("wall_color", "")),
            "floor_material": features.get("floor_material", features.get("floor_type", "")),
            "ceiling_treatment": features.get("ceiling_treatment", features.get("ceiling_type", "")),
            "furniture_items": features.get("detected_furniture", []),
            "lighting_type": str(features.get("lighting_sources", ["ceiling"])),
            "colour_palette": features.get("colour_palette", features.get("color_palette", [])),
            "detected_style": result["style"]["label"],
            "quality_tier": features.get("quality_tier", state.get("budget_tier", "mid")),
            "specific_changes": features.get("specific_changes", []),
            "estimated_wall_area_sqft": wall_sqft,
            "estimated_floor_area_sqft": floor_sqft,
            "room_condition": features.get("condition", "fair"),
            "layout_score": str(result.get("layout_report", {}).get("layout_score", "65/100")),
            "walkable_space": str(result.get("layout_report", {}).get("walkable_space", "45%")),
            "natural_light_quality": features.get("natural_light", "moderate"),
            "extraction_source": "gemini",
        }

        floor_str = vision_features_dict["floor_material"].lower()
        material_types = []
        if any(x in floor_str for x in ["tile", "vitrified", "ceramic"]):
            material_types.append("vitrified_tile")
        if any(x in floor_str for x in ["wood", "hardwood", "parquet"]):
            material_types.append("hardwood_floor")
        if "marble" in floor_str:
            material_types.append("marble")
        material_types = material_types or ["vitrified_tile"]

        return {
            "vision_features": vision_features_dict,
            "image_features": result.get("image_features", vision_features_dict),
            "room_features": features,
            "layout_report": result.get("layout_report", {}),
            "style_label": result["style"]["label"],
            "style_confidence": result["style"]["confidence"],
            "explainable_recommendations": result.get("recommendations", []),
            "material_quantities": result.get("material_quantities", {}),
            "detected_objects": list(result.get("spatial_map", {}).get("detected_objects", {}).keys()),
            "material_types": material_types,
            "wall_area_sqft": wall_sqft,
            "floor_area_sqft": floor_sqft,
            "room_dimensions": {
                "wall_area_sqft": wall_sqft,
                "floor_area_sqft": floor_sqft,
                "estimated_length_ft": features.get("estimated_length_ft", 14.0),
                "estimated_width_ft": features.get("estimated_width_ft", 12.0),
                "estimated_height_ft": features.get("estimated_height_ft", 9.0),
            },
            "detected_changes": features.get("specific_changes", []),
            "visual_style": [result["style"]["label"]],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CV ↔ Gemini fusion
    # ─────────────────────────────────────────────────────────────────────────

    def _fuse_cv_into_state(
        self,
        state_updates: Dict[str, Any],
        cv: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        v3.0: Fuse CV pipeline output into the state update dict.

        Fusion rules (v2.0 rules preserved, v3.0 additions marked):
          - detected_objects : merge CV YOLO + Gemini lists (CV takes precedence)
          - style_label      : prefer CV if confidence > 0.65, else keep Gemini
                               [v3.0] also honour clip_style_label if confidence > 0.40
          - room_type        : prefer CV if confidence > 0.70, else keep Gemini
          - materials        : merge both, deduplicated
          - lighting         : add CV lighting to vision_features
          [v3.0] floor/wall area : replace Gemini guess if depth confidence > 0.50
          [v3.0] renovation scope: add damage-based scope to state
        """
        cv_room         = cv.get("room_type", "")
        cv_room_conf    = cv.get("room_type_confidence", 0.0)
        cv_style        = cv.get("style", "")
        cv_style_conf   = cv.get("style_confidence", 0.0)
        cv_objects      = cv.get("detected_objects", [])
        cv_materials    = cv.get("materials", [])
        cv_lighting     = cv.get("lighting", "")

        # ── Merge detected_objects (UNCHANGED from v2.0) ──────────────────────
        gemini_objects = state_updates.get("detected_objects", [])
        if cv_objects:
            merged = list(dict.fromkeys(cv_objects + gemini_objects))
            state_updates["detected_objects"] = merged
            vf = dict(state_updates.get("vision_features", {}))
            vf["furniture_items"] = merged
            state_updates["vision_features"] = vf

        # ── Style override — blended multi-signal confidence ─────────────────
        # Without fine-tuned weights, zero-shot CLIP is unreliable below 0.55.
        # We blend CLIP + Gemini confidences and apply an agreement bonus when
        # both sources independently agree on the same style label.
        clip_style      = cv.get("clip_style_label", "")
        clip_style_conf = cv.get("clip_style_confidence", 0.0)
        gemini_style    = state_updates.get("style_label", "")
        gemini_conf     = state_updates.get("style_confidence", 0.0)

        # Agreement bonus: if CLIP and Gemini agree, boost combined confidence
        styles_agree = bool(
            clip_style and gemini_style
            and clip_style.lower() == gemini_style.lower()
        )
        if styles_agree:
            blended_conf = round(min(0.95, clip_style_conf * 0.6 + gemini_conf * 0.4 + 0.10), 3)
        else:
            blended_conf = clip_style_conf

        # Thresholds: 0.55 when sources disagree, 0.42 when they agree
        CLIP_MIN_CONF   = 0.55   # raised from 0.40 — prevents false overrides
        CLIP_AGREE_CONF = 0.42   # lower bar when both sources independently agree
        effective_min   = CLIP_AGREE_CONF if styles_agree else CLIP_MIN_CONF

        if clip_style and blended_conf >= effective_min:
            state_updates["style_label"]      = clip_style
            state_updates["style_confidence"] = blended_conf
            vf = dict(state_updates.get("vision_features", {}))
            vf["detected_style"] = clip_style
            state_updates["vision_features"] = vf
            logger.info(
                f"[vision_analyzer/fuse] Style overridden by CLIP: "
                f"{clip_style} (clip={clip_style_conf:.2f} "
                f"blended={blended_conf:.2f} agree={styles_agree})"
            )
        elif cv_style and cv_style_conf >= 0.65:
            # Existing YOLO/CLIP style from v2.0 extractor (unchanged threshold)
            state_updates["style_label"]      = cv_style
            state_updates["style_confidence"] = cv_style_conf
            vf = dict(state_updates.get("vision_features", {}))
            vf["detected_style"] = cv_style
            state_updates["vision_features"] = vf

        # ── Room type override (UNCHANGED from v2.0) ─────────────────────────
        if cv_room and cv_room_conf >= 0.70:
            vf = dict(state_updates.get("vision_features", {}))
            vf["room_type"] = cv_room
            state_updates["vision_features"] = vf
            room_feat = dict(state_updates.get("room_features", {}))
            room_feat["room_type"] = cv_room
            state_updates["room_features"] = room_feat

        # ── Merge materials (UNCHANGED from v2.0) ────────────────────────────
        existing_types = list(state_updates.get("material_types", []))
        merged_mats    = list(dict.fromkeys(existing_types + cv_materials))
        state_updates["material_types"] = merged_mats

        # ── Lighting (UNCHANGED from v2.0) ───────────────────────────────────
        if cv_lighting:
            vf = dict(state_updates.get("vision_features", {}))
            vf["lighting_condition"] = cv_lighting
            state_updates["vision_features"] = vf

        # ── [v3.0] Replace Gemini's guessed area with DepthEstimator result ──
        area_conf = cv.get("area_measurement_confidence", 0.0)
        floor_m   = cv.get("measured_floor_area_sqft")
        wall_m    = cv.get("measured_wall_area_sqft")

        if floor_m is not None and area_conf > 0.50:
            state_updates["floor_area_sqft"] = floor_m
            state_updates["wall_area_sqft"]  = wall_m or state_updates.get("wall_area_sqft", 200.0)

            # Also update room_dimensions and vision_features
            rd = dict(state_updates.get("room_dimensions", {}))
            rd["floor_area_sqft"] = floor_m
            if wall_m is not None:
                rd["wall_area_sqft"] = wall_m
            if cv.get("measured_ceiling_height_ft"):
                rd["estimated_height_ft"] = cv["measured_ceiling_height_ft"]
            state_updates["room_dimensions"] = rd

            vf = dict(state_updates.get("vision_features", {}))
            vf["estimated_floor_area_sqft"] = floor_m
            if wall_m is not None:
                vf["estimated_wall_area_sqft"] = wall_m
            vf["area_measurement_method"]   = cv.get("area_measurement_method", "depth_model")
            state_updates["vision_features"] = vf

            logger.info(
                f"[vision_analyzer/fuse] Depth-estimated area applied: "
                f"floor={floor_m:.0f}sqft  wall={wall_m:.0f}sqft  "
                f"conf={area_conf:.2f}  method={cv.get('area_measurement_method')}"
            )

        # ── [v3.0] Damage-based renovation scope ─────────────────────────────
        scope_from_damage = cv.get("renovation_scope_from_damage")
        if scope_from_damage:
            state_updates["detected_renovation_scope"] = scope_from_damage

            # Surface damage issues → append to detected_changes
            detected_issues = cv.get("detected_damage", [])
            if detected_issues:
                existing_changes = list(state_updates.get("detected_changes", []))
                damage_labels = [
                    f"Structural issue detected: {issue.replace('_', ' ')}"
                    for issue in detected_issues
                ]
                state_updates["detected_changes"] = list(
                    dict.fromkeys(existing_changes + damage_labels)
                )

            # Waterproofing flag into room_features
            if cv.get("requires_waterproofing"):
                rf = dict(state_updates.get("room_features", {}))
                rf["requires_waterproofing"] = True
                state_updates["room_features"] = rf

            logger.info(
                f"[vision_analyzer/fuse] Damage scope applied: {scope_from_damage}  "
                f"issues={cv.get('detected_damage', [])}  "
                f"waterproof={cv.get('requires_waterproofing', False)}"
            )

        return state_updates

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline fallback (unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_pipeline_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.orchestrator.langgraph_orchestrator import node_vision_analysis

        compat = {
            k: v for k, v in state.items()
            if not isinstance(v, (bytes, bytearray))
        }

        result = node_vision_analysis(compat)
        features = result.get("image_features", {})
        dims = result.get("room_dimensions", {})
        wall_sqft = float(dims.get("wall_area_sqft", 200.0))
        floor_sqft = float(dims.get("floor_area_sqft", 120.0))
        theme = state.get("theme", "Modern Minimalist")

        vision_features_dict = {
            "wall_treatment": features.get("wall_treatment", ""),
            "floor_material": features.get("floor_material", ""),
            "ceiling_treatment": features.get("ceiling_treatment", ""),
            "furniture_items": features.get("furniture_items", []),
            "lighting_type": features.get("lighting_type", ""),
            "colour_palette": features.get("colour_palette", []),
            "detected_style": features.get("detected_style", theme),
            "quality_tier": features.get("quality_tier", "mid"),
            "specific_changes": features.get("specific_changes", []),
            "estimated_wall_area_sqft": wall_sqft,
            "estimated_floor_area_sqft": floor_sqft,
            "room_condition": features.get("room_condition", "fair"),
            "layout_score": "65/100",
            "walkable_space": "45%",
            "natural_light_quality": "moderate",
            "extraction_source": "pipeline_gemini",
        }

        return {
            "vision_features": vision_features_dict,
            "image_features": features,
            "room_features": features,
            "layout_report": {"layout_score": "65/100", "walkable_space": "45%", "issues_detected": []},
            "style_label": features.get("detected_style", theme),
            "style_confidence": 0.6,
            "explainable_recommendations": [],
            "material_quantities": {
                "paint_liters": wall_sqft * 0.074,
                "floor_tiles_sqft": floor_sqft * 1.1,
                "plywood_sqft": floor_sqft * 0.3,
            },
            "detected_objects": [],
            "material_types": ["vitrified_tile"],
            "wall_area_sqft": wall_sqft,
            "floor_area_sqft": floor_sqft,
            "room_dimensions": dims,
            "detected_changes": result.get("detected_changes", []),
            "visual_style": result.get("visual_style", [theme]),
        }

    def _static_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        theme = state.get("theme", "Modern Minimalist")
        tier = state.get("budget_tier", "mid")

        features = {
            "wall_treatment": f"{theme} painted walls",
            "floor_material": "vitrified tiles",
            "ceiling_treatment": "POP false ceiling",
            "furniture_items": [],
            "colour_palette": ["white", "grey", "neutral"],
            "detected_style": theme,
            "quality_tier": tier,
            "specific_changes": [
                "Wall treatment upgraded", "Flooring replaced",
                "Ceiling updated", "Lighting modernised",
            ],
            "estimated_wall_area_sqft": 200,
            "estimated_floor_area_sqft": 120,
            "room_condition": "fair",
        }

        return {
            "vision_features": {**features, "extraction_source": "static_fallback"},
            "image_features": features,
            "room_features": features,
            "layout_report": {"layout_score": "65/100", "walkable_space": "45%", "issues_detected": []},
            "style_label": theme,
            "style_confidence": 0.4,
            "explainable_recommendations": [],
            "material_quantities": {"paint_liters": 14.8, "floor_tiles_sqft": 132, "plywood_sqft": 36},
            "detected_objects": [],
            "material_types": ["vitrified_tile"],
            "wall_area_sqft": 200.0,
            "floor_area_sqft": 120.0,
            "room_dimensions": {
                "wall_area_sqft": 200, "floor_area_sqft": 120,
                "estimated_length_ft": 14.0, "estimated_width_ft": 12.0,
                "estimated_height_ft": 9.0,
            },
            "detected_changes": ["Room analysis in fallback mode"],
            "visual_style": [theme],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_image(state: Dict[str, Any]) -> Optional[bytes]:
        img_bytes = state.get("original_image_bytes")
        if img_bytes:
            return img_bytes
        img_b64 = state.get("original_image_b64", "")
        if img_b64:
            try:
                return base64.b64decode(img_b64)
            except Exception:
                pass
        return None