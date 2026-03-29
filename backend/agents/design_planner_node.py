"""
ARKEN — DesignPlannerAgentNode v2.0
=====================================
WHAT CHANGED:
  - Now reads cv_features (YOLOv8 + CLIP) from state to ground every
    material/BOQ recommendation to what is ACTUALLY in the uploaded image.
  - Pulls style-matched examples from the Interior Design datasets so
    recommendations reflect the detected style, not generic defaults.
  - Queries DIY dataset for renovation guidance relevant to the detected
    room issues (walls, lighting, plumbing etc.)
  - design_plan now includes image_grounded=True flag when CV data was used.
  - All existing downstream contracts preserved.

Pipeline position: node_rag_retrieval → [THIS] → node_budget_estimation
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Style → material preference mapping (from dataset analysis) ──────────────

STYLE_FLOOR_PREFERENCE: Dict[str, str] = {
    "Modern Minimalist":   "large-format vitrified tile (800x800mm)",
    "Scandinavian":        "light oak engineered wood",
    "Japandi":             "natural bamboo flooring",
    "Industrial":          "polished concrete or dark ceramic",
    "Bohemian":            "terracotta or patterned encaustic tile",
    "Contemporary Indian": "marble or large vitrified tile",
    "Traditional Indian":  "kadappa stone or teak wood",
    "Art Deco":            "geometric marble mosaic",
    "Mid-Century Modern":  "warm walnut hardwood",
    "Coastal":             "whitewashed wood-look tile",
    "Farmhouse":           "wide-plank reclaimed wood look",
}

STYLE_WALL_PREFERENCE: Dict[str, str] = {
    "Modern Minimalist":   "matte white or warm grey emulsion",
    "Scandinavian":        "soft white or pale blue emulsion",
    "Japandi":             "warm off-white or clay textured plaster",
    "Industrial":          "exposed concrete texture or dark grey",
    "Bohemian":            "terracotta, mustard, or jewel-tone emulsion",
    "Contemporary Indian": "warm beige with brass accent details",
    "Traditional Indian":  "ivory or warm white with carved panel detailing",
    "Art Deco":            "deep navy or forest green with gold trim",
    "Mid-Century Modern":  "warm white with walnut panel accent",
    "Coastal":             "ocean blue or sandy beige",
    "Farmhouse":           "shiplap white or rustic grey",
}

STYLE_CEILING_PREFERENCE: Dict[str, str] = {
    "Modern Minimalist":   "recessed LED false ceiling (plain white)",
    "Scandinavian":        "white exposed beam ceiling or plain white",
    "Japandi":             "natural wood slat ceiling or white flat",
    "Industrial":          "exposed concrete with track lighting",
    "Bohemian":            "macramé pendant clusters, no false ceiling",
    "Contemporary Indian": "multi-layer POP false ceiling with coves",
    "Traditional Indian":  "ornate POP cornice with centre medallion",
    "Art Deco":            "stepped geometric false ceiling with uplighting",
    "Mid-Century Modern":  "flush ceiling with Sputnik pendant",
    "Coastal":             "white-washed tray ceiling",
    "Farmhouse":           "exposed wooden beam ceiling",
}

# ── Object-specific renovation actions triggered by YOLO detections ──────────

OBJECT_RENOVATION_ACTIONS: Dict[str, List[str]] = {
    "sofa":         ["Reupholster or replace sofa fabric to match new style", "Add throw pillows aligned with colour palette"],
    "bed":          ["Update bed frame to match theme", "Add feature headboard wall treatment behind bed"],
    "dining table": ["Refinish dining table surface", "Update dining chairs to complement table"],
    "wardrobe":     ["Add laminate/acrylic shutters to wardrobe", "Install LED interior lighting in wardrobe"],
    "television":   ["Mount TV on feature wall", "Add floating TV unit in matching material"],
    "refrigerator": ["Panel refrigerator into kitchen cabinetry", "Update kitchen cabinet handles to match"],
    "sink":         ["Replace faucet with designer tap", "Update under-sink cabinet finish"],
    "chair":        ["Reupholster chairs in coordinating fabric", "Add cushioned seat pads"],
    "coffee table": ["Replace with style-matched centre table", "Add decorative tray and objects"],
    "indoor plant": ["Add curated planter collection per style", "Install built-in planter niche"],
}

# ── Lighting upgrades by detected lighting condition ─────────────────────────

LIGHTING_UPGRADES: Dict[str, List[str]] = {
    "dim":       ["Install recessed LED downlights (3000K warm)", "Add floor lamp for ambient fill", "Increase window glazing area if possible"],
    "artificial":["Add sheer curtains to diffuse natural light", "Install circadian rhythm tunable LED strips", "Position mirrors to bounce available light"],
    "natural":   ["Add smart dimmer switches to control glare", "Install UV-protective film on windows", "Layer with warm artificial accent lights for evening"],
    "warm":      ["Balance with 4000K task lighting in work areas", "Add cool-white under-cabinet lights in kitchen"],
    "cool":      ["Layer with 2700K Edison bulbs for warmth", "Add warm-toned table lamps for coziness"],
    "mixed":     ["Unify lighting colour temperature to 3000K", "Install scene control system for different moods"],
}


class DesignPlannerAgentNode:
    """
    Image-grounded design planner.
    Uses CV pipeline output (detected objects, style, lighting, materials)
    + dataset metadata to generate recommendations specific to the actual image.
    """

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        name = "design_planner_agent"
        logger.info(f"[{state.get('project_id', '')}] DesignPlannerAgent v2.0 starting")

        try:
            updates = self._plan(state)
        except Exception as e:
            logger.error(f"[design_planner] Error: {e}", exc_info=True)
            budget_inr = state.get("budget_inr", 750_000)
            updates = {
                "design_plan": {},
                "material_plan": {},
                "boq_line_items": [],
                "labour_estimate": int(budget_inr * 0.32),
                "total_cost_estimate": budget_inr,
                "schedule": {},
                "errors": (state.get("errors") or []) + [f"design_planner_agent: {e}"],
            }

        elapsed = round(time.perf_counter() - t0, 3)
        timings = dict(state.get("agent_timings") or {})
        timings[name] = elapsed
        updates["agent_timings"] = timings

        completed = list(state.get("completed_agents") or [])
        if name not in completed:
            completed.append(name)
        updates["completed_agents"] = completed

        logger.info(
            f"[design_planner] done in {elapsed}s — "
            f"total=₹{updates.get('total_cost_estimate', 0):,} "
            f"items={len(updates.get('boq_line_items', []))} "
            f"grounded={updates.get('design_plan', {}).get('image_grounded', False)}"
        )
        return updates

    # ─────────────────────────────────────────────────────────────────────────
    # Core planning — image-grounded
    # ─────────────────────────────────────────────────────────────────────────

    def _plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.design_planner import DesignPlannerAgent

        dims = state.get("room_dimensions") or {}
        floor_area_sqft = float(state.get("floor_area_sqft") or dims.get("floor_area_sqft", 120))
        wall_area_sqft = float(state.get("wall_area_sqft") or dims.get("wall_area_sqft", 200))
        budget_tier = state.get("budget_tier", "mid")
        budget_inr = state.get("budget_inr", 750_000)
        theme = state.get("theme", "Modern Minimalist")
        city = state.get("city", "Hyderabad")
        room_type = state.get("room_type", "bedroom")

        # ── Pull CV features from state ───────────────────────────────────────
        cv = state.get("cv_features") or {}
        detected_objects: List[str] = (
            state.get("detected_objects")
            or cv.get("detected_objects")
            or []
        )
        detected_style: str = (
            state.get("style_label")
            or cv.get("style")
            or theme
        )
        detected_lighting: str = cv.get("lighting", "mixed")
        detected_materials: List[str] = (
            state.get("material_types")
            or cv.get("materials")
            or []
        )
        cv_room_type: str = cv.get("room_type") or room_type
        image_grounded: bool = bool(cv and cv.get("extraction_source") not in ("cv_unavailable", ""))

        logger.info(
            f"[design_planner] CV grounding — style={detected_style} "
            f"objects={detected_objects[:5]} lighting={detected_lighting} "
            f"grounded={image_grounded}"
        )

        # ── Get style-matched dataset examples ────────────────────────────────
        dataset_context = self._get_dataset_style_context(detected_style, cv_room_type)

        # ── Get DIY renovation guidance for detected issues ───────────────────
        diy_guidance = self._get_diy_guidance(state, detected_objects, detected_lighting)

        # ── Build image-grounded quantities ───────────────────────────────────
        vision_quantities = state.get("material_quantities") or {}
        quantities = {
            "paint_liters": vision_quantities.get("paint_liters") or wall_area_sqft * 0.037 * 2,
            "floor_tiles_sqft": vision_quantities.get("floor_tiles_sqft") or floor_area_sqft * 1.1,
            "plywood_sqft": vision_quantities.get("plywood_sqft") or floor_area_sqft * 0.3,
            "wall_area_sqft": wall_area_sqft,
            "wall_tiles_sqft": vision_quantities.get("wall_tiles_sqft", 0),
        }

        # ── Pull Task 1 condition fields from state ───────────────────────────
        room_features = state.get("room_features") or state.get("vision_features") or {}
        condition_score  = (state.get("condition_score")
                            or room_features.get("condition_score") or 65)
        wall_condition   = (state.get("wall_condition")
                            or room_features.get("wall_condition") or "fair")
        floor_condition  = (state.get("floor_condition")
                            or room_features.get("floor_condition") or "fair")
        issues_detected  = (state.get("issues_detected")
                            or room_features.get("issues_detected") or [])
        renovation_scope = (state.get("renovation_scope")
                            or room_features.get("renovation_scope") or "partial")
        high_value_upgrades = (state.get("high_value_upgrades")
                               or room_features.get("high_value_upgrades") or [])

        # ── Run base design planner ───────────────────────────────────────────
        planner = DesignPlannerAgent()
        plan = planner.plan(
            theme=detected_style,          # use DETECTED style, not just user-chosen theme
            budget_inr=budget_inr,
            budget_tier=budget_tier,
            area_sqft=floor_area_sqft,
            room_type=cv_room_type,        # use CV-detected room type
            city=city,
            quantities=quantities,
            # NEW: condition fields from Task 1 Gemini extraction
            wall_condition=wall_condition,
            floor_condition=floor_condition,
            issues_detected=issues_detected,
            renovation_scope=renovation_scope,
            high_value_upgrades=high_value_upgrades,
            condition_score=int(condition_score),
        )

        # ── Build image-specific action items ─────────────────────────────────
        image_specific_actions = self._build_image_specific_actions(
            detected_objects=detected_objects,
            detected_style=detected_style,
            detected_lighting=detected_lighting,
            detected_materials=detected_materials,
            room_type=cv_room_type,
            budget_tier=budget_tier,
        )

        # ── Build grounded BOQ (enriched line items) ──────────────────────────
        # ── Use plan line_items directly — no re-processing ──────────────────────────────────
        # design_planner v3 already includes all labour as visible line items,
        # civil materials, false ceiling components, wardrobe, skirting etc.
        # _build_grounded_boq() is NOT called here — it strips rate_inr and
        # recalculates totals incorrectly. We use the plan items directly and
        # append any object-specific items detected in the image.
        grounded_boq = list(plan.get("line_items", []))

        # ── Append object-specific items from YOLO/CV detection ───────────────
        existing_categories = {str(i.get("category", "")).lower() for i in grounded_boq}
        tier = budget_tier.lower()

        if "sofa" in [o.lower() for o in detected_objects]:
            if "upholstery" not in existing_categories:
                _sofa_rates = {"basic": 12000, "mid": 28000, "premium": 65000}
                rate = _sofa_rates.get(tier, 28000)
                grounded_boq.append({
                    "category": "Upholstery", "brand": "Local Contractor",
                    "product": f"Sofa Reupholstery — {detected_style} fabric",
                    "qty": 1.0, "unit": "set", "rate_inr": rate, "total_inr": rate,
                    "priority": "nice_to_have", "tier_applied": tier,
                    "note": "Reupholster existing sofa in style-matched fabric",
                })

        if "bed" in [o.lower() for o in detected_objects]:
            if "headboard" not in existing_categories:
                _hb_rates = {"basic": 10000, "mid": 25000, "premium": 58000}
                rate = _hb_rates.get(tier, 25000)
                grounded_boq.append({
                    "category": "Headboard", "brand": "Local Carpenter",
                    "product": f"Custom {detected_style} Headboard with Feature Wall",
                    "qty": 1.0, "unit": "piece", "rate_inr": rate, "total_inr": rate,
                    "priority": "nice_to_have", "tier_applied": tier,
                    "note": "Custom headboard + feature wall treatment behind bed",
                })

        # ── Recompute totals from actual line items ────────────────────────────
        labour_total   = sum(i["total_inr"] for i in grounded_boq
                              if i.get("category", "").startswith("Labour"))
        material_total = sum(i["total_inr"] for i in grounded_boq
                              if not i.get("category", "").startswith("Labour"))
        subtotal       = labour_total + material_total
        gst            = int(subtotal * 0.18)
        cont_pct       = {"basic": 0.15, "mid": 0.12, "premium": 0.10}.get(tier, 0.12)
        contingency    = int(subtotal * cont_pct)
        grand_total    = subtotal + gst + contingency

        # ── Layout suggestions ────────────────────────────────────────────────
        layout_suggestions = self._get_layout_suggestions(state, detected_objects)

        # ── Schedule ─────────────────────────────────────────────────────────
        schedule = {}
        try:
            from agents.coordinator import ProjectCoordinatorAgent
            coordinator = ProjectCoordinatorAgent()
            schedule = coordinator.generate_schedule(
                area_sqft=floor_area_sqft,
                budget_inr=budget_inr,
                room_type=cv_room_type,
                city=city,
                start_date=date.today() + timedelta(days=7),
            )
        except Exception as se:
            logger.warning(f"[design_planner] Schedule generation failed: {se}")

        enriched_plan = {
            **plan,
            # Override with recomputed values from actual line items
            "total_inr":      grand_total,
            "material_inr":   material_total,
            "labour_inr":     labour_total,
            "gst_inr":        gst,
            "contingency_inr": contingency,
            "line_items":     grounded_boq,
            # Image-grounding metadata
            "image_grounded": image_grounded,
            "detected_style": detected_style,
            "detected_objects": detected_objects,
            "detected_lighting": detected_lighting,
            "detected_materials": detected_materials,
            "cv_room_type": cv_room_type,
            "image_specific_actions": image_specific_actions,
            "style_floor_recommendation": STYLE_FLOOR_PREFERENCE.get(detected_style, "vitrified tiles"),
            "style_wall_recommendation": STYLE_WALL_PREFERENCE.get(detected_style, "premium emulsion"),
            "style_ceiling_recommendation": STYLE_CEILING_PREFERENCE.get(detected_style, "POP false ceiling"),
            "layout_suggestions": layout_suggestions,
            "dataset_style_examples": dataset_context,
            "diy_guidance": diy_guidance,
        }

        return {
            "design_plan": enriched_plan,
            "material_plan": plan.get("recommendations", {}),
            "boq_line_items": grounded_boq,
            # labour_estimate is the sum of all Labour- line items (visible in BOQ table)
            "labour_estimate": labour_total,
            "total_cost_estimate": grand_total,
            "schedule": schedule,
            "detected_style_grounded": detected_style,
            "image_specific_actions": image_specific_actions,
            "diy_renovation_tips": diy_guidance,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Image-specific action builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_image_specific_actions(
        self,
        detected_objects: List[str],
        detected_style: str,
        detected_lighting: str,
        detected_materials: List[str],
        room_type: str,
        budget_tier: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate renovation actions SPECIFIC to what's detected in the image.
        Every action is traceable to a CV signal (object, style, lighting, material).
        """
        actions = []

        # 1. Object-specific actions (YOLO-grounded)
        for obj in detected_objects[:6]:
            obj_actions = OBJECT_RENOVATION_ACTIONS.get(obj.lower(), [])
            for action in obj_actions[:1]:  # top 1 per object
                actions.append({
                    "action": action,
                    "trigger": f"Detected object: {obj}",
                    "category": "object_specific",
                    "priority": "high",
                    "grounding": "yolo_detection",
                })

        # 2. Style-specific surface upgrades (CLIP-grounded)
        floor_rec = STYLE_FLOOR_PREFERENCE.get(detected_style, "vitrified tiles")
        actions.append({
            "action": f"Replace flooring with {floor_rec} to match detected {detected_style} aesthetic",
            "trigger": f"Detected style: {detected_style}",
            "category": "flooring",
            "priority": "high",
            "grounding": "clip_style_detection",
        })

        wall_rec = STYLE_WALL_PREFERENCE.get(detected_style, "premium emulsion")
        actions.append({
            "action": f"Apply {wall_rec} on walls — primary treatment for {detected_style} style",
            "trigger": f"Detected style: {detected_style}",
            "category": "walls",
            "priority": "high",
            "grounding": "clip_style_detection",
        })

        ceiling_rec = STYLE_CEILING_PREFERENCE.get(detected_style, "POP false ceiling with LED")
        actions.append({
            "action": f"Install {ceiling_rec} as ceiling treatment",
            "trigger": f"Detected style: {detected_style}",
            "category": "ceiling",
            "priority": "medium",
            "grounding": "clip_style_detection",
        })

        # 3. Lighting-specific upgrades (CLIP-grounded)
        lighting_upgrades = LIGHTING_UPGRADES.get(detected_lighting, [])
        for upgrade in lighting_upgrades[:2]:
            actions.append({
                "action": upgrade,
                "trigger": f"Detected lighting condition: {detected_lighting}",
                "category": "lighting",
                "priority": "medium",
                "grounding": "clip_lighting_detection",
            })

        # 4. Material-specific actions
        for material in detected_materials[:3]:
            if "wood" in material.lower():
                actions.append({
                    "action": "Sand and refinish existing wood surfaces to restore natural grain",
                    "trigger": f"Detected material: {material}",
                    "category": "materials",
                    "priority": "medium",
                    "grounding": "material_inference",
                })
            elif "marble" in material.lower():
                actions.append({
                    "action": "Polish and seal marble surfaces — add matching marble accessories",
                    "trigger": f"Detected material: {material}",
                    "category": "materials",
                    "priority": "low",
                    "grounding": "material_inference",
                })

        return actions[:12]

    # ─────────────────────────────────────────────────────────────────────────
    # Grounded BOQ builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_grounded_boq(
        self,
        base_boq: List[Dict],
        detected_style: str,
        detected_objects: List[str],
        floor_area_sqft: float,
        wall_area_sqft: float,
        budget_tier: str,
    ) -> List[Dict[str, Any]]:
        """
        Enriches base BOQ line items with image-specific descriptions.
        Replaces generic descriptions with style+object-specific ones.
        """
        enriched = []
        tier = budget_tier.lower()

        for item in base_boq:
            if not isinstance(item, dict):
                continue
            enriched_item = dict(item)
            category = str(item.get("category", "")).lower()

            # Override descriptions based on detected style
            if category == "paint" and detected_style:
                wall_rec = STYLE_WALL_PREFERENCE.get(detected_style, item.get("description", ""))
                enriched_item["description"] = (
                    f"{item.get('product', '')} — {wall_rec} "
                    f"({detected_style} palette)"
                )
                enriched_item["style_grounded"] = True

            elif category in ("tile", "flooring", "tiles"):
                floor_rec = STYLE_FLOOR_PREFERENCE.get(detected_style, item.get("description", ""))
                enriched_item["description"] = (
                    f"{item.get('product', '')} — {floor_rec} "
                    f"({detected_style} style)"
                )
                enriched_item["style_grounded"] = True

            enriched.append(enriched_item)

        # Add object-specific BOQ items not in base BOQ
        existing_categories = {str(i.get("category", "")).lower() for i in enriched}

        if "sofa" in detected_objects and "upholstery" not in existing_categories:
            enriched.append({
                "category": "Upholstery",
                "description": f"Sofa reupholstery in {detected_style}-appropriate fabric",
                "quantity": 1,
                "unit": "set",
                "total_inr": 15000 if tier == "basic" else 35000 if tier == "mid" else 75000,
                "priority": "medium",
                "grounding": "yolo_detected_sofa",
                "style_grounded": True,
            })

        if ("bed" in detected_objects or "bedroom" in str(detected_objects).lower()):
            if "headboard" not in existing_categories:
                enriched.append({
                    "category": "Headboard",
                    "description": f"Custom {detected_style} headboard with feature wall treatment",
                    "quantity": 1,
                    "unit": "piece",
                    "total_inr": 12000 if tier == "basic" else 28000 if tier == "mid" else 65000,
                    "priority": "medium",
                    "grounding": "yolo_detected_bed",
                    "style_grounded": True,
                })

        return enriched

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset context lookup
    # ─────────────────────────────────────────────────────────────────────────

    def _get_dataset_style_context(
        self,
        detected_style: str,
        room_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Pull matching examples from Interior Design datasets.
        Returns style-matched image records as context for recommendations.

        BUG FIX v2.1:
          - Deduplicates across both loaders by (style, room_type, materials_hash)
          - Caps same-style examples at 3 max; remaining slots filled with
            related styles to ensure variety in the UI panel
          - Falls back gracefully when dataset is unavailable
          - Never returns more than 2 examples with identical (style, room_type)
        """
        RELATED_STYLES: Dict[str, List[str]] = {
            "Modern Minimalist":   ["Scandinavian", "Japandi", "Contemporary Indian"],
            "Scandinavian":        ["Modern Minimalist", "Japandi"],
            "Japandi":             ["Scandinavian", "Modern Minimalist"],
            "Industrial":          ["Modern Minimalist", "Art Deco"],
            "Bohemian":            ["Contemporary Indian", "Traditional Indian"],
            "Contemporary Indian": ["Modern Minimalist", "Traditional Indian"],
            "Traditional Indian":  ["Contemporary Indian", "Bohemian"],
            "Art Deco":            ["Industrial", "Mid-Century Modern"],
            "Mid-Century Modern":  ["Art Deco", "Scandinavian"],
            "Coastal":             ["Scandinavian", "Bohemian"],
            "Farmhouse":           ["Coastal", "Scandinavian"],
        }

        try:
            from services.datasets.dataset_loader import ARKENDatasetRegistry
            registry = ARKENDatasetRegistry.get()

            examples: List[Dict[str, Any]] = []
            seen_signatures: set = set()   # (style, room_type, mat_tuple) → dedup key

            def _add_example(rec, source_style: str) -> bool:
                """Add record if not duplicate. Returns True if added."""
                mat_key = tuple(sorted((rec.materials or [])[:3]))
                sig = (source_style, rec.room_type, mat_key)
                if sig in seen_signatures:
                    return False
                # Max 2 examples with identical (style, room_type)
                same_style_room = sum(
                    1 for e in examples
                    if e["style"] == source_style and e["room_type"] == rec.room_type
                )
                if same_style_room >= 2:
                    return False
                seen_signatures.add(sig)
                examples.append({
                    "room_type": rec.room_type,
                    "style":     source_style,
                    "materials": (rec.materials or [])[:4],
                    "objects":   (rec.objects or [])[:5],
                    "source":    rec.source_dataset,
                })
                return True

            # ── Phase 1: exact style match from both loaders ──────────────────
            for loader in [registry.material_style, registry.interior_images]:
                if not loader.available:
                    continue
                matches = loader.get_by_style(detected_style)
                for rec in (matches or []):
                    if len(examples) >= 3:
                        break
                    _add_example(rec, detected_style)

            # ── Phase 2: room_type fallback for primary style ─────────────────
            if len(examples) < 3:
                for loader in [registry.material_style, registry.interior_images]:
                    if not loader.available:
                        continue
                    rt_matches = getattr(loader, "get_by_room_type", lambda x: [])(room_type)
                    for rec in (rt_matches or []):
                        if len(examples) >= 3:
                            break
                        # Only take room_type matches that share the same style
                        if getattr(rec, "style", "") == detected_style:
                            _add_example(rec, detected_style)

            # ── Phase 3: related styles to fill remaining slots (max 5 total) ─
            # This prevents the all-minimalist-bathroom problem
            related = RELATED_STYLES.get(detected_style, ["Modern Minimalist"])
            for related_style in related:
                if len(examples) >= 5:
                    break
                for loader in [registry.material_style, registry.interior_images]:
                    if not loader.available or len(examples) >= 5:
                        continue
                    rel_matches = loader.get_by_style(related_style)
                    for rec in (rel_matches or []):
                        if len(examples) >= 5:
                            break
                        _add_example(rec, related_style)

            if examples:
                primary_count = sum(1 for e in examples if e["style"] == detected_style)
                logger.info(
                    f"[design_planner] Dataset: {len(examples)} examples "
                    f"({primary_count} × {detected_style}, "
                    f"{len(examples) - primary_count} related styles)"
                )
            else:
                logger.debug(f"[design_planner] Dataset: no examples for {detected_style}")

            return examples[:5]

        except Exception as e:
            logger.debug(f"[design_planner] Dataset context lookup failed (non-critical): {e}")
            # Return synthetic fallback examples so the section is never empty
            return self._synthetic_style_examples(detected_style, room_type)

    def _synthetic_style_examples(
        self,
        detected_style: str,
        room_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Minimal synthetic examples used when the dataset loader is unavailable.
        Prevents the UI panel from being empty or showing stale cached data.
        """
        STYLE_MATERIALS: Dict[str, List[str]] = {
            "Modern Minimalist":   ["vitrified tile", "white emulsion", "POP false ceiling", "LED strip"],
            "Scandinavian":        ["light oak wood", "white paint", "linen fabric", "pendant lamp"],
            "Japandi":             ["bamboo", "clay plaster", "natural fibre", "paper lamp"],
            "Industrial":          ["exposed brick", "steel", "concrete", "Edison bulb"],
            "Bohemian":            ["terracotta tile", "woven fabric", "rattan", "macramé"],
            "Contemporary Indian": ["marble", "brass", "jali panel", "warm emulsion"],
            "Traditional Indian":  ["teak wood", "kadappa stone", "handloom fabric", "brass lamp"],
        }
        RELATED_STYLES: Dict[str, List[str]] = {
            "Modern Minimalist": ["Scandinavian", "Japandi"],
            "Scandinavian":      ["Modern Minimalist", "Japandi"],
            "Contemporary Indian": ["Traditional Indian", "Modern Minimalist"],
        }
        primary_mats = STYLE_MATERIALS.get(detected_style, ["vitrified tile", "emulsion paint"])
        related = RELATED_STYLES.get(detected_style, ["Scandinavian", "Contemporary Indian"])
        examples = [
            {"room_type": room_type, "style": detected_style,
             "materials": primary_mats, "objects": [], "source": "synthetic"},
        ]
        for rs in related[:2]:
            examples.append({
                "room_type": room_type, "style": rs,
                "materials": STYLE_MATERIALS.get(rs, primary_mats),
                "objects": [], "source": "synthetic",
            })
        return examples

    # ─────────────────────────────────────────────────────────────────────────
    # DIY guidance from dataset
    # ─────────────────────────────────────────────────────────────────────────

    def _get_diy_guidance(
        self,
        state: Dict[str, Any],
        detected_objects: List[str],
        detected_lighting: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant DIY renovation guidance grounded in the actual image signals.
        Query terms are derived ONLY from:
          - room_type (from CV detection)
          - detected_objects (from YOLO)
          - detected_lighting condition (from CLIP)
          - issues_detected (from Gemini vision)
          - detected_style (from CLIP)
        This prevents unrelated results (e.g. laundry hacks for a bedroom).
        """
        try:
            from services.datasets.dataset_loader import ARKENDatasetRegistry
            registry = ARKENDatasetRegistry.get()

            if not registry.diy_renovation.available:
                return []

            # ── Build query terms strictly from image signals ─────────────────
            query_terms: List[str] = []

            room_type: str = (
                state.get("room_type")
                or (state.get("cv_features") or {}).get("room_type", "bedroom")
            )
            detected_style: str = (
                state.get("style_label")
                or state.get("detected_style_grounded")
                or "Modern Minimalist"
            )

            # 1. Room-type specific terms (most specific, highest priority)
            ROOM_QUERY_MAP: Dict[str, List[str]] = {
                "bedroom":     ["bedroom", "wardrobe", "flooring", "walls", "lighting", "paint"],
                "kitchen":     ["kitchen", "tiles", "plumbing", "electrical", "platform", "cabinet"],
                "bathroom":    ["bathroom", "waterproofing", "tiles", "plumbing", "fittings"],
                "living_room": ["living room", "flooring", "walls", "ceiling", "lighting", "paint"],
                "full_home":   ["flooring", "walls", "electrical", "plumbing", "ceiling", "paint"],
                "study":       ["walls", "flooring", "lighting", "electrical"],
                "dining_room": ["flooring", "walls", "lighting"],
            }
            query_terms.extend(ROOM_QUERY_MAP.get(room_type, ["walls", "flooring", "lighting"]))

            # 2. Detected objects (YOLO-grounded)
            OBJECT_QUERY_MAP: Dict[str, List[str]] = {
                "wardrobe":     ["wardrobe", "carpentry", "plywood"],
                "bed":          ["bedroom", "flooring", "walls"],
                "television":   ["electrical", "lighting", "walls"],
                "sink":         ["plumbing", "waterproofing"],
                "toilet":       ["bathroom", "plumbing", "waterproofing", "tiles"],
                "refrigerator": ["kitchen", "electrical"],
                "sofa":         ["flooring", "walls", "living room"],
                "chair":        ["flooring", "walls"],
            }
            for obj in detected_objects[:6]:
                obj_terms = OBJECT_QUERY_MAP.get(obj.lower(), [])
                query_terms.extend(obj_terms)

            # 3. Lighting signals (CLIP-grounded)
            if detected_lighting in ("dim", "artificial"):
                query_terms.extend(["lighting", "electrical", "LED"])
            elif detected_lighting == "mixed":
                query_terms.extend(["lighting", "LED"])

            # 4. Issues from Gemini vision
            vision = state.get("vision_features") or state.get("room_features") or {}
            issues_str = " ".join(
                str(i).lower() for i in (
                    state.get("issues_detected")
                    or vision.get("issues_detected", [])
                )
            )
            if any(kw in issues_str for kw in ("crack", "cracking")):
                query_terms.extend(["walls", "structural", "crack"])
            if any(kw in issues_str for kw in ("damp", "seepage", "mould", "leak")):
                query_terms.extend(["waterproofing", "damp", "walls"])
            if any(kw in issues_str for kw in ("peel", "flak")):
                query_terms.extend(["paint", "walls", "preparation"])

            # 5. Style-specific material terms
            STYLE_QUERY_MAP: Dict[str, List[str]] = {
                "Modern Minimalist":   ["flooring", "false ceiling", "LED", "paint", "tiles"],
                "Scandinavian":        ["flooring", "paint", "lighting"],
                "Contemporary Indian": ["flooring", "false ceiling", "paint"],
                "Traditional Indian":  ["flooring", "carpentry", "paint"],
                "Industrial":          ["flooring", "electrical", "walls"],
            }
            query_terms.extend(STYLE_QUERY_MAP.get(detected_style, ["paint", "flooring"]))

            # Deduplicate and limit
            unique_terms = list(dict.fromkeys(query_terms))[:10]

            chunks = registry.diy_renovation.get_relevant_chunks(
                query_terms=unique_terms,
                max_results=4,
            )

            # ── Filter out any chunks that don't match the room context ────────
            # This is the safety net that prevents laundry/unrelated content
            room_keywords = set(ROOM_QUERY_MAP.get(room_type, []))
            room_keywords.add(room_type.replace("_", " "))

            relevant_chunks = []
            for chunk in chunks:
                chunk_text = (
                    chunk.content + " " + chunk.chapter_title + " " + chunk.category
                ).lower()
                # Accept if chunk content contains any room-specific keyword
                if any(kw.lower() in chunk_text for kw in room_keywords):
                    relevant_chunks.append(chunk)
                else:
                    logger.debug(
                        f"[design_planner] DIY chunk filtered out (not relevant to "
                        f"{room_type}): '{chunk.chapter_title}'"
                    )

            # Fall back to all matched chunks if filtering removes everything
            if not relevant_chunks:
                relevant_chunks = chunks[:3]

            guidance = []
            for chunk in relevant_chunks[:4]:
                summary = chunk.content[:300].strip()
                if len(chunk.content) > 300:
                    summary += "..."
                guidance.append({
                    "category": chunk.category,
                    "tip": chunk.chapter_title,
                    "guidance": summary,
                    "source": "India Renovation Knowledge Base",
                    "link": chunk.clip_link,
                })

            if guidance:
                logger.info(
                    f"[design_planner] DIY: retrieved {len(guidance)} guidance chunks "
                    f"for room_type={room_type}, style={detected_style}"
                )
            else:
                logger.debug(f"[design_planner] DIY: no relevant chunks for {room_type}")

            return guidance

        except Exception as e:
            logger.debug(f"[design_planner] DIY guidance lookup failed (non-critical): {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Layout suggestions
    # ─────────────────────────────────────────────────────────────────────────

    def _get_layout_suggestions(
        self,
        state: Dict[str, Any],
        detected_objects: List[str],
    ) -> List[str]:
        suggestions = []
        layout = state.get("layout_report") or {}
        room_type = state.get("room_type", "bedroom")
        user_goals = state.get("user_goals") or {}
        goal = user_goals.get("primary_goal", "personal_comfort") if isinstance(user_goals, dict) else "personal_comfort"

        # Layout issues from vision
        for issue in (layout.get("issues") or layout.get("issues_detected") or [])[:3]:
            suggestions.append(f"Fix: {issue}")
        suggestions.extend((layout.get("suggestions") or [])[:2])

        # Object-based layout advice
        if "sofa" in detected_objects and "coffee table" not in detected_objects:
            suggestions.append("Add a coffee table — currently missing for a complete seating arrangement")
        if "bed" in detected_objects and len(detected_objects) < 3:
            suggestions.append("Room appears sparse — consider bedside tables and a wardrobe for functional balance")
        if len(detected_objects) > 8:
            suggestions.append("Room is over-furnished — remove 2-3 items for better flow and visual space")

        # Goal-based
        if goal == "maximise_resale_value":
            suggestions.append(f"Prioritise kitchen/bathroom over {room_type} for max resale ROI")
        elif goal == "maximise_rental_yield":
            suggestions.append("Keep colour scheme neutral — tenant preference drives rental premium")
        elif goal == "luxury_upgrade":
            suggestions.append("Statement lighting as centrepiece — high visual impact per rupee")

        return suggestions[:6]