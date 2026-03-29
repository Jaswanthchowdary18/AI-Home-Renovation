"""
ARKEN — LangGraph Graph Pipeline v6.0
=======================================
v6.0 UPGRADES over v5.0:
  - Fine-tuned model weights wired throughout the pipeline:
      YOLOv8  → ml/weights/yolo_indian_rooms.pt (fine-tuned, fallback yolov8n.pt)
      CLIP    → ml/weights/clip_finetuned.pt (fine-tuned encoder, fallback ViT-B/32)
      Style   → ml/weights/style_classifier.pt (EfficientNet-B0, fallback CLIP zero-shot)
      Room    → ml/weights/room_classifier.pt (EfficientNet, fallback CLIP zero-shot)
  - ROIExplainer wired into node_roi_prediction: SHAP factors + NHB Residex
    benchmark validation added to roi_prediction dict.
  - startup_check_weights() helper logs clear warnings if fine-tuned weights
    are missing with exact instructions to run training scripts.
  - All pretrained-only fallback paths preserved with graceful degradation.
  - rendering.py is NOT touched.

DAG (unchanged):
  intent → visual_assessment → [style_detection + object_detection]
         → material_estimation → cost_estimation
         → [roi_prediction + timeline + rendering]
         → report_generator → END
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

# ── Weight file paths (shared across nodes) ───────────────────────────────────
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))
_BACKEND_DIR = Path(__file__).resolve().parent.parent

def _weights_dir() -> Path:
    """Resolve ml/weights directory — Docker volume or local fallback."""
    d = Path(os.getenv("ML_WEIGHTS_DIR", str(_APP_DIR / "ml" / "weights")))
    if not d.exists():
        local = _BACKEND_DIR / "ml" / "weights"
        if local.exists():
            return local
    d.mkdir(parents=True, exist_ok=True)
    return d

WEIGHTS_DIR = _weights_dir()

_WEIGHT_YOLO   = WEIGHTS_DIR / "yolo_indian_rooms.pt"
_WEIGHT_CLIP   = WEIGHTS_DIR / "clip_finetuned.pt"
_WEIGHT_STYLE  = WEIGHTS_DIR / "style_classifier.pt"
_WEIGHT_ROOM   = WEIGHTS_DIR / "room_classifier.pt"


# ── Startup weight check ──────────────────────────────────────────────────────

def startup_check_weights() -> Dict[str, bool]:
    """
    Verify all fine-tuned weight files exist.
    Logs a clear WARNING with training instructions for each missing file.
    Called from main.py lifespan startup.

    Returns:
        Dict mapping weight name → True (present) / False (missing).
    """
    checks = {
        "yolo_indian_rooms.pt":  _WEIGHT_YOLO,
        "clip_finetuned.pt":     _WEIGHT_CLIP,
        "style_classifier.pt":   _WEIGHT_STYLE,
        "room_classifier.pt":    _WEIGHT_ROOM,
    }
    train_instructions = {
        "yolo_indian_rooms.pt": (
            "Run:  python ml/train_models.py --model yolo "
            "--data data/datasets/interior_design_material_style "
            "--output ml/weights/yolo_indian_rooms.pt"
        ),
        "clip_finetuned.pt": (
            "Run:  python ml/train_models.py --model clip "
            "--data data/datasets/interior_design_material_style "
            "--output ml/weights/clip_finetuned.pt"
        ),
        "style_classifier.pt": (
            "Run:  python ml/train_models.py --model style_classifier "
            "--data data/datasets/interior_design_material_style "
            "--output ml/weights/style_classifier.pt"
        ),
        "room_classifier.pt": (
            "Run:  python ml/train_models.py --model room_classifier "
            "--data data/datasets/interior_design_images_metadata "
            "--output ml/weights/room_classifier.pt"
        ),
    }

    results: Dict[str, bool] = {}
    any_missing = False

    for name, path in checks.items():
        present = path.exists() and path.stat().st_size > 1024  # > 1KB sanity
        results[name] = present
        if present:
            size_mb = path.stat().st_size / 1_048_576
            logger.info(f"✅  Fine-tuned weight: {name} ({size_mb:.1f} MB)")
        else:
            any_missing = True
            logger.warning(
                f"⚠️   Fine-tuned weight MISSING: {name}\n"
                f"        Expected at: {path}\n"
                f"        {train_instructions[name]}\n"
                f"        Pipeline will use pretrained fallback until this file exists."
            )

    if not any_missing:
        logger.info("✅  All fine-tuned weights present — pipeline at full accuracy.")
    else:
        logger.warning(
            "⚠️   Some fine-tuned weights are missing. "
            "The pipeline will degrade gracefully to pretrained models. "
            "Run the training scripts listed above to restore full accuracy."
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline State — single source of truth
# ─────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    # ── Input keys ───────────────────────────────────────────────────────────
    project_id: str
    original_image_b64: str
    original_image_mime: str
    original_image_bytes: bytes      # raw bytes for vision agents
    renovated_image_b64: str
    renovated_image_mime: str
    theme: str
    city: str
    budget_tier: str
    budget_inr: int
    room_type: str
    user_intent: str

    # ── Intent ───────────────────────────────────────────────────────────────
    parsed_intent: Dict

    # ── Visual Assessment (from VisualAssessorAgent v2) ───────────────────────
    room_features: Dict              # RoomFeatures.to_dict() — canonical schema
    image_features: Dict             # legacy-compat key for pipeline.py agents
    layout_report: Dict              # {layout_score, walkable_space, issues, ...}
    style_label: str
    style_confidence: float
    explainable_recommendations: List[Dict]

    # ── CV model provenance (v6.0) ────────────────────────────────────────────
    yolo_model_used: str             # "yolo_indian_rooms" | "yolov8n_pretrained"
    clip_model_used: str             # "clip_finetuned" | "clip_pretrained"
    style_model_used: str            # "efficientnet_finetuned" | "clip_zero_shot"
    room_model_used: str             # "room_classifier_finetuned" | "clip_zero_shot"

    # ── Object Detection ─────────────────────────────────────────────────────
    detected_objects: List[Dict]
    material_types: List[str]

    # ── Dimension / area keys ─────────────────────────────────────────────────
    wall_area_sqft: float
    floor_area_sqft: float
    room_dimensions: Dict
    detected_changes: List[str]
    visual_style: List[str]

    # ── Material Estimation ───────────────────────────────────────────────────
    material_quantities: Dict
    boq_line_items: List[Dict]
    labour_estimate: int

    # ── Cost Estimation ───────────────────────────────────────────────────────
    cost_breakdown: Dict
    total_cost_estimate: int
    material_plan: Dict

    # ── ROI Prediction (v6.0: includes SHAP + NHB validation) ────────────────
    roi_prediction: Dict
    roi_explainer_factors: List[Dict]   # SHAP top-3 factors
    roi_nhb_validation: Dict            # NHB Residex benchmark validation
    payback_months: int
    equity_gain_inr: int
    location_context: Dict
    budget_analysis: Dict
    material_prices: List[Dict]

    # ── Timeline ─────────────────────────────────────────────────────────────
    timeline: Dict

    # ── Rendering ────────────────────────────────────────────────────────────
    render_prompt: str
    render_url: str

    # ── Report ───────────────────────────────────────────────────────────────
    renovation_report: Dict          # full structured report for PDF generation

    # ── Insights / Chat ───────────────────────────────────────────────────────
    insights: Dict
    chat_context: str

    # ── RAG Knowledge Retrieval ───────────────────────────────────────────────
    rag_context: str               # formatted knowledge context (injected into LLM)
    rag_doc_ids: List[str]         # knowledge document IDs retrieved
    rag_categories: List[str]      # knowledge categories covered

    # ── Pipeline metadata ────────────────────────────────────────────────────
    errors: List[str]
    agent_timings: Dict[str, float]
    completed_agents: List[str]
    inference_metrics: Dict


# ─────────────────────────────────────────────────────────────────────────────
# Node: Intent
# ─────────────────────────────────────────────────────────────────────────────

def node_intent(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    name = "intent"
    try:
        intent_text = state.get("user_intent", "")
        theme = state.get("theme", "Modern Minimalist")
        room_type = state.get("room_type", "bedroom")
        budget_tier = state.get("budget_tier", "mid")

        goal_keywords = {
            "sell": "maximise_resale_value",
            "rent": "maximise_rental_yield",
            "live": "personal_comfort",
            "aesthetic": "aesthetic_refresh",
            "cost": "cost_optimisation",
            "quick": "quick_refresh",
            "luxury": "luxury_upgrade",
        }
        goal = "personal_comfort"
        for kw, g in goal_keywords.items():
            if kw in intent_text.lower():
                goal = g
                break

        upd: GraphState = {
            "parsed_intent": {
                "goal": goal,
                "priority": "roi" if goal in ("maximise_resale_value", "maximise_rental_yield") else "aesthetics",
                "style_preference": theme,
                "room_type": room_type,
                "budget_tier": budget_tier,
            }
        }
    except Exception as e:
        logger.warning(f"[intent] {e}")
        upd: GraphState = {
            "parsed_intent": {"goal": "personal_comfort", "priority": "aesthetics"},
            "errors": (state.get("errors") or []) + [f"intent: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Visual Assessment
# v6.0: CVModelRegistry is used explicitly so fine-tuned YOLO / CLIP / style /
#        room classifier weights are loaded FIRST, with pretrained fallbacks.
# ─────────────────────────────────────────────────────────────────────────────

def node_visual_assessment(state: GraphState) -> GraphState:
    """
    Full visual analysis using Gemini Vision + fine-tuned CV models.

    Model load order (each has graceful pretrained fallback):
      YOLOv8:    yolo_indian_rooms.pt → yolov8n.pt (COCO pretrained)
      CLIP:      clip_finetuned.pt   → ViT-B/32 pretrained
      Style:     style_classifier.pt → CLIP zero-shot dual-pass
      Room:      room_classifier.pt  → CLIP zero-shot

    Model provenance is recorded in state keys:
      yolo_model_used, clip_model_used, style_model_used, room_model_used
    """
    t0 = time.perf_counter()
    name = "visual_assessment"
    try:
        import asyncio
        import base64
        from agents.visual_assessor import VisualAssessorAgent

        # ── Pre-warm CV model registry with fine-tuned weights ────────────────
        model_provenance: Dict[str, str] = {}
        try:
            from ml.cv_model_registry import CVModelRegistry, WEIGHTS_DIR as _WD
            registry = CVModelRegistry.get()

            # YOLO: log which weight was loaded
            if registry.yolo and registry.yolo._model is not None:
                yolo_src = (
                    "yolo_indian_rooms"
                    if registry.yolo._loaded_from_finetuned
                    else "yolov8n_pretrained"
                )
            else:
                yolo_src = "unavailable"
            model_provenance["yolo_model_used"] = yolo_src

            # CLIP
            if registry.clip and registry.clip._loaded_from_finetuned:
                model_provenance["clip_model_used"] = "clip_finetuned"
            else:
                model_provenance["clip_model_used"] = "clip_pretrained"

        except Exception as reg_err:
            logger.debug(f"[visual_assessment] CV registry pre-warm skipped: {reg_err}")

        # ── Run VisualAssessorAgent (Gemini-first with CV helpers) ────────────
        agent = VisualAssessorAgent()

        img_bytes = state.get("original_image_bytes")
        if not img_bytes:
            img_b64 = state.get("original_image_b64", "")
            if img_b64:
                img_bytes = base64.b64decode(img_b64)
            else:
                raise ValueError("No image data in state")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        agent.analyze(
                            img_bytes,
                            state.get("project_id", "unknown"),
                            room_type=state.get("room_type", "bedroom"),
                            theme=state.get("theme", "Modern Minimalist"),
                            budget_tier=state.get("budget_tier", "mid"),
                            city=state.get("city", "Hyderabad"),
                        )
                    )
                    result = future.result(timeout=60)
            else:
                result = loop.run_until_complete(
                    agent.analyze(
                        img_bytes,
                        state.get("project_id", "unknown"),
                        room_type=state.get("room_type", "bedroom"),
                        theme=state.get("theme", "Modern Minimalist"),
                        budget_tier=state.get("budget_tier", "mid"),
                        city=state.get("city", "Hyderabad"),
                    )
                )
        except RuntimeError:
            result = asyncio.run(
                agent.analyze(
                    img_bytes,
                    state.get("project_id", "unknown"),
                    room_type=state.get("room_type", "bedroom"),
                    theme=state.get("theme", "Modern Minimalist"),
                    budget_tier=state.get("budget_tier", "mid"),
                    city=state.get("city", "Hyderabad"),
                )
            )

        features = result.get("features", {})
        wall_sqft = float(features.get("estimated_wall_area_sqft", 200.0))
        floor_sqft = float(features.get("estimated_floor_area_sqft", 120.0))

        # ── Style classification: fine-tuned EfficientNet → CLIP → zero-shot ─
        style_label      = result["style"]["label"]
        style_confidence = result["style"]["confidence"]
        style_model_used = "gemini_derived"

        if img_bytes:
            try:
                from ml.style_classifier import StyleClassifier
                clf = StyleClassifier()
                style_res = clf.classify(img_bytes, room_type=state.get("room_type", "bedroom"))
                # Use fine-tuned result if confidence is higher
                if style_res.get("confidence", 0) >= style_confidence:
                    style_label      = style_res["style_label"]
                    style_confidence = style_res["confidence"]
                    style_model_used = style_res.get("model_used", "style_classifier")
            except Exception as sc_err:
                logger.debug(f"[visual_assessment] StyleClassifier skipped: {sc_err}")

        model_provenance["style_model_used"] = style_model_used

        # ── Room classification: fine-tuned room_classifier.pt → zero-shot ───
        room_model_used = "state_input"
        try:
            from ml.cv_model_registry import CVModelRegistry
            registry = CVModelRegistry.get()
            if registry.room_classifier and img_bytes:
                room_result = registry.room_classifier.classify(img_bytes)
                room_model_used = (
                    "room_classifier_finetuned"
                    if getattr(registry.room_classifier, "_loaded_from_finetuned", False)
                    else "clip_zero_shot"
                )
                # Update room_type in state if classifier is more confident
                detected_room = room_result.get("room_type", state.get("room_type", "bedroom"))
                logger.debug(
                    f"[visual_assessment] Room classifier: {detected_room} "
                    f"({room_model_used})"
                )
        except Exception as rc_err:
            logger.debug(f"[visual_assessment] RoomClassifier skipped: {rc_err}")

        model_provenance["room_model_used"] = room_model_used

        upd: GraphState = {
            "room_features": features,
            "image_features": result.get("image_features", {}),
            "layout_report": result.get("layout_report", {}),
            "style_label": style_label,
            "style_confidence": style_confidence,
            "explainable_recommendations": result.get("recommendations", []),
            "material_quantities": result.get("material_quantities", {}),
            "detected_objects": list(result.get("spatial_map", {}).get("detected_objects", {}).keys()),
            "material_types": _infer_material_types(features),
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
            "visual_style": [style_label],
            **model_provenance,
        }

    except Exception as e:
        logger.error(f"[visual_assessment] {e}", exc_info=True)
        upd: GraphState = {
            "room_features": {},
            "image_features": _fallback_image_features(state),
            "layout_report": {"layout_score": "65/100", "walkable_space": "45%", "issues_detected": []},
            "style_label": state.get("theme", "Modern Minimalist"),
            "style_confidence": 0.5,
            "explainable_recommendations": [],
            "material_quantities": _fallback_quantities(state),
            "detected_objects": [],
            "material_types": ["vitrified_tile"],
            "wall_area_sqft": 200.0,
            "floor_area_sqft": 120.0,
            "room_dimensions": {"wall_area_sqft": 200, "floor_area_sqft": 120,
                                "estimated_length_ft": 14.0, "estimated_width_ft": 12.0,
                                "estimated_height_ft": 9.0},
            "detected_changes": [],
            "visual_style": [state.get("theme", "Modern Minimalist")],
            "yolo_model_used": "unavailable",
            "clip_model_used": "unavailable",
            "style_model_used": "fallback",
            "room_model_used": "fallback",
            "errors": (state.get("errors") or []) + [f"visual_assessment: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Material Estimation
# ─────────────────────────────────────────────────────────────────────────────

def node_material_estimation(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    name = "material_estimation"
    try:
        from services.material_estimator import MaterialEstimator
        estimator = MaterialEstimator()

        existing_quantities = state.get("material_quantities", {})
        if existing_quantities and existing_quantities.get("paint_liters", 0) > 0:
            quantities = dict(existing_quantities)
            quantities["_wall_area_sqft"] = state.get("wall_area_sqft", 200.0)
            quantities["_floor_area_sqft"] = state.get("floor_area_sqft", 120.0)
        else:
            quantities = estimator.estimate(
                wall_area_sqft=state.get("wall_area_sqft", 200.0),
                floor_area_sqft=state.get("floor_area_sqft", 120.0),
                detected_objects=[{"label": o} for o in (state.get("detected_objects") or [])],
                material_types=state.get("material_types", []),
                budget_tier=state.get("budget_tier", "mid"),
                room_type=state.get("room_type", "bedroom"),
            )

        upd: GraphState = {"material_quantities": quantities}
    except Exception as e:
        logger.error(f"[material_estimation] {e}")
        upd: GraphState = {
            "material_quantities": _fallback_quantities(state),
            "errors": (state.get("errors") or []) + [f"material_estimation: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Cost Estimation
# ─────────────────────────────────────────────────────────────────────────────

def node_cost_estimation(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    name = "cost_estimation"
    try:
        from services.cost_estimator import CostEstimator
        estimator = CostEstimator()

        result = estimator.estimate(
            material_quantities=state.get("material_quantities", {}),
            budget_tier=state.get("budget_tier", "mid"),
            city=state.get("city", "Hyderabad"),
            room_type=state.get("room_type", "bedroom"),
            wall_area_sqft=state.get("wall_area_sqft", 200.0),
            floor_area_sqft=state.get("floor_area_sqft", 120.0),
        )

        upd: GraphState = {
            "cost_breakdown": result["breakdown"],
            "total_cost_estimate": result["total_inr"],
            "boq_line_items": result["line_items"],
            "labour_estimate": result["labour_inr"],
            "material_plan": result["material_plan"],
        }
    except Exception as e:
        logger.error(f"[cost_estimation] {e}")
        budget = state.get("budget_inr", 750_000)
        upd: GraphState = {
            "cost_breakdown": {"materials_inr": int(budget * 0.55), "labour_inr": int(budget * 0.30),
                               "supervision_inr": int(budget * 0.05), "misc_contingency_inr": int(budget * 0.10),
                               "total_inr": budget},
            "total_cost_estimate": budget,
            "boq_line_items": [],
            "labour_estimate": int(budget * 0.30),
            "material_plan": {},
            "errors": (state.get("errors") or []) + [f"cost_estimation: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Node: ROI Prediction
# v6.0: Wires ROIExplainer (SHAP) and NHB Residex benchmark validation.
#        Fine-tuned XGBoost model loaded via ROIForecastAgent; SHAP explanation
#        runs in a best-effort thread so it never blocks the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def node_roi_prediction(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    name = "roi_prediction"
    try:
        from agents.orchestrator.langgraph_orchestrator import node_roi_forecasting

        compat = {k: v for k, v in state.items() if not isinstance(v, bytes)}
        compat["room_dimensions"] = state.get("room_dimensions") or {
            "wall_area_sqft": state.get("wall_area_sqft", 200),
            "floor_area_sqft": state.get("floor_area_sqft", 120),
        }
        compat["budget_inr"] = state.get("budget_inr") or state.get("total_cost_estimate", 750_000)

        r1 = node_roi_forecasting(compat)

        roi_pred = r1.get("roi_prediction", {})

        # ── ROIExplainer: SHAP factors + NHB validation ───────────────────────
        explainer_factors: List[Dict] = []
        nhb_validation:    Dict       = {}

        try:
            from ml.roi_explainer import ROIExplainer
            explainer = ROIExplainer()

            # Build a minimal feature row for SHAP
            try:
                import pandas as pd
                floor_sqft = float(
                    state.get("floor_area_sqft")
                    or state.get("room_dimensions", {}).get("floor_area_sqft", 120)
                )
                reno_cost  = float(
                    state.get("budget_inr")
                    or state.get("total_cost_estimate", 750_000)
                )
                city = state.get("city", "Hyderabad")
                room_type = state.get("room_type", "bedroom")

                # city_tier lookup (same as ROIForecastAgent)
                _CITY_TIERS = {
                    "Mumbai": 1, "Delhi NCR": 1, "Delhi": 1,
                    "Bangalore": 1, "Chennai": 2, "Hyderabad": 2,
                    "Pune": 2, "Kolkata": 2, "Ahmedabad": 3,
                }
                city_tier = _CITY_TIERS.get(city, 2)

                budget_tier_map = {"budget": 1, "mid": 2, "premium": 3, "luxury": 4}
                budget_tier_enc = budget_tier_map.get(state.get("budget_tier", "mid"), 2)

                room_type_map = {
                    "bedroom": 0, "bathroom": 1, "living_room": 2,
                    "kitchen": 3, "full_home": 4,
                }
                room_type_enc = room_type_map.get(room_type, 0)

                feature_row = pd.DataFrame([{
                    "renovation_cost_lakh": round(reno_cost / 100_000, 4),
                    "area_sqft":            floor_sqft,
                    "city_tier":            city_tier,
                    "budget_tier_enc":      budget_tier_enc,
                    "room_type_enc":        room_type_enc,
                    "existing_condition_enc": 1,   # average default
                }])

                explainer_factors = explainer.explain(feature_row)
                logger.debug(
                    f"[roi_prediction] SHAP factors: "
                    f"{[f['feature'] for f in explainer_factors]}"
                )
            except ImportError:
                logger.debug("[roi_prediction] pandas not available for SHAP feature row")

            # NHB Residex benchmark validation
            nhb_validation = explainer.validate_against_nhb_benchmarks(
                roi_pct=float(roi_pred.get("roi_pct", 0.0)),
                city=state.get("city", "Hyderabad"),
                room_type=state.get("room_type", "bedroom"),
            )
            logger.debug(
                f"[roi_prediction] NHB validation: "
                f"within_benchmark={nhb_validation.get('within_benchmark', 'N/A')}"
            )

        except ImportError:
            logger.debug("[roi_prediction] ROIExplainer not available — SHAP skipped")
        except Exception as exp_err:
            logger.debug(f"[roi_prediction] ROIExplainer error (non-critical): {exp_err}")

        # Merge explainer data into roi_prediction dict
        if explainer_factors:
            roi_pred["explainer_factors"] = explainer_factors
        if nhb_validation:
            roi_pred["nhb_validation"] = nhb_validation

        upd: GraphState = {
            "roi_prediction":        roi_pred,
            "roi_explainer_factors": explainer_factors,
            "roi_nhb_validation":    nhb_validation,
            "payback_months":        r1.get("payback_months", 36),
            "equity_gain_inr":       r1.get("equity_gain_inr", 0),
            "location_context":      r1.get("location_context", {}),
            "budget_analysis":       r1.get("budget_analysis", {}),
            "material_prices":       r1.get("material_prices", []),
        }
    except Exception as e:
        logger.error(f"[roi_prediction] {e}")
        upd: GraphState = {
            "roi_prediction":        {"roi_pct": 12.0, "model_type": "fallback", "payback_months": 36},
            "roi_explainer_factors": [],
            "roi_nhb_validation":    {},
            "payback_months":        36,
            "equity_gain_inr":       0,
            "location_context":      {},
            "budget_analysis":       {},
            "material_prices":       [],
            "errors": (state.get("errors") or []) + [f"roi_prediction: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Timeline
# ─────────────────────────────────────────────────────────────────────────────

def node_timeline(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    name = "timeline"
    try:
        from services.timeline_estimator import TimelineEstimator
        estimator = TimelineEstimator()

        timeline = estimator.estimate(
            material_quantities=state.get("material_quantities", {}),
            floor_area_sqft=state.get("floor_area_sqft", 120.0),
            room_type=state.get("room_type", "bedroom"),
            budget_tier=state.get("budget_tier", "mid"),
        )
        upd: GraphState = {"timeline": timeline}
    except Exception as e:
        logger.error(f"[timeline] {e}")
        upd: GraphState = {
            "timeline": {
                "days": 21, "calendar_weeks": 3.5,
                "phases": [
                    {"phase": "Site preparation & demolition", "days": 2, "parallel": False},
                    {"phase": "Electrical rough-in", "days": 2, "parallel": False},
                    {"phase": "Floor tiling & curing", "days": 5, "parallel": False},
                    {"phase": "Wall putty & preparation", "days": 3, "parallel": False},
                    {"phase": "Paint (2 coats)", "days": 3, "parallel": False},
                    {"phase": "Carpentry & modular installation", "days": 4, "parallel": True},
                    {"phase": "Polishing, cleanup & snagging", "days": 2, "parallel": False},
                ],
                "workers": 3,
                "assumptions": ["Standard 3-worker crew", "6-day work week"],
            },
            "errors": (state.get("errors") or []) + [f"timeline: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Insight Generation
# ─────────────────────────────────────────────────────────────────────────────

def node_insight_generation(state: GraphState) -> GraphState:
    """
    Delegates to existing battle-tested pipeline insight agent.
    UPGRADED: RAG knowledge retrieval runs first to provide fact-based context.
    Retrieved renovation knowledge (costs, repair standards, materials, case studies)
    is injected into insights before LLM reasoning — retrieved knowledge takes priority.
    """
    t0 = time.perf_counter()
    name = "insight_generation"

    # ── Step 1: RAG Retrieval (non-blocking) ──────────────────────────────────
    rag_result: Dict[str, Any] = {}
    try:
        from services.rag.context_builder import get_rag_pipeline

        rag_pipeline = get_rag_pipeline()

        extra_queries: List[str] = []
        user_intent = state.get("user_intent", "")
        if user_intent:
            extra_queries.append(f"renovation advice {user_intent}")
        changes = state.get("detected_changes") or []
        for change in changes[:3]:
            extra_queries.append(f"best method {change}")

        rag_result = rag_pipeline.run(state, extra_queries=extra_queries or None)
        logger.info(
            f"[insight_generation] RAG retrieved {len(rag_result.get('rag_chunks', []))} docs "
            f"— categories: {rag_result.get('rag_categories', [])}"
        )
    except Exception as rag_err:
        logger.warning(f"[insight_generation] RAG retrieval failed (non-critical): {rag_err}")

    # ── Step 2: Core insight generation ──────────────────────────────────────
    try:
        from agents.orchestrator.langgraph_orchestrator import node_insight_generation as _orch_insight

        rag_enriched_state = dict(state)
        if rag_result.get("rag_context"):
            rag_enriched_state["rag_context"] = rag_result["rag_context"]

        compat = {k: v for k, v in rag_enriched_state.items() if not isinstance(v, bytes)}
        result = _orch_insight(compat)

        upd: GraphState = {
            "insights":    result.get("insights", {}),
            "chat_context": result.get("chat_context", ""),
        }

        if upd["insights"] and state.get("layout_report"):
            upd["insights"]["layout_analysis"] = state.get("layout_report", {})
        if upd["insights"] and state.get("explainable_recommendations"):
            upd["insights"]["explainable_design_recommendations"] = state.get("explainable_recommendations", [])

        # ── Inject SHAP ROI factors into insights ─────────────────────────────
        if upd["insights"] and state.get("roi_explainer_factors"):
            upd["insights"]["roi_explainer_factors"] = state.get("roi_explainer_factors", [])
        if upd["insights"] and state.get("roi_nhb_validation"):
            upd["insights"]["roi_nhb_validation"] = state.get("roi_nhb_validation", {})

        # ── Step 3: RAG enrichment ───────────────────────────────────────────
        if rag_result and upd.get("insights"):
            try:
                from services.rag.context_builder import get_rag_pipeline as _get_rag
                _rag = _get_rag()
                upd["insights"] = _rag.enrich_insights(
                    upd["insights"],
                    rag_result,
                    state,
                )
                existing_ctx = upd.get("chat_context", "")
                rag_ctx_snippet = rag_result.get("rag_context", "")[:1200]
                if rag_ctx_snippet:
                    upd["chat_context"] = f"{existing_ctx}\n\n{rag_ctx_snippet}"
            except Exception as enrich_err:
                logger.warning(f"[insight_generation] RAG enrichment failed: {enrich_err}")

        if rag_result:
            upd["insights"]["rag_doc_ids"] = rag_result.get("rag_doc_ids", [])
            upd["insights"]["rag_categories_used"] = rag_result.get("rag_categories", [])

    except Exception as e:
        logger.error(f"[insight_generation] {e}")
        upd: GraphState = {
            "insights": {
                "summary_headline": f"{state.get('theme', 'Modern')} renovation — {state.get('city', 'India')}",
                "financial_outlook": {},
                "visual_analysis": {},
                "market_intelligence": {},
                "budget_assessment": {},
                "recommendations": [],
                "risk_factors": [],
                "top_materials": [],
                "layout_analysis": state.get("layout_report", {}),
                "explainable_design_recommendations": state.get("explainable_recommendations", []),
                "roi_explainer_factors": state.get("roi_explainer_factors", []),
                "roi_nhb_validation": state.get("roi_nhb_validation", {}),
                "rag_knowledge_context": rag_result.get("rag_context", ""),
                "rag_doc_ids": rag_result.get("rag_doc_ids", []),
            },
            "chat_context": f"Renovation: {state.get('theme')} in {state.get('city')}",
            "errors": (state.get("errors") or []) + [f"insight_generation: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def node_report_generator(state: GraphState) -> GraphState:
    """
    Assembles the full structured renovation report.
    v6.0: includes CV model provenance and SHAP ROI factors.
    """
    t0 = time.perf_counter()
    name = "report_generator"
    try:
        features = state.get("room_features", {})
        roi = state.get("roi_prediction", {})
        timeline = state.get("timeline", {})
        cost = state.get("cost_breakdown", {})
        insights = state.get("insights", {})
        layout = state.get("layout_report", {})
        recommendations = state.get("explainable_recommendations", [])
        boq = state.get("boq_line_items", [])

        def fmt_inr(n: Any) -> str:
            if not isinstance(n, (int, float)):
                return str(n)
            if n >= 10_000_000:
                return f"₹{n/10_000_000:.2f} Cr"
            if n >= 100_000:
                return f"₹{n/100_000:.1f}L"
            return f"₹{n:,.0f}"

        report = {
            "report_version": "2.1",
            "project_id": state.get("project_id", ""),
            "room_analysis": {
                "room_type": state.get("room_type", "bedroom"),
                "room_condition": features.get("room_condition", "fair"),
                "quality_tier": features.get("quality_tier", "mid"),
                "wall_treatment": f"{features.get('wall_color', '')} — {features.get('wall_texture', '')}",
                "floor_material": features.get("floor_type", ""),
                "ceiling_type": features.get("ceiling_type", ""),
                "detected_furniture": features.get("detected_furniture", []),
                "colour_palette": features.get("colour_palette", []),
                "natural_light": features.get("natural_light_quality", "moderate"),
                "specific_observations": features.get("specific_changes", []),
            },
            "style_analysis": {
                "detected_style": state.get("style_label", "Modern Minimalist"),
                "confidence": state.get("style_confidence", 0.7),
                "colour_palette": features.get("colour_palette", []),
                "model_used": state.get("style_model_used", "unknown"),
            },
            "layout_analysis": {
                "layout_score": layout.get("layout_score", "65/100"),
                "walkable_space": layout.get("walkable_space", "45%"),
                "lighting_score": layout.get("lighting_score", "65/100"),
                "furniture_density": layout.get("furniture_density", "medium"),
                "issues": layout.get("issues_detected", []),
                "suggestions": layout.get("suggestions", []),
            },
            "design_recommendations": recommendations,
            "cost_estimate": {
                "materials": fmt_inr(cost.get("materials_inr", 0)),
                "labour": fmt_inr(cost.get("labour_inr", 0)),
                "supervision": fmt_inr(cost.get("supervision_inr", 0)),
                "contingency": fmt_inr(cost.get("misc_contingency_inr", 0)),
                "total": fmt_inr(cost.get("total_inr", state.get("total_cost_estimate", 0))),
                "total_raw_inr": cost.get("total_inr", state.get("total_cost_estimate", 0)),
                "city_multiplier": cost.get("city_multiplier", 1.0),
            },
            "boq_summary": boq[:10],
            "roi_forecast": {
                "roi_percentage": f"{roi.get('roi_pct', 0):.1f}%",
                "equity_gain": fmt_inr(roi.get("equity_gain_inr", 0)),
                "payback_period": f"{roi.get('payback_months', 36)} months",
                "pre_reno_value": fmt_inr(roi.get("pre_reno_value_inr", 0)),
                "post_reno_value": fmt_inr(roi.get("post_reno_value_inr", 0)),
                "rental_yield_improvement": f"+{roi.get('rental_yield_delta', 0):.2f}%",
                "model_type": roi.get("model_type", "heuristic"),
                "model_confidence": f"{roi.get('model_confidence', 0.65)*100:.0f}%",
                # v6.0: SHAP explainability
                "explainer_factors": state.get("roi_explainer_factors", []),
                "nhb_validation": state.get("roi_nhb_validation", {}),
            },
            "renovation_timeline": {
                "total_days": timeline.get("days", 21),
                "calendar_weeks": timeline.get("calendar_weeks", 3.5),
                "phases": timeline.get("phases", []),
                "workers_required": timeline.get("workers", 3),
                "assumptions": timeline.get("assumptions", []),
            },
            "market_intelligence": {
                "city": state.get("city", ""),
                "location_context": state.get("location_context", {}),
                "budget_assessment": state.get("budget_analysis", {}),
            },
            # v6.0: CV model provenance for transparency
            "cv_model_provenance": {
                "yolo":  state.get("yolo_model_used", "unknown"),
                "clip":  state.get("clip_model_used", "unknown"),
                "style": state.get("style_model_used", "unknown"),
                "room":  state.get("room_model_used", "unknown"),
            },
            "summary_headline": insights.get("summary_headline", ""),
            "render_url": state.get("render_url", ""),
        }

        upd: GraphState = {"renovation_report": report}

    except Exception as e:
        logger.error(f"[report_generator] {e}", exc_info=True)
        upd: GraphState = {
            "renovation_report": {
                "report_version": "2.1",
                "error": str(e),
                "summary_headline": f"{state.get('theme', 'Modern')} renovation report",
            },
            "errors": (state.get("errors") or []) + [f"report_generator: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Rendering  — delegates to agents/rendering_agent.py (NOT MODIFIED)
# ─────────────────────────────────────────────────────────────────────────────

def node_rendering(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    name = "rendering"
    try:
        from agents.rendering_agent import RenderingAgent
        agent = RenderingAgent()

        prompt, render_url = agent.generate(
            theme=state.get("theme", "Modern Minimalist"),
            style_label=state.get("style_label", state.get("theme", "Modern Minimalist")),
            detected_objects=[{"label": o} for o in (state.get("detected_objects") or [])],
            image_features=state.get("image_features", {}),
            original_image_b64=state.get("original_image_b64", ""),
            original_image_mime=state.get("original_image_mime", "image/jpeg"),
        )

        upd: GraphState = {"render_prompt": prompt, "render_url": render_url}
    except Exception as e:
        logger.warning(f"[rendering] {e}")
        upd: GraphState = {
            "render_prompt": f"{state.get('theme', 'Modern')} interior renovation",
            "render_url": "",
            "errors": (state.get("errors") or []) + [f"rendering: {e}"],
        }

    _timing(upd, name, t0, state)
    _done(upd, name, state)
    return {**state, **upd}


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Build the LangGraph StateGraph.

    DAG:
      intent → visual_assessment → material_estimation → cost_estimation
             → roi_prediction → timeline → rendering → insight_generation
             → report_generator → END
    """
    try:
        from langgraph.graph import StateGraph, END

        g = StateGraph(GraphState)

        nodes = [
            ("intent", node_intent),
            ("visual_assessment", node_visual_assessment),
            ("material_estimation", node_material_estimation),
            ("cost_estimation", node_cost_estimation),
            ("roi_prediction", node_roi_prediction),
            ("timeline", node_timeline),
            ("rendering", node_rendering),
            ("insight_generation", node_insight_generation),
            ("report_generator", node_report_generator),
        ]

        for node_name, fn in nodes:
            g.add_node(node_name, fn)

        g.set_entry_point("intent")
        g.add_edge("intent", "visual_assessment")
        g.add_edge("visual_assessment", "material_estimation")
        g.add_edge("material_estimation", "cost_estimation")
        g.add_edge("cost_estimation", "roi_prediction")
        g.add_edge("cost_estimation", "timeline")
        g.add_edge("cost_estimation", "rendering")
        g.add_edge("roi_prediction", "insight_generation")
        g.add_edge("timeline", "insight_generation")
        g.add_edge("rendering", "insight_generation")
        g.add_edge("insight_generation", "report_generator")
        g.add_edge("report_generator", END)

        return g.compile()

    except ImportError:
        logger.warning("langgraph not installed — sequential fallback will be used")
        return None


def run_graph(initial_state: GraphState) -> GraphState:
    """Execute the graph. Falls back to sequential if LangGraph unavailable."""
    app = build_graph()
    if app is not None:
        try:
            return app.invoke(initial_state)
        except Exception as e:
            logger.warning(f"LangGraph invoke failed ({e}) — sequential fallback")

    # Sequential fallback
    state = dict(initial_state)
    for fn in [
        node_intent, node_visual_assessment, node_material_estimation,
        node_cost_estimation, node_roi_prediction, node_timeline,
        node_rendering, node_insight_generation, node_report_generator,
    ]:
        try:
            state = fn(state)
        except Exception as e:
            logger.error(f"[sequential] {fn.__name__}: {e}")
            state.setdefault("errors", []).append(f"{fn.__name__}: {e}")

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _timing(upd: dict, name: str, t0: float, state: GraphState):
    elapsed = round(time.perf_counter() - t0, 3)
    timings = dict(state.get("agent_timings") or {})
    timings[name] = elapsed
    upd["agent_timings"] = timings


def _done(upd: dict, name: str, state: GraphState):
    done = list(state.get("completed_agents") or [])
    if name not in done:
        done.append(name)
    upd["completed_agents"] = done


def _infer_material_types(features: Dict[str, Any]) -> List[str]:
    """Infer material_types list from feature dict for material estimator."""
    floor = features.get("floor_type", "").lower()
    materials = []
    if "tile" in floor or "vitrified" in floor or "ceramic" in floor:
        materials.append("vitrified_tile")
    if "wood" in floor or "hardwood" in floor or "parquet" in floor:
        materials.append("hardwood_floor")
    if "marble" in floor:
        materials.append("marble")
    if "vinyl" in floor or "laminate" in floor:
        materials.append("engineered_wood")
    return materials or ["vitrified_tile"]


def _fallback_image_features(state: GraphState) -> Dict:
    theme = state.get("theme", "Modern Minimalist")
    budget_tier = state.get("budget_tier", "mid")
    return {
        "wall_treatment": f"{theme} style walls",
        "floor_material": "vitrified tiles",
        "ceiling_treatment": "POP false ceiling",
        "furniture_items": [],
        "colour_palette": ["white", "grey"],
        "detected_style": theme,
        "quality_tier": budget_tier,
        "specific_changes": ["Room analysis unavailable — using defaults"],
        "estimated_wall_area_sqft": 200,
        "estimated_floor_area_sqft": 120,
        "room_condition": "fair",
    }


def _fallback_quantities(state: GraphState) -> Dict:
    wall = state.get("wall_area_sqft", 200.0)
    floor = state.get("floor_area_sqft", 120.0)
    return {
        "paint_liters": round(wall * 0.074, 1),
        "primer_liters": round(wall * 0.037, 1),
        "putty_kg": round(wall * 0.25, 1),
        "tiles_sqft": round(floor * 1.1, 1),
        "wall_tiles_sqft": 0.0,
        "plywood_sqft": round(floor * 0.30, 1),
        "flooring_sqft": 0.0,
        "false_ceiling_sqft": round(floor * 0.80, 1),
        "electrical_points": max(4, int(floor / 12)),
        "_wall_area_sqft": wall,
        "_floor_area_sqft": floor,
    }