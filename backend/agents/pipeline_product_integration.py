"""
ARKEN — Pipeline Product Suggestion Integration v2.0
======================================================
SAVE AS: backend/agents/pipeline_product_integration.py

v2.0 Changes (BUG 4 FIX):
  - Added inject_products_into_boq(pipeline_state) function.
    After ProductSuggesterAgent completes, detected products are injected
    as BOQ line items under category "Furniture & Fixtures" and the
    budget_estimate["total_cost_inr"] is recalculated to include them.
  - Added "products_in_boq" and "products_subtotal_inr" to state so
    analyze.py summary builder can surface them to the frontend.
  - ProductSuggesterAgent v2.0 guarantees a price_inr field on every
    item — this is consumed here for the BOQ injection.

All v1.0 behaviours preserved:
  - Async, never raises, returns state unchanged on any failure.
  - Reads renovated_image_b64 (rendered image) exclusively.
  - style_label resolved from multiple state key fallback chain.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BUG 4 FIX: inject_products_into_boq
# ─────────────────────────────────────────────────────────────────────────────

def inject_products_into_boq(pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge detected products from product_suggestions into boq_line_items
    and update budget_estimate["total_cost_inr"].

    Reads from state:
      - product_suggestions["items"]  — list of detected products (v2.0 schema)
      - boq_line_items                — existing BOQ list
      - budget_estimate               — existing budget dict

    Writes to state:
      - boq_line_items               — extended with Furniture & Fixtures items
      - budget_estimate              — total_cost_inr updated, products fields added
      - products_in_boq              — list of injected product items
      - products_subtotal_inr        — sum of injected product prices

    Never raises — all exceptions are caught and logged.
    """
    try:
        suggestions = pipeline_state.get("product_suggestions") or {}
        # ProductSuggesterAgent v2.0 uses "items" key; v1.0 used "shop_this_look"
        raw_items: List[Dict] = (
            suggestions.get("items") or
            suggestions.get("shop_this_look") or
            []
        )

        if not raw_items:
            logger.info("[PipelineProducts] No product items to inject into BOQ")
            pipeline_state.setdefault("products_in_boq", [])
            pipeline_state.setdefault("products_subtotal_inr", 0)
            return pipeline_state

        # ── Build BOQ line items for each detected product ────────────────────
        injected_items: List[Dict] = []
        products_subtotal = 0

        for product in raw_items:
            price_inr = int(product.get("price_inr", 0))
            if price_inr <= 0:
                # Try price_range_inr["mid"] as fallback
                pr = product.get("price_range_inr", {})
                price_inr = int(pr.get("mid", pr.get("low", 0)))

            if price_inr <= 0:
                # No valid price — skip this item rather than inject ₹0
                logger.debug(
                    f"[PipelineProducts] Skipping '{product.get('item_name')}' — no price_inr"
                )
                continue

            boq_item = {
                "category":    "Furniture & Fixtures",
                "brand":       product.get("brand", "Recommended"),
                "product":     product.get("name") or product.get("item_name", "Item"),
                "sku":         product.get("sku", "N/A"),
                "qty":         1,
                "unit":        "unit",
                "rate_inr":    price_inr,
                "total_inr":   price_inr,
                "priority":    "nice_to_have",
                "source":      "shop_this_look_detection",
                "confidence":  product.get("confidence", 1.0),
                "tier_applied": pipeline_state.get("budget_tier", "mid"),
                "note": (
                    f"Detected in renovated image — {product.get('style_match_note', '')} "
                    f"(confidence={product.get('confidence', 'N/A')})"
                ),
            }

            injected_items.append(boq_item)
            products_subtotal += price_inr

        if not injected_items:
            logger.info("[PipelineProducts] No products had valid prices — nothing injected into BOQ")
            pipeline_state.setdefault("products_in_boq", [])
            pipeline_state.setdefault("products_subtotal_inr", 0)
            return pipeline_state

        # ── Extend boq_line_items ─────────────────────────────────────────────
        existing_boq = list(pipeline_state.get("boq_line_items") or [])
        # Avoid duplicating if already injected (idempotent)
        already_injected = any(
            item.get("source") == "shop_this_look_detection"
            for item in existing_boq
        )
        if not already_injected:
            existing_boq.extend(injected_items)
            pipeline_state["boq_line_items"] = existing_boq
        else:
            logger.info("[PipelineProducts] Products already injected into BOQ — skipping duplicate")
            products_subtotal = sum(
                item.get("total_inr", 0) for item in existing_boq
                if item.get("source") == "shop_this_look_detection"
            )

        # ── Update budget_estimate total ──────────────────────────────────────
        budget_estimate = dict(pipeline_state.get("budget_estimate") or {})
        existing_total = int(budget_estimate.get("total_cost_inr", 0))

        if not already_injected and products_subtotal > 0:
            budget_estimate["total_cost_inr"]          = existing_total + products_subtotal
            budget_estimate["products_included_in_boq"]  = True
            budget_estimate["products_subtotal_inr"]     = products_subtotal
            budget_estimate["renovation_cost_inr"]       = existing_total  # original without products
            pipeline_state["budget_estimate"] = budget_estimate

        # ── Write convenience state keys for analyze.py summary ──────────────
        pipeline_state["products_in_boq"]        = injected_items
        pipeline_state["products_subtotal_inr"]  = products_subtotal

        logger.info(
            f"[PipelineProducts] Injected {len(injected_items)} products into BOQ — "
            f"subtotal=₹{products_subtotal:,} | new total=₹{budget_estimate.get('total_cost_inr', 0):,}"
        )

    except Exception as e:
        logger.error(f"[PipelineProducts] inject_products_into_boq failed: {e}", exc_info=True)
        pipeline_state.setdefault("products_in_boq", [])
        pipeline_state.setdefault("products_subtotal_inr", 0)

    return pipeline_state


# ─────────────────────────────────────────────────────────────────────────────
# enrich_with_product_suggestions (entry point called from analyze.py)
# ─────────────────────────────────────────────────────────────────────────────

async def enrich_with_product_suggestions(
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Post-pipeline step: detect furniture in the rendered image and
    inject shop_this_look links + BOQ line items into the final state.

    BUG 4 FIX: After ProductSuggesterAgent runs, inject_products_into_boq()
    is called to merge detected products into boq_line_items and update the
    total_cost_inr in budget_estimate.

    Reads from state:
      - renovated_image_b64   (set by rendering step — RENDERED image, not original)
      - room_type, style_label (or detected_style), budget_tier, city
      - explainable_recommendations / design_plan (for fallback)

    Writes to state:
      - product_suggestions   (shop_this_look payload)
      - boq_line_items        (extended with Furniture & Fixtures)
      - budget_estimate       (total updated to include products)
      - products_in_boq       (list of injected items)
      - products_subtotal_inr (sum of product prices)

    Never raises — all exceptions caught and logged.
    """
    try:
        from agents.product_suggester_agent import ProductSuggesterAgent

        # ── Extract RENDERED (renovated) image ────────────────────────────────
        # ADDITIONAL FIX: explicitly use renovated_image_b64, never original_image_b64
        image_b64   = pipeline_state.get("renovated_image_b64", "")
        image_bytes = None
        mime_type   = pipeline_state.get("renovated_image_mime") or "image/png"

        if image_b64:
            import base64
            try:
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                image_bytes = base64.b64decode(image_b64)
                logger.info(
                    f"[PipelineProducts] Using renovated_image_b64 ({len(image_bytes)} bytes) "
                    f"for product detection"
                )
            except (ValueError, Exception) as e:
                logger.warning(f"[PipelineProducts] base64 decode failed: {e}")
        else:
            logger.info(
                "[PipelineProducts] renovated_image_b64 not in state — "
                "product detection will use design_recommendations fallback"
            )

        # ── Resolve metadata ──────────────────────────────────────────────────
        room_type   = str(pipeline_state.get("room_type", "bedroom"))
        budget_tier = str(pipeline_state.get("budget_tier", "mid"))
        city        = str(pipeline_state.get("city", "Hyderabad"))

        style_label = (
            pipeline_state.get("style_label") or
            pipeline_state.get("detected_style") or
            (pipeline_state.get("vision_features") or {}).get("detected_style") or
            (pipeline_state.get("image_features") or {}).get("detected_style") or
            pipeline_state.get("theme") or
            "Modern Minimalist"
        )

        # Design recommendations for fallback
        _design_plan = pipeline_state.get("design_plan") or {}
        _dp_recs = _design_plan.get("recommendations") if isinstance(_design_plan, dict) else None
        if isinstance(_dp_recs, dict):
            _dp_recs = None
        design_recommendations = (
            pipeline_state.get("explainable_recommendations") or
            pipeline_state.get("recommendations") or
            (_dp_recs if isinstance(_dp_recs, list) else None) or
            []
        )

        # ── Call ProductSuggesterAgent ────────────────────────────────────────
        agent  = ProductSuggesterAgent()
        result = agent.suggest(
            rendered_image_bytes=image_bytes,
            room_type=room_type,
            style_label=str(style_label),
            budget_tier=budget_tier,
            city=city,
            design_recommendations=design_recommendations,
            mime_type=mime_type,
        )

        pipeline_state["product_suggestions"] = result

        logger.info(
            f"[PipelineProducts] ProductSuggesterAgent done — "
            f"{result.get('items_detected', 0)} items detected via "
            f"{result.get('detection_source', 'unknown')} | "
            f"room={room_type} style={style_label} budget={budget_tier}"
        )

        # ── BUG 4 FIX: Inject into BOQ and update total ───────────────────────
        pipeline_state = inject_products_into_boq(pipeline_state)

    except ImportError as e:
        logger.warning(f"[PipelineProducts] ProductSuggesterAgent not available: {e}")
        pipeline_state.setdefault("product_suggestions", None)
        pipeline_state.setdefault("products_in_boq", [])
        pipeline_state.setdefault("products_subtotal_inr", 0)
    except Exception as e:
        logger.error(f"[PipelineProducts] enrich_with_product_suggestions failed: {e}", exc_info=True)
        pipeline_state.setdefault("product_suggestions", None)
        pipeline_state.setdefault("products_in_boq", [])
        pipeline_state.setdefault("products_subtotal_inr", 0)

    return pipeline_state
