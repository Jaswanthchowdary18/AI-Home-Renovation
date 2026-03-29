"""
ARKEN — Structured Image Feature Schema v1.0
=============================================
Defines the canonical schema for all visual features extracted from room images.
Every downstream agent MUST receive data in this schema — never raw Gemini dicts.

This is the single source of truth for:
  - What the visual assessor extracts
  - What each downstream agent expects
  - What gets stored in DB / vector store
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Feature Schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoomFeatures:
    """
    All visual features extracted from a single room image.
    All fields have safe defaults so downstream agents never see None.
    """
    # ── Room identity ────────────────────────────────────────────────────────
    room_type: str = "bedroom"
    quality_tier: str = "mid"               # budget / mid / premium (from visual)
    room_condition: str = "fair"            # new / good / fair / poor

    # ── Surfaces ─────────────────────────────────────────────────────────────
    wall_color: str = "off-white"
    wall_texture: str = "smooth paint"
    floor_type: str = "vitrified tiles"
    floor_pattern: str = "plain"
    ceiling_type: str = "plain POP"
    ceiling_color: str = "white"

    # ── Furniture ────────────────────────────────────────────────────────────
    detected_furniture: List[str] = field(default_factory=list)
    furniture_positions: Dict[str, str] = field(default_factory=dict)
    # e.g. {"sofa": "center-left", "tv_unit": "north wall"}

    # ── Lighting ─────────────────────────────────────────────────────────────
    lighting_type: str = "overhead tube light"
    lighting_sources: List[str] = field(default_factory=list)
    natural_light_quality: str = "moderate"  # good / moderate / poor

    # ── Space ────────────────────────────────────────────────────────────────
    estimated_wall_area_sqft: float = 200.0
    estimated_floor_area_sqft: float = 120.0
    estimated_length_ft: float = 14.0
    estimated_width_ft: float = 12.0
    estimated_height_ft: float = 9.0
    free_space_percentage: float = 40.0     # pct of floor not covered by furniture
    room_area_estimate: str = "medium"      # small / medium / large

    # ── Style ────────────────────────────────────────────────────────────────
    colour_palette: List[str] = field(default_factory=lambda: ["white", "grey"])
    detected_style: str = "Modern Minimalist"
    style_confidence: float = 0.70

    # ── Change log ───────────────────────────────────────────────────────────
    specific_changes: List[str] = field(default_factory=list)
    renovation_potential: List[str] = field(default_factory=list)
    # e.g. ["replace flooring", "add false ceiling", "upgrade lighting"]

    # ── Layout analysis ──────────────────────────────────────────────────────
    layout_score: int = 70                  # 0-100
    walkable_space_pct: float = 55.0
    layout_issues: List[str] = field(default_factory=list)
    layout_suggestions: List[str] = field(default_factory=list)

    # ── Physics / spatial ────────────────────────────────────────────────────
    wall_coverage_pct: float = 60.0        # % of wall area occupied
    lighting_score: int = 65               # 0-100
    furniture_density: str = "medium"      # sparse / medium / dense

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoomFeatures":
        """Safe deserialization — unknown keys are silently ignored."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_gemini_response(cls, raw: Dict[str, Any], room_type: str = "bedroom") -> "RoomFeatures":
        """
        Map a raw Gemini JSON response to the canonical schema.
        Handles variant key names from different prompt versions.
        """
        def _get(*keys: str, default: Any = "") -> Any:
            for k in keys:
                v = raw.get(k)
                if v is not None:
                    return v
            return default

        # Parse furniture list (Gemini returns various formats)
        furn_raw = _get("furniture_items", "detected_furniture", default=[])
        if isinstance(furn_raw, str):
            furn_list = [f.strip() for f in furn_raw.split(",") if f.strip()]
        elif isinstance(furn_raw, list):
            furn_list = [str(f) for f in furn_raw]
        else:
            furn_list = []

        # Parse colour palette
        palette_raw = _get("colour_palette", "color_palette", default=["white", "grey"])
        if isinstance(palette_raw, str):
            palette = [c.strip() for c in palette_raw.split(",")]
        elif isinstance(palette_raw, list):
            palette = [str(c) for c in palette_raw]
        else:
            palette = ["white", "grey"]

        # Derive free space from furniture density
        furn_density_map = {"sparse": 65.0, "medium": 45.0, "dense": 25.0}
        density_hint = _get("furniture_density", "furniture_placement", default="medium")
        if isinstance(density_hint, str):
            density_str = "dense" if "dense" in density_hint.lower() else \
                          "sparse" if "sparse" in density_hint.lower() else "medium"
        else:
            density_str = "medium"
        free_space = furn_density_map.get(density_str, 45.0)

        # Layout score from raw or derived
        layout_score_raw = _get("layout_score", "space_efficiency_score", default=None)
        if isinstance(layout_score_raw, (int, float)):
            layout_score = int(min(100, max(0, layout_score_raw)))
        else:
            # Derive from condition + density
            condition = str(_get("room_condition", default="fair")).lower()
            base = {"new": 80, "good": 72, "fair": 60, "poor": 45}.get(condition, 60)
            density_penalty = {"sparse": 0, "medium": -5, "dense": -15}.get(density_str, 0)
            layout_score = max(0, min(100, base + density_penalty))

        # Issues from Gemini or derived
        issues_raw = _get("layout_issues", "space_issues", default=[])
        if isinstance(issues_raw, list):
            issues = [str(i) for i in issues_raw]
        elif isinstance(issues_raw, str) and issues_raw:
            issues = [issues_raw]
        else:
            issues = []

        # Renovation potential
        changes = _get("specific_changes", "renovation_suggestions", default=[])
        if isinstance(changes, list):
            changes = [str(c) for c in changes]
        elif isinstance(changes, str):
            changes = [changes]

        return cls(
            room_type=room_type,
            quality_tier=str(_get("quality_tier", default="mid")).lower(),
            room_condition=str(_get("room_condition", default="fair")).lower(),
            wall_color=str(_get("wall_colour", "wall_color", "wall_treatment", default="off-white")),
            wall_texture=str(_get("wall_texture", "wall_finish", default="smooth paint")),
            floor_type=str(_get("floor_material", "floor_type", default="vitrified tiles")),
            floor_pattern=str(_get("floor_pattern", default="plain")),
            ceiling_type=str(_get("ceiling_treatment", "ceiling_type", default="plain POP")),
            ceiling_color=str(_get("ceiling_colour", "ceiling_color", default="white")),
            detected_furniture=furn_list,
            furniture_positions={},
            lighting_type=str(_get("lighting_type", "light_sources", default="overhead")),
            lighting_sources=list(_get("lighting_sources", default=[])) if isinstance(_get("lighting_sources", default=[]), list) else [],
            natural_light_quality=str(_get("natural_light", "natural_light_quality", default="moderate")),
            estimated_wall_area_sqft=float(_get("estimated_wall_area_sqft", default=200.0)),
            estimated_floor_area_sqft=float(_get("estimated_floor_area_sqft", default=120.0)),
            estimated_length_ft=float(_get("estimated_length_ft", default=14.0)),
            estimated_width_ft=float(_get("estimated_width_ft", default=12.0)),
            estimated_height_ft=float(_get("estimated_height_ft", default=9.0)),
            free_space_percentage=free_space,
            room_area_estimate=str(_get("room_size", "room_area_estimate", default="medium")),
            colour_palette=palette,
            detected_style=str(_get("detected_style", "interior_style", default="Modern Minimalist")),
            style_confidence=float(_get("style_confidence", default=0.70)),
            specific_changes=changes,
            renovation_potential=list(_get("renovation_potential", default=[])) if isinstance(_get("renovation_potential", default=[]), list) else [],
            layout_score=layout_score,
            walkable_space_pct=free_space,
            layout_issues=issues,
            layout_suggestions=list(_get("layout_suggestions", default=[])) if isinstance(_get("layout_suggestions", default=[]), list) else [],
            wall_coverage_pct=float(_get("wall_coverage_pct", default=60.0)),
            lighting_score=int(_get("lighting_score", default=65)),
            furniture_density=density_str,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Gemini Vision Prompt — structured extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_extraction_prompt(room_type: str, theme: str, budget_tier: str, city: str) -> str:
    # Budget-tier-specific material guidance injected into prompt
    TIER_MATERIAL_CONTEXT = {
        "basic": (
            "Budget tier (₹2–5L): Focus on paint refresh, basic vitrified tiles (Kajaria/Somany GVT 600×600 ~₹60/sqft), "
            "standard electrical switches (Havells/Crabtree), and minor carpentry. "
            "Do NOT specify premium stone, Italian tiles, or imported hardware — these are out of scope at this budget."
        ),
        "mid": (
            "Mid tier (₹5–12L): Specify premium emulsion paint (Asian Paints Royale Sheen or Dulux Velvet Touch), "
            "large-format vitrified tiles (Kajaria Endura or Simpolo GVT 800×800 ~₹90–120/sqft), "
            "18mm BWR plywood modular carpentry (Greenply/Century), false ceiling for bedroom/living, "
            "and modular electrical (Legrand Myrius or Schneider). "
            "This is the maximum-ROI sweet spot for Indian residential renovation."
        ),
        "premium": (
            "Premium tier (₹12L+): Specify premium stone flooring (marble/granite/engineered stone), "
            "Italian-finish or acrylic-shutter carpentry, smart home integration (lighting control, automated blinds), "
            "imported CP fittings (Grohe/Kohler) where applicable, and designer lighting (Philips Hue or equivalent). "
            "All materials should be top-tier Indian market or imported equivalents."
        ),
    }
    tier_context = TIER_MATERIAL_CONTEXT.get(budget_tier, TIER_MATERIAL_CONTEXT["mid"])

    return f"""Analyse this {room_type} image in detail for an Indian renovation project. Provide a thorough, specific visual assessment.

Context:
- Room: {room_type} in {city}
- Design theme: {theme}
- Budget tier: {budget_tier} — {tier_context}

Respond ONLY with a JSON object containing EXACTLY these keys (no extra keys, no markdown):
{{
  "room_type": "{room_type}",
  "quality_tier": "budget|mid|premium — infer from materials/finishes visible",
  "room_condition": "new|good|fair|poor — based on wear, cleanliness, modernity",
  "wall_colour": "specific colour name visible (e.g. ivory white, warm grey)",
  "wall_texture": "type of finish (e.g. matte emulsion, textured plaster, wallpaper)",
  "floor_material": "exact material (e.g. vitrified tiles 600×600, hardwood, marble, carpet)",
  "floor_pattern": "pattern if any (e.g. 600x600 plain, herringbone, checkerboard)",
  "ceiling_type": "type (e.g. plain POP, gypsum false ceiling, exposed concrete)",
  "ceiling_colour": "colour of ceiling",
  "furniture_items": ["list every visible piece of furniture by name and approximate size"],
  "furniture_density": "sparse|medium|dense — how crowded is the room",
  "lighting_type": "primary lighting (e.g. tube light, LED downlights, pendant, natural)",
  "natural_light_quality": "good|moderate|poor — based on windows/brightness",
  "colour_palette": ["primary colour", "secondary colour", "accent colour"],
  "detected_style": "primary interior design style detected (one of: Modern Minimalist, Scandinavian, Traditional Indian, Industrial, Japandi, Bohemian, Art Deco, Contemporary)",
  "style_confidence": 0.75,
  "specific_changes": [
    "at least 5 specific visual observations about the current state",
    "e.g. Walls have old emulsion paint with visible cracks near skirting board",
    "e.g. Floor has 400x400 ceramic tiles with grout discolouration between joints",
    "e.g. Ceiling is plain white with single tube light fixture, no cove",
    "e.g. One aluminium-frame window visible on north wall, no curtain track",
    "e.g. Sofa placed against south wall leaving narrow 60cm passage to door"
  ],
  "renovation_potential": [
    "specific improvements visible as opportunities, matched to {budget_tier} budget",
    "e.g. Replace floor tiles with 800x800 GVT for visual space expansion (mid-tier appropriate)",
    "e.g. Add false ceiling with cove lighting to modernise — highest ROI upgrade for this room",
    "e.g. Repaint walls in trending warm tone — low cost, high visual impact"
  ],
  "layout_issues": [
    "any space planning issues you can actually see",
    "e.g. Furniture blocking natural movement path to window",
    "e.g. Corner space completely underutilised — no storage or accent element",
    "e.g. No dedicated study or work nook despite room size allowing it"
  ],
  "layout_suggestions": [
    "specific actionable layout improvements for this room",
    "e.g. Move bed 30cm forward to open passage to door — no cost change",
    "e.g. Add floor-to-ceiling wardrobe on east wall to utilise dead 8ft space"
  ],
  "estimated_wall_area_sqft": 200,
  "estimated_floor_area_sqft": 120,
  "recommended_materials": [
    "Specific Indian brand + product suited to {budget_tier} budget for this room",
    "e.g. Asian Paints Royale Sheen in Warm Ivory for walls",
    "e.g. Kajaria Endura 800x800 GVT in light grey for floor"
  ]
}}

IMPORTANT:
- Only describe what you ACTUALLY SEE — no guesses or hallucinations
- Be specific about colours (e.g. "warm ivory" not just "white")
- List every visible furniture item individually with approximate size
- material recommendations must match the {budget_tier} budget tier specified above
- Return ONLY the JSON — no markdown fences, no explanation text
"""


# ─────────────────────────────────────────────────────────────────────────────
# Layout Analyser — physics-aware spatial analysis
# ─────────────────────────────────────────────────────────────────────────────

class LayoutAnalyser:
    """
    Physics-aware layout analysis from extracted room features.
    Calculates walkable space, furniture collision, lighting estimation, layout score.
    """

    # Approximate floor footprint per furniture type (sqft)
    FURNITURE_FOOTPRINT: Dict[str, float] = {
        "sofa": 28.0, "couch": 28.0, "sectional sofa": 45.0,
        "bed": 35.0, "double bed": 42.0, "king bed": 50.0, "single bed": 24.0,
        "wardrobe": 20.0, "cabinet": 10.0, "bookshelf": 8.0,
        "dining table": 24.0, "study table": 15.0, "desk": 12.0,
        "coffee table": 8.0, "side table": 4.0, "tv unit": 12.0,
        "chair": 6.0, "dining chair": 4.0, "armchair": 10.0,
        "refrigerator": 6.0, "washing machine": 5.0,
    }

    # Minimum walkable corridor width (ft)
    MIN_CORRIDOR_FT = 3.0
    MIN_WALKABLE_PCT = 35.0

    def analyse(self, features: RoomFeatures) -> RoomFeatures:
        """Run full layout analysis and update features in place."""
        floor_sqft = features.estimated_floor_area_sqft

        # Calculate furniture footprint
        total_furniture_sqft = 0.0
        collision_issues = []

        for item in features.detected_furniture:
            item_lower = item.lower()
            footprint = 0.0
            for key, fp in self.FURNITURE_FOOTPRINT.items():
                if key in item_lower:
                    footprint = fp
                    break
            total_furniture_sqft += footprint

        # Walkable space
        walkable_sqft = max(0.0, floor_sqft - total_furniture_sqft)
        walkable_pct = round((walkable_sqft / max(floor_sqft, 1)) * 100, 1)
        features.walkable_space_pct = walkable_pct
        features.free_space_percentage = walkable_pct

        # Collision / overcrowding detection
        if total_furniture_sqft > floor_sqft * 0.65:
            collision_issues.append("Furniture density too high — movement paths severely restricted")
        elif total_furniture_sqft > floor_sqft * 0.50:
            collision_issues.append("High furniture density — some movement paths may be restricted")

        # Furniture-specific collision hints
        has_sofa = any("sofa" in f.lower() or "couch" in f.lower() for f in features.detected_furniture)
        has_bed = any("bed" in f.lower() for f in features.detected_furniture)
        has_dining = any("dining" in f.lower() for f in features.detected_furniture)

        if has_sofa and has_dining and floor_sqft < 180:
            collision_issues.append("Sofa and dining furniture in small space — zone separation needed")
        if has_bed and floor_sqft < 100:
            collision_issues.append("Bed may restrict access to wardrobe or bathroom — consider space-saving bed")

        # Existing layout issues from vision
        all_issues = list(set(features.layout_issues + collision_issues))
        features.layout_issues = all_issues

        # Layout score computation
        base_score = 70
        # Walkable space bonus/penalty
        if walkable_pct >= 55:
            base_score += 10
        elif walkable_pct < 35:
            base_score -= 20
        elif walkable_pct < 45:
            base_score -= 10

        # Lighting score
        lighting_bonus = {
            "good": 10, "moderate": 0, "poor": -15
        }.get(features.natural_light_quality, 0)
        base_score += lighting_bonus

        # Issue penalties
        base_score -= min(30, len(all_issues) * 5)

        # Condition bonus
        condition_bonus = {"new": 10, "good": 5, "fair": 0, "poor": -10}.get(features.room_condition, 0)
        base_score += condition_bonus

        features.layout_score = max(10, min(100, base_score))

        # Lighting score
        lighting_type_lower = features.lighting_type.lower()
        if "led" in lighting_type_lower or "downlight" in lighting_type_lower:
            features.lighting_score = 80
        elif "tube" in lighting_type_lower or "fluorescent" in lighting_type_lower:
            features.lighting_score = 50
        elif "natural" in lighting_type_lower or features.natural_light_quality == "good":
            features.lighting_score = 75
        else:
            features.lighting_score = 60

        # Wall coverage
        furn_density_map = {"sparse": 30.0, "medium": 50.0, "dense": 75.0}
        features.wall_coverage_pct = furn_density_map.get(features.furniture_density, 50.0)

        return features

    def format_report(self, features: RoomFeatures) -> Dict[str, Any]:
        """Return a human-readable layout report dict."""
        return {
            "layout_score": f"{features.layout_score}/100",
            "walkable_space": f"{features.walkable_space_pct:.0f}%",
            "lighting_score": f"{features.lighting_score}/100",
            "wall_coverage": f"{features.wall_coverage_pct:.0f}%",
            "furniture_density": features.furniture_density,
            "issues_detected": features.layout_issues,
            "suggestions": features.layout_suggestions,
            "natural_light": features.natural_light_quality,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Style Detector — rule-based + embedding
# ─────────────────────────────────────────────────────────────────────────────

class StyleDetector:
    """
    Detects interior design style from extracted features.
    Uses rule-based matching on colour palette + materials + furniture.
    Returns style label + confidence score.
    """

    # Style signatures: (colour keywords, material keywords, furniture keywords)
    STYLE_RULES: Dict[str, Dict] = {
        "Scandinavian": {
            "colours": ["white", "beige", "light grey", "natural", "cream", "birch"],
            "materials": ["wood", "oak", "pine", "linen", "wool", "cotton", "rattan"],
            "furniture": ["simple", "functional", "low profile"],
            "base_score": 0,
        },
        "Modern Minimalist": {
            "colours": ["white", "grey", "black", "charcoal", "monochrome"],
            "materials": ["concrete", "glass", "steel", "matte", "lacquer", "acrylic"],
            "furniture": ["sleek", "hidden storage", "built-in", "clean lines"],
            "base_score": 0,
        },
        "Industrial": {
            "colours": ["dark grey", "charcoal", "rust", "raw", "gunmetal", "black"],
            "materials": ["concrete", "brick", "metal", "exposed", "steel", "iron"],
            "furniture": ["pipe", "raw wood", "vintage", "factory"],
            "base_score": 0,
        },
        "Traditional Indian": {
            "colours": ["terracotta", "ochre", "saffron", "deep red", "gold", "maroon"],
            "materials": ["teak", "jali", "brass", "marble", "silk", "carved wood"],
            "furniture": ["carved", "ornate", "brass", "jharokha", "jali"],
            "base_score": 0,
        },
        "Japandi": {
            "colours": ["warm grey", "sage", "muted", "earthy", "cream", "brown"],
            "materials": ["bamboo", "paper", "wabi", "natural", "unfinished", "matte"],
            "furniture": ["low", "minimal", "functional", "natural"],
            "base_score": 0,
        },
        "Bohemian": {
            "colours": ["terracotta", "jewel", "teal", "mustard", "burgundy", "vibrant"],
            "materials": ["macramé", "rattan", "jute", "kilim", "eclectic", "layered"],
            "furniture": ["eclectic", "vintage", "layered", "mix"],
            "base_score": 0,
        },
        "Art Deco": {
            "colours": ["gold", "black", "emerald", "navy", "deep teal", "cream"],
            "materials": ["velvet", "marble", "brass", "geometric", "mirrored", "lacquer"],
            "furniture": ["geometric", "bold", "ornate", "symmetrical"],
            "base_score": 0,
        },
        "Contemporary": {
            "colours": ["warm white", "beige", "taupe", "navy", "sage"],
            "materials": ["mixed", "fabric", "wood", "metal"],
            "furniture": ["comfortable", "updated", "transitional"],
            "base_score": 5,  # Default fallback boost
        },
    }

    def detect(self, features: RoomFeatures) -> tuple[str, float]:
        """
        Returns (style_label, confidence_score 0..1).
        If Gemini already detected a style with high confidence, use it.
        Otherwise run rule-based detection.
        """
        # Trust Gemini if confidence is reasonable
        if features.style_confidence >= 0.65 and features.detected_style:
            return features.detected_style, features.style_confidence

        scores = {style: rules["base_score"] for style, rules in self.STYLE_RULES.items()}

        # Score from colours
        palette_text = " ".join(features.colour_palette).lower()
        wall_text = (features.wall_color + " " + features.wall_texture).lower()
        floor_text = features.floor_type.lower()
        furniture_text = " ".join(features.detected_furniture).lower()

        full_text = f"{palette_text} {wall_text} {floor_text} {furniture_text}"

        for style, rules in self.STYLE_RULES.items():
            for c in rules["colours"]:
                if c in full_text:
                    scores[style] += 2
            for m in rules["materials"]:
                if m in full_text:
                    scores[style] += 3
            for f_hint in rules["furniture"]:
                if f_hint in full_text:
                    scores[style] += 1

        best_style = max(scores, key=lambda s: scores[s])
        total_score = sum(scores.values())
        confidence = min(0.95, scores[best_style] / max(total_score, 1) + 0.3)

        return best_style, round(confidence, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Explainable Recommendation Generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DesignRecommendation:
    """A single explainable design recommendation linked to visual features."""
    title: str
    category: str                       # flooring / wall / ceiling / lighting / furniture / layout
    reasoning: List[str]               # why this is recommended (linked to features)
    estimated_cost_inr: str
    roi_impact_pct: str
    priority: str                       # high / medium / low
    specific_product: Optional[str] = None
    sku: Optional[str] = None


def generate_explainable_recommendations(
    features: RoomFeatures,
    budget_tier: str,
    city: str,
    detected_style: str,
    roi_pct: float,
) -> List[Dict[str, Any]]:
    """
    Generate design recommendations with explicit reasoning tied to visual features.
    Every recommendation explains WHY based on what was seen in the image.
    """
    recs: List[DesignRecommendation] = []
    tier = budget_tier.lower()

    # ── 1. Flooring recommendation ────────────────────────────────────────
    floor_lower = features.floor_type.lower()
    if "ceramic" in floor_lower or "old" in floor_lower or features.quality_tier == "budget":
        if tier == "basic":
            recs.append(DesignRecommendation(
                title="Upgrade to Glazed Vitrified Tiles (600x600)",
                category="flooring",
                reasoning=[
                    f"Current floor detected as '{features.floor_type}' — lower grade visible",
                    "600x600 GVT reflects more light, making room appear larger",
                    f"Aligns with {detected_style} aesthetic requirements",
                    "Kajaria GVT provides 15+ year durability with low maintenance",
                ],
                estimated_cost_inr="₹45-65/sqft (materials) + ₹30-35/sqft labour",
                roi_impact_pct=f"+{min(8, roi_pct * 0.5):.0f}–{min(12, roi_pct * 0.7):.0f}%",
                priority="high",
                specific_product="Kajaria Glazed Vitrified 600x600",
                sku="KJR-GVT-6060-BG",
            ))
        elif tier == "mid":
            recs.append(DesignRecommendation(
                title="Install Premium Vitrified Tiles (800x800)",
                category="flooring",
                reasoning=[
                    f"Current floor '{features.floor_type}' is below mid-tier standard",
                    f"Large format tiles improve {detected_style} visual coherence",
                    f"800x800 format has {round(features.estimated_floor_area_sqft / 5.8):.0f} fewer grout lines — cleaner look",
                    f"In {city} market, premium flooring adds 8-12% to rental yield",
                ],
                estimated_cost_inr="₹85-110/sqft (materials) + ₹45/sqft labour",
                roi_impact_pct=f"+{min(10, roi_pct * 0.6):.0f}–{min(14, roi_pct * 0.8):.0f}%",
                priority="high",
                specific_product="Nitco Vitrified 800x800 Polished",
                sku="NT-VT-8080",
            ))
        elif tier == "premium":
            recs.append(DesignRecommendation(
                title="Install Italian-Finish GVT Slab (800x1600)",
                category="flooring",
                reasoning=[
                    f"Current '{features.floor_type}' is inconsistent with premium tier expectations",
                    "1600mm slabs eliminate virtually all visible grout lines",
                    f"{detected_style} style at premium tier demands large-format stone-look flooring",
                    f"{city} luxury market commands +18-22% premium for premium flooring",
                ],
                estimated_cost_inr="₹185-285/sqft (materials) + ₹65/sqft labour",
                roi_impact_pct=f"+{min(15, roi_pct * 0.8):.0f}–{min(22, roi_pct):.0f}%",
                priority="high",
                specific_product="Simpolo GVT Slab 800x1600",
                sku="SP-GVT-8160",
            ))

    # ── 2. Wall treatment recommendation ─────────────────────────────────
    wall_lower = (features.wall_color + " " + features.wall_texture).lower()
    if "crack" in " ".join(features.specific_changes).lower() or \
       features.room_condition in ["fair", "poor"] or \
       "old" in wall_lower or "stain" in wall_lower:
        paint_products = {
            "basic": ("Asian Paints Apcolite Premium Emulsion", "AP-APG-20L", "₹185/litre"),
            "mid": ("Asian Paints Royale Sheen", "AP-RS-4L", "₹420/litre"),
            "premium": ("Dulux Velvet Touch Pearl Glo", "DX-VTP-4L", "₹720/litre"),
        }
        prod, sku, price = paint_products.get(tier, paint_products["mid"])
        recs.append(DesignRecommendation(
            title=f"Full Wall Repaint — {prod}",
            category="wall",
            reasoning=[
                f"Walls show {features.room_condition} condition — repaint will visually transform space immediately",
                f"Colour palette change from '{features.wall_color}' to {detected_style}-appropriate tone",
                f"Estimated {features.estimated_wall_area_sqft:.0f} sqft wall area — full treatment needed",
                f"Putty + primer + 2 coats {prod} — lasting 8-10 year finish",
            ],
            estimated_cost_inr=f"{price} + ₹40/sqft putty+primer+labour",
            roi_impact_pct="+5–8% (paint is highest-ROI renovation per rupee spent)",
            priority="high",
            specific_product=prod,
            sku=sku,
        ))

    # ── 3. False ceiling recommendation ──────────────────────────────────
    ceiling_lower = features.ceiling_type.lower()
    if tier in ["mid", "premium"] and ("plain" in ceiling_lower or "bare" in ceiling_lower):
        recs.append(DesignRecommendation(
            title="Add Gypsum False Ceiling with LED Cove Lighting",
            category="ceiling",
            reasoning=[
                f"Current ceiling '{features.ceiling_type}' is basic — false ceiling significantly modernises the space",
                f"LED cove lighting addresses detected lighting score of {features.lighting_score}/100",
                f"False ceiling reduces visual height from {features.estimated_height_ft:.0f}ft to 8ft — creates cozier feel",
                f"{detected_style} interiors standardly feature LED false ceilings",
                f"Covers estimated {features.estimated_floor_area_sqft:.0f} sqft floor area",
            ],
            estimated_cost_inr="₹120-150/sqft gypsum + ₹75/sqft labour + ₹5,000-15,000 LED strip",
            roi_impact_pct="+6–10% (false ceiling is a high-visibility upgrade in Indian market)",
            priority="medium" if tier == "mid" else "high",
            specific_product="Armstrong Gypsum Board 12mm + Philips Hue LED Strip",
        ))

    # ── 4. Lighting upgrade recommendation ────────────────────────────────
    if features.lighting_score < 70:
        light_type = features.lighting_type.lower()
        if "tube" in light_type or "fluorescent" in light_type or features.lighting_score < 55:
            recs.append(DesignRecommendation(
                title="Replace Fluorescent Lighting with LED Downlights",
                category="lighting",
                reasoning=[
                    f"Detected lighting '{features.lighting_type}' scores {features.lighting_score}/100 — below optimal",
                    "Fluorescent tubes cast flat, unflattering light unsuitable for {detected_style} aesthetic",
                    f"Natural light quality is '{features.natural_light_quality}' — supplemental lighting critical",
                    "LED downlights (4000K) reduce electricity consumption by 60% vs fluorescent",
                    f"Estimated {max(4, int(features.estimated_floor_area_sqft / 25))} downlights needed for {features.estimated_floor_area_sqft:.0f} sqft",
                ],
                estimated_cost_inr="₹350-500/downlight + ₹850/electrical point labour",
                roi_impact_pct="+3–5% (lighting is first impression for buyers/tenants)",
                priority="medium",
                specific_product="Philips SceneSwitch LED Downlight 7W",
                sku="PHL-SS-DL7W",
            ))

    # ── 5. Layout optimisation recommendation ────────────────────────────
    if features.layout_score < 65 or len(features.layout_issues) >= 2:
        recs.append(DesignRecommendation(
            title="Space Planning & Layout Optimisation",
            category="layout",
            reasoning=[
                f"Layout score: {features.layout_score}/100 — below optimal threshold of 70",
                f"Walkable space: {features.walkable_space_pct:.0f}% — {'adequate' if features.walkable_space_pct >= 40 else 'restricted'}",
            ] + features.layout_issues[:3],
            estimated_cost_inr="₹0 (repositioning) to ₹15,000 (space planner consultation)",
            roi_impact_pct="+2–4% (space efficiency improves livability score)",
            priority="medium" if features.layout_score >= 55 else "high",
        ))

    # Serialize to dicts
    return [
        {
            "title": r.title,
            "category": r.category,
            "reasoning": r.reasoning,
            "estimated_cost": r.estimated_cost_inr,
            "roi_impact": r.roi_impact_pct,
            "priority": r.priority,
            "product": r.specific_product,
            "sku": r.sku,
        }
        for r in recs
    ]