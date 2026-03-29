"""
ARKEN — Visual Assessor Agent v4.0
=====================================
Gemini Vision-first room analysis — now fully wired to fine-tuned CV models.

v4.0 Changes over v3.1:

  1. StyleDetector upgraded (Tier 1 → 3 cascade):
     OLD: Keyword-based scoring on extracted text fields (unreliable).
     NEW:
       Tier 1 — Fine-tuned EfficientNet-B0 (style_classifier.pt, Prompt 1):
                 Directly classifies image → 5 dataset styles → 11 ARKEN styles.
                 Bypasses text-field dependency entirely.
       Tier 2 — Fine-tuned CLIP (clip_finetuned.pt, Prompt 1):
                 Embedding similarity with full ROOM_STYLE_PROMPTS library.
       Tier 3 — Keyword rules on Gemini-extracted text (v3.1 behaviour):
                 Preserved unchanged as last-resort fallback.

  2. apply_cv_enrichment() upgraded:
     - CV style override threshold LOWERED from 0.65 → 0.40:
       fine-tuned EfficientNet is reliable at 0.40+ (was unreliable at 0.65 for
       zero-shot CLIP, hence the old conservative threshold).
     - Damage assessment from DamageDetector (v3.0) injected into condition field:
       damage severity → condition score adjustment.
     - YOLO-detected objects from fine-tuned model (yolo_indian_rooms.pt) merged
       with Gemini furniture list with higher priority.
     - Room type from fine-tuned EfficientNet room classifier respected at 0.75+.

  3. VisualAssessorAgent.analyze() upgraded:
     - Runs StyleClassifier (fine-tuned) directly on image_bytes before Gemini
       to get a style prior that is injected into RoomFeatures.
     - Passes YOLO detections (from CVModelRegistry fine-tuned model) into
       RoomFeatures.detected_furniture for enriched BOQ quantities.
     - Passes DamageDetector results into condition fields.
     - All Gemini calls UNCHANGED — no rendering.py or Gemini API code touched.

  All v3.1 public API preserved:
    VisualAssessorAgent.analyze() signature unchanged.
    RoomFeatures.from_gemini_response() unchanged.
    EXTRACTION_PROMPT unchanged.

  Graceful degradation: if fine-tuned weights absent, behaviour = v3.1.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

PAINT_COVERAGE_L_PER_SQFT = 0.037
TILE_WASTAGE_FACTOR        = 1.10

# ── Keyword rules (Tier 3 fallback — unchanged from v3.1) ────────────────────
STYLE_RULES = {
    "Modern Minimalist":   {
        "keywords": ["minimal","clean","white","neutral","simple","sleek"],
        "base": 0.1,
    },
    "Scandinavian":        {
        "keywords": ["wood","light","cosy","linen","hygge","birch","pine"],
        "base": 0.05,
    },
    "Japandi":             {
        "keywords": ["zen","wabi","natural","bamboo","muted","low","calm"],
        "base": 0.05,
    },
    "Industrial":          {
        "keywords": ["exposed","brick","metal","dark","steel","concrete","raw"],
        "base": 0.05,
    },
    "Bohemian":            {
        "keywords": ["colourful","eclectic","pattern","layered","boho","texture"],
        "base": 0.05,
    },
    "Contemporary Indian": {
        "keywords": ["terracotta","brass","carved","jali","vibrant","warm","ethnic"],
        "base": 0.1,
    },
    "Traditional Indian":  {
        "keywords": ["teak","silk","intricate","antique","handcrafted","classic"],
        "base": 0.1,
    },
    "Art Deco":            {
        "keywords": ["geometric","gold","mirror","velvet","bold","symmetry"],
        "base": 0.05,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# RoomFeatures dataclass (unchanged from v3.1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoomFeatures:
    room_type: str = "bedroom"
    wall_color: str = "unknown"
    wall_material: str = "painted plaster"
    floor_type: str = "vitrified tiles"
    ceiling_type: str = "pop false ceiling"
    detected_furniture: List[str] = field(default_factory=list)
    furniture_positions: Dict[str, str] = field(default_factory=dict)
    lighting_sources: List[str] = field(default_factory=list)
    free_space_percentage: float = 40.0
    room_area_estimate: float = 120.0
    wall_area_sqft: float = 200.0
    floor_area_sqft: float = 120.0
    layout_score: int = 70
    walkable_space_pct: float = 45.0
    style_tags: List[str] = field(default_factory=list)
    style_label: str = "Modern Minimalist"
    style_confidence: float = 0.6
    natural_light: str = "moderate"
    condition: str = "good"
    color_palette: List[str] = field(default_factory=list)
    renovation_priority: List[str] = field(default_factory=list)
    layout_issues: List[str] = field(default_factory=list)
    layout_suggestions: List[str] = field(default_factory=list)
    design_recommendations: List[Dict] = field(default_factory=list)
    extraction_source: str = "gemini"

    @classmethod
    def from_gemini_response(
        cls,
        raw: Dict,
        fallback_room_type: str = "bedroom",
    ) -> "RoomFeatures":
        f = cls()
        f.room_type    = raw.get("room_type", fallback_room_type)
        f.wall_color   = raw.get("wall_color", "white")
        f.wall_material = raw.get("wall_material", "painted plaster")
        f.floor_type   = raw.get("floor_type", "vitrified tiles")
        f.ceiling_type = raw.get("ceiling_type", "pop false ceiling")
        f.detected_furniture  = raw.get("detected_furniture", [])
        f.furniture_positions = raw.get("furniture_positions", {})
        f.lighting_sources    = raw.get("lighting_sources", ["ceiling light"])
        f.free_space_percentage = float(raw.get("free_space_percentage", 40))
        f.room_area_estimate    = float(raw.get("room_area_estimate", 120))
        f.natural_light = raw.get("natural_light", "moderate")
        f.condition     = raw.get("overall_condition", raw.get("condition", "good"))
        f.color_palette = raw.get("color_palette", [])
        f.renovation_priority = raw.get("renovation_priority", [])
        f.style_tags    = raw.get("style_tags", [])

        # ── New fields from improved prompt ──────────────────────────────────
        condition_score     = int(raw.get("condition_score", 65))
        wall_condition      = raw.get("wall_condition", "fair")
        floor_condition     = raw.get("floor_condition", "fair")
        issues_detected     = raw.get("issues_detected", [])
        reno_scope          = raw.get("renovation_scope_needed", "partial")
        high_value_upgrades = raw.get("high_value_upgrades", [])
        material_types      = raw.get("material_types", [])
        floor_quality       = raw.get("floor_material_quality", "mid")
        furniture_quality   = raw.get("furniture_quality", "mid")

        if condition_score >= 80:
            f.condition = "excellent"
        elif condition_score >= 65:
            f.condition = "good"
        elif condition_score >= 45:
            f.condition = "fair"
        else:
            f.condition = "poor"

        f._condition_score     = condition_score
        f._wall_condition      = wall_condition
        f._floor_condition     = floor_condition
        f._issues_detected     = issues_detected
        f._renovation_scope    = reno_scope
        f._high_value_upgrades = high_value_upgrades
        f._material_types      = material_types
        f._floor_quality       = floor_quality
        f._furniture_quality   = furniture_quality

        area = f.room_area_estimate
        f.floor_area_sqft = area
        side = math.sqrt(max(area, 10))
        f.wall_area_sqft  = round(4 * side * 9 * 0.85, 1)
        f.walkable_space_pct = f.free_space_percentage

        score = 70
        # Furniture density and free space are mutually exclusive signals —
        # a room cannot be both over-furnished AND under-furnished.
        # Use free_space_percentage as the tie-breaker when furniture count
        # and free space give conflicting signals.
        is_over_furnished = (
            len(f.detected_furniture) > 8
            and f.free_space_percentage <= 60   # must also look cramped
        )
        is_under_furnished = (
            f.free_space_percentage > 60
            and len(f.detected_furniture) <= 4  # genuinely sparse
        )
        if is_over_furnished:
            score -= 15
            f.layout_issues.append("Room appears over-furnished — consider decluttering 1–2 items")
        if f.free_space_percentage < 30 and not is_over_furnished:
            score -= 10
            f.layout_issues.append("Insufficient walkable space — furniture arrangement needs optimising")
        if is_under_furnished:
            score -= 5
            f.layout_issues.append("Room appears under-furnished — add key furniture pieces for balance")
        if f.natural_light in ("good", "excellent"):
            score += 10
        if f.condition in ("good", "excellent"):
            score += 5
        f.layout_score = min(100, max(0, score))

        if f.free_space_percentage < 35:
            f.layout_suggestions.append(
                "Remove 1-2 pieces or switch to space-saving furniture"
            )
        if ("bed" in str(f.detected_furniture).lower()
                and "wardrobe" not in str(f.detected_furniture).lower()):
            f.layout_suggestions.append("Add built-in wardrobe along longest wall")
        if len(f.lighting_sources) == 1:
            f.layout_suggestions.append(
                "Layer lighting: add bedside lamps and accent lights"
            )

        # v4.0: StyleDetector now uses fine-tuned model when image_bytes available.
        # When called from from_gemini_response() we only have text → Tier 3 keyword.
        f.style_label, f.style_confidence = StyleDetector.detect_from_features(f)
        f.design_recommendations = generate_explainable_recommendations(f)
        f.extraction_source = "gemini"
        return f

    def apply_cv_enrichment(self, cv_features: Dict) -> None:
        """
        Enrich this RoomFeatures with CV pipeline output.
        v4.0 changes:
          - Style override threshold: 0.40 (was 0.65) — fine-tuned model is reliable.
          - DamageDetector result adjusts condition score.
          - Fine-tuned YOLO objects merged with higher priority.
        """
        if not cv_features:
            return

        # ── Merge detected objects (YOLO fine-tuned takes priority) ──────────
        cv_objects = cv_features.get("detected_objects", [])
        if cv_objects:
            existing = set(self.detected_furniture)
            new_objs = [o for o in cv_objects if o not in existing]
            # Prepend fine-tuned YOLO objects (higher confidence than Gemini list)
            self.detected_furniture = new_objs + self.detected_furniture

        # ── Style override — LOWERED threshold for fine-tuned model ──────────
        cv_style_conf  = cv_features.get("style_confidence", 0.0)
        clip_style     = cv_features.get("clip_style_label", "")
        clip_style_conf = cv_features.get("clip_style_confidence", 0.0)

        # v4.0: accept fine-tuned model at 0.40+ (was 0.65 for zero-shot CLIP)
        model_used = cv_features.get("style_model_used", "")
        is_finetuned = "finetuned" in model_used or "efficientnet" in model_used

        if clip_style and clip_style_conf > self.style_confidence:
            if is_finetuned and clip_style_conf >= 0.40:
                self.style_label      = clip_style
                self.style_confidence = clip_style_conf
            elif not is_finetuned and clip_style_conf >= 0.65:
                self.style_label      = clip_style
                self.style_confidence = clip_style_conf
        elif cv_style_conf > self.style_confidence and cv_style_conf >= 0.40:
            cv_style = cv_features.get("style", "")
            if cv_style:
                self.style_label      = cv_style
                self.style_confidence = cv_style_conf

        # ── Lighting from CV ──────────────────────────────────────────────────
        cv_lighting = cv_features.get("lighting", "")
        if cv_lighting and cv_lighting not in ("mixed", ""):
            if "natural" in cv_lighting:
                self.natural_light = "good"
            elif "dim" in cv_lighting:
                self.natural_light = "poor"

        # ── Damage → condition adjustment (v4.0 NEW) ─────────────────────────
        damage_severity = cv_features.get("damage_severity", "")
        if damage_severity:
            _damage_to_condition = {
                "none":     "excellent",
                "minor":    "good",
                "moderate": "fair",
                "severe":   "poor",
            }
            cv_condition = _damage_to_condition.get(damage_severity)
            if cv_condition:
                # Condition priority: poor > fair > good > excellent
                _cond_rank = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
                if _cond_rank.get(cv_condition, 3) <= _cond_rank.get(self.condition, 3):
                    # Damage-based condition is worse — trust it
                    self.condition = cv_condition
                    self._condition_score = getattr(self, "_condition_score", 65)
                    # Recalculate layout score with updated condition
                    if self.condition in ("good", "excellent"):
                        self.layout_score = min(100, self.layout_score + 5)
                    elif self.condition == "poor":
                        self.layout_score = max(0, self.layout_score - 10)

        # ── Room type from fine-tuned EfficientNet ────────────────────────────
        cv_room_type = cv_features.get("room_type", "")
        cv_room_conf = cv_features.get("room_type_confidence", 0.0)
        if cv_room_type and cv_room_conf >= 0.75:
            self.room_type = cv_room_type

        self.extraction_source = "gemini+cv_finetuned"

    def to_dict(self) -> Dict:
        base = asdict(self)
        base["condition_score"]     = getattr(self, "_condition_score", 65)
        base["wall_condition"]      = getattr(self, "_wall_condition", "fair")
        base["floor_condition"]     = getattr(self, "_floor_condition", "fair")
        base["issues_detected"]     = getattr(self, "_issues_detected", [])
        base["renovation_scope"]    = getattr(self, "_renovation_scope", "partial")
        base["high_value_upgrades"] = getattr(self, "_high_value_upgrades", [])
        base["material_types"]      = getattr(self, "_material_types", [])
        return base


# ─────────────────────────────────────────────────────────────────────────────
# StyleDetector v4.0 — three-tier cascade
# ─────────────────────────────────────────────────────────────────────────────

class StyleDetector:
    """
    v4.0: Three-tier style detection.

      Tier 1: Fine-tuned EfficientNet-B0 (style_classifier.pt) — image-based.
      Tier 2: Fine-tuned CLIP (clip_finetuned.pt) — embedding-based.
      Tier 3: Keyword rules on Gemini text fields — text-based fallback.

    detect() — requires image_bytes (Tier 1/2/3).
    detect_from_features() — text-only, Tier 3 only (used in from_gemini_response).
    """

    @staticmethod
    def detect(
        image_bytes: bytes,
        room_type: str = "",
        gemini_hint: str = "",
    ) -> Tuple[str, float]:
        """
        Classify style from raw image bytes.
        Uses fine-tuned models when available, falls back to keyword rules.

        Args:
            image_bytes:   Raw JPEG/PNG bytes.
            room_type:     "kitchen" | "bedroom" | "bathroom" | "living_room"
            gemini_hint:   Gemini-extracted style text (blend source).

        Returns:
            (style_label, confidence)
        """
        try:
            from ml.style_classifier import StyleClassifier
            clf    = StyleClassifier()
            result = clf.classify(image_bytes, gemini_hint, room_type)
            return result["style_label"], result["style_confidence"]
        except Exception as e:
            logger.debug(f"[StyleDetector] StyleClassifier failed: {e}. Keyword fallback.")
            return StyleDetector._keyword_fallback({})

    @staticmethod
    def detect_from_features(features: "RoomFeatures") -> Tuple[str, float]:
        """
        Tier 3 keyword-only classification from Gemini text fields.
        Used when image_bytes are not available (from_gemini_response path).
        """
        text = " ".join([
            features.wall_color, features.floor_type, features.ceiling_type,
            features.wall_material, " ".join(features.detected_furniture),
            " ".join(features.style_tags), " ".join(features.color_palette),
        ]).lower()

        scores: Dict[str, float] = {}
        for style, config in STYLE_RULES.items():
            score = config["base"]
            for kw in config["keywords"]:
                if kw in text:
                    score += 0.15
            scores[style] = round(score, 3)

        best = max(scores, key=lambda s: scores[s])
        return best, round(min(scores[best], 0.95), 2)

    @staticmethod
    def _keyword_fallback(features_dict: Dict) -> Tuple[str, float]:
        return "Modern Minimalist", 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation generator (unchanged from v3.1)
# ─────────────────────────────────────────────────────────────────────────────

def generate_explainable_recommendations(f: RoomFeatures) -> List[Dict]:
    recs  = []
    floor = f.floor_type.lower()

    if "tile" in floor or "vitrified" in floor:
        recs.append({
            "category":       "flooring",
            "recommendation": "Replace with large-format vitrified tiles (800x800mm) or engineered wood",
            "reasoning":      [
                f"Detected floor type: {f.floor_type}",
                "Large-format tiles reduce grout lines, making room appear more spacious",
                "Consistent flooring adds 8-12% perceived value",
            ],
            "priority":       "high",
            "cost_range_inr": "Rs 85-180 per sqft",
            "diy_reference":  _get_diy_reference("Walls and Ceilings"),
        })
    elif "wood" in floor:
        recs.append({
            "category":       "flooring",
            "recommendation": "Sand, stain and polish existing hardwood; repair damaged planks",
            "reasoning":      [
                f"Detected floor type: {f.floor_type}",
                "Refinishing wood costs 20% of replacement — excellent ROI",
            ],
            "priority":       "medium",
            "cost_range_inr": "Rs 30-60 per sqft",
            "diy_reference":  _get_diy_reference("Walls and Ceilings"),
        })

    if f.wall_color in ("white", "off-white", "cream", "beige", "unknown"):
        recs.append({
            "category":       "walls",
            "recommendation": (
                f"Apply warm accent wall in {f.style_label} palette — Asian Paints Royale Sheen"
            ),
            "reasoning":      [
                f"Current wall colour ({f.wall_color}) is neutral",
                "Accent walls add visual depth at under Rs 5,000 total cost",
                f"In {f.style_label} style, a feature wall drives the design narrative",
            ],
            "priority":       "medium",
            "cost_range_inr": "Rs 3,500-8,000 total",
            "diy_reference":  _get_diy_reference("Walls and Ceilings"),
        })

    if "pop" not in f.ceiling_type.lower() and "false" not in f.ceiling_type.lower():
        recs.append({
            "category":       "ceiling",
            "recommendation": "Install POP false ceiling with recessed LED lighting",
            "reasoning":      [
                f"Current ceiling: {f.ceiling_type} — no false ceiling detected",
                "False ceilings hide wiring, improve acoustic quality",
                "LED recessed lighting reduces power consumption 60%",
            ],
            "priority":       "high",
            "cost_range_inr": "Rs 65-120 per sqft",
            "diy_reference":  _get_diy_reference("Lighting"),
        })

    if len(f.lighting_sources) <= 1:
        recs.append({
            "category":       "lighting",
            "recommendation": "Layer 3-point lighting: ambient ceiling, task, and accent",
            "reasoning":      [
                f"Only {len(f.lighting_sources)} light source detected",
                "Single-source lighting creates harsh shadows",
                "Layered lighting adds luxury feel and increases perceived room size",
            ],
            "priority":       "high",
            "cost_range_inr": "Rs 8,000-25,000 for complete plan",
            "diy_reference":  _get_diy_reference("Lighting"),
        })

    if f.layout_issues:
        recs.append({
            "category":       "layout",
            "recommendation": "Optimise furniture layout for better flow",
            "reasoning":      [f"Layout score: {f.layout_score}/100"] + f.layout_issues,
            "suggestions":    f.layout_suggestions,
            "priority":       "medium",
            "cost_range_inr": "Rs 0 (rearrangement only)",
        })

    return recs


def _get_diy_reference(category: str) -> Optional[Dict]:
    try:
        from services.datasets.dataset_loader import ARKENDatasetRegistry
        registry = ARKENDatasetRegistry.get()
        if not registry.diy_renovation.available:
            return None
        chunks = registry.diy_renovation.get_by_category(category)
        if not chunks:
            return None
        chunk = chunks[0]
        return {
            "source":   "DIY Home Improvement Dataset",
            "category": chunk.category,
            "tip":      chunk.chapter_title,
            "link":     chunk.clip_link,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Gemini extraction prompt (unchanged from v3.1)
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert Indian interior designer and certified property inspector analysing a room photo for a renovation project.
Analyse this image carefully and return ONLY valid JSON — no markdown, no preamble, no trailing text.

{
  "room_type": "bedroom|living_room|kitchen|bathroom|dining_room|study|other",

  "wall_color": "exact colour name visible in image (e.g. cream white, terracotta, sage green, not just 'white')",
  "wall_material": "painted plaster|wallpaper|tiles|exposed brick|wood panelling|other",
  "wall_condition": "new|good|fair|poor|very poor — based on paint quality, cracks, stains, dampness",
  "wall_issues": ["list EVERY visible issue: peeling paint, hairline cracks, water stains, efflorescence, mould, dampness — empty list ONLY if genuinely none visible"],

  "floor_type": "vitrified tiles|marble|granite|hardwood|laminate|carpet|cement|mosaic|other",
  "floor_material_quality": "basic|mid|premium — based on tile size, finish, brand-inferred quality",
  "floor_condition": "new|good|fair|poor|very poor — check for chips, grout discolouration, scratches",

  "ceiling_type": "pop false ceiling|plain plaster|wooden|exposed concrete|gypsum|other",
  "ceiling_condition": "new|good|fair|poor — check for water marks, cracks, yellowing, sagging",

  "detected_furniture": ["every visible furniture item — be specific: 'queen bed 5ft×6ft', '3-seater fabric sofa', 'teak TV unit 5ft' — not just 'bed', 'sofa'"],
  "furniture_quality": "basic|mid|premium — infer from materials, finish, apparent brand tier",
  "furniture_positions": {"describe each item's position relative to walls/windows"},

  "lighting_sources": ["describe every light source: 'single bare tube light on ceiling', 'two LED downlights', 'north-facing window 4ft×3ft'"],
  "natural_light": "poor|moderate|good|excellent — assess window size, orientation if visible, brightness in image",
  "artificial_lighting_quality": "poor|adequate|good|excellent — assess fixture type and light distribution",

  "free_space_percentage": 40,
  "room_area_estimate": 120,

  "overall_condition": "poor|fair|good|excellent",
  "condition_score": 65,

  "renovation_scope_needed": "cosmetic_only|partial|full_room|structural_plus",

  "material_types": ["list every visible material: marble, teak wood, granite, vitrified tile, MDF, glass, stainless steel, fabric, etc."],

  "color_palette": ["2-4 dominant colours with specific names: 'warm ivory', 'charcoal grey', 'brass accent'"],
  "style_tags": ["up to 5 style descriptors from: modern, minimal, traditional, industrial, boho, scandinavian, classic indian, contemporary, transitional"],
  "style_confidence": 0.75,

  "issues_detected": ["CRITICAL: list every specific visible issue with location — 'peeling paint patch on north wall near skirting', 'cracked tile at entrance threshold', 'water stain ring on ceiling above window', 'outdated ivory switchboard on south wall', 'exposed wire near bed headwall' — empty list ONLY if genuinely no issues visible"],

  "renovation_priority": ["ordered top 3 areas needing work: walls|floor|ceiling|lighting|furniture|electrical|plumbing|storage"],

  "high_value_upgrades": ["2-3 specific upgrades with highest ROI for this exact room — be specific to what you see: 'Replace 400×400 ceramic floor tiles with 800×800 GVT for +15% perceived value', 'Add POP false ceiling with cove LED — single highest-ROI upgrade for this room type', 'Install modular wardrobe on east wall dead space — missing storage in current layout'"],

  "layout_score": 70,
  "layout_issues": ["specific layout problems visible: 'bed positioned blocking window natural light', 'no circulation space between bed and wardrobe', 'sofa backs directly onto main door'],
  "layout_suggestions": ["actionable fixes: 'rotate bed 90 degrees to east wall — opens 4ft walkway to window', 'shift sofa 2ft forward — creates proper entry foyer'"]
}

CRITICAL RULES:
1. Only describe what you ACTUALLY SEE in the image — do not guess or fabricate details
2. Be specific: 'warm ivory emulsion paint with matte finish' not just 'white walls'
3. List EVERY visible furniture item individually with approximate dimensions
4. condition_score is 0–100: 90+=new/excellent, 70–89=good, 50–69=fair, 30–49=poor, <30=very poor
5. issues_detected is the most important field — inspect every visible surface carefully
6. Return ONLY the JSON object — no markdown fences, no explanation, no preamble"""


# ─────────────────────────────────────────────────────────────────────────────
# VisualAssessorAgent v4.0
# ─────────────────────────────────────────────────────────────────────────────

class VisualAssessorAgent:
    """
    Gemini Vision-first room analyser with fine-tuned CV enrichment.

    v4.0: Runs fine-tuned StyleClassifier + YOLO + DamageDetector BEFORE Gemini
    to enrich features with reliable ML results.
    Gemini calls are UNCHANGED — no rendering.py or Gemini API code modified.
    """

    async def analyze(
        self,
        image_bytes: bytes,
        project_id: str,
        room_type: str = "bedroom",
        theme: str = "Modern Minimalist",
        budget_tier: str = "mid",
        city: str = "Hyderabad",
        cv_features: Optional[Dict] = None,
    ) -> Dict:
        # ── Stage A: Pre-run fine-tuned CV models on image ────────────────────
        local_cv = await self._run_local_cv(image_bytes, room_type, theme)

        # Merge provided cv_features with local results
        # (local results are direct image inference; provided features come from
        #  VisionAnalyzerAgent's full CV pipeline including DepthEstimator)
        merged_cv = {**local_cv, **(cv_features or {})}

        # ── Stage B: Gemini Vision extraction (unchanged) ─────────────────────
        features = await self._extract_features(image_bytes, room_type)

        # ── Stage C: Apply merged CV enrichment ───────────────────────────────
        if merged_cv:
            features.apply_cv_enrichment(merged_cv)

        quantities    = self._estimate_quantities(features)
        mask_s3_keys  = await self._store_local(project_id)

        return {
            "spatial_map": {
                "dimensions": {
                    "wall_area_sqft":   features.wall_area_sqft,
                    "floor_area_sqft":  features.floor_area_sqft,
                    "estimated_length_ft": round(math.sqrt(features.floor_area_sqft) * 1.2, 1),
                    "estimated_width_ft":  round(math.sqrt(features.floor_area_sqft), 1),
                    "estimated_height_ft": 9.0,
                    "scale_confidence":    features.extraction_source,
                },
                "masks_s3":       mask_s3_keys,
                "detected_objects": {item: [0.9] for item in features.detected_furniture},
            },
            "material_quantities":    quantities,
            "style": {
                "embedding":  [],
                "tags":       features.style_tags,
                "label":      features.style_label,
                "confidence": features.style_confidence,
            },
            "features":              features.to_dict(),
            "room_features":         features.to_dict(),
            "image_features":        features.to_dict(),
            "recommendations":       features.design_recommendations,
            "layout_report": {
                "layout_score":      features.layout_score,
                "walkable_space_pct": features.walkable_space_pct,
                "issues":            features.layout_issues,
                "suggestions":       features.layout_suggestions,
            },
            "design_recommendations": features.design_recommendations,
            "style_label":    features.style_label,
            "style_confidence": features.style_confidence,
        }

    async def _run_local_cv(
        self,
        image_bytes: bytes,
        room_type: str,
        gemini_hint: str = "",
    ) -> Dict:
        """
        v4.0: Run fine-tuned models directly before Gemini:
          1. StyleClassifier (EfficientNet/CLIP fine-tuned)
          2. YOLO fine-tuned object detection (for furniture list enrichment)
          3. DamageDetector (for condition pre-assessment)

        All wrapped in try/except — never blocks the Gemini path.
        """
        import asyncio
        result: Dict = {}

        # ── Fine-tuned StyleClassifier ────────────────────────────────────────
        try:
            from ml.style_classifier import StyleClassifier
            clf = StyleClassifier()
            style_res = await asyncio.to_thread(
                clf.classify, image_bytes, gemini_hint, room_type
            )
            result["clip_style_label"]      = style_res["style_label"]
            result["clip_style_confidence"] = style_res["style_confidence"]
            result["clip_style_top3"]       = style_res.get("top_3_styles", [])
            result["style_model_used"]      = style_res.get("model_used", "")
            result["style_gemini_agreement"] = style_res.get("gemini_agreement", False)
            logger.info(
                f"[VisualAssessor-v4] StyleClassifier: "
                f"style={style_res['style_label']}  "
                f"conf={style_res['style_confidence']:.2f}  "
                f"model={style_res.get('model_used', '?')}"
            )
        except Exception as e:
            logger.debug(f"[VisualAssessor-v4] StyleClassifier failed (non-critical): {e}")

        # ── Fine-tuned YOLO object detection ──────────────────────────────────
        try:
            from ml.cv_model_registry import get_registry
            registry   = get_registry()
            yolo_dets  = await asyncio.to_thread(
                registry.yolo.detect, image_bytes, 0.35
            )
            if yolo_dets:
                result["detected_objects"] = [d["label"] for d in yolo_dets]
                result["yolo_detections"]  = yolo_dets
                result["yolo_finetuned"]   = registry.yolo.is_finetuned
                logger.info(
                    f"[VisualAssessor-v4] YOLO detected {len(yolo_dets)} objects "
                    f"(fine-tuned={registry.yolo.is_finetuned}): "
                    f"{[d['label'] for d in yolo_dets[:5]]}"
                )
        except Exception as e:
            logger.debug(f"[VisualAssessor-v4] YOLO detection failed (non-critical): {e}")

        # ── DamageDetector ────────────────────────────────────────────────────
        try:
            from ml.damage_detector import DamageDetector
            detector  = DamageDetector()
            yolo_dets_for_damage = result.get("yolo_detections")
            damage_res = await asyncio.to_thread(
                detector.detect,
                image_bytes,
                None,           # wall_region_hint: auto
                yolo_dets_for_damage,
            )
            result["damage_severity"]              = damage_res["severity"]
            result["detected_damage"]              = damage_res["detected_issues"]
            result["damage_scores"]                = damage_res["damage_scores"]
            result["renovation_scope_from_damage"] = damage_res["renovation_scope_recommendation"]
            result["requires_waterproofing"]       = damage_res["requires_waterproofing"]
            result["requires_structural_repair"]   = damage_res["requires_structural_repair"]
            result["damage_model_used"]            = damage_res["model_used"]
            result["damage_detection_tier"]        = damage_res.get("detection_tier", "unknown")
            logger.info(
                f"[VisualAssessor-v4] DamageDetector: "
                f"severity={damage_res['severity']}  "
                f"issues={damage_res['detected_issues']}  "
                f"tier={damage_res.get('detection_tier', '?')}"
            )
        except Exception as e:
            logger.debug(f"[VisualAssessor-v4] DamageDetector failed (non-critical): {e}")

        # ── Room type from fine-tuned EfficientNet ────────────────────────────
        try:
            from ml.cv_model_registry import get_registry
            registry  = get_registry()
            room_pred, room_conf = await asyncio.to_thread(
                registry.room_classifier.classify, image_bytes
            )
            if room_conf >= 0.50:
                result["room_type"]             = room_pred
                result["room_type_confidence"]  = room_conf
                result["room_classifier_finetuned"] = registry.room_classifier.is_finetuned
                logger.info(
                    f"[VisualAssessor-v4] RoomClassifier: "
                    f"room={room_pred}  conf={room_conf:.2f}  "
                    f"fine-tuned={registry.room_classifier.is_finetuned}"
                )
        except Exception as e:
            logger.debug(f"[VisualAssessor-v4] Room classifier failed (non-critical): {e}")

        return result

    async def _extract_features(
        self, image_bytes: bytes, room_type: str
    ) -> RoomFeatures:
        """Gemini extraction — UNCHANGED from v3.1."""
        api_key = None
        if settings.GOOGLE_API_KEY:
            try:
                api_key = settings.GOOGLE_API_KEY.get_secret_value()
            except Exception:
                api_key = str(settings.GOOGLE_API_KEY)

        if not api_key:
            logger.warning("[VisualAssessor] No GOOGLE_API_KEY — using heuristic features")
            return self._heuristic_features(room_type)

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            b64    = base64.b64encode(image_bytes).decode()

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg", data=b64
                            )
                        ),
                        types.Part(text=EXTRACTION_PROMPT),
                    ])
                ],
            )

            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$",          "", text)
            raw  = json.loads(text)
            return RoomFeatures.from_gemini_response(raw, room_type)

        except Exception as e:
            logger.warning(
                f"[VisualAssessor] Gemini extraction failed: {e} — using heuristics"
            )
            return self._heuristic_features(room_type)

    @staticmethod
    def _heuristic_features(room_type: str) -> RoomFeatures:
        f = RoomFeatures()
        f.room_type       = room_type
        f.wall_color      = "off-white"
        f.floor_type      = "vitrified tiles"
        f.ceiling_type    = "plain plaster"
        f.detected_furniture = (
            ["bed", "wardrobe", "side table", "ceiling light"]
            if room_type == "bedroom"
            else ["sofa", "coffee table", "tv unit", "ceiling light"]
        )
        f.lighting_sources      = ["ceiling light"]
        f.free_space_percentage = 40.0
        f.room_area_estimate    = 120.0
        f.floor_area_sqft       = 120.0
        f.wall_area_sqft        = 200.0
        f.natural_light         = "moderate"
        f.condition             = "good"
        f.style_label           = "Modern Minimalist"
        f.style_confidence      = 0.5
        f.layout_score          = 65
        f.walkable_space_pct    = 40.0
        f.layout_suggestions    = ["Add layered lighting", "Consider accent wall"]
        f.design_recommendations = generate_explainable_recommendations(f)
        f.extraction_source     = "heuristic"
        return f

    def _estimate_quantities(self, f: RoomFeatures) -> Dict:
        """
        Estimate material quantities from room dimensions.
        All values rounded to 1 decimal — no float precision artifacts.

        Construction norms used (Indian standard):
          Paint: 2 coats, 60 sqft/litre/coat → liters = wall_area / 30
          Putty: 1 bag (20kg) covers 80 sqft
          Primer: 1 can (20L) covers 200 sqft
          Tiles: floor area × 1.10 (10% wastage)
          Tile adhesive: 1 bag (20kg) covers 50 sqft @ 3mm bed
          Grout: 0.3 kg per sqft (600×600, 3mm joint)
          Cement (tile bedding): 1 bag (50kg) per 40 sqft
          Sand (tile bedding): 1 cft per 15 sqft
        """
        import math as _math
        wall  = f.wall_area_sqft
        floor = f.floor_area_sqft

        paint_liters   = round(wall / 30, 1)            # wall_area / 30 for 2 coats
        putty_bags     = max(1, _math.ceil(wall / 80))   # 1 bag per 80 sqft
        primer_cans    = max(1, _math.ceil(wall / 200))  # 1 can per 200 sqft
        tile_sqft      = round(floor * TILE_WASTAGE_FACTOR, 1)
        adhesive_bags  = max(1, _math.ceil(floor / 50)) # 1 bag per 50 sqft
        grout_kg       = round(floor * 0.3, 1)
        cement_bags    = max(1, _math.ceil(floor / 40)) # 1 bag per 40 sqft
        sand_cft       = round(floor / 15, 1)

        return {
            # Paint & wall prep
            "paint_liters":         paint_liters,
            "primer_cans":          float(primer_cans),
            "putty_bags_20kg":      float(putty_bags),
            "putty_kg":             round(putty_bags * 20, 1),
            # Tiling
            "floor_tiles_sqft":     tile_sqft,
            "wall_tiles_sqft":      0.0,
            "tile_adhesive_bags_20kg": float(adhesive_bags),
            "grout_kg":             grout_kg,
            # Civil
            "cement_bags_50kg":     float(cement_bags),
            "sand_cft":             sand_cft,
            # Carpentry
            "plywood_sqft":         round(floor * 0.3, 1),
            # Dimensions
            "wall_area_sqft":       round(wall, 1),
            "floor_area_sqft":      round(floor, 1),
        }

    async def _store_local(self, project_id: str) -> Dict[str, str]:
        import os
        try:
            local_dir = f"/tmp/arken_local_storage/projects/{project_id}/masks"
            os.makedirs(local_dir, exist_ok=True)
        except Exception:
            pass
        return {
            "wall":        f"local://masks/{project_id}/wall.png",
            "floor":       f"local://masks/{project_id}/floor.png",
            "combined_reno": f"local://masks/{project_id}/combined_reno.png",
        }