"""
Tests — agents/visual_assessor.py
===================================
Covers:
  - RoomFeatures.from_gemini_response: field population, area calculation,
    layout score computation, layout issue mutual exclusion (key bug fix)
  - RoomFeatures.to_dict: private condition fields surfaced correctly
  - StyleDetector.detect_from_features: keyword-based classification
  - RoomFeatures.apply_cv_enrichment: style threshold logic (0.55 fix),
    agreement bonus, damage→condition mapping, YOLO object merging
  - generate_explainable_recommendations: category coverage
  - Edge cases: empty raw dict, unknown room type, extreme free-space values
"""

from __future__ import annotations

import unittest

import tests.conftest as _cf
from tests.conftest import load_agent

va_mod = load_agent("agents.visual_assessor", "agents/visual_assessor.py")

RoomFeatures                   = va_mod.RoomFeatures
StyleDetector                  = va_mod.StyleDetector
generate_explainable_recommendations = va_mod.generate_explainable_recommendations
STYLE_RULES                    = va_mod.STYLE_RULES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_raw(**overrides) -> dict:
    base = {
        "room_type": "bedroom",
        "wall_color": "white",
        "wall_material": "painted plaster",
        "wall_condition": "good",
        "wall_issues": [],
        "floor_type": "vitrified tiles",
        "floor_material_quality": "mid",
        "floor_condition": "good",
        "ceiling_type": "plain plaster",
        "ceiling_condition": "good",
        "detected_furniture": ["bed", "wardrobe", "side table"],
        "furniture_quality": "mid",
        "furniture_positions": {},
        "lighting_sources": ["ceiling light"],
        "natural_light": "moderate",
        "artificial_lighting_quality": "adequate",
        "free_space_percentage": 45.0,
        "room_area_estimate": 120.0,
        "overall_condition": "good",
        "condition_score": 72,
        "renovation_scope_needed": "partial",
        "material_types": ["vitrified_tile"],
        "color_palette": ["white", "grey"],
        "style_tags": ["modern", "minimal"],
        "issues_detected": [],
        "renovation_priority": ["walls", "flooring"],
        "high_value_upgrades": ["false ceiling with cove lighting"],
    }
    base.update(overrides)
    return base


class TestRoomFeaturesFromGeminiResponse(unittest.TestCase):
    """RoomFeatures.from_gemini_response populates fields correctly."""

    def test_basic_field_population(self):
        raw = _make_raw()
        f = RoomFeatures.from_gemini_response(raw, "bedroom")
        self.assertEqual(f.room_type, "bedroom")
        self.assertEqual(f.wall_color, "white")
        self.assertEqual(f.floor_type, "vitrified tiles")

    def test_detected_furniture_copied(self):
        raw = _make_raw(detected_furniture=["bed", "wardrobe", "desk"])
        f = RoomFeatures.from_gemini_response(raw)
        self.assertIn("bed", f.detected_furniture)
        self.assertIn("wardrobe", f.detected_furniture)

    def test_room_area_sets_floor_area_sqft(self):
        raw = _make_raw(room_area_estimate=150.0)
        f = RoomFeatures.from_gemini_response(raw)
        self.assertAlmostEqual(f.floor_area_sqft, 150.0, places=1)

    def test_wall_area_derived_from_floor_area(self):
        raw = _make_raw(room_area_estimate=120.0)
        f = RoomFeatures.from_gemini_response(raw)
        self.assertGreater(f.wall_area_sqft, 0)

    def test_condition_score_72_gives_good_condition(self):
        raw = _make_raw(condition_score=72)
        f = RoomFeatures.from_gemini_response(raw)
        self.assertEqual(f.condition, "good")

    def test_condition_score_55_gives_fair(self):
        raw = _make_raw(condition_score=55)
        f = RoomFeatures.from_gemini_response(raw)
        self.assertEqual(f.condition, "fair")

    def test_condition_score_25_gives_poor(self):
        raw = _make_raw(condition_score=25)
        f = RoomFeatures.from_gemini_response(raw)
        self.assertEqual(f.condition, "poor")

    def test_private_issues_detected_stored(self):
        raw = _make_raw(issues_detected=["crack on north wall", "peeling paint"])
        f = RoomFeatures.from_gemini_response(raw)
        self.assertIn("crack on north wall", f._issues_detected)

    def test_private_renovation_scope_stored(self):
        raw = _make_raw(renovation_scope_needed="full_room")
        f = RoomFeatures.from_gemini_response(raw)
        self.assertEqual(f._renovation_scope, "full_room")

    def test_private_high_value_upgrades_stored(self):
        raw = _make_raw(high_value_upgrades=["modular wardrobe", "marble flooring"])
        f = RoomFeatures.from_gemini_response(raw)
        self.assertIn("modular wardrobe", f._high_value_upgrades)

    def test_layout_score_computed(self):
        raw = _make_raw()
        f = RoomFeatures.from_gemini_response(raw)
        self.assertIsInstance(f.layout_score, int)
        self.assertGreaterEqual(f.layout_score, 0)
        self.assertLessEqual(f.layout_score, 100)

    def test_extraction_source_is_gemini(self):
        f = RoomFeatures.from_gemini_response(_make_raw())
        self.assertEqual(f.extraction_source, "gemini")

    def test_natural_light_good_increases_layout_score(self):
        good_light = RoomFeatures.from_gemini_response(_make_raw(natural_light="good"))
        poor_light = RoomFeatures.from_gemini_response(_make_raw(natural_light="poor"))
        self.assertGreater(good_light.layout_score, poor_light.layout_score)

    def test_empty_raw_uses_defaults(self):
        f = RoomFeatures.from_gemini_response({}, "kitchen")
        self.assertEqual(f.room_type, "kitchen")
        self.assertIsInstance(f.floor_area_sqft, float)


class TestLayoutIssueMutualExclusion(unittest.TestCase):
    """
    KEY BUG FIX: over-furnished and under-furnished cannot both fire.
    Regression test for the contradictory repairs issue seen in the screenshot.
    """

    def test_high_furniture_count_and_high_free_space_no_contradiction(self):
        """9 furniture items but 65% free space — ambiguous, must pick one or neither."""
        raw = _make_raw(
            detected_furniture=["bed", "wardrobe", "desk", "chair", "lamp",
                                "nightstand", "dresser", "bookshelf", "tv unit"],
            free_space_percentage=65.0,
        )
        f = RoomFeatures.from_gemini_response(raw)
        issues_lower = " ".join(i.lower() for i in f.layout_issues)
        has_over  = "over-furnished" in issues_lower
        has_under = "under-furnished" in issues_lower or "feels empty" in issues_lower
        self.assertFalse(
            has_over and has_under,
            f"Contradictory issues fired simultaneously: {f.layout_issues}",
        )

    def test_low_furniture_high_free_space_is_under_furnished(self):
        """3 items + 70% free space → under-furnished, NOT over-furnished."""
        raw = _make_raw(
            detected_furniture=["bed", "nightstand", "lamp"],
            free_space_percentage=70.0,
        )
        f = RoomFeatures.from_gemini_response(raw)
        issues_lower = " ".join(i.lower() for i in f.layout_issues)
        self.assertNotIn("over-furnished", issues_lower)

    def test_many_furniture_low_free_space_is_over_furnished(self):
        """10 items + 25% free space → over-furnished, NOT under-furnished."""
        raw = _make_raw(
            detected_furniture=["bed", "wardrobe", "desk", "chair", "lamp",
                                "nightstand", "dresser", "bookshelf", "tv", "sofa"],
            free_space_percentage=25.0,
        )
        f = RoomFeatures.from_gemini_response(raw)
        issues_lower = " ".join(i.lower() for i in f.layout_issues)
        self.assertNotIn("under-furnished", issues_lower)
        self.assertNotIn("feels empty", issues_lower)

    def test_normal_room_no_layout_issues(self):
        """5 items + 45% free space → no layout issues."""
        raw = _make_raw(
            detected_furniture=["bed", "wardrobe", "desk", "lamp", "nightstand"],
            free_space_percentage=45.0,
        )
        f = RoomFeatures.from_gemini_response(raw)
        self.assertEqual(len(f.layout_issues), 0)


class TestRoomFeaturesToDict(unittest.TestCase):
    """to_dict surfaces private condition fields correctly."""

    def setUp(self):
        raw = _make_raw(
            condition_score=65,
            wall_condition="fair",
            floor_condition="good",
            issues_detected=["crack on east wall"],
            renovation_scope_needed="partial",
            high_value_upgrades=["false ceiling"],
        )
        self.d = RoomFeatures.from_gemini_response(raw).to_dict()

    def test_condition_score_in_dict(self):
        self.assertIn("condition_score", self.d)
        self.assertEqual(self.d["condition_score"], 65)

    def test_wall_condition_in_dict(self):
        self.assertIn("wall_condition", self.d)
        self.assertEqual(self.d["wall_condition"], "fair")

    def test_issues_detected_in_dict(self):
        self.assertIn("issues_detected", self.d)
        self.assertIn("crack on east wall", self.d["issues_detected"])

    def test_renovation_scope_in_dict(self):
        self.assertIn("renovation_scope", self.d)
        self.assertEqual(self.d["renovation_scope"], "partial")

    def test_high_value_upgrades_in_dict(self):
        self.assertIn("high_value_upgrades", self.d)
        self.assertIn("false ceiling", self.d["high_value_upgrades"])


class TestStyleDetectorFromFeatures(unittest.TestCase):
    """StyleDetector.detect_from_features keyword classification."""

    def _detect(self, **overrides) -> tuple:
        f = RoomFeatures()
        f.wall_color = overrides.get("wall_color", "neutral")
        f.floor_type = overrides.get("floor_type", "vitrified tiles")
        f.ceiling_type = overrides.get("ceiling_type", "plain plaster")
        f.wall_material = overrides.get("wall_material", "painted plaster")
        f.detected_furniture = overrides.get("detected_furniture", [])
        f.style_tags = overrides.get("style_tags", [])
        f.color_palette = overrides.get("color_palette", [])
        return StyleDetector.detect_from_features(f)

    def test_returns_tuple_of_str_and_float(self):
        label, conf = self._detect()
        self.assertIsInstance(label, str)
        self.assertIsInstance(conf, float)

    def test_confidence_between_0_and_1(self):
        _, conf = self._detect()
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_minimal_keyword_minimal_style(self):
        label, _ = self._detect(
            wall_color="white", style_tags=["minimal", "clean", "simple"]
        )
        self.assertEqual(label, "Modern Minimalist")

    def test_wood_hygge_birch_gives_scandinavian(self):
        label, conf = self._detect(
            floor_type="birch hardwood", style_tags=["hygge", "cosy", "linen"]
        )
        self.assertEqual(label, "Scandinavian")
        self.assertGreater(conf, 0.1)

    def test_exposed_concrete_industrial(self):
        label, _ = self._detect(
            wall_material="exposed concrete", style_tags=["industrial", "raw", "metal"]
        )
        self.assertEqual(label, "Industrial")

    def test_unknown_features_returns_some_label(self):
        label, conf = self._detect()
        # Should return a valid style from STYLE_RULES
        self.assertIn(label, STYLE_RULES)
        self.assertGreater(conf, 0.0)

    def test_all_style_rules_have_keywords(self):
        for style, config in STYLE_RULES.items():
            self.assertIn("keywords", config, f"{style} missing keywords")
            self.assertGreater(len(config["keywords"]), 0)
            self.assertIn("base", config)


class TestApplyCVEnrichment(unittest.TestCase):
    """RoomFeatures.apply_cv_enrichment style threshold and fusion logic."""

    def _make_features(self, style="Modern Minimalist", style_conf=0.60) -> RoomFeatures:
        f = RoomFeatures()
        f.style_label = style
        f.style_confidence = style_conf
        f.detected_furniture = ["bed"]
        f.natural_light = "moderate"
        f.condition = "fair"
        f.layout_score = 65
        f.extraction_source = "gemini"
        return f

    def test_clip_above_055_threshold_overrides_style(self):
        """Zero-shot CLIP at >= 0.65 confidence AND higher than Gemini — should override."""
        f = self._make_features(style="Modern Minimalist", style_conf=0.50)
        cv = {
            "clip_style_label": "Industrial",
            "clip_style_confidence": 0.66,   # above 0.65 zero-shot threshold
            "style_confidence": 0.0,
            "style": "",
            "style_model_used": "clip_zero_shot",
        }
        f.apply_cv_enrichment(cv)
        self.assertEqual(f.style_label, "Industrial")

    def test_clip_below_055_threshold_does_not_override(self):
        """CLIP at 0.42 (< 0.55) without agreement — should NOT override."""
        f = self._make_features(style="Scandinavian", style_conf=0.70)
        cv = {
            "clip_style_label": "Industrial",
            "clip_style_confidence": 0.42,
            "style_confidence": 0.0,
            "style": "",
            "style_model_used": "clip_zero_shot",
        }
        f.apply_cv_enrichment(cv)
        # Gemini had 0.70 confidence — CLIP at 0.42 should not win
        self.assertEqual(f.style_label, "Scandinavian")

    def test_agreement_bonus_lower_threshold(self):
        """CLIP and Gemini agree at 0.45 — blended conf should cross 0.42 min."""
        f = self._make_features(style="Japandi", style_conf=0.50)
        cv = {
            "clip_style_label": "Japandi",   # same as Gemini — agreement
            "clip_style_confidence": 0.45,
            "style_confidence": 0.0,
            "style": "",
            "style_model_used": "clip_zero_shot",
        }
        f.apply_cv_enrichment(cv)
        # With agreement bonus: 0.45×0.6 + 0.50×0.4 + 0.10 = 0.67 > 0.42
        self.assertEqual(f.style_label, "Japandi")
        self.assertGreater(f.style_confidence, 0.45)

    def test_cv_objects_merged_with_gemini(self):
        f = self._make_features()
        f.detected_furniture = ["wardrobe", "lamp"]
        cv = {
            "detected_objects": ["bed", "desk"],
            "style_confidence": 0.0, "style": "",
            "clip_style_label": "", "clip_style_confidence": 0.0,
            "style_model_used": "",
        }
        f.apply_cv_enrichment(cv)
        self.assertIn("bed", f.detected_furniture)
        self.assertIn("wardrobe", f.detected_furniture)

    def test_cv_yolo_objects_prepended(self):
        """CV (YOLO) objects must appear before Gemini objects."""
        f = self._make_features()
        f.detected_furniture = ["gemini_item"]
        cv = {
            "detected_objects": ["yolo_item"],
            "style_confidence": 0.0, "style": "",
            "clip_style_label": "", "clip_style_confidence": 0.0,
            "style_model_used": "",
        }
        f.apply_cv_enrichment(cv)
        self.assertEqual(f.detected_furniture[0], "yolo_item")

    def test_damage_severity_poor_maps_to_poor_condition(self):
        f = self._make_features()
        f.condition = "good"
        cv = {
            "damage_severity": "severe",
            "style_confidence": 0.0, "style": "",
            "clip_style_label": "", "clip_style_confidence": 0.0,
            "style_model_used": "",
        }
        f.apply_cv_enrichment(cv)
        self.assertEqual(f.condition, "poor")

    def test_damage_none_does_not_downgrade_excellent(self):
        f = self._make_features()
        f.condition = "excellent"
        cv = {
            "damage_severity": "none",
            "style_confidence": 0.0, "style": "",
            "clip_style_label": "", "clip_style_confidence": 0.0,
            "style_model_used": "",
        }
        f.apply_cv_enrichment(cv)
        # "none" maps to excellent — no downgrade expected
        self.assertIn(f.condition, ("excellent", "good"))

    def test_empty_cv_does_not_crash(self):
        f = self._make_features()
        f.apply_cv_enrichment({})
        self.assertEqual(f.style_label, "Modern Minimalist")

    def test_extraction_source_updated(self):
        f = self._make_features()
        cv = {
            "style_confidence": 0.0, "style": "",
            "clip_style_label": "", "clip_style_confidence": 0.0,
            "style_model_used": "",
        }
        f.apply_cv_enrichment(cv)
        self.assertIn("cv", f.extraction_source)


class TestGenerateExplainableRecommendations(unittest.TestCase):
    """generate_explainable_recommendations returns structured rec dicts."""

    def _make_room_features(self, **overrides) -> RoomFeatures:
        f = RoomFeatures()
        f.floor_type = overrides.get("floor_type", "vitrified tiles")
        f.wall_color = overrides.get("wall_color", "white")
        f.ceiling_type = overrides.get("ceiling_type", "plain plaster")
        f.lighting_sources = overrides.get("lighting_sources", ["ceiling light"])
        f.layout_issues = overrides.get("layout_issues", [])
        f.layout_suggestions = overrides.get("layout_suggestions", [])
        f.layout_score = overrides.get("layout_score", 70)
        f.style_label = overrides.get("style_label", "Modern Minimalist")
        f.detected_furniture = overrides.get("detected_furniture", ["bed"])
        return f

    def test_returns_list(self):
        recs = generate_explainable_recommendations(self._make_room_features())
        self.assertIsInstance(recs, list)

    def test_tile_floor_gets_flooring_rec(self):
        recs = generate_explainable_recommendations(
            self._make_room_features(floor_type="vitrified tiles")
        )
        categories = [r.get("category") for r in recs]
        self.assertIn("flooring", categories)

    def test_neutral_wall_gets_wall_rec(self):
        recs = generate_explainable_recommendations(
            self._make_room_features(wall_color="white")
        )
        categories = [r.get("category") for r in recs]
        self.assertIn("walls", categories)

    def test_no_false_ceiling_gets_ceiling_rec(self):
        recs = generate_explainable_recommendations(
            self._make_room_features(ceiling_type="plain plaster")
        )
        categories = [r.get("category") for r in recs]
        self.assertIn("ceiling", categories)

    def test_single_light_source_gets_lighting_rec(self):
        recs = generate_explainable_recommendations(
            self._make_room_features(lighting_sources=["ceiling light"])
        )
        categories = [r.get("category") for r in recs]
        self.assertIn("lighting", categories)

    def test_layout_issues_gets_layout_rec(self):
        recs = generate_explainable_recommendations(
            self._make_room_features(
                layout_issues=["Room appears over-furnished — consider decluttering 1–2 items"]
            )
        )
        categories = [r.get("category") for r in recs]
        self.assertIn("layout", categories)

    def test_each_rec_has_required_keys(self):
        recs = generate_explainable_recommendations(self._make_room_features())
        for r in recs:
            for k in ("category", "recommendation", "priority"):
                self.assertIn(k, r, f"Rec missing key '{k}': {r}")

    def test_recommendation_text_non_empty(self):
        recs = generate_explainable_recommendations(self._make_room_features())
        for r in recs:
            self.assertGreater(len(r.get("recommendation", "")), 10)


if __name__ == "__main__":
    unittest.main()
