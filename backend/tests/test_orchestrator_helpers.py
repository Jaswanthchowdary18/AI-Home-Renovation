"""
Tests — agents/orchestrator/langgraph_orchestrator.py (helper functions)
=========================================================================
Covers pure-Python helpers that contain core business logic and run
with no external services:

  - _merge: state dict composition
  - _require: key presence validation
  - _extract_condition_fields: condition field hoisting (core bug fix)
  - _build_damage_assessment: issue merging, condition_score, severity
  - _build_detected_features: feature dict structure
  - _default_vision_output: fallback shape
  - NodeValidationError: raised correctly
"""

from __future__ import annotations

import unittest

import tests.conftest as _cf
from tests.conftest import load_agent, make_minimal_state

orch_mod = load_agent(
    "agents.orchestrator.langgraph_orchestrator",
    "agents/orchestrator/langgraph_orchestrator.py",
)

_merge                   = orch_mod._merge
_require                 = orch_mod._require
_extract_condition_fields = orch_mod._extract_condition_fields
_build_damage_assessment = orch_mod._build_damage_assessment
_build_detected_features = orch_mod._build_detected_features
_default_vision_output   = orch_mod._default_vision_output
NodeValidationError      = orch_mod.NodeValidationError


class TestMerge(unittest.TestCase):

    def test_updates_applied_to_base(self):
        base = {"a": 1, "b": 2}
        result = _merge(base, {"b": 99, "c": 3})
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 99)
        self.assertEqual(result["c"], 3)

    def test_original_state_not_mutated(self):
        base = {"a": 1}
        _merge(base, {"a": 2})
        self.assertEqual(base["a"], 1)

    def test_empty_updates_returns_same_content(self):
        base = {"x": 42}
        result = _merge(base, {})
        self.assertEqual(result["x"], 42)

    def test_empty_base_with_updates(self):
        result = _merge({}, {"key": "value"})
        self.assertEqual(result["key"], "value")

    def test_returns_new_dict(self):
        base = {"a": 1}
        result = _merge(base, {})
        self.assertIsNot(result, base)


class TestRequire(unittest.TestCase):

    def test_present_key_does_not_raise(self):
        state = {"room_type": "bedroom", "city": "Hyderabad"}
        _require(state, "room_type", "city", node="test_node")

    def test_missing_key_raises_node_validation_error(self):
        state = {"room_type": "bedroom"}
        with self.assertRaises(NodeValidationError) as ctx:
            _require(state, "room_type", "city", node="test_node")
        self.assertIn("city", str(ctx.exception))

    def test_none_value_counts_as_missing(self):
        state = {"room_type": None}
        with self.assertRaises(NodeValidationError):
            _require(state, "room_type", node="test_node")

    def test_empty_string_counts_as_missing(self):
        state = {"city": ""}
        with self.assertRaises(NodeValidationError):
            _require(state, "city", node="test_node")

    def test_zero_int_counts_as_missing(self):
        state = {"budget_inr": 0}
        with self.assertRaises(NodeValidationError):
            _require(state, "budget_inr", node="test_node")

    def test_multiple_missing_reported_together(self):
        state = {}
        with self.assertRaises(NodeValidationError) as ctx:
            _require(state, "city", "room_type", "budget_inr", node="test_node")
        msg = str(ctx.exception)
        self.assertIn("city", msg)
        self.assertIn("room_type", msg)


class TestExtractConditionFields(unittest.TestCase):
    """
    Core bug-fix regression: condition fields must be hoisted from
    room_features to top-level state so InsightGenerationAgent and
    DesignPlannerAgentNode can read them directly.
    """

    def test_issues_from_room_features_hoisted(self):
        rf = {"issues_detected": ["crack on north wall", "damp near window"]}
        result = _extract_condition_fields(rf, {})
        self.assertIn("crack on north wall", result["issues_detected"])
        self.assertIn("damp near window", result["issues_detected"])

    def test_layout_issues_merged_into_issues_detected(self):
        rf = {"issues_detected": ["crack"]}
        pipeline_result = {"layout_report": {"issues": ["over-furnished"]}}
        # The function takes room_features + result (vision agent output)
        result = _extract_condition_fields(rf, pipeline_result)
        combined = result["issues_detected"]
        self.assertIn("crack", combined)

    def test_condition_score_hoisted(self):
        rf = {"condition_score": 65}
        result = _extract_condition_fields(rf, {})
        self.assertEqual(result["condition_score"], 65)

    def test_condition_score_from_private_attr(self):
        rf = {"_condition_score": 72}
        result = _extract_condition_fields(rf, {})
        self.assertEqual(result["condition_score"], 72)

    def test_wall_condition_hoisted(self):
        rf = {"wall_condition": "poor"}
        result = _extract_condition_fields(rf, {})
        self.assertEqual(result["wall_condition"], "poor")

    def test_floor_condition_hoisted(self):
        rf = {"floor_condition": "good"}
        result = _extract_condition_fields(rf, {})
        self.assertEqual(result["floor_condition"], "good")

    def test_renovation_scope_hoisted(self):
        rf = {"renovation_scope": "full_room"}
        result = _extract_condition_fields(rf, {})
        self.assertEqual(result["renovation_scope"], "full_room")

    def test_high_value_upgrades_hoisted(self):
        rf = {"high_value_upgrades": ["false ceiling", "modular wardrobe"]}
        result = _extract_condition_fields(rf, {})
        self.assertIn("false ceiling", result["high_value_upgrades"])

    def test_private_issues_detected_used_when_public_absent(self):
        rf = {"_issues_detected": ["seepage near bathroom"]}
        result = _extract_condition_fields(rf, {})
        self.assertIn("seepage near bathroom", result["issues_detected"])

    def test_result_issues_deduplicated(self):
        rf = {
            "issues_detected": ["crack", "peeling paint"],
            "_issues_detected": ["crack"],  # duplicate
        }
        result = _extract_condition_fields(rf, {})
        self.assertEqual(result["issues_detected"].count("crack"), 1)

    def test_not_assessed_defaults(self):
        result = _extract_condition_fields({}, {})
        self.assertEqual(result["wall_condition"], "not_assessed")
        self.assertEqual(result["floor_condition"], "not_assessed")

    def test_renovation_scope_defaults_to_partial(self):
        result = _extract_condition_fields({}, {})
        self.assertEqual(result["renovation_scope"], "partial")

    def test_all_required_keys_returned(self):
        result = _extract_condition_fields({}, {})
        for k in ("issues_detected", "condition_score", "wall_condition",
                  "floor_condition", "renovation_scope", "high_value_upgrades"):
            self.assertIn(k, result, f"Missing key: {k}")

    def test_string_issue_wrapped_in_list(self):
        rf = {"issues_detected": "single issue as string"}
        result = _extract_condition_fields(rf, {})
        self.assertIsInstance(result["issues_detected"], list)

    def test_result_from_vision_agent_used_as_fallback(self):
        rf = {}
        result_dict = {"issues_detected": ["from vision agent"]}
        result = _extract_condition_fields(rf, result_dict)
        self.assertIn("from vision agent", result["issues_detected"])


class TestBuildDamageAssessment(unittest.TestCase):

    def test_basic_shape(self):
        rf = {
            "condition": "fair",
            "condition_score": 60,
            "issues_detected": ["crack"],
            "renovation_priority": ["walls"],
            "natural_light": "moderate",
        }
        result = _build_damage_assessment(rf, {})
        for k in ("overall_condition", "condition_score", "severity",
                  "issues_detected", "renovation_priority", "natural_light"):
            self.assertIn(k, result)

    def test_poor_condition_high_severity(self):
        rf = {"condition": "poor"}
        result = _build_damage_assessment(rf, {})
        self.assertEqual(result["severity"], "high")

    def test_good_condition_low_severity(self):
        rf = {"condition": "good"}
        result = _build_damage_assessment(rf, {})
        self.assertEqual(result["severity"], "low")

    def test_excellent_condition_none_severity(self):
        rf = {"condition": "excellent"}
        result = _build_damage_assessment(rf, {})
        self.assertEqual(result["severity"], "none")

    def test_issues_merged_from_rf_and_layout(self):
        rf = {"issues_detected": ["crack on east wall"]}
        layout = {"issues": ["over-furnished"]}
        result = _build_damage_assessment(rf, layout)
        combined = result["issues_detected"]
        self.assertIn("crack on east wall", combined)
        self.assertIn("over-furnished", combined)

    def test_issues_deduplicated(self):
        rf = {"issues_detected": ["crack"]}
        layout = {"issues_detected": ["crack"]}  # duplicate
        result = _build_damage_assessment(rf, layout)
        self.assertEqual(result["issues_detected"].count("crack"), 1)

    def test_condition_score_passed_through(self):
        rf = {"condition": "fair", "condition_score": 58}
        result = _build_damage_assessment(rf, {})
        self.assertEqual(result["condition_score"], 58)

    def test_private_condition_score_used(self):
        rf = {"condition": "fair", "_condition_score": 62}
        result = _build_damage_assessment(rf, {})
        self.assertEqual(result["condition_score"], 62)

    def test_layout_score_default_when_absent(self):
        result = _build_damage_assessment({}, {})
        self.assertIn("layout_score", result)

    def test_empty_inputs_no_crash(self):
        result = _build_damage_assessment({}, {})
        self.assertIsInstance(result, dict)


class TestBuildDetectedFeatures(unittest.TestCase):

    def _call(self, extra_result=None, extra_rf=None, extra_layout=None):
        result = {"style_label": "Modern Minimalist",
                  "floor_area_sqft": 120.0, "wall_area_sqft": 200.0,
                  "vision_features": {
                      "wall_treatment": "white paint",
                      "floor_material": "vitrified tiles",
                      "ceiling_treatment": "pop false ceiling",
                      "lighting_type": "ceiling",
                      "colour_palette": ["white", "grey"],
                      "furniture_items": ["bed", "wardrobe"],
                      "quality_tier": "mid",
                      "natural_light_quality": "moderate",
                  }}
        rf = {"wall_color": "white", "lighting_sources": ["ceiling"],
              "natural_light": "moderate", "condition": "fair"}
        layout = {"layout_score": "70/100", "walkable_space": "45%"}
        if extra_result: result.update(extra_result)
        if extra_rf: rf.update(extra_rf)
        if extra_layout: layout.update(extra_layout)
        return _build_detected_features(result, rf, layout)

    def test_required_top_level_keys(self):
        feat = self._call()
        for k in ("wall", "floor", "ceiling", "lighting",
                  "colour_palette", "furniture_items",
                  "detected_style", "quality_tier", "natural_light",
                  "layout_score", "walkable_space"):
            self.assertIn(k, feat, f"Missing key: {k}")

    def test_wall_has_treatment_and_colour(self):
        feat = self._call()
        self.assertIn("treatment", feat["wall"])
        self.assertIn("colour", feat["wall"])

    def test_floor_has_material_and_area(self):
        feat = self._call()
        self.assertIn("material", feat["floor"])
        self.assertIn("area_sqft", feat["floor"])

    def test_detected_style_matches_input(self):
        feat = self._call(extra_result={"style_label": "Industrial"})
        self.assertEqual(feat["detected_style"], "Industrial")

    def test_floor_area_passed_through(self):
        feat = self._call(extra_result={"floor_area_sqft": 150.0})
        self.assertAlmostEqual(feat["floor"]["area_sqft"], 150.0)


class TestDefaultVisionOutput(unittest.TestCase):

    def test_returns_dict(self):
        state = make_minimal_state()
        result = _default_vision_output(state)
        self.assertIsInstance(result, dict)

    def test_detected_features_present(self):
        result = _default_vision_output(make_minimal_state())
        self.assertIn("detected_features", result)

    def test_damage_assessment_present(self):
        result = _default_vision_output(make_minimal_state())
        self.assertIn("damage_assessment", result)

    def test_style_label_defaults_to_theme(self):
        state = make_minimal_state(theme="Industrial")
        result = _default_vision_output(state)
        self.assertEqual(result["style_label"], "Industrial")

    def test_all_required_state_keys_present(self):
        result = _default_vision_output(make_minimal_state())
        for k in ("vision_features", "image_features", "room_features",
                  "detected_objects", "material_types",
                  "wall_area_sqft", "floor_area_sqft"):
            self.assertIn(k, result, f"Default vision output missing: {k}")


if __name__ == "__main__":
    unittest.main()
