"""
Tests — agents/graph_state.py
===============================
Covers:
  - ARKENGraphState TypedDict field completeness
  - make_initial_state: required inputs, field defaults, image encoding
  - extract_contract_state: 8 canonical keys present
  - make_initial_state: validation (missing city raises, negative budget raises)
"""

from __future__ import annotations

import base64
import unittest

import tests.conftest as _cf
from tests.conftest import load_agent

gs_mod = load_agent("agents.graph_state", "agents/graph_state.py")

ARKENGraphState      = gs_mod.ARKENGraphState
make_initial_state   = gs_mod.make_initial_state
extract_contract_state = gs_mod.extract_contract_state


class TestMakeInitialState(unittest.TestCase):

    def _make(self, **overrides):
        defaults = dict(
            project_id="proj-001",
            user_id="user-001",
            city="Hyderabad",
            budget_inr=750_000,
            room_type="bedroom",
            budget_tier="mid",
        )
        defaults.update(overrides)
        return make_initial_state(**defaults)

    def test_basic_creation_succeeds(self):
        state = self._make()
        self.assertIsInstance(state, dict)

    def test_project_id_stored(self):
        state = self._make(project_id="my-proj")
        self.assertEqual(state["project_id"], "my-proj")

    def test_city_stored(self):
        state = self._make(city="Mumbai")
        self.assertEqual(state["city"], "Mumbai")

    def test_budget_inr_stored(self):
        state = self._make(budget_inr=1_000_000)
        self.assertEqual(state["budget_inr"], 1_000_000)

    def test_room_type_stored(self):
        state = self._make(room_type="kitchen")
        self.assertEqual(state["room_type"], "kitchen")

    def test_errors_initialized_as_empty_list(self):
        state = self._make()
        self.assertEqual(state["errors"], [])

    def test_completed_agents_initialized_as_empty_list(self):
        state = self._make()
        self.assertEqual(state["completed_agents"], [])

    def test_agent_timings_initialized_as_empty_dict(self):
        state = self._make()
        self.assertEqual(state["agent_timings"], {})

    def test_condition_fields_default_not_assessed(self):
        state = self._make()
        self.assertIsNone(state["condition_score"])
        self.assertEqual(state["wall_condition"], "not_assessed")
        self.assertEqual(state["floor_condition"], "not_assessed")

    def test_issues_detected_defaults_empty(self):
        state = self._make()
        self.assertEqual(state["issues_detected"], [])

    def test_renovation_scope_defaults_not_assessed(self):
        state = self._make()
        self.assertEqual(state["renovation_scope"], "not_assessed")

    def test_within_budget_defaults_true(self):
        state = self._make()
        self.assertTrue(state["within_budget"])

    def test_image_bytes_to_b64_auto_encoding(self):
        raw = b"\x89PNG\r\n\x1a\n"
        state = self._make(image_bytes=raw)
        self.assertEqual(state["original_image_bytes"], raw)
        expected_b64 = base64.b64encode(raw).decode()
        self.assertEqual(state["original_image_b64"], expected_b64)

    def test_image_b64_to_bytes_auto_decoding(self):
        raw = b"test image data"
        b64 = base64.b64encode(raw).decode()
        state = self._make(image_b64=b64)
        self.assertEqual(state["original_image_bytes"], raw)

    def test_theme_stored(self):
        state = self._make(theme="Industrial")
        self.assertEqual(state["theme"], "Industrial")

    def test_missing_city_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            make_initial_state(
                project_id="p", user_id="u",
                city="", budget_inr=750_000,
            )

    def test_negative_budget_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            make_initial_state(
                project_id="p", user_id="u",
                city="Hyderabad", budget_inr=-1,
            )

    def test_contract_keys_initialized_empty(self):
        state = self._make()
        for k in ("design_plan", "cost_estimate", "roi_prediction",
                  "insights", "final_report", "damage_assessment"):
            self.assertIn(k, state)
            self.assertIsInstance(state[k], dict)

    def test_pipeline_version_set(self):
        state = self._make()
        self.assertIn("pipeline_version", state)
        self.assertIsInstance(state["pipeline_version"], str)


class TestExtractContractState(unittest.TestCase):

    def _make_state(self, **overrides):
        base = {
            "user_goal": "personal_comfort",
            "original_image_b64": "dGVzdA==",
            "detected_features": {"floor": {"material": "marble"}},
            "retrieved_knowledge": [{"doc_id": "doc1"}],
            "design_plan": {"line_items": []},
            "cost_estimate": {"total_inr": 750_000},
            "roi_prediction": {"roi_pct": 12.0},
            "insights": {"summary_headline": "Test"},
            "final_report": {"report_id": "r1"},
            "renovation_report": {"report_id": "r1"},
        }
        base.update(overrides)
        return base

    def test_returns_dict(self):
        result = extract_contract_state(self._make_state())
        self.assertIsInstance(result, dict)

    def test_all_8_contract_keys_present(self):
        result = extract_contract_state(self._make_state())
        required = [
            "user_goal", "uploaded_images", "detected_features",
            "retrieved_knowledge", "design_plan", "cost_estimate",
            "roi_prediction", "insights",
        ]
        for k in required:
            self.assertIn(k, result, f"Contract key missing: {k}")

    def test_uploaded_images_is_list(self):
        result = extract_contract_state(self._make_state())
        self.assertIsInstance(result["uploaded_images"], list)

    def test_final_report_included(self):
        result = extract_contract_state(self._make_state())
        self.assertIn("final_report", result)

    def test_user_goal_passed_through(self):
        result = extract_contract_state(self._make_state(user_goal="luxury_upgrade"))
        self.assertEqual(result["user_goal"], "luxury_upgrade")

    def test_missing_keys_default_to_empty(self):
        result = extract_contract_state({})
        self.assertEqual(result["user_goal"], "")
        self.assertIsInstance(result["detected_features"], dict)
        self.assertIsInstance(result["retrieved_knowledge"], list)

    def test_renovation_report_used_as_final_report_fallback(self):
        state = self._make_state()
        del state["final_report"]
        result = extract_contract_state(state)
        self.assertIn("final_report", result)

    def test_does_not_include_internal_keys(self):
        result = extract_contract_state(self._make_state())
        for internal_key in ("cv_features", "agent_timings", "completed_agents",
                             "errors", "original_image_bytes"):
            self.assertNotIn(internal_key, result,
                             f"Internal key leaked into contract: {internal_key}")


if __name__ == "__main__":
    unittest.main()
