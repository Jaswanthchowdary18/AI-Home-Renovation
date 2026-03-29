"""
Tests — agents/design_planner_node.py
=======================================
Covers:
  - _build_image_specific_actions: required keys, action count, triggers
  - _synthetic_style_examples: variety (not all same style), required keys
  - _get_layout_suggestions: object-based triggers, max 6 items
  - STYLE_FLOOR_PREFERENCE / STYLE_WALL_PREFERENCE / STYLE_CEILING_PREFERENCE:
      completeness and non-empty values
  - OBJECT_RENOVATION_ACTIONS: all keys map to non-empty action lists
  - LIGHTING_UPGRADES: all lighting conditions have upgrade suggestions
"""

from __future__ import annotations

import unittest

import tests.conftest as _cf
from tests.conftest import load_agent, make_minimal_state

dp_mod = load_agent("agents.design_planner_node", "agents/design_planner_node.py")

DesignPlannerAgentNode  = dp_mod.DesignPlannerAgentNode
STYLE_FLOOR_PREFERENCE  = dp_mod.STYLE_FLOOR_PREFERENCE
STYLE_WALL_PREFERENCE   = dp_mod.STYLE_WALL_PREFERENCE
STYLE_CEILING_PREFERENCE = dp_mod.STYLE_CEILING_PREFERENCE
OBJECT_RENOVATION_ACTIONS = dp_mod.OBJECT_RENOVATION_ACTIONS
LIGHTING_UPGRADES       = dp_mod.LIGHTING_UPGRADES


# ─────────────────────────────────────────────────────────────────────────────
# Lookup table integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestStylePreferenceTables(unittest.TestCase):

    def test_floor_preferences_all_non_empty(self):
        for style, pref in STYLE_FLOOR_PREFERENCE.items():
            self.assertIsInstance(pref, str)
            self.assertGreater(len(pref), 5, f"Empty floor pref for {style}")

    def test_wall_preferences_all_non_empty(self):
        for style, pref in STYLE_WALL_PREFERENCE.items():
            self.assertIsInstance(pref, str)
            self.assertGreater(len(pref), 5)

    def test_ceiling_preferences_all_non_empty(self):
        for style, pref in STYLE_CEILING_PREFERENCE.items():
            self.assertIsInstance(pref, str)
            self.assertGreater(len(pref), 5)

    def test_modern_minimalist_in_all_tables(self):
        for table in (STYLE_FLOOR_PREFERENCE, STYLE_WALL_PREFERENCE,
                      STYLE_CEILING_PREFERENCE):
            self.assertIn("Modern Minimalist", table)

    def test_styles_consistent_across_tables(self):
        """All styles with floor prefs should also have wall prefs."""
        for style in STYLE_FLOOR_PREFERENCE:
            self.assertIn(style, STYLE_WALL_PREFERENCE,
                          f"{style} in floor prefs but not wall prefs")

    def test_floor_pref_mentions_flooring_material(self):
        """Floor preferences should describe a flooring material."""
        flooring_keywords = ["tile", "wood", "bamboo", "marble", "concrete",
                             "stone", "terrazzo", "plank"]
        for style, pref in STYLE_FLOOR_PREFERENCE.items():
            pref_lower = pref.lower()
            self.assertTrue(
                any(kw in pref_lower for kw in flooring_keywords),
                f"{style} floor pref doesn't mention a material: {pref}",
            )


class TestObjectRenovationActions(unittest.TestCase):

    def test_sofa_has_actions(self):
        self.assertIn("sofa", OBJECT_RENOVATION_ACTIONS)
        self.assertGreater(len(OBJECT_RENOVATION_ACTIONS["sofa"]), 0)

    def test_bed_has_actions(self):
        self.assertIn("bed", OBJECT_RENOVATION_ACTIONS)
        self.assertGreater(len(OBJECT_RENOVATION_ACTIONS["bed"]), 0)

    def test_all_actions_are_non_empty_strings(self):
        for obj, actions in OBJECT_RENOVATION_ACTIONS.items():
            for action in actions:
                self.assertIsInstance(action, str)
                self.assertGreater(len(action), 10,
                                   f"Short action for {obj}: '{action}'")

    def test_tv_has_actions(self):
        self.assertIn("television", OBJECT_RENOVATION_ACTIONS)


class TestLightingUpgrades(unittest.TestCase):

    def test_dim_has_upgrades(self):
        self.assertIn("dim", LIGHTING_UPGRADES)
        self.assertGreater(len(LIGHTING_UPGRADES["dim"]), 0)

    def test_artificial_has_upgrades(self):
        self.assertIn("artificial", LIGHTING_UPGRADES)

    def test_all_upgrade_strings_non_empty(self):
        for condition, upgrades in LIGHTING_UPGRADES.items():
            for upgrade in upgrades:
                self.assertIsInstance(upgrade, str)
                self.assertGreater(len(upgrade), 10)


# ─────────────────────────────────────────────────────────────────────────────
# _build_image_specific_actions
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildImageSpecificActions(unittest.TestCase):

    def setUp(self):
        self.agent = DesignPlannerAgentNode()

    def _call(self, **kw) -> list:
        defaults = dict(
            detected_objects=["bed", "wardrobe"],
            detected_style="Modern Minimalist",
            detected_lighting="moderate",
            detected_materials=["vitrified_tile"],
            room_type="bedroom",
            budget_tier="mid",
        )
        defaults.update(kw)
        return self.agent._build_image_specific_actions(**defaults)

    def test_returns_list(self):
        self.assertIsInstance(self._call(), list)

    def test_max_12_actions(self):
        actions = self._call(
            detected_objects=["sofa", "bed", "wardrobe", "tv", "desk",
                              "mirror", "plant", "coffee table", "chair",
                              "refrigerator", "sink", "dining table"],
        )
        self.assertLessEqual(len(actions), 12)

    def test_each_action_has_required_keys(self):
        actions = self._call()
        for a in actions:
            for k in ("action", "trigger", "category", "priority", "grounding"):
                self.assertIn(k, a, f"Action missing key '{k}': {a}")

    def test_action_text_non_empty(self):
        actions = self._call()
        for a in actions:
            self.assertGreater(len(a["action"]), 10)

    def test_flooring_action_present_for_any_style(self):
        """Every style should produce a flooring action."""
        actions = self._call(detected_style="Industrial")
        categories = [a["category"] for a in actions]
        self.assertIn("flooring", categories)

    def test_wall_action_present(self):
        actions = self._call()
        categories = [a["category"] for a in actions]
        self.assertIn("walls", categories)

    def test_ceiling_action_present(self):
        actions = self._call()
        categories = [a["category"] for a in actions]
        self.assertIn("ceiling", categories)

    def test_lighting_action_for_dim_lighting(self):
        actions = self._call(detected_lighting="dim")
        categories = [a["category"] for a in actions]
        self.assertIn("lighting", categories)

    def test_object_specific_action_for_detected_sofa(self):
        actions = self._call(detected_objects=["sofa"])
        triggers = [a["trigger"].lower() for a in actions]
        self.assertTrue(any("sofa" in t for t in triggers))

    def test_grounding_values_are_valid(self):
        valid_groundings = {
            "yolo_detection", "clip_style_detection",
            "clip_lighting_detection", "material_inference",
        }
        actions = self._call()
        for a in actions:
            self.assertIn(a["grounding"], valid_groundings,
                          f"Unknown grounding: {a['grounding']}")

    def test_priority_values_are_valid(self):
        valid_priorities = {"high", "medium", "low"}
        actions = self._call()
        for a in actions:
            self.assertIn(a["priority"], valid_priorities)

    def test_floor_action_references_detected_style(self):
        actions = self._call(detected_style="Scandinavian")
        flooring_actions = [a for a in actions if a["category"] == "flooring"]
        self.assertGreater(len(flooring_actions), 0)
        self.assertIn("Scandinavian", flooring_actions[0]["action"])

    def test_wood_material_triggers_wood_action(self):
        actions = self._call(detected_materials=["wood"])
        material_actions = [a for a in actions if a["category"] == "materials"]
        self.assertGreater(len(material_actions), 0)

    def test_marble_material_triggers_polish_action(self):
        actions = self._call(detected_materials=["marble"])
        material_actions = [a for a in actions if a["category"] == "materials"]
        # Marble should suggest polishing/sealing
        combined = " ".join(a["action"].lower() for a in material_actions)
        self.assertIn("marble", combined)


# ─────────────────────────────────────────────────────────────────────────────
# _synthetic_style_examples
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticStyleExamples(unittest.TestCase):

    def setUp(self):
        self.agent = DesignPlannerAgentNode()

    def _call(self, style="Modern Minimalist", room_type="bedroom") -> list:
        return self.agent._synthetic_style_examples(style, room_type)

    def test_returns_list(self):
        self.assertIsInstance(self._call(), list)

    def test_not_empty(self):
        self.assertGreater(len(self._call()), 0)

    def test_variety_of_styles_no_all_same(self):
        """Core fix: must not return all identical style labels."""
        examples = self._call(style="Modern Minimalist")
        styles = [e["style"] for e in examples]
        unique_styles = set(styles)
        self.assertGreater(
            len(unique_styles), 1,
            f"All examples are the same style: {styles}",
        )

    def test_primary_style_present(self):
        examples = self._call(style="Modern Minimalist")
        styles = [e["style"] for e in examples]
        self.assertIn("Modern Minimalist", styles)

    def test_each_example_has_required_keys(self):
        for ex in self._call():
            for k in ("room_type", "style", "materials", "source"):
                self.assertIn(k, ex, f"Example missing key '{k}': {ex}")

    def test_materials_is_list(self):
        for ex in self._call():
            self.assertIsInstance(ex["materials"], list)

    def test_room_type_matches_input(self):
        examples = self._call(room_type="kitchen")
        for ex in examples:
            self.assertEqual(ex["room_type"], "kitchen")

    def test_scandinavian_has_related_styles(self):
        examples = self._call(style="Scandinavian")
        styles = [e["style"] for e in examples]
        related = {"Modern Minimalist", "Japandi"}
        self.assertTrue(
            any(s in related for s in styles),
            f"Scandinavian examples should include related styles: {styles}",
        )

    def test_contemporary_indian_has_related(self):
        examples = self._call(style="Contemporary Indian")
        styles = [e["style"] for e in examples]
        # Should include Traditional Indian or Modern Minimalist
        related = {"Traditional Indian", "Modern Minimalist"}
        self.assertTrue(any(s in related for s in styles))

    def test_max_5_examples(self):
        self.assertLessEqual(len(self._call()), 5)


# ─────────────────────────────────────────────────────────────────────────────
# _get_layout_suggestions
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLayoutSuggestions(unittest.TestCase):

    def setUp(self):
        self.agent = DesignPlannerAgentNode()

    def _call(self, state=None, detected_objects=None) -> list:
        s = state or make_minimal_state()
        return self.agent._get_layout_suggestions(s, detected_objects or [])

    def test_returns_list(self):
        self.assertIsInstance(self._call(), list)

    def test_max_6_suggestions(self):
        self.assertLessEqual(len(self._call()), 6)

    def test_sofa_without_coffee_table_triggers_suggestion(self):
        suggestions = self._call(detected_objects=["sofa"])
        combined = " ".join(s.lower() for s in suggestions)
        self.assertIn("coffee table", combined)

    def test_sparse_bedroom_triggers_suggestion(self):
        """Bed with only 2 detected objects → suggests adding bedside tables."""
        suggestions = self._call(detected_objects=["bed", "lamp"])
        combined = " ".join(s.lower() for s in suggestions)
        self.assertIn("bedside", combined)

    def test_over_furnished_room_triggers_declutter(self):
        objects = ["bed", "sofa", "desk", "wardrobe", "tv", "bookshelf",
                   "lamp", "chair", "nightstand"]
        suggestions = self._call(detected_objects=objects)
        combined = " ".join(s.lower() for s in suggestions)
        self.assertIn("over-furnished", combined)

    def test_layout_report_issues_included(self):
        state = make_minimal_state()
        state["layout_report"]["issues"] = ["No clear pathway to window"]
        suggestions = self._call(state=state)
        combined = " ".join(suggestions)
        self.assertIn("No clear pathway", combined)

    def test_resale_goal_adds_suggestion(self):
        state = make_minimal_state()
        state["user_goals"] = {"primary_goal": "maximise_resale_value"}
        suggestions = self._call(state=state)
        combined = " ".join(s.lower() for s in suggestions)
        self.assertIn("kitchen", combined)

    def test_rental_goal_adds_neutral_suggestion(self):
        state = make_minimal_state()
        state["user_goals"] = {"primary_goal": "maximise_rental_yield"}
        suggestions = self._call(state=state)
        combined = " ".join(s.lower() for s in suggestions)
        self.assertIn("neutral", combined)

    def test_luxury_goal_adds_lighting_suggestion(self):
        state = make_minimal_state()
        state["user_goals"] = {"primary_goal": "luxury_upgrade"}
        suggestions = self._call(state=state)
        combined = " ".join(s.lower() for s in suggestions)
        self.assertIn("lighting", combined)


if __name__ == "__main__":
    unittest.main()
