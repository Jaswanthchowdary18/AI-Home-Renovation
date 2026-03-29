"""
Tests — agents/roi_agent_node.py :: _build_vastu_analysis
===========================================================
Covers:
  - Pancha Bhuta balance: all 5 elements present in output
  - Zone map: non-empty, relevant zones for each room type
  - Renovation Vastu rules: room-type-specific do/don't lists
  - Colour analysis: score assigned, known colours detected
  - Colour remedy: fires for red/black/dark walls
  - Object findings: mirror, bed, sofa, plant, TV, wardrobe, desk
  - Lighting energy: poor/dim fires deficiency; good fires positive
  - Space energy: cluttered fires; open fires positive
  - Overall score: within [20, 95], increases with good light/colours
  - Top improvements: at least 3 actionable items
  - Vastu market data: required keys present
  - Image-grounded flag: True when objects detected
  - Edge cases: empty state, unknown room type
"""

from __future__ import annotations

import unittest

import tests.conftest as _cf
from tests.conftest import load_agent, make_roi_state

roi_node_mod = load_agent("agents.roi_agent_node", "agents/roi_agent_node.py")
ROIAgentNode = roi_node_mod.ROIAgentNode


def _vastu(state: dict) -> dict:
    agent = ROIAgentNode.__new__(ROIAgentNode)
    return agent._build_vastu_analysis(state, state.get("room_type", "bedroom"))


class TestPanchaBhutaBalance(unittest.TestCase):

    def test_returns_5_elements(self):
        result = _vastu(make_roi_state())
        pb = result["pancha_bhuta_balance"]
        self.assertEqual(len(pb), 5)

    def test_all_elements_named(self):
        pb = _vastu(make_roi_state())["pancha_bhuta_balance"]
        names = [el["element"] for el in pb]
        for expected in ("Prithvi", "Jal", "Agni", "Vayu", "Akasha"):
            self.assertTrue(
                any(expected in n for n in names),
                f"Element '{expected}' missing from Pancha Bhuta: {names}",
            )

    def test_each_element_has_required_keys(self):
        pb = _vastu(make_roi_state())["pancha_bhuta_balance"]
        for el in pb:
            for k in ("element", "status", "ideal_zone"):
                self.assertIn(k, el, f"Element missing key '{k}': {el}")

    def test_status_is_present_or_weak(self):
        pb = _vastu(make_roi_state())["pancha_bhuta_balance"]
        for el in pb:
            self.assertIn(el["status"], ("present", "weak"))

    def test_weak_element_has_deficiency_fix(self):
        pb = _vastu(make_roi_state())["pancha_bhuta_balance"]
        for el in pb:
            if el["status"] == "weak":
                self.assertIsNotNone(
                    el.get("deficiency_fix"),
                    f"Weak element '{el['element']}' has no deficiency_fix",
                )
                self.assertIsInstance(el["deficiency_fix"], str)
                self.assertGreater(len(el["deficiency_fix"]), 10)

    def test_earth_present_when_stone_floor(self):
        state = make_roi_state()
        state["room_features"]["floor_type"] = "marble"
        pb = _vastu(state)["pancha_bhuta_balance"]
        earth = next(e for e in pb if "Prithvi" in e["element"])
        self.assertEqual(earth["status"], "present")

    def test_fire_present_when_red_or_orange_color(self):
        state = make_roi_state()
        state["room_features"]["color_palette"] = ["red", "orange"]
        state["room_features"]["wall_color"] = "orange"
        pb = _vastu(state)["pancha_bhuta_balance"]
        fire = next(e for e in pb if "Agni" in e["element"])
        self.assertEqual(fire["status"], "present")


class TestVastuZoneMap(unittest.TestCase):

    def test_zone_map_non_empty(self):
        zm = _vastu(make_roi_state())["vastu_zone_map"]
        self.assertGreater(len(zm), 0)

    def test_zone_map_has_max_4_zones(self):
        zm = _vastu(make_roi_state())["vastu_zone_map"]
        self.assertLessEqual(len(zm), 4)

    def test_each_zone_has_required_keys(self):
        zm = _vastu(make_roi_state())["vastu_zone_map"]
        for z in zm:
            for k in ("zone", "ideal_for", "avoid", "renovation_tip"):
                self.assertIn(k, z, f"Zone missing key '{k}': {z}")

    def test_northeast_zone_always_included(self):
        """North-east is always relevant for all room types."""
        zm = _vastu(make_roi_state())["vastu_zone_map"]
        zones = [z["zone"] for z in zm]
        self.assertTrue(
            any("north-east" in z.lower() or "northeast" in z.lower() for z in zones),
            f"North-east zone missing: {zones}",
        )

    def test_kitchen_includes_southeast(self):
        state = make_roi_state(room_type="kitchen")
        zm = _vastu(state)["vastu_zone_map"]
        zones = [z["zone"].lower() for z in zm]
        self.assertTrue(any("south-east" in z for z in zones))

    def test_bedroom_includes_southwest(self):
        state = make_roi_state(room_type="bedroom")
        zm = _vastu(state)["vastu_zone_map"]
        zones = [z["zone"].lower() for z in zm]
        self.assertTrue(any("south" in z for z in zones))


class TestRenovationVastuRules(unittest.TestCase):

    def test_rules_present_for_bedroom(self):
        rules = _vastu(make_roi_state(room_type="bedroom"))["renovation_vastu_rules"]
        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)

    def test_rules_present_for_kitchen(self):
        rules = _vastu(make_roi_state(room_type="kitchen"))["renovation_vastu_rules"]
        self.assertGreater(len(rules), 0)

    def test_rules_present_for_living_room(self):
        rules = _vastu(make_roi_state(room_type="living_room"))["renovation_vastu_rules"]
        self.assertGreater(len(rules), 0)

    def test_each_rule_has_do_and_dont(self):
        rules = _vastu(make_roi_state(room_type="bedroom"))["renovation_vastu_rules"]
        for r in rules:
            self.assertIn("do", r, f"Rule missing 'do': {r}")
            self.assertIn("dont", r, f"Rule missing 'dont': {r}")
            self.assertGreater(len(r["do"]), 10)
            self.assertGreater(len(r["dont"]), 10)

    def test_each_rule_has_rule_label(self):
        rules = _vastu(make_roi_state())["renovation_vastu_rules"]
        for r in rules:
            self.assertIn("rule", r)


class TestColourAnalysis(unittest.TestCase):

    def test_colour_analysis_present(self):
        ca = _vastu(make_roi_state())["colour_analysis"]
        self.assertIsInstance(ca, dict)

    def test_colour_analysis_has_required_keys(self):
        ca = _vastu(make_roi_state())["colour_analysis"]
        for k in ("detected_color", "vastu_score", "element", "explanation"):
            self.assertIn(k, ca, f"colour_analysis missing key '{k}'")

    def test_white_wall_gives_excellent_score(self):
        state = make_roi_state()
        state["room_features"]["wall_color"] = "white"
        state["room_features"]["color_palette"] = ["white"]
        ca = _vastu(state)["colour_analysis"]
        self.assertEqual(ca["vastu_score"], "excellent")

    def test_cream_wall_gives_excellent_score(self):
        state = make_roi_state()
        state["room_features"]["wall_color"] = "cream"
        ca = _vastu(state)["colour_analysis"]
        self.assertEqual(ca["vastu_score"], "excellent")

    def test_explanation_non_empty(self):
        ca = _vastu(make_roi_state())["colour_analysis"]
        self.assertGreater(len(ca["explanation"]), 20)


class TestColourRemedy(unittest.TestCase):

    def test_red_wall_has_remedy(self):
        state = make_roi_state()
        state["room_features"]["wall_color"] = "red"
        state["room_features"]["color_palette"] = ["red"]
        remedy = _vastu(state)["colour_remedy"]
        self.assertIsNotNone(remedy)
        self.assertIn("remedy", remedy)

    def test_black_wall_has_remedy(self):
        state = make_roi_state()
        state["room_features"]["wall_color"] = "black"
        state["room_features"]["color_palette"] = ["black"]
        remedy = _vastu(state)["colour_remedy"]
        self.assertIsNotNone(remedy)

    def test_neutral_wall_no_remedy(self):
        state = make_roi_state()
        state["room_features"]["wall_color"] = "white"
        state["room_features"]["color_palette"] = ["white"]
        remedy = _vastu(state)["colour_remedy"]
        self.assertIsNone(remedy)

    def test_remedy_has_issue_and_replacement(self):
        state = make_roi_state()
        state["room_features"]["wall_color"] = "red"
        state["room_features"]["color_palette"] = ["red"]
        remedy = _vastu(state)["colour_remedy"]
        if remedy:
            self.assertIn("issue", remedy)
            self.assertIn("replacement", remedy)


class TestObjectFindings(unittest.TestCase):

    def test_mirror_in_bedroom_is_important(self):
        state = make_roi_state(room_type="bedroom")
        state["room_features"]["detected_furniture"] = ["mirror", "bed"]
        state["cv_features"]["detected_objects"] = ["mirror"]
        findings = _vastu(state)["object_findings"]
        mirror_findings = [f for f in findings if "mirror" in f.get("object", "").lower()]
        self.assertGreater(len(mirror_findings), 0)
        self.assertEqual(mirror_findings[0]["severity"], "important")

    def test_plant_is_positive(self):
        state = make_roi_state()
        state["room_features"]["detected_furniture"] = ["indoor plant", "bed"]
        state["cv_features"]["detected_objects"] = ["indoor plant"]
        findings = _vastu(state)["object_findings"]
        plant_findings = [f for f in findings if "plant" in f.get("object", "").lower()]
        self.assertGreater(len(plant_findings), 0)
        self.assertEqual(plant_findings[0]["severity"], "positive")

    def test_tv_in_bedroom_is_important(self):
        state = make_roi_state(room_type="bedroom")
        state["room_features"]["detected_furniture"] = ["television", "bed"]
        state["cv_features"]["detected_objects"] = ["television"]
        findings = _vastu(state)["object_findings"]
        tv_findings = [f for f in findings if "television" in f.get("object", "").lower()
                       or "tv" in f.get("object", "").lower()]
        self.assertGreater(len(tv_findings), 0)
        self.assertEqual(tv_findings[0]["severity"], "important")

    def test_each_finding_has_action(self):
        state = make_roi_state()
        state["room_features"]["detected_furniture"] = ["mirror", "bed"]
        state["cv_features"]["detected_objects"] = ["mirror"]
        findings = _vastu(state)["object_findings"]
        for f in findings:
            self.assertIn("action", f)
            self.assertGreater(len(f["action"]), 10)


class TestLightingEnergy(unittest.TestCase):

    def test_poor_light_fires_deficiency(self):
        state = make_roi_state()
        state["room_features"]["natural_light"] = "poor"
        le = _vastu(state)["lighting_energy"]
        self.assertGreater(len(le), 0)
        combined = " ".join(item.get("aspect", "") for item in le).lower()
        self.assertIn("deficien", combined)

    def test_good_light_fires_positive(self):
        state = make_roi_state()
        state["room_features"]["natural_light"] = "excellent"
        le = _vastu(state)["lighting_energy"]
        self.assertGreater(len(le), 0)
        combined = " ".join(item.get("aspect", "") for item in le).lower()
        self.assertIn("good", combined)

    def test_each_lighting_finding_has_impact(self):
        state = make_roi_state()
        state["room_features"]["natural_light"] = "dim"
        le = _vastu(state)["lighting_energy"]
        for item in le:
            self.assertIn("impact", item)


class TestSpaceEnergy(unittest.TestCase):

    def test_cluttered_fires_when_low_free_space(self):
        state = make_roi_state()
        state["room_features"]["free_space_percentage"] = 20.0
        se = _vastu(state)["space_energy"]
        self.assertGreater(len(se), 0)
        combined = " ".join(item.get("aspect", "") for item in se).lower()
        self.assertIn("cluttered", combined)

    def test_open_fires_when_high_free_space(self):
        state = make_roi_state()
        state["room_features"]["free_space_percentage"] = 65.0
        se = _vastu(state)["space_energy"]
        self.assertGreater(len(se), 0)
        combined = " ".join(item.get("aspect", "") for item in se).lower()
        self.assertIn("open", combined)

    def test_normal_space_no_space_energy(self):
        state = make_roi_state()
        state["room_features"]["free_space_percentage"] = 45.0
        se = _vastu(state)["space_energy"]
        self.assertEqual(len(se), 0)


class TestOverallVastuScore(unittest.TestCase):

    def test_score_within_valid_range(self):
        score = _vastu(make_roi_state())["overall_score"]
        self.assertGreaterEqual(score, 20)
        self.assertLessEqual(score, 95)

    def test_excellent_light_increases_score(self):
        state_good = make_roi_state()
        state_good["room_features"]["natural_light"] = "excellent"
        state_poor = make_roi_state()
        state_poor["room_features"]["natural_light"] = "poor"
        score_good = _vastu(state_good)["overall_score"]
        score_poor = _vastu(state_poor)["overall_score"]
        self.assertGreater(score_good, score_poor)

    def test_black_wall_decreases_score(self):
        state_white = make_roi_state()
        state_white["room_features"]["wall_color"] = "white"
        state_white["room_features"]["color_palette"] = ["white"]
        state_black = make_roi_state()
        state_black["room_features"]["wall_color"] = "black"
        state_black["room_features"]["color_palette"] = ["black"]
        score_white = _vastu(state_white)["overall_score"]
        score_black = _vastu(state_black)["overall_score"]
        self.assertGreater(score_white, score_black)

    def test_mirror_in_bedroom_decreases_score(self):
        state_no_mirror = make_roi_state(room_type="bedroom")
        state_no_mirror["room_features"]["detected_furniture"] = ["bed"]

        state_mirror = make_roi_state(room_type="bedroom")
        state_mirror["room_features"]["detected_furniture"] = ["mirror", "bed"]
        state_mirror["cv_features"]["detected_objects"] = ["mirror"]
        state_mirror["detected_objects"] = ["mirror"]

        score_no = _vastu(state_no_mirror)["overall_score"]
        score_mirror = _vastu(state_mirror)["overall_score"]
        self.assertGreater(score_no, score_mirror)

    def test_label_matches_score(self):
        result = _vastu(make_roi_state())
        score = result["overall_score"]
        label = result["overall_label"]
        if score >= 85:
            self.assertEqual(label, "Excellent")
        elif score >= 70:
            self.assertEqual(label, "Good")
        elif score >= 55:
            self.assertEqual(label, "Fair")
        else:
            self.assertEqual(label, "Needs attention")


class TestTopImprovements(unittest.TestCase):

    def test_at_least_3_improvements(self):
        improvements = _vastu(make_roi_state())["top_improvements"]
        self.assertGreaterEqual(len(improvements), 3)

    def test_max_5_improvements(self):
        improvements = _vastu(make_roi_state())["top_improvements"]
        self.assertLessEqual(len(improvements), 5)

    def test_each_improvement_is_non_empty_string(self):
        for item in _vastu(make_roi_state())["top_improvements"]:
            self.assertIsInstance(item, str)
            self.assertGreater(len(item), 15)


class TestVastuMarketData(unittest.TestCase):

    def test_market_data_present(self):
        vmd = _vastu(make_roi_state())["vastu_market_data"]
        self.assertIsInstance(vmd, dict)

    def test_buyer_consideration_pct(self):
        vmd = _vastu(make_roi_state())["vastu_market_data"]
        self.assertIn("buyer_consideration_pct", vmd)
        self.assertGreater(vmd["buyer_consideration_pct"], 50)

    def test_resale_impact_non_empty(self):
        vmd = _vastu(make_roi_state())["vastu_market_data"]
        self.assertIn("resale_impact", vmd)
        self.assertGreater(len(vmd["resale_impact"]), 20)

    def test_source_present(self):
        vmd = _vastu(make_roi_state())["vastu_market_data"]
        self.assertIn("source", vmd)


class TestVastuEdgeCases(unittest.TestCase):

    def test_empty_state_no_crash(self):
        result = _vastu({})
        self.assertIsInstance(result, dict)
        self.assertIn("overall_score", result)

    def test_image_grounded_true_when_objects_detected(self):
        state = make_roi_state()
        state["detected_objects"] = ["bed", "wardrobe"]
        result = _vastu(state)
        self.assertTrue(result["image_grounded"])

    def test_image_grounded_false_when_no_objects(self):
        state = make_roi_state()
        state["detected_objects"] = []
        state["cv_features"]["detected_objects"] = []
        state["room_features"]["detected_furniture"] = []
        state["room_features"]["wall_color"] = "unknown"
        result = _vastu(state)
        self.assertFalse(result["image_grounded"])

    def test_unknown_room_type_still_returns_result(self):
        state = make_roi_state(room_type="balcony")
        result = _vastu(state)
        self.assertIn("overall_score", result)

    def test_vastu_disclaimer_present(self):
        result = _vastu(make_roi_state())
        self.assertIn("vastu_disclaimer", result)
        self.assertGreater(len(result["vastu_disclaimer"]), 50)

    def test_vastu_materials_present(self):
        result = _vastu(make_roi_state())
        self.assertIn("vastu_materials", result)
        for k in ("flooring", "walls", "ceiling"):
            self.assertIn(k, result["vastu_materials"])


if __name__ == "__main__":
    unittest.main()
