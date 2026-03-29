"""
Tests — agents/insight_generation_agent.py
============================================
Covers:
  - _extract_priority_repairs:
      * No contradictory issues (over vs under-furnished)
      * Deduplication by category (no two structural of same type)
      * Layout issues capped at 1
      * Max 5 repairs total
      * Every repair has: issue, severity, how_to_fix, estimated_cost_inr
      * Fallback when no issues and good condition
      * Severity escalation for crack/seepage/leak
  - _build_renovation_sequence:
      * Every step has title, description, estimated_cost_inr, duration_days
      * Steps ordered correctly (site prep before painting, painting before fixtures)
      * Bedroom sequence has no plumbing rough-in
      * Bathroom sequence has waterproofing
      * Cost INR computed from budget
  - _build_room_intelligence:
      * All 4 sections present (what_we_detected, whats_working_well,
        quick_wins, material_spotlight)
      * Quick wins capped at 3
      * Triggered correctly (no ceiling quick-win when false ceiling exists)
      * Working well: good light → positive entry; poor condition → not positive
  - _build_insights_dict:
      * Required top-level keys present
      * data_quality has confidence_tier
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

import tests.conftest as _cf
from tests.conftest import load_agent, make_minimal_state

iga_mod = load_agent(
    "agents.insight_generation_agent",
    "agents/insight_generation_agent.py",
)
InsightGenerationAgent = iga_mod.InsightGenerationAgent


# ─────────────────────────────────────────────────────────────────────────────
# Priority Repairs
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractPriorityRepairs(unittest.TestCase):

    def setUp(self):
        self.agent = InsightGenerationAgent()

    def _call(self, state: dict) -> list:
        return self.agent._extract_priority_repairs(state, {})

    def test_max_5_repairs(self):
        state = make_minimal_state(issues_detected=[
            "crack on north wall", "peeling paint on south wall",
            "water stain on ceiling", "mould near window",
            "broken switchboard", "damp patch near bathroom wall",
        ])
        repairs = self._call(state)
        self.assertLessEqual(len(repairs), 5)

    def test_every_repair_has_required_keys(self):
        state = make_minimal_state(issues_detected=["crack on east wall"])
        repairs = self._call(state)
        self.assertGreater(len(repairs), 0)
        for r in repairs:
            for k in ("issue", "severity", "how_to_fix", "estimated_cost_inr",
                      "must_fix_first", "category"):
                self.assertIn(k, r, f"Repair missing key '{k}': {r}")

    def test_how_to_fix_non_empty(self):
        state = make_minimal_state(issues_detected=["crack on east wall"])
        repairs = self._call(state)
        for r in repairs:
            self.assertIsInstance(r["how_to_fix"], str)
            self.assertGreater(len(r["how_to_fix"]), 10)

    def test_estimated_cost_inr_is_non_negative(self):
        state = make_minimal_state(
            issues_detected=["peeling paint"],
            cost_estimate={"total_inr": 750_000},
        )
        repairs = self._call(state)
        for r in repairs:
            self.assertGreaterEqual(r["estimated_cost_inr"], 0)

    def test_no_duplicate_categories(self):
        state = make_minimal_state(issues_detected=[
            "crack on north wall",
            "crack on south wall",   # same category — should be deduped
            "peeling paint",
        ])
        repairs = self._call(state)
        categories = [r["category"] for r in repairs]
        # structural should appear at most once
        self.assertLessEqual(categories.count("structural"), 1)

    def test_layout_issues_capped_at_1(self):
        state = make_minimal_state()
        # Put two layout-type issues in room_features layout_issues
        state["room_features"]["layout_issues"] = [
            "Room appears over-furnished — consider decluttering 1–2 items",
            "Insufficient walkable space — furniture arrangement needs optimising",
        ]
        state["layout_report"]["issues"] = state["room_features"]["layout_issues"]
        repairs = self._call(state)
        layout_repairs = [r for r in repairs if r.get("category") == "layout"]
        self.assertLessEqual(len(layout_repairs), 1)

    def test_no_contradictory_layout_repairs(self):
        """over-furnished and under-furnished must not both appear."""
        state = make_minimal_state()
        state["layout_report"]["issues"] = [
            "Room appears over-furnished — consider decluttering 1–2 items",
            "Room appears under-furnished — add key furniture pieces for balance",
        ]
        repairs = self._call(state)
        issues_text = " ".join(r["issue"].lower() for r in repairs)
        has_over  = "over-furnished" in issues_text
        has_under = "under-furnished" in issues_text
        self.assertFalse(
            has_over and has_under,
            f"Contradictory layout repairs: {[r['issue'] for r in repairs]}",
        )

    def test_crack_escalates_to_high_severity(self):
        state = make_minimal_state(
            condition_score=68,
            issues_detected=["crack on east wall"],
        )
        repairs = self._call(state)
        crack_repairs = [r for r in repairs if "crack" in r["issue"].lower()]
        self.assertGreater(len(crack_repairs), 0)
        self.assertIn(crack_repairs[0]["severity"], ("high", "critical"))
        self.assertTrue(crack_repairs[0]["must_fix_first"])

    def test_good_condition_no_issues_shows_upgrades(self):
        """When room is fine and no issues, fallback shows high-value upgrades."""
        state = make_minimal_state(
            condition_score=82,
            issues_detected=[],
            high_value_upgrades=["false ceiling with cove lighting", "modular wardrobe"],
        )
        state["damage_assessment"]["issues_detected"] = []
        state["layout_report"]["issues"] = []
        repairs = self._call(state)
        self.assertGreater(len(repairs), 0)
        # Should be upgrade-type items
        upgrade_repairs = [r for r in repairs if r.get("is_upgrade") or r.get("category") == "upgrade"]
        self.assertGreater(len(upgrade_repairs), 0)

    def test_severity_levels_valid(self):
        state = make_minimal_state(issues_detected=["damp patch", "peeling paint"])
        repairs = self._call(state)
        valid_severities = {"low", "medium", "high", "critical"}
        for r in repairs:
            self.assertIn(r["severity"], valid_severities)

    def test_reads_from_damage_assessment_fallback(self):
        """Issues should be found even if only in damage_assessment."""
        state = make_minimal_state(issues_detected=[])
        state["damage_assessment"]["issues_detected"] = ["mould on bathroom wall"]
        repairs = self._call(state)
        issues_text = " ".join(r["issue"].lower() for r in repairs)
        self.assertIn("mould", issues_text)

    def test_empty_state_no_crash(self):
        repairs = self._call({})
        self.assertIsInstance(repairs, list)


# ─────────────────────────────────────────────────────────────────────────────
# Renovation Sequence
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildRenovationSequence(unittest.TestCase):

    def setUp(self):
        self.agent = InsightGenerationAgent()

    def _call(self, state: dict) -> list:
        return self.agent._build_renovation_sequence(state, {})

    def test_every_step_has_title(self):
        steps = self._call(make_minimal_state())
        self.assertGreater(len(steps), 0)
        for s in steps:
            self.assertIn("title", s)
            self.assertIsInstance(s["title"], str)
            self.assertGreater(len(s["title"]), 3)

    def test_every_step_has_description(self):
        steps = self._call(make_minimal_state())
        for s in steps:
            self.assertIn("description", s)
            self.assertGreater(len(s["description"]), 10)

    def test_every_step_has_duration_days(self):
        steps = self._call(make_minimal_state())
        for s in steps:
            self.assertIn("duration_days", s)
            self.assertGreater(s["duration_days"], 0)

    def test_every_step_has_cost_inr(self):
        state = make_minimal_state()
        state["cost_estimate"] = {"total_inr": 500_000}
        steps = self._call(state)
        for s in steps:
            self.assertIn("estimated_cost_inr", s)
            self.assertGreaterEqual(s["estimated_cost_inr"], 0)

    def test_step_numbers_sequential(self):
        steps = self._call(make_minimal_state())
        for i, s in enumerate(steps, 1):
            self.assertEqual(s["step"], i)

    def test_bedroom_has_no_plumbing_rough_in(self):
        state = make_minimal_state(room_type="bedroom")
        steps = self._call(state)
        titles = [s["title"].lower() for s in steps]
        self.assertFalse(
            any("plumbing rough" in t for t in titles),
            "Bedroom should not have plumbing rough-in step",
        )

    def test_bathroom_has_waterproofing(self):
        state = make_minimal_state(room_type="bathroom")
        steps = self._call(state)
        titles = [s["title"].lower() for s in steps]
        self.assertTrue(
            any("waterproof" in t for t in titles),
            "Bathroom must have waterproofing step",
        )

    def test_bathroom_has_plumbing_fixtures(self):
        state = make_minimal_state(room_type="bathroom")
        steps = self._call(state)
        titles = [s["title"].lower() for s in steps]
        self.assertTrue(any("plumbing fixture" in t for t in titles))

    def test_bedroom_has_ceiling_step(self):
        state = make_minimal_state(room_type="bedroom")
        steps = self._call(state)
        titles = [s["title"].lower() for s in steps]
        self.assertTrue(any("ceiling" in t for t in titles))

    def test_painting_steps_come_after_tiling(self):
        state = make_minimal_state(room_type="living_room")
        steps = self._call(state)
        titles = [s["title"].lower() for s in steps]
        tiling_idx = next((i for i, t in enumerate(titles) if "tile" in t or "tiling" in t), None)
        paint_idx  = next((i for i, t in enumerate(titles) if "paint coat" in t), None)
        if tiling_idx is not None and paint_idx is not None:
            self.assertLess(tiling_idx, paint_idx)

    def test_handover_is_last_step(self):
        steps = self._call(make_minimal_state())
        self.assertIn("handover", steps[-1]["title"].lower())

    def test_putty_before_final_paint(self):
        steps = self._call(make_minimal_state())
        titles = [s["title"].lower() for s in steps]
        putty_idx = next((i for i, t in enumerate(titles) if "putty" in t), None)
        paint_idx = next((i for i, t in enumerate(titles) if "final paint" in t), None)
        if putty_idx is not None and paint_idx is not None:
            self.assertLess(putty_idx, paint_idx)

    def test_crack_adds_structural_step(self):
        state = make_minimal_state(issues_detected=["crack on north wall"])
        steps = self._call(state)
        titles = [s["title"].lower() for s in steps]
        self.assertTrue(any("crack" in t for t in titles))

    def test_cosmetic_only_skips_site_prep(self):
        state = make_minimal_state(renovation_scope="cosmetic_only")
        steps = self._call(state)
        titles = [s["title"].lower() for s in steps]
        self.assertFalse(any("site preparation" in t for t in titles))

    def test_cost_proportional_to_budget(self):
        state_small = make_minimal_state()
        state_small["cost_estimate"] = {"total_inr": 200_000}
        state_big = make_minimal_state()
        state_big["cost_estimate"] = {"total_inr": 2_000_000}
        steps_small = self._call(state_small)
        steps_big   = self._call(state_big)
        total_small = sum(s["estimated_cost_inr"] for s in steps_small)
        total_big   = sum(s["estimated_cost_inr"] for s in steps_big)
        self.assertGreater(total_big, total_small)


# ─────────────────────────────────────────────────────────────────────────────
# Room Intelligence
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildRoomIntelligence(unittest.TestCase):

    def setUp(self):
        self.agent = InsightGenerationAgent()

    def _call(self, state: dict) -> dict:
        return self.agent._build_room_intelligence(state)

    def test_all_four_sections_present(self):
        ri = self._call(make_minimal_state())
        for section in ("what_we_detected", "whats_working_well",
                        "quick_wins", "material_spotlight"):
            self.assertIn(section, ri, f"Section missing: {section}")

    def test_what_we_detected_is_list(self):
        ri = self._call(make_minimal_state())
        self.assertIsInstance(ri["what_we_detected"], list)

    def test_what_we_detected_non_empty(self):
        ri = self._call(make_minimal_state())
        self.assertGreater(len(ri["what_we_detected"]), 0)

    def test_each_detected_item_has_label_value_detail(self):
        ri = self._call(make_minimal_state())
        for item in ri["what_we_detected"]:
            self.assertIn("label", item)
            self.assertIn("value", item)
            self.assertIn("detail", item)

    def test_quick_wins_capped_at_3(self):
        ri = self._call(make_minimal_state())
        self.assertLessEqual(len(ri["quick_wins"]), 3)

    def test_quick_wins_has_at_least_1(self):
        ri = self._call(make_minimal_state())
        self.assertGreater(len(ri["quick_wins"]), 0)

    def test_quick_win_has_required_keys(self):
        ri = self._call(make_minimal_state())
        for qw in ri["quick_wins"]:
            for k in ("action", "impact", "duration", "cost_range", "roi_note"):
                self.assertIn(k, qw, f"Quick win missing key '{k}': {qw}")

    def test_good_natural_light_in_working_well(self):
        state = make_minimal_state()
        state["room_features"]["natural_light"] = "excellent"
        ri = self._call(state)
        working = " ".join(w["point"].lower() for w in ri["whats_working_well"])
        self.assertIn("natural light", working)

    def test_existing_false_ceiling_in_working_well(self):
        state = make_minimal_state()
        state["room_features"]["ceiling_type"] = "pop false ceiling"
        ri = self._call(state)
        working = " ".join(w["point"].lower() for w in ri["whats_working_well"])
        self.assertIn("ceiling", working)

    def test_neutral_wall_triggers_feature_wall_quick_win(self):
        state = make_minimal_state()
        state["room_features"]["wall_color"] = "white"
        ri = self._call(state)
        actions = " ".join(qw["action"].lower() for qw in ri["quick_wins"])
        self.assertIn("wall", actions)

    def test_existing_false_ceiling_does_not_trigger_cove_quick_win(self):
        """If false ceiling exists, cove LED strip quick-win shouldn't fire."""
        state = make_minimal_state()
        state["room_features"]["ceiling_type"] = "pop false ceiling"
        ri = self._call(state)
        actions = " ".join(qw["action"].lower() for qw in ri["quick_wins"])
        # "cove" should not appear as a quick win if ceiling already exists
        self.assertNotIn("cove led strip along ceiling perimeter", actions)

    def test_material_spotlight_present(self):
        ri = self._call(make_minimal_state())
        self.assertIsInstance(ri["material_spotlight"], list)

    def test_material_spotlight_item_has_required_keys(self):
        state = make_minimal_state(material_types=["vitrified_tile"])
        ri = self._call(state)
        for item in ri["material_spotlight"]:
            for k in ("name", "quality", "durability", "upgrade_to", "upgrade_cost"):
                self.assertIn(k, item, f"Material item missing key '{k}'")

    def test_section_title_present(self):
        ri = self._call(make_minimal_state())
        self.assertIn("section_title", ri)
        self.assertEqual(ri["section_title"], "Room Intelligence")

    def test_image_grounded_flag_present(self):
        ri = self._call(make_minimal_state())
        self.assertIn("image_grounded", ri)

    def test_room_type_in_subtitle(self):
        state = make_minimal_state(room_type="kitchen")
        ri = self._call(state)
        self.assertIn("kitchen", ri["section_subtitle"].lower())

    def test_condition_score_in_detected_items(self):
        state = make_minimal_state(condition_score=75)
        ri = self._call(state)
        labels = [item["label"] for item in ri["what_we_detected"]]
        self.assertIn("Room Condition", labels)

    def test_poor_condition_not_in_working_well(self):
        state = make_minimal_state(condition_score=35)
        state["room_features"]["condition"] = "poor"
        ri = self._call(state)
        working = " ".join(w["point"].lower() for w in ri["whats_working_well"])
        self.assertNotIn("solid room condition", working)

    def test_empty_state_no_crash(self):
        ri = self._call({})
        self.assertIsInstance(ri, dict)
        self.assertIn("what_we_detected", ri)


# ─────────────────────────────────────────────────────────────────────────────
# _build_insights_dict  (contract shape)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildInsightsDict(unittest.TestCase):

    def setUp(self):
        self.agent = InsightGenerationAgent()

    def _call(self, state: dict) -> dict:
        from tests.conftest import load_agent as _la
        ie_mod = _la.__module__  # just to avoid unused-import warning
        return self.agent._build_insights_dict(state, {})

    def test_required_top_level_keys(self):
        result = self._call(make_minimal_state())
        required = [
            "summary_headline", "data_quality", "visual_analysis",
            "financial_outlook", "recommendations",
            "dataset_style_examples", "diy_renovation_tips",
            "image_grounded", "dataset_grounded", "rag_grounded",
        ]
        for k in required:
            self.assertIn(k, result, f"Insights dict missing key '{k}'")

    def test_data_quality_has_confidence_tier(self):
        dq = self._call(make_minimal_state())["data_quality"]
        self.assertIn("confidence_tier", dq)
        self.assertIn(dq["confidence_tier"], ("high", "medium", "low"))

    def test_visual_analysis_has_style_detected(self):
        va = self._call(make_minimal_state())["visual_analysis"]
        self.assertIn("style_detected", va)

    def test_financial_outlook_has_roi_pct(self):
        state = make_minimal_state()
        state["roi_prediction"] = {"roi_pct": 12.5}
        fo = self._call(state)["financial_outlook"]
        self.assertIn("roi_pct", fo)

    def test_summary_headline_non_empty_string(self):
        headline = self._call(make_minimal_state())["summary_headline"]
        self.assertIsInstance(headline, str)
        self.assertGreater(len(headline), 5)

    def test_recommendations_is_list(self):
        recs = self._call(make_minimal_state())["recommendations"]
        self.assertIsInstance(recs, list)


if __name__ == "__main__":
    unittest.main()
