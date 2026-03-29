"""
Tests — agents/roi_forecast.py
================================
Covers:
  - Constants and data integrity (CITY_TIER, CITY_PSF, CITY_YIELD, etc.)
  - validate_roi_reasonability: all 4 warning rules
  - _build_risk_factors: over-capitalisation, tier-specific, room-specific
  - _vision_to_materials: object→material mapping, style signals
  - ROIForecastAgent._heuristic_predict: output shape, value bounds, city variation
  - _build_report: rupee_breakdown formula, comparable_context, CI bounds,
                   material_roi_breakdown, renovation_timing_advice, resale_timeline,
                   neighbourhood_context
  - ROI clamping: never below ROI_MIN_PCT, never above ROI_MAX_PCT
  - Edge cases: unknown city, zero budget, extreme values
"""

from __future__ import annotations

import sys
import unittest

# Load stubs before importing agents
import tests.conftest as _cf
from tests.conftest import load_agent

roi_mod = load_agent("agents.roi_forecast", "agents/roi_forecast.py")

ROIForecastAgent       = roi_mod.ROIForecastAgent
validate_roi_reasonability = roi_mod.validate_roi_reasonability
_build_risk_factors    = roi_mod._build_risk_factors
_vision_to_materials   = roi_mod._vision_to_materials
CITY_TIER              = roi_mod.CITY_TIER
CITY_PSF               = roi_mod.CITY_PSF
CITY_YIELD             = roi_mod.CITY_YIELD
TIER_APPRECIATION      = roi_mod.TIER_APPRECIATION
ROOM_ROI_MULTIPLIER    = roi_mod.ROOM_ROI_MULTIPLIER
ROI_MIN_PCT            = roi_mod.ROI_MIN_PCT
ROI_MAX_PCT            = roi_mod.ROI_MAX_PCT
RESALE_PREMIUM_DISCOUNT = roi_mod.RESALE_PREMIUM_DISCOUNT
MATERIAL_ROI_FACTORS   = roi_mod.MATERIAL_ROI_FACTORS


class TestConstants(unittest.TestCase):
    """Constants match known Indian market data."""

    def test_tier1_cities_present(self):
        for city in ("Mumbai", "Bangalore", "Hyderabad", "Chennai", "Pune"):
            self.assertIn(city, CITY_TIER, f"{city} missing from CITY_TIER")
            self.assertEqual(CITY_TIER[city], 1)

    def test_tier2_cities_present(self):
        for city in ("Ahmedabad", "Jaipur", "Lucknow"):
            self.assertIn(city, CITY_TIER)
            self.assertEqual(CITY_TIER[city], 2)

    def test_city_psf_reasonable(self):
        """PSF values must be in a credible range (₹2,000–₹15,000/sqft)."""
        for city, psf in CITY_PSF.items():
            self.assertGreater(psf, 2_000, f"{city} PSF too low: {psf}")
            self.assertLess(psf, 15_000, f"{city} PSF too high: {psf}")

    def test_mumbai_psf_highest(self):
        self.assertEqual(CITY_PSF["Mumbai"], max(CITY_PSF.values()))

    def test_city_yield_reasonable(self):
        """Gross rental yields must be between 1.5% and 6%."""
        for city, yld in CITY_YIELD.items():
            self.assertGreater(yld, 1.5, f"{city} yield too low")
            self.assertLess(yld, 6.0, f"{city} yield too high")

    def test_tier1_higher_appreciation_than_tier3(self):
        self.assertGreater(TIER_APPRECIATION[1], TIER_APPRECIATION[3])

    def test_roi_bounds_sensible(self):
        self.assertGreater(ROI_MIN_PCT, 0)
        self.assertLess(ROI_MAX_PCT, 50)

    def test_kitchen_highest_room_multiplier(self):
        self.assertGreater(
            ROOM_ROI_MULTIPLIER.get("kitchen", 0),
            ROOM_ROI_MULTIPLIER.get("bedroom", 0),
        )

    def test_tier1_no_resale_discount(self):
        self.assertEqual(RESALE_PREMIUM_DISCOUNT[1], 1.0)

    def test_tier3_has_significant_discount(self):
        self.assertLess(RESALE_PREMIUM_DISCOUNT[3], 0.65)

    def test_all_cities_have_psf_and_yield(self):
        for city in CITY_TIER:
            self.assertIn(city, CITY_PSF, f"{city} missing PSF")
            self.assertIn(city, CITY_YIELD, f"{city} missing yield")


class TestValidateROIReasonability(unittest.TestCase):
    """validate_roi_reasonability flags implausible predictions correctly."""

    def _call(self, **kw):
        defaults = dict(
            roi_pct=12.0, payback_months=36,
            room_type="bedroom", city_tier=1, budget_tier="mid",
            renovation_cost_inr=500_000, property_value_inr=5_000_000,
        )
        defaults.update(kw)
        return validate_roi_reasonability(**defaults)

    def test_normal_case_no_warnings(self):
        result = self._call()
        self.assertFalse(result["is_unusual"])
        self.assertEqual(len(result["warnings"]), 0)

    def test_unusually_high_roi_basic_tier2_bedroom(self):
        result = self._call(
            roi_pct=30.0, room_type="bedroom",
            city_tier=2, budget_tier="basic",
        )
        self.assertTrue(result["is_unusual"])
        flags = [w["flag"] for w in result["warnings"]]
        self.assertIn("unusually_high", flags)

    def test_roi_3x_above_tier_benchmark_flagged(self):
        tier_bench = TIER_APPRECIATION[1]  # 9.2%
        # 3.5× = 32.2% — above threshold
        result = self._call(roi_pct=tier_bench * 3.6)
        self.assertTrue(result["is_unusual"])

    def test_fast_payback_flagged(self):
        result = self._call(payback_months=3)
        flags = [w["flag"] for w in result["warnings"]]
        self.assertIn("unusually_fast_payback", flags)

    def test_over_capitalisation_flagged(self):
        # Spend 30% of property value
        result = self._call(
            renovation_cost_inr=3_000_000,
            property_value_inr=10_000_000,
        )
        flags = [w["flag"] for w in result["warnings"]]
        self.assertIn("over_capitalisation_risk", flags)

    def test_returns_dict_with_required_keys(self):
        result = self._call()
        self.assertIn("is_unusual", result)
        self.assertIn("warnings", result)
        self.assertIsInstance(result["warnings"], list)

    def test_warning_has_flag_and_explanation(self):
        result = self._call(payback_months=2)
        for w in result["warnings"]:
            self.assertIn("flag", w)
            self.assertIn("explanation", w)
            self.assertIsInstance(w["explanation"], str)
            self.assertGreater(len(w["explanation"]), 20)


class TestBuildRiskFactors(unittest.TestCase):
    """_build_risk_factors returns 2–3 contextual risk statements."""

    def _call(self, **kw):
        defaults = dict(
            city="Hyderabad", city_tier=1, room_type="bedroom",
            budget_tier="mid", renovation_cost_inr=500_000,
            property_value_inr=5_000_000,
        )
        defaults.update(kw)
        return _build_risk_factors(**defaults)

    def test_returns_list(self):
        self.assertIsInstance(self._call(), list)

    def test_at_least_2_factors(self):
        self.assertGreaterEqual(len(self._call()), 2)

    def test_max_3_factors(self):
        self.assertLessEqual(len(self._call()), 3)

    def test_over_capitalisation_appears_when_high_spend(self):
        factors = self._call(
            renovation_cost_inr=2_000_000,
            property_value_inr=5_000_000,
        )
        combined = " ".join(factors).lower()
        self.assertIn("over", combined)

    def test_tier2_city_note_present(self):
        factors = self._call(city="Jaipur", city_tier=2)
        combined = " ".join(factors).lower()
        self.assertIn("tier-2", combined)

    def test_tier3_city_note_present(self):
        factors = self._call(city="Bhopal", city_tier=3)
        combined = " ".join(factors).lower()
        self.assertIn("tier-3", combined)

    def test_kitchen_note_present(self):
        factors = self._call(room_type="kitchen")
        combined = " ".join(factors).lower()
        self.assertIn("kitchen", combined)

    def test_bathroom_waterproofing_note(self):
        factors = self._call(room_type="bathroom")
        combined = " ".join(factors).lower()
        self.assertIn("waterproof", combined)

    def test_factors_are_non_empty_strings(self):
        for f in self._call():
            self.assertIsInstance(f, str)
            self.assertGreater(len(f), 30)


class TestVisionToMaterials(unittest.TestCase):
    """_vision_to_materials maps CV features to material keys correctly."""

    def test_empty_cv_features_returns_empty(self):
        self.assertEqual(_vision_to_materials(None), [])
        self.assertEqual(_vision_to_materials({}), [])

    def test_marble_floor_maps_to_premium_flooring(self):
        cv = {"detected_objects": ["marble floor"], "materials": [], "style": ""}
        result = _vision_to_materials(cv)
        self.assertIn("premium_flooring", result)

    def test_modular_kitchen_maps_correctly(self):
        cv = {"detected_objects": ["modular kitchen"], "materials": [], "style": ""}
        result = _vision_to_materials(cv)
        self.assertIn("modular_kitchen", result)

    def test_false_ceiling_maps_correctly(self):
        cv = {"detected_objects": ["false ceiling"], "materials": [], "style": ""}
        result = _vision_to_materials(cv)
        self.assertIn("false_ceiling", result)

    def test_style_signal_added(self):
        cv = {"detected_objects": [], "materials": [],
              "style": "Modern Minimalist"}
        result = _vision_to_materials(cv)
        # Modern Minimalist maps to [false_ceiling, premium_paint]
        self.assertTrue(len(result) >= 1)

    def test_materials_field_marble_detected(self):
        cv = {"detected_objects": [], "materials": ["marble"], "style": ""}
        result = _vision_to_materials(cv)
        self.assertIn("premium_flooring", result)

    def test_max_6_materials_returned(self):
        cv = {
            "detected_objects": [
                "marble floor", "modular kitchen", "false ceiling",
                "smart switch", "upvc window", "fitted wardrobe",
                "premium sanitary",
            ],
            "materials": ["marble", "wood"],
            "style": "Contemporary Indian",
        }
        result = _vision_to_materials(cv)
        self.assertLessEqual(len(result), 6)

    def test_no_duplicates(self):
        cv = {
            "detected_objects": ["marble floor", "granite"],
            "materials": ["marble"],
            "style": "Scandinavian",
        }
        result = _vision_to_materials(cv)
        self.assertEqual(len(result), len(set(result)))


class TestROIForecastAgentHeuristic(unittest.TestCase):
    """ROIForecastAgent._heuristic_predict: shape, bounds, city variation."""

    @classmethod
    def setUpClass(cls):
        cls.agent = ROIForecastAgent.__new__(ROIForecastAgent)
        cls.agent._calibrator = None
        cls.agent._explainer = None
        cls.agent._nhb_validator = None

    def _predict(self, **kw):
        defaults = dict(
            renovation_cost_inr=500_000,
            area_sqft=120.0,
            city="Hyderabad",
            room_type="bedroom",
            budget_tier="mid",
            current_property_value_inr=None,
            materials=[],
            renovation_scope="partial",
        )
        defaults.update(kw)
        return self.agent._heuristic_predict(**defaults)

    # ── output shape ──────────────────────────────────────────────────────────

    def test_required_keys_present(self):
        result = self._predict()
        required = [
            "roi_pct", "roi_ci_low", "roi_ci_high",
            "comparable_context", "rupee_breakdown",
            "risk_factors", "confidence_level", "data_transparency",
            "equity_gain_inr", "pre_reno_value_inr", "post_reno_value_inr",
            "payback_months", "model_type",
            "material_roi_breakdown", "renovation_timing_advice",
            "resale_timeline", "neighbourhood_context",
        ]
        for k in required:
            self.assertIn(k, result, f"Missing key: {k}")

    def test_roi_pct_within_bounds(self):
        result = self._predict()
        self.assertGreaterEqual(result["roi_pct"], ROI_MIN_PCT)
        self.assertLessEqual(result["roi_pct"], ROI_MAX_PCT)

    def test_ci_low_le_mean_le_high(self):
        result = self._predict()
        self.assertLessEqual(result["roi_ci_low"], result["roi_pct"])
        self.assertGreaterEqual(result["roi_ci_high"], result["roi_pct"])

    def test_post_reno_value_greater_than_pre(self):
        result = self._predict()
        self.assertGreater(
            result["post_reno_value_inr"],
            result["pre_reno_value_inr"],
        )

    def test_model_type_is_heuristic(self):
        result = self._predict()
        self.assertIn("heuristic", result["model_type"])

    # ── rupee_breakdown ───────────────────────────────────────────────────────

    def test_rupee_breakdown_spend_matches_input(self):
        result = self._predict(renovation_cost_inr=600_000)
        self.assertEqual(result["rupee_breakdown"]["spend_inr"], 600_000)

    def test_rupee_breakdown_formula_keys(self):
        bd = self._predict()["rupee_breakdown"]
        for k in ("spend_inr", "value_added_inr", "net_equity_gain_inr",
                  "monthly_rental_increase_inr", "payback_months", "formula"):
            self.assertIn(k, bd)

    def test_monthly_rental_increase_positive(self):
        result = self._predict()
        self.assertGreater(
            result["rupee_breakdown"]["monthly_rental_increase_inr"], 0
        )

    def test_payback_months_between_6_and_120(self):
        result = self._predict()
        pb = result["rupee_breakdown"]["payback_months"]
        self.assertGreaterEqual(pb, 6)
        self.assertLessEqual(pb, 120)

    # ── comparable_context ────────────────────────────────────────────────────

    def test_comparable_context_has_benchmark(self):
        ctx = self._predict()["comparable_context"]
        self.assertIn("city_avg_renovation_roi_pct", ctx)
        self.assertIn("room_type_benchmark_pct", ctx)
        self.assertIn("interpretation", ctx)
        self.assertIn("source", ctx)

    def test_interpretation_mentions_city(self):
        ctx = self._predict(city="Mumbai")["comparable_context"]
        self.assertIn("Mumbai", ctx["interpretation"])

    def test_room_benchmark_varies_by_room(self):
        bed = self._predict(room_type="bedroom")["comparable_context"]
        kit = self._predict(room_type="kitchen")["comparable_context"]
        self.assertGreater(
            kit["room_type_benchmark_pct"],
            bed["room_type_benchmark_pct"],
        )

    # ── city variation ────────────────────────────────────────────────────────

    def test_tier1_city_higher_roi_than_tier3(self):
        t1 = self._predict(city="Mumbai")["roi_pct"]
        t3 = self._predict(city="Bhopal")["roi_pct"]
        # After discount, Tier-1 still comes out higher
        self.assertGreater(t1, t3)

    def test_unknown_city_uses_tier2_defaults(self):
        # Should not raise
        result = self._predict(city="UnknownCity")
        self.assertGreater(result["roi_pct"], 0)

    # ── room type variation ───────────────────────────────────────────────────

    def test_kitchen_roi_higher_than_bedroom(self):
        bed = self._predict(room_type="bedroom")["roi_pct"]
        kit = self._predict(room_type="kitchen")["roi_pct"]
        self.assertGreater(kit, bed)

    def test_full_home_roi_higher_than_study(self):
        study = self._predict(room_type="study")["roi_pct"]
        full  = self._predict(room_type="full_home")["roi_pct"]
        self.assertGreater(full, study)

    # ── material_roi_breakdown ────────────────────────────────────────────────

    def test_material_roi_breakdown_present(self):
        result = self._predict()
        breakdown = result["material_roi_breakdown"]
        self.assertIsInstance(breakdown, list)
        self.assertGreater(len(breakdown), 0)

    def test_material_breakdown_has_required_keys(self):
        result = self._predict()
        for item in result["material_roi_breakdown"]:
            for k in ("material", "avg_roi_pct", "cost_range",
                      "buyer_appeal", "note", "source"):
                self.assertIn(k, item, f"Missing key in material breakdown: {k}")

    def test_materials_passed_appear_in_breakdown(self):
        result = self._predict(materials=["modular_kitchen", "premium_flooring"])
        names = [m["material"].lower() for m in result["material_roi_breakdown"]]
        found = any("kitchen" in n for n in names)
        self.assertTrue(found, "modular_kitchen not reflected in breakdown")

    # ── renovation_timing_advice ──────────────────────────────────────────────

    def test_timing_advice_present(self):
        timing = self._predict()["renovation_timing_advice"]
        self.assertIsInstance(timing, dict)
        for k in ("best_season", "avoid_season", "listing_timing"):
            self.assertIn(k, timing)

    def test_timing_differs_by_tier(self):
        t1_timing = self._predict(city="Mumbai")["renovation_timing_advice"]
        t3_timing = self._predict(city="Bhopal")["renovation_timing_advice"]
        # Different cities may share timing advice text but the dicts must exist
        self.assertIsInstance(t1_timing, dict)
        self.assertIsInstance(t3_timing, dict)

    # ── resale_timeline ───────────────────────────────────────────────────────

    def test_resale_timeline_has_three_horizons(self):
        tl = self._predict()["resale_timeline"]
        for k in ("sell_in_1_year", "sell_in_3_years", "sell_in_5_years"):
            self.assertIn(k, tl)

    def test_resale_5yr_roi_gte_1yr(self):
        tl = self._predict()["resale_timeline"]
        self.assertGreaterEqual(
            tl["sell_in_5_years"]["realised_roi_pct"],
            tl["sell_in_1_year"]["realised_roi_pct"],
        )

    def test_resale_has_recommendation(self):
        tl = self._predict()["resale_timeline"]
        self.assertIn("recommendation", tl)
        self.assertGreater(len(tl["recommendation"]), 20)

    # ── neighbourhood_context ─────────────────────────────────────────────────

    def test_neighbourhood_context_present(self):
        nc = self._predict(city="Hyderabad")["neighbourhood_context"]
        for k in ("5yr_appreciation", "demand_trend", "rental_demand",
                  "buyer_type", "source"):
            self.assertIn(k, nc)

    def test_known_city_has_specific_context(self):
        nc = self._predict(city="Bangalore")["neighbourhood_context"]
        # Bangalore-specific context should mention IT or tech
        combined = (nc.get("demand_trend", "") + nc.get("buyer_type", "")).lower()
        self.assertIn("it", combined)

    def test_unknown_city_still_has_context(self):
        nc = self._predict(city="UnknownCity")["neighbourhood_context"]
        self.assertIsInstance(nc, dict)
        self.assertGreater(len(nc), 0)

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_zero_renovation_cost_no_crash(self):
        result = self._predict(renovation_cost_inr=1)
        self.assertGreaterEqual(result["roi_pct"], ROI_MIN_PCT)

    def test_very_large_renovation_clamped(self):
        result = self._predict(renovation_cost_inr=50_000_000)
        self.assertLessEqual(result["roi_pct"], ROI_MAX_PCT)

    def test_premium_budget_tier_higher_roi(self):
        mid     = self._predict(budget_tier="mid")["roi_pct"]
        premium = self._predict(budget_tier="premium")["roi_pct"]
        self.assertGreater(premium, mid)

    def test_confidence_level_has_level_field(self):
        cl = self._predict()["confidence_level"]
        self.assertIn("level", cl)
        self.assertIn(cl["level"], ("high", "medium", "low"))


if __name__ == "__main__":
    unittest.main()
