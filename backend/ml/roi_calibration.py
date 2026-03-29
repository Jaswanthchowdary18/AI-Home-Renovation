"""
ARKEN — ROI Calibration Module v2.0
======================================
Real-data-driven ROI multipliers + NHB Residex 2024 benchmark validation.

v2.0 Changes over v1.0:
  - NHBBenchmarkValidator class added:
      Validates ROI predictions against NHB Residex 2024 city×room_type ranges.
      Flags predictions outside ±2 standard deviations of the benchmark.
      Delegates to ROIExplainer.validate_against_nhb_benchmarks() (single
      source of truth for benchmark data) with its own formatting layer.
      Provides get_benchmark_summary() for dashboard display.

  - ROICalibrator v1.0 PRESERVED UNCHANGED:
      calibrate_from_real_data(), get_room_multiplier(), get_city_psf(),
      get_rental_yield(), get_calibration_report() all unchanged.

  - get_nhb_validator() — new module-level singleton accessor.

Usage:
    from ml.roi_calibration import get_calibrator, get_nhb_validator

    # Existing v1.0 API (unchanged)
    cal = get_calibrator()
    room_mult = cal.get_room_multiplier("kitchen", fallback=1.35)

    # New v2.0 NHB validator
    validator = get_nhb_validator()
    result = validator.validate(roi_pct=18.5, city="Bangalore", room_type="kitchen")
    summary = validator.get_benchmark_summary(city="Bangalore")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))
_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _resolve(app_rel: str, local_rel: str) -> Path:
    a = _APP_DIR    / app_rel
    b = _BACKEND_DIR / local_rel
    return a if a.exists() else b


_PROPERTY_CSV = _resolve(
    "data/datasets/property_transactions/india_property_transactions.csv",
    "data/datasets/property_transactions/india_property_transactions.csv",
)
_HOUSING_DIR  = _resolve(
    "data/datasets/india_housing_prices",
    "data/datasets/india_housing_prices",
)
_RENT_CSV = _resolve(
    "data/datasets/House Price India/House_Rent_Dataset.csv",
    "data/datasets/House Price India/House_Rent_Dataset.csv",
)


# ─────────────────────────────────────────────────────────────────────────────
# ROICalibrator v1.0 (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

class ROICalibrator:
    """
    Computes real-data-driven multipliers for ROI prediction.

    If calibration succeeds, get_room_multiplier(), get_city_psf(), and
    get_rental_yield() return values learned from real data.
    If it fails (data missing), they return the supplied fallback value —
    making this a transparent, zero-risk upgrade over hardcoded constants.

    v1.0 API preserved unchanged.
    """

    def __init__(self) -> None:
        self._calibrated:       bool             = False
        self._room_multipliers: Dict[str, float] = {}
        self._city_psf:         Dict[str, int]   = {}
        self._rental_yields:    Dict[str, float] = {}

    def calibrate_from_real_data(self) -> bool:
        success  = False
        success |= self._calibrate_room_multipliers()
        success |= self._calibrate_city_psf()
        success |= self._calibrate_rental_yields()
        self._calibrated = success

        if success:
            logger.info(
                f"[ROICalibrator] Calibrated from real data: "
                f"room_multipliers={len(self._room_multipliers)}, "
                f"city_psf={len(self._city_psf)}, "
                f"rental_yields={len(self._rental_yields)}"
            )
        else:
            logger.warning(
                "[ROICalibrator] All calibration steps failed — "
                "will use hardcoded fallback constants."
            )
        return success

    def _calibrate_room_multipliers(self) -> bool:
        if not _PROPERTY_CSV.exists():
            logger.debug(f"[ROICalibrator] Property CSV not found: {_PROPERTY_CSV}")
            return False
        try:
            import pandas as pd

            df = pd.read_csv(str(_PROPERTY_CSV))
            if "data_source" in df.columns:
                df = df[df["data_source"].str.contains("real", case=False, na=False)]

            reno = df[df["roi_pct"].notna()].copy()
            if len(reno) < 100:
                logger.debug(f"[ROICalibrator] Too few renovation rows: {len(reno)}")
                return False

            bedroom_roi = reno[reno["room_renovated"] == "bedroom"]["roi_pct"]
            if len(bedroom_roi) < 20:
                return False
            bedroom_mean = float(bedroom_roi.mean())
            if bedroom_mean <= 0:
                return False

            multipliers: Dict[str, float] = {}
            for room in ["kitchen", "bathroom", "living_room", "full_home", "bedroom"]:
                subset = reno[reno["room_renovated"] == room]["roi_pct"]
                if len(subset) >= 20:
                    room_mean = float(subset.mean())
                    multipliers[room] = round(room_mean / bedroom_mean, 3)
                    logger.debug(
                        f"[ROICalibrator] {room}: mean={room_mean:.2f}%  "
                        f"mult={multipliers[room]:.3f} (n={len(subset)})"
                    )

            if multipliers:
                self._room_multipliers = multipliers
                return True

        except Exception as exc:
            logger.debug(f"[ROICalibrator] Room multiplier error: {exc}")
        return False

    def _calibrate_city_psf(self) -> bool:
        if not _HOUSING_DIR.exists():
            logger.debug(f"[ROICalibrator] Housing dir not found: {_HOUSING_DIR}")
            return False

        city_map = {
            "Bangalore": "Bangalore", "Mumbai":    "Mumbai",
            "Chennai":   "Chennai",   "Delhi":     "Delhi NCR",
            "Hyderabad": "Hyderabad", "Kolkata":   "Kolkata",
        }
        found_any = False
        try:
            import pandas as pd

            for csv_name, arken_city in city_map.items():
                csv_path = _HOUSING_DIR / f"{csv_name}.csv"
                if not csv_path.exists():
                    continue
                try:
                    cdf = pd.read_csv(str(csv_path))
                    cdf["psf"] = cdf["Price"] / cdf["Area"].clip(lower=1)
                    cdf = cdf[(cdf["psf"] >= 1500) & (cdf["psf"] <= 80000)]
                    if len(cdf) >= 50:
                        median_psf = int(cdf["psf"].median())
                        self._city_psf[arken_city] = median_psf
                        logger.debug(
                            f"[ROICalibrator] {arken_city} PSF: ₹{median_psf:,} "
                            f"(n={len(cdf):,})"
                        )
                        found_any = True
                except Exception as exc:
                    logger.debug(f"[ROICalibrator] {csv_name} PSF failed: {exc}")

        except ImportError:
            logger.debug("[ROICalibrator] pandas not available for PSF calibration")

        return found_any

    def _calibrate_rental_yields(self) -> bool:
        if not _RENT_CSV.exists():
            logger.debug(f"[ROICalibrator] Rent CSV not found: {_RENT_CSV}")
            return False
        if not self._city_psf:
            return False

        city_map = {
            "Bangalore": "Bangalore", "Mumbai":    "Mumbai",
            "Chennai":   "Chennai",   "Delhi":     "Delhi NCR",
            "Hyderabad": "Hyderabad", "Kolkata":   "Kolkata",
        }
        try:
            import pandas as pd

            rdf = pd.read_csv(str(_RENT_CSV))
            rdf = rdf[rdf["Size"] > 0].copy()
            rdf["rent_psf"] = rdf["Rent"] / rdf["Size"]
            rdf = rdf[(rdf["rent_psf"] > 1) & (rdf["rent_psf"] < 500)]

            found_any = False
            for csv_city, arken_city in city_map.items():
                subset = rdf[rdf["City"] == csv_city]
                if len(subset) < 20:
                    continue
                psf = self._city_psf.get(arken_city)
                if not psf:
                    continue
                median_rps  = float(subset["rent_psf"].median())
                gross_yield = round(median_rps * 12 / psf, 4)
                self._rental_yields[arken_city] = gross_yield
                logger.debug(
                    f"[ROICalibrator] {arken_city} yield: {gross_yield*100:.2f}% "
                    f"(rent_psf=₹{median_rps:.2f}, n={len(subset)})"
                )
                found_any = True

            return found_any

        except Exception as exc:
            logger.debug(f"[ROICalibrator] Rental yield error: {exc}")
            return False

    def get_room_multiplier(self, room_type: str, fallback: float) -> float:
        if self._calibrated and room_type in self._room_multipliers:
            return self._room_multipliers[room_type]
        return fallback

    def get_city_psf(self, city: str, fallback: int) -> int:
        if self._calibrated and city in self._city_psf:
            return self._city_psf[city]
        return fallback

    def get_rental_yield(self, city: str, fallback: float) -> float:
        if self._calibrated and city in self._rental_yields:
            return self._rental_yields[city]
        return fallback

    def get_calibration_report(self) -> Dict[str, Any]:
        return {
            "calibrated":               self._calibrated,
            "room_multipliers_learned": len(self._room_multipliers),
            "cities_with_real_psf":     len(self._city_psf),
            "cities_with_real_yield":   len(self._rental_yields),
            "source":                   "real_kaggle_data" if self._calibrated else "hardcoded_constants",
            "room_multipliers":         dict(self._room_multipliers),
            "city_psf_sample":          {k: v for k, v in list(self._city_psf.items())[:3]},
        }


# ─────────────────────────────────────────────────────────────────────────────
# NHBBenchmarkValidator v2.0 (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class NHBBenchmarkValidator:
    """
    Validates ROI predictions against NHB Residex 2024 benchmarks.

    Delegates to ROIExplainer.validate_against_nhb_benchmarks() which holds
    the authoritative benchmark data (derived from 32,210 real transaction rows).

    Additional methods:
      - get_benchmark_summary(city) → all room_type benchmarks for a city
      - get_expected_range(city, room_type) → (low, high) 80% probability range
      - flag_for_display(validation_result) → user-facing severity label

    Usage:
        validator = NHBBenchmarkValidator()
        result = validator.validate(18.5, "Bangalore", "kitchen")
        summary = validator.get_benchmark_summary("Mumbai")
    """

    # Severity thresholds
    _Z_WARN     = 2.0   # |z| > 2.0  → flag (outside ±2 std dev)
    _Z_CRITICAL = 3.0   # |z| > 3.0  → critical flag

    def validate(
        self,
        roi_pct: float,
        city: str,
        room_type: str,
    ) -> Dict[str, Any]:
        """
        Validate a ROI prediction against NHB benchmarks.

        Args:
            roi_pct:   Predicted ROI percentage.
            city:      City name (e.g. "Bangalore", "Mumbai", "Jaipur").
            room_type: Room type (e.g. "kitchen", "bedroom").

        Returns:
            {
              "within_benchmark":     bool,
              "benchmark_mean":       float,
              "benchmark_std":        float,
              "benchmark_n":          int,
              "z_score":              float,
              "flag":                 None | "unusually_high" | "unusually_low",
              "severity":             "ok" | "warning" | "critical",
              "flag_threshold_2sd":   float,
              "data_source":          str,
              "note":                 str,
              "recommended_action":   str,   # NEW — actionable guidance
            }
        """
        try:
            from ml.roi_explainer import get_explainer
            explainer = get_explainer()
            result = explainer.validate_against_nhb_benchmarks(
                roi_pct=roi_pct, city=city, room_type=room_type
            )
        except Exception as e:
            logger.warning(f"[NHBBenchmarkValidator] ROIExplainer unavailable: {e}. "
                           "Using direct benchmark lookup.")
            result = self._direct_validate(roi_pct, city, room_type)

        # Enrich with severity + recommended action
        z  = abs(result.get("z_score", 0.0))
        flag = result.get("flag")

        if z > self._Z_CRITICAL:
            severity = "critical"
        elif z > self._Z_WARN:
            severity = "warning"
        else:
            severity = "ok"

        result["severity"] = severity
        result["recommended_action"] = self._recommended_action(flag, severity, city, room_type)
        return result

    def _direct_validate(
        self, roi_pct: float, city: str, room_type: str
    ) -> Dict[str, Any]:
        """
        Direct benchmark lookup without ROIExplainer dependency.
        Uses the same NHB_BENCHMARKS data from roi_explainer.py.
        """
        # Import benchmark data directly
        try:
            from ml.roi_explainer import NHB_BENCHMARKS, _TIER_MEAN_FALLBACK, _TIER_STD_FALLBACK, _CITY_TIER
        except ImportError:
            # Last-resort hardcoded fallback
            return {
                "within_benchmark": True, "benchmark_mean": 14.0,
                "benchmark_std": 6.5, "benchmark_n": 0,
                "z_score": 0.0, "flag": None,
                "flag_threshold_2sd": 27.0, "data_source": "hardcoded_fallback",
                "note": "NHB benchmark lookup unavailable.",
            }

        rt = (room_type or "bedroom").lower().replace(" ", "_")
        if rt not in ("bathroom", "bedroom", "full_home", "kitchen", "living_room"):
            rt = "bedroom"

        city_bench = NHB_BENCHMARKS.get(city)
        if city_bench and rt in city_bench:
            bench  = city_bench[rt]
            bm_mean = bench["mean"]
            bm_std  = bench["std"]
            bm_n    = bench["n"]
            source  = "real_kaggle_transaction_derived_32k_rows"
        else:
            tier    = _CITY_TIER.get(city, 2)
            bm_mean = _TIER_MEAN_FALLBACK[tier].get(rt, 12.0)
            bm_std  = _TIER_STD_FALLBACK[tier].get(rt, 6.0)
            bm_n    = 0
            source  = f"nhb_tier{tier}_estimated_anarock2024"

        safe_std   = max(bm_std, 0.5)
        z_score    = (roi_pct - bm_mean) / safe_std
        upper_2sd  = bm_mean + 2.0 * safe_std
        lower_2sd  = bm_mean - 2.0 * safe_std
        within     = lower_2sd <= roi_pct <= upper_2sd
        flag: Optional[str] = (
            "unusually_high" if z_score > 2.0 else
            "unusually_low"  if z_score < -2.0 else None
        )
        room_disp = rt.replace("_", " ")
        note = (
            f"ROI {roi_pct:.1f}% {'within' if within else 'outside'} normal range "
            f"for {room_disp} in {city} "
            f"(NHB benchmark: {bm_mean:.1f}% ± {safe_std:.1f}%)."
        )

        return {
            "within_benchmark": within, "benchmark_mean": round(bm_mean, 2),
            "benchmark_std": round(bm_std, 2), "benchmark_n": bm_n,
            "z_score": round(float(z_score), 3), "flag": flag,
            "flag_threshold_2sd": round(upper_2sd if flag == "unusually_high" else lower_2sd, 2),
            "data_source": source, "note": note,
        }

    @staticmethod
    def _recommended_action(
        flag: Optional[str],
        severity: str,
        city: str,
        room_type: str,
    ) -> str:
        """Return an actionable recommendation based on the validation result."""
        rt = room_type.replace("_", " ")
        if flag == "unusually_high":
            if severity == "critical":
                return (
                    f"Predicted ROI is exceptionally high for a {rt} renovation in {city}. "
                    "Verify renovation cost and current property value inputs before presenting "
                    "this forecast to the user. Consider lowering max displayed ROI."
                )
            return (
                f"ROI is above the typical range for {rt} renovations in {city}. "
                "Clearly communicate to the user that this is an optimistic scenario "
                "and actual results depend on local buyer demand and finish quality."
            )
        if flag == "unusually_low":
            if severity == "critical":
                return (
                    f"Predicted ROI is very low for a {rt} renovation in {city}. "
                    "This renovation may not recover its cost. Suggest the user focus "
                    "on rental yield improvement rather than near-term resale value."
                )
            return (
                f"ROI is below average for {rt} renovations in {city}. "
                "Highlight rental income improvement as the primary benefit rather than resale value."
            )
        return (
            f"ROI is within the normal range for {rt} renovations in {city}. "
            "No special caveats required for this prediction."
        )

    def get_benchmark_summary(self, city: str) -> Dict[str, Any]:
        """
        Return all room_type benchmarks for a given city.

        Args:
            city: City name (e.g. "Bangalore").

        Returns:
            {
                "city": str,
                "data_source": str,
                "benchmarks": {
                    "kitchen":     {"mean": 16.25, "std": 8.18, "range_2sd": [0.0, 32.6], "n": 1225},
                    "bedroom":     {...},
                    ...
                }
            }
        """
        try:
            from ml.roi_explainer import NHB_BENCHMARKS, _TIER_MEAN_FALLBACK, _TIER_STD_FALLBACK, _CITY_TIER
        except ImportError:
            return {"city": city, "data_source": "unavailable", "benchmarks": {}}

        city_bench = NHB_BENCHMARKS.get(city)
        room_types = ["bathroom", "bedroom", "full_home", "kitchen", "living_room"]

        if city_bench:
            source = "real_kaggle_transaction_derived_32k_rows"
        else:
            tier       = _CITY_TIER.get(city, 2)
            city_bench = {
                rt: {
                    "mean": _TIER_MEAN_FALLBACK[tier].get(rt, 12.0),
                    "std":  _TIER_STD_FALLBACK[tier].get(rt, 6.0),
                    "n": 0,
                }
                for rt in room_types
            }
            source = f"nhb_tier{tier}_estimated_anarock2024"

        benchmarks: Dict[str, Any] = {}
        for rt in room_types:
            if rt in city_bench:
                bench    = city_bench[rt]
                mean_    = bench["mean"]
                std_     = max(bench["std"], 0.5)
                n        = bench.get("n", 0)
                lo_2sd   = round(max(0.0, mean_ - 2 * std_), 2)
                hi_2sd   = round(mean_ + 2 * std_, 2)
                benchmarks[rt] = {
                    "mean":       round(mean_, 2),
                    "std":        round(bench["std"], 2),
                    "range_2sd":  [lo_2sd, hi_2sd],
                    "n":          n,
                    "label":      f"{mean_:.1f}% (±{std_:.1f}%)",
                }

        return {"city": city, "data_source": source, "benchmarks": benchmarks}

    def get_expected_range(
        self,
        city: str,
        room_type: str,
        confidence: float = 0.80,
    ) -> Dict[str, float]:
        """
        Return the expected ROI range for city × room_type at given confidence.

        Args:
            city:       City name.
            room_type:  Room type.
            confidence: Probability coverage (default 0.80 = ±1.28 std dev).

        Returns:
            {"low": float, "high": float, "mean": float, "confidence": float}
        """
        import math

        # Z-score for given coverage (symmetric two-tailed)
        # 0.80 → z=1.282, 0.90 → z=1.645, 0.95 → z=1.960
        z_map = {0.80: 1.282, 0.85: 1.440, 0.90: 1.645, 0.95: 1.960}
        z = z_map.get(confidence, 1.282)

        summary = self.get_benchmark_summary(city)
        rt = (room_type or "bedroom").lower().replace(" ", "_")
        bench = summary.get("benchmarks", {}).get(rt, {})

        if not bench:
            return {"low": 5.0, "high": 25.0, "mean": 12.0, "confidence": confidence}

        mean_ = bench["mean"]
        std_  = max(bench["std"], 0.5)

        return {
            "low":        round(max(0.0, mean_ - z * std_), 2),
            "high":       round(mean_ + z * std_, 2),
            "mean":       mean_,
            "confidence": confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────────────────────

_calibrator:     Optional[ROICalibrator]        = None
_nhb_validator:  Optional[NHBBenchmarkValidator] = None


def get_calibrator() -> ROICalibrator:
    """
    Return the singleton ROICalibrator, calibrating from real data on first call.
    Thread-safe for read access after first call.
    """
    global _calibrator
    if _calibrator is None:
        _calibrator = ROICalibrator()
        _calibrator.calibrate_from_real_data()
    return _calibrator


def get_nhb_validator() -> NHBBenchmarkValidator:
    """
    Return the singleton NHBBenchmarkValidator.
    Lightweight — no data loading required (all data is hardcoded from real stats).
    """
    global _nhb_validator
    if _nhb_validator is None:
        _nhb_validator = NHBBenchmarkValidator()
    return _nhb_validator