"""
ARKEN — Live Price Fetcher v1.0
================================
Smart cache+staleness system for material prices using india_material_prices_historical.csv.

Features:
  - Load latest prices from historical CSV dataset
  - Linear drift correction based on material trend_slope
  - Staleness metadata (fresh/stale/very_stale)
  - Price alert status with buy-now recommendation
  - Singleton with LRU in-memory cache

All monetary values in INR.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Dataset path ──────────────────────────────────────────────────────────────
_DATASET_ROOT = Path(os.getenv("ARKEN_DATASET_DIR", "/app/data/datasets"))
_CSV_PATH = _DATASET_ROOT / "material_prices" / "india_material_prices_historical.csv"

# ── Freshness thresholds (days) ────────────────────────────────────────────────
FRESH_DAYS = 30
STALE_DAYS = 90

# ── Material trend slopes (annual fractional increase) ───────────────────────
# Source: SEED_DATA in price_forecast.py
MATERIAL_TREND_SLOPES: Dict[str, float] = {
    "cement_opc53_per_bag_50kg":      0.06,
    "steel_tmt_fe500_per_kg":         0.05,
    "teak_wood_per_cft":              0.04,
    "kajaria_tiles_per_sqft":         0.03,
    "copper_wire_per_kg":             0.10,
    "sand_river_per_brass":           0.09,
    "bricks_per_1000":                0.05,
    "granite_per_sqft":               0.04,
    "asian_paints_premium_per_litre": 0.04,
    "pvc_upvc_window_per_sqft":       0.05,
    "modular_kitchen_per_sqft":       0.07,
    "bathroom_sanitary_set":          0.05,
}

# ── Seed fallback prices (INR, Q1 2026 estimate) ─────────────────────────────
SEED_PRICES: Dict[str, float] = {
    "cement_opc53_per_bag_50kg":      400.0,
    "steel_tmt_fe500_per_kg":         65.0,
    "teak_wood_per_cft":              3000.0,
    "kajaria_tiles_per_sqft":         90.0,
    "copper_wire_per_kg":             850.0,
    "sand_river_per_brass":           3700.0,
    "bricks_per_1000":                9000.0,
    "granite_per_sqft":               195.0,
    "asian_paints_premium_per_litre": 350.0,
    "pvc_upvc_window_per_sqft":       950.0,
    "modular_kitchen_per_sqft":       1350.0,
    "bathroom_sanitary_set":          21000.0,
}

# ── City cost multipliers (from CITY_COST_MULTIPLIER in price_forecast.py) ───
CITY_MULTIPLIERS: Dict[str, float] = {
    "Mumbai": 1.25, "Delhi NCR": 1.20, "Bangalore": 1.15,
    "Hyderabad": 1.00, "Pune": 1.05, "Chennai": 1.05,
    "Kolkata": 0.95, "Ahmedabad": 0.92, "Surat": 0.90,
    "Jaipur": 0.88, "Lucknow": 0.85, "Chandigarh": 0.95,
    "Nagpur": 0.87, "Indore": 0.86, "Bhopal": 0.84,
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────────────────────────────────────

class _PriceDataCache:
    """Thread-safe in-memory cache for historical price data loaded from CSV."""

    def __init__(self):
        self._data: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def get_or_load(self) -> Dict[str, Any]:
        with self._lock:
            if self._data is not None:
                return self._data
            self._data = self._load_csv()
            return self._data

    @staticmethod
    def _load_csv() -> Dict[str, Any]:
        """
        Load the historical CSV and build a fast lookup structure.

        Returns:
            {
                "latest": {(material_key, city): (date, price)},
                "history": {(material_key, city): [(date, price), ...]},
                "loaded": bool,
                "csv_path": str,
            }
        """
        result: Dict[str, Any] = {
            "latest": {},
            "history": {},
            "loaded": False,
            "csv_path": str(_CSV_PATH),
        }

        csv_path = _CSV_PATH
        # Also try local fallback paths
        fallback_paths = [
            csv_path,
            Path("/app/data/datasets/material_prices/india_material_prices_historical.csv"),
            Path("data/datasets/material_prices/india_material_prices_historical.csv"),
        ]

        actual_path = None
        for p in fallback_paths:
            if p.exists():
                actual_path = p
                break

        if actual_path is None:
            logger.warning(f"[PriceFetcher] Historical CSV not found at {csv_path} — using seed prices.")
            return result

        try:
            import csv
            with open(actual_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mat_key = row.get("material_key", "").strip()
                    city = row.get("city", "").strip()
                    date_str = row.get("date", "").strip()
                    price_str = row.get("price_inr", "").strip()

                    if not all([mat_key, city, date_str, price_str]):
                        continue

                    try:
                        row_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        price = float(price_str)
                    except (ValueError, TypeError):
                        continue

                    key = (mat_key, city)
                    # Track history
                    if key not in result["history"]:
                        result["history"][key] = []
                    result["history"][key].append((row_date, price))

                    # Track latest
                    if key not in result["latest"] or row_date > result["latest"][key][0]:
                        result["latest"][key] = (row_date, price)

            result["loaded"] = True
            total_pairs = len(result["latest"])
            logger.info(f"[PriceFetcher] Loaded CSV: {total_pairs} material-city price pairs from {actual_path}")

        except Exception as e:
            logger.warning(f"[PriceFetcher] CSV load error: {e}")

        return result


_price_cache = _PriceDataCache()


# ─────────────────────────────────────────────────────────────────────────────
# Live Price Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class LivePriceFetcher:
    """
    Provides current material prices with staleness-awareness and trend extrapolation.

    Singleton — use LivePriceFetcher.get() to access shared instance.
    """

    _instance: Optional["LivePriceFetcher"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache = _price_cache

    @classmethod
    def get(cls) -> "LivePriceFetcher":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_current_prices(self, material_key: str, city: str) -> Dict[str, Any]:
        """
        Get current price for a material+city combination.

        Returns:
            {
                current_price_inr:   float,
                source:              "historical_csv" | "seed_extrapolated",
                confidence:          float (0.5–0.9),
                days_since_verified: int,
                trend_direction:     "up" | "stable" | "down",
                last_updated:        str (ISO date),
                price_freshness:     "fresh" | "stale" | "very_stale",
                city_multiplier:     float,
                material_key:        str,
                city:                str,
            }
        """
        data = self._cache.get_or_load()
        today = date.today()

        key = (material_key, city)
        city_multiplier = CITY_MULTIPLIERS.get(city, 1.0)

        if data["loaded"] and key in data["latest"]:
            last_date, last_price = data["latest"][key]
            days_since = (today - last_date).days

            # Apply drift correction
            corrected_price = self._apply_drift(material_key, last_price, days_since)

            # Apply city multiplier (CSV already has city-specific prices — no double-apply needed)
            # If the CSV was generated with city multipliers baked in, use corrected_price directly
            current_price = corrected_price

            # Trend direction from last 90 days
            trend = self._compute_trend(data, material_key, city, days=90)

            # Freshness
            freshness = self._freshness(days_since)
            confidence = 0.9 if freshness == "fresh" else (0.75 if freshness == "stale" else 0.60)

            return {
                "current_price_inr":   round(current_price, 2),
                "source":              "historical_csv",
                "confidence":          confidence,
                "days_since_verified": days_since,
                "trend_direction":     trend,
                "last_updated":        last_date.isoformat(),
                "price_freshness":     freshness,
                "city_multiplier":     city_multiplier,
                "material_key":        material_key,
                "city":                city,
            }

        # Fallback: seed price + city multiplier + extrapolation
        seed_price = SEED_PRICES.get(material_key)
        if seed_price is None:
            # Try partial match
            for k, v in SEED_PRICES.items():
                if material_key in k or k in material_key:
                    seed_price = v
                    break

        if seed_price is None:
            seed_price = 1000.0  # generic fallback

        # Apply city multiplier and extrapolate from known seed date (2024-01-01)
        seed_date = date(2024, 1, 1)
        days_since_seed = (today - seed_date).days
        extrapolated = self._apply_drift(material_key, seed_price * city_multiplier, days_since_seed)

        return {
            "current_price_inr":   round(extrapolated, 2),
            "source":              "seed_extrapolated",
            "confidence":          0.5,
            "days_since_verified": days_since_seed,
            "trend_direction":     "up" if MATERIAL_TREND_SLOPES.get(material_key, 0.05) > 0.03 else "stable",
            "last_updated":        seed_date.isoformat(),
            "price_freshness":     "very_stale",
            "city_multiplier":     city_multiplier,
            "material_key":        material_key,
            "city":                city,
        }

    def get_price_alert_status(
        self,
        material_key: str,
        city: str,
        user_target_price: float,
    ) -> Dict[str, Any]:
        """
        Evaluate whether now is a good time to buy given a user's target price.

        Returns:
            {
                should_buy_now:      bool,
                days_until_target:   int,
                alert_message:       str,
                current_price_inr:   float,
                target_price_inr:    float,
                price_gap_pct:       float,
            }
        """
        price_info = self.get_current_prices(material_key, city)
        current = price_info["current_price_inr"]
        trend = price_info["trend_direction"]
        freshness = price_info["price_freshness"]
        slope = MATERIAL_TREND_SLOPES.get(material_key, 0.05)

        gap = (current - user_target_price) / max(user_target_price, 1.0)
        gap_pct = round(gap * 100, 1)

        if current <= user_target_price:
            should_buy = True
            days_est = 0
            if trend == "up":
                msg = (
                    f"Price at ₹{current:,.0f} is AT or BELOW your target ₹{user_target_price:,.0f}. "
                    f"BUY NOW — price is trending up and may rise further."
                )
            else:
                msg = (
                    f"Price at ₹{current:,.0f} meets your target ₹{user_target_price:,.0f}. "
                    f"Good time to purchase."
                )
        elif trend == "down" or trend == "stable":
            # Estimate days to reach target price at current drift rate
            daily_rate = slope / 365.0
            if daily_rate > 0 and current > user_target_price:
                # Price going up — won't reach target, suggest waiting for seasonal dip
                days_est = 180  # estimate seasonal low cycle
                should_buy = False
                msg = (
                    f"Current price ₹{current:,.0f} is {abs(gap_pct):.1f}% above target ₹{user_target_price:,.0f}. "
                    f"Price trend is {trend}. Consider purchasing in ~{days_est} days during seasonal price dip."
                )
            else:
                days_est = 0
                should_buy = True
                msg = (
                    f"Price ₹{current:,.0f} is near target. Trend is {trend} — acceptable to buy now."
                )
        else:
            # Trending up, current above target
            daily_change = current * slope / 365.0
            if daily_change > 0:
                days_est = min(int(abs(current - user_target_price) / daily_change), 730)
            else:
                days_est = 365
            should_buy = gap_pct < 5.0  # buy if within 5% of target
            msg = (
                f"Price ₹{current:,.0f} is {abs(gap_pct):.1f}% above target ₹{user_target_price:,.0f}. "
                f"Price trending UP — target may not be reached. "
                + ("Consider buying now to avoid further increases." if should_buy else
                   f"Wait for market correction or revise target upward.")
            )

        if freshness in ("stale", "very_stale"):
            msg += f" (Note: price data is {freshness} — {price_info['days_since_verified']} days old.)"

        return {
            "should_buy_now":    should_buy,
            "days_until_target": days_est,
            "alert_message":     msg,
            "current_price_inr": current,
            "target_price_inr":  user_target_price,
            "price_gap_pct":     gap_pct,
        }

    def get_all_material_prices(self, city: str) -> Dict[str, Dict[str, Any]]:
        """Get current prices for all materials in SEED_PRICES for a given city."""
        return {
            mat_key: self.get_current_prices(mat_key, city)
            for mat_key in SEED_PRICES
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _apply_drift(self, material_key: str, last_price: float, days_elapsed: int) -> float:
        """Apply linear drift correction from last known price."""
        if days_elapsed <= 0:
            return last_price
        annual_slope = MATERIAL_TREND_SLOPES.get(material_key, 0.05)
        daily_rate = annual_slope / 365.0
        corrected = last_price * (1 + daily_rate * days_elapsed)
        return corrected

    @staticmethod
    def _compute_trend(
        data: Dict[str, Any],
        material_key: str,
        city: str,
        days: int = 90,
    ) -> str:
        """Compute price trend direction from last N days of history."""
        key = (material_key, city)
        history = data.get("history", {}).get(key, [])
        if len(history) < 2:
            return "stable"

        cutoff = date.today() - timedelta(days=days)
        recent = [(d, p) for d, p in history if d >= cutoff]

        if len(recent) < 2:
            # Use full history
            recent = sorted(history, key=lambda x: x[0])[-min(6, len(history)):]

        if len(recent) < 2:
            return "stable"

        recent.sort(key=lambda x: x[0])
        first_price = recent[0][1]
        last_price = recent[-1][1]

        if first_price <= 0:
            return "stable"

        change_pct = (last_price - first_price) / first_price * 100

        if change_pct > 3.0:
            return "up"
        elif change_pct < -3.0:
            return "down"
        return "stable"

    @staticmethod
    def _freshness(days_since: int) -> str:
        if days_since <= FRESH_DAYS:
            return "fresh"
        elif days_since <= STALE_DAYS:
            return "stale"
        return "very_stale"


# ── Convenience singleton accessor ────────────────────────────────────────────

def get_price_fetcher() -> LivePriceFetcher:
    """Return the singleton LivePriceFetcher instance."""
    return LivePriceFetcher.get()
