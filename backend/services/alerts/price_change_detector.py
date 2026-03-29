"""
ARKEN — Price Change Detector v1.0
=====================================
Monitors material price trends and generates actionable buy signals and
market summaries for the report generation agent.

Calls PriceForecastAgent.forecast_for_project() internally — does NOT
import or modify any agent file.

Design:
  - Stateless service — every call recomputes from latest forecast data.
  - Safe fallback: returns empty signals if forecast agent unavailable.
  - Used by report_agent_node.py to enrich final renovation reports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Urgency thresholds (% price change over 90 days)
_URGENT_PCT   = 10.0   # "high" urgency — buy immediately
_MODERATE_PCT = 5.0    # "medium" urgency — buy soon
_LOW_PCT      = 2.0    # "low" urgency — watch

# Supported room types (mirrors SEED_DATA relevance)
_ALL_ROOMS = ["kitchen", "bathroom", "bedroom", "living_room", "full_home"]


def _urgency(pct_change: float) -> str:
    abs_pct = abs(pct_change)
    if abs_pct >= _URGENT_PCT:
        return "high"
    if abs_pct >= _MODERATE_PCT:
        return "medium"
    if abs_pct >= _LOW_PCT:
        return "low"
    return "none"


class PriceChangeDetector:
    """
    Analyses material price forecasts and generates buy signals and
    market summaries. Stateless — safe to call from multiple threads.
    """

    def get_buy_signals(
        self,
        city: str,
        room_type: str = "full_home",
        area_sqft: float = 900.0,
    ) -> List[Dict[str, Any]]:
        """
        Return materials where buy_now_signal is True and price is rising.

        Args:
            city:       e.g. "Hyderabad"
            room_type:  filters to materials relevant to this room
            area_sqft:  project size for budget impact calculation

        Returns:
            List of signal dicts:
            {
                material:        str   (display name)
                material_key:    str   (internal key)
                current_price:   float (INR)
                forecast_90d:    float (INR, 90-day forecast)
                pct_change:      float (% change over 90 days)
                urgency_level:   str   ("high" | "medium" | "low" | "none")
                reason:          str   (human-readable signal reason)
                trend:           str   ("up" | "down" | "stable")
                data_quality:    str
            }
        """
        try:
            from agents.price_forecast import PriceForecastAgent
        except ImportError as exc:
            logger.warning(f"[PriceChangeDetector] Cannot import PriceForecastAgent: {exc}")
            return []

        try:
            agent = PriceForecastAgent()
            forecasts = agent.forecast_for_project(
                city=city,
                room_type=room_type,
                area_sqft=area_sqft,
            )
        except Exception as exc:
            logger.warning(f"[PriceChangeDetector] forecast_for_project failed: {exc}")
            return []

        signals: List[Dict[str, Any]] = []

        for item in forecasts:
            if not isinstance(item, dict):
                continue

            pct_change   = float(item.get("pct_change_90d", 0.0))
            buy_now      = bool(item.get("buy_now_signal", False))
            trend        = str(item.get("trend", "stable"))
            current      = float(item.get("current_price_inr", 0.0))
            forecast_90d = float(item.get("forecast_90d_inr", current))
            mat_key      = str(item.get("material_key", ""))
            mat_name     = str(item.get("display_name", mat_key))
            reason       = str(item.get("buy_signal_reason", item.get("trend_reason", "")))
            data_quality = str(item.get("data_quality", "unknown"))

            # Only surface actionable signals (price rising meaningfully)
            urgency = _urgency(pct_change)
            if urgency == "none" and not buy_now:
                continue

            signals.append({
                "material":     mat_name,
                "material_key": mat_key,
                "current_price": round(current, 2),
                "forecast_90d":  round(forecast_90d, 2),
                "pct_change":    round(pct_change, 2),
                "urgency_level": urgency,
                "reason":        reason or self._default_reason(trend, pct_change, mat_name),
                "trend":         trend,
                "data_quality":  data_quality,
            })

        # Sort: high urgency first, then by pct_change descending
        urgency_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        signals.sort(key=lambda x: (urgency_order.get(x["urgency_level"], 3), -x["pct_change"]))
        return signals

    @staticmethod
    def _default_reason(trend: str, pct_change: float, material: str) -> str:
        if trend == "up":
            return (
                f"{material} prices forecast to rise {pct_change:.1f}% "
                "over the next 90 days. Purchase now to lock in current rates."
            )
        if trend == "down":
            return (
                f"{material} prices may fall {abs(pct_change):.1f}% "
                "over the next 90 days. Consider deferring non-urgent purchases."
            )
        return f"{material} prices are stable. No urgency to purchase immediately."

    def get_market_summary(self, city: str) -> Dict[str, Any]:
        """
        Aggregate buy signals across all materials for a city.

        Returns:
            {
                city:                  str
                rising_count:          int
                falling_count:         int
                stable_count:          int
                top_buy_now_materials: List[str]   (top 3 urgent)
                market_sentiment:      str  ("bullish" | "neutral" | "bearish")
                total_materials:       int
                high_urgency_count:    int
                generated_at:          str  (ISO timestamp)
            }
        """
        generated_at = datetime.now(tz=timezone.utc).isoformat()

        try:
            from agents.price_forecast import PriceForecastAgent
            agent = PriceForecastAgent()
            forecasts = agent.forecast_for_project(
                city=city,
                room_type="full_home",
                area_sqft=1000.0,
            )
        except Exception as exc:
            logger.warning(f"[PriceChangeDetector] get_market_summary forecast failed: {exc}")
            return {
                "city":                  city,
                "rising_count":          0,
                "falling_count":         0,
                "stable_count":          0,
                "top_buy_now_materials": [],
                "market_sentiment":      "neutral",
                "total_materials":       0,
                "high_urgency_count":    0,
                "generated_at":          generated_at,
                "error":                 str(exc),
            }

        rising_count   = 0
        falling_count  = 0
        stable_count   = 0
        high_urgency   = 0
        buy_now_items: List[Dict] = []

        for item in forecasts:
            if not isinstance(item, dict):
                continue
            trend      = str(item.get("trend", "stable"))
            pct_change = float(item.get("pct_change_90d", 0.0))
            mat_name   = str(item.get("display_name", item.get("material_key", "")))
            buy_now    = bool(item.get("buy_now_signal", False))

            if trend == "up":
                rising_count += 1
            elif trend == "down":
                falling_count += 1
            else:
                stable_count += 1

            urg = _urgency(pct_change)
            if urg == "high":
                high_urgency += 1
            if buy_now and urg in ("high", "medium"):
                buy_now_items.append({"name": mat_name, "pct": pct_change, "urg": urg})

        total = rising_count + falling_count + stable_count

        # Sentiment: bullish if >40% rising, bearish if >40% falling, else neutral
        if total > 0:
            rising_pct = rising_count / total
            falling_pct = falling_count / total
            if rising_pct >= 0.4:
                sentiment = "bullish"
            elif falling_pct >= 0.4:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
        else:
            sentiment = "neutral"

        # Top 3 buy-now materials by urgency then pct_change
        buy_now_items.sort(key=lambda x: ({"high": 0, "medium": 1, "low": 2}.get(x["urg"], 3), -x["pct"]))
        top_buy_now = [x["name"] for x in buy_now_items[:3]]

        return {
            "city":                  city,
            "rising_count":          rising_count,
            "falling_count":         falling_count,
            "stable_count":          stable_count,
            "top_buy_now_materials": top_buy_now,
            "market_sentiment":      sentiment,
            "total_materials":       total,
            "high_urgency_count":    high_urgency,
            "generated_at":          generated_at,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_detector_instance: Optional[PriceChangeDetector] = None


def get_price_change_detector() -> PriceChangeDetector:
    """Return singleton PriceChangeDetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PriceChangeDetector()
    return _detector_instance
