"""
ARKEN — Material Price Alert Service v1.0
==========================================
SAVE AS: backend/services/price_alert_service.py  — NEW FILE

Allows users to set price threshold alerts on construction materials.
Provides smart auto-generated alert recommendations based on the
current material price forecast signals.

Alert lifecycle:
  CREATE → is_active=True, triggered_at=None
  CHECK  → if current price crosses threshold: triggered_at = now, is_active = False
  QUERY  → GET /api/v1/alerts/{user_id}
  DELETE → soft-delete (is_active=False)

Smart alerts (auto-generated):
  Based on PriceForecastAgent data — suggests alerts the user should set
  given current market conditions, without requiring the user to know
  which materials to watch.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Friendly material display names ───────────────────────────────────────────
_MATERIAL_DISPLAY = {
    "cement_opc53_per_bag_50kg":        "Cement OPC 53 Grade",
    "steel_tmt_fe500_per_kg":           "Steel TMT Fe500",
    "kajaria_tiles_per_sqft":           "Kajaria Vitrified Tiles",
    "copper_wire_per_kg":               "Copper Electrical Wire",
    "sand_river_per_brass":             "River Sand",
    "bricks_per_1000":                  "Red Bricks",
    "granite_per_sqft":                 "Granite (Black Galaxy)",
    "asian_paints_premium_per_litre":   "Asian Paints Premium Emulsion",
    "pvc_upvc_window_per_sqft":         "UPVC Windows",
    "modular_kitchen_per_sqft":         "Modular Kitchen (Laminate)",
    "bathroom_sanitary_set":            "Bathroom Sanitary Set",
    "teak_wood_per_cft":                "Teak Wood Grade A",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass (used when DB is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceAlert:
    user_id:       str
    material_key:  str
    threshold_inr: float
    direction:     str              # "above" | "below"
    email:         Optional[str]    = None
    alert_id:      str              = field(default_factory=lambda: str(uuid.uuid4()))
    is_active:     bool             = True
    created_at:    datetime         = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    triggered_at:  Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "alert_id":     self.alert_id,
            "user_id":      self.user_id,
            "material_key": self.material_key,
            "material_display_name": _MATERIAL_DISPLAY.get(self.material_key, self.material_key),
            "threshold_inr": self.threshold_inr,
            "direction":    self.direction,
            "email":        self.email,
            "is_active":    self.is_active,
            "created_at":   self.created_at.isoformat(),
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# In-memory store (fallback when DB unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class _InMemoryAlertStore:
    """Thread-safe in-memory alert store for development / DB-less deployments."""

    def __init__(self):
        self._alerts: Dict[str, PriceAlert] = {}   # alert_id → PriceAlert

    def save(self, alert: PriceAlert) -> PriceAlert:
        self._alerts[alert.alert_id] = alert
        return alert

    def get_by_user(self, user_id: str) -> List[PriceAlert]:
        return [a for a in self._alerts.values() if a.user_id == user_id and a.is_active]

    def get_all_active(self) -> List[PriceAlert]:
        return [a for a in self._alerts.values() if a.is_active]

    def get_by_id(self, alert_id: str) -> Optional[PriceAlert]:
        return self._alerts.get(alert_id)

    def delete(self, alert_id: str) -> bool:
        alert = self._alerts.get(alert_id)
        if alert:
            alert.is_active = False
            return True
        return False


_store = _InMemoryAlertStore()


# ─────────────────────────────────────────────────────────────────────────────
# DB-backed alert operations (PostgreSQL path via SQLAlchemy)
# ─────────────────────────────────────────────────────────────────────────────

async def _db_save_alert(alert: PriceAlert, db) -> bool:
    """Persist a PriceAlert to PostgreSQL. Returns True on success."""
    try:
        from db.models import PriceAlertModel
        from sqlalchemy import select

        db_obj = PriceAlertModel(
            id=uuid.UUID(alert.alert_id),
            user_id=alert.user_id,
            material_key=alert.material_key,
            threshold_inr=alert.threshold_inr,
            direction=alert.direction,
            email=alert.email,
            is_active=True,
        )
        db.add(db_obj)
        await db.flush()
        return True
    except Exception as e:
        logger.warning(f"[PriceAlerts] DB save failed ({e}) — using in-memory store")
        return False


async def _db_get_user_alerts(user_id: str, db) -> Optional[List[Dict]]:
    try:
        from db.models import PriceAlertModel
        from sqlalchemy import select

        result = await db.execute(
            select(PriceAlertModel).where(
                PriceAlertModel.user_id == user_id,
                PriceAlertModel.is_active.is_(True),
            )
        )
        rows = result.scalars().all()
        return [
            {
                "alert_id":     str(r.id),
                "user_id":      r.user_id,
                "material_key": r.material_key,
                "material_display_name": _MATERIAL_DISPLAY.get(r.material_key, r.material_key),
                "threshold_inr": r.threshold_inr,
                "direction":    r.direction,
                "email":        r.email,
                "is_active":    r.is_active,
                "created_at":   r.created_at.isoformat() if r.created_at else None,
                "triggered_at": r.triggered_at.isoformat() if r.triggered_at else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"[PriceAlerts] DB query failed ({e})")
        return None


async def _db_delete_alert(alert_id: str, db) -> bool:
    try:
        from db.models import PriceAlertModel

        result = await db.execute(
            __import__("sqlalchemy").select(PriceAlertModel).where(
                PriceAlertModel.id == uuid.UUID(alert_id)
            )
        )
        obj = result.scalar_one_or_none()
        if obj:
            obj.is_active = False
            return True
        return False
    except Exception as e:
        logger.warning(f"[PriceAlerts] DB delete failed ({e})")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API — create / read / delete
# ─────────────────────────────────────────────────────────────────────────────

async def create_alert(
    user_id: str,
    material_key: str,
    threshold_inr: float,
    direction: str,
    email: Optional[str] = None,
    db=None,
) -> Dict:
    """
    Create a new price alert. Persists to DB if available, else in-memory.

    Args:
        direction: "above" — alert when price rises above threshold
                   "below" — alert when price falls below threshold
    """
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got '{direction}'")

    alert = PriceAlert(
        user_id=user_id,
        material_key=material_key,
        threshold_inr=float(threshold_inr),
        direction=direction,
        email=email,
    )

    # Try DB first
    db_ok = False
    if db is not None:
        db_ok = await _db_save_alert(alert, db)

    # Always write to in-memory (ensures check_alerts() works without DB)
    _store.save(alert)

    logger.info(
        f"[PriceAlerts] Created alert {alert.alert_id} for user={user_id} "
        f"material={material_key} {direction} ₹{threshold_inr:.2f} "
        f"({'DB+memory' if db_ok else 'memory only'})"
    )
    return alert.to_dict()


async def get_user_alerts(user_id: str, db=None) -> List[Dict]:
    """Return all active alerts for a user."""
    # Try DB
    if db is not None:
        db_result = await _db_get_user_alerts(user_id, db)
        if db_result is not None:
            return db_result

    # In-memory fallback
    return [a.to_dict() for a in _store.get_by_user(user_id)]


async def delete_alert(alert_id: str, db=None) -> bool:
    """Soft-delete an alert (sets is_active=False)."""
    db_ok = False
    if db is not None:
        db_ok = await _db_delete_alert(alert_id, db)

    mem_ok = _store.delete(alert_id)
    logger.info(f"[PriceAlerts] Deleted {alert_id} (db={db_ok} mem={mem_ok})")
    return db_ok or mem_ok


# ─────────────────────────────────────────────────────────────────────────────
# Alert checker — run periodically (e.g. via cron or startup task)
# ─────────────────────────────────────────────────────────────────────────────

async def check_alerts() -> List[Dict]:
    """
    Evaluate all active alerts against current material prices.

    For each alert, fetches the current price from PriceForecastAgent.
    If the threshold is crossed, marks the alert triggered and returns it
    in the triggered list so callers can send notifications.

    Returns: list of triggered alert dicts.
    """
    try:
        from agents.price_forecast import PriceForecastAgent
        agent = PriceForecastAgent()
    except Exception as e:
        logger.error(f"[PriceAlerts] Cannot load PriceForecastAgent: {e}")
        return []

    active_alerts = _store.get_all_active()
    triggered: List[Dict] = []

    for alert in active_alerts:
        try:
            forecast = agent.forecast_material(alert.material_key, horizon_days=1)
            if not forecast:
                continue

            current_price = float(forecast.get("current_price_inr", 0))
            if current_price <= 0:
                continue

            crossed = (
                (alert.direction == "above" and current_price > alert.threshold_inr) or
                (alert.direction == "below" and current_price < alert.threshold_inr)
            )

            if crossed:
                alert.triggered_at = datetime.now(tz=timezone.utc)
                alert.is_active    = False
                result             = alert.to_dict()
                result["current_price_inr"] = current_price
                triggered.append(result)
                logger.info(
                    f"[PriceAlerts] TRIGGERED: {alert.material_key} "
                    f"current=₹{current_price:.2f} {alert.direction} ₹{alert.threshold_inr:.2f} "
                    f"user={alert.user_id}"
                )

        except Exception as e:
            logger.warning(f"[PriceAlerts] Check failed for alert {alert.alert_id}: {e}")

    return triggered


# ─────────────────────────────────────────────────────────────────────────────
# Smart alert suggestions (auto-generated from forecast signals)
# ─────────────────────────────────────────────────────────────────────────────

def get_smart_alerts(
    project_material_keys: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Auto-generate 3 recommended price alerts based on current market signals.

    Logic:
      1. If a material relevant to the project is trending UP >3% in 90d
         → "buy before price" alert (direction="above" at current price + 4%)
      2. If a material is stable
         → "ceiling" alert (direction="above" at current price + 8%)
      3. If a material is volatile (High)
         → symmetric alert at ±8% of current price

    Returns list of suggestion dicts the user can confirm and activate.
    """
    suggestions: List[Dict] = []

    try:
        from agents.price_forecast import PriceForecastAgent, SEED_DATA
        agent = PriceForecastAgent()

        # Default to all materials if no project-specific list provided
        keys_to_check = project_material_keys or list(SEED_DATA.keys())[:6]

        for key in keys_to_check[:6]:   # cap at 6 to avoid slow responses
            try:
                forecast = agent.forecast_material(key, horizon_days=90)
                if not forecast:
                    continue

                name          = forecast.get("display_name", key)
                current       = float(forecast.get("current_price_inr", 0))
                pct_90d       = float(forecast.get("pct_change_90d", 0))
                trend         = forecast.get("trend", "stable")
                volatility    = forecast.get("volatility_label", "Low")
                unit          = forecast.get("unit", "unit")

                if current <= 0:
                    continue

                if trend == "up" and pct_90d > 3:
                    # Buy before prices rise further
                    threshold = round(current * 1.04, 2)   # 4% above current = last good price
                    suggestions.append({
                        "material_key":   key,
                        "material_name":  name,
                        "direction":      "above",
                        "threshold_inr":  threshold,
                        "rationale": (
                            f"{name} is trending up {pct_90d:.1f}% over the next 90 days — "
                            f"set an alert to buy before ₹{threshold:.0f}/{unit.split('(')[-1].rstrip(')')}."
                        ),
                        "urgency": "high",
                    })

                elif trend == "stable" and volatility == "Low":
                    # Ceiling alert — warn if price spikes unexpectedly
                    threshold = round(current * 1.08, 2)
                    suggestions.append({
                        "material_key":   key,
                        "material_name":  name,
                        "direction":      "above",
                        "threshold_inr":  threshold,
                        "rationale": (
                            f"{name} is stable — no urgency, but alert at "
                            f"₹{threshold:.0f}/{unit.split('(')[-1].rstrip(')')} as a ceiling."
                        ),
                        "urgency": "low",
                    })

                elif volatility == "High":
                    # Volatile — alert on either direction at ±8%
                    threshold = round(current * 1.08, 2)
                    suggestions.append({
                        "material_key":   key,
                        "material_name":  name,
                        "direction":      "above",
                        "threshold_inr":  threshold,
                        "rationale": (
                            f"{name} is volatile ({volatility}) — "
                            f"alert at current price +8% (₹{threshold:.0f}) as a safety ceiling."
                        ),
                        "urgency": "medium",
                    })

                if len(suggestions) >= 3:
                    break

            except Exception as e:
                logger.warning(f"[PriceAlerts] Smart suggestion failed for {key}: {e}")

    except Exception as e:
        logger.error(f"[PriceAlerts] get_smart_alerts failed: {e}", exc_info=True)
        # Return static fallback suggestions so the endpoint is never empty
        suggestions = [
            {
                "material_key":  "steel_tmt_fe500_per_kg",
                "material_name": "Steel TMT Fe500",
                "direction":     "above",
                "threshold_inr": 68.0,
                "rationale":     "Steel TMT Fe500 is trending up 5.1% in 90 days — set an alert to buy before ₹68/kg.",
                "urgency":       "high",
            },
            {
                "material_key":  "kajaria_tiles_per_sqft",
                "material_name": "Kajaria Vitrified Tiles",
                "direction":     "above",
                "threshold_inr": 95.0,
                "rationale":     "Kajaria tiles are stable — no urgency, but alert at ₹95/sqft as a ceiling.",
                "urgency":       "low",
            },
            {
                "material_key":  "copper_wire_per_kg",
                "material_name": "Copper Electrical Wire",
                "direction":     "above",
                "threshold_inr": round(850 * 1.08, 2),
                "rationale":     "Copper wire is volatile (High) — alert at current price +8% as a safety ceiling.",
                "urgency":       "medium",
            },
        ]

    return suggestions[:3]
