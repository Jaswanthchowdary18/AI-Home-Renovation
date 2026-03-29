"""
ARKEN — Price Alert API Routes v1.0
=====================================
SAVE AS: backend/api/routes/alerts.py  — NEW FILE

Endpoints:
  POST   /api/v1/alerts/              — create alert
  GET    /api/v1/alerts/{user_id}     — get all active alerts for a user
  DELETE /api/v1/alerts/{alert_id}    — soft-delete alert
  GET    /api/v1/alerts/smart-suggestions/{project_id}
                                      — AI-recommended alerts for this project
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, EmailStr

from db.session import AsyncSession, get_db
from services.price_alert_service import (
    create_alert,
    delete_alert,
    get_smart_alerts,
    get_user_alerts,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request models ─────────────────────────────────────────────────────────────

class CreateAlertRequest(BaseModel):
    user_id:       str
    material_key:  str = Field(
        description="Key from SEED_DATA in price_forecast.py, e.g. 'steel_tmt_fe500_per_kg'"
    )
    threshold_inr: float = Field(gt=0, description="Price threshold in INR")
    direction:     str   = Field(
        description="'above' — alert when price rises above threshold; "
                    "'below' — alert when price falls below threshold"
    )
    email: Optional[str] = Field(
        default=None,
        description="Email address to notify when alert triggers (optional)"
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/",
    summary="Create a material price alert",
    status_code=status.HTTP_201_CREATED,
)
async def create_price_alert(
    req: CreateAlertRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Set a threshold alert on any construction material.

    When the current market price crosses the threshold (above or below),
    the alert is marked as triggered. Use GET /{user_id} to poll for
    triggered alerts or integrate with a notification webhook.
    """
    if req.direction not in ("above", "below"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="direction must be 'above' or 'below'",
        )

    try:
        alert = await create_alert(
            user_id=req.user_id,
            material_key=req.material_key,
            threshold_inr=req.threshold_inr,
            direction=req.direction,
            email=req.email,
            db=db,
        )
        return alert
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"[Alerts] create_alert failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert creation failed: {e}",
        )


@router.get(
    "/smart-suggestions/{project_id}",
    summary="Get AI-recommended price alerts for this project",
)
async def smart_suggestions(project_id: str):
    """
    Returns up to 3 recommended price alerts auto-generated from current
    material price forecasts. Suggestions are based on trend direction,
    volatility, and 90-day price change signals.

    Example output:
      "Steel prices are trending up 5.1% — set an alert before ₹68/kg"
      "Kajaria tiles stable — ceiling alert at ₹95/sqft"
      "Copper wire volatile — alert at +8% of current price"
    """
    try:
        suggestions = get_smart_alerts()
        return {
            "project_id":  project_id,
            "suggestions": suggestions,
            "count":       len(suggestions),
            "note": (
                "These alerts are auto-recommended based on current material price trends. "
                "Confirm each one to activate it."
            ),
        }
    except Exception as e:
        logger.error(f"[Alerts] smart_suggestions failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Smart suggestions failed: {e}",
        )


@router.get(
    "/{user_id}",
    summary="Get all active price alerts for a user",
)
async def get_alerts(user_id: str, db: AsyncSession = Depends(get_db)):
    """Return all active (not yet triggered or deleted) alerts for the given user."""
    try:
        alerts = await get_user_alerts(user_id, db=db)
        return {
            "user_id": user_id,
            "alerts":  alerts,
            "count":   len(alerts),
        }
    except Exception as e:
        logger.error(f"[Alerts] get_user_alerts failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert query failed: {e}",
        )


@router.delete(
    "/{alert_id}",
    summary="Delete (deactivate) a price alert",
)
async def delete_price_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    """Soft-delete an alert. The alert record is preserved but marked inactive."""
    try:
        deleted = await delete_alert(alert_id, db=db)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found",
            )
        return {"deleted": True, "alert_id": alert_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Alerts] delete_alert failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert deletion failed: {e}",
        )
