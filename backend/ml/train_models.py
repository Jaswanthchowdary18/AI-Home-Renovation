"""
ARKEN — ML Model Training Script
==================================
Trains all three property ML models on real housing datasets.

Usage:
    docker exec arken-backend python -m ml.train_models

What this does:
  1. Loads and unifies all housing datasets via HousingDataPreprocessor
  2. Trains PropertyValueModel (RF + GB + XGBoost + LightGBM ensemble)
  3. Trains RenovationCostModel (XGBoost + RF + LightGBM)
  4. Trains ROIModel (XGBoost + LightGBM + RF on real data)
  5. Saves weights to /app/ml/weights/

Without this script:
  - Models auto-train on first request (may take 30–120s)
  - This script pre-trains to avoid cold-start latency

Training time estimates (CPU):
  PropertyValueModel:  ~60s on 250K rows
  RenovationCostModel: ~30s on 30K rows
  ROIModel:            ~30s on 30K rows

Dataset requirements:
  At least one of:
    /app/data/datasets/india_housing_prices.csv  (recommended, 250K rows)
    /app/data/datasets/Housing.csv               (545 rows)
    /app/data/datasets/House Price India.csv     (14K rows)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def train_all():
    logger.info("=== ARKEN ML Model Training ===")
    t0 = time.perf_counter()

    # Check datasets
    from ml.housing_preprocessor import get_preprocessor
    prep = get_preprocessor()
    df   = prep.load_all()

    if len(df) < 100:
        logger.error(
            f"Only {len(df)} rows loaded. "
            "Ensure datasets are in /app/data/datasets/ — see README_DATASETS.md"
        )
        sys.exit(1)

    logger.info(f"Dataset loaded: {len(df):,} rows from sources: "
                f"{df['source'].value_counts().to_dict()}")

    # 1. Property Value Model
    logger.info("--- Training PropertyValueModel ---")
    from ml.property_models import PropertyValueModel
    pvm = PropertyValueModel()
    logger.info(f"✓ PropertyValueModel ready")

    # 2. Renovation Cost Model
    logger.info("--- Training RenovationCostModel ---")
    from ml.property_models import RenovationCostModel
    rcm = RenovationCostModel()
    logger.info("✓ RenovationCostModel ready")

    # 3. ROI Model
    logger.info("--- Training ROIModel ---")
    from ml.property_models import ROIModel
    roi = ROIModel()
    logger.info("✓ ROIModel ready")

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(f"=== All models trained in {elapsed}s ===")
    logger.info(f"Weights saved to: /app/ml/weights/")

    # Quick sanity check
    logger.info("--- Running sanity check predictions ---")
    from ml.property_models import get_ml_manager
    mgr = get_ml_manager()

    val = mgr.property_value.predict(size_sqft=1200, city_tier=1, age_years=8, bhk=3)
    logger.info(f"PropertyValue (1200sqft, Tier1, 8yr): ₹{val['value_inr']:,} "
                f"(±{val['price_per_sqft']:,}/sqft)")

    cost = mgr.renovation_cost.predict(
        room_type="living_room", budget_tier="mid",
        area_sqft=250, city_tier=1, scope="partial"
    )
    logger.info(f"RenovationCost (living_room, mid, 250sqft): ₹{cost['renovation_cost_inr']:,}")

    import pandas as pd
    X_roi = pd.DataFrame([{
        "renovation_cost_lakh": 5.0, "size_sqft": 1200, "city_tier": 1,
        "room_type_enc": 2, "budget_tier_enc": 1, "age_years": 10,
        "furnished": 1, "reno_intensity": 0.04, "scope_enc": 1,
        "amenity_count": 3, "has_parking": 1,
    }])
    roi_mean, roi_low, roi_high = mgr.roi.predict(X_roi)
    logger.info(f"ROI (bathroom, mid, Tier1): {roi_mean:.1f}% [{roi_low:.1f}–{roi_high:.1f}%]")

    logger.info("=== Training and validation complete ===")


if __name__ == "__main__":
    train_all()
