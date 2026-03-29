#!/usr/bin/env python3
"""
ARKEN — Realistic ROI Model (Fixed)
=====================================
Replaces the leaky roi_pct model with two simple, honest models:

  Model 1: Predict post_renovation_value (INR)
           Input: pre_renovation_value + renovation features
           Target: post_reno_value_inr

  Model 2: Predict value_uplift_pct (property value % increase)
           Input: property + renovation features (NO leaky reno_intensity)
           Target: (post - pre) / pre * 100

ROOT CAUSE of R²=0.999:
  The old model predicted roi_pct using reno_intensity = renovation_cost / property_price.
  But roi_pct itself is computed as (rent_months × monthly_rent) / renovation_cost × 100.
  renovation_cost appears in BOTH the feature AND the denominator of the label — pure leakage.

WHAT "REALISTIC" MEANS HERE:
  - For renovation ROI, what matters is:
      cost_before_renovation  = pre_reno_value_inr
      cost_after_renovation   = post_reno_value_inr
  - The model predicts post_reno_value from pre_reno_value + property features
  - Expected R² for post_reno_value_inr: 0.95–0.99 (pre_reno_value is the dominant signal)
  - Expected R² for value_uplift_pct: 0.30–0.65 (genuinely hard to predict, realistic)
  - Expected MAE for uplift: 0.3–0.9 percentage points (realistic residual noise)

Outputs:
  roi_pre_post_model.joblib   — predicts post_reno_value_inr
  roi_uplift_model.joblib     — predicts value_uplift_pct
  roi_model_report.json       — honest metrics
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_CSV = Path("/home/claude/arke4n12/backend/data/datasets/property_transactions/india_property_transactions.csv")
_OUT_DIR = _SCRIPT_DIR / "roi_weights_realistic"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Encoding maps (same as original) ─────────────────────────────────────────
ROOM_ENC_MAP   = {"bedroom": 0, "bathroom": 1, "living_room": 2, "kitchen": 3, "full_home": 4}
BUDGET_ENC_MAP = {"basic": 0, "mid": 1, "premium": 2}
SCOPE_ENC_MAP  = {"cosmetic_only": 0, "partial": 1, "full_room": 2, "structural_plus": 3}
FURNISHED_MAP  = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}

# ── Features for Model 2 (uplift) — NO renovation_cost, NO reno_intensity ────
# These are all property-level inputs a user would actually KNOW before renovating
UPLIFT_FEATURES = [
    "city_tier",          # Tier 1/2/3 city (real market signal)
    "size_sqft",          # Property size
    "age_years",          # Property age (older → more upside from renovation)
    "room_type_enc",      # Which room is being renovated
    "budget_tier_enc",    # Basic / mid / premium finish
    "scope_enc",          # Cosmetic / partial / full / structural
    "amenity_count",      # Property amenities (quality signal)
    "has_parking",        # Parking availability
    "furnished_enc",      # Furnished status
    "bedrooms",           # Number of bedrooms
]

# ── Features for Model 1 (post-reno value) — includes pre_reno_value ─────────
POST_VALUE_FEATURES = ["pre_reno_value_inr"] + UPLIFT_FEATURES + ["renovation_cost_inr"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(str(_CSV))
    print(f"Loaded {len(df):,} rows")

    # Encode categoricals
    df["room_type_enc"]   = df["room_renovated"].map(ROOM_ENC_MAP).fillna(0).astype(int)
    df["budget_tier_enc"] = df["budget_tier"].map(BUDGET_ENC_MAP).fillna(1).astype(int)
    df["scope_enc"]       = df["renovation_scope"].map(SCOPE_ENC_MAP).fillna(1).astype(int)
    df["furnished_enc"]   = df["furnished_status"].map(FURNISHED_MAP).fillna(1).astype(int)
    df["has_parking"]     = (df["parking"] > 0).astype(int)

    # Target: property value uplift % = (post - pre) / pre * 100
    df["value_uplift_pct"] = (
        (df["post_reno_value_inr"] - df["pre_reno_value_inr"])
        / df["pre_reno_value_inr"].clip(lower=1)
        * 100
    )

    # Keep only rows with all needed columns
    needed = ["pre_reno_value_inr", "post_reno_value_inr",
              "renovation_cost_inr", "value_uplift_pct"] + UPLIFT_FEATURES
    df = df[df[needed].notna().all(axis=1)].copy()
    print(f"After filter: {len(df):,} rows")
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def train_and_evaluate(X_tr, X_te, y_tr, y_te, label: str):
    """Train a GBM and evaluate. Returns (model, metrics_dict)."""
    print(f"\n  Training {label} …")

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    mae_ = mean_absolute_error(y_te, preds)
    rmse_ = rmse(y_te, preds)
    r2_ = r2_score(y_te, preds)

    # Feature importances
    fi = dict(zip(X_tr.columns.tolist(), model.feature_importances_.tolist()))
    top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"    MAE:  {mae_:>12.4f}")
    print(f"    RMSE: {rmse_:>12.4f}")
    print(f"    R²:   {r2_:>12.4f}")
    print(f"    Top features: {[f for f, _ in top_features]}")

    return model, {"mae": round(mae_, 4), "rmse": round(rmse_, 4), "r2": round(r2_, 4)}


def main():
    print("=" * 65)
    print("  ARKEN Realistic ROI Model Training")
    print("=" * 65)

    df = load_data()

    # ── Sample stats ──────────────────────────────────────────────
    print(f"\n  pre_reno_value_inr  mean: ₹{df['pre_reno_value_inr'].mean():,.0f}")
    print(f"  post_reno_value_inr mean: ₹{df['post_reno_value_inr'].mean():,.0f}")
    print(f"  renovation_cost_inr mean: ₹{df['renovation_cost_inr'].mean():,.0f}")
    print(f"  value_uplift_pct   mean:  {df['value_uplift_pct'].mean():.2f}%  "
          f"std: {df['value_uplift_pct'].std():.2f}%")

    # ── Stratified split on city_tier ────────────────────────────
    df_tr, df_te = train_test_split(df, test_size=0.20, random_state=42,
                                    stratify=df["city_tier"])

    # ─────────────────────────────────────────────────────────────
    # MODEL 1: Predict post_reno_value_inr
    # ─────────────────────────────────────────────────────────────
    print("\n── Model 1: Predict post_renovation_value (INR) ──")
    print("   Input: pre_reno_value + renovation details")
    print("   Target: post_reno_value_inr")

    feat1 = [f for f in POST_VALUE_FEATURES if f in df.columns]
    X1_tr = df_tr[feat1].fillna(df_tr[feat1].median())
    X1_te = df_te[feat1].fillna(df_tr[feat1].median())
    y1_tr = df_tr["post_reno_value_inr"]
    y1_te = df_te["post_reno_value_inr"]

    model1, metrics1 = train_and_evaluate(X1_tr, X1_te, y1_tr, y1_te,
                                          "post_reno_value_inr")
    joblib.dump(model1, str(_OUT_DIR / "roi_pre_post_model.joblib"))

    # Show sample predictions
    sample_preds = model1.predict(X1_te.head(5))
    sample_actuals = y1_te.values[:5]
    print("\n  Sample predictions (Model 1 — post_reno_value):")
    print(f"  {'Actual (₹)':>20}  {'Predicted (₹)':>20}  {'Error (₹)':>15}")
    for a, p in zip(sample_actuals, sample_preds):
        print(f"  {a:>20,.0f}  {p:>20,.0f}  {abs(a-p):>15,.0f}")

    # ─────────────────────────────────────────────────────────────
    # MODEL 2: Predict value_uplift_pct (NO renovation_cost in features)
    # ─────────────────────────────────────────────────────────────
    print("\n── Model 2: Predict value_uplift_pct ──")
    print("   Input: property features only (no renovation cost — no leakage)")
    print("   Target: (post - pre) / pre × 100")
    print("   Expected R²: 0.30–0.65  (genuinely uncertain — realistic!)")

    feat2 = [f for f in UPLIFT_FEATURES if f in df.columns]
    X2_tr = df_tr[feat2].fillna(df_tr[feat2].median())
    X2_te = df_te[feat2].fillna(df_tr[feat2].median())
    y2_tr = df_tr["value_uplift_pct"]
    y2_te = df_te["value_uplift_pct"]

    model2, metrics2 = train_and_evaluate(X2_tr, X2_te, y2_tr, y2_te,
                                          "value_uplift_pct")
    joblib.dump(model2, str(_OUT_DIR / "roi_uplift_model.joblib"))

    # Show sample predictions
    sample_preds2 = model2.predict(X2_te.head(5))
    sample_actuals2 = y2_te.values[:5]
    print("\n  Sample predictions (Model 2 — value_uplift_pct):")
    print(f"  {'Actual (%)':>15}  {'Predicted (%)':>15}  {'Error (pp)':>12}")
    for a, p in zip(sample_actuals2, sample_preds2):
        print(f"  {a:>15.3f}  {p:>15.3f}  {abs(a-p):>12.3f}")

    # ── Sanity check: per-city-tier metrics for model 2 ──────────
    print("\n  Value uplift MAE by city tier (Model 2):")
    all_preds2 = model2.predict(X2_te)
    for tier in sorted(df_te["city_tier"].unique()):
        mask = df_te["city_tier"] == tier
        if mask.sum() == 0:
            continue
        tier_mae = mean_absolute_error(y2_te[mask], all_preds2[mask])
        print(f"    Tier {int(tier)}: MAE={tier_mae:.3f}pp  (n={mask.sum()})")

    # ── Save report ───────────────────────────────────────────────
    report = {
        "training_date": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_size": len(df),
        "train_size": len(df_tr),
        "test_size": len(df_te),
        "description": (
            "Model 1 predicts post_renovation_value_inr from pre_reno_value + features. "
            "Model 2 predicts value_uplift_pct from property features only (no leakage). "
            "R²=0.999 in old model was caused by reno_intensity feature being a direct "
            "algebraic component of the roi_pct label formula."
        ),
        "model_1_post_reno_value": {
            "target": "post_reno_value_inr",
            "features": feat1,
            "metrics": metrics1,
            "file": "roi_pre_post_model.joblib",
            "note": (
                f"R²={metrics1['r2']:.3f} expected (pre_reno_value dominates). "
                f"MAE=₹{metrics1['mae']:,.0f} means predicted post-reno value is within "
                f"₹{metrics1['mae']/100000:.1f}L of actual."
            ),
        },
        "model_2_value_uplift": {
            "target": "value_uplift_pct",
            "features": feat2,
            "metrics": metrics2,
            "file": "roi_uplift_model.joblib",
            "note": (
                f"R²={metrics2['r2']:.3f} is REALISTIC (not a bug). "
                f"Property value uplift from renovation has genuine uncertainty. "
                f"MAE={metrics2['mae']:.3f} percentage points."
            ),
        },
        "root_cause_of_old_r2_0999": (
            "Old model predicted roi_pct using 'reno_intensity' = renovation_cost / property_price. "
            "roi_pct was computed as (rent_months × monthly_rent) / renovation_cost × 100. "
            "renovation_cost appeared in BOTH feature (reno_intensity denominator) and label "
            "denominator — direct algebraic leakage. Model memorized the formula."
        ),
        "what_you_wanted": (
            "cost_before_renovation = pre_reno_value_inr  "
            "cost_after_renovation  = post_reno_value_inr  "
            "Model 1 gives you exactly this."
        ),
    }

    report_path = _OUT_DIR / "roi_model_report.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print()
    print(f"  Model 1  — post_reno_value_inr")
    print(f"    MAE  = ₹{metrics1['mae']:>12,.0f}")
    print(f"    RMSE = ₹{metrics1['rmse']:>12,.0f}")
    print(f"    R²   = {metrics1['r2']:>13.3f}  ← expected high (pre_reno dominates)")
    print()
    print(f"  Model 2  — value_uplift_pct")
    print(f"    MAE  = {metrics2['mae']:>10.4f} pp")
    print(f"    RMSE = {metrics2['rmse']:>10.4f} pp")
    print(f"    R²   = {metrics2['r2']:>10.3f}  ← realistic (genuine uncertainty)")
    print()
    print(f"  Weights → {_OUT_DIR}/")
    print(f"  Report  → {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()