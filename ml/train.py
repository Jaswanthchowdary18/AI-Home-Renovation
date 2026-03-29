"""
ARKEN — ML Training Pipeline v2.0
====================================
Train and serialise the ROI ensemble + Prophet/XGBoost price models.
ALL SYNTHETIC DATA GENERATION HAS BEEN REMOVED.

v2.0 Changes over v1.0:
  - generate_roi_dataset() DELETED. No np.random, no synthetic rows.
  - load_real_roi_data() loads india_property_transactions.csv via
    RenovationDataPreprocessor. Raises DataError if < 500 rows.
  - walk_forward_cross_validation() uses temporal folds (split by
    transaction_date, oldest → newest) to avoid data leakage.
  - train_price_models_from_real_data() fits Prophet + XGBoost per
    material×city from india_material_prices_historical.csv.
  - model_report.json: dataset_size, training_date, feature_cols,
    cv_results, city_coverage, model_versions.
  - All random seeds fixed at 42 for reproducibility.
  - Human-readable summary table printed after training.

Usage:
    cd backend && python ../ml/train.py --model roi
    cd backend && python ../ml/train.py --model price
    cd backend && python ../ml/train.py --model all
    cd backend && python ../ml/train.py --model all --csv-path /custom/path.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arken.train")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ML_DIR      = Path(__file__).resolve().parent          # ml/
_BACKEND_DIR = _ML_DIR.parent / "backend"               # backend/ (if running from repo root)

# Handle both run-from-backend and run-from-repo-root layouts
if not (_BACKEND_DIR / "ml").exists():
    _BACKEND_DIR = Path(__file__).resolve().parent.parent  # ml/../ = backend/

_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))
_WEIGHTS_DIR = Path(os.getenv("ML_WEIGHTS_DIR", str(_APP_DIR / "ml" / "weights")))

# Fallback for local dev
if not _WEIGHTS_DIR.exists() and not _WEIGHTS_DIR.parent.exists():
    _WEIGHTS_DIR = _BACKEND_DIR / "ml" / "weights"

_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_data(app_rel: str, local_rel: str) -> Path:
    a = _APP_DIR    / app_rel
    b = _BACKEND_DIR / local_rel
    return a if a.exists() else b

_PROPERTY_CSV = _resolve_data(
    "data/datasets/property_transactions/india_property_transactions.csv",
    "data/datasets/property_transactions/india_property_transactions.csv",
)
_MATERIAL_CSV = _resolve_data(
    "data/datasets/material_prices/india_material_prices_historical.csv",
    "data/datasets/material_prices/india_material_prices_historical.csv",
)

# Add backend to path so we can import project modules
sys.path.insert(0, str(_BACKEND_DIR))

# ── Random seed ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── FEATURE_COLS (must match housing_preprocessor.FEATURE_COLS) ───────────────
FEATURE_COLS = [
    "renovation_cost_lakh", "size_sqft", "city_tier",
    "room_type_enc", "budget_tier_enc", "age_years",
    "furnished", "reno_intensity", "scope_enc",
    "amenity_count", "has_parking",
]

# ── Encoding maps (must match housing_preprocessor.py exactly) ─────────────────
ROOM_ENC_MAP    = {"bedroom": 0, "bathroom": 1, "living_room": 2, "kitchen": 3, "full_home": 4}
BUDGET_ENC_MAP  = {"basic": 0, "mid": 1, "premium": 2}
SCOPE_ENC_MAP   = {"cosmetic_only": 0, "partial": 1, "full_room": 2, "structural_plus": 3}
FURNISHED_MAP   = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}

# ── Minimum dataset size guard ─────────────────────────────────────────────────
_MIN_ROI_ROWS     = 500
_MIN_PRICE_ROWS   = 50    # per material×city combo (after interpolation)

# ── Price model output dirs ─────────────────────────────────────────────────────
_PROPHET_DIR  = _WEIGHTS_DIR / "prophet_models"
_PROPHET_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class DataError(ValueError):
    """Raised when real data is missing or below minimum quality threshold."""


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# Real data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_real_roi_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and preprocess real renovation data from india_property_transactions.csv.

    Uses RenovationDataPreprocessor from housing_preprocessor.py when available
    (production path). Falls back to direct CSV load + feature engineering when
    running the script standalone outside the FastAPI app.

    Raises:
        DataError: If CSV is missing, unreadable, or has fewer than _MIN_ROI_ROWS rows.

    Returns:
        Preprocessed DataFrame with FEATURE_COLS + roi_pct available.
    """
    target_csv = csv_path or _PROPERTY_CSV

    if not target_csv.exists():
        raise DataError(
            f"india_property_transactions.csv not found at:\n  {target_csv}\n\n"
            "To generate it from real Kaggle data, run:\n"
            "  cd backend\n"
            "  python data/datasets/property_transactions/build_real_roi_dataset.py\n\n"
            "That script reads:\n"
            "  backend/data/datasets/india_housing_prices/{City}.csv  (32,963 rows)\n"
            "  backend/data/datasets/House Price India/House_Rent_Dataset.csv (4,746 rows)\n"
            "and derives every roi_pct label from real observed rental yields."
        )

    # Try RenovationDataPreprocessor first (canonical path)
    try:
        from ml.housing_preprocessor import RenovationDataPreprocessor, FEATURE_COLS as HP_FEATURES
        rp = RenovationDataPreprocessor()
        rp._PROPERTY_TRANSACTIONS_CSV_OVERRIDE = target_csv   # allow override
        df = rp.load()
        if len(df) >= _MIN_ROI_ROWS:
            logger.info(f"[load_real_roi_data] Loaded {len(df):,} rows via RenovationDataPreprocessor")
            return df
    except Exception as e:
        logger.warning(f"[load_real_roi_data] RenovationDataPreprocessor failed ({e}), loading CSV directly")

    # Direct CSV load + feature engineering (standalone mode)
    try:
        df = pd.read_csv(str(target_csv))
    except Exception as e:
        raise DataError(f"Failed to read CSV at {target_csv}: {e}") from e

    numeric_cols = [
        "size_sqft", "age_years", "bedrooms", "floor_number", "total_floors",
        "parking", "amenity_count", "city_tier",
        "transaction_price_inr", "price_per_sqft",
        "pre_reno_value_inr", "post_reno_value_inr", "renovation_cost_inr",
        "roi_pct", "rental_yield_pct", "payback_months",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering
    df["renovation_cost_lakh"] = df["renovation_cost_inr"] / 100_000
    df["reno_intensity"] = (
        df["renovation_cost_inr"] / df["transaction_price_inr"].clip(lower=1)
    ).clip(upper=0.5)
    df["room_type_enc"]   = df["room_renovated"].map(ROOM_ENC_MAP).fillna(0).astype(float)
    df["budget_tier_enc"] = df["budget_tier"].map(BUDGET_ENC_MAP).fillna(1).astype(float)
    df["scope_enc"]       = df["renovation_scope"].map(SCOPE_ENC_MAP).fillna(1).astype(float)
    df["furnished"]       = df["furnished_status"].map(FURNISHED_MAP).fillna(1).astype(float)
    df["has_parking"]     = (df["parking"] > 0).astype(int)

    roi_rows = int(df["roi_pct"].notna().sum())
    if roi_rows < _MIN_ROI_ROWS:
        raise DataError(
            f"Only {roi_rows} rows with valid roi_pct found in {target_csv} "
            f"(minimum required: {_MIN_ROI_ROWS}).\n"
            "Run build_real_roi_dataset.py to regenerate from the Kaggle housing datasets."
        )

    # Guard against synthetic data
    if "data_source" in df.columns:
        bad = [s for s in df["data_source"].dropna().str.lower().unique() if "synthetic" in s]
        if bad:
            raise DataError(
                f"SYNTHETIC DATA detected in 'data_source': {bad}. "
                "Regenerate the CSV from real data before training."
            )

    logger.info(
        f"[load_real_roi_data] {len(df):,} total rows | "
        f"{roi_rows:,} rows with roi_pct | "
        f"roi range: [{df['roi_pct'].min():.1f}, {df['roi_pct'].max():.1f}]%"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Temporal walk-forward cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cross_validation(
    df_reno: pd.DataFrame,
    models: List,
    feature_cols: List[str],
    n_folds: int = 5,
) -> Dict[str, Any]:
    """
    Temporal walk-forward cross-validation for ROI ensemble.

    Splits data chronologically (oldest → newest) into n_folds.
    For fold k: train on folds 0..k-1, validate on fold k.
    This prevents data leakage that random KFold would cause with time-series data.

    Args:
        df_reno:      Rows with roi_pct not null, indexed by date.
        models:       List of trained sklearn estimators (XGB, RF, GBM).
        feature_cols: Features to use.
        n_folds:      Number of temporal folds (default 5).

    Returns:
        Dict with per-fold and overall MAE/RMSE/R², saved to roi_cv_results.json.
    """
    from sklearn.metrics import r2_score

    # Sort by transaction_date for temporal ordering
    if "transaction_date" in df_reno.columns:
        try:
            df_sorted = df_reno.sort_values("transaction_date").reset_index(drop=True)
        except Exception:
            df_sorted = df_reno.copy().reset_index(drop=True)
    else:
        df_sorted = df_reno.copy().reset_index(drop=True)

    avail_feats = [c for c in feature_cols if c in df_sorted.columns]
    X = df_sorted[avail_feats].copy()
    y = df_sorted["roi_pct"].copy()

    # Fill NaNs
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    n = len(X)
    fold_size   = n // (n_folds + 1)          # +1 so fold 0 always has training data
    fold_results = []

    logger.info(f"[walk_forward_cv] {n:,} rows | {n_folds} folds | fold_size≈{fold_size:,}")

    for k in range(1, n_folds + 1):
        train_end = k * fold_size
        val_start = train_end
        val_end   = min((k + 1) * fold_size, n)

        if train_end < 50 or val_end - val_start < 10:
            logger.debug(f"[walk_forward_cv] Fold {k} too small, skipping")
            continue

        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_val = X.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]

        # Retrain each model on the temporal training slice
        fold_preds_all = []
        for mdl in models:
            try:
                mdl_clone = _clone_model(mdl)
                mdl_clone.fit(X_tr, y_tr)
                fold_preds_all.append(mdl_clone.predict(X_val))
            except Exception as e:
                logger.debug(f"[walk_forward_cv] Fold {k} model {type(mdl).__name__} failed: {e}")

        if not fold_preds_all:
            continue

        ens_preds = np.mean(fold_preds_all, axis=0)
        y_val_arr = y_val.values

        fold_mae  = _mae(y_val_arr, ens_preds)
        fold_rmse = _rmse(y_val_arr, ens_preds)
        try:
            fold_r2 = float(r2_score(y_val_arr, ens_preds))
        except Exception:
            fold_r2 = float("nan")

        fold_results.append({
            "fold":            k,
            "train_rows":      int(train_end),
            "val_rows":        int(val_end - val_start),
            "mae_pct":         round(fold_mae, 4),
            "rmse_pct":        round(fold_rmse, 4),
            "r2":              round(fold_r2, 4) if not np.isnan(fold_r2) else None,
        })
        logger.info(
            f"  Fold {k}: train={train_end:,} val={val_end-val_start:,} | "
            f"MAE={fold_mae:.3f}% RMSE={fold_rmse:.3f}% R²={fold_r2:.3f}"
        )

    if not fold_results:
        return {"error": "No folds completed successfully"}

    overall_mae  = float(np.mean([f["mae_pct"]  for f in fold_results]))
    overall_rmse = float(np.mean([f["rmse_pct"] for f in fold_results]))
    overall_r2   = float(np.mean([f["r2"]       for f in fold_results if f["r2"] is not None]))

    cv_results = {
        "method":        "walk_forward_temporal",
        "n_folds":       n_folds,
        "overall_mae":   round(overall_mae, 4),
        "overall_rmse":  round(overall_rmse, 4),
        "overall_r2":    round(overall_r2, 4),
        "fold_results":  fold_results,
    }

    # Save to weights dir
    cv_path = _WEIGHTS_DIR / "roi_cv_results.json"
    try:
        with open(cv_path, "w", encoding="utf-8") as fh:
            json.dump(cv_results, fh, indent=2)
        logger.info(f"[walk_forward_cv] CV results saved → {cv_path}")
    except Exception as e:
        logger.warning(f"[walk_forward_cv] Could not save CV results: {e}")

    return cv_results


def _clone_model(model):
    """Clone an sklearn estimator without copying fitted state."""
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        import copy
        return copy.deepcopy(model)


# ─────────────────────────────────────────────────────────────────────────────
# ROI model training
# ─────────────────────────────────────────────────────────────────────────────

def train_roi_model(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Train XGBoost + RandomForest + GradientBoosting ensemble on real data.
    Runs walk-forward cross-validation and saves model_report.json.

    Returns:
        model_report dict (also saved to ml/weights/model_report.json).
    """
    try:
        import joblib
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(f"scikit-learn / joblib required: pip install scikit-learn joblib | {e}")

    logger.info("=" * 60)
    logger.info("[train_roi] Loading real renovation data …")

    # ── Load real data ─────────────────────────────────────────────────────────
    df = load_real_roi_data(csv_path)
    df_reno = df[df["roi_pct"].notna()].copy()
    avail_feats = [c for c in FEATURE_COLS if c in df_reno.columns]
    missing     = [c for c in FEATURE_COLS if c not in df_reno.columns]
    if missing:
        logger.warning(f"[train_roi] Features not in dataset (will be skipped): {missing}")

    X = df_reno[avail_feats].copy()
    y = df_reno["roi_pct"].copy()

    # Fill remaining NaNs
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Stratified split by city_tier to preserve tier distribution
    strat = df_reno["city_tier"] if "city_tier" in df_reno.columns else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=strat
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.12, random_state=RANDOM_SEED
    )

    logger.info(
        f"[train_roi] Split — train={len(X_tr):,}  val={len(X_val):,}  "
        f"test={len(X_te):,}  features={avail_feats}"
    )

    trained_models = []
    model_versions: Dict[str, Dict] = {}

    # ── XGBoost ────────────────────────────────────────────────────────────────
    try:
        import xgboost as xgb
        xg = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=40,
        )
        xg.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xg_pred_tr  = xg.predict(X_tr)
        xg_pred_te  = xg.predict(X_te)
        xg_mae_tr   = _mae(y_tr.values, xg_pred_tr)
        xg_mae_te   = _mae(y_te.values, xg_pred_te)
        xg_rmse_te  = _rmse(y_te.values, xg_pred_te)
        xg_r2_te    = float(r2_score(y_te, xg_pred_te))
        logger.info(
            f"  XGBoost — train_MAE={xg_mae_tr:.3f}% "
            f"test_MAE={xg_mae_te:.3f}% RMSE={xg_rmse_te:.3f}% R²={xg_r2_te:.3f}"
        )
        joblib.dump(xg, _WEIGHTS_DIR / "roi_xgb.joblib")
        trained_models.append(xg)
        model_versions["xgboost"] = {
            "train_mae": round(xg_mae_tr, 4), "test_mae": round(xg_mae_te, 4),
            "rmse": round(xg_rmse_te, 4), "r2": round(xg_r2_te, 4),
            "n_estimators": 800, "file": "roi_xgb.joblib",
        }
    except Exception as e:
        logger.warning(f"[train_roi] XGBoost failed: {e}")

    # ── RandomForest ───────────────────────────────────────────────────────────
    try:
        rf = RandomForestRegressor(
            n_estimators=600,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.7,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        rf.fit(X_tr, y_tr)
        rf_pred_tr = rf.predict(X_tr)
        rf_pred_te = rf.predict(X_te)
        rf_mae_tr  = _mae(y_tr.values, rf_pred_tr)
        rf_mae_te  = _mae(y_te.values, rf_pred_te)
        rf_rmse_te = _rmse(y_te.values, rf_pred_te)
        rf_r2_te   = float(r2_score(y_te, rf_pred_te))
        logger.info(
            f"  RandomForest — train_MAE={rf_mae_tr:.3f}% "
            f"test_MAE={rf_mae_te:.3f}% RMSE={rf_rmse_te:.3f}% R²={rf_r2_te:.3f}"
        )
        joblib.dump(rf, _WEIGHTS_DIR / "roi_rf.joblib")
        trained_models.append(rf)
        model_versions["random_forest"] = {
            "train_mae": round(rf_mae_tr, 4), "test_mae": round(rf_mae_te, 4),
            "rmse": round(rf_rmse_te, 4), "r2": round(rf_r2_te, 4),
            "n_estimators": 600, "file": "roi_rf.joblib",
        }
    except Exception as e:
        logger.warning(f"[train_roi] RandomForest failed: {e}")

    # ── GradientBoosting ───────────────────────────────────────────────────────
    try:
        gbm = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
        )
        gbm.fit(X_tr, y_tr)
        gbm_pred_tr = gbm.predict(X_tr)
        gbm_pred_te = gbm.predict(X_te)
        gbm_mae_tr  = _mae(y_tr.values, gbm_pred_tr)
        gbm_mae_te  = _mae(y_te.values, gbm_pred_te)
        gbm_rmse_te = _rmse(y_te.values, gbm_pred_te)
        gbm_r2_te   = float(r2_score(y_te, gbm_pred_te))
        logger.info(
            f"  GradientBoosting — train_MAE={gbm_mae_tr:.3f}% "
            f"test_MAE={gbm_mae_te:.3f}% RMSE={gbm_rmse_te:.3f}% R²={gbm_r2_te:.3f}"
        )
        joblib.dump(gbm, _WEIGHTS_DIR / "roi_gbm.joblib")
        trained_models.append(gbm)
        model_versions["gradient_boosting"] = {
            "train_mae": round(gbm_mae_tr, 4), "test_mae": round(gbm_mae_te, 4),
            "rmse": round(gbm_rmse_te, 4), "r2": round(gbm_r2_te, 4),
            "n_estimators": 500, "file": "roi_gbm.joblib",
        }
    except Exception as e:
        logger.warning(f"[train_roi] GradientBoosting failed: {e}")

    if not trained_models:
        raise RuntimeError("All ROI models failed to train. Check logs above.")

    # ── Ensemble evaluation ────────────────────────────────────────────────────
    ens_raw   = np.array([m.predict(X_te) for m in trained_models])
    ens_preds = np.mean(ens_raw, axis=0)
    ens_mae   = _mae(y_te.values, ens_preds)
    ens_rmse  = _rmse(y_te.values, ens_preds)
    ens_r2    = float(r2_score(y_te, ens_preds))
    logger.info(
        f"  Ensemble ({len(trained_models)} models) — "
        f"MAE={ens_mae:.3f}% RMSE={ens_rmse:.3f}% R²={ens_r2:.3f}"
    )

    # ── Walk-forward cross-validation (temporal) ───────────────────────────────
    logger.info("[train_roi] Running walk-forward CV on temporal folds …")
    cv_results = walk_forward_cross_validation(df_reno, trained_models, avail_feats, n_folds=5)

    # ── City coverage ──────────────────────────────────────────────────────────
    city_coverage: Dict[str, int] = {}
    if "city" in df_reno.columns:
        city_coverage = df_reno["city"].value_counts().to_dict()

    # ── Feature importance from RF ─────────────────────────────────────────────
    rf_model = next((m for m in trained_models
                     if type(m).__name__ == "RandomForestRegressor"), None)
    feature_importances: List[Dict] = []
    if rf_model is not None:
        try:
            imp = rf_model.feature_importances_
            idx = np.argsort(imp)[::-1]
            feature_importances = [
                {"feature": avail_feats[i], "importance": round(float(imp[i]), 4)}
                for i in idx
            ]
        except Exception:
            pass

    # ── model_report.json ──────────────────────────────────────────────────────
    report = {
        "training_date":        datetime.now(tz=timezone.utc).isoformat(),
        "dataset_size":         len(df_reno),
        "train_size":           len(X_tr),
        "val_size":             len(X_val),
        "test_size":            len(X_te),
        "data_source":          "india_property_transactions_real_kaggle_derived",
        "feature_cols":         avail_feats,
        "ensemble_mae":         round(ens_mae, 4),
        "ensemble_rmse":        round(ens_rmse, 4),
        "ensemble_r2":          round(ens_r2, 4),
        "cv_results":           cv_results,
        "model_versions":       model_versions,
        "city_coverage":        city_coverage,
        "feature_importances":  feature_importances,
        "n_models":             len(trained_models),
    }

    report_path = _WEIGHTS_DIR / "model_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"[train_roi] model_report.json saved → {report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Price model training (Prophet + XGBoost per material×city)
# ─────────────────────────────────────────────────────────────────────────────

def train_price_models_from_real_data(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Train Prophet + XGBoost price models from india_material_prices_historical.csv.

    For each material×city combination:
      1. Fits a Prophet model with Indian construction seasonality (yearly_seasonality=True).
      2. Fits an XGBoost regressor on lag/rolling/calendar features.
      3. Saves Prophet model to ml/weights/prophet_models/{material}_{city}.pkl
      4. Saves XGBoost model to ml/weights/price_xgb.joblib (single model, all combos)

    Returns:
        Summary dict with model counts, materials trained, MAPE per material.
    """
    try:
        import joblib
        from prophet import Prophet
    except ImportError:
        raise ImportError("prophet required: pip install prophet")

    try:
        import xgboost as xgb
        _HAS_XGB = True
    except ImportError:
        _HAS_XGB = False
        logger.warning("[train_price] XGBoost not installed — skipping XGB price model")

    target_csv = csv_path or _MATERIAL_CSV
    if not target_csv.exists():
        raise DataError(
            f"india_material_prices_historical.csv not found at {target_csv}.\n"
            "Run: python data/datasets/material_prices/build_extended_material_prices.py"
        )

    logger.info("=" * 60)
    logger.info(f"[train_price] Loading material price data from {target_csv}")

    df = pd.read_csv(str(target_csv), parse_dates=["date"])
    logger.info(
        f"[train_price] {len(df):,} rows | "
        f"{df['material_key'].nunique()} materials | "
        f"{df['city'].nunique()} cities"
    )

    materials = sorted(df["material_key"].unique())
    cities    = sorted(df["city"].unique())

    prophet_count = 0
    skipped_count = 0
    mape_per_material: Dict[str, List[float]] = {}

    # XGBoost training data — aggregate all combos
    xgb_records: List[Dict] = []

    for material in materials:
        for city in cities:
            subset = df[
                (df["material_key"] == material) & (df["city"] == city)
            ].sort_values("date").copy()

            if len(subset) < 12:   # need at least 1 year of monthly data
                skipped_count += 1
                logger.debug(
                    f"[train_price] Skipping {material}×{city}: only {len(subset)} rows"
                )
                continue

            # ── Prophet ───────────────────────────────────────────────────────
            try:
                prophet_df = subset[["date", "price_inr"]].rename(
                    columns={"date": "ds", "price_inr": "y"}
                )

                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="additive",
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                )

                # Indian construction seasonality: stronger Oct-Feb, weaker Jun-Aug
                m.add_seasonality(
                    name="indian_construction",
                    period=365.25,
                    fourier_order=5,
                )

                m.fit(prophet_df)

                # Evaluate: hold out last 6 months
                if len(prophet_df) >= 18:
                    train_p = prophet_df.iloc[:-6]
                    test_p  = prophet_df.iloc[-6:]
                    m_eval  = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05,
                    )
                    m_eval.fit(train_p)
                    future    = m_eval.make_future_dataframe(periods=6, freq="MS")
                    forecast  = m_eval.predict(future)
                    pred_vals = forecast["yhat"].iloc[-6:].values
                    true_vals = test_p["y"].values
                    mape = float(np.mean(np.abs((true_vals - pred_vals) / np.maximum(true_vals, 1.0))) * 100)
                    mape_per_material.setdefault(material, []).append(mape)
                    logger.debug(f"  {material}×{city}: Prophet MAPE={mape:.1f}%")

                # Save Prophet model
                safe_name = f"{material}__{city.replace(' ', '_')}.pkl"
                joblib.dump(m, _PROPHET_DIR / safe_name)
                prophet_count += 1

            except Exception as e:
                logger.warning(f"[train_price] Prophet failed for {material}×{city}: {e}")
                skipped_count += 1

            # ── Build XGBoost features ─────────────────────────────────────────
            if _HAS_XGB:
                try:
                    s = subset.copy().set_index("date").sort_index()
                    prices = s["price_inr"]
                    for lag in [1, 2, 3, 6, 12]:
                        if len(prices) > lag:
                            s[f"lag_{lag}"] = prices.shift(lag)
                    for window in [3, 6, 12]:
                        if len(prices) > window:
                            s[f"roll_mean_{window}"] = prices.rolling(window).mean()
                            s[f"roll_std_{window}"]  = prices.rolling(window).std()
                    s["month"]     = s.index.month
                    s["quarter"]   = s.index.quarter
                    s["year"]      = s.index.year
                    s["material"]  = material
                    s["city"]      = city
                    xgb_records.append(s.dropna().reset_index())
                except Exception as e:
                    logger.debug(f"[train_price] XGB feature build failed for {material}×{city}: {e}")

    # ── XGBoost global price regressor ────────────────────────────────────────
    xgb_report: Dict = {}
    if _HAS_XGB and xgb_records:
        try:
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import mean_absolute_error

            all_xgb = pd.concat(xgb_records, ignore_index=True)
            le_mat  = LabelEncoder().fit(all_xgb["material"])
            le_city = LabelEncoder().fit(all_xgb["city"])
            all_xgb["mat_enc"]  = le_mat.transform(all_xgb["material"])
            all_xgb["city_enc"] = le_city.transform(all_xgb["city"])

            xgb_feature_cols = [
                c for c in all_xgb.columns
                if c.startswith("lag_") or c.startswith("roll_")
                or c in ["month", "quarter", "year", "mat_enc", "city_enc"]
            ]
            Xx = all_xgb[xgb_feature_cols].fillna(0)
            yx = all_xgb["price_inr"]

            split = int(len(Xx) * 0.85)
            X_xtr, X_xte = Xx.iloc[:split], Xx.iloc[split:]
            y_xtr, y_xte = yx.iloc[:split], yx.iloc[split:]

            xg_price = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                verbosity=0,
            )
            xg_price.fit(X_xtr, y_xtr)
            xg_pred_price = xg_price.predict(X_xte)
            xg_mae_price  = _mae(y_xte.values, xg_pred_price)
            xg_mape_price = float(
                np.mean(np.abs((y_xte.values - xg_pred_price) / np.maximum(y_xte.values, 1.0))) * 100
            )
            logger.info(
                f"[train_price] XGB price model — "
                f"MAE=₹{xg_mae_price:.2f} MAPE={xg_mape_price:.1f}%"
            )

            # Save encoder-augmented model bundle
            bundle = {
                "model": xg_price,
                "le_material": le_mat,
                "le_city": le_city,
                "feature_cols": xgb_feature_cols,
            }
            joblib.dump(bundle, _WEIGHTS_DIR / "price_xgb.joblib")
            xgb_report = {
                "mae_inr": round(xg_mae_price, 2),
                "mape_pct": round(xg_mape_price, 2),
                "train_rows": int(len(X_xtr)),
                "test_rows": int(len(X_xte)),
            }
        except Exception as e:
            logger.warning(f"[train_price] XGB global price model failed: {e}")

    # Per-material MAPE summary
    mat_mape_summary = {
        mat: round(float(np.mean(mapes)), 2)
        for mat, mapes in mape_per_material.items()
        if mapes
    }

    summary = {
        "prophet_models_trained": prophet_count,
        "models_skipped":         skipped_count,
        "materials":              materials,
        "cities":                 cities,
        "total_rows":             len(df),
        "prophet_dir":            str(_PROPHET_DIR),
        "mape_per_material":      mat_mape_summary,
        "xgboost_price":          xgb_report,
        "training_date":          datetime.now(tz=timezone.utc).isoformat(),
    }

    price_report_path = _WEIGHTS_DIR / "price_model_report.json"
    with open(price_report_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(
        f"[train_price] Done. Prophet models: {prophet_count}  Skipped: {skipped_count}\n"
        f"             Report → {price_report_path}"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Summary table printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_table(roi_report: Optional[Dict], price_report: Optional[Dict]) -> None:
    """Print human-readable summary of training results."""
    print("\n" + "=" * 72)
    print("ARKEN ML Training Summary")
    print("=" * 72)

    if roi_report:
        print(f"\nROI Ensemble — {roi_report.get('n_models', '?')} models")
        print(f"  Dataset size : {roi_report.get('dataset_size', '?'):,} rows (real Kaggle data)")
        print(f"  Features     : {roi_report.get('feature_cols', [])}")
        print()
        print(f"  {'Model':<25} {'Train MAE':>12} {'Test MAE':>12} {'RMSE':>10} {'R²':>8}")
        print("  " + "-" * 68)
        for name, metrics in roi_report.get("model_versions", {}).items():
            print(
                f"  {name:<25} "
                f"{metrics.get('train_mae', 0):.3f}%{' ':>6} "
                f"{metrics.get('test_mae', 0):.3f}%{' ':>6} "
                f"{metrics.get('rmse', 0):.3f}%{' ':>4} "
                f"{metrics.get('r2', 0):.3f}"
            )
        print(
            f"\n  Ensemble:  "
            f"Test MAE={roi_report.get('ensemble_mae', 0):.3f}%  "
            f"RMSE={roi_report.get('ensemble_rmse', 0):.3f}%  "
            f"R²={roi_report.get('ensemble_r2', 0):.3f}"
        )
        cv = roi_report.get("cv_results", {})
        if cv:
            print(
                f"\n  Walk-forward CV ({cv.get('n_folds', '?')} temporal folds):  "
                f"MAE={cv.get('overall_mae', 0):.3f}%  "
                f"RMSE={cv.get('overall_rmse', 0):.3f}%  "
                f"R²={cv.get('overall_r2', 0):.3f}"
            )
        top_feats = roi_report.get("feature_importances", [])[:5]
        if top_feats:
            print(
                f"\n  Top features: " +
                ", ".join(f"{f['feature']}({f['importance']:.3f})" for f in top_feats)
            )

    if price_report:
        print(f"\nPrice Forecast Models")
        print(f"  Prophet models trained : {price_report.get('prophet_models_trained', 0)}")
        print(f"  Materials              : {len(price_report.get('materials', []))}")
        print(f"  Cities                 : {len(price_report.get('cities', []))}")
        mape = price_report.get("mape_per_material", {})
        if mape:
            avg_mape = float(np.mean(list(mape.values())))
            print(f"  Prophet avg MAPE       : {avg_mape:.1f}%")
        xgb_price = price_report.get("xgboost_price", {})
        if xgb_price:
            print(
                f"  XGB price MAPE         : {xgb_price.get('mape_pct', 0):.1f}%  "
                f"MAE=₹{xgb_price.get('mae_inr', 0):.0f}"
            )

    print("\n" + "=" * 72)
    print("Weights saved to:", _WEIGHTS_DIR)
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARKEN ML Training Pipeline v2.0 — real data only, no synthetic rows."
    )
    parser.add_argument(
        "--model",
        choices=["roi", "price", "all"],
        default="all",
        help="Which model(s) to train (default: all).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional override path for india_property_transactions.csv.",
    )
    parser.add_argument(
        "--price-csv",
        type=str,
        default=None,
        help="Optional override path for india_material_prices_historical.csv.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Directory to save model weights (default: ml/weights/).",
    )
    args = parser.parse_args()

    global _WEIGHTS_DIR, _PROPHET_DIR
    if args.weights_dir:
        _WEIGHTS_DIR = Path(args.weights_dir)
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        _PROPHET_DIR = _WEIGHTS_DIR / "prophet_models"
        _PROPHET_DIR.mkdir(parents=True, exist_ok=True)

    csv_path   = Path(args.csv_path)   if args.csv_path   else None
    price_csv  = Path(args.price_csv)  if args.price_csv  else None

    roi_report   = None
    price_report = None

    try:
        if args.model in ("roi", "all"):
            roi_report = train_roi_model(csv_path)

        if args.model in ("price", "all"):
            price_report = train_price_models_from_real_data(price_csv)

        _print_summary_table(roi_report, price_report)

    except DataError as e:
        logger.error(f"\n[DataError] {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()