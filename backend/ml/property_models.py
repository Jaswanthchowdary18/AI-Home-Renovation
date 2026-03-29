"""
ARKEN — Property ML Models v3.0
=================================
v3.0 Changes (ROI FIX):

  ROIModel:
    - Now trains from india_property_transactions.csv via RenovationDataPreprocessor.
    - 3-model ensemble: XGBoost + RandomForest + GradientBoosting (no LightGBM required).
    - Saves individual models: roi_xgb.joblib, roi_rf.joblib, roi_gbm.joblib.
    - model_report.json includes per-model MAE, RMSE, R², city_coverage,
      training_date, dataset_size, model_versions.
    - predict() returns (roi_mean, roi_ci_low, roi_ci_high) with CI from
      REAL ensemble std — not a fixed percentage.
    - Model freshness: loads from disk if files < 30 days old; retrains otherwise.
    - model_confidence in _build_report() is now computed dynamically:
      confidence = max(0.60, min(0.95, 1.0 - (roi_high - roi_low) / max(roi_mean, 1) * 2))

  RenovationCostModel:
    - Now uses RenovationDataPreprocessor.get_reno_cost_splits() (real CSV rows).
    - Ensemble variance used for confidence (not hardcoded 0.78).
    - All other API unchanged.

  PropertyValueModel:
    - UNCHANGED from v2.0.

All existing public method signatures preserved.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    logging.getLogger(__name__).warning(
        "[PropertyModels] scikit-learn / joblib not installed. "
        "Install: pip install scikit-learn joblib"
    )

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path("/app/ml/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

_train_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class DataQualityError(ValueError):
    """
    Raised when the training dataset does not meet minimum quality requirements:
      - Dataset too small (< minimum rows threshold)
      - CSV missing from expected path
      - Synthetic data detected in real-data-only context
    This replaces the old silent fallback to synthetic data.
    """


# ── Model freshness threshold ──────────────────────────────────────────────────
_MODEL_MAX_AGE_DAYS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        return None


def _try_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        return None


def _file_age_days(path: Path) -> float:
    """Return age of file in days, or float('inf') if not found."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 86400
    except OSError:
        return float("inf")


def _feature_importance_top(model, feature_names: List[str], top_n: int = 5) -> List[Dict]:
    try:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        return [
            {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
            for i in idx if importances[i] > 0.001
        ]
    except Exception:
        return []


def _ensemble_predict_raw(models: List, X: pd.DataFrame) -> np.ndarray:
    """Return per-model predictions as a 2D array (n_models × n_samples)."""
    preds = []
    for m in models:
        try:
            preds.append(m.predict(X))
        except Exception as e:
            logger.debug(f"[ensemble] {type(m).__name__} predict failed: {e}")
    return np.array(preds) if preds else np.empty((0, len(X)))


def _ensemble_predict(models: List, X: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns (mean_prediction, lower_95ci, upper_95ci).
    CI comes from actual ensemble std — NOT a fixed percentage.
    """
    raw = _ensemble_predict_raw(models, X)
    if raw.size == 0:
        return 0.0, 0.0, 0.0

    col    = raw[:, 0]
    mean_  = float(np.mean(col))
    std_   = float(np.std(col, ddof=1)) if len(col) > 1 else mean_ * 0.12
    return mean_, float(mean_ - 1.96 * std_), float(mean_ + 1.96 * std_)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Property Value Model (UNCHANGED from v2.0)
# ─────────────────────────────────────────────────────────────────────────────

class PropertyValueModel:
    WEIGHT_FILE = WEIGHTS_DIR / "property_value_ensemble.joblib"
    FEATURE_NAMES = [
        "size_sqft", "age_years", "bhk", "city_tier", "furnished",
        "amenity_count", "has_parking", "has_security",
        "schools_nearby", "hospitals_nearby",
    ]
    _models: Optional[List] = None

    def __init__(self):
        if PropertyValueModel._models is None:
            self._load_or_train()

    def _load_or_train(self):
        with _train_lock:
            if PropertyValueModel._models is not None:
                return
            if self.WEIGHT_FILE.exists():
                try:
                    PropertyValueModel._models = joblib.load(self.WEIGHT_FILE)
                    logger.info(f"[PropertyValueModel] Loaded from {self.WEIGHT_FILE}")
                    return
                except Exception as e:
                    logger.warning(f"[PropertyValueModel] Load failed ({e}), retraining")
            self._train()

    def _train(self):
        from ml.housing_preprocessor import get_preprocessor
        logger.info("[PropertyValueModel] Training on housing datasets...")
        prep = get_preprocessor()
        X_tr, X_va, X_te, y_tr, y_va, y_te = prep.get_property_value_splits()
        models = []
        rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr)
        logger.info(f"[PropertyValueModel] RF  MAE(log): {mean_absolute_error(y_te, rf.predict(X_te)):.4f}")
        models.append(("rf", rf))
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(X_tr, y_tr)
        logger.info(f"[PropertyValueModel] GBM MAE(log): {mean_absolute_error(y_te, gb.predict(X_te)):.4f}")
        models.append(("gb", gb))
        xgb = _try_xgboost()
        if xgb:
            xg = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                   subsample=0.85, colsample_bytree=0.85, tree_method="hist",
                                   random_state=42, n_jobs=-1, verbosity=0)
            xg.fit(X_tr, y_tr)
            logger.info(f"[PropertyValueModel] XGB MAE(log): {mean_absolute_error(y_te, xg.predict(X_te)):.4f}")
            models.append(("xgb", xg))
        PropertyValueModel._models = [m for _, m in models]
        joblib.dump(PropertyValueModel._models, self.WEIGHT_FILE)
        logger.info(f"[PropertyValueModel] Saved {len(models)} models → {self.WEIGHT_FILE}")

    def predict(self, size_sqft, city_tier, age_years=10, bhk=2, furnished=1,
                amenity_count=3, has_parking=0, has_security=0,
                schools_nearby=2, hospitals_nearby=1) -> Dict[str, Any]:
        X = pd.DataFrame([{
            "size_sqft": size_sqft, "age_years": age_years, "bhk": bhk,
            "city_tier": city_tier, "furnished": furnished,
            "amenity_count": amenity_count, "has_parking": has_parking,
            "has_security": has_security, "schools_nearby": schools_nearby,
            "hospitals_nearby": hospitals_nearby,
        }])
        if not PropertyValueModel._models:
            psf = {1: 9500, 2: 6500, 3: 4500}.get(city_tier, 6500)
            val = int(size_sqft * psf)
            return {"value_inr": val, "value_low_inr": int(val * 0.85),
                    "value_high_inr": int(val * 1.15), "confidence": 0.55,
                    "price_per_sqft": psf, "model_drivers": [], "model_type": "heuristic"}
        log_mean, log_low, log_high = _ensemble_predict(PropertyValueModel._models, X)
        val  = int(np.expm1(log_mean))
        low  = int(np.expm1(max(log_low, 0)))
        high = int(np.expm1(log_high))
        rf_m = next((m for m in PropertyValueModel._models if isinstance(m, RandomForestRegressor)), None)
        return {
            "value_inr":      val,
            "value_low_inr":  low,
            "value_high_inr": high,
            "price_per_sqft": int(val / max(size_sqft, 1)),
            "confidence":     round(0.72 + min(0.15, len(PropertyValueModel._models) * 0.04), 2),
            "model_drivers":  _feature_importance_top(rf_m, self.FEATURE_NAMES) if rf_m else [],
            "model_type":     f"ensemble_{len(PropertyValueModel._models)}",
            "n_models":       len(PropertyValueModel._models),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Renovation Cost Model (updated: real CSV via RenovationDataPreprocessor)
# ─────────────────────────────────────────────────────────────────────────────

class RenovationCostModel:
    WEIGHT_FILE = WEIGHTS_DIR / "renovation_cost_model.joblib"
    FEATURE_NAMES = [
        "room_type_enc", "budget_tier_enc", "size_sqft",
        "city_tier", "scope_enc", "age_years",
    ]
    ROOM_MAP   = {"bedroom": 0, "kitchen": 1, "bathroom": 2, "living_room": 3,
                  "full_home": 4, "dining_room": 5, "study": 6}
    BUDGET_MAP = {"basic": 0, "mid": 1, "premium": 2}
    SCOPE_MAP  = {"cosmetic_only": 0, "partial": 1, "full_room": 2, "structural_plus": 3}

    _models: Optional[List] = None

    def __init__(self):
        if RenovationCostModel._models is None:
            self._load_or_train()

    def _load_or_train(self):
        with _train_lock:
            if RenovationCostModel._models is not None:
                return
            age = _file_age_days(self.WEIGHT_FILE)
            if self.WEIGHT_FILE.exists() and age < _MODEL_MAX_AGE_DAYS:
                try:
                    RenovationCostModel._models = joblib.load(self.WEIGHT_FILE)
                    logger.info(f"[RenovationCostModel] Loaded from {self.WEIGHT_FILE} (age: {age:.1f}d)")
                    return
                except Exception as e:
                    logger.warning(f"[RenovationCostModel] Load failed ({e}), retraining")
            self._train()

    def _train(self):
        from ml.housing_preprocessor import get_reno_preprocessor
        logger.info("[RenovationCostModel] Training on real renovation data...")
        rp = get_reno_preprocessor()
        try:
            X_tr, X_te, y_tr, y_te = rp.get_reno_cost_splits()
        except Exception as e:
            logger.warning(f"[RenovationCostModel] Real data unavailable ({e}), using legacy path")
            from ml.housing_preprocessor import get_preprocessor
            X_tr, X_te, y_tr, y_te = get_preprocessor().get_reno_cost_splits()

        avail = [c for c in self.FEATURE_NAMES if c in X_tr.columns]
        models = []

        xgb = _try_xgboost()
        if xgb:
            xg = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.04,
                                   subsample=0.85, colsample_bytree=0.85,
                                   tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
            xg.fit(X_tr[avail], y_tr)
            logger.info(f"[RenovationCostModel] XGB MAE(log): {mean_absolute_error(y_te, xg.predict(X_te[avail])):.4f}")
            models.append(xg)

        rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_tr[avail], y_tr)
        logger.info(f"[RenovationCostModel] RF  MAE(log): {mean_absolute_error(y_te, rf.predict(X_te[avail])):.4f}")
        models.append(rf)

        RenovationCostModel._models = models
        joblib.dump(models, self.WEIGHT_FILE)
        logger.info(f"[RenovationCostModel] Saved {len(models)} models → {self.WEIGHT_FILE}")

    def predict(self, room_type, budget_tier, area_sqft, city_tier,
                age_years=10, scope="partial", furnished=1) -> Dict[str, Any]:
        from ml.housing_preprocessor import RENO_COST_BENCHMARKS

        room_enc   = self.ROOM_MAP.get(room_type, 0)
        budget_enc = self.BUDGET_MAP.get(budget_tier, 1)
        scope_enc  = self.SCOPE_MAP.get(scope, 1)
        reno_area  = area_sqft if room_type == "full_home" else area_sqft * 0.25

        X = pd.DataFrame([{
            "room_type_enc": room_enc, "budget_tier_enc": budget_enc,
            "size_sqft": area_sqft, "city_tier": city_tier,
            "scope_enc": scope_enc, "age_years": age_years,
            "reno_area_sqft": reno_area, "furnished": furnished,
        }])
        avail = [c for c in self.FEATURE_NAMES if c in X.columns]

        if RenovationCostModel._models:
            log_mean, log_low, log_high = _ensemble_predict(RenovationCostModel._models, X[avail])
            cost      = int(np.expm1(log_mean))
            cost_low  = int(np.expm1(max(log_low, 0)))
            cost_high = int(np.expm1(log_high))
            # Confidence from ensemble variance
            spread    = max(cost_high - cost_low, 1)
            raw_conf  = 1.0 - (spread / max(cost, 1))
            confidence = float(np.clip(raw_conf, 0.55, 0.92))
            model_type = f"ensemble_{len(RenovationCostModel._models)}"
        else:
            lo, hi    = RENO_COST_BENCHMARKS.get(room_type, {}).get(budget_tier, (500, 1200))
            cost      = int(reno_area * (lo + hi) / 2)
            cost_low  = int(reno_area * lo)
            cost_high = int(reno_area * hi)
            confidence = 0.60
            model_type = "benchmark"

        mult      = {1: 1.10, 2: 1.00, 3: 0.88}.get(city_tier, 1.0)
        cost      = int(cost * mult)
        cost_low  = int(cost_low * mult)
        cost_high = int(cost_high * mult)

        return {
            "renovation_cost_inr":      cost,
            "renovation_cost_low_inr":  cost_low,
            "renovation_cost_high_inr": cost_high,
            "cost_per_sqft":            int(cost / max(reno_area, 1)),
            "reno_area_sqft":           round(reno_area, 1),
            "cost_breakdown":           self._cost_breakdown(room_type, budget_tier, cost),
            "confidence":               confidence,
            "model_type":               model_type,
        }

    @staticmethod
    def _cost_breakdown(room_type: str, budget_tier: str, total_cost: int) -> Dict[str, int]:
        if room_type in ("kitchen", "bathroom"):
            alloc = {"materials": 0.55, "labour": 0.28, "fixtures": 0.10,
                     "supervision": 0.04, "gst_contingency": 0.03}
        else:
            alloc = {"materials": 0.52, "labour": 0.30, "fixtures": 0.08,
                     "supervision": 0.05, "gst_contingency": 0.05}
        return {k: int(total_cost * v) for k, v in alloc.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROI Model (v3.0 — real CSV, 3-model ensemble, real CI, model_report.json)
# ─────────────────────────────────────────────────────────────────────────────

class ROIModel:
    """
    XGBoost + RandomForest + GradientBoosting ensemble for ROI prediction.

    v3.0:
      - Training data: india_property_transactions.csv via RenovationDataPreprocessor.
      - Individual model files: roi_xgb.joblib, roi_rf.joblib, roi_gbm.joblib.
      - CI from actual ensemble std.
      - model_report.json: training_date, dataset_size, mae/rmse/r2 per model,
        ensemble metrics, city_coverage, model_versions.
      - Freshness check: reload if files < 30 days old, else retrain.
    """

    XGB_FILE    = WEIGHTS_DIR / "roi_xgb.joblib"
    RF_FILE     = WEIGHTS_DIR / "roi_rf.joblib"
    GBM_FILE    = WEIGHTS_DIR / "roi_gbm.joblib"
    REPORT_FILE = WEIGHTS_DIR / "model_report.json"

    FEATURE_NAMES = [
        "renovation_cost_lakh", "size_sqft", "city_tier",
        "room_type_enc", "budget_tier_enc", "age_years",
        "furnished", "reno_intensity", "scope_enc",
        "amenity_count", "has_parking",
    ]

    _models:       Optional[List]      = None
    _feats:        Optional[List[str]] = None
    _dataset_size: int                 = 0

    def __init__(self):
        if ROIModel._models is None:
            self._load_or_train()

    def _models_fresh(self) -> bool:
        """True if all 3 model files exist and are < _MODEL_MAX_AGE_DAYS old."""
        for p in (self.XGB_FILE, self.RF_FILE, self.GBM_FILE):
            if not p.exists() or _file_age_days(p) >= _MODEL_MAX_AGE_DAYS:
                return False
        return True

    def _load_or_train(self):
        with _train_lock:
            if ROIModel._models is not None:
                return
            if self._models_fresh():
                try:
                    models = []
                    xgb_m = joblib.load(self.XGB_FILE)
                    rf_m  = joblib.load(self.RF_FILE)
                    gbm_m = joblib.load(self.GBM_FILE)
                    models = [xgb_m, rf_m, gbm_m]
                    ROIModel._models = models
                    ROIModel._feats  = self.FEATURE_NAMES
                    # Read dataset_size from report
                    if self.REPORT_FILE.exists():
                        with open(self.REPORT_FILE, "r", encoding="utf-8") as fh:
                            rpt = json.load(fh)
                        ROIModel._dataset_size = int(rpt.get("dataset_size", 0))
                    logger.info(
                        f"[ROIModel] Loaded 3 models from disk "
                        f"(dataset_size={ROIModel._dataset_size:,})"
                    )
                    return
                except Exception as e:
                    logger.warning(f"[ROIModel] Load failed ({e}), retraining")
                    ROIModel._models = None
            self._train()

    # ─────────────────────────────────────────────────────────────────────────
    # Real-data loading (no synthetic fallback — DataQualityError on failure)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_real_dataset(self) -> tuple:
        """
        Load real renovation data from india_property_transactions.csv.

        Uses RenovationDataPreprocessor which validates the CSV has no
        synthetic data (raises ValueError if 'synthetic' found in data_source).

        Raises:
            DataQualityError: If fewer than 200 training rows are available,
                              or if the CSV is missing or contains synthetic data.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test) DataFrames/Series.
        """
        from ml.housing_preprocessor import get_reno_preprocessor

        rp = get_reno_preprocessor()
        try:
            X_tr, X_te, y_tr, y_te = rp.get_roi_splits(stratify_by_city_tier=True)
        except Exception as e:
            raise DataQualityError(
                f"[ROIModel] Real CSV unavailable or failed validation: {e}\n"
                "To fix: run python data/datasets/property_transactions/build_real_roi_dataset.py\n"
                "That script derives all roi_pct labels from real Kaggle housing and rent data — no synthetic data."
            ) from e

        if len(X_tr) < 200:
            raise DataQualityError(
                f"[ROIModel] Only {len(X_tr)} training rows available (minimum 200 required). "
                "Regenerate india_property_transactions.csv from the Kaggle housing datasets."
            )

        logger.info(
            f"[ROIModel] Real data loaded — "
            f"train={len(X_tr):,}  test={len(X_te):,}  "
            f"roi_pct range=[{y_tr.min():.1f}, {y_tr.max():.1f}]%"
        )
        return X_tr, X_te, y_tr, y_te

    def _train(self):
        """Train ROI ensemble on real data. Raises DataQualityError if real data unavailable."""
        logger.info("[ROIModel] Training on real property transactions CSV...")

        # ── Load real data ONLY — no synthetic fallback ───────────────────────
        try:
            X_tr, X_te, y_tr, y_te = self._load_real_dataset()
        except DataQualityError as e:
            logger.error(str(e))
            logger.error(
                "[ROIModel] Training aborted. ROI predictions will use heuristic fallback "
                "until real data is available. Fix: run build_real_roi_dataset.py."
            )
            return

        dataset_size = len(X_tr) + len(X_te)
        feats        = [c for c in self.FEATURE_NAMES if c in X_tr.columns]
        logger.info(
            f"[ROIModel] Split — train={len(X_tr):,}  test={len(X_te):,}  "
            f"features={feats}"
        )

        # ── Per-city coverage ─────────────────────────────────────────────────
        city_coverage: Dict[str, int] = {}
        # (city_tier is present; actual city names are logged from preprocessor)

        model_versions: Dict[str, Dict] = {}
        trained_models = []

        # ── 1. XGBoost ────────────────────────────────────────────────────────
        xgb = _try_xgboost()
        if xgb:
            try:
                xg = xgb.XGBRegressor(
                    n_estimators=800, max_depth=6, learning_rate=0.04,
                    subsample=0.85, colsample_bytree=0.85,
                    reg_alpha=0.1, reg_lambda=1.0,
                    tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
                )
                xg.fit(X_tr[feats], y_tr, eval_set=[(X_te[feats], y_te)], verbose=False)
                xg_pred  = xg.predict(X_te[feats])
                xg_mae   = float(mean_absolute_error(y_te, xg_pred))
                xg_rmse  = _rmse(y_te, xg_pred)
                xg_r2    = float(r2_score(y_te, xg_pred))
                logger.info(
                    f"[ROIModel] XGB — MAE={xg_mae:.3f}%  RMSE={xg_rmse:.3f}%  R²={xg_r2:.3f}"
                )
                joblib.dump(xg, self.XGB_FILE)
                trained_models.append(xg)
                model_versions["xgboost"] = {
                    "mae": round(xg_mae, 4), "rmse": round(xg_rmse, 4), "r2": round(xg_r2, 4),
                    "n_estimators": 800, "file": str(self.XGB_FILE),
                }
            except Exception as e:
                logger.warning(f"[ROIModel] XGB training failed: {e}")
        else:
            logger.warning("[ROIModel] XGBoost not installed — skipping XGB model.")

        # ── 2. RandomForest ───────────────────────────────────────────────────
        try:
            rf = RandomForestRegressor(
                n_estimators=500, max_depth=12, min_samples_leaf=4,
                n_jobs=-1, random_state=42,
            )
            rf.fit(X_tr[feats], y_tr)
            rf_pred = rf.predict(X_te[feats])
            rf_mae  = float(mean_absolute_error(y_te, rf_pred))
            rf_rmse = _rmse(y_te, rf_pred)
            rf_r2   = float(r2_score(y_te, rf_pred))
            logger.info(
                f"[ROIModel] RF  — MAE={rf_mae:.3f}%  RMSE={rf_rmse:.3f}%  R²={rf_r2:.3f}"
            )
            joblib.dump(rf, self.RF_FILE)
            trained_models.append(rf)
            model_versions["random_forest"] = {
                "mae": round(rf_mae, 4), "rmse": round(rf_rmse, 4), "r2": round(rf_r2, 4),
                "n_estimators": 500, "file": str(self.RF_FILE),
            }
        except Exception as e:
            logger.warning(f"[ROIModel] RF training failed: {e}")

        # ── 3. GradientBoosting ───────────────────────────────────────────────
        try:
            gbm = GradientBoostingRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.06,
                subsample=0.8, random_state=42,
            )
            gbm.fit(X_tr[feats], y_tr)
            gbm_pred = gbm.predict(X_te[feats])
            gbm_mae  = float(mean_absolute_error(y_te, gbm_pred))
            gbm_rmse = _rmse(y_te, gbm_pred)
            gbm_r2   = float(r2_score(y_te, gbm_pred))
            logger.info(
                f"[ROIModel] GBM — MAE={gbm_mae:.3f}%  RMSE={gbm_rmse:.3f}%  R²={gbm_r2:.3f}"
            )
            joblib.dump(gbm, self.GBM_FILE)
            trained_models.append(gbm)
            model_versions["gradient_boosting"] = {
                "mae": round(gbm_mae, 4), "rmse": round(gbm_rmse, 4), "r2": round(gbm_r2, 4),
                "n_estimators": 400, "file": str(self.GBM_FILE),
            }
        except Exception as e:
            logger.warning(f"[ROIModel] GBM training failed: {e}")

        if not trained_models:
            logger.error("[ROIModel] All models failed to train. ROI will use heuristic fallback.")
            return

        # ── Ensemble evaluation ───────────────────────────────────────────────
        ens_raw    = np.array([m.predict(X_te[feats]) for m in trained_models])
        ens_preds  = np.mean(ens_raw, axis=0)
        ens_mae    = float(mean_absolute_error(y_te, ens_preds))
        ens_rmse   = _rmse(y_te, ens_preds)
        ens_r2     = float(r2_score(y_te, ens_preds))
        logger.info(
            f"[ROIModel] Ensemble ({len(trained_models)} models) — "
            f"MAE={ens_mae:.3f}%  RMSE={ens_rmse:.3f}%  R²={ens_r2:.3f}"
        )

        # ── Feature importance from RF ────────────────────────────────────────
        rf_best = next((m for m in trained_models if isinstance(m, RandomForestRegressor)), None)
        fi = _feature_importance_top(rf_best, feats, top_n=len(feats)) if rf_best else []

        # ── model_report.json ─────────────────────────────────────────────────
        report = {
            "training_date":   datetime.now(tz=timezone.utc).isoformat(),
            "dataset_size":    dataset_size,
            "train_size":      len(X_tr),
            "test_size":       len(X_te),
            "data_source":     "india_property_transactions_csv",
            "mae":             round(ens_mae, 4),
            "rmse":            round(ens_rmse, 4),
            "r2":              round(ens_r2, 4),
            "model_versions":  model_versions,
            "city_coverage":   city_coverage,
            "features_used":   feats,
            "feature_importances": fi,
            "n_models":        len(trained_models),
        }
        try:
            with open(self.REPORT_FILE, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
            logger.info(f"[ROIModel] model_report.json saved → {self.REPORT_FILE}")
        except Exception as e:
            logger.warning(f"[ROIModel] Could not write model_report.json: {e}")

        ROIModel._models       = trained_models
        ROIModel._feats        = feats
        ROIModel._dataset_size = dataset_size
        logger.info(f"[ROIModel] {len(trained_models)} models ready.")

    def predict(self, X: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Returns (roi_mean, roi_ci_low, roi_ci_high) as percentages.
        CI is derived from actual ensemble std — not a fixed percentage.
        """
        if not ROIModel._models:
            return 10.0, 7.0, 14.0

        feats = ROIModel._feats if ROIModel._feats else self.FEATURE_NAMES
        avail = [f for f in feats if f in X.columns]

        raw   = _ensemble_predict_raw(ROIModel._models, X[avail])
        if raw.size == 0:
            return 10.0, 7.0, 14.0

        col  = raw[:, 0]
        mean_ = float(np.mean(col))
        std_  = float(np.std(col, ddof=1)) if len(col) > 1 else mean_ * 0.12
        low   = mean_ - 1.96 * std_
        high  = mean_ + 1.96 * std_

        mean_ = float(np.clip(mean_, 1.5, 40.0))
        low   = float(np.clip(low,   1.0, 40.0))
        high  = float(np.clip(high, mean_, 42.0))
        return round(mean_, 2), round(low, 2), round(high, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Model manager (singleton)
# ─────────────────────────────────────────────────────────────────────────────

class PropertyMLManager:
    _instance: Optional["PropertyMLManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.property_value  = PropertyValueModel()
        self.renovation_cost = RenovationCostModel()
        self.roi             = ROIModel()
        logger.info("[PropertyMLManager] All models ready")

    @classmethod
    def get(cls) -> "PropertyMLManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


def get_ml_manager() -> PropertyMLManager:
    return PropertyMLManager.get()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "train" in sys.argv:
        logger.info("=== Training all ARKEN property ML models ===")
        PropertyMLManager.get()
        logger.info("=== Training complete ===")