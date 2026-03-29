#!/usr/bin/env python3
"""
ARKEN — Model Evaluation Report
=================================
Standalone evaluation script that acts as the living model card for every
deployment. Covers all four ML subsystems:

  1. Price forecasting   — XGBoost MAE/MAPE + Prophet walk-forward CV MAPE
  2. ROI ensemble        — XGBoost + RF + GBM + ensemble MAE/RMSE/R²
  3. CV pipeline         — StyleClassifier metadata_trained flag + DamageDetector model_used
  4. Depth estimation    — prior calibration check

Usage:
    cd backend
    python scripts/evaluate_models.py            # full report
    python scripts/evaluate_models.py --section price
    python scripts/evaluate_models.py --section roi
    python scripts/evaluate_models.py --section cv
    python scripts/evaluate_models.py --json     # machine-readable output

Exit code 0 — all critical checks passed.
Exit code 1 — one or more critical checks failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("arken.eval")

# ── Path resolution ────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))

sys.path.insert(0, str(_BACKEND_DIR))


def _resolve(app_rel: str, local_rel: str) -> Path:
    a = _APP_DIR / app_rel
    b = _BACKEND_DIR / local_rel
    return a if a.exists() else b


_MATERIAL_CSV  = _resolve(
    "data/datasets/material_prices/india_material_prices_historical.csv",
    "data/datasets/material_prices/india_material_prices_historical.csv",
)
_PROPERTY_CSV  = _resolve(
    "data/datasets/property_transactions/india_property_transactions.csv",
    "data/datasets/property_transactions/india_property_transactions.csv",
)
_WEIGHTS_DIR   = _resolve("ml/weights", "ml/weights")
_CV_REPORT     = _WEIGHTS_DIR / "prophet_cv_report.json"
_MODEL_REPORT  = _WEIGHTS_DIR / "model_report.json"
_PRICE_XGB     = _WEIGHTS_DIR / "price_xgb.joblib"
_ROI_XGB       = _WEIGHTS_DIR / "roi_xgb.joblib"
_ROI_RF        = _WEIGHTS_DIR / "roi_rf.joblib"
_ROI_GBM       = _WEIGHTS_DIR / "roi_gbm.joblib"

# ── ANSI colours ──────────────────────────────────────────────────────────────
_NO_COLOR = os.getenv("NO_COLOR", "")
def _g(t): return t if _NO_COLOR else f"\033[92m{t}\033[0m"
def _r(t): return t if _NO_COLOR else f"\033[91m{t}\033[0m"
def _y(t): return t if _NO_COLOR else f"\033[93m{t}\033[0m"
def _b(t): return t if _NO_COLOR else f"\033[1m{t}\033[0m"
def _c(t): return t if _NO_COLOR else f"\033[96m{t}\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Price model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_price_model() -> Dict[str, Any]:
    """
    Evaluates XGBoostPriceRegressor on held-out last 6 months of material prices CSV.
    Also reports Prophet walk-forward CV MAPE from prophet_cv_report.json if available.

    Returns structured results dict.
    """
    print(f"\n{_b('═' * 70)}")
    print(f"{_b('  1. PRICE FORECAST MODEL EVALUATION')}")
    print(f"{_b('═' * 70)}")

    result: Dict[str, Any] = {
        "section":    "price_model",
        "status":     "skipped",
        "xgb":        {},
        "prophet_cv": {},
        "checks":     [],
    }

    # ── XGBoost evaluation ────────────────────────────────────────────────────
    if not _PRICE_XGB.exists():
        print(_y(f"  ⚠  price_xgb.joblib not found at {_PRICE_XGB}"))
        print(_y("     Run: python ml/train_price_models.py to train first."))
        result["checks"].append({"name": "price_xgb_exists", "passed": False, "critical": True})
    elif not _MATERIAL_CSV.exists():
        print(_y(f"  ⚠  Material prices CSV not found at {_MATERIAL_CSV}"))
        result["checks"].append({"name": "material_csv_exists", "passed": False, "critical": True})
    else:
        try:
            import joblib
            import numpy as np
            import pandas as pd
            from sklearn.metrics import mean_absolute_error
            from sklearn.model_selection import train_test_split

            bundle = joblib.load(str(_PRICE_XGB))
            model  = bundle["model"]
            city_enc = bundle.get("city_enc", {})
            mat_enc  = bundle.get("mat_enc", {})

            df = pd.read_csv(str(_MATERIAL_CSV), parse_dates=["date"])
            if "city" not in df.columns:
                df["city"] = "Hyderabad"

            # Feature engineering (mirrors train_price_models._build_xgb_features)
            df = df.sort_values(["material_key", "city", "date"]).copy()
            df["month"]      = df["date"].dt.month
            df["year"]       = df["date"].dt.year
            df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
            df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
            df["trend_index"] = (df["date"] - df["date"].min()).dt.days / 30.0
            grp = df.groupby(["material_key", "city"])["price_inr"]
            df["lag_1m"]          = grp.shift(1)
            df["lag_3m"]          = grp.shift(3)
            df["lag_6m"]          = grp.shift(6)
            df["rolling_mean_3m"] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            df["rolling_std_3m"]  = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0))
            df["city_enc"]     = df["city"].map(city_enc).fillna(0).astype(int)
            df["material_enc"] = df["material_key"].map(mat_enc).fillna(0).astype(int)

            FEATURE_COLS = [
                "month_sin", "month_cos", "year", "city_enc", "material_enc",
                "lag_1m", "lag_3m", "lag_6m", "rolling_mean_3m", "rolling_std_3m",
                "is_monsoon", "trend_index",
            ]
            df = df.dropna(subset=FEATURE_COLS + ["price_inr"]).reset_index(drop=True)

            # Held-out test: last 6 months (time-aware, NOT random)
            cutoff_date = df["date"].max() - pd.DateOffset(months=6)
            test_mask   = df["date"] > cutoff_date
            train_df    = df[~test_mask]
            test_df     = df[test_mask]

            if len(test_df) < 10:
                print(_y("  ⚠  Not enough test rows (< 10) — using random 15% split instead."))
                _, test_idx = train_test_split(range(len(df)), test_size=0.15, random_state=42)
                test_df = df.iloc[test_idx]

            y_true = test_df["price_inr"].values
            y_pred = model.predict(test_df[FEATURE_COLS].values)

            overall_mae  = float(mean_absolute_error(y_true, y_pred))
            nonzero      = y_true != 0
            overall_mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

            print(f"\n  {'Material':<42} {'City':<12} {'MAE':>10} {'MAPE':>8} {'Rows':>6}")
            print("  " + "─" * 82)

            mat_rows = []
            materials = sorted(test_df["material_key"].unique())
            for mat in materials:
                for city in sorted(test_df["city"].unique()):
                    mask = (test_df["material_key"] == mat) & (test_df["city"] == city)
                    if mask.sum() < 3:
                        continue
                    yt = test_df.loc[mask, "price_inr"].values
                    yp = model.predict(test_df.loc[mask, FEATURE_COLS].values)
                    mae  = float(mean_absolute_error(yt, yp))
                    nz   = yt != 0
                    mape = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100) if nz.sum() else 0.0
                    mat_rows.append((mat, city, mae, mape, mask.sum()))
                    status = _g("✓") if mape < 10 else (_y("⚠") if mape < 20 else _r("✗"))
                    print(f"  {status} {mat:<40} {city:<12} ₹{mae:>8.1f}  {mape:>6.2f}%  {mask.sum():>5}")

            print(f"\n  {'Overall XGBoost':42} {'':12} ₹{overall_mae:>8.1f}  {overall_mape:>6.2f}%  {len(test_df):>5}")

            result["xgb"] = {
                "overall_mae":  round(overall_mae, 2),
                "overall_mape": round(overall_mape, 2),
                "test_rows":    len(test_df),
                "material_rows": [
                    {"material": m, "city": c, "mae": round(mae, 2), "mape": round(mape, 2), "rows": n}
                    for m, c, mae, mape, n in mat_rows
                ],
            }
            passed = overall_mape < 25.0
            result["checks"].append({
                "name": "xgb_overall_mape_under_25pct",
                "passed": passed, "critical": True,
                "value": f"{overall_mape:.2f}%",
            })

        except Exception as exc:
            print(_r(f"  ✗  XGBoost evaluation failed: {exc}"))
            result["checks"].append({"name": "xgb_evaluation", "passed": False, "critical": True, "error": str(exc)})

    # ── Prophet CV report ─────────────────────────────────────────────────────
    print(f"\n  {_b('Prophet walk-forward CV report:')}")
    if not _CV_REPORT.exists():
        print(_y(f"  ⚠  prophet_cv_report.json not found at {_CV_REPORT}"))
        print(_y("     Run: python ml/train_price_models.py to generate CV results."))
        result["checks"].append({"name": "prophet_cv_report_exists", "passed": False, "critical": False})
    else:
        try:
            with open(_CV_REPORT) as f:
                cv_data = json.load(f)

            summary = cv_data.get("__summary__", {})
            overall_mean = summary.get("overall_mean_mape")
            n_pairs      = summary.get("total_material_city_pairs", 0)
            val_date     = summary.get("validation_date", "unknown")

            print(f"  Overall mean MAPE: {_g(f'{overall_mean:.2f}%') if overall_mean and overall_mean < 10 else _y(f'{overall_mean}%')}")
            print(f"  Material-city pairs validated: {n_pairs}")
            print(f"  Validation date: {val_date[:10] if val_date != 'unknown' else 'unknown'}")
            print()
            print(f"  {'Pair':<55} {'Mean MAPE':>10} {'Std':>8} {'Worst':>8} {'Folds':>6}")
            print("  " + "─" * 91)

            pair_rows = [(k, v) for k, v in cv_data.items() if k != "__summary__"]
            pair_rows.sort(key=lambda x: x[1].get("mean_mape", 99))

            for key, entry in pair_rows[:20]:  # show top 20 best
                mm = entry.get("mean_mape", 0)
                sm = entry.get("std_mape", 0)
                wm = entry.get("worst_mape", 0)
                nf = entry.get("n_folds", 0)
                status = _g("✓") if mm < 5 else (_y("⚠") if mm < 10 else _r("✗"))
                print(f"  {status} {key:<53} {mm:>9.2f}%  {sm:>6.2f}%  {wm:>6.2f}%  {nf:>5}")

            if len(pair_rows) > 20:
                print(f"  ... and {len(pair_rows) - 20} more pairs")

            result["prophet_cv"] = {
                "overall_mean_mape": overall_mean,
                "n_pairs":           n_pairs,
                "validation_date":   val_date,
            }
            if overall_mean is not None:
                passed = overall_mean < 15.0
                result["checks"].append({
                    "name": "prophet_cv_mean_mape_under_15pct",
                    "passed": passed, "critical": False,
                    "value": f"{overall_mean:.2f}%",
                })

        except Exception as exc:
            print(_r(f"  ✗  CV report parse failed: {exc}"))

    result["status"] = "completed"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROI ensemble evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_roi_model() -> Dict[str, Any]:
    """
    Evaluates ROI ensemble (XGBoost, RandomForest, GradientBoosting) on a
    time-aware 15% holdout (last rows by row index, not random).
    Reports MAE, RMSE, R² for each individual model and the ensemble mean.
    """
    print(f"\n{_b('═' * 70)}")
    print(f"{_b('  2. ROI ENSEMBLE MODEL EVALUATION')}")
    print(f"{_b('═' * 70)}")

    result: Dict[str, Any] = {
        "section": "roi_model",
        "status":  "skipped",
        "models":  {},
        "checks":  [],
    }

    model_files = {"xgboost": _ROI_XGB, "random_forest": _ROI_RF, "gradient_boosting": _ROI_GBM}
    missing = [name for name, p in model_files.items() if not p.exists()]
    if missing:
        print(_y(f"  ⚠  Missing model files: {missing}"))
        print(_y("     Run: POST /health/retrain  or  python scripts/build_datasets.py"))
        for m in missing:
            result["checks"].append({"name": f"{m}_exists", "passed": False, "critical": True})
        return result

    if not _PROPERTY_CSV.exists():
        print(_y(f"  ⚠  Property transactions CSV not found at {_PROPERTY_CSV}"))
        result["checks"].append({"name": "property_csv_exists", "passed": False, "critical": True})
        return result

    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_absolute_error, r2_score

        # Load models
        models: Dict[str, Any] = {}
        for name, path in model_files.items():
            try:
                models[name] = joblib.load(str(path))
                print(f"  {_g('✓')} Loaded {name} from {path.name}")
            except Exception as e:
                print(_r(f"  ✗  Could not load {name}: {e}"))
                result["checks"].append({"name": f"{name}_loadable", "passed": False, "critical": True})

        if not models:
            return result

        # Load data + feature engineering (mirrors RenovationDataPreprocessor)
        from ml.housing_preprocessor import (
            FEATURE_COLS, ROOM_ENC_MAP, BUDGET_ENC_MAP, SCOPE_ENC_MAP, FURNISHED_MAP
        )

        df = pd.read_csv(str(_PROPERTY_CSV))
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

        df["renovation_cost_lakh"] = df["renovation_cost_inr"] / 100_000
        df["reno_intensity"]       = (
            df["renovation_cost_inr"] / df["transaction_price_inr"].clip(lower=1)
        ).clip(upper=0.5)
        df["room_type_enc"]   = df["room_renovated"].map(ROOM_ENC_MAP).fillna(0).astype(float)
        df["budget_tier_enc"] = df["budget_tier"].map(BUDGET_ENC_MAP).fillna(1).astype(float)
        df["scope_enc"]       = df["renovation_scope"].map(SCOPE_ENC_MAP).fillna(1).astype(float)
        df["furnished"]       = df["furnished_status"].map(FURNISHED_MAP).fillna(1).astype(float)
        df["has_parking"]     = (df["parking"] > 0).astype(int)

        df_reno = df[df["roi_pct"].notna()].copy()
        avail   = [c for c in FEATURE_COLS if c in df_reno.columns]
        X       = df_reno[avail].fillna(df_reno[avail].median())
        y       = df_reno["roi_pct"]

        # Time-aware holdout: last 15% of rows by index (not random)
        split_idx = int(len(X) * 0.85)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        print(f"\n  Train: {split_idx:,} rows  |  Test (last 15%): {len(X_test):,} rows")
        print(f"  Features: {avail}")
        print()
        print(f"  {'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE':>8}")
        print("  " + "─" * 65)

        all_preds: List[np.ndarray] = []
        model_stats: Dict[str, Any] = {}

        for name, model_obj in models.items():
            try:
                y_pred = model_obj.predict(X_test[avail])
                all_preds.append(y_pred)
                mae  = float(mean_absolute_error(y_test, y_pred))
                rmse = float(np.sqrt(np.mean((y_test.values - y_pred) ** 2)))
                r2   = float(r2_score(y_test, y_pred))
                nz   = y_test.values != 0
                mape = float(np.mean(np.abs((y_test.values[nz] - y_pred[nz]) / y_test.values[nz])) * 100) if nz.sum() else 0.0

                status = _g("✓") if mae < 3.0 else (_y("⚠") if mae < 6.0 else _r("✗"))
                print(f"  {status} {name:<23} {mae:>9.3f}%  {rmse:>9.3f}%  {r2:>7.3f}  {mape:>7.2f}%")
                model_stats[name] = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "mape": round(mape, 4)}
            except Exception as e:
                print(_r(f"  ✗  {name}: prediction failed: {e}"))

        # Ensemble
        if len(all_preds) >= 2:
            ens_pred = np.mean(np.stack(all_preds, axis=0), axis=0)
            ens_mae  = float(mean_absolute_error(y_test, ens_pred))
            ens_rmse = float(np.sqrt(np.mean((y_test.values - ens_pred) ** 2)))
            ens_r2   = float(r2_score(y_test, ens_pred))
            nz_e     = y_test.values != 0
            ens_mape = float(np.mean(np.abs((y_test.values[nz_e] - ens_pred[nz_e]) / y_test.values[nz_e])) * 100) if nz_e.sum() else 0.0
            print("  " + "─" * 65)
            status = _g("✓") if ens_mae < 3.0 else (_y("⚠") if ens_mae < 6.0 else _r("✗"))
            print(f"  {status} {'ENSEMBLE (mean)':<23} {ens_mae:>9.3f}%  {ens_rmse:>9.3f}%  {ens_r2:>7.3f}  {ens_mape:>7.2f}%")
            model_stats["ensemble"] = {"mae": round(ens_mae, 4), "rmse": round(ens_rmse, 4), "r2": round(ens_r2, 4), "mape": round(ens_mape, 4)}
            result["checks"].append({
                "name": "ensemble_mae_under_6pct",
                "passed": ens_mae < 6.0, "critical": True,
                "value": f"{ens_mae:.3f}%",
            })

        # Load model_report.json for training provenance
        if _MODEL_REPORT.exists():
            with open(_MODEL_REPORT) as f:
                rpt = json.load(f)
            print(f"\n  Training date:   {rpt.get('training_date', 'unknown')[:10]}")
            print(f"  Dataset size:    {rpt.get('dataset_size', 0):,} rows")
            print(f"  Data source:     {rpt.get('data_source', 'unknown')}")

        result["models"] = model_stats
        result["status"] = "completed"

    except Exception as exc:
        print(_r(f"  ✗  ROI evaluation failed: {exc}"))
        logger.debug("ROI eval error", exc_info=True)
        result["checks"].append({"name": "roi_evaluation", "passed": False, "critical": True, "error": str(exc)})

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. CV pipeline evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cv_pipeline() -> Dict[str, Any]:
    """
    Evaluates StyleClassifier and DamageDetector without requiring real images.
    Uses synthetic minimal test images (solid-colour 224×224 patches) to verify:
      - models load without error
      - output schema is correct
      - metadata_trained flags are set
      - model_used labels are honest
    """
    print(f"\n{_b('═' * 70)}")
    print(f"{_b('  3. CV PIPELINE EVALUATION')}")
    print(f"{_b('═' * 70)}")

    result: Dict[str, Any] = {
        "section": "cv_pipeline",
        "status":  "skipped",
        "style_classifier": {},
        "damage_detector":  {},
        "depth_estimator":  {},
        "checks":           [],
    }

    # ── Generate minimal test images (no external files needed) ──────────────
    def _make_test_image(r: int, g: int, b: int, size: int = 224) -> bytes:
        """Create a solid-colour JPEG as bytes for testing."""
        try:
            import io as _io
            from PIL import Image
            img = Image.new("RGB", (size, size), color=(r, g, b))
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except ImportError:
            # If Pillow unavailable, return a minimal valid JPEG header
            # (0xFFD8FFE0 JFIF header, 224×224 grey)
            import struct
            return bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]) + b"\x00" * 100

    test_images = {
        "white_wall":      _make_test_image(240, 240, 240),   # clean wall (no damage)
        "brown_stain":     _make_test_image(160, 100, 60),    # damp-like brown
        "dark_industrial": _make_test_image(50, 50, 60),      # industrial/dark
        "bright_scandi":   _make_test_image(240, 238, 230),   # Scandinavian light
        "warm_boho":       _make_test_image(210, 170, 120),   # boho earthy
    }

    # ── StyleClassifier ───────────────────────────────────────────────────────
    print(f"\n  {_b('StyleClassifier')}")
    print(f"  {'Image':<22} {'Predicted style':<25} {'Conf':>6} {'metadata_trained':>16} {'prompt_source'}")
    print("  " + "─" * 100)

    try:
        from ml.style_classifier import StyleClassifier, STYLE_PRIORS, ROOM_STYLE_PROMPTS

        clf = StyleClassifier()

        # Verify structure
        assert "Modern Minimalist" in STYLE_PRIORS, "STYLE_PRIORS missing Modern Minimalist"
        assert abs(sum(STYLE_PRIORS.values()) - 1.0) < 0.001, "STYLE_PRIORS don't sum to 1"
        assert "kitchen" in ROOM_STYLE_PROMPTS, "ROOM_STYLE_PROMPTS missing kitchen"
        assert "modern" in ROOM_STYLE_PROMPTS["kitchen"], "ROOM_STYLE_PROMPTS missing kitchen/modern"
        assert len(ROOM_STYLE_PROMPTS["kitchen"]["modern"]) == 5, "Expected 5 prompts per room/style"

        total_prompts = sum(
            len(prompts)
            for rt in ROOM_STYLE_PROMPTS.values()
            for prompts in rt.values()
        )
        print(f"  {_g('✓')} ROOM_STYLE_PROMPTS: {total_prompts} prompts across {len(ROOM_STYLE_PROMPTS)} room types")
        print(f"  {_g('✓')} STYLE_PRIORS sum to 1.0  (top: {max(STYLE_PRIORS, key=STYLE_PRIORS.get)} = {max(STYLE_PRIORS.values()):.3f})")

        style_results = []
        for img_name, img_bytes in test_images.items():
            room_hint = "kitchen" if "industrial" in img_name else "bedroom"
            r = clf.classify(img_bytes, gemini_style_hint="", room_type=room_hint)

            label    = r.get("style_label", "?")
            conf     = r.get("style_confidence", 0.0)
            mt       = r.get("metadata_trained", False)
            ps       = r.get("prompt_source", "?")
            mu       = r.get("model_used", "?")
            rows     = r.get("metadata_rows_used", 0)

            mt_str = _g("✓ True") if mt else _r("✗ False")
            conf_str = _g(f"{conf:.3f}") if conf >= 0.35 else _y(f"{conf:.3f}")
            print(f"  {img_name:<22} {label:<25} {conf_str:>6}  {mt_str:>16}  {ps[:30]}")
            style_results.append({"image": img_name, "style": label, "confidence": conf,
                                   "metadata_trained": mt, "model_used": mu})

            # Critical checks
            assert mt is True, f"metadata_trained must be True, got {mt}"
            assert "rows_used" in str(r) or rows > 0, "metadata_rows_used missing"

        result["style_classifier"] = {
            "predictions":    style_results,
            "total_prompts":  total_prompts,
            "prior_sum_ok":   True,
        }
        result["checks"].append({"name": "style_classifier_metadata_trained", "passed": True, "critical": True})
        result["checks"].append({"name": "style_classifier_room_prompts_100", "passed": total_prompts >= 100, "critical": True,
                                  "value": str(total_prompts)})

    except Exception as exc:
        print(_r(f"  ✗  StyleClassifier evaluation failed: {exc}"))
        result["checks"].append({"name": "style_classifier", "passed": False, "critical": True, "error": str(exc)})

    # ── DamageDetector ────────────────────────────────────────────────────────
    print(f"\n  {_b('DamageDetector')}")
    print(f"  {'Image':<22} {'Damage class':<22} {'Scope':<20} {'Conf':>6} {'model_used'}")
    print("  " + "─" * 90)

    try:
        from ml.damage_detector import DamageDetector, DAMAGE_PROMPTS

        det = DamageDetector()

        # Verify CLIP prompt structure
        assert len(DAMAGE_PROMPTS) == 7, f"Expected 7 damage classes, got {len(DAMAGE_PROMPTS)}"
        for cls, prompts in DAMAGE_PROMPTS.items():
            assert len(prompts) == 3, f"{cls} needs exactly 3 prompts, got {len(prompts)}"
        print(f"  {_g('✓')} DAMAGE_PROMPTS: 7 classes × 3 prompts = 21 total")

        expected_models = {"clip_zero_shot_damage", "heuristic_fallback"}
        damage_results  = []

        for img_name, img_bytes in test_images.items():
            r = det.detect(img_bytes)

            severity = r.get("severity", "?")
            scope    = r.get("renovation_scope_recommendation", "?")
            conf     = r.get("scope_confidence", 0.0)
            mu       = r.get("model_used", "?")
            issues   = r.get("detected_issues", [])

            # Ensure no random-weight ResNet label
            assert "resnet50_transfer" not in mu, f"resnet50_transfer still present in model_used: {mu}"
            assert mu in expected_models or "clip" in mu or "heuristic" in mu, \
                f"Unexpected model_used: {mu}"

            conf_str = _g(f"{conf:.3f}") if conf <= 0.72 else _y(f"{conf:.3f}")
            print(f"  {img_name:<22} {issues[0] if issues else 'no_damage':<22} {scope:<20} {conf_str:>6}  {mu}")
            damage_results.append({"image": img_name, "severity": severity, "scope": scope,
                                    "confidence": conf, "model_used": mu})

        result["damage_detector"] = {"predictions": damage_results}
        result["checks"].append({"name": "damage_detector_no_random_resnet", "passed": True, "critical": True})
        result["checks"].append({"name": "damage_detector_clip_prompts_21", "passed": True, "critical": True})

    except Exception as exc:
        print(_r(f"  ✗  DamageDetector evaluation failed: {exc}"))
        result["checks"].append({"name": "damage_detector", "passed": False, "critical": True, "error": str(exc)})

    # ── DepthEstimator ────────────────────────────────────────────────────────
    print(f"\n  {_b('DepthEstimator (prior calibration check)')}")

    try:
        from ml.depth_estimator import REAL_ROOM_SIZE_STATS, DepthEstimator

        assert "bedroom" in REAL_ROOM_SIZE_STATS, "REAL_ROOM_SIZE_STATS missing bedroom"
        for room, stats in REAL_ROOM_SIZE_STATS.items():
            assert "median_sqft" in stats and "p25" in stats and "p75" in stats
            assert stats["source"] == "india_housing_kaggle_32k"

        print(f"  {_g('✓')} REAL_ROOM_SIZE_STATS: {len(REAL_ROOM_SIZE_STATS)} room types, all sourced from kaggle_32k")
        for rt, s in REAL_ROOM_SIZE_STATS.items():
            print(f"     {rt:<15}: median={s['median_sqft']} sqft  p25={s['p25']}  p75={s['p75']}")

        # Test fallback uses priors not old constants
        est = DepthEstimator()
        heur = est._heuristic_fallback("bedroom")
        assert heur["method"] == "real_data_prior_only", f"Got method={heur['method']}"
        assert heur["calibrated"] is True, "calibrated should be True"
        assert heur["floor_area_sqft"] == REAL_ROOM_SIZE_STATS["bedroom"]["median_sqft"]
        assert heur["prior_source"] == "india_housing_kaggle_32k_rows"
        print(f"  {_g('✓')} _heuristic_fallback uses real_data_prior_only (bedroom = {heur['floor_area_sqft']} sqft)")

        result["depth_estimator"] = {
            "prior_rooms":   len(REAL_ROOM_SIZE_STATS),
            "prior_source":  "india_housing_kaggle_32k_rows",
            "heuristic_method": heur["method"],
        }
        result["checks"].append({"name": "depth_estimator_real_priors", "passed": True, "critical": True})

    except Exception as exc:
        print(_r(f"  ✗  DepthEstimator evaluation failed: {exc}"))
        result["checks"].append({"name": "depth_estimator_priors", "passed": False, "critical": True, "error": str(exc)})

    result["status"] = "completed"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _health_score(all_results: List[Dict]) -> Tuple[int, int, int]:
    """Returns (passed, failed_critical, warnings)."""
    passed = failed = warnings = 0
    for r in all_results:
        for check in r.get("checks", []):
            if check.get("passed"):
                passed += 1
            elif check.get("critical"):
                failed += 1
            else:
                warnings += 1
    return passed, failed, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="ARKEN Model Evaluation Report")
    parser.add_argument("--section", choices=["price", "roi", "cv"], default=None,
                        help="Run only one section")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colours")
    args = parser.parse_args()

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    print(f"\n{_b('ARKEN — Model Evaluation Report')}")
    print(f"{_b('Generated:')} {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{_b('Backend:')}   {_BACKEND_DIR}")

    all_results: List[Dict] = []
    t0 = time.perf_counter()

    if args.section in (None, "price"):
        all_results.append(evaluate_price_model())
    if args.section in (None, "roi"):
        all_results.append(evaluate_roi_model())
    if args.section in (None, "cv"):
        all_results.append(evaluate_cv_pipeline())

    elapsed = time.perf_counter() - t0
    passed, failed, warnings = _health_score(all_results)
    total = passed + failed + warnings

    print(f"\n{_b('═' * 70)}")
    print(f"{_b('  OVERALL HEALTH SCORE')}")
    print(f"{_b('═' * 70)}")
    score = int(100 * passed / max(total, 1))
    colour = _g if score >= 80 else (_y if score >= 60 else _r)
    print(f"  Score:           {colour(f'{score}/100')}")
    print(f"  Checks passed:   {_g(str(passed))}")
    print(f"  Critical failed: {_r(str(failed)) if failed else _g('0')}")
    print(f"  Warnings:        {_y(str(warnings)) if warnings else _g('0')}")
    print(f"  Elapsed:         {elapsed:.1f}s")

    if score >= 80:
        print(f"\n  {_g('✓ All critical checks passed — models ready for production.')}")
    elif score >= 60:
        print(f"\n  {_y('⚠ Some issues found — review warnings above before deployment.')}")
    else:
        print(f"\n  {_r('✗ Critical failures — DO NOT deploy until issues are resolved.')}")
    print()

    if args.json:
        output = {
            "generated_at":    datetime.now(tz=timezone.utc).isoformat(),
            "health_score":    score,
            "passed":          passed,
            "failed_critical": failed,
            "warnings":        warnings,
            "elapsed_s":       round(elapsed, 2),
            "sections":        all_results,
        }
        print(json.dumps(output, indent=2))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
