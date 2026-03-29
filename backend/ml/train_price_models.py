#!/usr/bin/env python3
"""
ARKEN — Standalone Price Model Training Script
================================================
Trains and evaluates both the XGBoostPriceRegressor and ProphetForecastEngine
models against the real historical CSV.

Usage:
    python backend/ml/train_price_models.py [--csv PATH] [--output-dir PATH]

Options:
    --csv        Path to india_material_prices_historical.csv
                 Default: /app/data/datasets/material_prices/india_material_prices_historical.csv
    --output-dir Directory to save trained models
                 Default: /app/ml/weights

Outputs:
    - backend/ml/weights/price_xgb.joblib          (XGBoost model bundle)
    - backend/ml/weights/prophet_models/<key>_<city>.pkl  (per Prophet model)
    - Console evaluation report: MAE, MAPE per material for both models
"""

import argparse
import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arken.train_price")

# ── Optional dependency guards ─────────────────────────────────────────────────

try:
    import numpy as np
    import pandas as pd
    _NUMPY_OK = True
except ImportError:
    logger.error("numpy / pandas not installed. Run: pip install numpy pandas")
    sys.exit(1)

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import joblib
    _XGB_OK = True
except ImportError:
    _XGB_OK = False
    logger.warning("XGBoost / scikit-learn / joblib not installed — XGB training skipped.")
    logger.warning("  Install: pip install xgboost scikit-learn joblib")

try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    _PROPHET_OK = False
    logger.warning("Facebook Prophet not installed — Prophet training skipped.")
    logger.warning("  Install: pip install prophet")

# ── Constants (kept in sync with price_forecast.py) ───────────────────────────

SEED_DATA_KEYS = [
    "cement_opc53_per_bag_50kg",
    "steel_tmt_fe500_per_kg",
    "teak_wood_per_cft",
    "kajaria_tiles_per_sqft",
    "copper_wire_per_kg",
    "sand_river_per_brass",
    "bricks_per_1000",
    "granite_per_sqft",
    "asian_paints_premium_per_litre",
    "pvc_upvc_window_per_sqft",
    "modular_kitchen_per_sqft",
    "bathroom_sanitary_set",
]

CITIES = ["Hyderabad", "Mumbai", "Bangalore", "Delhi NCR", "Chennai", "Pune"]

_FEATURE_COLS = [
    "month_sin", "month_cos", "year", "city_enc", "material_enc",
    "lag_1m", "lag_3m", "lag_6m",
    "rolling_mean_3m", "rolling_std_3m",
    "is_monsoon", "trend_index",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(str(csv_path), parse_dates=["date"])
    required = {"date", "material_key", "price_inr"}
    if not required.issubset(df.columns):
        logger.error(f"CSV missing columns: {required - set(df.columns)}")
        sys.exit(1)

    if "city" not in df.columns:
        df["city"] = "Hyderabad"

    df = df.sort_values(["material_key", "city", "date"]).reset_index(drop=True)
    logger.info(
        f"Loaded {len(df):,} rows  |  "
        f"{df['material_key'].nunique()} materials  |  "
        f"{df['city'].nunique()} cities  |  "
        f"Date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost training
# ─────────────────────────────────────────────────────────────────────────────

def _build_xgb_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    df = df.copy()
    df["month"]       = df["date"].dt.month
    df["year"]        = df["date"].dt.year
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["is_monsoon"]  = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["trend_index"] = (df["date"] - df["date"].min()).dt.days / 30.0

    grp = df.groupby(["material_key", "city"])["price_inr"]
    df["lag_1m"]          = grp.shift(1)
    df["lag_3m"]          = grp.shift(3)
    df["lag_6m"]          = grp.shift(6)
    df["rolling_mean_3m"] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["rolling_std_3m"]  = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0))

    city_cats     = df["city"].astype("category")
    city_enc      = {v: k for k, v in enumerate(city_cats.cat.categories)}
    df["city_enc"] = df["city"].map(city_enc).fillna(0).astype(int)

    mat_cats      = df["material_key"].astype("category")
    mat_enc       = {v: k for k, v in enumerate(mat_cats.cat.categories)}
    df["material_enc"] = df["material_key"].map(mat_enc).fillna(0).astype(int)

    return df, city_enc, mat_enc


def train_xgboost(df: pd.DataFrame, output_dir: Path) -> Optional[Dict]:
    if not _XGB_OK:
        logger.warning("Skipping XGBoost training (dependencies missing).")
        return None

    logger.info("=" * 60)
    logger.info("TRAINING XGBoostPriceRegressor")
    logger.info("=" * 60)
    t0 = time.time()

    feat_df, city_enc, mat_enc = _build_xgb_features(df)
    feat_df = feat_df.dropna(subset=_FEATURE_COLS + ["price_inr"])

    X = feat_df[_FEATURE_COLS].values
    y = feat_df["price_inr"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    logger.info(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    nonzero = y_test != 0
    mape   = float(np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100)

    elapsed = time.time() - t0
    logger.info(f"  Overall  →  MAE: ₹{mae:.2f}   MAPE: {mape:.2f}%   ({elapsed:.1f}s)")

    # Per-material evaluation
    logger.info("\n  Per-material evaluation (test set):")
    logger.info(f"  {'Material':<42} {'MAE':>10} {'MAPE':>8}")
    logger.info("  " + "-" * 62)

    mat_col = feat_df.iloc[len(X_train):]["material_key"].values  # approximate split
    # More accurate: re-split with index tracking
    feat_df_clean = feat_df.dropna(subset=_FEATURE_COLS + ["price_inr"]).reset_index(drop=True)
    _, test_idx   = train_test_split(range(len(feat_df_clean)), test_size=0.15, random_state=42)
    test_df       = feat_df_clean.iloc[test_idx]
    preds_full    = model.predict(test_df[_FEATURE_COLS].values)

    material_report = []
    for mat in sorted(SEED_DATA_KEYS):
        mask = test_df["material_key"].values == mat
        if mask.sum() == 0:
            continue
        y_m    = test_df["price_inr"].values[mask]
        y_p_m  = preds_full[mask]
        mae_m  = mean_absolute_error(y_m, y_p_m)
        nz     = y_m != 0
        mape_m = float(np.mean(np.abs((y_m[nz] - y_p_m[nz]) / y_m[nz])) * 100) if nz.sum() > 0 else 0
        material_report.append((mat, mae_m, mape_m))
        logger.info(f"  {mat:<42} ₹{mae_m:>8.2f}  {mape_m:>6.2f}%")

    # Save model
    save_path = output_dir / "price_xgb.joblib"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "city_enc": city_enc, "mat_enc": mat_enc}, str(save_path))
    logger.info(f"\n  ✓ XGBoost model saved → {save_path}")

    return {
        "model": model, "city_enc": city_enc, "mat_enc": mat_enc,
        "overall_mae": mae, "overall_mape": mape,
        "material_report": material_report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prophet training
# ─────────────────────────────────────────────────────────────────────────────

def _month_factor(month: int) -> float:
    return {1: 1.02, 2: 1.03, 3: 1.05, 4: 1.01, 5: 0.99, 6: 0.96,
            7: 0.94, 8: 0.93, 9: 0.97, 10: 1.04, 11: 1.06, 12: 1.03}.get(month, 1.0)


def _prophet_evaluate(model_entry: Dict, test_df: pd.DataFrame) -> Tuple[float, float]:
    """Evaluate a fitted Prophet model on held-out test data."""
    model   = model_entry["model"]
    has_reg = model_entry["has_regressor"]

    future = model.make_future_dataframe(periods=len(test_df), freq="MS")
    if has_reg:
        future["inflation_proxy"] = model_entry["last_inflation"]
    forecast = model.predict(future)

    # Align on dates
    forecast = forecast.set_index("ds")
    test_dates = pd.to_datetime(test_df["ds"])
    preds  = forecast.loc[forecast.index.isin(test_dates), "yhat"].values
    actuals = test_df["y"].values[:len(preds)]
    if len(preds) == 0:
        return float("nan"), float("nan")
    mae  = float(np.mean(np.abs(actuals - preds)))
    nz   = actuals != 0
    mape = float(np.mean(np.abs((actuals[nz] - preds[nz]) / actuals[nz])) * 100) if nz.sum() > 0 else 0.0
    return mae, mape


def train_prophet(df: pd.DataFrame, output_dir: Path) -> Dict:
    if not _PROPHET_OK:
        logger.warning("Skipping Prophet training (dependencies missing).")
        return {}

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING ProphetForecastEngine")
    logger.info("=" * 60)

    prophet_dir = output_dir / "prophet_models"
    prophet_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, List] = {}
    n_fitted  = 0
    n_skipped = 0
    t0_total  = time.time()

    logger.info(f"\n  {'Material':<42} {'City':<12} {'Rows':>6} {'MAE':>10} {'MAPE':>8}")
    logger.info("  " + "-" * 82)

    for mat in sorted(SEED_DATA_KEYS):
        mat_report = []
        for city in CITIES:
            sub = df[(df["material_key"] == mat) & (df["city"] == city)].copy()
            if len(sub) < 24:
                n_skipped += 1
                continue

            sub = sub.sort_values("date").reset_index(drop=True)
            prophet_df = sub.rename(columns={"date": "ds", "price_inr": "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

            # Train/test split: last 6 months held out
            cutoff   = len(prophet_df) - 6
            train_df = prophet_df.iloc[:cutoff]
            test_df  = prophet_df.iloc[cutoff:]

            prophet_df["inflation_proxy"] = prophet_df["y"].pct_change(periods=12).fillna(0) * 100

            try:
                t0 = time.time()
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",
                    interval_width=0.80,
                    changepoint_prior_scale=0.15,
                    seasonality_prior_scale=10.0,
                )
                model.add_seasonality(
                    name="indian_construction",
                    period=365.25,
                    fourier_order=3,
                    prior_scale=5.0,
                )
                infl_std = prophet_df["inflation_proxy"].std()
                has_reg  = infl_std > 0.01
                if has_reg:
                    model.add_regressor("inflation_proxy")
                    train_fit = prophet_df.iloc[:cutoff][["ds", "y", "inflation_proxy"]]
                    model.fit(train_fit)
                else:
                    model.fit(prophet_df.iloc[:cutoff][["ds", "y"]])

                last_inflation = float(prophet_df["inflation_proxy"].iloc[-1])
                entry = {
                    "model": model,
                    "has_regressor": has_reg,
                    "last_inflation": last_inflation,
                    "n_rows": len(prophet_df),
                }

                mae, mape = _prophet_evaluate(
                    entry, test_df.rename(columns={"date": "ds", "price_inr": "y"})
                    if "ds" not in test_df.columns else test_df
                )

                elapsed = time.time() - t0
                logger.info(
                    f"  {mat:<42} {city:<12} {len(prophet_df):>6}  "
                    f"₹{mae:>8.2f}  {mape:>6.2f}%  ({elapsed:.1f}s)"
                )

                # Save prophet model
                safe_city = city.replace(" ", "_")
                pkl_path  = prophet_dir / f"{mat}__{safe_city}.pkl"
                with open(str(pkl_path), "wb") as f:
                    pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)

                mat_report.append({"city": city, "mae": mae, "mape": mape, "n_rows": len(prophet_df)})
                n_fitted += 1

            except Exception as e:
                logger.warning(f"  ✗ Failed {mat}|{city}: {e}")
                n_skipped += 1

        report[mat] = mat_report

    elapsed_total = time.time() - t0_total
    logger.info(f"\n  ✓ Prophet: {n_fitted} models saved → {prophet_dir}")
    logger.info(f"    Skipped: {n_skipped}  |  Total time: {elapsed_total:.1f}s")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Walk-forward cross-validation for Prophet price forecasting.

    Methodology:
      - Expanding windows: train on first N months, predict next 3 months, expand by 3
      - Minimum training window: 18 months
      - Collect MAPE per fold per material per city
      - Report: mean_mape, std_mape, worst_mape, best_mape, n_folds
      - Saves prophet_cv_report.json to output_dir

    Returns:
        Summary dict with overall_mean_mape and validation coverage counts.
    """
    if not _PROPHET_OK:
        logger.warning("[ProphetCV] Prophet not installed — skipping walk-forward CV.")
        return {"overall_mean_mape": None, "skipped": True}

    from datetime import datetime as _dt

    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD CROSS-VALIDATION (Prophet)")
    logger.info("=" * 60)

    results: Dict = {}
    material_keys = df["material_key"].unique()
    cities        = df["city"].unique()
    total_pairs   = len(material_keys) * len(cities)
    done          = 0

    for material_key in sorted(material_keys):
        for city in sorted(cities):
            done += 1
            subset = (
                df[(df["material_key"] == material_key) & (df["city"] == city)]
                .sort_values("date")
                .reset_index(drop=True)
            )
            if len(subset) < 24:
                continue

            prices = subset["price_inr"].values
            dates  = subset["date"].values
            fold_mapes: List[float] = []

            # Expanding window: train on [:start_end], predict [start_end:start_end+3]
            for start_end in range(18, len(prices) - 3, 3):
                train_y = prices[:start_end]
                test_y  = prices[start_end:start_end + 3]

                train_df = pd.DataFrame({
                    "ds": pd.to_datetime(dates[:start_end]),
                    "y":  train_y.astype(float),
                })

                try:
                    m = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode="multiplicative",
                        interval_width=0.80,
                    )
                    # Suppress Stan output
                    import logging as _log
                    _log.getLogger("cmdstanpy").setLevel(_log.WARNING)
                    _log.getLogger("prophet").setLevel(_log.WARNING)

                    m.fit(train_df, iter=300)
                    future = m.make_future_dataframe(periods=3, freq="MS")
                    fc     = m.predict(future)
                    pred_y = fc["yhat"].values[-3:]

                    nonzero = test_y != 0
                    if nonzero.sum() > 0:
                        mape = float(
                            np.mean(
                                np.abs((test_y[nonzero] - pred_y[nonzero]) / test_y[nonzero])
                            ) * 100
                        )
                        fold_mapes.append(mape)
                except Exception:
                    continue

            if fold_mapes:
                key = f"{material_key}|{city}"
                results[key] = {
                    "mean_mape":  round(float(np.mean(fold_mapes)), 2),
                    "std_mape":   round(float(np.std(fold_mapes)),  2),
                    "worst_mape": round(float(np.max(fold_mapes)),  2),
                    "best_mape":  round(float(np.min(fold_mapes)),  2),
                    "n_folds":    len(fold_mapes),
                    "material":   material_key,
                    "city":       city,
                }
                logger.info(
                    f"  [{done:>3}/{total_pairs}] {material_key:<42} {city:<12} "
                    f"mean_MAPE={results[key]['mean_mape']:.2f}%  "
                    f"n_folds={len(fold_mapes)}"
                )

    # Overall summary
    all_mapes = [v["mean_mape"] for v in results.values()]
    summary = {
        "overall_mean_mape":         round(float(np.mean(all_mapes)), 2) if all_mapes else None,
        "overall_median_mape":       round(float(np.median(all_mapes)), 2) if all_mapes else None,
        "materials_validated":       len(set(k.split("|")[0] for k in results)),
        "cities_validated":          len(set(k.split("|")[1] for k in results)),
        "total_material_city_pairs": len(results),
        "validation_date":           _dt.utcnow().isoformat(),
    }
    results["__summary__"] = summary

    # Save
    cv_report_path = output_dir / "prophet_cv_report.json"
    cv_report_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(str(cv_report_path), "w") as f:
        _json.dump(results, f, indent=2)

    logger.info(
        f"\n[ProphetCV] Walk-forward CV complete. "
        f"Overall mean MAPE: {summary['overall_mean_mape']}%  "
        f"({summary['total_material_city_pairs']} material-city pairs)"
    )
    logger.info(f"[ProphetCV] Report saved → {cv_report_path}")
    return summary


def print_summary(xgb_result, prophet_result):
    print("\n")
    print("=" * 70)
    print("  ARKEN PRICE FORECAST MODEL EVALUATION REPORT")
    print("=" * 70)

    if xgb_result:
        print(f"\n  XGBoost (global model across all materials & cities)")
        print(f"    Overall MAE:  ₹{xgb_result['overall_mae']:.2f}")
        print(f"    Overall MAPE: {xgb_result['overall_mape']:.2f}%")
        print()
        print(f"  {'Material':<42}  {'MAE':>10}  {'MAPE':>8}")
        print("  " + "-" * 64)
        for mat, mae, mape in sorted(xgb_result.get("material_report", []), key=lambda x: x[2]):
            print(f"  {mat:<42}  ₹{mae:>8.2f}  {mape:>6.2f}%")
    else:
        print("\n  XGBoost: not trained (dependencies missing)")

    if prophet_result:
        print(f"\n  Prophet (per material × city)")
        print(f"  {'Material':<42}  {'Avg MAE':>10}  {'Avg MAPE':>10}")
        print("  " + "-" * 66)
        for mat, city_results in sorted(prophet_result.items()):
            if not city_results:
                continue
            avg_mae  = float(np.mean([r["mae"]  for r in city_results if not np.isnan(r["mae"])]))
            avg_mape = float(np.mean([r["mape"] for r in city_results if not np.isnan(r["mape"])]))
            print(f"  {mat:<42}  ₹{avg_mae:>8.2f}  {avg_mape:>8.2f}%")
    else:
        print("\n  Prophet: not trained (dependencies missing)")

    print()
    print("=" * 70)
    print("  Training complete. Models ready for PriceForecastAgent v5.0.")
    print("=" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARKEN — Train XGBoost + Prophet price forecast models"
    )
    parser.add_argument(
        "--csv",
        default="/app/data/datasets/material_prices/india_material_prices_historical.csv",
        help="Path to india_material_prices_historical.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/app/ml/weights",
        help="Directory to save trained model files",
    )
    args = parser.parse_args()

    csv_path   = Path(args.csv)
    output_dir = Path(args.output_dir)

    logger.info("ARKEN Price Model Training Script")
    logger.info(f"  CSV:        {csv_path}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  XGBoost:    {'available' if _XGB_OK else 'NOT INSTALLED'}")
    logger.info(f"  Prophet:    {'available' if _PROPHET_OK else 'NOT INSTALLED'}")

    df = load_csv(csv_path)

    xgb_result     = train_xgboost(df, output_dir)
    prophet_result = train_prophet(df, output_dir)

    print_summary(xgb_result, prophet_result)

    # Walk-forward cross-validation — runs after training so models are already saved
    logger.info("\nRunning walk-forward cross-validation for Prophet...")
    cv_summary = walk_forward_cv(df, output_dir)
    if cv_summary.get("overall_mean_mape") is not None:
        logger.info(
            f"[ProphetCV] Cross-validation complete — "
            f"overall mean MAPE: {cv_summary['overall_mean_mape']:.2f}%  "
            f"across {cv_summary['total_material_city_pairs']} material-city pairs"
        )



if __name__ == "__main__":
    main()