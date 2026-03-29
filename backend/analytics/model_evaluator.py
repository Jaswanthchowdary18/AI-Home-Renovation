"""
ARKEN — Model Evaluator v1.0
===============================
Runs proper held-out evaluation across all three ML model families and
produces a dashboard-ready metrics dict served at GET /api/v1/health/metrics.

Evaluation methods:
  1. evaluate_style_classifier() — accuracy, F1, confusion matrix on val_data.csv
  2. evaluate_roi_model()        — MAE, RMSE, R², CI coverage on held-out test set
  3. evaluate_price_forecast()   — MAPE, MAE, directional accuracy per material

Results are cached to ml/weights/*_eval_latest.json (24h TTL).
Stale cache triggers re-evaluation on next get_all_metrics() call.

Design:
  - Lazy imports for torch/CLIP — evaluator import does NOT slow startup.
  - Never raises — returns {"error": ..., "partial_results": ...} on failure.
  - Thread-safe: evaluations can be triggered from any FastAPI route.

Usage:
    from analytics.model_evaluator import ModelEvaluator
    ev = ModelEvaluator()
    metrics = ev.get_all_metrics()   # cached, fast
    ev.run_full_evaluation()         # force re-evaluation
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))

def _resolve(app_rel: str, local_rel: str) -> Path:
    a = _APP_DIR    / app_rel
    b = _BACKEND_DIR / local_rel
    return a if a.exists() else b

_WEIGHTS_DIR = _resolve("ml/weights", "ml/weights")
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

_STYLE_EVAL_PATH = _WEIGHTS_DIR / "style_eval_latest.json"
_ROI_EVAL_PATH   = _WEIGHTS_DIR / "roi_eval_latest.json"
_PRICE_EVAL_PATH = _WEIGHTS_DIR / "price_eval_latest.json"

_CACHE_TTL_HOURS = 24
_EVAL_LOCK       = threading.Lock()

# Dataset paths
_VAL_CSV_PATHS = [
    _resolve("data/datasets/interior_design_images_metadata/val_data.csv",
             "data/datasets/interior_design_images_metadata/val_data.csv"),
    _resolve("data/datasets/interior_design_material_style/val_data.csv",
             "data/datasets/interior_design_material_style/val_data.csv"),
]
_MATERIAL_CSV = _resolve(
    "data/datasets/material_prices/india_material_prices_historical.csv",
    "data/datasets/material_prices/india_material_prices_historical.csv",
)


def _is_cache_fresh(path: Path, ttl_hours: int = _CACHE_TTL_HOURS) -> bool:
    """Return True if the JSON cache file exists and is younger than ttl_hours."""
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < ttl_hours


def _save_json(path: Path, data: Dict) -> None:
    data["evaluated_at"] = datetime.now(tz=timezone.utc).isoformat()
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    except Exception as e:
        logger.warning(f"[ModelEvaluator] Could not save {path.name}: {e}")


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


class ModelEvaluator:
    """
    Runs held-out evaluation for all three ARKEN ML model families.

    Thread-safe: a global lock prevents concurrent evaluation runs.
    All evaluations degrade gracefully — never raise to the caller.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Style Classifier Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_style_classifier(self) -> Dict[str, Any]:
        """
        Evaluate StyleClassifier on interior_design_material_style/val_data.csv.

        v2.0 — Fine-tuned model support:
          PRIMARY:  style_classifier.pt (fine-tuned EfficientNet-B0 on 5-class dataset)
          FALLBACK: clip_finetuned.pt CLIP encoder → zero-shot CLIP (dual-pass)

        The method auto-detects which tier is active from StyleClassifier's
        result["model_used"] and records it in the eval JSON as `model_tier`.

        Computes:
          - accuracy, top3_accuracy
          - macro_f1, per_class_f1, per_class_precision, per_class_recall
          - confusion_matrix_dict
          - model_tier: "efficientnet_finetuned" | "clip_finetuned" | "clip_zero_shot"
          - weight_path: actual weight file that was loaded (for traceability)

        Saves to ml/weights/style_eval_latest.json.

        Returns:
            {
              accuracy: float,
              top3_accuracy: float,
              macro_f1: float,
              per_class_f1: {style: float},
              per_class_precision: {style: float},
              per_class_recall: {style: float},
              confusion_matrix_dict: {true: {pred: count}},
              samples_evaluated: int,
              errors_skipped: int,
              model_tier: str,
              weight_path: str,
              model_used_counts: {model_path: count},
              evaluated_at: str,
            }
        """
        t0 = time.perf_counter()
        logger.info("[ModelEvaluator] Starting style classifier evaluation (v2 — fine-tuned primary)…")

        try:
            import csv as csv_module
            import sys
            sys.path.insert(0, str(_BACKEND_DIR))
            from ml.style_classifier import StyleClassifier, _CSV_TO_ARKEN

            # ── Determine which weight tier is active ─────────────────────────
            # Weight resolution mirrors StyleClassifier.__init__ priority order
            _wdir = _WEIGHTS_DIR
            _style_pt  = _wdir / "style_classifier.pt"
            _clip_ft   = _wdir / "clip_finetuned.pt"

            if _style_pt.exists() and _style_pt.stat().st_size > 1024:
                weight_path = str(_style_pt)
                expected_tier = "efficientnet_finetuned"
                logger.info(
                    f"[ModelEvaluator] Fine-tuned style_classifier.pt found "
                    f"({_style_pt.stat().st_size / 1e6:.1f} MB) — using as primary"
                )
            elif _clip_ft.exists() and _clip_ft.stat().st_size > 1024:
                weight_path = str(_clip_ft)
                expected_tier = "clip_finetuned"
                logger.info(
                    f"[ModelEvaluator] clip_finetuned.pt found — using as primary CLIP encoder"
                )
            else:
                weight_path = "pretrained_clip_zero_shot"
                expected_tier = "clip_zero_shot"
                logger.warning(
                    "[ModelEvaluator] No fine-tuned weights found — evaluating "
                    "zero-shot CLIP baseline. Accuracy will be lower than fine-tuned."
                )

            # ── Find val_data.csv ─────────────────────────────────────────────
            # Prefer interior_design_material_style (5-class dataset matching training)
            csv_path: Optional[Path] = None
            preferred_csv = _resolve(
                "data/datasets/interior_design_material_style/val_data.csv",
                "data/datasets/interior_design_material_style/val_data.csv",
            )
            if preferred_csv.exists():
                csv_path = preferred_csv
                logger.info(f"[ModelEvaluator] Using val CSV: {csv_path}")
            else:
                for candidate in _VAL_CSV_PATHS:
                    if candidate.exists():
                        csv_path = candidate
                        break

            if csv_path is None:
                return self._eval_error(
                    "style_classifier",
                    "val_data.csv not found. Expected at: "
                    "data/datasets/interior_design_material_style/val_data.csv"
                )

            # ── Resolve image root ────────────────────────────────────────────
            img_root = csv_path.parent
            for candidate in [
                _BACKEND_DIR / "data" / "datasets" / "interior_design_material_style",
                _BACKEND_DIR / "data" / "datasets" / "interior_design_images_metadata",
                _BACKEND_DIR / "data" / "raw",
                _BACKEND_DIR.parent / "data" / "raw",
            ]:
                if candidate.exists():
                    img_root = candidate
                    break

            # ── Load records from CSV ─────────────────────────────────────────
            records: List = []
            with open(csv_path, "r", encoding="utf-8-sig") as fh:
                reader = csv_module.DictReader(fh)
                for row in reader:
                    raw_path  = row.get("image_path", "").replace("\\", "/").replace("../", "")
                    room_type = row.get("room_type", "bedroom").strip()
                    style     = row.get("style", "").strip()
                    if not style:
                        continue
                    for root in [img_root, _BACKEND_DIR / "data" / "datasets", _BACKEND_DIR]:
                        candidate_path = root / raw_path
                        if candidate_path.exists():
                            records.append((candidate_path, room_type, style))
                            break

            if not records:
                return self._eval_error(
                    "style_classifier",
                    f"No images resolved from {csv_path}. "
                    "Place dataset images at interior_design_material_style/{{room_type}}/{{style}}/."
                )

            logger.info(
                f"[ModelEvaluator] Loaded {len(records)} val records from {csv_path.name}"
            )

            # ── Run evaluation ────────────────────────────────────────────────
            # Instantiate StyleClassifier once — it will auto-load fine-tuned
            # weights in priority order (EfficientNet → CLIP fine-tuned → zero-shot)
            clf = StyleClassifier()

            true_labels:  List[str]          = []
            pred_labels:  List[str]          = []
            top3_correct: int                = 0
            model_used_counts: Dict[str, int] = {}
            errors = 0
            observed_tiers: List[str]        = []

            # Cap at 500 samples for speed; shuffle for representative sampling
            import random
            eval_records = records.copy()
            random.shuffle(eval_records)
            eval_records = eval_records[:500]

            for img_path, room_type, true_style in eval_records:
                try:
                    image_bytes = img_path.read_bytes()
                    result      = clf.classify(image_bytes, room_type=room_type)
                    pred        = result["style_label"]
                    mu          = result.get("model_used", "unknown")
                    model_used_counts[mu] = model_used_counts.get(mu, 0) + 1
                    observed_tiers.append(mu)

                    # Map dataset style label → ARKEN canonical style
                    arken_true = _CSV_TO_ARKEN.get(true_style.lower(), true_style)
                    true_labels.append(arken_true)
                    pred_labels.append(pred)

                    # Top-3 accuracy
                    top3 = [x["style"] for x in result.get("top_3_styles", [])]
                    if arken_true in top3:
                        top3_correct += 1

                except Exception as e:
                    logger.debug(f"[ModelEvaluator] Skipped {img_path.name}: {e}")
                    errors += 1

            if not true_labels:
                return self._eval_error("style_classifier", "No images evaluated successfully")

            # ── Compute metrics ───────────────────────────────────────────────
            from collections import Counter, defaultdict

            correct  = sum(t == p for t, p in zip(true_labels, pred_labels))
            accuracy = correct / len(true_labels)
            top3_acc = top3_correct / len(true_labels)

            all_labels = sorted(set(true_labels + pred_labels))
            per_class_tp: Dict[str, int] = defaultdict(int)
            per_class_fp: Dict[str, int] = defaultdict(int)
            per_class_fn: Dict[str, int] = defaultdict(int)

            for t, p in zip(true_labels, pred_labels):
                if t == p:
                    per_class_tp[t] += 1
                else:
                    per_class_fp[p] += 1
                    per_class_fn[t] += 1

            per_class_precision: Dict[str, float] = {}
            per_class_recall:    Dict[str, float] = {}
            per_class_f1:        Dict[str, float] = {}

            for label in all_labels:
                tp  = per_class_tp[label]
                fp  = per_class_fp[label]
                fn  = per_class_fn[label]
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                per_class_precision[label] = round(prec, 4)
                per_class_recall[label]    = round(rec, 4)
                per_class_f1[label]        = round(f1, 4)

            macro_f1 = round(sum(per_class_f1.values()) / max(len(per_class_f1), 1), 4)

            # Confusion matrix
            confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for t, p in zip(true_labels, pred_labels):
                confusion[t][p] += 1

            # Determine actual model tier from observed model_used values
            if any("efficientnet" in m.lower() or "finetuned" in m.lower()
                   for m in observed_tiers):
                actual_tier = "efficientnet_finetuned"
            elif any("clip_finetuned" in m.lower() for m in observed_tiers):
                actual_tier = "clip_finetuned"
            else:
                actual_tier = "clip_zero_shot"

            elapsed = round(time.perf_counter() - t0, 2)
            result_dict = {
                "accuracy":              round(accuracy, 4),
                "top3_accuracy":         round(top3_acc, 4),
                "macro_f1":              macro_f1,
                "per_class_f1":          per_class_f1,
                "per_class_precision":   per_class_precision,
                "per_class_recall":      per_class_recall,
                "confusion_matrix_dict": {k: dict(v) for k, v in confusion.items()},
                "samples_evaluated":     len(true_labels),
                "errors_skipped":        errors,
                "model_tier":            actual_tier,
                "weight_path":           weight_path,
                "model_used_counts":     model_used_counts,
                "val_csv":               str(csv_path),
                "eval_time_seconds":     elapsed,
            }

            _save_json(_STYLE_EVAL_PATH, result_dict)
            logger.info(
                f"[ModelEvaluator] Style eval done in {elapsed}s: "
                f"acc={accuracy:.3f} macro_f1={macro_f1:.3f} "
                f"tier={actual_tier} n={len(true_labels)}"
            )
            return result_dict

        except Exception as e:
            logger.error(f"[ModelEvaluator] evaluate_style_classifier failed: {e}", exc_info=True)
            return self._eval_error("style_classifier", str(e))

            return self._eval_error("style_classifier", str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # 2. ROI Model Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_roi_model(self) -> Dict[str, Any]:
        """
        Evaluate ROI ensemble on held-out test rows from RenovationDataPreprocessor.

        Computes: MAE, RMSE, R², MAE by city, MAE by room type, CI coverage.
        Saves to ml/weights/roi_eval_latest.json.

        Returns:
            {
              mae_pct: float,
              rmse_pct: float,
              r2: float,
              mae_by_city: {city: float},
              mae_by_room_type: {room: float},
              ci_coverage_pct: float,
              test_rows: int,
              evaluated_at: str,
            }
        """
        t0 = time.perf_counter()
        logger.info("[ModelEvaluator] Starting ROI model evaluation …")

        try:
            import sys
            import numpy as np
            sys.path.insert(0, str(_BACKEND_DIR))
            from ml.housing_preprocessor import get_reno_preprocessor
            from ml.property_models import ROIModel

            rp = get_reno_preprocessor()
            try:
                X_tr, X_te, y_tr, y_te = rp.get_roi_splits(stratify_by_city_tier=True)
            except Exception as e:
                return self._eval_error("roi_model", f"Data loading failed: {e}")

            if len(X_te) < 10:
                return self._eval_error(
                    "roi_model",
                    f"Only {len(X_te)} test rows — need ≥10 for evaluation."
                )

            # Check weights exist before instantiating ROIModel
            # (ROIModel.__init__ triggers training if weights missing — avoid that in evaluator)
            roi_weight_files = [
                _WEIGHTS_DIR / "roi_xgb.joblib",
                _WEIGHTS_DIR / "roi_rf.joblib",
                _WEIGHTS_DIR / "roi_gbm.joblib",
            ]
            weights_present = all(p.exists() for p in roi_weight_files)
            if not weights_present:
                return self._eval_error(
                    "roi_model",
                    "ROI model weights not found. Run: python ml/train.py --model roi"
                )

            # Load trained model (weights exist — no training triggered)
            roi_model = ROIModel()
            if not ROIModel._models:
                return self._eval_error("roi_model", "ROI ensemble failed to load from weights.")

            # Predict
            preds_mean:  List[float] = []
            preds_low:   List[float] = []
            preds_high:  List[float] = []

            for i in range(len(X_te)):
                row_df = X_te.iloc[[i]]
                try:
                    mean_, low_, high_ = roi_model.predict(row_df)
                    preds_mean.append(float(mean_))
                    preds_low.append(float(low_))
                    preds_high.append(float(high_))
                except Exception:
                    preds_mean.append(0.0)
                    preds_low.append(0.0)
                    preds_high.append(0.0)

            y_true = y_te.values
            y_pred = np.array(preds_mean)

            # Overall metrics
            abs_errors = np.abs(y_true - y_pred)
            mae  = float(np.mean(abs_errors))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = float(1 - ss_res / max(ss_tot, 1e-10))

            # CI coverage
            low_arr  = np.array(preds_low)
            high_arr = np.array(preds_high)
            in_ci    = np.sum((y_true >= low_arr) & (y_true <= high_arr))
            ci_coverage = float(in_ci / max(len(y_true), 1) * 100)

            # MAE by city (if city column available in test features)
            mae_by_city: Dict[str, float] = {}
            try:
                df_full = rp.load()
                if "city" in df_full.columns:
                    df_reno = df_full[df_full["roi_pct"].notna()].copy()
                    feats   = [c for c in ROIModel.FEATURE_NAMES if c in df_reno.columns]
                    X_all   = df_reno[feats + ["city"]].copy()
                    y_all   = df_reno["roi_pct"].copy()
                    # Use test indices
                    X_test_with_city = X_all.iloc[-len(X_te):]
                    y_test_series    = y_all.iloc[-len(X_te):]
                    for city_name in X_test_with_city["city"].unique():
                        mask = X_test_with_city["city"] == city_name
                        y_c  = y_test_series[mask].values
                        idx  = mask.values.nonzero()[0]
                        if len(idx) >= 3:
                            p_c  = y_pred[idx]
                            mae_by_city[city_name] = round(float(np.mean(np.abs(y_c - p_c))), 4)
            except Exception as e:
                logger.debug(f"[ModelEvaluator] MAE by city skipped: {e}")

            # MAE by room type
            mae_by_room: Dict[str, float] = {}
            try:
                ROOM_DEC = {0: "bedroom", 1: "bathroom", 2: "living_room", 3: "kitchen", 4: "full_home"}
                if "room_type_enc" in X_te.columns:
                    for enc, room_name in ROOM_DEC.items():
                        mask = X_te["room_type_enc"].values == enc
                        if mask.sum() >= 3:
                            mae_by_room[room_name] = round(
                                float(np.mean(abs_errors[mask])), 4
                            )
            except Exception as e:
                logger.debug(f"[ModelEvaluator] MAE by room skipped: {e}")

            elapsed = round(time.perf_counter() - t0, 2)
            result_dict = {
                "mae_pct":             round(mae, 4),
                "rmse_pct":            round(rmse, 4),
                "r2":                  round(r2, 4),
                "mae_by_city":         mae_by_city,
                "mae_by_room_type":    mae_by_room,
                "ci_coverage_pct":     round(ci_coverage, 2),
                "test_rows":           len(X_te),
                "eval_time_seconds":   elapsed,
            }

            _save_json(_ROI_EVAL_PATH, result_dict)
            logger.info(
                f"[ModelEvaluator] ROI eval done in {elapsed}s: "
                f"MAE={mae:.3f}% RMSE={rmse:.3f}% R²={r2:.3f} CI_cov={ci_coverage:.1f}%"
            )
            return result_dict

        except Exception as e:
            logger.error(f"[ModelEvaluator] evaluate_roi_model failed: {e}", exc_info=True)
            return self._eval_error("roi_model", str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Price Forecast Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_price_forecast(self) -> Dict[str, Any]:
        """
        Hold out last 6 months of material price data per material, train on
        the rest, predict the held-out period, compute MAPE and directional accuracy.

        Saves to ml/weights/price_eval_latest.json.

        Returns:
            {
              overall_mape_pct: float,
              overall_mae_inr: float,
              directional_accuracy_pct: float,
              per_material: {material: {mape, mae_inr, directional_accuracy, obs_count}},
              materials_evaluated: int,
              evaluated_at: str,
            }
        """
        t0 = time.perf_counter()
        logger.info("[ModelEvaluator] Starting price forecast evaluation …")

        try:
            import numpy as np
            import pandas as pd

            if not _MATERIAL_CSV.exists():
                return self._eval_error(
                    "price_forecast",
                    f"Material prices CSV not found at {_MATERIAL_CSV}"
                )

            df = pd.read_csv(str(_MATERIAL_CSV), parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)

            materials = sorted(df["material_key"].unique())
            per_material: Dict[str, Dict] = {}
            all_mapes:   List[float] = []
            all_maes:    List[float] = []
            all_dir_acc: List[float] = []

            for material in materials:
                try:
                    subset = df[df["material_key"] == material].sort_values("date")
                    if len(subset) < 18:  # need >12 train + 6 test
                        continue

                    # Hold out last 6 months
                    train = subset.iloc[:-6]
                    test  = subset.iloc[-6:]

                    # Fit Prophet on training data
                    try:
                        from prophet import Prophet
                        prophet_df = train[["date", "price_inr"]].rename(
                            columns={"date": "ds", "price_inr": "y"}
                        )
                        m = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            changepoint_prior_scale=0.05,
                            seasonality_mode="additive",
                        )
                        m.fit(prophet_df)
                        future   = m.make_future_dataframe(periods=6, freq="MS")
                        forecast = m.predict(future)
                        pred_vals = forecast["yhat"].iloc[-6:].values
                    except Exception:
                        # Fallback to linear trend if Prophet fails
                        from numpy.polynomial import polynomial as P
                        x      = np.arange(len(train))
                        y      = train["price_inr"].values
                        coeffs = P.polyfit(x, y, 1)
                        x_test = np.arange(len(train), len(train) + 6)
                        pred_vals = P.polyval(x_test, coeffs)

                    true_vals = test["price_inr"].values

                    # MAPE
                    ape = np.abs((true_vals - pred_vals) / np.maximum(np.abs(true_vals), 1.0))
                    mape = float(np.mean(ape) * 100)

                    # MAE in INR
                    mae_inr = float(np.mean(np.abs(true_vals - pred_vals)))

                    # Directional accuracy
                    true_direction = np.sign(np.diff(true_vals))
                    pred_direction = np.sign(np.diff(pred_vals))
                    if len(true_direction) > 0:
                        dir_acc = float(np.mean(true_direction == pred_direction) * 100)
                    else:
                        dir_acc = 50.0

                    per_material[material] = {
                        "mape_pct":              round(mape, 2),
                        "mae_inr":               round(mae_inr, 2),
                        "directional_accuracy_pct": round(dir_acc, 2),
                        "obs_count":             len(test),
                        "train_rows":            len(train),
                    }
                    all_mapes.append(mape)
                    all_maes.append(mae_inr)
                    all_dir_acc.append(dir_acc)

                except Exception as e:
                    logger.debug(f"[ModelEvaluator] Price eval failed for {material}: {e}")

            if not all_mapes:
                return self._eval_error(
                    "price_forecast",
                    "No materials had enough data for evaluation (need ≥18 monthly rows each)."
                )

            elapsed = round(time.perf_counter() - t0, 2)
            result_dict = {
                "overall_mape_pct":           round(float(sum(all_mapes) / len(all_mapes)), 2),
                "overall_mae_inr":            round(float(sum(all_maes) / len(all_maes)), 2),
                "directional_accuracy_pct":   round(float(sum(all_dir_acc) / len(all_dir_acc)), 2),
                "per_material":               per_material,
                "materials_evaluated":        len(per_material),
                "eval_time_seconds":          elapsed,
            }

            _save_json(_PRICE_EVAL_PATH, result_dict)
            logger.info(
                f"[ModelEvaluator] Price eval done in {elapsed}s: "
                f"MAPE={result_dict['overall_mape_pct']:.1f}% "
                f"dir_acc={result_dict['directional_accuracy_pct']:.1f}%"
            )
            return result_dict

        except Exception as e:
            logger.error(f"[ModelEvaluator] evaluate_price_forecast failed: {e}", exc_info=True)
            return self._eval_error("price_forecast", str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # Combined metrics
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Return combined metrics from all three evaluators.
        Uses 24-hour JSON cache — re-evaluates only when stale.

        This is the primary method called by GET /api/v1/health/metrics.

        Returns:
            {
              style_classifier: {...},
              roi_model: {...},
              price_forecast: {...},
              overall_health_score: float,
              generated_at: str,
            }
        """
        style_result = (
            _load_json(_STYLE_EVAL_PATH) if _is_cache_fresh(_STYLE_EVAL_PATH)
            else self.evaluate_style_classifier()
        )
        roi_result = (
            _load_json(_ROI_EVAL_PATH) if _is_cache_fresh(_ROI_EVAL_PATH)
            else self.evaluate_roi_model()
        )
        price_result = (
            _load_json(_PRICE_EVAL_PATH) if _is_cache_fresh(_PRICE_EVAL_PATH)
            else self.evaluate_price_forecast()
        )

        # Compute overall health score (0-100)
        scores: List[float] = []
        style_acc = style_result.get("accuracy")
        if style_acc is not None:
            scores.append(float(style_acc) * 100)

        roi_r2 = roi_result.get("r2")
        if roi_r2 is not None:
            scores.append(max(0.0, float(roi_r2)) * 100)

        price_dir = price_result.get("directional_accuracy_pct")
        if price_dir is not None:
            scores.append(float(price_dir))

        overall_score = round(sum(scores) / max(len(scores), 1), 1) if scores else 0.0

        return {
            "style_classifier":   style_result  or {"error": "not evaluated"},
            "roi_model":          roi_result     or {"error": "not evaluated"},
            "price_forecast":     price_result   or {"error": "not evaluated"},
            "overall_health_score": overall_score,
            "cache_ttl_hours":    _CACHE_TTL_HOURS,
            "generated_at":       datetime.now(tz=timezone.utc).isoformat(),
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Force re-evaluation of all three models, regardless of cache.
        Acquires the global evaluation lock to prevent concurrent runs.

        Returns:
            Combined metrics dict (same schema as get_all_metrics()).
        """
        with _EVAL_LOCK:
            logger.info("[ModelEvaluator] Running full evaluation (all models) …")
            style_result = self.evaluate_style_classifier()
            roi_result   = self.evaluate_roi_model()
            price_result = self.evaluate_price_forecast()

            scores: List[float] = []
            if style_result.get("accuracy"):
                scores.append(float(style_result["accuracy"]) * 100)
            if roi_result.get("r2") is not None:
                scores.append(max(0.0, float(roi_result["r2"])) * 100)
            if price_result.get("directional_accuracy_pct"):
                scores.append(float(price_result["directional_accuracy_pct"]))

            overall = round(sum(scores) / max(len(scores), 1), 1) if scores else 0.0
            logger.info(f"[ModelEvaluator] Full evaluation complete. Score: {overall}/100")

            return {
                "style_classifier":     style_result,
                "roi_model":            roi_result,
                "price_forecast":       price_result,
                "overall_health_score": overall,
                "generated_at":         datetime.now(tz=timezone.utc).isoformat(),
            }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _eval_error(model: str, error: str) -> Dict[str, Any]:
        """Return a structured error result that never causes the caller to crash."""
        logger.warning(f"[ModelEvaluator] {model}: {error}")
        return {
            "error":           error,
            "partial_results": {},
            "model":           model,
            "evaluated_at":    datetime.now(tz=timezone.utc).isoformat(),
        }