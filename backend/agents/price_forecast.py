"""
ARKEN — Price Forecast Agent v5.0
====================================
v5.0 Changes over v4.0 (REAL ML ENSEMBLE):

  1. ProphetForecastEngine:
       - Facebook Prophet time-series model per material × city combination.
       - yearly_seasonality=True, custom Indian construction seasonality added.
       - Confidence intervals from Prophet uncertainty (yhat_lower / yhat_upper).
       - Models cached in memory after first fit; warm-up on __init__.
       - Graceful fallback to linear-real if Prophet not installed.

  2. XGBoostPriceRegressor:
       - XGBoost regression on lag/rolling/calendar features.
       - Trained once at startup; model persisted to backend/ml/weights/price_xgb.joblib.
       - Loaded from disk on startup if file exists; retrained otherwise.
       - Logs MAE + MAPE to logger.info.

  3. Ensemble _fit_and_forecast():
       - Final forecast = Prophet 60% + XGBoost 40% (configurable).
       - data_quality = "real_ml_ensemble" | "real_prophet_only" |
                        "real_xgb_only" | "linear_real" | "estimated_seed_based"
       - buy_now_signal based on real ML prediction.
       - confidence from Prophet interval width.

  4. New output keys (additive only):
       - "ml_model_used": str
       - "training_data_rows": int
       - "forecast_method": str

All public API surface UNCHANGED:
  PriceForecastAgent.forecast_all()
  PriceForecastAgent.forecast_material()
  PriceForecastAgent.forecast_for_project()
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ── Material price CSV path ────────────────────────────────────────────────────
_MATERIAL_CSV_PATH = Path(
    "/app/data/datasets/material_prices/india_material_prices_historical.csv"
)

# ── ML model persistence paths ────────────────────────────────────────────────
_XGB_MODEL_PATH    = Path("/app/ml/weights/price_xgb.joblib")
_PROPHET_DIR       = Path("/app/ml/weights/prophet_models")

# ── Seed verification date (baseline for last_verified_date) ──────────────────
_SEED_LAST_VERIFIED = "2026-01-01"

# ── Ensemble weights ──────────────────────────────────────────────────────────
_PROPHET_WEIGHT  = 0.60
_XGB_WEIGHT      = 0.40

# ── Thread pool for blocking model operations in async context ─────────────────
_executor = ThreadPoolExecutor(max_workers=4)

# ── UPDATED: Real Q1 2026 Indian material prices ──────────────────────────────
SEED_DATA: Dict[str, Dict] = {
    "cement_opc53_per_bag_50kg": {
        "unit": "per 50kg bag", "current_inr": 400, "yearly_avg_inr": 390,
        "seasonality": "additive", "trend_slope": 0.06, "display_name": "Cement OPC 53 Grade",
        "category": "structural",
        "relevance": ["structural", "flooring", "walls", "kitchen", "bathroom", "full_home"],
    },
    "steel_tmt_fe500_per_kg": {
        "unit": "per kg (Fe500 grade)", "current_inr": 65, "yearly_avg_inr": 62,
        "seasonality": "multiplicative", "trend_slope": 0.05, "display_name": "Steel TMT Fe500",
        "category": "structural", "relevance": ["structural", "full_home"],
    },
    "teak_wood_per_cft": {
        "unit": "per cubic foot (Grade A)", "current_inr": 3000, "yearly_avg_inr": 2850,
        "seasonality": "additive", "trend_slope": 0.04, "display_name": "Teak Wood Grade A",
        "category": "woodwork",
        "relevance": ["kitchen", "bedroom", "living_room", "study", "full_home"],
    },
    "kajaria_tiles_per_sqft": {
        "unit": "per sqft (600x600 glazed)", "current_inr": 90, "yearly_avg_inr": 87,
        "seasonality": "additive", "trend_slope": 0.03, "display_name": "Kajaria Vitrified Tiles",
        "category": "flooring",
        "relevance": ["flooring", "kitchen", "bathroom", "living_room", "full_home"],
    },
    "copper_wire_per_kg": {
        "unit": "per kg (electrical grade)", "current_inr": 850, "yearly_avg_inr": 800,
        "seasonality": "multiplicative", "trend_slope": 0.10, "display_name": "Copper Electrical Wire",
        "category": "electrical",
        "relevance": ["electrical", "kitchen", "bathroom", "full_home", "smart_home"],
    },
    "sand_river_per_brass": {
        "unit": "per brass (100 cft)", "current_inr": 3700, "yearly_avg_inr": 3400,
        "seasonality": "additive", "trend_slope": 0.09, "display_name": "River Sand",
        "category": "raw_material",
        "relevance": ["structural", "flooring", "walls", "full_home"],
    },
    "bricks_per_1000": {
        "unit": "per 1000 units", "current_inr": 9000, "yearly_avg_inr": 8200,
        "seasonality": "additive", "trend_slope": 0.05, "display_name": "Red Bricks",
        "category": "structural", "relevance": ["structural", "full_home"],
    },
    "granite_per_sqft": {
        "unit": "per sqft (Black Galaxy)", "current_inr": 195, "yearly_avg_inr": 180,
        "seasonality": "additive", "trend_slope": 0.04, "display_name": "Granite (Black Galaxy)",
        "category": "flooring", "relevance": ["flooring", "kitchen", "bathroom", "full_home"],
    },
    "asian_paints_premium_per_litre": {
        "unit": "per litre (Royale Aspira)", "current_inr": 350, "yearly_avg_inr": 330,
        "seasonality": "additive", "trend_slope": 0.04,
        "display_name": "Asian Paints Premium Emulsion",
        "category": "paint", "relevance": ["walls", "bedroom", "living_room", "kitchen", "full_home"],
    },
    "pvc_upvc_window_per_sqft": {
        "unit": "per sqft (UPVC frame + glass)", "current_inr": 950, "yearly_avg_inr": 900,
        "seasonality": "additive", "trend_slope": 0.05, "display_name": "UPVC Windows",
        "category": "fixtures", "relevance": ["bedroom", "living_room", "full_home"],
    },
    "modular_kitchen_per_sqft": {
        "unit": "per sqft (mid-range laminate)", "current_inr": 1350, "yearly_avg_inr": 1250,
        "seasonality": "additive", "trend_slope": 0.07, "display_name": "Modular Kitchen (Laminate)",
        "category": "kitchen", "relevance": ["kitchen", "full_home"],
    },
    "bathroom_sanitary_set": {
        "unit": "per set (Hindware/Cera standard)", "current_inr": 21000, "yearly_avg_inr": 19500,
        "seasonality": "additive", "trend_slope": 0.05, "display_name": "Bathroom Sanitary Set",
        "category": "plumbing", "relevance": ["bathroom", "full_home"],
    },
}

CITY_COST_MULTIPLIER: Dict[str, float] = {
    "Mumbai": 1.25, "Delhi NCR": 1.20, "Bangalore": 1.15,
    "Hyderabad": 1.00, "Pune": 1.05, "Chennai": 1.05,
    "Kolkata": 0.95, "Ahmedabad": 0.92, "Surat": 0.90,
    "Jaipur": 0.88, "Lucknow": 0.85, "Chandigarh": 0.95,
    "Nagpur": 0.87, "Indore": 0.86, "Bhopal": 0.84,
}

TREND_DRIVERS: Dict[str, Dict[str, str]] = {
    "cement_opc53_per_bag_50kg": {
        "up": "Cement prices rising due to increased infrastructure demand, coal cost escalation.",
        "stable": "Cement prices stable — balanced demand from housing sector.",
        "down": "Cement softening on oversupply from new plant commissions.",
    },
    "steel_tmt_fe500_per_kg": {
        "up": "Steel trending upward on global iron ore tightening and strong domestic demand.",
        "stable": "Steel prices consolidating — global markets stable.",
        "down": "Steel retreating as Chinese export surplus enters Indian markets.",
    },
    "copper_wire_per_kg": {
        "up": "Copper elevated on global EV/green energy demand surge and MCX inventory drawdowns.",
        "stable": "Copper stable — MCX inventories balanced.",
        "down": "Copper correcting on weaker Chinese manufacturing PMI.",
    },
    "kajaria_tiles_per_sqft": {
        "up": "Tile prices inching up — gas cost pressures on kilns.",
        "stable": "Tiles flat — competitive market with multiple domestic producers.",
        "down": "Tiles easing on aggressive competition and import pressure.",
    },
    "sand_river_per_brass": {
        "up": "Sand prices surging on mining ban enforcement and monsoon logistics disruption.",
        "stable": "Sand supply normalised — mining permits regularised post-monsoon.",
        "down": "M-sand adoption increasing, reducing pressure on river sand pricing.",
    },
    "asian_paints_premium_per_litre": {
        "up": "Asian Paints raised prices 2-3% due to crude oil derivative cost increase.",
        "stable": "Paint prices stable — AP maintaining market share over competitors.",
        "down": "Paint prices easing on lower crude oil input costs.",
    },
    "modular_kitchen_per_sqft": {
        "up": "Modular kitchen demand rising post-pandemic; plywood and hardware costs up.",
        "stable": "Kitchen segment stable — strong competition keeps prices in check.",
        "down": "Kitchen prices softening — new entrants increasing supply.",
    },
    "default": {
        "up": "Price trending upward on material demand increase.",
        "stable": "Price stable within seasonal norms.",
        "down": "Price softening on market correction.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# RealPriceDataManager  (unchanged public API from v4.0)
# ─────────────────────────────────────────────────────────────────────────────

class RealPriceDataManager:
    """
    Manages loading of real historical material price data.

    When india_material_prices_historical.csv exists:
      - Loads the CSV and provides actual time-series for each material.
      - data_quality → "historical"

    When the CSV does not exist:
      - Falls back to SEED_DATA-based synthetic history generation.
      - Marks all outputs with data_quality="estimated_seed_based".
    """

    _instance: Optional["RealPriceDataManager"] = None
    _real_data: Optional[pd.DataFrame] = None
    _data_quality: str = "estimated_seed_based"

    def __new__(cls) -> "RealPriceDataManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        if not _MATERIAL_CSV_PATH.exists():
            logger.warning(
                "[RealPriceDataManager] india_material_prices_historical.csv not found at "
                f"{_MATERIAL_CSV_PATH}. "
                "All price forecasts will use estimated seed-based data. "
                "See backend/data/datasets/material_prices/README.md."
            )
            self._real_data    = None
            self._data_quality = "estimated_seed_based"
            return

        try:
            df = pd.read_csv(str(_MATERIAL_CSV_PATH), parse_dates=["date"])
            required_cols = {"date", "material_key", "price_inr"}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                logger.warning(
                    f"[RealPriceDataManager] CSV missing required columns: {missing}. "
                    "Falling back to seed-based estimates."
                )
                self._real_data    = None
                self._data_quality = "estimated_seed_based"
                return

            df = df.sort_values("date").reset_index(drop=True)
            self._real_data    = df
            self._data_quality = "historical"
            logger.info(
                f"[RealPriceDataManager] Loaded {len(df):,} rows of real price history "
                f"from {_MATERIAL_CSV_PATH}. "
                f"Materials covered: {df['material_key'].nunique()}"
            )
        except Exception as e:
            logger.error(f"[RealPriceDataManager] Failed to load CSV: {e}. Using seed-based data.")
            self._real_data    = None
            self._data_quality = "estimated_seed_based"

    @property
    def has_real_data(self) -> bool:
        return self._real_data is not None

    @property
    def data_quality(self) -> str:
        return self._data_quality

    def get_series(
        self, material_key: str, city: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        if self._real_data is None:
            return None
        df = self._real_data[self._real_data["material_key"] == material_key].copy()
        if df.empty:
            return None
        if city and "city" in df.columns:
            city_df = df[df["city"] == city]
            if not city_df.empty:
                df = city_df
        return df.sort_values("date")[["date", "price_inr"]].reset_index(drop=True)

    def get_all_series_for_material(self, material_key: str) -> Optional[pd.DataFrame]:
        """Return city-averaged monthly series for a material (for XGB training)."""
        if self._real_data is None:
            return None
        df = self._real_data[self._real_data["material_key"] == material_key].copy()
        if df.empty:
            return None
        df = df.sort_values("date")
        monthly = (
            df.groupby(["date", "city"])["price_inr"]
            .mean()
            .reset_index()
        )
        return monthly


# Module-level singleton
_price_manager: Optional[RealPriceDataManager] = None


def _get_price_manager() -> RealPriceDataManager:
    global _price_manager
    if _price_manager is None:
        _price_manager = RealPriceDataManager()
    return _price_manager


# ─────────────────────────────────────────────────────────────────────────────
# ProphetForecastEngine
# ─────────────────────────────────────────────────────────────────────────────

class ProphetForecastEngine:
    """
    Facebook Prophet-based time-series forecasting engine.

    - One model per (material_key, city) combination.
    - Models are fit once and cached in-memory.
    - Uses yearly seasonality + custom Indian construction seasonality
      (peak Oct–Mar, slow Jun–Sep monsoon).
    - Falls back to None if Prophet not installed.
    """

    _PROPHET_AVAILABLE: Optional[bool] = None

    def __init__(self):
        self._models: Dict[str, object] = {}   # key: "{material_key}|{city}"
        self._check_prophet()

    @classmethod
    def _check_prophet(cls) -> bool:
        if cls._PROPHET_AVAILABLE is None:
            try:
                import prophet  # noqa: F401
                # Eagerly verify that the Stan backend (CmdStan) is actually
                # usable. Prophet imports fine even when CmdStan is not installed,
                # but model.fit() raises AttributeError: 'Prophet' object has no
                # attribute 'stan_backend'. We catch this here so the warning is
                # emitted once at startup instead of on every forecast call.
                from prophet import Prophet as _P
                _probe = _P(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                import pandas as _pd
                import numpy as _np
                _dates = _pd.date_range("2022-01-01", periods=30, freq="MS")
                _probe_df = _pd.DataFrame({
                    "ds": _dates,
                    "y": _np.linspace(100, 130, 30),
                })
                _probe.fit(_probe_df)           # will raise if CmdStan missing
                cls._PROPHET_AVAILABLE = True
                logger.info("[ProphetForecastEngine] Facebook Prophet + CmdStan verified and available.")
            except AttributeError as e:
                if "stan_backend" in str(e) or "stan" in str(e).lower():
                    cls._PROPHET_AVAILABLE = False
                    logger.warning(
                        "[ProphetForecastEngine] Prophet stan_backend (CmdStan) missing. "
                        "Prophet disabled for this session — using XGBoost fallback. "
                        "To fix: add RUN python -c \"import cmdstanpy; cmdstanpy.install_cmdstan()\" "
                        "to your Dockerfile after pip install, then rebuild."
                    )
                else:
                    cls._PROPHET_AVAILABLE = False
                    logger.warning(f"[ProphetForecastEngine] Prophet check failed: {e}. Using XGBoost fallback.")
            except Exception as e:
                cls._PROPHET_AVAILABLE = False
                logger.warning(
                    f"[ProphetForecastEngine] Prophet not available ({type(e).__name__}: {e}). "
                    "Using XGBoost fallback."
                )
        return cls._PROPHET_AVAILABLE

    def _cache_key(self, material_key: str, city: str) -> str:
        return f"{material_key}|{city}"

    def _build_prophet_df(self, series: pd.DataFrame) -> pd.DataFrame:
        """Convert date/price_inr series to Prophet's ds/y format."""
        df = series.rename(columns={"date": "ds", "price_inr": "y"}).copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)

        # Add inflation proxy: YoY % change
        df["inflation_proxy"] = df["y"].pct_change(periods=12).fillna(0) * 100
        return df

    def _fit_single(self, material_key: str, city: str, series: pd.DataFrame) -> bool:
        """
        Fit a Prophet model for one material×city. Returns True on success.

        stan_backend fix: Prophet ≥1.1.4 uses cmdstanpy. If the stan backend is
        not initialised (AttributeError: 'Prophet' object has no attribute
        'stan_backend'), we catch it specifically, mark Prophet unavailable for
        this process, and let callers fall back to XGB / linear-real.
        Run  python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"  to fix
        permanently.
        """
        if not self._check_prophet():
            return False

        try:
            from prophet import Prophet

            prophet_df = self._build_prophet_df(series)
            if len(prophet_df) < 24:
                logger.debug(
                    f"[ProphetForecastEngine] Skipping {material_key}|{city}: "
                    f"only {len(prophet_df)} rows (need ≥24)."
                )
                return False

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

            if prophet_df["inflation_proxy"].std() > 0.01:
                model.add_regressor("inflation_proxy")
                model.fit(prophet_df[["ds", "y", "inflation_proxy"]])
            else:
                model.fit(prophet_df[["ds", "y"]])

            ckey = self._cache_key(material_key, city)
            self._models[ckey] = {
                "model": model,
                "has_regressor": prophet_df["inflation_proxy"].std() > 0.01,
                "last_inflation": float(prophet_df["inflation_proxy"].iloc[-1]),
                "n_rows": len(prophet_df),
            }
            return True

        except AttributeError as e:
            if "stan_backend" in str(e):
                # CmdStan not installed — disable Prophet for this process so
                # callers fall back cleanly to XGBoost / linear-real forecasting.
                ProphetForecastEngine._PROPHET_AVAILABLE = False
                logger.warning(
                    "[ProphetForecastEngine] Prophet stan_backend missing. "
                    "Run: python -m cmdstanpy install_cmdstan to fix permanently. "
                    "Prophet disabled for this session — using XGBoost fallback."
                )
            else:
                logger.warning(
                    f"[ProphetForecastEngine] Fit failed for {material_key}|{city}: {e}"
                )
            return False

        except Exception as e:
            logger.warning(
                f"[ProphetForecastEngine] Fit failed for {material_key}|{city}: {e}"
            )
            return False

    def warm_up(self, pm: RealPriceDataManager) -> int:
        """
        Fit Prophet models for all material × city combinations found in real data.
        Returns number of models successfully fitted.
        """
        if not self._check_prophet() or not pm.has_real_data:
            return 0

        n_fitted = 0
        cities = list(CITY_COST_MULTIPLIER.keys())

        for material_key in SEED_DATA:
            for city in cities:
                series = pm.get_series(material_key, city)
                if series is not None and len(series) >= 24:
                    if self._fit_single(material_key, city, series):
                        n_fitted += 1

        logger.info(
            f"[ProphetForecastEngine] Warm-up complete: {n_fitted} models fitted "
            f"({len(SEED_DATA)} materials × up to {len(cities)} cities)."
        )
        return n_fitted

    def forecast(
        self,
        material_key: str,
        city: str,
        horizon: int = 90,
        series: Optional[pd.DataFrame] = None,
        pm: Optional[RealPriceDataManager] = None,
    ) -> Optional[List[Dict]]:
        """
        Generate a horizon-day daily forecast.
        Returns list of {ds, yhat, yhat_lower, yhat_upper} or None on failure.

        city=None is resolved to "Hyderabad" (national base, multiplier=1.0).
        This prevents 'material|None' cache keys and stan_backend fit errors.
        """
        if not self._check_prophet():
            return None

        # Resolve None city to national base city
        resolved_city = city or "Hyderabad"
        ckey = self._cache_key(material_key, resolved_city)

        # Lazy-fit if not in cache
        if ckey not in self._models:
            src = series if series is not None else (
                pm.get_series(material_key, resolved_city) if pm else None
            )
            if src is None or len(src) < 24:
                return None
            if not self._fit_single(material_key, resolved_city, src):
                return None

        entry = self._models[ckey]
        model = entry["model"]

        try:
            future = model.make_future_dataframe(periods=horizon, freq="D")
            if entry["has_regressor"]:
                # Forward-fill last known inflation value
                future["inflation_proxy"] = entry["last_inflation"]

            forecast_df = model.predict(future)
            tail = forecast_df.tail(horizon).copy()

            records = []
            for _, row in tail.iterrows():
                records.append({
                    "ds":         row["ds"].strftime("%Y-%m-%d"),
                    "yhat":       round(float(row["yhat"]),       2),
                    "yhat_lower": round(float(row["yhat_lower"]), 2),
                    "yhat_upper": round(float(row["yhat_upper"]), 2),
                })
            return records

        except Exception as e:
            logger.warning(
                f"[ProphetForecastEngine] Predict failed for {material_key}|{city}: {e}"
            )
            return None

    def get_n_rows(self, material_key: str, city: str) -> int:
        ckey = self._cache_key(material_key, city)
        entry = self._models.get(ckey)
        return entry["n_rows"] if entry else 0


# ─────────────────────────────────────────────────────────────────────────────
# XGBoostPriceRegressor
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostPriceRegressor:
    """
    XGBoost-based price regressor trained on the full historical CSV.

    Features:
        month_sin, month_cos, year, city_enc, material_enc,
        lag_1m, lag_3m, lag_6m, rolling_mean_3m, rolling_std_3m,
        is_monsoon, trend_index

    The 30/60/90-day predictions are produced by rolling the last known
    price forward using a recursive single-step strategy.
    """

    _XGB_AVAILABLE: Optional[bool] = None

    def __init__(self):
        self._model = None
        self._city_enc: Dict[str, int]     = {}
        self._mat_enc:  Dict[str, int]     = {}
        self._is_fitted: bool              = False
        self._check_xgb()

    @classmethod
    def _check_xgb(cls) -> bool:
        if cls._XGB_AVAILABLE is None:
            try:
                import xgboost  # noqa: F401
                import joblib    # noqa: F401
                import sklearn   # noqa: F401
                cls._XGB_AVAILABLE = True
                logger.info("[XGBoostPriceRegressor] XGBoost + joblib + sklearn available.")
            except ImportError as e:
                cls._XGB_AVAILABLE = False
                logger.warning(
                    f"[XGBoostPriceRegressor] Missing dependency: {e}. "
                    "Install: pip install xgboost joblib scikit-learn. "
                    "XGBoost branch disabled."
                )
        return cls._XGB_AVAILABLE

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from a DataFrame with columns:
        date, material_key, price_inr, city, [city_enc, material_enc]
        """
        df = df.sort_values(["material_key", "city", "date"]).copy()
        df["month"]         = df["date"].dt.month
        df["year"]          = df["date"].dt.year
        df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
        df["is_monsoon"]    = df["month"].isin([6, 7, 8, 9]).astype(int)
        df["trend_index"]   = (df["date"] - df["date"].min()).dt.days / 30.0

        # Lag & rolling features per material × city group
        grp = df.groupby(["material_key", "city"])["price_inr"]
        df["lag_1m"]         = grp.shift(1)
        df["lag_3m"]         = grp.shift(3)
        df["lag_6m"]         = grp.shift(6)
        df["rolling_mean_3m"] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df["rolling_std_3m"]  = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0))

        # Encode city & material
        if self._city_enc:
            df["city_enc"] = df["city"].map(self._city_enc).fillna(0).astype(int)
        else:
            cats = df["city"].astype("category")
            self._city_enc = dict(enumerate(cats.cat.categories))
            self._city_enc = {v: k for k, v in self._city_enc.items()}
            df["city_enc"] = df["city"].map(self._city_enc).fillna(0).astype(int)

        if self._mat_enc:
            df["material_enc"] = df["material_key"].map(self._mat_enc).fillna(0).astype(int)
        else:
            cats = df["material_key"].astype("category")
            self._mat_enc = dict(enumerate(cats.cat.categories))
            self._mat_enc = {v: k for k, v in self._mat_enc.items()}
            df["material_enc"] = df["material_key"].map(self._mat_enc).fillna(0).astype(int)

        return df

    _FEATURE_COLS = [
        "month_sin", "month_cos", "year", "city_enc", "material_enc",
        "lag_1m", "lag_3m", "lag_6m",
        "rolling_mean_3m", "rolling_std_3m",
        "is_monsoon", "trend_index",
    ]

    def load_or_train(self, pm: RealPriceDataManager) -> bool:
        """Load model from disk if exists, otherwise train from real data."""
        if not self._check_xgb():
            return False

        try:
            import joblib
            if _XGB_MODEL_PATH.exists():
                bundle = joblib.load(str(_XGB_MODEL_PATH))
                self._model    = bundle["model"]
                self._city_enc = bundle["city_enc"]
                self._mat_enc  = bundle["mat_enc"]
                self._is_fitted = True
                logger.info(
                    f"[XGBoostPriceRegressor] Loaded model from {_XGB_MODEL_PATH}."
                )
                return True
        except Exception as e:
            logger.warning(
                f"[XGBoostPriceRegressor] Could not load saved model ({e}). Retraining."
            )

        return self._train(pm)

    def _train(self, pm: RealPriceDataManager) -> bool:
        if not self._check_xgb() or not pm.has_real_data:
            return False

        try:
            import xgboost as xgb
            import joblib
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error

            df = pm._real_data.copy()
            if "city" not in df.columns:
                df["city"] = "Hyderabad"

            df = self._build_features(df)
            df = df.dropna(subset=self._FEATURE_COLS + ["price_inr"])

            X = df[self._FEATURE_COLS].values
            y = df["price_inr"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42
            )

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
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = model.predict(X_test)
            mae    = mean_absolute_error(y_test, y_pred)
            # MAPE
            nonzero = y_test != 0
            mape = float(np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100)

            logger.info(
                f"[XGBoostPriceRegressor] Training complete. "
                f"Test MAE: ₹{mae:.2f}  |  MAPE: {mape:.2f}%  "
                f"(train: {len(X_train)} rows, test: {len(X_test)} rows)"
            )

            self._model    = model
            self._is_fitted = True

            # Persist
            _XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": model, "city_enc": self._city_enc, "mat_enc": self._mat_enc},
                str(_XGB_MODEL_PATH),
            )
            logger.info(f"[XGBoostPriceRegressor] Model saved to {_XGB_MODEL_PATH}.")
            return True

        except Exception as e:
            logger.error(f"[XGBoostPriceRegressor] Training failed: {e}", exc_info=True)
            return False

    def predict_horizon(
        self,
        material_key: str,
        city: str,
        series: pd.DataFrame,
        horizon: int = 90,
    ) -> Optional[List[float]]:
        """
        Produce horizon-day recursive single-step price predictions.
        Returns list of [price_d1, ..., price_d{horizon}] or None.
        """
        if not self._is_fitted or self._model is None:
            return None

        try:
            # Resolve None city → Hyderabad (national base, multiplier=1.0)
            resolved_city = city if city else "Hyderabad"
            city_enc = self._city_enc.get(resolved_city, 0)
            mat_enc  = self._mat_enc.get(material_key, 0)

            prices = list(series["price_inr"].values[-12:])  # keep last 12 months
            dates  = list(series["date"].values[-12:])
            last_date = pd.Timestamp(dates[-1])
            trend_base = (last_date - pd.Timestamp("2020-01-01")).days / 30.0

            predictions = []
            for i in range(1, horizon + 1):
                future_date = last_date + pd.Timedelta(days=i)
                month = future_date.month
                year  = future_date.year

                lag_1m = float(prices[-1])        if len(prices) >= 1  else prices[-1]
                lag_3m = float(prices[-3])        if len(prices) >= 3  else prices[0]
                lag_6m = float(prices[-6])        if len(prices) >= 6  else prices[0]
                rm3    = float(np.mean(prices[-3:])) if len(prices) >= 3 else float(np.mean(prices))
                rs3    = float(np.std(prices[-3:]))  if len(prices) >= 3 else 0.0

                feat = np.array([[
                    np.sin(2 * np.pi * month / 12),
                    np.cos(2 * np.pi * month / 12),
                    year,
                    city_enc,
                    mat_enc,
                    lag_1m, lag_3m, lag_6m,
                    rm3, rs3,
                    int(month in [6, 7, 8, 9]),
                    trend_base + i / 30.0,
                ]])

                pred = float(self._model.predict(feat)[0])
                predictions.append(pred)
                prices.append(pred)

            return predictions

        except Exception as e:
            logger.warning(
                f"[XGBoostPriceRegressor] Predict failed for {material_key}|{city}: {e}"
            )
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Module-level engine singletons (initialised lazily then warmed up)
# ─────────────────────────────────────────────────────────────────────────────

_prophet_engine:  Optional[ProphetForecastEngine]  = None
_xgb_regressor:   Optional[XGBoostPriceRegressor]  = None


def _get_prophet_engine() -> ProphetForecastEngine:
    global _prophet_engine
    if _prophet_engine is None:
        _prophet_engine = ProphetForecastEngine()
    return _prophet_engine


def _get_xgb_regressor() -> XGBoostPriceRegressor:
    global _xgb_regressor
    if _xgb_regressor is None:
        _xgb_regressor = XGBoostPriceRegressor()
    return _xgb_regressor


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (kept from v4.0)
# ─────────────────────────────────────────────────────────────────────────────

def _month_factor(month: int) -> float:
    return {1: 1.02, 2: 1.03, 3: 1.05, 4: 1.01, 5: 0.99, 6: 0.96,
            7: 0.94, 8: 0.93, 9: 0.97, 10: 1.04, 11: 1.06, 12: 1.03}.get(month, 1.0)


# Rate-limited warning tracker — prevents hundreds of identical log lines per request
_clamp_warned: set = set()


def _validate_forecast(
    yhat: float, base: float, material_key: str,
    current_inr: float = 0.0, city_multiplier: float = 1.0,
) -> float:
    """
    Clamp yhat to [base*0.25, base*5.00].

    Bounds are deliberately wide because:
      - XGB is trained on city-specific CSV data (already city-scaled).
      - Prophet models cover the full 2020-2026 range (wide natural spread).
      - Tight bounds caused hundreds of spurious clamping warnings per request.

    Warnings fire ONCE per material key per process to avoid log spam.
    Delta must be > 25% of ref_price to trigger a warning at all.
    """
    global _clamp_warned
    clamped = max(base * 0.25, min(base * 5.00, yhat))
    if clamped != yhat:
        ref_price = (current_inr * city_multiplier) if current_inr > 0 else base
        if (
            ref_price > 0
            and abs(clamped - yhat) / ref_price > 0.25
            and material_key not in _clamp_warned
        ):
            logger.warning(
                f"[PriceForecast] {material_key}: prediction {yhat:.2f} "
                f"clamped to {clamped:.2f} — outside expected range. "
                f"(Warning fires once per material per process.)"
            )
            _clamp_warned.add(material_key)
    return clamped


def _build_trend_narrative(
    material_key, trend, pct_change, volatility_label, city=None, city_multiplier=1.0,
):
    drivers        = TREND_DRIVERS.get(material_key, TREND_DRIVERS["default"])
    seed           = SEED_DATA[material_key]
    base_narrative = drivers.get(trend, drivers.get("stable", ""))
    direction      = "rise" if pct_change > 0 else "fall" if pct_change < 0 else "remain stable"
    magnitude      = abs(pct_change)
    impact_word    = (
        "marginally"    if magnitude < 1 else
        "moderately"    if magnitude < 4 else
        "significantly" if magnitude < 8 else
        "substantially"
    )
    narrative = (
        f"{seed['display_name']} prices expected to {direction} {impact_word} by "
        f"{magnitude:.1f}% over 90 days. {base_narrative} Volatility: {volatility_label}."
    )
    if city and city_multiplier != 1.0:
        city_note = "above" if city_multiplier > 1.0 else "below"
        narrative += (
            f" {city} local pricing is {abs((city_multiplier - 1.0) * 100):.0f}% "
            f"{city_note} national average due to logistics and demand factors."
        )
    return narrative


def _build_confidence_note(
    material_key: str,
    data_quality: str,
    volatility_label: str,
    confidence: float,
    has_real_data: bool,
    ml_model_used: str = "seed_fallback",
) -> str:
    seed = SEED_DATA.get(material_key, {})
    name = seed.get("display_name", material_key)

    if has_real_data and data_quality not in ("estimated_seed_based",):
        model_desc = {
            "prophet+xgboost": "Facebook Prophet + XGBoost ensemble",
            "prophet":         "Facebook Prophet time-series model",
            "xgboost":         "XGBoost regression model",
            "linear_real":     "linear trend fit on real historical data",
        }.get(ml_model_used, ml_model_used)
        return (
            f"{name} forecast produced by {model_desc} trained on verified Indian "
            f"market price data. "
            f"Model confidence: {confidence * 100:.0f}%. "
            f"Volatility: {volatility_label}. "
            "This is the most reliable forecast type available in ARKEN."
        )
    else:
        return (
            f"{name} forecast is based on seed prices last verified {_SEED_LAST_VERIFIED}, "
            f"extrapolated using known seasonal patterns and trend slopes. "
            f"Model confidence: {confidence * 100:.0f}%. "
            f"Volatility: {volatility_label}. "
            "For higher accuracy, supply real historical price data — see "
            "backend/data/datasets/material_prices/README.md."
        )


def _compute_budget_impact(material_key, pct_change_90d, area_sqft, room_type):
    seed = SEED_DATA[material_key]
    if area_sqft is None or area_sqft <= 0:
        return {"estimated_impact_inr": 0, "action": "N/A", "urgency": "low"}
    qty_estimate = 0.0
    category = seed.get("category", "")
    if category == "flooring" and material_key == "kajaria_tiles_per_sqft":
        qty_estimate = area_sqft * 1.10
    elif category == "paint" and "per_litre" in material_key:
        qty_estimate = area_sqft * 0.12
    elif category == "structural" and "cement" in material_key:
        qty_estimate = max(10, area_sqft * 0.08)
    elif category == "flooring" and "granite" in material_key:
        qty_estimate = area_sqft * 0.50
    elif category == "kitchen" and room_type == "kitchen":
        qty_estimate = max(5, area_sqft * 0.30)
    elif category == "plumbing" and room_type in ("bathroom", "full_home"):
        qty_estimate = 1.0
    elif category == "electrical":
        qty_estimate = area_sqft * 0.15
    if qty_estimate <= 0:
        return {"estimated_impact_inr": 0, "action": "monitor", "urgency": "low"}
    current_spend = seed["current_inr"] * qty_estimate
    future_spend  = current_spend * (1 + pct_change_90d / 100)
    delta_inr     = int(future_spend - current_spend)
    if abs(pct_change_90d) < 2:
        urgency, action = "low", "No action required — price movement minimal."
    elif pct_change_90d > 6:
        urgency, action = "high", "Procure now or lock in rates to avoid cost overrun."
    elif pct_change_90d > 2:
        urgency, action = "medium", "Consider early procurement or advance rate agreement."
    else:
        urgency, action = "low", "Watch market — prices softening, defer if possible."
    return {
        "estimated_qty":        round(qty_estimate, 1),
        "current_spend_inr":    int(current_spend),
        "projected_spend_inr":  int(future_spend),
        "delta_inr":            delta_inr,
        "action":               action,
        "urgency":              urgency,
    }


def _linear_real_forecast(
    series: pd.DataFrame,
    horizon: int,
    material_key: str,
    city_multiplier: float,
) -> Tuple[List[Dict], float, float]:
    """Linear trend on real historical data. Same as v4 _fit_real_data_forecast."""
    seed           = SEED_DATA[material_key]
    prices         = series["price_inr"].values.astype(float)
    adjusted_prices = prices * city_multiplier
    adjusted_base   = seed["current_inr"] * city_multiplier
    x               = np.arange(len(adjusted_prices), dtype=float)
    adj_coeffs      = np.polyfit(x, adjusted_prices, 1)
    residuals       = adjusted_prices - np.polyval(adj_coeffs, x)
    sigma           = max(residuals.std(), adjusted_base * 0.01)
    current         = float(prices[-1]) * city_multiplier
    last_date       = series["date"].iloc[-1]

    records = []
    for i in range(horizon):
        future_idx = len(prices) + i
        dt         = last_date + timedelta(days=i + 1)
        yhat_raw   = np.polyval(adj_coeffs, future_idx) * _month_factor(dt.month)
        yhat       = _validate_forecast(yhat_raw, adjusted_base, material_key,
                                        seed["current_inr"], city_multiplier)
        lower      = _validate_forecast(yhat - 1.96 * sigma, adjusted_base * 0.40,
                                        material_key, seed["current_inr"], city_multiplier)
        upper      = _validate_forecast(yhat + 1.96 * sigma, adjusted_base * 3.0,
                                        material_key, seed["current_inr"], city_multiplier)
        records.append({
            "ds":         dt.strftime("%Y-%m-%d"),
            "yhat":       round(float(yhat),  2),
            "yhat_lower": round(float(lower), 2),
            "yhat_upper": round(float(upper), 2),
        })
    return records, current, sigma


def _seed_fallback_forecast(
    material_key: str,
    horizon: int,
    city: Optional[str],
    city_multiplier: float,
) -> Tuple[List[Dict], float, str]:
    """Original seed-based synthetic forecast. Returns (records, current_price, data_quality)."""
    seed          = SEED_DATA[material_key]
    base          = seed["current_inr"]
    adjusted_base = base * city_multiplier
    slope         = seed["trend_slope"] / 365
    days_history  = 730

    np.random.seed(abs(hash(material_key)) % (2 ** 31))
    start          = datetime(2023, 1, 1)
    history_dates  = [start + timedelta(days=i) for i in range(days_history)]
    history_prices = []
    for i, dt in enumerate(history_dates):
        trend  = adjusted_base * (1 + slope * i)
        mf     = _month_factor(dt.month)
        noise  = np.random.normal(0, adjusted_base * 0.015)
        raw    = max(trend * mf + noise, adjusted_base * 0.7)
        history_prices.append(_validate_forecast(raw, adjusted_base, material_key,
                                                   base, city_multiplier))
    x      = np.arange(days_history, dtype=float)
    y      = np.array(history_prices)
    coeffs = np.polyfit(x, y, 1)
    sigma  = max((y - np.polyval(coeffs, x)).std(), adjusted_base * 0.01)

    records = []
    for i in range(days_history, days_history + horizon):
        dt       = start + timedelta(days=i)
        yhat_raw = np.polyval(coeffs, i) * _month_factor(dt.month)
        yhat     = _validate_forecast(yhat_raw, adjusted_base, material_key,
                                      base, city_multiplier)
        lower    = _validate_forecast(yhat - 1.96 * sigma, adjusted_base * 0.40,
                                      material_key, base, city_multiplier)
        upper    = _validate_forecast(yhat + 1.96 * sigma, adjusted_base * 3.0,
                                      material_key, base, city_multiplier)
        records.append({
            "ds":         dt.strftime("%Y-%m-%d"),
            "yhat":       round(float(yhat),  2),
            "yhat_lower": round(float(lower), 2),
            "yhat_upper": round(float(upper), 2),
        })

    current = base * city_multiplier
    return records, current, "estimated_seed_based"


# ─────────────────────────────────────────────────────────────────────────────
# Core forecast orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _load_cv_mape(material_key: str, city: str) -> Optional[float]:
    """
    Load walk-forward CV MAPE for this material+city combo if available.
    Returns mean_mape as float (e.g. 4.2 for 4.2%), or None if CV not run yet.
    """
    cv_path = _PROPHET_DIR.parent / "prophet_cv_report.json"
    if not cv_path.exists():
        return None
    try:
        import json as _json
        with open(str(cv_path)) as f:
            cv_data = _json.load(f)
        key   = f"{material_key}|{city}"
        entry = cv_data.get(key)
        if entry and entry.get("mean_mape") is not None:
            return float(entry["mean_mape"])
    except Exception:
        pass
    return None


def _fit_and_forecast(
    material_key, horizon, city=None, area_sqft=None, room_type=None,
):
    """
    Orchestrates forecast from best available model.

    Priority:
      1. ProphetForecastEngine + XGBoostPriceRegressor ensemble (real data)
      2. Prophet only (real data, XGB unavailable)
      3. XGBoost only  (real data, Prophet unavailable)
      4. Linear-real   (real data, both ML libs absent)
      5. Seed fallback  (no CSV)
    """
    seed            = SEED_DATA[material_key]
    city_multiplier = CITY_COST_MULTIPLIER.get(city or "", 1.0)
    adjusted_base   = seed["current_inr"] * city_multiplier
    pm              = _get_price_manager()
    real_series     = pm.get_series(material_key, city)
    n_rows          = len(real_series) if real_series is not None else 0

    prophet  = _get_prophet_engine()
    xgb_reg  = _get_xgb_regressor()

    # ── 1. Seed fallback if no CSV ─────────────────────────────────────────────
    if real_series is None or n_rows < 24:
        forecast_records, current, data_quality = _seed_fallback_forecast(
            material_key, horizon, city, city_multiplier
        )
        ml_model_used    = "seed_fallback"
        forecast_method  = "Synthetic seed-based linear extrapolation (no real data available)"
        training_rows    = 0
        sigma_for_conf   = adjusted_base * 0.05

    else:
        current = float(real_series["price_inr"].iloc[-1]) * city_multiplier

        # ── Get Prophet forecast ───────────────────────────────────────────────
        prophet_records = prophet.forecast(material_key, city, horizon, real_series, pm)

        # ── Get XGB predictions (flat list of horizon prices) ──────────────────
        xgb_prices: Optional[List[float]] = None
        if xgb_reg._is_fitted:
            xgb_prices = xgb_reg.predict_horizon(material_key, city, real_series, horizon)

        # ── Build ensemble forecast_records ───────────────────────────────────
        if prophet_records is not None and xgb_prices is not None:
            # Weighted ensemble.
            # Both prophet_records["yhat"] and xgb_p are already in city-scale
            # (Prophet trained on city-specific series; XGB trained on city-specific CSV).
            # Blend directly — NO additional city_multiplier applied here.
            forecast_records = []
            for i, (p_rec, xgb_p) in enumerate(zip(prophet_records, xgb_prices)):
                blended_yhat  = _PROPHET_WEIGHT * p_rec["yhat"]  + _XGB_WEIGHT * xgb_p
                blended_lower = p_rec["yhat_lower"]
                blended_upper = p_rec["yhat_upper"]
                # Shift CI proportionally with ensemble adjustment
                delta         = blended_yhat - p_rec["yhat"]
                blended_lower = _validate_forecast(blended_lower + delta,
                                                   adjusted_base * 0.40, material_key,
                                                   seed["current_inr"], city_multiplier)
                blended_upper = _validate_forecast(blended_upper + delta,
                                                   adjusted_base * 3.0, material_key,
                                                   seed["current_inr"], city_multiplier)
                blended_yhat  = _validate_forecast(blended_yhat, adjusted_base,
                                                   material_key, seed["current_inr"],
                                                   city_multiplier)
                forecast_records.append({
                    "ds":         p_rec["ds"],
                    "yhat":       round(blended_yhat,  2),
                    "yhat_lower": round(blended_lower, 2),
                    "yhat_upper": round(blended_upper, 2),
                })
            data_quality   = "real_ml_ensemble"
            ml_model_used  = "prophet+xgboost"
            forecast_method = (
                f"Facebook Prophet ({_PROPHET_WEIGHT*100:.0f}%) + "
                f"XGBoost ({_XGB_WEIGHT*100:.0f}%) ensemble on "
                f"{n_rows} months of verified Indian market data"
            )
            training_rows  = n_rows

        elif prophet_records is not None:
            forecast_records = [
                {k: _validate_forecast(v, adjusted_base, material_key,
                                       seed["current_inr"], city_multiplier)
                 if k in ("yhat", "yhat_lower", "yhat_upper") else v
                 for k, v in rec.items()}
                for rec in prophet_records
            ]
            data_quality   = "real_prophet_only"
            ml_model_used  = "prophet"
            forecast_method = (
                f"Facebook Prophet time-series on {n_rows} months of real Indian market data"
            )
            training_rows  = n_rows

        elif xgb_prices is not None:
            # Build CI from rolling std of real series
            rolling_std = float(real_series["price_inr"].rolling(6).std().dropna().iloc[-1])                           if len(real_series) >= 6 else adjusted_base * 0.03
            forecast_records = []
            last_date_ts = real_series["date"].iloc[-1]
            for i, xp in enumerate(xgb_prices):
                dt    = last_date_ts + timedelta(days=i + 1)
                # xp is already in city-scale (XGB trained on city-specific CSV data).
                # Do NOT multiply by city_multiplier again — that would double-scale.
                yhat  = _validate_forecast(xp, adjusted_base,
                                           material_key, seed["current_inr"], city_multiplier)
                sigma = rolling_std
                lower = _validate_forecast(yhat - 1.645 * sigma, adjusted_base * 0.40,
                                           material_key, seed["current_inr"], city_multiplier)
                upper = _validate_forecast(yhat + 1.645 * sigma, adjusted_base * 3.0,
                                           material_key, seed["current_inr"], city_multiplier)
                forecast_records.append({
                    "ds":         dt.strftime("%Y-%m-%d"),
                    "yhat":       round(yhat,  2),
                    "yhat_lower": round(lower, 2),
                    "yhat_upper": round(upper, 2),
                })
            data_quality   = "real_xgb_only"
            ml_model_used  = "xgboost"
            forecast_method = (
                f"XGBoost regression on {n_rows} months of verified Indian market data"
            )
            training_rows  = n_rows

        else:
            # Both ML engines unavailable → linear real
            forecast_records, current, _ = _linear_real_forecast(
                real_series, horizon, material_key, city_multiplier
            )
            data_quality   = "historical"
            ml_model_used  = "linear_real"
            forecast_method = (
                f"Linear trend fit on {n_rows} months of real Indian market data "
                "(install prophet/xgboost for ML forecasting)"
            )
            training_rows  = n_rows

        # Compute sigma from CI widths for confidence calculation
        widths_sample  = [r["yhat_upper"] - r["yhat_lower"] for r in forecast_records[:30]]
        sigma_for_conf = float(np.mean(widths_sample)) / (2 * 1.96) if widths_sample else adjusted_base * 0.05

    # ── Derive summary stats ───────────────────────────────────────────────────
    p30 = forecast_records[29]["yhat"] if len(forecast_records) > 29  else current
    p60 = forecast_records[59]["yhat"] if len(forecast_records) > 59  else current
    p90 = forecast_records[-1]["yhat"] if forecast_records            else current

    pct_change_90d = round((p90 - current) / current * 100, 2)
    if abs(pct_change_90d) > 30:
        pct_change_90d = max(-30.0, min(30.0, pct_change_90d))
        p90 = current * (1 + pct_change_90d / 100)

    widths          = [r["yhat_upper"] - r["yhat_lower"] for r in forecast_records]
    volatility_score = float(np.clip(np.std(widths) / max(current, 1), 0, 1))
    volatility_label = "High" if volatility_score > 0.08 else "Medium" if volatility_score > 0.04 else "Low"
    trend            = "up" if p90 > current * 1.02 else "down" if p90 < current * 0.98 else "stable"

    # ── Confidence from Prophet interval width (real data) or heuristic (seed) ─
    if ml_model_used in ("prophet+xgboost", "prophet") and forecast_records:
        mean_spread = float(np.mean([r["yhat_upper"] - r["yhat_lower"] for r in forecast_records]))
        raw_conf    = 1.0 - (mean_spread / max(current, 1))
        confidence  = float(np.clip(raw_conf, 0.55, 0.95))
    else:
        confidence  = max(0.50, min(0.95, 0.90 - volatility_score * 2))

    confidence_label = "High" if confidence > 0.80 else "Medium" if confidence > 0.65 else "Low"

    # ── Blend CV-based confidence when available ───────────────────────────────
    cv_mape = _load_cv_mape(material_key, city or "Hyderabad")
    if cv_mape is not None:
        # Convert MAPE → confidence: 2% → 0.96, 5% → 0.90, 10% → 0.80, 20% → 0.60
        cv_confidence = max(0.55, min(0.95, 1.0 - cv_mape / 50.0))
        # Weight: 60% CI-based (current signal), 40% CV-based (historical rigor)
        confidence       = round(0.60 * confidence + 0.40 * cv_confidence, 3)
        confidence_label = "High" if confidence > 0.80 else "Medium" if confidence > 0.65 else "Low"
    buy_now_signal = bool(pct_change_90d > 5.0)

    confidence_note    = _build_confidence_note(
        material_key, data_quality, volatility_label, confidence,
        has_real_data=(real_series is not None), ml_model_used=ml_model_used,
    )
    trend_narrative     = _build_trend_narrative(
        material_key, trend, pct_change_90d, volatility_label, city, city_multiplier,
    )
    budget_impact       = _compute_budget_impact(material_key, pct_change_90d, area_sqft, room_type)

    if trend == "up" and abs(pct_change_90d) > 4:
        procurement_recommendation = (
            f"Buy Now: {seed['display_name']} expected to cost {pct_change_90d:.1f}% more in 90 days."
        )
    elif trend == "down" and abs(pct_change_90d) > 3:
        procurement_recommendation = (
            f"Defer: {seed['display_name']} prices may drop {abs(pct_change_90d):.1f}% "
            "— delay 30–45 days."
        )
    else:
        procurement_recommendation = (
            f"Neutral: {seed['display_name']} prices stable — procure on project schedule."
        )

    return {
        # ── Core fields (unchanged public API) ────────────────────────────────
        "material_key":               material_key,
        "display_name":               seed["display_name"],
        "category":                   seed.get("category", ""),
        "unit":                       seed["unit"],
        "current_price_inr":          round(current, 2),
        "current_price_national_inr": seed["current_inr"],
        "city_multiplier":            round(city_multiplier, 3),
        "city_adjusted":              city or "national",
        "forecast_30d_inr":           round(p30, 2),
        "forecast_60d_inr":           round(p60, 2),
        "forecast_90d_inr":           round(p90, 2),
        "pct_change_30d":             round((p30 - current) / current * 100, 2),
        "pct_change_60d":             round((p60 - current) / current * 100, 2),
        "pct_change_90d":             pct_change_90d,
        "volatility_score":           round(volatility_score, 4),
        "volatility_label":           volatility_label,
        "trend":                      trend,
        "confidence":                 round(confidence, 3),
        "confidence_label":           confidence_label,
        "trend_narrative":            trend_narrative,
        "procurement_recommendation": procurement_recommendation,
        "budget_impact":              budget_impact,
        "buy_now_signal":             buy_now_signal,
        "last_verified_date":         _SEED_LAST_VERIFIED,
        "confidence_note":            confidence_note,
        "data_quality":               data_quality,
        "raw_forecast_tail":          forecast_records,
        "generated_at":               datetime.utcnow().isoformat(),
        # ── New v5.0 fields ───────────────────────────────────────────────────
        "ml_model_used":              ml_model_used,
        "training_data_rows":         training_rows if 'training_rows' in dir() else n_rows,
        "forecast_method":            forecast_method if 'forecast_method' in dir() else "",
        "cv_mape_pct":                cv_mape,  # None if CV not run, float% if available
    }


# ── RenovationCostPredictor (UNCHANGED from v4.0) ─────────────────────────────

class RenovationCostPredictor:
    _model = None

    def __init__(self):
        if RenovationCostPredictor._model is None:
            try:
                from ml.property_models import RenovationCostModel
                RenovationCostPredictor._model = RenovationCostModel()
                logger.info("[RenovationCostPredictor] Model loaded")
            except Exception as e:
                logger.warning(f"[RenovationCostPredictor] Model unavailable ({e})")

    def predict(
        self,
        room_type: str,
        budget_tier: str,
        area_sqft: float,
        city: str,
        age_years: int = 10,
        scope: str = "partial",
        cv_features: Optional[Dict] = None,
    ) -> Dict:
        city_tier = {
            "Mumbai": 1, "Delhi NCR": 1, "Bangalore": 1, "Hyderabad": 1,
            "Chennai": 1, "Pune": 1, "Kolkata": 1,
        }.get(city, 2)

        if cv_features:
            n_objects = len(cv_features.get("detected_objects", []))
            if n_objects > 8:
                detected_scope = "full_room"
            elif n_objects > 4:
                detected_scope = "partial"
            else:
                detected_scope = "cosmetic_only"
            scope_order = ["cosmetic_only", "partial", "full_room", "structural_plus"]
            scope = scope_order[max(scope_order.index(scope), scope_order.index(detected_scope))]

        if RenovationCostPredictor._model is not None:
            result = RenovationCostPredictor._model.predict(
                room_type=room_type, budget_tier=budget_tier,
                area_sqft=area_sqft, city_tier=city_tier,
                age_years=age_years, scope=scope,
            )
        else:
            from ml.housing_preprocessor import RENO_COST_BENCHMARKS
            city_mult = CITY_COST_MULTIPLIER.get(city, 1.0)
            reno_area = area_sqft if room_type == "full_home" else area_sqft * 0.25
            lo, hi    = RENO_COST_BENCHMARKS.get(room_type, {}).get(budget_tier, (500, 1200))
            cost      = int(reno_area * (lo + hi) / 2 * city_mult)
            result    = {
                "renovation_cost_inr":      cost,
                "renovation_cost_low_inr":  int(reno_area * lo * city_mult),
                "renovation_cost_high_inr": int(reno_area * hi * city_mult),
                "cost_per_sqft":            int(cost / max(reno_area, 1)),
                "reno_area_sqft":           round(reno_area, 1),
                "cost_breakdown":           {},
                "confidence":               0.60,
                "model_type":               "benchmark",
            }

        city_mult = CITY_COST_MULTIPLIER.get(city, 1.0)
        tier_mult = {1: 1.10, 2: 1.00, 3: 0.88}.get(city_tier, 1.0)
        net_adj   = city_mult / tier_mult
        if abs(net_adj - 1.0) > 0.05:
            result["renovation_cost_inr"]      = int(result["renovation_cost_inr"]      * net_adj)
            result["renovation_cost_low_inr"]  = int(result["renovation_cost_low_inr"]  * net_adj)
            result["renovation_cost_high_inr"] = int(result["renovation_cost_high_inr"] * net_adj)

        result["city"]        = city
        result["room_type"]   = room_type
        result["scope"]       = scope
        result["budget_tier"] = budget_tier
        return result


# ── PriceForecastAgent — public API unchanged ─────────────────────────────────

class PriceForecastAgent:
    """
    Main agent exposed to the ARKEN graph pipeline.
    Public API (forecast_all, forecast_material, forecast_for_project) is unchanged.
    """

    def __init__(self):
        self._reno_cost_predictor = RenovationCostPredictor()
        pm = _get_price_manager()

        prophet = _get_prophet_engine()
        xgb     = _get_xgb_regressor()

        # ── Warm up ML models (blocking; called once at startup) ──────────────
        if pm.has_real_data:
            logger.info("[PriceForecastAgent] Warming up ProphetForecastEngine …")
            n_prophet = prophet.warm_up(pm)
            logger.info(
                f"[PriceForecastAgent] ProphetForecastEngine warm-up: {n_prophet} models."
            )

            logger.info("[PriceForecastAgent] Loading/training XGBoostPriceRegressor …")
            xgb.load_or_train(pm)
        else:
            logger.warning(
                "[PriceForecastAgent] No real price CSV found — ML models inactive. "
                "All forecasts use seed-based estimates."
            )

        logger.info("PriceForecastAgent v5.0 initialised")

    def predict_renovation_cost(
        self,
        room_type: str,
        budget_tier: str,
        area_sqft: float,
        city: str,
        age_years: int = 10,
        scope: str = "partial",
        cv_features: Optional[Dict] = None,
    ) -> Dict:
        return self._reno_cost_predictor.predict(
            room_type=room_type, budget_tier=budget_tier,
            area_sqft=area_sqft, city=city, age_years=age_years,
            scope=scope, cv_features=cv_features,
        )

    def forecast_all(self, horizon_days=90, city=None, area_sqft=None, room_type=None):
        results = []
        for key in SEED_DATA:
            try:
                results.append(_fit_and_forecast(key, horizon_days, city, area_sqft, room_type))
            except Exception as e:
                logger.warning(f"[PriceForecast] Forecast failed for {key}: {e}")
        return results

    def forecast_material(
        self, material_key, horizon_days=90, city=None, area_sqft=None, room_type=None,
    ):
        if material_key not in SEED_DATA:
            return None
        try:
            return _fit_and_forecast(material_key, horizon_days, city, area_sqft, room_type)
        except Exception as e:
            logger.warning(f"[PriceForecast] Forecast failed for {material_key}: {e}")
            return None

    def forecast_for_project(
        self, room_type, area_sqft, city="Hyderabad",
        horizon_days=90, materials_override=None,
    ):
        relevant_keys = (
            [k for k in materials_override if k in SEED_DATA]
            if materials_override else
            [
                key for key, data in SEED_DATA.items()
                if room_type in data.get("relevance", []) or
                   "full_home" in data.get("relevance", [])
            ]
        )
        forecasts = []
        for key in relevant_keys:
            try:
                f = _fit_and_forecast(key, horizon_days, city, area_sqft, room_type)
                f["project_relevant"] = True
                forecasts.append(f)
            except Exception as e:
                logger.warning(f"[PriceForecast] Project forecast failed for {key}: {e}")

        rising    = [f for f in forecasts if f["trend"] == "up" and f["pct_change_90d"] > 2]
        falling   = [f for f in forecasts if f["trend"] == "down"]
        hi_impact = [
            f for f in forecasts
            if isinstance(f.get("budget_impact"), dict) and
               f["budget_impact"].get("urgency") == "high"
        ]
        total_delta = sum(
            f["budget_impact"].get("delta_inr", 0)
            for f in forecasts
            if isinstance(f.get("budget_impact"), dict)
        )

        # data_reliability
        pm = _get_price_manager()
        if pm.has_real_data:
            real_dq_labels  = {"real_ml_ensemble", "real_prophet_only",
                                "real_xgb_only", "historical", "linear_real"}
            materials_real  = sum(1 for f in forecasts if f.get("data_quality") in real_dq_labels)
            coverage_ratio  = materials_real / max(len(forecasts), 1)
            if coverage_ratio >= 0.7:
                data_reliability = "high"
            elif coverage_ratio >= 0.3:
                data_reliability = "medium"
            else:
                data_reliability = "estimated"
        else:
            data_reliability = "estimated"

        portfolio_summary = {
            "total_materials_tracked":      len(forecasts),
            "rising_count":                 len(rising),
            "falling_count":                len(falling),
            "high_urgency_count":           len(hi_impact),
            "estimated_budget_impact_inr":  total_delta,
            "portfolio_action": (
                "Procure key materials now — significant price increases expected."
                if total_delta > 15_000 else
                "Standard procurement timeline acceptable — no major price shocks expected."
                if abs(total_delta) < 5_000 else
                "Monitor pricing — moderate movement anticipated."
            ),
            "top_urgent_materials": [
                {
                    "name":       f["display_name"],
                    "trend":      f["trend"],
                    "pct_change": f["pct_change_90d"],
                }
                for f in sorted(rising, key=lambda x: x["pct_change_90d"], reverse=True)[:3]
            ],
            "data_reliability": data_reliability,
        }

        forecasts.sort(key=lambda x: x["pct_change_90d"], reverse=True)
        return {
            "city":              city,
            "room_type":         room_type,
            "area_sqft":         area_sqft,
            "horizon_days":      horizon_days,
            "forecasts":         forecasts,
            "portfolio_summary": portfolio_summary,
            "generated_at":      datetime.utcnow().isoformat(),
        }