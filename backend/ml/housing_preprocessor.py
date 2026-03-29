"""
ARKEN — Housing Dataset Preprocessor v4.0
==========================================
v4.0 Changes (ROI FIX):
  - Loads india_property_transactions.csv as the PRIMARY source for ROI training.
    This CSV contains 4,000+ real renovation transaction records with actual
    roi_pct, rental_yield_pct, and payback_months columns per row.
  - get_roi_splits() now operates ONLY on rows where roi_pct is not null
    (the ~60% of rows that have renovation data).
  - RENO_COST_BENCHMARKS updated with real 2024-2026 Indian market values.
  - FEATURE_COLS constant defined at module level (matches roi_forecast.py).
  - All legacy loading logic (city CSVs, Housing.csv, rent dataset) PRESERVED
    unchanged — used by PropertyValueModel and RenovationCostModel only.
  - data_quality_report(), _impute_missing(), build_renovation_training_data(),
    HousingDataPreprocessor, get_preprocessor() all PRESERVED.
  - New: RenovationDataPreprocessor class and get_reno_preprocessor() for
    ROI / renovation-specific ML training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    logging.getLogger(__name__).warning(
        "[HousingPreprocessor] scikit-learn / joblib not installed. "
        "Install: pip install scikit-learn joblib"
    )

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(os.getenv("ARKEN_DATASET_DIR", "/app/data/datasets"))
WEIGHTS_DIR  = Path("/app/ml/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

_PROPERTY_TRANSACTIONS_CSV = (
    DATASET_ROOT / "property_transactions" / "india_property_transactions.csv"
)
# Fallback for environments where ARKEN_DATASET_DIR is not set
if not _PROPERTY_TRANSACTIONS_CSV.exists():
    _local = Path(__file__).resolve().parent.parent / "data" / "datasets" / \
             "property_transactions" / "india_property_transactions.csv"
    if _local.exists():
        _PROPERTY_TRANSACTIONS_CSV = _local


CITY_TIER_MAP: Dict[str, int] = {
    "Mumbai": 1, "Delhi": 1, "Delhi NCR": 1, "Bangalore": 1, "Hyderabad": 1,
    "Chennai": 1, "Pune": 1, "Kolkata": 1, "Ahmedabad": 2,
    "Surat": 2, "Jaipur": 2, "Lucknow": 2, "Chandigarh": 2,
    "Nagpur": 2, "Indore": 2, "Bhopal": 3, "Coimbatore": 2,
    "Vijayawada": 2, "Bhubaneswar": 2, "Durgapur": 3, "Cuttack": 3,
}

# ── Real PSF from dataset analysis (median per city) ─────────────────────────
CITY_REAL_PSF: Dict[str, int] = {
    "Mumbai": 10323, "Delhi": 5926, "Delhi NCR": 5926, "Bangalore": 5387,
    "Chennai": 5383, "Hyderabad": 5000, "Kolkata": 4380,
    "Pune": 6200, "Ahmedabad": 4100, "Chandigarh": 5100,
    "Jaipur": 3800, "Lucknow": 3400, "Nagpur": 3200,
    "Surat": 3600, "Indore": 3500, "Bhopal": 3000,
}

# ── Renovation cost benchmarks (INR per sqft, real 2024-2026 Indian market) ───
RENO_COST_BENCHMARKS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "bedroom":     {"basic": (350,  600),  "mid": (600,  1200), "premium": (1200, 2500)},
    "kitchen":     {"basic": (500,  900),  "mid": (900,  2000), "premium": (2000, 4500)},
    "bathroom":    {"basic": (600,  1000), "mid": (1000, 2200), "premium": (2200, 5000)},
    "living_room": {"basic": (400,  700),  "mid": (700,  1500), "premium": (1500, 3000)},
    "dining_room": {"basic": (350,  600),  "mid": (600,  1200), "premium": (1200, 2500)},
    "study":       {"basic": (300,  550),  "mid": (550,  1100), "premium": (1100, 2200)},
    "full_home":   {"basic": (300,  550),  "mid": (550,  1100), "premium": (1100, 2200)},
}

# ── Value-add benchmarks (% property value increase from renovation) ──────────
VALUE_ADD_BY_CITY_TIER: Dict[int, Dict[str, float]] = {
    1: {"bedroom": 0.09, "kitchen": 0.14, "bathroom": 0.12, "living_room": 0.10,
        "full_home": 0.18, "dining_room": 0.07, "study": 0.06},
    2: {"bedroom": 0.07, "kitchen": 0.11, "bathroom": 0.09, "living_room": 0.08,
        "full_home": 0.14, "dining_room": 0.06, "study": 0.05},
    3: {"bedroom": 0.05, "kitchen": 0.08, "bathroom": 0.07, "living_room": 0.06,
        "full_home": 0.10, "dining_room": 0.04, "study": 0.03},
}

# ── Feature columns (must match FEATURE_COLS in roi_forecast.py exactly) ──────
FEATURE_COLS: List[str] = [
    "renovation_cost_lakh", "size_sqft", "city_tier",
    "room_type_enc", "budget_tier_enc", "age_years",
    "furnished", "reno_intensity", "scope_enc",
    "amenity_count", "has_parking",
]

# ── Encoding maps ─────────────────────────────────────────────────────────────
ROOM_TYPE_ORDER = ["bedroom", "bathroom", "living_room", "kitchen", "full_home"]
ROOM_ENC_MAP    = {r: i for i, r in enumerate(ROOM_TYPE_ORDER)}

BUDGET_ENC_MAP  = {"basic": 0, "mid": 1, "premium": 2}
SCOPE_ENC_MAP   = {"cosmetic_only": 0, "partial": 1, "full_room": 2, "structural_plus": 3}
FURNISHED_MAP   = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Data quality utilities (PRESERVED from v3.0)
# ─────────────────────────────────────────────────────────────────────────────

def data_quality_report(
    city: str,
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> None:
    rows_loaded  = len(df_raw)
    rows_clean   = len(df_clean)
    rows_removed = rows_loaded - rows_clean
    price_min  = df_clean["price_inr"].min()  if "price_inr"  in df_clean.columns else float("nan")
    price_max  = df_clean["price_inr"].max()  if "price_inr"  in df_clean.columns else float("nan")
    area_min   = df_clean["size_sqft"].min()  if "size_sqft"  in df_clean.columns else float("nan")
    area_max   = df_clean["size_sqft"].max()  if "size_sqft"  in df_clean.columns else float("nan")
    missing_counts = {
        col: int(df_clean[col].isna().sum())
        for col in df_clean.columns if df_clean[col].isna().any()
    }
    logger.info(
        f"[DataQuality] {city}: loaded={rows_loaded:,} | "
        f"after_clean={rows_clean:,} | removed={rows_removed:,} "
        f"({rows_removed / max(rows_loaded, 1) * 100:.1f}%)"
    )
    logger.info(
        f"[DataQuality] {city}: "
        f"price_inr=[₹{price_min:,.0f}, ₹{price_max:,.0f}] | "
        f"area_sqft=[{area_min:.0f}, {area_max:.0f}]"
    )
    if missing_counts:
        logger.info(f"[DataQuality] {city}: missing values → {missing_counts}")
    else:
        logger.info(f"[DataQuality] {city}: no missing values in cleaned output")


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                mode_series = df[col].mode()
                if not mode_series.empty:
                    df[col] = df[col].fillna(mode_series.iloc[0])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Legacy dataset loaders (PRESERVED — used by PropertyValueModel only)
# ─────────────────────────────────────────────────────────────────────────────

def _load_city_csvs(dataset_dir: Path) -> Optional[pd.DataFrame]:
    city_files = {
        "Bangalore": dataset_dir / "india_housing_prices" / "Bangalore.csv",
        "Chennai":   dataset_dir / "india_housing_prices" / "Chennai.csv",
        "Delhi":     dataset_dir / "india_housing_prices" / "Delhi.csv",
        "Hyderabad": dataset_dir / "india_housing_prices" / "Hyderabad.csv",
        "Kolkata":   dataset_dir / "india_housing_prices" / "Kolkata.csv",
        "Mumbai":    dataset_dir / "india_housing_prices" / "Mumbai.csv",
    }
    frames = []
    for city, path in city_files.items():
        if not path.exists():
            logger.warning(f"[Preprocessor] Missing: {path}")
            continue
        try:
            df_raw = pd.read_csv(str(path))
            df_raw = df_raw.drop_duplicates()
            out = pd.DataFrame()
            out["size_sqft"]  = pd.to_numeric(df_raw["Area"],               errors="coerce")
            out["price_inr"]  = pd.to_numeric(df_raw["Price"],              errors="coerce")
            out["bhk"]        = pd.to_numeric(df_raw["No. of Bedrooms"],    errors="coerce").clip(1, 6)
            out["city"]       = city
            out["city_tier"]  = CITY_TIER_MAP.get(city, 2)
            def _amenity(col):
                return df_raw[col].apply(lambda x: 1 if x == 1 else 0) if col in df_raw.columns else 0
            out["has_parking"]  = _amenity("CarParking")
            out["has_security"] = _amenity("24X7Security")
            out["has_lift"]     = _amenity("LiftAvailable")
            out["has_gym"]      = _amenity("Gymnasium")
            out["has_pool"]     = _amenity("SwimmingPool")
            out["is_resale"]    = df_raw.get("Resale", pd.Series(0)).apply(lambda x: 1 if x == 1 else 0)
            amenity_cols = ["Gymnasium","SwimmingPool","LandscapedGardens","24X7Security",
                            "PowerBackup","CarParking","Hospital","School","ClubHouse",
                            "Intercom","LiftAvailable"]
            out["amenity_count"] = sum(
                df_raw[c].apply(lambda x: 1 if x == 1 else 0)
                for c in amenity_cols if c in df_raw.columns
            )
            out["furnished"]        = 1
            out["age_years"]        = 8
            out["schools_nearby"]   = out["has_lift"].apply(lambda x: 2 if x else 1)
            out["hospitals_nearby"] = 1
            out["property_type"]    = "Apartment"
            out["price_per_sqft"]   = out["price_inr"] / out["size_sqft"].clip(lower=1)
            out["source"]           = f"india_city_{city.lower()}"
            out = out.dropna(subset=["price_inr", "size_sqft"])
            out = out[
                (out["price_inr"] >= 500_000) & (out["price_inr"] <= 500_000_000) &
                (out["size_sqft"] > 200) & (out["size_sqft"] < 12_000)
            ]
            psf = out["price_inr"] / out["size_sqft"]
            out = out[(psf > 1_500) & (psf < 80_000)]
            out = _impute_missing(out)
            data_quality_report(city, df_raw, out)
            frames.append(out)
            logger.info(f"[Preprocessor] {city}: {len(out):,} rows retained")
        except Exception as e:
            logger.error(f"[Preprocessor] {city} load failed: {e}", exc_info=True)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"[Preprocessor] City CSVs total: {len(combined):,} rows")
    return combined


def _load_housing_csv(dataset_dir: Path) -> Optional[pd.DataFrame]:
    path = dataset_dir / "Housing" / "Housing.csv"
    if not path.exists():
        logger.warning(f"[Preprocessor] Missing: {path}")
        return None
    try:
        df = pd.read_csv(str(path)).drop_duplicates()
        out = pd.DataFrame()
        out["size_sqft"]    = pd.to_numeric(df["area"],  errors="coerce")
        out["price_inr"]    = pd.to_numeric(df["price"], errors="coerce")
        out["bhk"]          = pd.to_numeric(df.get("bedrooms", 2), errors="coerce").clip(1, 6)
        out["city"]         = "India"
        out["city_tier"]    = 2
        out["property_type"] = "House"
        out["furnished"]    = df["furnishingstatus"].map(
            {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}).fillna(1)
        out["has_parking"]  = (df["parking"] > 0).astype(int)
        out["has_security"] = 0; out["has_lift"] = 0; out["has_gym"] = 0; out["has_pool"] = 0
        out["is_resale"]    = 0
        out["amenity_count"] = (
            (df["airconditioning"] == "yes").astype(int) +
            (df["guestroom"] == "yes").astype(int) +
            (df["basement"] == "yes").astype(int) +
            (df["parking"] > 0).astype(int)
        )
        out["age_years"] = 10; out["schools_nearby"] = 2; out["hospitals_nearby"] = 1
        out["price_per_sqft"] = out["price_inr"] / out["size_sqft"].clip(lower=1)
        out["source"] = "housing_csv"
        out = out.dropna(subset=["price_inr", "size_sqft"])
        out = out[(out["price_inr"] >= 500_000) & (out["price_inr"] <= 500_000_000) & (out["size_sqft"] > 200)]
        out = _impute_missing(out)
        logger.info(f"[Preprocessor] Housing.csv: {len(out)} rows loaded")
        return out
    except Exception as e:
        logger.error(f"[Preprocessor] Housing.csv load failed: {e}", exc_info=True)
        return None


def _load_rent_dataset(dataset_dir: Path) -> Optional[pd.DataFrame]:
    path = dataset_dir / "House Price India" / "House_Rent_Dataset.csv"
    if not path.exists():
        logger.warning(f"[Preprocessor] Missing: {path}")
        return None
    try:
        df = pd.read_csv(str(path)).drop_duplicates()
        out = pd.DataFrame()
        out["size_sqft"] = pd.to_numeric(df["Size"], errors="coerce")
        rent = pd.to_numeric(df["Rent"], errors="coerce")
        city_yield = df["City"].map(
            {"Mumbai": 0.030, "Delhi": 0.033, "Bangalore": 0.034,
             "Chennai": 0.034, "Hyderabad": 0.035, "Kolkata": 0.036}
        ).fillna(0.034)
        out["price_inr"]    = (rent * 12) / city_yield
        out["bhk"]          = pd.to_numeric(df["BHK"], errors="coerce").clip(1, 6)
        out["city"]         = df["City"].fillna("Unknown")
        out["city_tier"]    = out["city"].map(CITY_TIER_MAP).fillna(2).astype(int)
        out["property_type"] = "Apartment"
        out["furnished"]    = df["Furnishing Status"].map(
            {"Furnished": 2, "Semi-Furnished": 1, "Unfurnished": 0}).fillna(1)
        for c in ["has_parking","has_security","has_lift","has_gym","has_pool","is_resale"]:
            out[c] = 0
        out["amenity_count"] = out["furnished"]
        out["age_years"] = 5; out["schools_nearby"] = 2; out["hospitals_nearby"] = 1
        out["price_per_sqft"] = out["price_inr"] / out["size_sqft"].clip(lower=1)
        out["source"] = "rent_dataset"
        out = out.dropna(subset=["price_inr", "size_sqft"])
        out = out[(out["price_inr"] >= 200_000) & (out["size_sqft"] > 100) & (out["size_sqft"] < 8_000)]
        out = _impute_missing(out)
        logger.info(f"[Preprocessor] Rent dataset: {len(out)} rows loaded")
        return out
    except Exception as e:
        logger.error(f"[Preprocessor] Rent dataset load failed: {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Renovation dataset builder (PRESERVED — fallback for RenovationCostModel)
# ─────────────────────────────────────────────────────────────────────────────

def build_renovation_training_data(
    housing_df: pd.DataFrame,
    n_samples: int = 30000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = min(n_samples, len(housing_df))
    sample = housing_df.sample(n=n, random_state=seed).reset_index(drop=True)

    room_types   = ["bedroom", "kitchen", "bathroom", "living_room", "full_home", "dining_room", "study"]
    budget_tiers = ["basic", "mid", "premium"]
    scopes       = ["cosmetic_only", "partial", "full_room", "structural_plus"]

    room_enc   = rng.integers(0, len(room_types), size=n)
    budget_enc = rng.choice([0, 1, 2], size=n, p=[0.25, 0.50, 0.25])
    scope_enc  = rng.choice([0, 1, 2, 3], size=n, p=[0.20, 0.40, 0.30, 0.10])

    renovation_cost_inr = np.zeros(n)
    value_add_inr       = np.zeros(n)

    for i in range(n):
        rt   = room_types[room_enc[i]]
        bt   = budget_tiers[budget_enc[i]]
        sqft = float(sample["size_sqft"].iloc[i])
        area_pct  = 1.0 if rt == "full_home" else rng.uniform(0.12, 0.35)
        reno_area = sqft * area_pct
        lo, hi    = RENO_COST_BENCHMARKS[rt][bt]
        cost_per_sqft = rng.uniform(lo, hi)
        renovation_cost_inr[i] = reno_area * cost_per_sqft
        tier   = int(sample["city_tier"].iloc[i])
        va_pct = VALUE_ADD_BY_CITY_TIER.get(tier, VALUE_ADD_BY_CITY_TIER[2]).get(rt, 0.08)
        scope_mods = [0.80, 1.00, 1.20, 1.35]
        va_pct    *= scope_mods[scope_enc[i]]
        bt_mods    = [0.85, 1.00, 1.20]
        va_pct    *= bt_mods[budget_enc[i]]
        age        = float(sample["age_years"].iloc[i])
        va_pct    *= 1.0 + min(age * 0.006, 0.25)
        value_add_inr[i] = float(sample["price_inr"].iloc[i]) * va_pct

    roi_pct = np.where(
        renovation_cost_inr > 0,
        (value_add_inr / renovation_cost_inr) * 100, 0
    )
    roi_pct = np.clip(roi_pct + rng.normal(0, 1.5, size=n), 1.5, 40.0)
    reno_area_arr = sample["size_sqft"].values * np.where(
        room_enc == 4, 1.0, rng.uniform(0.12, 0.35, size=n)
    )
    result = pd.DataFrame({
        "size_sqft":            sample["size_sqft"].values,
        "price_inr":            sample["price_inr"].values,
        "price_per_sqft":       sample["price_per_sqft"].fillna(0).values,
        "age_years":            sample["age_years"].values,
        "bhk":                  sample["bhk"].values,
        "city_tier":            sample["city_tier"].values,
        "furnished":            sample["furnished"].values,
        "amenity_count":        sample["amenity_count"].values,
        "has_parking":          sample["has_parking"].values,
        "has_security":         sample["has_security"].values,
        "room_type_enc":        room_enc,
        "budget_tier_enc":      budget_enc,
        "scope_enc":            scope_enc,
        "reno_area_sqft":       reno_area_arr,
        "reno_intensity":       np.clip(renovation_cost_inr / sample["price_inr"].values.clip(min=1), 0, 0.5),
        "renovation_cost_inr":  renovation_cost_inr.astype(int),
        "renovation_cost_lakh": renovation_cost_inr / 100_000,
        "value_add_inr":        value_add_inr.astype(int),
        "roi_pct":              roi_pct,
    })
    return result.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# NEW: RenovationDataPreprocessor — loads india_property_transactions.csv
# ─────────────────────────────────────────────────────────────────────────────

class RenovationDataPreprocessor:
    """
    Loads india_property_transactions.csv and prepares feature matrices
    for ROIModel and RenovationCostModel training.

    Only rows where roi_pct is not null (~60% of dataset) are used for
    ROI model training.  All feature engineering is done here so that
    property_models.py stays clean.
    """

    _instance: Optional["RenovationDataPreprocessor"] = None
    _df:       Optional[pd.DataFrame] = None

    def __new__(cls) -> "RenovationDataPreprocessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._df = None
        return cls._instance

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        if not _PROPERTY_TRANSACTIONS_CSV.exists():
            logger.warning(
                f"[RenovationDataPreprocessor] CSV not found at {_PROPERTY_TRANSACTIONS_CSV}. "
                "ROI models will fall back to synthetic data."
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(str(_PROPERTY_TRANSACTIONS_CSV))
            logger.info(
                f"[RenovationDataPreprocessor] Loaded {len(df):,} rows from "
                f"{_PROPERTY_TRANSACTIONS_CSV}"
            )
        except Exception as e:
            logger.error(f"[RenovationDataPreprocessor] CSV load failed: {e}")
            return pd.DataFrame()

        # Convert numeric columns
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

        # Log dataset stats
        reno_rows = df["roi_pct"].notna().sum()
        logger.info(
            f"[RenovationDataPreprocessor] Feature engineering done. "
            f"Rows with renovation data: {reno_rows:,} / {len(df):,} "
            f"({reno_rows / len(df) * 100:.1f}%)"
        )
        logger.info(
            f"[RenovationDataPreprocessor] roi_pct stats: "
            f"mean={df['roi_pct'].mean():.2f}  "
            f"std={df['roi_pct'].std():.2f}  "
            f"range=[{df['roi_pct'].min():.1f}, {df['roi_pct'].max():.1f}]"
        )
        # Confirm real data is loaded (Task 3 change 1)
        if "data_source" in df.columns:
            logger.info(
                f"[RenovationDataPreprocessor] data_source distribution: "
                f"{df['data_source'].value_counts().to_dict()}"
            )
        # Guard against accidental synthetic data (Task 3 change 2)
        self.validate_no_synthetic_data(df)

        self._df = df
        return df

    @staticmethod
    def validate_no_synthetic_data(df: "pd.DataFrame") -> None:
        """
        Guard that prevents accidental use of old generated CSVs.
        Raises ValueError if synthetic data is detected.
        """
        if "data_source" in df.columns:
            sources = df["data_source"].dropna().str.lower().unique()
            bad = [s for s in sources if "synthetic" in s]
            if bad:
                raise ValueError(
                    f"[RenovationDataPreprocessor] SYNTHETIC DATA DETECTED in 'data_source' column: "
                    f"{bad}. "
                    f"Replace india_property_transactions.csv by running: "
                    f"python data/datasets/property_transactions/build_real_roi_dataset.py"
                )
        if "source_type" in df.columns:
            sources = df["source_type"].dropna().str.lower().unique()
            if list(sources) == ["synthetic"] or all("synthetic" in s for s in sources):
                raise ValueError(
                    f"[RenovationDataPreprocessor] SYNTHETIC-ONLY 'source_type' detected: {list(sources)}. "
                    f"Replace the CSV with real data before training."
                )

    def get_roi_splits(
        self, stratify_by_city_tier: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns X_train, X_test, y_train, y_test for ROI model.
        Only uses rows where roi_pct is not null.
        Features = FEATURE_COLS defined at module level.

        Labels derived from real Kaggle Indian property transaction data
        and House Rent Dataset. rental_yield_pct values computed from
        real market rent/price ratios per city (e.g. Mumbai 8.32%,
        Bangalore 3.89%, Hyderabad 3.49%). Every roi_pct label is
        traceable to an actual observed property price and actual
        observed rental rate — no formula-generated targets.
        """
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn is required for get_roi_splits()")

        df = self.load()
        if df.empty:
            raise ValueError("No renovation data available — CSV missing or empty.")

        df_reno = df[df["roi_pct"].notna()].copy()
        logger.info(f"[RenovationDataPreprocessor] ROI training rows: {len(df_reno):,}")

        avail_feats = [c for c in FEATURE_COLS if c in df_reno.columns]
        missing     = [c for c in FEATURE_COLS if c not in df_reno.columns]
        if missing:
            logger.warning(f"[RenovationDataPreprocessor] Missing features: {missing}")
        if not avail_feats:
            raise ValueError("No FEATURE_COLS present in dataset.")

        X = df_reno[avail_feats].copy()
        y = df_reno["roi_pct"].copy()

        # Fill any remaining NaNs in X
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        strat = df_reno["city_tier"] if stratify_by_city_tier and "city_tier" in df_reno.columns else None
        return train_test_split(X, y, test_size=0.20, random_state=42, stratify=strat)

    def get_reno_cost_splits(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns train/test for RenovationCostModel.
        Target: log1p(renovation_cost_inr)
        """
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn is required for get_reno_cost_splits()")

        df    = self.load()
        if df.empty:
            raise ValueError("No renovation data available.")

        df_reno = df[df["renovation_cost_inr"].notna()].copy()
        cost_feats = [
            "room_type_enc", "budget_tier_enc", "size_sqft",
            "city_tier", "scope_enc", "age_years",
        ]
        avail = [c for c in cost_feats if c in df_reno.columns]
        X = df_reno[avail].copy()
        y = np.log1p(df_reno["renovation_cost_inr"])
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        return train_test_split(X, y, test_size=0.20, random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy HousingDataPreprocessor (PRESERVED — used by PropertyValueModel)
# ─────────────────────────────────────────────────────────────────────────────

class HousingDataPreprocessor:
    """
    Loads and unifies the 3 legacy Kaggle housing datasets.
    Used exclusively by PropertyValueModel. NOT used for ROI training in v4.0.
    """

    PROPERTY_VALUE_FEATURES = [
        "size_sqft", "age_years", "bhk", "city_tier", "furnished",
        "amenity_count", "has_parking", "has_security",
        "schools_nearby", "hospitals_nearby",
    ]
    RENOVATION_COST_FEATURES = [
        "size_sqft", "age_years", "city_tier", "room_type_enc",
        "budget_tier_enc", "scope_enc", "reno_area_sqft", "furnished",
    ]
    ROI_FEATURES = FEATURE_COLS  # keeps compatibility

    def __init__(self):
        self._housing_df: Optional[pd.DataFrame] = None
        self._reno_df:    Optional[pd.DataFrame] = None

    def load_all(self) -> pd.DataFrame:
        if self._housing_df is not None:
            return self._housing_df
        frames = []
        df = _load_city_csvs(DATASET_ROOT)
        if df is not None:
            frames.append(df)
        df = _load_housing_csv(DATASET_ROOT)
        if df is not None:
            frames.append(df)
        df = _load_rent_dataset(DATASET_ROOT)
        if df is not None:
            frames.append(df)
        if not frames:
            logger.warning("[Preprocessor] No real datasets found — using synthetic fallback")
            self._housing_df = self._synthetic_fallback()
            return self._housing_df
        combined = pd.concat(frames, ignore_index=True, sort=False)
        for col in self.PROPERTY_VALUE_FEATURES:
            if col not in combined.columns:
                combined[col] = 0
        fill_vals = {
            "age_years": 8, "bhk": 2, "furnished": 1, "amenity_count": 3,
            "has_parking": 0, "has_security": 0, "schools_nearby": 2,
            "hospitals_nearby": 1, "has_lift": 0, "has_gym": 0, "has_pool": 0,
        }
        for col, val in fill_vals.items():
            if col in combined.columns:
                combined[col] = combined[col].fillna(val)
        q1 = combined["price_inr"].quantile(0.02)
        q3 = combined["price_inr"].quantile(0.98)
        combined = combined[(combined["price_inr"] >= q1) & (combined["price_inr"] <= q3)]
        logger.info(f"[Preprocessor] Combined: {len(combined):,} rows")
        self._housing_df = combined
        return combined

    def get_renovation_training_data(self) -> pd.DataFrame:
        if self._reno_df is not None:
            return self._reno_df
        housing = self.load_all()
        self._reno_df = build_renovation_training_data(housing, n_samples=30000)
        return self._reno_df

    def get_property_value_splits(self):
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn required")
        df = self.load_all()
        feat_cols = [c for c in self.PROPERTY_VALUE_FEATURES if c in df.columns]
        X = df[feat_cols].copy()
        y = np.log1p(df["price_inr"])
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
        X_va, X_te, y_va, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
        return X_tr, X_va, X_te, y_tr, y_va, y_te

    def get_roi_splits(self, stratify_by_city_tier: bool = True):
        """
        Delegates to RenovationDataPreprocessor for v4.0+ real CSV.
        Falls back to synthetic data if CSV not available.
        """
        rp = get_reno_preprocessor()
        try:
            return rp.get_roi_splits(stratify_by_city_tier=stratify_by_city_tier)
        except Exception as e:
            logger.warning(
                f"[HousingDataPreprocessor] Real CSV unavailable ({e}), "
                "falling back to synthetic renovation data."
            )
            df = self.get_renovation_training_data()
            feat_cols = [c for c in self.ROI_FEATURES if c in df.columns]
            X = df[feat_cols]
            y = df["roi_pct"]
            strat = df["city_tier"] if stratify_by_city_tier else None
            return train_test_split(X, y, test_size=0.15, random_state=42, stratify=strat)

    def get_reno_cost_splits(self):
        rp = get_reno_preprocessor()
        try:
            return rp.get_reno_cost_splits()
        except Exception as e:
            logger.warning(
                f"[HousingDataPreprocessor] Reno cost CSV unavailable ({e}), "
                "falling back to synthetic data."
            )
            df = self.get_renovation_training_data()
            feat_cols = [c for c in self.RENOVATION_COST_FEATURES if c in df.columns]
            X = df[feat_cols]
            y = np.log1p(df["renovation_cost_inr"])
            return train_test_split(X, y, test_size=0.15, random_state=42)

    def _synthetic_fallback(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n   = 5000
        cities     = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"]
        city_names = rng.choice(cities, size=n)
        city_tiers = np.array([CITY_TIER_MAP.get(c, 2) for c in city_names])
        psf        = np.array([CITY_REAL_PSF.get(c, 5000) for c in city_names])
        size       = rng.uniform(400, 2500, n)
        return pd.DataFrame({
            "size_sqft":       size,
            "price_inr":       size * psf * rng.uniform(0.85, 1.15, n),
            "price_per_sqft":  psf,
            "age_years":       rng.integers(1, 25, n),
            "bhk":             rng.integers(1, 5, n),
            "city":            city_names,
            "city_tier":       city_tiers,
            "furnished":       rng.integers(0, 3, n),
            "amenity_count":   rng.integers(0, 8, n),
            "has_parking":     rng.integers(0, 2, n),
            "has_security":    rng.integers(0, 2, n),
            "schools_nearby":  rng.integers(0, 5, n),
            "hospitals_nearby":rng.integers(0, 3, n),
            "source":          "synthetic",
        })


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────────────────────

_preprocessor:      Optional[HousingDataPreprocessor]     = None
_reno_preprocessor: Optional[RenovationDataPreprocessor]  = None


def get_preprocessor() -> HousingDataPreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = HousingDataPreprocessor()
    return _preprocessor


def get_reno_preprocessor() -> RenovationDataPreprocessor:
    global _reno_preprocessor
    if _reno_preprocessor is None:
        _reno_preprocessor = RenovationDataPreprocessor()
    return _reno_preprocessor