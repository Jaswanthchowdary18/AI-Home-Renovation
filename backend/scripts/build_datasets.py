"""
ARKEN PropTech — Dataset & Model Build Script
=============================================
One-command setup that:

  1. Generates india_material_prices_historical.csv
     (6,912 rows: 12 materials × 8 cities × 72 months)

  2. Generates india_property_transactions.csv
     (7,000 rows: ~5,250 with renovation ROI data for ML training)

  3. (Optional) Trains both ML models:
       • ROIModel         → /app/ml/weights/roi_ensemble.joblib
       • RenovationCostModel → /app/ml/weights/reno_cost_xgb.joblib
       Skipped if --data-only flag is passed.

Usage (from repository root):
    # Full build: data + models
    python backend/scripts/build_datasets.py

    # Data only (skip ML training)
    python backend/scripts/build_datasets.py --data-only

    # Custom output paths
    python backend/scripts/build_datasets.py \\
        --material-csv /custom/path/material_prices.csv \\
        --property-csv /custom/path/property_transactions.csv

    # Quick test (fewer rows, fast)
    python backend/scripts/build_datasets.py --quick

Environment variables honoured:
    ARKEN_DATASET_DIR   Override dataset root (default: /app/data/datasets)
    ARKEN_WEIGHTS_DIR   Override ML weights dir (default: /app/ml/weights)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths resolved from environment or defaults ────────────────────────────────
_REPO_ROOT   = Path(__file__).parent.parent.resolve()   # backend/
_DATASET_ROOT = Path(os.getenv("ARKEN_DATASET_DIR", "/app/data/datasets"))
_WEIGHTS_DIR  = Path(os.getenv("ARKEN_WEIGHTS_DIR", "/app/ml/weights"))

_MATERIAL_GENERATOR  = (
    _REPO_ROOT / "data" / "datasets" / "material_prices" / "generate_material_prices.py"
)
_PROPERTY_GENERATOR  = (
    _REPO_ROOT / "data" / "datasets" / "property_transactions" / "generate_property_data.py"
)
_MATERIAL_CSV_OUT  = _DATASET_ROOT / "material_prices" / "india_material_prices_historical.csv"
_PROPERTY_CSV_OUT  = _DATASET_ROOT / "property_transactions" / "india_property_transactions.csv"


def _run(cmd: list, step: str) -> bool:
    """
    Run a subprocess command.  Returns True on success, False on failure.
    Streams stdout/stderr live so the user sees progress.
    """
    logger.info(f"── {step} ─────────────────────────────────────────────────────")
    logger.info("  CMD: " + " ".join(str(c) for c in cmd))
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.monotonic() - t0
        logger.info(f"  ✓  {step} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  ✗  {step} FAILED (returncode={e.returncode})")
        return False
    except FileNotFoundError as e:
        logger.error(f"  ✗  {step} FAILED — file not found: {e}")
        return False


def step_generate_material_prices(
    output_csv: Path,
    seed: int,
) -> bool:
    """Step 1 — Run generate_material_prices.py."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_MATERIAL_GENERATOR),
        "--output", str(output_csv),
        "--seed",   str(seed),
    ]
    return _run(cmd, "Step 1: Generate Material Prices CSV")


def step_generate_property_data(
    output_csv: Path,
    rows: int,
    seed: int,
) -> bool:
    """Step 2 — Run generate_property_data.py."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_PROPERTY_GENERATOR),
        "--output", str(output_csv),
        "--rows",   str(rows),
        "--seed",   str(seed),
    ]
    return _run(cmd, "Step 2: Generate Property Transactions CSV")


def step_train_models(weights_dir: Path) -> bool:
    """
    Step 3 — Train ROIModel and RenovationCostModel.

    Imports property_models directly so the training runs in the same Python
    process (no subprocess needed — avoids import path issues).
    """
    logger.info("── Step 3: Train ML Models ──────────────────────────────────")

    # Add backend to sys.path so imports work from anywhere
    backend_dir = _REPO_ROOT
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    weights_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    try:
        logger.info("  Importing ROIModel …")
        from ml.property_models import ROIModel, RenovationCostModel
    except ImportError as e:
        logger.error(
            f"  ✗  Cannot import property_models: {e}\n"
            "     Make sure scikit-learn, xgboost, and joblib are installed:\n"
            "     pip install scikit-learn xgboost joblib"
        )
        return False

    # --- Train ROI ensemble ---
    try:
        logger.info("  Training ROIModel (3-model ensemble: XGBoost + RF + GBM) …")
        roi_model = ROIModel()
        logger.info("  ✓  ROIModel trained")
    except Exception as e:
        logger.error(f"  ✗  ROIModel training failed: {e}", exc_info=True)
        return False

    # --- Train Renovation Cost model ---
    try:
        logger.info("  Training RenovationCostModel (XGBoost) …")
        cost_model = RenovationCostModel()
        logger.info("  ✓  RenovationCostModel trained")
    except Exception as e:
        logger.warning(
            f"  ⚠  RenovationCostModel training failed ({e}). "
            "ROI model still usable; cost predictions will fall back to benchmarks."
        )

    elapsed = time.monotonic() - t0
    logger.info(f"  ✓  Step 3 completed in {elapsed:.1f}s")
    return True


def _check_generators_exist() -> bool:
    ok = True
    for p in [_MATERIAL_GENERATOR, _PROPERTY_GENERATOR]:
        if not p.exists():
            logger.error(f"Generator script not found: {p}")
            ok = False
    return ok


def _print_summary(
    material_ok: bool,
    property_ok: bool,
    models_ok: bool,
    data_only: bool,
    material_csv: Path,
    property_csv: Path,
) -> None:
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║          ARKEN Dataset Build — Summary                  ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")

    def _row(label: str, status: bool, path: str = "") -> str:
        tick = "✓" if status else "✗"
        line = f"║  {tick}  {label:<30s}"
        if path:
            line += f"  {path}"
        return line.ljust(61) + "║"

    logger.info(_row("Material prices CSV",   material_ok, str(material_csv)))
    logger.info(_row("Property transactions CSV", property_ok, str(property_csv)))
    if not data_only:
        logger.info(_row("ML model training",  models_ok))

    all_ok = material_ok and property_ok and (data_only or models_ok)
    logger.info("╠══════════════════════════════════════════════════════════╣")
    if all_ok:
        logger.info("║  ✓  BUILD SUCCESSFUL                                    ║")
    else:
        logger.info("║  ✗  BUILD FAILED — see errors above                     ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    if all_ok:
        logger.info("Next steps:")
        logger.info("  1. Start the backend:  uvicorn main:app --reload")
        logger.info("  2. Test the forecast endpoint:")
        logger.info("       curl http://localhost:8000/api/forecast/materials?city=Mumbai")
        logger.info("  3. Check data_quality field in response")
        logger.info("     → 'real_ml_ensemble' means Prophet+XGBoost is active")
        logger.info("     → 'estimated_seed_based' means CSV was not found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARKEN one-command dataset + model build script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Generate CSV files only; skip ML model training"
    )
    parser.add_argument(
        "--material-csv", type=str, default=str(_MATERIAL_CSV_OUT),
        help=f"Output path for material prices CSV (default: {_MATERIAL_CSV_OUT})"
    )
    parser.add_argument(
        "--property-csv", type=str, default=str(_PROPERTY_CSV_OUT),
        help=f"Output path for property transactions CSV (default: {_PROPERTY_CSV_OUT})"
    )
    parser.add_argument(
        "--rows", type=int, default=7000,
        help="Number of property transaction rows to generate (default: 7000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode: fewer rows (2000 property, data-only)"
    )
    args = parser.parse_args()

    if args.quick:
        args.rows      = 2000
        args.data_only = True
        logger.info("Quick mode: 2000 property rows, skipping ML training.")

    material_csv = Path(args.material_csv)
    property_csv = Path(args.property_csv)

    logger.info("=" * 62)
    logger.info("  ARKEN PropTech — Dataset & Model Build")
    logger.info(f"  Backend root  : {_REPO_ROOT}")
    logger.info(f"  Material CSV  : {material_csv}")
    logger.info(f"  Property CSV  : {property_csv}")
    logger.info(f"  Property rows : {args.rows:,}")
    logger.info(f"  Seed          : {args.seed}")
    logger.info(f"  Data only     : {args.data_only}")
    logger.info("=" * 62)

    if not _check_generators_exist():
        logger.error(
            "One or more generator scripts are missing. "
            "Run: python backend/scripts/build_datasets.py from the repository root."
        )
        sys.exit(1)

    t_total = time.monotonic()

    # Step 1
    material_ok = step_generate_material_prices(material_csv, args.seed)

    # Step 2
    property_ok = step_generate_property_data(property_csv, args.rows, args.seed)

    # Step 3
    models_ok = True
    if not args.data_only:
        if not (material_ok and property_ok):
            logger.warning(
                "One or more data generation steps failed. "
                "Skipping ML training to avoid training on incomplete data."
            )
            models_ok = False
        else:
            models_ok = step_train_models(_WEIGHTS_DIR)
    else:
        logger.info("Skipping ML training (--data-only flag set).")

    total_elapsed = time.monotonic() - t_total
    logger.info(f"\nTotal build time: {total_elapsed:.1f}s")

    _print_summary(
        material_ok=material_ok,
        property_ok=property_ok,
        models_ok=models_ok,
        data_only=args.data_only,
        material_csv=material_csv,
        property_csv=property_csv,
    )

    all_ok = material_ok and property_ok and (args.data_only or models_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
