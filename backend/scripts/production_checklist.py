#!/usr/bin/env python3
"""
ARKEN — Production Deployment Checklist
==========================================
Runs all pre-deployment checks and prints a colour-coded results table.
Exits with code 0 if all CRITICAL checks pass; exits with code 1 if any fail.

Usage:
    cd backend
    python scripts/production_checklist.py

    # Non-zero exit on any critical failure (for CI gates):
    python scripts/production_checklist.py --strict

    # JSON output (for monitoring integration):
    python scripts/production_checklist.py --json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── ANSI colours ──────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_NO_COLOR = os.getenv("NO_COLOR", "")


def _c(color: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"{color}{text}{_RESET}"


def _ok(msg: str) -> str:
    return _c(_GREEN, f"  ✓  {msg}")


def _fail(msg: str) -> str:
    return _c(_RED, f"  ✗  {msg}")


def _warn(msg: str) -> str:
    return _c(_YELLOW, f"  ⚠  {msg}")


def _info(msg: str) -> str:
    return _c(_CYAN, f"  ℹ  {msg}")


# ── Path resolution ────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))

# Add backend to sys.path for imports
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _resolve(app_rel: str, local_rel: str) -> Path:
    a = _APP_DIR / app_rel
    b = _BACKEND_DIR / local_rel
    return a if a.exists() else b


# ── Check result type ─────────────────────────────────────────────────────────
# (name, passed, critical, detail)
CheckResult = Tuple[str, bool, bool, str]


class Checklist:
    def __init__(self) -> None:
        self._results: List[CheckResult] = []

    def add(self, name: str, passed: bool, critical: bool, detail: str) -> None:
        self._results.append((name, passed, critical, detail))

    @property
    def all_critical_passed(self) -> bool:
        return all(passed for _, passed, critical, _ in self._results if critical)

    @property
    def total(self) -> int:
        return len(self._results)

    @property
    def passed(self) -> int:
        return sum(1 for _, p, _, _ in self._results if p)

    @property
    def failed_critical(self) -> int:
        return sum(1 for _, p, c, _ in self._results if not p and c)

    @property
    def warnings(self) -> int:
        return sum(1 for _, p, c, _ in self._results if not p and not c)

    def print_table(self) -> None:
        width = 72
        print("\n" + _c(_BOLD, "═" * width))
        print(_c(_BOLD, "  ARKEN Production Deployment Checklist"))
        print(_c(_BOLD, "═" * width))

        current_section = ""
        for name, passed, critical, detail in self._results:
            # Section headers from name prefix "SECTION: ..."
            if name.startswith("__section__"):
                section = name[11:]
                if section != current_section:
                    current_section = section
                    print(f"\n  {_c(_BOLD, section)}")
                continue

            icon   = "✓" if passed else ("✗" if critical else "⚠")
            colour = _GREEN if passed else (_RED if critical else _YELLOW)
            label  = _c(colour, f"  [{icon}]")
            detail_str = f"  {detail}" if detail else ""
            print(f"{label}  {name}")
            if detail_str:
                print(_c(_CYAN, f"      {detail}"))

        print("\n" + "─" * width)
        status_colour = _GREEN if self.all_critical_passed else _RED
        print(_c(status_colour, f"  Result: {self.passed}/{self.total} checks passed  |  "
                                f"{self.failed_critical} critical failures  |  "
                                f"{self.warnings} warnings"))
        print("─" * width + "\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed":               self.passed,
            "total":                self.total,
            "failed_critical":      self.failed_critical,
            "warnings":             self.warnings,
            "all_critical_passed":  self.all_critical_passed,
            "checks": [
                {
                    "name":     name,
                    "passed":   passed,
                    "critical": critical,
                    "detail":   detail,
                }
                for name, passed, critical, detail in self._results
                if not name.startswith("__section__")
            ],
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_security(cl: Checklist) -> None:
    cl.add("__section__Security", True, False, "")

    # .env must not contain the known-bad Google API key
    env_path = _BACKEND_DIR / ".env"
    bad_key  = "AIzaSyBNKExIcDRRsZ0WUgxBVoRBbvegeMs2IHM"
    if env_path.exists():
        content = env_path.read_text(errors="replace")
        if bad_key in content:
            cl.add(
                ".env does not contain compromised Google API key",
                False, True,
                "CRITICAL: revoke AIzaSyBNKEx... at console.cloud.google.com/apis/credentials",
            )
        else:
            cl.add(
                ".env does not contain compromised Google API key",
                True, True, "",
            )

        # Check for placeholder SECRET_KEY
        if "GENERATE_WITH_openssl_rand_hex_32" in content or \
           "arken-dev-secret-key" in content or \
           "your-super-secret-key" in content:
            cl.add(
                "SECRET_KEY is set (not placeholder)",
                False, True,
                "Generate with: openssl rand -hex 32",
            )
        else:
            cl.add("SECRET_KEY is set (not placeholder)", True, True, "")
    else:
        cl.add(
            ".env file exists",
            False, True,
            f"Create {env_path} from .env.example",
        )

    # .env.example must not contain the bad key
    example_path = _BACKEND_DIR / ".env.example"
    if example_path.exists():
        ex_content = example_path.read_text(errors="replace")
        if bad_key in ex_content:
            cl.add(
                ".env.example does not contain real API keys",
                False, True,
                ".env.example still has the compromised key — run the env fix",
            )
        else:
            cl.add(".env.example does not contain real API keys", True, True, "")
    else:
        cl.add(".env.example exists", False, False, "Missing .env.example template")


def check_datasets(cl: Checklist) -> None:
    cl.add("__section__Dataset CSVs", True, False, "")

    required_csvs = [
        (
            "Material prices CSV",
            _resolve(
                "data/datasets/material_prices/india_material_prices_historical.csv",
                "data/datasets/material_prices/india_material_prices_historical.csv",
            ),
            True,
        ),
        (
            "Property transactions CSV",
            _resolve(
                "data/datasets/property_transactions/india_property_transactions.csv",
                "data/datasets/property_transactions/india_property_transactions.csv",
            ),
            True,
        ),
    ]

    for name, path, critical in required_csvs:
        if path.exists():
            size_kb = path.stat().st_size // 1024
            # Check minimum row count (quick head read)
            try:
                with open(path, "r") as f:
                    row_count = sum(1 for _ in f) - 1  # subtract header
                cl.add(name, True, critical, f"{path.name} — {row_count:,} rows, {size_kb:,} KB")
            except Exception:
                cl.add(name, True, critical, f"{path.name} — {size_kb:,} KB")
        else:
            cl.add(
                name, False, critical,
                f"Not found at {path}. Run: python scripts/build_datasets.py",
            )

    # Check data freshness (days since last date in material prices CSV)
    mat_csv = _resolve(
        "data/datasets/material_prices/india_material_prices_historical.csv",
        "data/datasets/material_prices/india_material_prices_historical.csv",
    )
    if mat_csv.exists():
        try:
            # Fast: just read the last line to get the latest date
            with open(mat_csv, "rb") as f:
                f.seek(0, 2)  # end
                # Read last ~200 bytes
                f.seek(max(0, f.tell() - 200))
                last_chunk = f.read().decode(errors="replace")
            last_line = [l for l in last_chunk.strip().splitlines() if l][-1]
            date_str  = last_line.split(",")[0].strip().strip('"')
            from datetime import datetime
            last_dt   = datetime.strptime(date_str[:10], "%Y-%m-%d")
            days_old  = (datetime.now() - last_dt).days
            fresh     = days_old <= int(os.getenv("DATA_FRESHNESS_ALERT_DAYS", "45"))
            cl.add(
                "Material prices CSV freshness",
                fresh, False,
                f"Last date: {date_str[:10]} ({days_old} days ago)"
                + ("" if fresh else " — consider refreshing"),
            )
        except Exception as exc:
            cl.add("Material prices CSV freshness", False, False, f"Could not read date: {exc}")


def check_ml_models(cl: Checklist) -> None:
    cl.add("__section__ML Model Files", True, False, "")

    weights_dir = _resolve("ml/weights", "ml/weights")
    model_files = [
        ("Price XGB model",          "price_xgb.joblib",             True),
        ("ROI XGB model",            "roi_xgb.joblib",               True),
        ("ROI RandomForest model",   "roi_rf.joblib",                True),
        ("ROI GradientBoosting model","roi_gbm.joblib",              True),
        ("Renovation cost model",    "renovation_cost_model.joblib", False),
        ("Model report",             "model_report.json",            False),
    ]

    warn_days = 30
    crit_days = 60

    for display_name, fname, critical in model_files:
        fpath = weights_dir / fname
        if not fpath.exists():
            cl.add(
                display_name, False, critical,
                f"Not found at {fpath}. Run: python scripts/build_datasets.py",
            )
            continue

        # File age check
        mtime    = fpath.stat().st_mtime
        age_days = (time.time() - mtime) / 86400
        if age_days > crit_days:
            detail = f"Age: {age_days:.0f} days — CRITICAL: retrain at POST /health/retrain"
            cl.add(display_name, False, False, detail)  # age warning only, not critical
        elif age_days > warn_days:
            detail = f"Age: {age_days:.0f} days — consider retraining (POST /health/retrain)"
            cl.add(display_name, True, False, detail)
        else:
            cl.add(display_name, True, critical, f"Age: {age_days:.0f} days")


def check_chromadb(cl: Checklist) -> None:
    cl.add("__section__ChromaDB RAG Corpus", True, False, "")

    chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(
        _resolve("data/chroma", "data/chroma")
    )))

    if not chroma_dir.exists():
        cl.add(
            "ChromaDB directory exists",
            False, True,
            f"Not found at {chroma_dir}. Run: python scripts/build_rag_corpus.py",
        )
        return

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            coll  = client.get_collection("arken_knowledge_v2")
            count = coll.count()
            passed = count >= 500
            cl.add(
                "ChromaDB has > 500 chunks",
                passed, True,
                f"{count:,} chunks in arken_knowledge_v2"
                + ("" if passed else " — run: python scripts/build_rag_corpus.py"),
            )
            # Domain coverage check
            if count > 0:
                try:
                    sample  = coll.get(limit=min(count, 500), include=["metadatas"])
                    domains = {m.get("domain") for m in (sample.get("metadatas") or []) if m}
                    cl.add(
                        "RAG corpus has multiple domains",
                        len(domains) >= 4, False,
                        f"{len(domains)} domains: {', '.join(sorted(d for d in domains if d))}",
                    )
                except Exception:
                    pass
        except Exception as exc:
            cl.add(
                "ChromaDB collection 'arken_knowledge_v2' exists",
                False, True,
                f"Collection not found: {exc}. Run: python scripts/build_rag_corpus.py",
            )
    except ImportError:
        cl.add(
            "ChromaDB installed",
            False, True,
            "chromadb not installed. Run: pip install chromadb>=0.5.0",
        )
    except Exception as exc:
        cl.add("ChromaDB accessible", False, True, f"Error: {exc}")


def check_redis(cl: Checklist) -> None:
    cl.add("__section__External Services", True, False, "")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis as redis_lib
        r = redis_lib.from_url(redis_url, socket_connect_timeout=3, socket_timeout=3)
        pong = r.ping()
        cl.add("Redis is reachable", bool(pong), False, f"URL: {redis_url.split('@')[-1]}")
    except ImportError:
        cl.add("Redis (redis-py installed)", False, False, "pip install redis — optional but recommended")
    except Exception as exc:
        cl.add(
            "Redis is reachable", False, False,
            f"Redis not reachable ({redis_url.split('@')[-1]}): {exc}. "
            "App will use in-memory fallback cache.",
        )


def check_postgresql(cl: Checklist) -> None:
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url or "localhost" in db_url:
        host_hint = "localhost"
    else:
        try:
            # e.g. postgresql+asyncpg://user:pass@host:5432/db
            host_hint = db_url.split("@")[-1].split("/")[0]
        except Exception:
            host_hint = "unknown"

    try:
        import psycopg2  # type: ignore
        # Convert asyncpg URL to psycopg2 URL
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        conn = psycopg2.connect(sync_url, connect_timeout=5)
        conn.close()
        cl.add("PostgreSQL is reachable", True, False, f"Host: {host_hint}")
    except ImportError:
        cl.add(
            "PostgreSQL (psycopg2 installed)",
            False, False,
            "psycopg2 not installed — install psycopg2-binary for connection check",
        )
    except Exception as exc:
        cl.add(
            "PostgreSQL is reachable",
            False, False,
            f"Not reachable ({host_hint}): {exc}. App may fail to start without DB.",
        )


def check_python_packages(cl: Checklist) -> None:
    cl.add("__section__Python Packages", True, False, "")

    required: List[Tuple[str, str, bool]] = [
        ("fastapi",            "fastapi",            True),
        ("uvicorn",            "uvicorn",            True),
        ("pydantic",           "pydantic",           True),
        ("pydantic-settings",  "pydantic_settings",  True),
        ("pandas",             "pandas",             True),
        ("numpy",              "numpy",              True),
        ("scikit-learn",       "sklearn",            True),
        ("joblib",             "joblib",             True),
        ("xgboost",            "xgboost",            False),
        ("chromadb",           "chromadb",           True),
        ("sentence-transformers","sentence_transformers", False),
        ("google-generativeai","google.generativeai", False),
        ("Pillow",             "PIL",                False),
        ("httpx",              "httpx",              False),
        ("redis",              "redis",              False),
        ("aiosqlite",          "aiosqlite",          False),
        ("sqlalchemy",         "sqlalchemy",         False),
        ("alembic",            "alembic",            False),
    ]

    for display, import_name, critical in required:
        spec = importlib.util.find_spec(import_name.split(".")[0])
        if spec is not None:
            try:
                mod = importlib.import_module(import_name.split(".")[0])
                ver = getattr(mod, "__version__", "?")
                cl.add(display, True, critical, f"v{ver}")
            except Exception:
                cl.add(display, True, critical, "installed")
        else:
            cl.add(
                display, False, critical,
                f"Not installed. Run: pip install {display}",
            )


def check_ml_model_ages(cl: Checklist) -> None:
    cl.add("__section__ML Model Freshness", True, False, "")

    weights_dir = _resolve("ml/weights", "ml/weights")
    report_path = weights_dir / "model_report.json"

    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
            training_date = report.get("training_date", "")
            mae           = report.get("mae")
            r2            = report.get("r2")
            dataset_size  = report.get("dataset_size", 0)

            if training_date:
                try:
                    from datetime import datetime
                    dt       = datetime.fromisoformat(training_date.replace("Z", "+00:00"))
                    days_old = (datetime.now(dt.tzinfo) - dt).days
                    fresh    = days_old <= 30
                    cl.add(
                        "ROI model trained within 30 days",
                        fresh, False,
                        f"Trained {days_old} days ago | MAE={mae} | R²={r2} | "
                        f"rows={dataset_size:,}" + ("" if fresh else " — retrain recommended"),
                    )
                except Exception as exc:
                    cl.add("ROI model training date parseable", False, False, str(exc))
            else:
                cl.add("model_report.json has training_date", False, False, "Field missing")

        except Exception as exc:
            cl.add("model_report.json is valid JSON", False, False, str(exc))
    else:
        cl.add(
            "model_report.json exists",
            False, False,
            f"Not found at {report_path}. Models need training first.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_checklist() -> Checklist:
    cl = Checklist()

    check_security(cl)
    check_datasets(cl)
    check_ml_models(cl)
    check_chromadb(cl)
    check_redis(cl)
    check_postgresql(cl)
    check_python_packages(cl)
    check_ml_model_ages(cl)

    return cl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARKEN production deployment pre-flight checklist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 if ANY check fails (including warnings), not just critical ones",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON instead of colour table",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour output",
    )
    args = parser.parse_args()

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    cl = run_checklist()

    if args.json:
        print(json.dumps(cl.to_dict(), indent=2))
    else:
        cl.print_table()

    if args.strict:
        return 0 if (cl.failed_critical == 0 and cl.warnings == 0) else 1
    return 0 if cl.all_critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
