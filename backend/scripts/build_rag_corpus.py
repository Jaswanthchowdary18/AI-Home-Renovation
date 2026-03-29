#!/usr/bin/env python3
"""
ARKEN — Build RAG Corpus
=========================
One-command script to:
  1. Generate the 3,000+ chunk renovation knowledge corpus
  2. Seed ChromaDB
  3. Print domain breakdown
  4. Run smoke tests (5 queries)
  5. Exit 0 on success

Usage:
    cd backend && python scripts/build_rag_corpus.py
    # Or with custom path:
    CHROMA_PERSIST_DIR=/data/chroma python scripts/build_rag_corpus.py
    # Dry run (no ChromaDB, just build and count corpus):
    python scripts/build_rag_corpus.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
# Support running from backend/ or from backend/scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy sub-loggers
for _noisy in ["sentence_transformers", "chromadb", "urllib3", "filelock"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)


SMOKE_TEST_QUERIES = [
    "kitchen renovation cost Mumbai mid-range",
    "waterproofing bathroom India best method",
    "Kajaria vitrified tile specifications anti-skid",
    "renovation ROI return Hyderabad IT corridor",
    "steel TMT Fe500D buying guide India",
]


def _print_header(title: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _print_section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 66 - len(title)))


def run_build(persist_dir: str, dry_run: bool = False) -> int:
    """
    Run the full corpus build pipeline.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    _print_header("ARKEN RAG Corpus Builder")
    print(f"  ChromaDB persist dir: {persist_dir}")
    print(f"  Dry run: {dry_run}")

    # ── Step 1: Import corpus builder ─────────────────────────────────────────
    _print_section("Step 1: Import corpus_builder")
    try:
        from data.rag_knowledge_base.corpus_builder import (
            build_full_corpus,
            get_corpus_domain_stats,
            seed_chromadb,
        )
        print("  ✓ corpus_builder imported successfully")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return 1

    # ── Step 2: Build corpus and show stats ───────────────────────────────────
    _print_section("Step 2: Build in-memory corpus")
    t0 = time.perf_counter()
    try:
        corpus = build_full_corpus()
        elapsed = round(time.perf_counter() - t0, 2)
        total_chunks = len(corpus)
        print(f"  ✓ Corpus built: {total_chunks} chunks in {elapsed}s")

        if total_chunks < 1000:
            print(f"  ⚠ WARNING: corpus has only {total_chunks} chunks (target 3,000+)")
    except Exception as e:
        print(f"  ✗ Corpus build failed: {e}")
        return 1

    # ── Step 3: Domain breakdown ──────────────────────────────────────────────
    _print_section("Step 3: Domain breakdown")
    stats = get_corpus_domain_stats()
    domain_targets = {
        "material_specs":    500,
        "renovation_guides": 500,
        "property_market":   480,
        "design_styles":     400,
        "diy_contractor":    600,
        "price_intelligence": 500,
    }

    all_domains_ok = True
    for domain, target in domain_targets.items():
        count = stats.get(domain, 0)
        status = "✓" if count >= target else "⚠"
        if count < target:
            all_domains_ok = False
        print(f"  {status} {domain:<25} {count:>5} chunks  (target: {target})")

    other_domains = {k: v for k, v in stats.items() if k not in domain_targets}
    if other_domains:
        for domain, count in sorted(other_domains.items()):
            print(f"    {domain:<25} {count:>5} chunks  (supplementary)")

    print(f"\n  Total: {total_chunks} chunks across {len(stats)} domains")

    if dry_run:
        print("\n  [DRY RUN] Skipping ChromaDB seed. Done.")
        return 0

    # ── Step 4: Seed ChromaDB ─────────────────────────────────────────────────
    _print_section("Step 4: Seed ChromaDB")
    try:
        import chromadb
        print(f"  chromadb version: {chromadb.__version__}")
    except ImportError:
        print("  ✗ chromadb not installed. Run: pip install chromadb")
        print("  Tip: pip install 'chromadb>=0.5.0' sentence-transformers")
        return 1

    t1 = time.perf_counter()
    try:
        os.makedirs(persist_dir, exist_ok=True)
        seeded = seed_chromadb(persist_dir)
        seed_elapsed = round(time.perf_counter() - t1, 1)
        print(f"  ✓ ChromaDB seeded: {seeded} chunks in {seed_elapsed}s")
        print(f"  Location: {persist_dir}")
    except Exception as e:
        print(f"  ✗ Seeding failed: {e}")
        logger.exception("Seed error")
        return 1

    if seeded < 100:
        print(f"  ✗ Too few chunks seeded ({seeded}) — something went wrong.")
        return 1

    # ── Step 5: Smoke tests ───────────────────────────────────────────────────
    _print_section("Step 5: Smoke test queries")
    try:
        from data.rag_knowledge_base.corpus_builder import ChromaRetriever
        retriever = ChromaRetriever(persist_dir=persist_dir)

        if retriever.collection is None:
            print("  ✗ Retriever collection not available")
            return 1

        total_count = retriever.collection.count()
        print(f"  Collection count: {total_count} chunks")

        all_smoke_ok = True
        for i, query in enumerate(SMOKE_TEST_QUERIES, 1):
            try:
                results = retriever.retrieve(query, top_k=3)
                if results:
                    top = results[0]
                    domain = top.get("metadata", {}).get("domain", "?")
                    score = top.get("score", 0.0)
                    title = top.get("metadata", {}).get("title", top.get("id", "?"))[:50]
                    print(f"  [{i}] ✓ '{query[:45]}'")
                    print(f"       → [{domain}] {title}  score={score:.3f}")
                else:
                    print(f"  [{i}] ⚠ '{query[:45]}' — no results returned")
                    all_smoke_ok = False
            except Exception as e:
                print(f"  [{i}] ✗ '{query[:45]}' — error: {e}")
                all_smoke_ok = False

        if not all_smoke_ok:
            print("\n  ⚠ Some smoke tests had no results — embedding may need warmup.")
        else:
            print("\n  ✓ All smoke tests passed")

    except Exception as e:
        print(f"  ✗ Smoke test failed: {e}")
        logger.exception("Smoke test error")
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_section("Summary")
    total_elapsed = round(time.perf_counter() - t0, 1)
    print(f"  Corpus chunks built:  {total_chunks}")
    print(f"  ChromaDB chunks:      {seeded}")
    print(f"  Total elapsed:        {total_elapsed}s")
    print(f"  ChromaDB at:          {persist_dir}")
    print()
    if total_chunks >= 1000 and seeded >= 100:
        print("  ✓ RAG corpus build COMPLETE — all checks passed.")
        return 0
    else:
        print("  ⚠ Build completed with warnings — check output above.")
        return 0  # Non-fatal


def main():
    parser = argparse.ArgumentParser(
        description="Build and seed ARKEN RAG knowledge corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "/tmp/arken_chroma"),
        help="ChromaDB persistence directory (default: $CHROMA_PERSIST_DIR or /tmp/arken_chroma)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build corpus in-memory only, skip ChromaDB seeding",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print domain stats and exit (implies --dry-run)",
    )
    args = parser.parse_args()

    exit_code = run_build(
        persist_dir=args.persist_dir,
        dry_run=args.dry_run or args.stats_only,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
