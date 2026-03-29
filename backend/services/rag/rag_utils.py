"""
ARKEN — RAG Utilities v2.0
============================
SAVE AS: backend/services/rag/rag_utils.py — REPLACE existing

v2.0 Changes over v1.0 (PROBLEM 1 FIX):
  1. When building context for the chat agent, only chunks with
     source_quality = "india_specific" or "general_diy" are used.
     "filtered_us_content" chunks are never included in prompts.
  2. All context strings are prefixed with the Indian context header:
     "Context source: Indian renovation standards and practices"
  3. build_india_safe_context() helper added for explicit safe use.

All other API (warm_up_rag, get_rag_health, run_rag_diagnostics,
CachedRAGPipeline, get_cached_pipeline) UNCHANGED.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from services.rag.knowledge_loader import (
    SOURCE_QUALITY_FILTERED,
    SOURCE_QUALITY_GENERAL,
    SOURCE_QUALITY_INDIA,
    build_context_header,
)

logger = logging.getLogger(__name__)


# ── PROBLEM 1 FIX: Context safety filter ──────────────────────────────────────

def _is_safe_for_chat(source_quality: str) -> bool:
    """
    Return True if a chunk should be included in chat agent context.
    Only india_specific and general_diy chunks are safe.
    filtered_us_content is never surfaced to users.
    """
    return source_quality in (SOURCE_QUALITY_INDIA, SOURCE_QUALITY_GENERAL)


def build_india_safe_context(chunks: list, max_chars: int = 4000) -> str:
    """
    Build a context string from retrieved chunks, filtering out any
    US-specific content and prepending the Indian context header.

    Args:
        chunks: list of (KnowledgeDocument, score) tuples or KnowledgeDocument objects
        max_chars: maximum total context length

    Returns:
        context string safe to inject into chat agent prompts
    """
    # Handle both (doc, score) tuples and plain doc objects
    docs = []
    for item in chunks:
        if isinstance(item, tuple) and len(item) >= 2:
            doc, _ = item[0], item[1]
        else:
            doc = item
        sq = getattr(doc, "source_quality", SOURCE_QUALITY_INDIA)
        if _is_safe_for_chat(sq):
            docs.append(doc)

    if not docs:
        return build_context_header() + "No relevant Indian renovation knowledge found."

    parts = [build_context_header()]
    total = len(parts[0])

    for doc in docs:
        entry = f"[{doc.category.upper()}] {doc.title}\n{doc.content}\n\n"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "".join(parts)


# ── Warm-up (unchanged) ────────────────────────────────────────────────────────

def warm_up_rag(force_rebuild: bool = False) -> bool:
    """
    Pre-initialise FAISS index at startup.
    Returns True if successful, False if vector store unavailable.
    """
    try:
        from services.rag.vector_store import get_knowledge_store
        t0 = time.perf_counter()
        store = get_knowledge_store()
        if force_rebuild:
            store.initialise(force_rebuild=True)
        elapsed = round(time.perf_counter() - t0, 3)
        counts = store.get_category_counts()
        total = sum(counts.values())
        logger.info(
            f"[rag_utils] Warm-up complete in {elapsed}s — "
            f"{total} chunks across {len(counts)} domains: {counts}"
        )
        return True
    except Exception as e:
        logger.error(f"[rag_utils] Warm-up failed: {e}")
        return False


async def async_warm_up_rag() -> bool:
    """Async wrapper for warm_up_rag()."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, warm_up_rag)


# ── Health check (unchanged) ──────────────────────────────────────────────────

def get_rag_health() -> Dict[str, Any]:
    """Return a health status dict for /health or /startup_checks endpoints."""
    try:
        from services.rag.vector_store import get_knowledge_store
        store = get_knowledge_store()

        if not store.is_ready:
            return {"status": "unavailable", "error": "Store not initialised"}

        counts    = store.get_category_counts()
        total     = sum(counts.values())
        cache     = store.cache_stats()
        embedder  = type(store._embedder).__name__ if store._embedder else "unknown"
        hybrid    = store._bm25 is not None

        # Count by source quality
        from services.rag.knowledge_loader import load_all_documents
        all_docs = load_all_documents()
        india_count   = sum(1 for d in all_docs if d.source_quality == SOURCE_QUALITY_INDIA)
        general_count = sum(1 for d in all_docs if d.source_quality == SOURCE_QUALITY_GENERAL)
        filtered_count = sum(1 for d in all_docs if d.source_quality == SOURCE_QUALITY_FILTERED)

        return {
            "status":                  "ready" if total > 0 else "degraded",
            "doc_count":               total,
            "categories":              counts,
            "cache_entries":           cache.get("query_cache_entries", 0),
            "embedder":                embedder,
            "hybrid_search":           hybrid,
            "india_specific_chunks":   india_count,
            "general_diy_chunks":      general_count,
            "filtered_us_chunks":      filtered_count,
            "context_header":          build_context_header().strip(),
        }
    except Exception as e:
        logger.warning(f"[rag_utils] Health check failed: {e}")
        return {"status": "unavailable", "error": str(e)}


# ── Diagnostics (unchanged) ───────────────────────────────────────────────────

def run_rag_diagnostics(sample_queries: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run diagnostic queries and return quality metrics."""
    from services.rag.vector_store import get_knowledge_store

    default_queries = [
        "IS:732 Indian electrical wiring standard",
        "vitrified tile cost India Kajaria",
        "Asian Paints Royale emulsion Indian walls",
        "CPVC vs GI pipe Indian plumbing",
        "M20 concrete grade IS:456 India",
        "Gyproc false ceiling cost India",
        "modular kitchen BWP plywood India",
        "Vastu Shastra bedroom direction",
        "waterproofing Dr Fixit bathroom India",
        "MCB ELCB distribution board India",
    ]

    queries = sample_queries or default_queries
    store   = get_knowledge_store()
    results = []

    for query in queries:
        t0   = time.perf_counter()
        hits = store.search(query, k=3, hybrid=True)
        elapsed = round(time.perf_counter() - t0, 3)

        # PROBLEM 1 FIX: flag US content in diagnostics
        safe_hits = [
            (doc, score) for doc, score in hits
            if _is_safe_for_chat(getattr(doc, "source_quality", SOURCE_QUALITY_INDIA))
        ]

        results.append({
            "query":          query,
            "top_k": [
                {
                    "doc_id":        doc.doc_id,
                    "category":      doc.category,
                    "source_quality": getattr(doc, "source_quality", SOURCE_QUALITY_INDIA),
                    "score":         round(score, 4),
                }
                for doc, score in hits
            ],
            "safe_hit_count": len(safe_hits),
            "latency_s":      elapsed,
            "found":          len(safe_hits) > 0,
            "top_score":      round(hits[0][1], 4) if hits else 0.0,
        })

    hit_rate    = sum(1 for r in results if r["found"]) / max(len(results), 1)
    avg_latency = sum(r["latency_s"] for r in results) / max(len(results), 1)
    avg_top_score = sum(r["top_score"] for r in results) / max(len(results), 1)

    report = {
        "total_queries":   len(results),
        "hit_rate":        round(hit_rate, 3),
        "avg_latency_s":   round(avg_latency, 4),
        "avg_top_score":   round(avg_top_score, 4),
        "query_results":   results,
        "context_prefix":  build_context_header().strip(),
    }

    logger.info(
        f"[rag_utils] Diagnostics — hit_rate={hit_rate:.0%}, "
        f"avg_latency={avg_latency*1000:.1f}ms"
    )
    return report


# ── Cached RAG Pipeline ───────────────────────────────────────────────────────

class CachedRAGPipeline:
    """
    Pipeline wrapper that caches full RAG results per state fingerprint.

    v2.0: run_cached() filters out US-specific content and prepends
    the Indian context header to all returned context strings.
    """

    def __init__(self, ttl_seconds: int = 1800):
        self._ttl      = ttl_seconds
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from services.rag.context_builder import get_rag_pipeline
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    def _state_fingerprint(self, state: Dict[str, Any]) -> str:
        key_fields = {
            "project_id":  state.get("project_id", ""),
            "room_type":   state.get("room_type", ""),
            "city":        state.get("city", ""),
            "budget_tier": state.get("budget_tier", ""),
            "theme":       state.get("theme", ""),
            "intent":      state.get("user_intent", "")[:80],
            "condition":   (state.get("vision_features") or {}).get("room_condition", ""),
        }
        raw = json.dumps(key_fields, sort_keys=True).encode()
        return "rag_result:" + hashlib.md5(raw).hexdigest()

    async def run_cached(
        self,
        state: Dict[str, Any],
        extra_queries: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the RAG pipeline with caching.
        PROBLEM 1 FIX: context string is filtered and prefixed.
        """
        cache_key = self._state_fingerprint(state)

        # Try cache
        try:
            from services.cache import cache_service
            cached = await cache_service.get(cache_key)
            if cached and isinstance(cached, dict):
                logger.debug(f"[rag_utils] Cache HIT: {cache_key[:16]}")
                return cached
        except Exception as e:
            logger.debug(f"[rag_utils] Cache lookup failed (non-critical): {e}")

        # Cache miss — run pipeline
        result = self._get_pipeline().run(
            state, extra_queries=extra_queries, agent_type=agent_type
        )

        # PROBLEM 1 FIX: sanitise context before caching and returning
        raw_context = result.get("rag_context", "")
        if raw_context and not raw_context.startswith("Context source:"):
            result["rag_context"] = build_context_header() + raw_context

        # Strip US-filtered content from retrieved_chunks if present
        if "retrieved_chunks" in result:
            result["retrieved_chunks"] = [
                chunk for chunk in result["retrieved_chunks"]
                if _is_safe_for_chat(getattr(chunk, "source_quality", SOURCE_QUALITY_INDIA))
            ]

        cacheable = {
            "rag_context":      result.get("rag_context", ""),
            "rag_doc_ids":      result.get("rag_doc_ids", []),
            "rag_categories":   result.get("rag_categories", []),
            "rag_domain_stats": result.get("rag_domain_stats", {}),
        }

        try:
            from services.cache import cache_service
            await cache_service.set(cache_key, cacheable, ttl=self._ttl)
            logger.debug(f"[rag_utils] Cache SET: {cache_key[:16]} (TTL={self._ttl}s)")
        except Exception as e:
            logger.debug(f"[rag_utils] Cache store failed (non-critical): {e}")

        return result


# ── Category coverage reporter ────────────────────────────────────────────────

def report_knowledge_coverage() -> str:
    """Return a human-readable summary of loaded knowledge documents."""
    from services.rag.knowledge_loader import load_all_documents, ALL_CATEGORIES
    docs = load_all_documents(chunk=False)

    lines = [
        "ARKEN Knowledge Base Coverage (v3.0 — India-first):",
        f"  Total source documents: {len(docs)}",
        f"  India-specific: {sum(1 for d in docs if d.source_quality == SOURCE_QUALITY_INDIA)}",
        f"  General DIY:    {sum(1 for d in docs if d.source_quality == SOURCE_QUALITY_GENERAL)}",
        f"  Filtered (US):  {sum(1 for d in docs if d.source_quality == SOURCE_QUALITY_FILTERED)}",
        "",
    ]
    for cat in ALL_CATEGORIES:
        cat_docs = [d for d in docs if d.category == cat]
        if not cat_docs:
            continue
        tag_set = set()
        for d in cat_docs:
            tag_set.update(d.tags)
        lines.append(
            f"  {cat:<35} {len(cat_docs):>3} docs   tags: {', '.join(sorted(tag_set)[:6])}"
        )
    return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────

_cached_pipeline: Optional[CachedRAGPipeline] = None


def get_cached_pipeline() -> CachedRAGPipeline:
    """Return singleton CachedRAGPipeline."""
    global _cached_pipeline
    if _cached_pipeline is None:
        _cached_pipeline = CachedRAGPipeline(ttl_seconds=1800)
    return _cached_pipeline