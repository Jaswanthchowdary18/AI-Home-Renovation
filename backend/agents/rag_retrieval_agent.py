"""
ARKEN — RAG Knowledge Retrieval Agent Node v2.1
================================================
LangGraph node performing structured knowledge retrieval AFTER vision analysis
and BEFORE design planning.

Position in DAG:
  node_vision_analysis → [THIS NODE] → node_design_planning

What's upgraded in v2.1:
  - Uses the new 6-domain KnowledgeCategory system
  - Calls retrieve_for_agent() for agent-targeted sub-retrievals
  - Injects rag_budget_context, rag_design_context, rag_roi_context
    as separate state keys (one per downstream agent)
  - Serialises RetrievedChunk.document.title (new field in v2.0)
  - rag_retrieval_stats includes domain_breakdown dict

CRITICAL: Does NOT call any Gemini/vision/image-generation APIs.
          Only consumes upstream agent outputs via state dict.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RAGRetrievalAgent:
    """
    LangGraph node: vision outputs → domain knowledge retrieval → structured context.
    """

    def __init__(self):
        # v2.2: Auto-seed knowledge base if ChromaDB is empty
        self.ensure_knowledge_seeded()

    def ensure_knowledge_seeded(self) -> None:
        """
        Check if ChromaDB has been seeded with renovation knowledge.
        If the collection has fewer than 50 chunks, automatically trigger seeding.

        This is a non-blocking best-effort operation — any failure is logged
        as a warning and does NOT prevent the agent from running.
        """
        try:
            from services.rag.retriever import get_retriever
            retriever = get_retriever()
            count = 0
            if hasattr(retriever, "collection") and retriever.collection is not None:
                count = retriever.collection.count()
            if count < 50:
                logger.warning(
                    f"[RAG] ChromaDB has only {count} chunks — auto-seeding knowledge base..."
                )
                # Use importlib to load seed_knowledge.py by file path.
                # `from data.rag_knowledge_base.seed_knowledge import ...` fails
                # because data/ and data/rag_knowledge_base/ have no __init__.py,
                # so Python cannot treat them as packages. Loading by path bypasses this.
                import importlib.util as _ilu
                import os as _os
                _seed_path = _os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    "data", "rag_knowledge_base", "seed_knowledge.py",
                )
                if _os.path.exists(_seed_path):
                    _spec = _ilu.spec_from_file_location("seed_knowledge", _seed_path)
                    _mod  = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    persist_dir = _os.getenv("CHROMA_PERSIST_DIR", "/tmp/arken_chroma")
                    seeded = _mod.seed_chromadb(persist_dir)
                    logger.info(
                        f"[RAG] Knowledge base seeded: {seeded} chunks at '{persist_dir}'."
                    )
                else:
                    logger.warning(
                        f"[RAG] seed_knowledge.py not found at '{_seed_path}'. "
                        "RAG will use knowledge_loader fallback instead."
                    )
            else:
                logger.info(f"[RAG] ChromaDB ready with {count} knowledge chunks.")
        except Exception as e:
            logger.warning(
                f"[RAG] Could not verify/seed knowledge base: {e}. "
                "RAG will proceed with whatever chunks are available."
            )

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        name = "rag_retrieval_agent"
        logger.info(f"[{state.get('project_id', '')}] RAGRetrievalAgent v2.1 starting")

        try:
            updates = self._retrieve(state)
        except Exception as e:
            logger.warning(f"[rag_retrieval] Retrieval failed ({e}) — using empty context")
            updates = self._fallback_context(state, str(e))

        elapsed = round(time.perf_counter() - t0, 3)

        # Merge timings + completion
        timings = dict(state.get("agent_timings") or {})
        timings[name] = elapsed
        updates["agent_timings"] = timings

        completed = list(state.get("completed_agents") or [])
        if name not in completed:
            completed.append(name)
        updates["completed_agents"] = completed

        stats = updates.get("rag_retrieval_stats", {})
        logger.info(
            f"[rag_retrieval] done in {elapsed}s — "
            f"docs={stats.get('doc_count', 0)} "
            f"queries={stats.get('query_count', 0)} "
            f"top_score={stats.get('top_score', 0.0):.3f} "
            f"domains={stats.get('domain_breakdown', {})}"
        )
        return updates

    # ─────────────────────────────────────────────────────────────────────────
    # Core retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from services.rag.retriever import get_retriever
        from services.rag.context_builder import RAGContextBuilder, get_rag_pipeline

        pipeline = get_rag_pipeline()
        retriever = get_retriever()
        builder = RAGContextBuilder()

        # Extra query from user intent
        extra_queries: List[str] = []
        intent = (state.get("user_intent") or "").strip()
        if intent:
            extra_queries.append(intent)

        # ── Full retrieval (all domains) ──────────────────────────────────────
        result = pipeline.run(state, extra_queries=extra_queries or None)
        chunks = result.get("rag_chunks", [])

        # ── Agent-specific context strings ────────────────────────────────────
        # Each downstream node can pick the context most relevant to it
        budget_result = pipeline.run_for_agent(state, agent_type="budget_estimator")
        design_result = pipeline.run_for_agent(state, agent_type="design_planner")
        roi_result    = pipeline.run_for_agent(state, agent_type="roi_predictor")

        # ── Serialise chunks for state (JSON-serialisable) ────────────────────
        retrieved_knowledge: List[Dict[str, Any]] = []
        rag_sources: List[str] = []
        top_score = 0.0
        domain_breakdown: Dict[str, int] = {}

        for chunk in chunks:
            doc = chunk.document
            retrieved_knowledge.append({
                "doc_id":      doc.doc_id,
                "title":       doc.title,
                "category":    doc.category,
                "subcategory": doc.subcategory,
                "text":        doc.content[:600],
                "score":       chunk.score,
                "rank":        chunk.rank,
                "query":       chunk.query_used[:80] if chunk.query_used else "",
                "method":      chunk.retrieval_method,
            })
            rag_sources.append(doc.doc_id)
            if chunk.score > top_score:
                top_score = chunk.score
            domain_breakdown[doc.category] = domain_breakdown.get(doc.category, 0) + 1

        return {
            "retrieved_knowledge":  retrieved_knowledge,
            "rag_context":          result.get("rag_context", ""),
            "rag_sources":          list(dict.fromkeys(rag_sources)),
            # Agent-specific contexts
            "rag_budget_context":   budget_result.get("rag_context", ""),
            "rag_design_context":   design_result.get("rag_context", ""),
            "rag_roi_context":      roi_result.get("rag_context", ""),
            # Stats
            "rag_retrieval_stats": {
                "doc_count":        len(retrieved_knowledge),
                "query_count":      len(chunks),
                "top_score":        round(top_score, 4),
                "domain_breakdown": domain_breakdown,
                "has_repair":       any(
                    d["category"] in ("construction_materials", "repair")
                    for d in retrieved_knowledge
                ),
                "has_cost":         any(
                    d["category"] in ("renovation_costs", "cost")
                    for d in retrieved_knowledge
                ),
                "has_case":         any(
                    d["category"] in ("real_estate_value_factors", "case_study")
                    for d in retrieved_knowledge
                ),
                "has_design":       any(
                    d["category"] in ("interior_design",)
                    for d in retrieved_knowledge
                ),
                "has_space_rules":  any(
                    d["category"] in ("space_planning_rules",)
                    for d in retrieved_knowledge
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_context(self, state: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Return minimal context so downstream nodes are never blocked."""
        room  = state.get("room_type", "bedroom")
        city  = state.get("city", "Hyderabad")
        tier  = state.get("budget_tier", "mid")
        theme = state.get("theme", "Modern Minimalist")

        rag_context = (
            f"=== ARKEN KNOWLEDGE BASE CONTEXT ===\n"
            f"Knowledge retrieval unavailable (error: {error}).\n"
            f"Proceeding with heuristic estimates for {room} renovation "
            f"in {city} ({tier} budget, {theme} style).\n"
            f"Standard cost assumptions:\n"
            f"  - Vitrified tile flooring: ₹85–180/sqft\n"
            f"  - Interior wall paint: ₹8,000–15,000 per room\n"
            f"  - False ceiling (POP/gypsum): ₹65–120/sqft\n"
            f"  - LED lighting package: ₹15,000–40,000\n"
            f"  - Labour: 30–35% of materials cost\n"
        )

        errors = list(state.get("errors") or [])
        errors.append(f"rag_retrieval: {error}")

        return {
            "retrieved_knowledge":  [],
            "rag_context":          rag_context,
            "rag_sources":          [],
            "rag_budget_context":   rag_context,
            "rag_design_context":   rag_context,
            "rag_roi_context":      rag_context,
            "rag_retrieval_stats": {
                "doc_count":        0,
                "query_count":      0,
                "top_score":        0.0,
                "domain_breakdown": {},
                "has_repair":       False,
                "has_cost":         False,
                "has_case":         False,
                "has_design":       False,
                "has_space_rules":  False,
            },
            "errors": errors,
        }


# ── LangGraph-compatible sync node function ───────────────────────────────────

def node_rag_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function (sync wrapper).
    Inserted between node_vision_analysis and node_design_planning in the graph.
    """
    agent = RAGRetrievalAgent()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, agent.run(dict(state)))
                result = future.result(timeout=90)
        else:
            result = loop.run_until_complete(agent.run(dict(state)))
    except RuntimeError:
        result = asyncio.run(agent.run(dict(state)))

    return {**state, **result}