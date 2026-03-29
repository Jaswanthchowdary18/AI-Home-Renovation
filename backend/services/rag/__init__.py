"""
ARKEN — RAG Package v2.0
=========================
Production-quality Retrieval-Augmented Generation system for renovation knowledge.

Architecture:
  knowledge_loader  — 6 domain knowledge categories, document chunking
  vector_store      — FAISS dense + BM25 keyword hybrid store with LRU cache
  retriever         — Domain-aware query routing, MMR diversity, re-ranking
  context_builder   — Structured prompt injection per agent type

Public API:
    from services.rag import (
        get_rag_pipeline,
        get_retriever,
        get_knowledge_store,
        KnowledgeCategory,
    )

    # Full pipeline (retrieval + context building)
    pipeline = get_rag_pipeline()
    result = pipeline.run(state)
    # result["rag_context"] → str, ready for LLM injection

    # Agent-targeted pipeline
    result = pipeline.run_for_agent(state, agent_type="budget_estimator")

    # Direct retrieval
    retriever = get_retriever()
    chunks = retriever.retrieve(state)

    # Agent-specific retrieval
    chunks = retriever.retrieve_for_agent("roi_predictor", state)

    # Direct knowledge store search (hybrid)
    store = get_knowledge_store()
    results = store.search("cracked interior wall repair India", k=5)

    # Category-specific search
    results = store.search_by_category("kitchen renovation cost", category="renovation_costs")
"""

from services.rag.context_builder import (
    get_rag_pipeline,
    RAGContextBuilder,
    RenovationRAGPipeline,
)
from services.rag.retriever import (
    get_retriever,
    RenovationRetriever,
    RetrievedChunk,
    RetrievalQuery,
)
from services.rag.vector_store import (
    get_knowledge_store,
    FAISSKnowledgeStore,
)
from services.rag.knowledge_loader import (
    load_all_documents,
    load_by_category,
    load_by_tags,
    get_category_stats,
    KnowledgeDocument,
    KnowledgeCategory,
)

__all__ = [
    # Pipeline
    "get_rag_pipeline",
    "RenovationRAGPipeline",
    # Retriever
    "get_retriever",
    "RenovationRetriever",
    "RetrievedChunk",
    "RetrievalQuery",
    # Vector store
    "get_knowledge_store",
    "FAISSKnowledgeStore",
    # Context builder
    "RAGContextBuilder",
    # Knowledge loader
    "load_all_documents",
    "load_by_category",
    "load_by_tags",
    "get_category_stats",
    "KnowledgeDocument",
    "KnowledgeCategory",
]