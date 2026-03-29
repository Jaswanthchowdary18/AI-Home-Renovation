"""
ARKEN — Dataset RAG Ingestion Script
======================================
Ingests the DIY renovation knowledge base into the ARKEN RAG vector store.

Run once after placing datasets (see README_DATASETS.md):
    python -m services.datasets.ingest_datasets

What this does:
  1. Loads DIY_dataset.csv → 1,066 knowledge chunks
  2. Embeds each chunk using sentence-transformers (all-MiniLM-L6-v2)
  3. Inserts into FAISS/ChromaDB vector store used by RAG retriever
  4. Chunks are retrievable by design_planner_node, budget_estimator_agent, roi_agent_node

The DIY dataset covers:
  - Walls and Ceilings (358 chunks)
  - Plumbing (229 chunks)
  - Doors (158 chunks)
  - Basements (109 chunks)
  - Power Tools (90 chunks)
  - Lighting (40 chunks)
  - Mechanical, Electrical, Toilets (remaining)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def ingest_diy_dataset():
    """Load DIY chunks and insert into the RAG vector store."""
    from services.datasets.dataset_loader import ARKENDatasetRegistry

    registry = ARKENDatasetRegistry.get()

    if not registry.diy_renovation.available:
        logger.error(
            "[ingest] DIY dataset not found. "
            "Place DIY_dataset.csv in backend/data/datasets/diy_renovation/"
        )
        return 0

    docs = registry.diy_renovation.to_rag_documents()
    logger.info(f"[ingest] Loaded {len(docs)} DIY knowledge chunks")

    # Try ARKEN's existing RAG infrastructure
    try:
        from services.rag.knowledge_loader import KnowledgeLoader
        loader = KnowledgeLoader()
        inserted = loader.ingest_documents(docs)
        logger.info(f"[ingest] ✓ Inserted {inserted} chunks via KnowledgeLoader")
        return inserted
    except Exception as e:
        logger.warning(f"[ingest] KnowledgeLoader failed: {e} — trying vector_store directly")

    try:
        from services.rag.vector_store import VectorStore
        vs = VectorStore()
        texts = [d["text"] for d in docs]
        metadatas = [d["metadata"] for d in docs]
        ids = [d["id"] for d in docs]
        vs.add_documents(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"[ingest] ✓ Inserted {len(docs)} chunks via VectorStore")
        return len(docs)
    except Exception as e:
        logger.error(f"[ingest] Vector store insertion failed: {e}")
        return 0


def ingest_style_metadata():
    """
    Index interior image metadata for style-based retrieval.
    Creates a lightweight JSON index for CLIP similarity lookups.
    """
    from services.datasets.dataset_loader import ARKENDatasetRegistry
    import json

    registry = ARKENDatasetRegistry.get()
    all_records = (
        registry.material_style.load() +
        registry.interior_images.load()
    )

    if not all_records:
        logger.warning("[ingest] No interior image records found")
        return 0

    index = []
    for rec in all_records:
        index.append({
            "image_path": str(rec.image_path),
            "room_type": rec.room_type,
            "style": rec.style,
            "materials": rec.materials,
            "objects": rec.objects,
            "source": rec.source_dataset,
        })

    index_path = Path("/app/data/datasets/style_index.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"[ingest] ✓ Style index written: {len(index)} records → {index_path}")
    return len(index)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("=== ARKEN Dataset Ingestion ===")

    diy_count = ingest_diy_dataset()
    logger.info(f"DIY knowledge chunks ingested: {diy_count}")

    style_count = ingest_style_metadata()
    logger.info(f"Style metadata indexed: {style_count}")

    logger.info("=== Ingestion complete ===")
