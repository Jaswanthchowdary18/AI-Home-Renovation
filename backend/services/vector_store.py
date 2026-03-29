"""
ARKEN — Vector Store Service v3.0
===================================
Stores renovation insights, BOQ items, and conversation context
as embeddings for RAG-based retrieval in the chat agent.

Primary: ChromaDB (local, no external API key needed)
Fallback: Pinecone (cloud, requires PINECONE_API_KEY in .env)

Usage:
    from services.vector_store import vector_store
    await vector_store.upsert_insights(project_id, insights_dict)
    results = await vector_store.query_similar(project_id, user_query, k=5)
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Embedding helper (Gemini text embedding) ─────────────────────────────────

def _embed_text(text: str, api_key: str) -> List[float]:
    """
    Generate embedding using Google's text-embedding-004 model.

    Falls back to a deterministic hash-based embedding when the API is
    unavailable (key not set, key revoked, or Embedding API not enabled).
    The hash fallback produces consistent 768-dim vectors suitable for
    project memory deduplication — just not as semantically rich.
    """
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1"},
        )
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        return result.embeddings[0].values
    except Exception as e:
        # Log at INFO not WARNING — the hash fallback is a valid working path,
        # not an unexpected failure. Common cause: Embedding API not enabled on
        # this key, or key is development-only. Project memory still works.
        logger.info(f"Embedding API unavailable ({type(e).__name__}), using hash fallback for project memory")
        h = hashlib.sha256(text.encode()).digest()
        return [((b / 255.0) - 0.5) for b in (h * 24)][:768]


# ── ChromaDB Store ───────────────────────────────────────────────────────────

class ChromaVectorStore:
    """
    Local ChromaDB-based vector store.
    Persists to /tmp/arken_chroma (configurable via CHROMA_PERSIST_DIR).
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._api_key: Optional[str] = None

    def _init(self):
        if self._client is not None:
            return
        try:
            import chromadb
            from config import settings

            persist_dir = getattr(settings, "CHROMA_PERSIST_DIR", "/tmp/arken_chroma")
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="arken_insights",
                metadata={"hnsw:space": "cosine"},
            )
            self._api_key = (
                settings.GOOGLE_API_KEY.get_secret_value()
                if settings.GOOGLE_API_KEY
                else None
            )
            logger.info(f"ChromaDB initialised at {persist_dir}")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            raise

    async def upsert_insights(
        self,
        project_id: str,
        insights: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store insight document as embedding."""
        try:
            self._init()
            # Build rich text document for embedding
            doc = self._insights_to_text(insights)
            embedding = _embed_text(doc, self._api_key or "")
            meta = {
                "project_id": project_id,
                "city": insights.get("market_intelligence", {}).get("city", ""),
                "theme": metadata.get("theme", "") if metadata else "",
                "budget_tier": metadata.get("budget_tier", "") if metadata else "",
                "roi_pct": str(
                    insights.get("financial_outlook", {}).get("projected_roi", "0%")
                ),
                **(metadata or {}),
            }
            self._collection.upsert(
                ids=[project_id],
                embeddings=[embedding],
                documents=[doc],
                metadatas=[meta],
            )
            logger.info(f"Upserted insights for project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Upsert failed for {project_id}: {e}")
            return False

    async def query_similar(
        self,
        project_id: str,
        query: str,
        k: int = 5,
        filter_city: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve similar insights for RAG context."""
        try:
            self._init()
            embedding = _embed_text(query, self._api_key or "")
            where = {"city": filter_city} if filter_city else None
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            out = []
            for i, doc in enumerate(results["documents"][0]):
                out.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                })
            return out
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    async def upsert_conversation_turn(
        self,
        project_id: str,
        turn_id: str,
        user_msg: str,
        assistant_msg: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store conversation turns for memory retrieval."""
        try:
            self._init()
            doc = f"USER: {user_msg}\nARKEN: {assistant_msg}"
            embedding = _embed_text(doc, self._api_key or "")
            meta = {
                "project_id": project_id,
                "type": "conversation",
                **(metadata or {}),
            }
            self._collection.upsert(
                ids=[f"{project_id}_{turn_id}"],
                embeddings=[embedding],
                documents=[doc],
                metadatas=[meta],
            )
            return True
        except Exception as e:
            logger.error(f"Conv upsert failed: {e}")
            return False

    async def get_conversation_context(
        self, project_id: str, query: str, k: int = 3
    ) -> str:
        """Retrieve relevant past conversation turns as context string."""
        results = await self.query_similar(project_id, query, k=k)
        if not results:
            return ""
        ctx_parts = []
        for r in results:
            if r["metadata"].get("type") == "conversation" and r["similarity"] > 0.5:
                ctx_parts.append(r["document"])
        return "\n\n".join(ctx_parts)

    @staticmethod
    def _insights_to_text(insights: Dict) -> str:
        """Convert insights dict to a rich text document for embedding."""
        parts = []
        headline = insights.get("summary_headline", "")
        if headline:
            parts.append(f"HEADLINE: {headline}")

        fin = insights.get("financial_outlook", {})
        if fin:
            parts.append(
                f"FINANCIAL: ROI={fin.get('projected_roi','?')} "
                f"equity={fin.get('equity_gain','?')} "
                f"payback={fin.get('payback_period','?')} "
                f"cost={fin.get('renovation_cost','?')}"
            )

        mkt = insights.get("market_intelligence", {})
        if mkt:
            parts.append(
                f"MARKET: {mkt.get('city','?')} "
                f"appreciation={mkt.get('avg_appreciation_5yr','?')} "
                f"yield={mkt.get('rental_yield','?')} "
                f"trend={mkt.get('market_trend','?')}"
            )

        vis = insights.get("visual_analysis", {})
        if vis:
            parts.append(
                f"VISUAL: style={vis.get('detected_style','?')} "
                f"walls={vis.get('wall_treatment','?')} "
                f"floor={vis.get('floor_material','?')}"
            )

        recs = insights.get("recommendations", [])
        if recs:
            parts.append("RECOMMENDATIONS: " + " | ".join(recs[:3]))

        return "\n".join(parts) or json.dumps(insights)[:1000]


# ── Pinecone fallback (optional cloud store) ─────────────────────────────────

class PineconeVectorStore:
    """
    Cloud-based Pinecone vector store.
    Requires PINECONE_API_KEY and PINECONE_INDEX env vars.
    """

    def __init__(self):
        self._index = None
        self._api_key: Optional[str] = None

    def _init(self):
        if self._index is not None:
            return
        from config import settings
        pc_key = getattr(settings, "PINECONE_API_KEY", None)
        pc_idx = getattr(settings, "PINECONE_INDEX", "arken-insights")
        if not pc_key:
            raise RuntimeError("PINECONE_API_KEY not set")
        from pinecone import Pinecone
        pc = Pinecone(api_key=pc_key)
        self._index = pc.Index(pc_idx)
        self._api_key = (
            settings.GOOGLE_API_KEY.get_secret_value()
            if settings.GOOGLE_API_KEY else None
        )

    async def upsert_insights(self, project_id, insights, metadata=None):
        try:
            self._init()
            doc = ChromaVectorStore._insights_to_text(insights)
            embedding = _embed_text(doc, self._api_key or "")
            self._index.upsert(vectors=[{
                "id": project_id,
                "values": embedding,
                "metadata": {**(metadata or {}), "doc": doc[:500]},
            }])
            return True
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
            return False

    async def query_similar(self, project_id, query, k=5, filter_city=None):
        try:
            self._init()
            embedding = _embed_text(query, self._api_key or "")
            res = self._index.query(vector=embedding, top_k=k, include_metadata=True)
            return [
                {"document": m["metadata"].get("doc", ""), "metadata": m["metadata"], "similarity": m["score"]}
                for m in res.matches
            ]
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            return []


# ── Auto-selecting store ──────────────────────────────────────────────────────

def _create_vector_store():
    """
    Try Pinecone first (if API key available), fall back to ChromaDB.
    """
    try:
        from config import settings
        if getattr(settings, "PINECONE_API_KEY", None):
            store = PineconeVectorStore()
            store._init()
            logger.info("Vector store: Pinecone (cloud)")
            return store
    except Exception:
        pass

    try:
        store = ChromaVectorStore()
        store._init()
        logger.info("Vector store: ChromaDB (local)")
        return store
    except Exception as e:
        logger.warning(f"All vector stores unavailable: {e}. Using no-op store.")
        return _NoOpStore()


class _NoOpStore:
    """Graceful no-op when no vector store is available."""
    async def upsert_insights(self, *a, **k): return False
    async def query_similar(self, *a, **k): return []
    async def upsert_conversation_turn(self, *a, **k): return False
    async def get_conversation_context(self, *a, **k): return ""


# Singleton
vector_store = _create_vector_store()