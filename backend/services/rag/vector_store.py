"""
ARKEN — RAG Vector Store v2.0
================================
Production-grade knowledge store with:

  - FAISS dense vector search (primary)
  - BM25 keyword retrieval (secondary — hybrid fusion)
  - Reciprocal Rank Fusion (RRF) for hybrid result merging
  - In-memory query result caching (LRU + TTL)
  - Category-filtered sub-index views for faster domain search
  - Thread-safe singleton with lazy init
  - Disk persistence with content-hash invalidation

CRITICAL: This module is entirely independent of image generation,
Gemini vision pipelines, and rendering — it only processes text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.rag.knowledge_loader import KnowledgeDocument, load_all_documents

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PERSIST_DIR = Path(os.environ.get("RAG_PERSIST_DIR", "/tmp/arken_rag"))
INDEX_FILE = PERSIST_DIR / "faiss_index_v2.bin"
DOCSTORE_FILE = PERSIST_DIR / "docstore_v2.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CACHE_VERSION = "2.0"
QUERY_CACHE_SIZE = 256       # LRU entries
QUERY_CACHE_TTL = 3600       # seconds


# ── LRU Query Cache ───────────────────────────────────────────────────────────

class _LRUQueryCache:
    """Thread-safe LRU cache for query results with TTL expiry."""

    def __init__(self, maxsize: int = QUERY_CACHE_SIZE, ttl: int = QUERY_CACHE_TTL):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _cache_key(self, query: str, k: int, category: Optional[str], hybrid: bool) -> str:
        raw = f"{query}|{k}|{category or ''}|{int(hybrid)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, k: int, category: Optional[str], hybrid: bool) -> Optional[Any]:
        key = self._cache_key(query, k, category, hybrid)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, ts = entry
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def put(self, query: str, k: int, category: Optional[str], hybrid: bool, value: Any) -> None:
        key = self._cache_key(query, k, category, hybrid)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.monotonic())
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ── Embedding Backends ────────────────────────────────────────────────────────

class SentenceTransformerEmbedder:
    """Primary embedder — sentence-transformers (local, no API key)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"[vector_store] SentenceTransformer loaded: {model_name} (dim={self.dim})")

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(
            self._model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


class TFIDFEmbedder:
    """Fallback embedder — TF-IDF + SVD when sentence-transformers unavailable."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._vectorizer = None
        self._svd = None
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        tfidf = self._vectorizer.fit_transform(texts)
        actual_dim = min(self.dim, tfidf.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        self._svd.fit(tfidf)
        self.dim = actual_dim
        self._fitted = True
        logger.info(f"[vector_store] TF-IDF/SVD embedder fitted: dim={self.dim}")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedder.fit() must be called before embed()")
        tfidf = self._vectorizer.transform(texts)
        return np.array(self._svd.transform(tfidf), dtype=np.float32)


def _build_embedder() -> Any:
    try:
        return SentenceTransformerEmbedder(EMBEDDING_MODEL)
    except ImportError:
        logger.warning("[vector_store] sentence-transformers not installed — using TF-IDF fallback")
        return TFIDFEmbedder(dim=256)


# ── BM25 Keyword Index ────────────────────────────────────────────────────────

class BM25Index:
    """
    BM25 keyword search index.
    Falls back to simple TF-IDF scoring when rank_bm25 is not installed.
    """

    def __init__(self, documents: List[KnowledgeDocument]):
        self._docs = documents
        self._corpus = [_doc_to_keywords(doc) for doc in documents]
        self._bm25 = None
        self._tfidf_matrix = None
        self._tfidf_vocab: Dict[str, int] = {}
        self._build()

    def _build(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            tokenised = [text.lower().split() for text in self._corpus]
            self._bm25 = BM25Okapi(tokenised, k1=1.5, b=0.75)
            logger.info("[vector_store] BM25 keyword index built (rank_bm25)")
        except ImportError:
            # Fallback: simple TF scoring
            self._bm25 = None
            self._build_tfidf_fallback()

    def _build_tfidf_fallback(self) -> None:
        """Build a minimal in-memory TF-based index as fallback."""
        vocab: Dict[str, int] = {}
        for text in self._corpus:
            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        self._tfidf_vocab = vocab
        n_docs = len(self._corpus)
        n_vocab = len(vocab)
        matrix = np.zeros((n_docs, n_vocab), dtype=np.float32)
        for i, text in enumerate(self._corpus):
            words = text.lower().split()
            for word in words:
                if word in vocab:
                    matrix[i, vocab[word]] += 1.0
            # Normalise by document length
            row_sum = matrix[i].sum()
            if row_sum > 0:
                matrix[i] /= row_sum
        self._tfidf_matrix = matrix
        logger.info("[vector_store] TF keyword fallback index built")

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Returns list of (doc_index, score) sorted by descending score.
        """
        if self._bm25 is not None:
            query_tokens = query.lower().split()
            scores = self._bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:k * 2]
            return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
        else:
            return self._tfidf_search(query, k)

    def _tfidf_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self._tfidf_matrix is None:
            return []
        query_vec = np.zeros(len(self._tfidf_vocab), dtype=np.float32)
        for word in query.lower().split():
            if word in self._tfidf_vocab:
                query_vec[self._tfidf_vocab[word]] += 1.0
        if query_vec.sum() == 0:
            return []
        query_vec /= query_vec.sum()
        scores = self._tfidf_matrix @ query_vec
        top_indices = np.argsort(scores)[::-1][:k * 2]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def _doc_to_keywords(doc: KnowledgeDocument) -> str:
    """Build keyword-rich string for BM25 indexing."""
    return " ".join([
        doc.content,
        f"category {doc.category}",
        f"subcategory {doc.subcategory}",
        " ".join(doc.tags),
        doc.title,
        doc.metadata.get("city", ""),
        " ".join(doc.metadata.get("room_types", [])),
    ])


# ── NumPy fallback FAISS index ────────────────────────────────────────────────

class NumpyFallbackIndex:
    """Pure NumPy cosine similarity index when FAISS is unavailable."""

    def __init__(self, embeddings: np.ndarray):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        self._embeddings = embeddings / norms

    def search(self, query: np.ndarray, k: int):
        norms = np.linalg.norm(query, axis=1, keepdims=True) + 1e-10
        q = query / norms
        scores = (q @ self._embeddings.T).flatten()
        top_k = np.argsort(scores)[::-1][:k]
        return scores[top_k].reshape(1, -1), top_k.reshape(1, -1)


# ── FAISS Knowledge Store ─────────────────────────────────────────────────────

class FAISSKnowledgeStore:
    """
    Production FAISS-backed knowledge store.

    Features:
    - Dense vector search (primary)
    - BM25 keyword search (secondary)
    - Reciprocal Rank Fusion for hybrid merging
    - LRU query result cache
    - Category sub-index maps for O(1) filtered retrieval
    - Thread-safe lazy initialisation + disk persistence
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._index = None
        self._documents: List[KnowledgeDocument] = []
        self._embedder = None
        self._bm25: Optional[BM25Index] = None
        self._category_index: Dict[str, List[int]] = {}   # category → [doc_positions]
        self._query_cache = _LRUQueryCache()
        self._initialized = False

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialise(self, force_rebuild: bool = False) -> None:
        """Build or load FAISS index. Thread-safe, idempotent."""
        with self._lock:
            if self._initialized and not force_rebuild:
                return
            t0 = time.perf_counter()
            documents = load_all_documents(chunk=True)
            cache_hash = _compute_docs_hash(documents)

            if not force_rebuild and self._try_load_from_disk(cache_hash):
                self._initialized = True
                elapsed = round(time.perf_counter() - t0, 3)
                logger.info(f"[vector_store] Loaded from disk cache in {elapsed}s ({len(documents)} chunks)")
                return

            logger.info(f"[vector_store] Building index for {len(documents)} chunks...")
            self._build_index(documents)
            self._try_save_to_disk(cache_hash)
            self._initialized = True
            elapsed = round(time.perf_counter() - t0, 3)
            logger.info(
                f"[vector_store] Index built in {elapsed}s — "
                f"{len(documents)} chunks, dim={self._embedder.dim}, "
                f"categories={list(self._category_index.keys())}"
            )

    def _build_index(self, documents: List[KnowledgeDocument]) -> None:
        self._documents = documents
        self._embedder = _build_embedder()

        texts = [_doc_to_text(doc) for doc in documents]

        if isinstance(self._embedder, TFIDFEmbedder):
            self._embedder.fit(texts)

        embeddings = self._embedder.embed(texts)

        # Build FAISS index
        try:
            import faiss
            dim = embeddings.shape[1]
            if len(documents) < 200:
                index = faiss.IndexFlatIP(dim)
            else:
                nlist = min(32, len(documents) // 8)
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(embeddings)
            index.add(embeddings)
            self._index = index
            logger.info(f"[vector_store] FAISS index: {index.ntotal} vectors, dim={dim}")
        except ImportError:
            logger.warning("[vector_store] FAISS not installed — using NumPy cosine similarity")
            self._index = NumpyFallbackIndex(embeddings)

        # Build BM25 keyword index
        try:
            self._bm25 = BM25Index(documents)
        except Exception as e:
            logger.warning(f"[vector_store] BM25 build failed (non-critical): {e}")
            self._bm25 = None

        # Build category sub-index (position maps)
        self._category_index = {}
        for pos, doc in enumerate(documents):
            self._category_index.setdefault(doc.category, []).append(pos)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _try_load_from_disk(self, expected_hash: str) -> bool:
        try:
            if not INDEX_FILE.exists() or not DOCSTORE_FILE.exists():
                return False
            with open(DOCSTORE_FILE, "rb") as f:
                stored = pickle.load(f)
            if stored.get("hash") != expected_hash or stored.get("version") != CACHE_VERSION:
                return False
            try:
                import faiss
                self._index = faiss.read_index(str(INDEX_FILE))
            except ImportError:
                self._index = stored.get("numpy_index")
            self._documents = stored["documents"]
            self._embedder = stored["embedder"]
            self._bm25 = stored.get("bm25")
            self._category_index = stored.get("category_index", {})
            return True
        except Exception as e:
            logger.warning(f"[vector_store] Cache load failed: {e}")
            return False

    def _try_save_to_disk(self, cache_hash: str) -> None:
        try:
            PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            stored: Dict[str, Any] = {
                "hash": cache_hash,
                "version": CACHE_VERSION,
                "documents": self._documents,
                "embedder": self._embedder,
                "bm25": self._bm25,
                "category_index": self._category_index,
            }
            try:
                import faiss
                faiss.write_index(self._index, str(INDEX_FILE))
            except (ImportError, AttributeError):
                stored["numpy_index"] = self._index
            with open(DOCSTORE_FILE, "wb") as f:
                pickle.dump(stored, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"[vector_store] Persisted to disk: {PERSIST_DIR}")
        except Exception as e:
            logger.warning(f"[vector_store] Disk persist failed (non-critical): {e}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.10,
        hybrid: bool = True,
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """
        Hybrid search: dense vector + BM25 keyword fusion.

        Args:
            query: Free-text search query.
            k: Number of results to return.
            category_filter: Optional category string to restrict results.
            score_threshold: Minimum normalised score (0–1) to include.
            hybrid: If True, fuse vector + BM25. If False, vector-only.

        Returns:
            List of (document, fused_score) sorted by descending score.
        """
        if not self._initialized:
            self.initialise()

        # Check cache
        cached = self._query_cache.get(query, k, category_filter, hybrid)
        if cached is not None:
            return cached

        # Normalise category alias
        from services.rag.knowledge_loader import CATEGORY_ALIAS_MAP
        if category_filter:
            category_filter = CATEGORY_ALIAS_MAP.get(category_filter, category_filter)

        # Dense vector search
        fetch_k = min(k * 6, max(len(self._documents), 1))
        query_vec = self._embedder.embed([query])   # shape (1, dim)
        distances, indices = self._index.search(query_vec, fetch_k)
        vector_results: Dict[int, float] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._documents):
                vector_results[int(idx)] = float(dist)

        # BM25 keyword search (if available and hybrid mode)
        keyword_results: Dict[int, float] = {}
        if hybrid and self._bm25 is not None:
            try:
                bm25_hits = self._bm25.search(query, k=fetch_k)
                if bm25_hits:
                    max_bm25 = max(s for _, s in bm25_hits) or 1.0
                    for idx, score in bm25_hits:
                        keyword_results[idx] = score / max_bm25   # normalise to [0,1]
            except Exception as e:
                logger.debug(f"[vector_store] BM25 search error (non-critical): {e}")

        # Reciprocal Rank Fusion
        fused_scores = _reciprocal_rank_fusion(
            vector_results, keyword_results, k=fetch_k, rrf_k=60
        )

        # Apply category filter + threshold + deduplicate parent docs
        results: List[Tuple[KnowledgeDocument, float]] = []
        seen_parent_ids: set = set()

        for pos, fused_score in fused_scores:
            if pos >= len(self._documents):
                continue
            doc = self._documents[pos]

            # Category filter
            if category_filter and doc.category != category_filter:
                continue

            # Score threshold
            if fused_score < score_threshold:
                continue

            # Soft dedup: prefer highest-scoring chunk per parent doc
            parent = doc.parent_doc_id or doc.doc_id
            if parent in seen_parent_ids:
                continue
            seen_parent_ids.add(parent)

            results.append((doc, round(fused_score, 4)))
            if len(results) >= k:
                break

        # Cache and return
        self._query_cache.put(query, k, category_filter, hybrid, results)
        return results

    def search_by_category(
        self,
        query: str,
        category: str,
        k: int = 5,
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """Category-targeted search using the category sub-index for efficiency."""
        return self.search(query, k=k, category_filter=category, hybrid=True)

    def get_all_documents(self) -> List[KnowledgeDocument]:
        if not self._initialized:
            self.initialise()
        return list(self._documents)

    def get_category_counts(self) -> Dict[str, int]:
        if not self._initialized:
            self.initialise()
        return {cat: len(positions) for cat, positions in self._category_index.items()}

    def cache_stats(self) -> Dict[str, Any]:
        return {
            "query_cache_entries": self._query_cache.size,
            "query_cache_max": self._query_cache._maxsize,
            "query_cache_ttl": self._query_cache._ttl,
        }

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._index is not None


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    vector_scores: Dict[int, float],
    keyword_scores: Dict[int, float],
    k: int,
    rrf_k: int = 60,
    vector_weight: float = 0.65,
    keyword_weight: float = 0.35,
) -> List[Tuple[int, float]]:
    """
    Merge dense vector and BM25 keyword results using Reciprocal Rank Fusion.

    RRF score = Σ weight_i / (rrf_k + rank_i)
    Lower rank number = higher relevance.
    """
    all_doc_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
    fused: Dict[int, float] = {}

    # Rank vector results
    sorted_vector = sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)
    vector_rank = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_vector)}

    # Rank keyword results
    sorted_keyword = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    keyword_rank = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_keyword)}

    for doc_id in all_doc_ids:
        score = 0.0
        if doc_id in vector_rank:
            score += vector_weight / (rrf_k + vector_rank[doc_id])
        if doc_id in keyword_rank:
            score += keyword_weight / (rrf_k + keyword_rank[doc_id])
        fused[doc_id] = score

    sorted_fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # Normalise scores to [0, 1]
    if sorted_fused:
        max_score = sorted_fused[0][1]
        if max_score > 0:
            sorted_fused = [(doc_id, score / max_score) for doc_id, score in sorted_fused]

    return sorted_fused[:k * 2]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_to_text(doc: KnowledgeDocument) -> str:
    """Build rich text for embedding (title + content + metadata signals)."""
    parts = [
        doc.title,
        doc.content,
        f"Category: {doc.category}",
        f"Subcategory: {doc.subcategory}",
        f"Tags: {', '.join(doc.tags)}",
    ]
    city = doc.metadata.get("city", "")
    if city:
        parts.append(f"City: {city}")
    room_types = doc.metadata.get("room_types", [])
    if room_types and room_types != ["all"]:
        parts.append(f"Rooms: {', '.join(room_types)}")
    return " ".join(parts)


def _compute_docs_hash(documents: List[KnowledgeDocument]) -> str:
    payload = json.dumps(
        [{"id": d.doc_id, "c": d.content[:64]} for d in documents],
        sort_keys=True,
    ).encode()
    return hashlib.md5(payload).hexdigest()


# ── Singleton ─────────────────────────────────────────────────────────────────

_store_instance: Optional[FAISSKnowledgeStore] = None
_store_lock = threading.Lock()


def get_knowledge_store() -> FAISSKnowledgeStore:
    """Return the singleton FAISSKnowledgeStore, initialising on first call."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = FAISSKnowledgeStore()
                _store_instance.initialise()
    return _store_instance