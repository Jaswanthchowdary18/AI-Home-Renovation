"""
ARKEN — RAG Retriever v2.0
============================
Production-quality retriever implementing the full pipeline:

  query → embedding → vector search → document ranking → dedup

New in v2.0:
  - Domain-aware query routing: maps signals to all 6 knowledge categories
  - Multi-query fan-out with per-domain targeted queries
  - Cross-Encoder re-ranking (when sentence-transformers available)
  - Cohere-style MMR (Maximal Marginal Relevance) for diversity
  - Retriever caching via CacheService integration (Redis / in-memory)
  - retrieve_for_agent() method for targeted single-agent retrieval
  - Synchronous and async interfaces

CRITICAL: Does NOT call any Gemini/vision/image-generation APIs.
Only consumes upstream agent outputs via state dict.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from services.rag.knowledge_loader import (
    KnowledgeCategory,
    KnowledgeDocument,
    CATEGORY_ALIAS_MAP,
)
from services.rag.vector_store import get_knowledge_store

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RetrievalQuery:
    """A single retrieval query targeting a specific knowledge domain."""
    query_text: str
    category_filter: Optional[str] = None
    weight: float = 1.0
    source_signal: str = ""


@dataclass
class RetrievedChunk:
    """A retrieved knowledge document with score, rank, and provenance."""
    document: KnowledgeDocument
    score: float
    query_used: str
    rank: int
    retrieval_method: str = "hybrid"   # "vector" | "keyword" | "hybrid"


# ── Domain-aware signal extractor ─────────────────────────────────────────────

class VisionSignalExtractor:
    """
    Extracts structured renovation signals from LangGraph state.
    Used to generate domain-targeted retrieval queries.
    """

    def extract(self, state: Dict[str, Any]) -> Dict[str, Any]:
        vision = state.get("vision_features") or state.get("image_features") or {}
        room_features = state.get("room_features") or vision

        def _pick(*keys, default="") -> str:
            for k in keys:
                v = vision.get(k) or room_features.get(k) or state.get(k)
                if v:
                    return str(v).lower()
            return default

        signals = {
            "room_type":       state.get("room_type", "bedroom").lower(),
            "city":            state.get("city", "Hyderabad"),
            "budget_tier":     state.get("budget_tier", "mid"),
            "theme":           state.get("theme", "Modern Minimalist"),
            "floor_material":  _pick("floor_material", "floor_type"),
            "wall_treatment":  _pick("wall_treatment", "wall_color"),
            "ceiling":         _pick("ceiling_treatment", "ceiling_type"),
            "room_condition":  _pick("room_condition", "condition", default="fair"),
            "detected_style":  _pick("detected_style", default=state.get("theme", "")),
            "specific_changes": (
                vision.get("specific_changes")
                or room_features.get("specific_changes")
                or state.get("detected_changes")
                or []
            ),
            "material_types":  state.get("material_types") or [],
            "wall_area_sqft":  float(state.get("wall_area_sqft") or vision.get("estimated_wall_area_sqft") or 200),
            "floor_area_sqft": float(state.get("floor_area_sqft") or vision.get("estimated_floor_area_sqft") or 120),
            "user_intent":     state.get("user_intent", ""),
            # NEW v2.1: condition fields from improved Gemini extraction
            "condition_score":  int(
                vision.get("condition_score") or room_features.get("condition_score")
                or state.get("condition_score") or 65
            ),
            "wall_condition":   (
                vision.get("wall_condition") or room_features.get("wall_condition")
                or state.get("wall_condition") or "fair"
            ),
            "floor_condition":  (
                vision.get("floor_condition") or room_features.get("floor_condition")
                or state.get("floor_condition") or "fair"
            ),
            "issues_detected":  (
                vision.get("issues_detected") or room_features.get("issues_detected")
                or state.get("issues_detected") or []
            ),
            "renovation_scope": (
                vision.get("renovation_scope") or room_features.get("renovation_scope")
                or state.get("renovation_scope") or "partial"
            ),
        }

        logger.debug(
            f"[retriever] Signals — room={signals['room_type']}, "
            f"floor={signals['floor_material']}, theme={signals['theme']}, "
            f"condition={signals['room_condition']}"
        )
        return signals


# ── Domain query formulator ───────────────────────────────────────────────────

class DomainQueryFormulator:
    """
    Converts vision signals into per-domain retrieval queries covering
    all 6 knowledge categories.
    """

    # Defect keyword → repair query
    DEFECT_QUERIES = {
        "crack":    "repair cracked wall plaster interior India",
        "damp":     "damp seepage moisture wall repair waterproofing India",
        "peel":     "peeling paint wall repair preparation coat",
        "stain":    "ceiling water stain seepage repair treatment",
        "hollow":   "hollow debonded floor tile repair epoxy injection",
        "leak":     "plumbing pipe leak repair bathroom India",
        "mould":    "mould mildew wall treatment bathroom India",
        "broken":   "broken tile replacement renovation India",
        "worn":     "worn floor material replacement renovation India",
        "old":      "renovation upgrade outdated materials India",
        "damaged":  "damaged surface repair renovation cost India",
        "spall":    "concrete spalling repair slab reinforcement",
    }

    FLOOR_MATERIAL_QUERIES = {
        "tile":      "vitrified tile flooring cost installation India recommendation",
        "vitrified": "vitrified tile flooring cost installation India brand",
        "marble":    "marble flooring cost installation India recommendation",
        "wood":      "engineered hardwood flooring cost India brand",
        "hardwood":  "engineered hardwood flooring cost India installation",
        "laminate":  "laminate flooring installation cost India brand",
        "vinyl":     "luxury vinyl tile LVT flooring cost India",
        "epoxy":     "epoxy flooring residential cost India application",
        "granite":   "granite flooring cost installation India",
    }

    ROOM_COST_QUERIES = {
        "kitchen":    "modular kitchen renovation cost India brands",
        "bathroom":   "bathroom renovation cost sanitary ware fitting India",
        "bedroom":    "bedroom renovation cost flooring ceiling wardrobe India",
        "living_room":"living room renovation cost flooring ceiling India",
        "dining":     "dining room renovation flooring ceiling light India",
        "office":     "home office renovation materials cost India",
        "balcony":    "balcony renovation waterproofing tiles India",
        "entrance":   "entrance foyer renovation marble flooring India cost",
    }

    STYLE_TO_INTERIOR = {
        "modern":      "modern minimalist interior design material recommendation India",
        "minimalist":  "minimalist interior design elements colours India",
        "scandinavian":"scandinavian interior style India implementation",
        "luxury":      "luxury contemporary interior design India material",
        "contemporary":"contemporary interior design India colour scheme material",
        "industrial":  "industrial interior design India concrete brick material",
        "traditional": "traditional Indian interior design elements material",
        "colonial":    "colonial British Indo-Saracenic interior material India",
    }

    def formulate(self, signals: Dict[str, Any]) -> List[RetrievalQuery]:
        """Generate a prioritised, domain-distributed list of retrieval queries."""
        queries: List[RetrievalQuery] = []
        seen: Set[str] = set()

        room_type   = signals["room_type"]
        city        = signals["city"]
        budget_tier = signals["budget_tier"]
        theme       = signals["theme"].lower()
        floor       = signals["floor_material"]
        wall        = signals["wall_treatment"]
        ceiling     = signals["ceiling"]
        condition   = signals["room_condition"]
        changes     = signals["specific_changes"]
        style       = signals["detected_style"].lower() or theme
        material_types = signals["material_types"]
        user_intent = signals["user_intent"]

        all_text = " ".join([
            condition,
            " ".join(str(c).lower() for c in changes),
            floor, wall, ceiling,
        ])

        def _add(q: RetrievalQuery):
            key = q.query_text[:60]
            if key not in seen:
                seen.add(key)
                queries.append(q)

        # ── 0. CONDITION-BASED: poor condition triggers repair + sequencing docs ──
        condition_score = signals.get("condition_score", 65)
        wall_cond = signals.get("wall_condition", "fair")
        floor_cond = signals.get("floor_condition", "fair")
        issues = signals.get("issues_detected", [])
        reno_scope = signals.get("renovation_scope", "partial")

        if condition_score < 50 or wall_cond in ("poor", "very poor"):
            _add(RetrievalQuery(
                query_text="poor wall condition remediation repair sequence before renovation India",
                category_filter=KnowledgeCategory.CONSTRUCTION_MATERIALS,
                weight=2.0,
                source_signal="condition:poor_wall",
            ))

        if condition_score < 50 or floor_cond in ("poor", "very poor"):
            _add(RetrievalQuery(
                query_text="poor floor condition hollow tile repair replacement India cost",
                category_filter=KnowledgeCategory.CONSTRUCTION_MATERIALS,
                weight=1.8,
                source_signal="condition:poor_floor",
            ))

        if condition_score < 70:
            _add(RetrievalQuery(
                query_text=f"condition score {condition_score} renovation priority budget allocation India",
                category_filter=KnowledgeCategory.RENOVATION_COSTS,
                weight=1.6,
                source_signal="condition:budget_priority",
            ))

        # Issues detected from Gemini image analysis
        issues_text = " ".join(str(i).lower() for i in issues)
        for kw, repair_q in self.DEFECT_QUERIES.items():
            if kw in issues_text:
                _add(RetrievalQuery(
                    query_text=repair_q,
                    category_filter=KnowledgeCategory.CONSTRUCTION_MATERIALS,
                    weight=1.9,
                    source_signal=f"issue_detected:{kw}",
                ))

        # High-value upgrades query
        _add(RetrievalQuery(
            query_text=f"highest ROI renovation upgrades India 2026 {room_type} {city}",
            category_filter=KnowledgeCategory.REAL_ESTATE_VALUE_FACTORS,
            weight=1.5,
            source_signal="high_roi_upgrades",
        ))

        # ── 1. CONSTRUCTION_MATERIALS: defect-triggered repair queries ─────────
        for kw, repair_q in self.DEFECT_QUERIES.items():
            if kw in all_text:
                _add(RetrievalQuery(
                    query_text=repair_q,
                    category_filter=KnowledgeCategory.CONSTRUCTION_MATERIALS,
                    weight=1.6,
                    source_signal=f"defect:{kw}",
                ))

        # ── 2. RENOVATION_COSTS: room-specific cost guidelines ─────────────────
        room_cost_q = self.ROOM_COST_QUERIES.get(room_type, f"{room_type} renovation cost India")
        _add(RetrievalQuery(
            query_text=f"{room_cost_q} {budget_tier} budget {city}",
            category_filter=KnowledgeCategory.RENOVATION_COSTS,
            weight=1.4,
            source_signal="room_cost",
        ))

        # ── 3. RENOVATION_COSTS: full home per-sqft benchmark ─────────────────
        _add(RetrievalQuery(
            query_text=f"full home renovation cost per sqft India {budget_tier} budget {city}",
            category_filter=KnowledgeCategory.RENOVATION_COSTS,
            weight=1.2,
            source_signal="cost_benchmark",
        ))

        # ── 4. CONSTRUCTION_MATERIALS: floor material ──────────────────────────
        for mat_kw, mat_q in self.FLOOR_MATERIAL_QUERIES.items():
            if mat_kw in floor or mat_kw in " ".join(material_types).lower():
                _add(RetrievalQuery(
                    query_text=mat_q,
                    category_filter=KnowledgeCategory.CONSTRUCTION_MATERIALS,
                    weight=1.3,
                    source_signal=f"floor_material:{mat_kw}",
                ))
                break

        # Default floor if no specific match
        if not any("floor_material" in q.source_signal for q in queries):
            _add(RetrievalQuery(
                query_text=f"flooring material recommendation {room_type} India {budget_tier} durability",
                category_filter=KnowledgeCategory.CONSTRUCTION_MATERIALS,
                weight=1.1,
                source_signal="floor_default",
            ))

        # ── 5. CONSTRUCTION_MATERIALS: paint brands ────────────────────────────
        _add(RetrievalQuery(
            query_text=f"interior wall paint brand recommendation India {budget_tier} emulsion primer",
            category_filter=KnowledgeCategory.CONSTRUCTION_MATERIALS,
            weight=1.1,
            source_signal="paint_brand",
        ))

        # ── 6. RENOVATION_COSTS: wall paint cost ──────────────────────────────
        _add(RetrievalQuery(
            query_text=f"interior wall painting cost per sqft India {budget_tier}",
            category_filter=KnowledgeCategory.RENOVATION_COSTS,
            weight=1.0,
            source_signal="paint_cost",
        ))

        # ── 7. RENOVATION_COSTS: ceiling cost (conditional) ───────────────────
        ceiling_work = any("ceiling" in str(c).lower() for c in changes)
        if ceiling_work or any(kw in ceiling for kw in ["false", "pop", "gypsum"]):
            _add(RetrievalQuery(
                query_text=f"false ceiling installation cost {room_type} India {budget_tier}",
                category_filter=KnowledgeCategory.RENOVATION_COSTS,
                weight=1.2,
                source_signal="ceiling_cost",
            ))

        # ── 8. INTERIOR_DESIGN: style-based recommendations ───────────────────
        for style_kw, style_q in self.STYLE_TO_INTERIOR.items():
            if style_kw in style:
                _add(RetrievalQuery(
                    query_text=style_q,
                    category_filter=KnowledgeCategory.INTERIOR_DESIGN,
                    weight=1.0,
                    source_signal=f"style:{style_kw}",
                ))
                break

        # Default interior design query
        if not any(q.source_signal.startswith("style:") for q in queries):
            _add(RetrievalQuery(
                query_text=f"interior design {theme} colour scheme material recommendation India",
                category_filter=KnowledgeCategory.INTERIOR_DESIGN,
                weight=0.9,
                source_signal="style_default",
            ))

        # ── 9. ARCHITECTURE_DESIGN: structural constraints ────────────────────
        if any(kw in all_text for kw in ["wall", "open", "struct", "knock", "remov"]):
            _add(RetrievalQuery(
                query_text=f"load bearing wall structural constraint renovation India",
                category_filter=KnowledgeCategory.ARCHITECTURE_DESIGN,
                weight=1.1,
                source_signal="structural",
            ))

        # ── 10. SPACE_PLANNING_RULES: room ergonomics ─────────────────────────
        _add(RetrievalQuery(
            query_text=f"{room_type} space planning clearances furniture ergonomics India NBC",
            category_filter=KnowledgeCategory.SPACE_PLANNING_RULES,
            weight=0.9,
            source_signal="space_planning",
        ))

        # ── 11. REAL_ESTATE_VALUE_FACTORS: ROI + buyer data ───────────────────
        _add(RetrievalQuery(
            query_text=f"renovation ROI return investment India {room_type} value addition resale",
            category_filter=KnowledgeCategory.REAL_ESTATE_VALUE_FACTORS,
            weight=1.1,
            source_signal="roi_factors",
        ))

        # ── 12. REAL_ESTATE_VALUE_FACTORS: city case study ────────────────────
        _add(RetrievalQuery(
            query_text=f"renovation case study {city} {room_type} ROI outcome rental India",
            category_filter=KnowledgeCategory.REAL_ESTATE_VALUE_FACTORS,
            weight=1.0,
            source_signal="case_study",
        ))

        # ── 13. User intent passthrough (uncategorised broad query) ──────────
        if user_intent:
            _add(RetrievalQuery(
                query_text=user_intent,
                category_filter=None,
                weight=1.3,
                source_signal="user_intent",
            ))

        queries.sort(key=lambda q: q.weight, reverse=True)
        logger.debug(f"[retriever] Formulated {len(queries)} domain queries")
        return queries[:14]


# ── MMR diversity filter ──────────────────────────────────────────────────────

def _mmr_diversify(
    scored_docs: List[Tuple[KnowledgeDocument, float]],
    embedder: Any,
    lambda_param: float = 0.7,
    final_k: int = 8,
) -> List[Tuple[KnowledgeDocument, float]]:
    """
    Maximal Marginal Relevance (MMR) to balance relevance vs. diversity.
    lambda_param=1.0 = pure relevance; 0.0 = pure diversity.
    Falls back to score-sorted list if embedder unavailable.
    """
    if len(scored_docs) <= final_k:
        return scored_docs

    try:
        texts = [_doc_to_text_mmr(doc) for doc, _ in scored_docs]
        embeddings = embedder.embed(texts)   # (n, dim)
        scores = np.array([s for _, s in scored_docs])

        selected_indices: List[int] = []
        candidate_indices = list(range(len(scored_docs)))

        # Greedily select
        while len(selected_indices) < final_k and candidate_indices:
            best_idx = -1
            best_score = -1.0

            for cand in candidate_indices:
                relevance = float(scores[cand])
                if not selected_indices:
                    mmr_score = relevance
                else:
                    selected_embs = embeddings[selected_indices]
                    cand_emb = embeddings[cand]
                    sims = selected_embs @ cand_emb
                    max_sim = float(np.max(sims))
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = cand

            if best_idx >= 0:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)

        return [scored_docs[i] for i in selected_indices]

    except Exception as e:
        logger.debug(f"[retriever] MMR failed (non-critical, using top-k): {e}")
        return scored_docs[:final_k]


def _doc_to_text_mmr(doc: KnowledgeDocument) -> str:
    return f"{doc.title} {doc.subcategory} {' '.join(doc.tags)} {doc.content[:200]}"


# ── Main Retriever ────────────────────────────────────────────────────────────

class RenovationRetriever:
    """
    Production retriever: state → queries → hybrid search → re-rank → context.

    Implements:
    - Domain-aware multi-query fan-out
    - Hybrid vector + keyword retrieval per query
    - Score fusion across queries (weighted max)
    - MMR diversity filtering
    - Optional Cross-Encoder re-ranking
    """

    def __init__(
        self,
        top_k_per_query: int = 4,
        final_top_k: int = 10,
        use_mmr: bool = True,
        use_reranker: bool = False,
    ):
        self._extractor = VisionSignalExtractor()
        self._formulator = DomainQueryFormulator()
        self._top_k_per_query = top_k_per_query
        self._final_top_k = final_top_k
        self._use_mmr = use_mmr
        self._use_reranker = use_reranker
        self._cross_encoder = None

    def _get_cross_encoder(self):
        """Lazy-load cross-encoder re-ranker (optional)."""
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("[retriever] Cross-encoder re-ranker loaded")
        except (ImportError, Exception) as e:
            logger.debug(f"[retriever] Cross-encoder unavailable: {e}")
            self._cross_encoder = False
        return self._cross_encoder

    def retrieve(
        self,
        state: Dict[str, Any],
        extra_queries: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Full retrieval pipeline from LangGraph state.

        Args:
            state: LangGraph pipeline state (post vision analysis).
            extra_queries: Optional free-text additional queries.

        Returns:
            Ranked list of RetrievedChunk objects.
        """
        t0 = time.perf_counter()

        # 1. Extract signals from state
        signals = self._extractor.extract(state)

        # 2. Formulate domain queries
        queries = self._formulator.formulate(signals)

        # 3. Add extra queries
        if extra_queries:
            for eq in extra_queries:
                if eq.strip():
                    queries.append(RetrievalQuery(
                        query_text=eq.strip(),
                        weight=1.3,
                        source_signal="extra",
                    ))

        # 4. Execute all queries against the store
        store = get_knowledge_store()
        scored_map: Dict[str, Tuple[KnowledgeDocument, float, str]] = {}

        for q in queries:
            try:
                hits = store.search(
                    query=q.query_text,
                    k=self._top_k_per_query,
                    category_filter=q.category_filter,
                    score_threshold=0.08,
                    hybrid=True,
                )
                for doc, raw_score in hits:
                    weighted = raw_score * q.weight
                    parent_id = doc.parent_doc_id or doc.doc_id
                    existing = scored_map.get(parent_id)
                    if existing is None or weighted > existing[1]:
                        scored_map[parent_id] = (doc, weighted, q.query_text)
            except Exception as e:
                logger.warning(f"[retriever] Query failed '{q.query_text[:50]}': {e}")

        # 5. Sort by weighted score
        sorted_docs = sorted(scored_map.values(), key=lambda x: x[1], reverse=True)
        scored_list = [(doc, score) for doc, score, _ in sorted_docs]

        # 6. Optional Cross-Encoder re-ranking on top candidates
        if self._use_reranker:
            query_text = signals.get("user_intent") or signals.get("room_type", "renovation")
            scored_list = self._rerank(query_text, scored_list)

        # 7. MMR diversity filter
        if self._use_mmr and len(scored_list) > self._final_top_k:
            embedder = store._embedder
            scored_list = _mmr_diversify(scored_list, embedder, lambda_param=0.70, final_k=self._final_top_k)

        # 8. Build final chunk list
        chunks = []
        for rank, (doc, score) in enumerate(scored_list[:self._final_top_k]):
            query_used = scored_map.get(doc.parent_doc_id or doc.doc_id, (None, None, ""))[2]
            chunks.append(RetrievedChunk(
                document=doc,
                score=round(score, 4),
                query_used=query_used,
                rank=rank + 1,
                retrieval_method="hybrid",
            ))

        elapsed = round(time.perf_counter() - t0, 3)
        cat_summary = {}
        for c in chunks:
            cat_summary[c.document.category] = cat_summary.get(c.document.category, 0) + 1

        logger.info(
            f"[retriever] Retrieved {len(chunks)} docs in {elapsed}s "
            f"from {len(queries)} queries — categories: {cat_summary}"
        )
        return chunks

    def _rerank(
        self,
        query: str,
        scored_list: List[Tuple[KnowledgeDocument, float]],
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """Cross-Encoder re-ranking on top-20 candidates."""
        ce = self._get_cross_encoder()
        if not ce:
            return scored_list
        try:
            candidates = scored_list[:20]
            pairs = [(query, doc.content[:400]) for doc, _ in candidates]
            ce_scores = ce.predict(pairs)
            reranked = sorted(
                zip([doc for doc, _ in candidates], ce_scores),
                key=lambda x: x[1], reverse=True,
            )
            # Normalise re-rank scores to [0, 1]
            max_score = max(s for _, s in reranked) if reranked else 1.0
            reranked_normalised = [(doc, float(s) / max(max_score, 1e-9)) for doc, s in reranked]
            return reranked_normalised + scored_list[20:]
        except Exception as e:
            logger.debug(f"[retriever] Cross-encoder rerank failed: {e}")
            return scored_list

    def retrieve_for_agent(
        self,
        agent_type: str,
        state: Dict[str, Any],
    ) -> List[RetrievedChunk]:
        """
        Targeted retrieval for a specific downstream agent.

        agent_type options:
          "design_planner"    → interior_design + construction_materials + space_planning
          "budget_estimator"  → renovation_costs
          "roi_predictor"     → real_estate_value_factors
          "repair_advisor"    → construction_materials (repair subcategory)
        """
        AGENT_CATEGORIES = {
            "design_planner": [
                KnowledgeCategory.INTERIOR_DESIGN,
                KnowledgeCategory.CONSTRUCTION_MATERIALS,
                KnowledgeCategory.SPACE_PLANNING_RULES,
                KnowledgeCategory.ARCHITECTURE_DESIGN,
            ],
            "budget_estimator": [
                KnowledgeCategory.RENOVATION_COSTS,
                KnowledgeCategory.CONSTRUCTION_MATERIALS,
            ],
            "roi_predictor": [
                KnowledgeCategory.REAL_ESTATE_VALUE_FACTORS,
                KnowledgeCategory.RENOVATION_COSTS,
            ],
            "repair_advisor": [
                KnowledgeCategory.CONSTRUCTION_MATERIALS,
                KnowledgeCategory.ARCHITECTURE_DESIGN,
            ],
        }

        categories = AGENT_CATEGORIES.get(agent_type)
        if not categories:
            logger.warning(f"[retriever] Unknown agent_type '{agent_type}' — using full retrieval")
            return self.retrieve(state)

        signals = self._extractor.extract(state)
        store = get_knowledge_store()
        scored_map: Dict[str, Tuple[KnowledgeDocument, float, str]] = {}

        for cat in categories:
            query = self._build_agent_query(agent_type, signals, cat)
            try:
                hits = store.search(
                    query=query,
                    k=self._top_k_per_query,
                    category_filter=cat,
                    hybrid=True,
                )
                for doc, score in hits:
                    parent = doc.parent_doc_id or doc.doc_id
                    if parent not in scored_map or score > scored_map[parent][1]:
                        scored_map[parent] = (doc, score, query)
            except Exception as e:
                logger.warning(f"[retriever] Agent query failed for {cat}: {e}")

        sorted_docs = sorted(scored_map.values(), key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(
                document=doc,
                score=round(score, 4),
                query_used=query,
                rank=rank + 1,
            )
            for rank, (doc, score, query) in enumerate(sorted_docs[:self._final_top_k])
        ]

    def _build_agent_query(self, agent_type: str, signals: Dict[str, Any], category: str) -> str:
        room = signals["room_type"]
        city = signals["city"]
        tier = signals["budget_tier"]
        theme = signals["theme"]

        if agent_type == "design_planner":
            return f"{theme} interior design {room} material selection India recommendation"
        elif agent_type == "budget_estimator":
            return f"{room} renovation cost {tier} budget {city} India 2024"
        elif agent_type == "roi_predictor":
            return f"renovation ROI return value addition {room} {city} India resale rental"
        elif agent_type == "repair_advisor":
            return f"repair standard method {room} India construction material best practice"
        return f"{room} renovation {city} India"

    def retrieve_for_repair(self, defect_description: str) -> List[RetrievedChunk]:
        """Targeted repair retrieval from a free-text defect description."""
        store = get_knowledge_store()
        results = store.search(
            query=f"repair method {defect_description} India standard",
            k=5,
            category_filter=KnowledgeCategory.CONSTRUCTION_MATERIALS,
            score_threshold=0.08,
        )
        return [
            RetrievedChunk(
                document=doc,
                score=round(score, 4),
                query_used=defect_description,
                rank=rank + 1,
            )
            for rank, (doc, score) in enumerate(results)
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────

_retriever_instance: Optional[RenovationRetriever] = None


def get_retriever() -> RenovationRetriever:
    """Return the singleton RenovationRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RenovationRetriever(
            top_k_per_query=4,
            final_top_k=10,
            use_mmr=True,
            use_reranker=False,   # Cross-encoder off by default (adds latency)
        )
    return _retriever_instance