"""
ARKEN — RAG Context Builder v2.0
===================================
Formats retrieved knowledge chunks into structured context blocks for LLM injection.

Key improvements over v1.0:
  - Structured sections per knowledge domain (all 6 categories)
  - Agent-specific context templates (budget_estimator, roi_predictor, design_planner)
  - Relevance-weighted excerpt selection with keyword highlighting
  - Token-budget aware truncation (char-based, configurable)
  - Integration with CacheService for pipeline-level caching
  - Backward-compatible build() and build_chat_context_addendum() APIs

Context format:
  === ARKEN KNOWLEDGE BASE CONTEXT ===
  [Domain Header]
    • Source: doc_id (score)
      ... excerpt ...
  [Next Domain]
    ...
  ---
  Context personalised for: <room> in <city>. Budget: ₹X.

CRITICAL: Does NOT modify or call any Gemini/vision/image-generation APIs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from services.rag.knowledge_loader import KnowledgeCategory
from services.rag.retriever import RetrievedChunk, RenovationRetriever, get_retriever

logger = logging.getLogger(__name__)


# ── Section headers ────────────────────────────────────────────────────────────

DOMAIN_HEADERS: Dict[str, str] = {
    KnowledgeCategory.RENOVATION_COSTS:          "## 💰 RENOVATION COST GUIDELINES",
    KnowledgeCategory.CONSTRUCTION_MATERIALS:    "## 🔩 CONSTRUCTION MATERIALS & REPAIR STANDARDS",
    KnowledgeCategory.INTERIOR_DESIGN:           "## 🎨 INTERIOR DESIGN PRINCIPLES",
    KnowledgeCategory.ARCHITECTURE_DESIGN:       "## 🏗 ARCHITECTURAL DESIGN CONSTRAINTS",
    KnowledgeCategory.SPACE_PLANNING_RULES:      "## 📐 SPACE PLANNING RULES & CLEARANCES",
    KnowledgeCategory.REAL_ESTATE_VALUE_FACTORS: "## 📊 REAL ESTATE VALUE & ROI ANALYSIS",
    # Legacy aliases
    "cost":       "## 💰 RENOVATION COST GUIDELINES",
    "repair":     "## 🔩 REPAIR STANDARDS",
    "material":   "## 🔩 MATERIAL RECOMMENDATIONS",
    "case_study": "## 📊 RENOVATION CASE STUDIES",
}

CONTEXT_INTRO = """\
=== ARKEN KNOWLEDGE BASE CONTEXT (Retrieved) ===
The following information has been retrieved from ARKEN's verified renovation knowledge base.
INSTRUCTION: Prioritise this retrieved knowledge over general assumptions.
Cite specific costs, brand names, standards, and case study outcomes from this context.\
"""

# Ordered rendering priority for domain sections
DOMAIN_RENDER_ORDER = [
    KnowledgeCategory.RENOVATION_COSTS,
    KnowledgeCategory.CONSTRUCTION_MATERIALS,
    KnowledgeCategory.INTERIOR_DESIGN,
    KnowledgeCategory.SPACE_PLANNING_RULES,
    KnowledgeCategory.ARCHITECTURE_DESIGN,
    KnowledgeCategory.REAL_ESTATE_VALUE_FACTORS,
    # Legacy
    "cost", "repair", "material", "case_study",
]

# Agent-specific context prompts injected before the domain sections
AGENT_CONTEXT_PROMPTS: Dict[str, str] = {
    "budget_estimator": (
        "Focus on: exact cost ranges per sqft, city multipliers, labour rates, "
        "and brand-specific pricing for the specified budget tier.\n"
    ),
    "roi_predictor": (
        "Focus on: ROI percentages, value addition benchmarks, payback periods, "
        "rental yield improvements, and case study outcomes for comparable projects.\n"
    ),
    "design_planner": (
        "Focus on: material specifications, brand recommendations, colour palettes, "
        "style principles, and space planning clearances for the given theme.\n"
    ),
    "repair_advisor": (
        "Focus on: repair methods, step-by-step procedures, materials required, "
        "IS standards compliance, and cost per sqft for remediation.\n"
    ),
}


# ── Context builder ────────────────────────────────────────────────────────────

class RAGContextBuilder:
    """
    Builds structured, domain-organised context strings from retrieved knowledge.

    Primary output is injected into LLM prompts across:
    - node_insight_generation (LangGraph)
    - node_budget_estimation (LangGraph)
    - node_roi_forecasting (LangGraph)
    - Chat Q&A endpoints
    """

    def __init__(
        self,
        max_chars_per_doc: int = 1000,
        max_total_chars: int = 10000,
    ):
        self._max_chars_per_doc = max_chars_per_doc
        self._max_total_chars = max_total_chars

    def build(
        self,
        chunks: List[RetrievedChunk],
        state: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        agent_type: Optional[str] = None,
    ) -> str:
        """
        Build the full RAG context string.

        Args:
            chunks:           Retrieved documents from RenovationRetriever.
            state:            LangGraph state (used for personalisation note).
            include_metadata: Include source attribution lines.
            agent_type:       Optional — injects agent-specific instruction header.

        Returns:
            Formatted context string ready for LLM injection.
        """
        if state is None:
            state = {}

        if not chunks:
            return self._empty_context(state)

        # Group by category
        by_category: Dict[str, List[RetrievedChunk]] = {}
        for chunk in chunks:
            cat = chunk.document.category
            by_category.setdefault(cat, []).append(chunk)

        sections: List[str] = [CONTEXT_INTRO]

        # Optional agent-specific header
        if agent_type and agent_type in AGENT_CONTEXT_PROMPTS:
            sections.append(f"**Agent Focus:** {AGENT_CONTEXT_PROMPTS[agent_type]}")

        total_chars = len(CONTEXT_INTRO)

        # Render sections in priority order, deduplicated
        rendered_categories: set = set()
        for cat in DOMAIN_RENDER_ORDER:
            if cat in rendered_categories:
                continue
            cat_chunks = by_category.get(cat)
            if not cat_chunks:
                continue
            rendered_categories.add(cat)

            header = DOMAIN_HEADERS.get(cat, f"## {cat.upper().replace('_', ' ')}")
            section_lines = [header]

            for chunk in cat_chunks:
                doc_text = self._format_chunk(chunk, include_metadata)
                if total_chars + len(doc_text) > self._max_total_chars:
                    section_lines.append(f"  [Additional {cat} context omitted — token budget]")
                    break
                section_lines.append(doc_text)
                total_chars += len(doc_text)

            sections.append("\n".join(section_lines))

        # Personalisation footer
        sections.append(self._personalisation_note(state, chunks))

        return "\n\n".join(sections)

    def _format_chunk(self, chunk: RetrievedChunk, include_metadata: bool) -> str:
        """Format a single retrieved chunk with source attribution."""
        doc = chunk.document
        content = doc.content

        if len(content) > self._max_chars_per_doc:
            content = content[:self._max_chars_per_doc].rsplit("\n", 1)[0]
            content += "\n  [...truncated]"

        parts = [f"  **[{doc.title}]** _(relevance: {chunk.score:.2f}, rank #{chunk.rank})_"]
        # Indent content for visual hierarchy
        indented = "\n".join(f"  {line}" for line in content.split("\n"))
        parts.append(indented)

        if include_metadata and doc.metadata:
            meta_str = self._format_metadata(doc.metadata)
            if meta_str:
                parts.append(f"  _Applies to: {meta_str}_")

        return "\n".join(parts)

    @staticmethod
    def _format_metadata(metadata: Dict[str, Any]) -> str:
        parts = []
        if "room_types" in metadata:
            rts = metadata["room_types"]
            if rts and rts != ["all"]:
                parts.append("rooms: " + ", ".join(rts))
        if "budget_tiers" in metadata:
            parts.append("tiers: " + ", ".join(metadata["budget_tiers"]))
        if "city" in metadata:
            parts.append("city: " + metadata["city"])
        if "standards" in metadata:
            parts.append("standards: " + ", ".join(metadata["standards"]))
        return "; ".join(parts)

    @staticmethod
    def _personalisation_note(state: Dict[str, Any], chunks: List[RetrievedChunk]) -> str:
        city = state.get("city", "India")
        budget_tier = state.get("budget_tier", "mid")
        budget_inr = state.get("budget_inr", 0)
        room_type = state.get("room_type", "room")
        theme = state.get("theme", "")
        categories = sorted({c.document.category for c in chunks})

        budget_str = ""
        if budget_inr:
            budget_str = (
                f" Budget: ₹{budget_inr/100000:.1f}L ({budget_tier} tier)."
                if budget_inr >= 100000
                else f" Budget: ₹{budget_inr:,} ({budget_tier} tier)."
            )

        return (
            "---\n"
            f"**Context personalised for:** {room_type} renovation in {city}.{budget_str}"
            + (f" Style: {theme}." if theme else "") + "\n"
            f"**Domains retrieved:** {', '.join(categories)}.\n"
            f"**Total knowledge chunks:** {len(chunks)}.\n"
            "Use the above figures, brand names, and standards in your analysis."
        )

    @staticmethod
    def _empty_context(state: Dict[str, Any]) -> str:
        city = state.get("city", "India")
        room_type = state.get("room_type", "room")
        budget_tier = state.get("budget_tier", "mid")
        return (
            "=== KNOWLEDGE BASE CONTEXT ===\n"
            "No specific knowledge documents retrieved for this query.\n"
            f"Project: {room_type} renovation in {city} ({budget_tier} budget).\n"
            "Apply general Indian renovation cost benchmarks: "
            "₹800–2,500/sqft for full renovation depending on tier.\n"
            "City premium: Mumbai/Delhi +25–35%, Bangalore +15%, Hyderabad base."
        )

    def build_agent_context(
        self,
        chunks: List[RetrievedChunk],
        state: Dict[str, Any],
        agent_type: str,
    ) -> str:
        """
        Build agent-specific context with targeted instructions.

        agent_type: "budget_estimator" | "roi_predictor" | "design_planner" | "repair_advisor"
        """
        return self.build(chunks, state, agent_type=agent_type)

    def build_chat_context_addendum(
        self,
        chunks: List[RetrievedChunk],
        user_question: str,
    ) -> str:
        """
        Compact context for chat Q&A — top chunk per category, relevance-filtered.
        """
        if not chunks:
            return ""

        top_per_cat: Dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            cat = chunk.document.category
            if cat not in top_per_cat or chunk.score > top_per_cat[cat].score:
                top_per_cat[cat] = chunk

        # Filter by minimum score
        top_per_cat = {cat: c for cat, c in top_per_cat.items() if c.score >= 0.15}

        if not top_per_cat:
            return ""

        parts = [f"Relevant renovation knowledge for: '{user_question[:80]}'"]
        for cat, chunk in top_per_cat.items():
            doc = chunk.document
            display_cat = cat.replace("_", " ").upper()
            relevant = _extract_relevant_paragraph(doc.content, user_question)
            parts.append(f"[{display_cat}] {relevant}")

        return "\n\n".join(parts)


# ── Paragraph extraction ──────────────────────────────────────────────────────

def _extract_relevant_paragraph(content: str, query: str) -> str:
    """Extract the most query-relevant paragraph using keyword overlap scoring."""
    query_words = set(re.sub(r"[^a-z0-9 ]", "", query.lower()).split())
    query_words -= {"the", "a", "an", "is", "in", "of", "for", "to", "and", "or", "how"}

    paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 40]
    if not paragraphs:
        return content[:400]

    scored = []
    for para in paragraphs:
        para_words = set(re.sub(r"[^a-z0-9 ]", "", para.lower()).split())
        overlap = len(query_words & para_words)
        scored.append((overlap, len(para), para))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    best = scored[0][2]
    return best[:500] + ("..." if len(best) > 500 else "")


# ── Full pipeline ─────────────────────────────────────────────────────────────

class RenovationRAGPipeline:
    """
    Convenience wrapper running the complete RAG pipeline.

    state → retrieval → context → enriched insights

    This is the primary integration point for LangGraph nodes.
    Fully backward-compatible with v1.0 API.
    """

    def __init__(self):
        self._retriever: Optional[RenovationRetriever] = None
        self._builder = RAGContextBuilder()

    @property
    def retriever(self) -> RenovationRetriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    def run(
        self,
        state: Dict[str, Any],
        extra_queries: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline.

        Returns dict with:
          rag_context:       str — formatted context for LLM injection
          rag_chunks:        List[RetrievedChunk]
          rag_doc_ids:       List[str]
          rag_categories:    List[str]
          rag_domain_stats:  Dict[str, int] — chunk count per domain
        """
        try:
            chunks = self.retriever.retrieve(state, extra_queries=extra_queries)
            context = self._builder.build(chunks, state, agent_type=agent_type)

            doc_ids = [c.document.doc_id for c in chunks]
            categories = list({c.document.category for c in chunks})
            domain_stats: Dict[str, int] = {}
            for c in chunks:
                domain_stats[c.document.category] = domain_stats.get(c.document.category, 0) + 1

            logger.info(
                f"[rag_pipeline] Built — {len(chunks)} chunks, "
                f"domains={domain_stats}, chars={len(context)}"
            )

            return {
                "rag_context":      context,
                "rag_chunks":       chunks,
                "rag_doc_ids":      doc_ids,
                "rag_categories":   categories,
                "rag_domain_stats": domain_stats,
            }

        except Exception as e:
            logger.error(f"[rag_pipeline] Failed: {e}", exc_info=True)
            return {
                "rag_context":      self._builder._empty_context(state),
                "rag_chunks":       [],
                "rag_doc_ids":      [],
                "rag_categories":   [],
                "rag_domain_stats": {},
            }

    def run_for_agent(
        self,
        state: Dict[str, Any],
        agent_type: str,
    ) -> Dict[str, Any]:
        """
        Run targeted RAG pipeline for a specific downstream agent.
        Produces focused context relevant to that agent's reasoning task.
        """
        try:
            chunks = self.retriever.retrieve_for_agent(agent_type, state)
            context = self._builder.build_agent_context(chunks, state, agent_type)
            return {
                "rag_context":      context,
                "rag_chunks":       chunks,
                "rag_doc_ids":      [c.document.doc_id for c in chunks],
                "rag_categories":   list({c.document.category for c in chunks}),
                "rag_domain_stats": _count_by_category(chunks),
            }
        except Exception as e:
            logger.error(f"[rag_pipeline] Agent run failed ({agent_type}): {e}", exc_info=True)
            return {
                "rag_context":      self._builder._empty_context(state),
                "rag_chunks":       [],
                "rag_doc_ids":      [],
                "rag_categories":   [],
                "rag_domain_stats": {},
            }

    def enrich_insights(
        self,
        insights: Dict[str, Any],
        rag_result: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge RAG knowledge into the existing insights dict.

        Enriches:
          insights["rag_knowledge_context"]   → raw context string
          insights["cost_references"]          → cost docs excerpts
          insights["material_references"]      → material docs excerpts
          insights["repair_guidance"]          → repair standards
          insights["case_study_references"]    → real project outcomes
          insights["space_planning_notes"]     → clearance rules
          insights["architecture_constraints"] → structural constraints
          insights["knowledge_sources"]        → doc IDs used
        """
        chunks: List[RetrievedChunk] = rag_result.get("rag_chunks", [])
        enriched = dict(insights)
        enriched["rag_knowledge_context"] = rag_result.get("rag_context", "")
        enriched["knowledge_sources"] = rag_result.get("rag_doc_ids", [])

        if not chunks:
            return enriched

        # Bucket by category
        buckets: Dict[str, List[Dict]] = {}
        for chunk in chunks:
            doc = chunk.document
            excerpt = {
                "source":      doc.doc_id,
                "title":       doc.title,
                "subcategory": doc.subcategory,
                "score":       chunk.score,
                "summary":     _first_n_lines(doc.content, 5),
            }
            buckets.setdefault(doc.category, []).append(excerpt)

        # Map new categories (and legacy aliases) to insight fields
        FIELD_MAP = {
            KnowledgeCategory.RENOVATION_COSTS:          "cost_references",
            KnowledgeCategory.CONSTRUCTION_MATERIALS:    "material_references",
            KnowledgeCategory.INTERIOR_DESIGN:           "design_references",
            KnowledgeCategory.ARCHITECTURE_DESIGN:       "architecture_constraints",
            KnowledgeCategory.SPACE_PLANNING_RULES:      "space_planning_notes",
            KnowledgeCategory.REAL_ESTATE_VALUE_FACTORS: "case_study_references",
            # Legacy
            "cost":       "cost_references",
            "repair":     "repair_guidance",
            "material":   "material_references",
            "case_study": "case_study_references",
        }

        for cat, excerpts in buckets.items():
            field = FIELD_MAP.get(cat)
            if field and excerpts:
                enriched[field] = excerpts

        # Apply cost overrides
        cost_refs = buckets.get(KnowledgeCategory.RENOVATION_COSTS, []) or buckets.get("cost", [])
        if cost_refs:
            enriched = _apply_cost_overrides(enriched, cost_refs, state)

        # Apply case study ROI signals
        case_refs = (
            buckets.get(KnowledgeCategory.REAL_ESTATE_VALUE_FACTORS, [])
            or buckets.get("case_study", [])
        )
        if case_refs:
            enriched = _apply_case_study_signals(enriched, case_refs, state)

        return enriched


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_n_lines(content: str, n: int) -> str:
    lines = [l for l in content.split("\n") if l.strip()][:n]
    return "\n".join(lines)


def _count_by_category(chunks: List[RetrievedChunk]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for c in chunks:
        stats[c.document.category] = stats.get(c.document.category, 0) + 1
    return stats


def _apply_cost_overrides(
    insights: Dict[str, Any],
    cost_excerpts: List[Dict],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Inject the best cost reference into budget_assessment."""
    if not cost_excerpts:
        return insights
    budget_tier = state.get("budget_tier", "mid")
    best = max(cost_excerpts, key=lambda x: x["score"])
    cost_note = (
        f"Cost reference ({best['subcategory']}, {budget_tier} tier): "
        + best["summary"].split("\n")[0]
    )
    budget = insights.get("budget_assessment", {})
    if isinstance(budget, dict):
        budget["rag_cost_reference"] = cost_note
        insights["budget_assessment"] = budget
    return insights


def _apply_case_study_signals(
    insights: Dict[str, Any],
    case_excerpts: List[Dict],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Enrich ROI signals with case study data."""
    if not case_excerpts:
        return insights
    city = state.get("city", "India").lower()
    city_match = next(
        (c for c in case_excerpts if city in c["source"].lower()),
        case_excerpts[0],
    )
    financial = insights.get("financial_outlook", {})
    if isinstance(financial, dict):
        financial["rag_case_study_reference"] = (
            f"Real project benchmark ({city_match['title']}): "
            + city_match["summary"].split("\n")[0]
        )
        insights["financial_outlook"] = financial
    return insights


# ── Singleton ──────────────────────────────────────────────────────────────────

_pipeline_instance: Optional[RenovationRAGPipeline] = None


def get_rag_pipeline() -> RenovationRAGPipeline:
    """Return the singleton RenovationRAGPipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RenovationRAGPipeline()
    return _pipeline_instance