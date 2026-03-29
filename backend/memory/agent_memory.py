"""
ARKEN — Lightweight Agent Memory System v1.0
============================================
Stores and retrieves per-user renovation memory:
  - Previous renovation plans
  - Budget constraints history
  - Design preferences
  - Goals / intent history

Storage backends:
  Primary:  ChromaDB (already in project via vector_store.py)
  Fallback: In-memory dict + JSON file persistence

Usage:
    from memory.agent_memory import agent_memory
    await agent_memory.save(user_id, project_id, memory_data)
    ctx = await agent_memory.recall(user_id)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MEMORY_FILE = Path(os.getenv("ARKEN_MEMORY_PATH", "/tmp/arken_memory.json"))


class AgentMemory:
    """
    Lightweight persistent memory for the ARKEN multi-agent system.
    Remembers past renovation plans, budgets, and design preferences per user.
    Falls back gracefully when vector store is unavailable.
    """

    def __init__(self):
        self._cache: Dict[str, List[Dict]] = {}
        self._load_from_disk()

    # ── Public API ────────────────────────────────────────────────────────

    async def save(
        self,
        user_id: str,
        project_id: str,
        memory_data: Dict[str, Any],
    ) -> bool:
        """
        Persist a renovation session to memory.
        memory_data should include: budget_inr, city, theme, budget_tier, goals, roi_pct.
        """
        try:
            record = {
                "project_id": project_id,
                "timestamp": datetime.utcnow().isoformat(),
                "budget_inr": memory_data.get("budget_inr", 0),
                "city": memory_data.get("city", ""),
                "theme": memory_data.get("theme", ""),
                "budget_tier": memory_data.get("budget_tier", "mid"),
                "room_type": memory_data.get("room_type", "bedroom"),
                "goals": memory_data.get("goals", []),
                "roi_pct": memory_data.get("roi_pct", 0),
                "design_preferences": memory_data.get("design_preferences", []),
                "style_label": memory_data.get("style_label", ""),
            }
            if user_id not in self._cache:
                self._cache[user_id] = []
            # Keep last 10 sessions per user
            self._cache[user_id] = ([record] + self._cache[user_id])[:10]
            self._save_to_disk()

            # Also persist to vector store for semantic retrieval
            await self._upsert_vector(user_id, project_id, record)
            return True
        except Exception as e:
            logger.warning(f"[memory] save failed: {e}")
            return False

    async def recall(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant past sessions for a user.
        Returns structured context for agent injection.
        """
        sessions = self._cache.get(user_id, [])
        if not sessions:
            return self._empty_context()

        recent = sessions[:limit]

        # Extract patterns
        budgets = [s.get("budget_inr", 0) for s in recent if s.get("budget_inr")]
        cities = list({s.get("city") for s in recent if s.get("city")})
        themes = list({s.get("theme") for s in recent if s.get("theme")})
        budget_tiers = list({s.get("budget_tier") for s in recent if s.get("budget_tier")})
        all_goals: List[str] = []
        all_prefs: List[str] = []
        for s in recent:
            all_goals.extend(s.get("goals", []))
            all_prefs.extend(s.get("design_preferences", []))

        avg_budget = int(sum(budgets) / len(budgets)) if budgets else 0
        roi_scores = [s.get("roi_pct", 0) for s in recent if s.get("roi_pct")]
        avg_roi = round(sum(roi_scores) / len(roi_scores), 1) if roi_scores else 0

        summary = []
        if avg_budget:
            summary.append(f"Average past budget: ₹{avg_budget:,}")
        if cities:
            summary.append(f"Cities worked in: {', '.join(cities)}")
        if themes:
            summary.append(f"Preferred styles: {', '.join(themes)}")
        if all_goals:
            summary.append(f"Past goals: {'; '.join(list(set(all_goals))[:4])}")
        if avg_roi:
            summary.append(f"Average ROI from past projects: {avg_roi}%")

        return {
            "has_history": True,
            "session_count": len(sessions),
            "recent_sessions": recent,
            "past_budgets": budgets,
            "past_budget_constraints": budgets,
            "avg_budget_inr": avg_budget,
            "preferred_cities": cities,
            "preferred_themes": themes,
            "preferred_budget_tiers": budget_tiers,
            "past_design_preferences": list(set(all_prefs))[:8],
            "past_renovation_goals": list(set(all_goals))[:6],
            "avg_roi_pct": avg_roi,
            "memory_summary": " | ".join(summary),
            "memory_context": "\n".join(summary),
        }

    async def get_budget_constraints(self, user_id: str) -> List[int]:
        """Return list of past budget_inr values for this user."""
        sessions = self._cache.get(user_id, [])
        return [s["budget_inr"] for s in sessions if s.get("budget_inr")]

    async def get_design_preferences(self, user_id: str) -> List[str]:
        """Return accumulated design preferences."""
        sessions = self._cache.get(user_id, [])
        prefs: List[str] = []
        for s in sessions:
            prefs.extend(s.get("design_preferences", []))
        return list(set(prefs))[:10]

    # ── Internals ─────────────────────────────────────────────────────────

    def _load_from_disk(self):
        try:
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE) as f:
                    self._cache = json.load(f)
                logger.info(f"[memory] Loaded {len(self._cache)} user memories from {MEMORY_FILE}")
        except Exception as e:
            logger.warning(f"[memory] Could not load from disk: {e}")
            self._cache = {}

    def _save_to_disk(self):
        try:
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(MEMORY_FILE, "w") as f:
                json.dump(self._cache, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[memory] Could not save to disk: {e}")

    async def _upsert_vector(
        self, user_id: str, project_id: str, record: Dict
    ) -> None:
        """Store memory record in vector store for semantic recall."""
        try:
            from services.vector_store import vector_store
            doc = (
                f"User {user_id} renovation: "
                f"budget={record.get('budget_inr',0)} city={record.get('city','')} "
                f"theme={record.get('theme','')} tier={record.get('budget_tier','')} "
                f"goals={record.get('goals',[])} roi={record.get('roi_pct',0)}"
            )
            await vector_store.upsert_insights(
                f"memory_{user_id}_{project_id}",
                {"summary_headline": doc, "financial_outlook": {"projected_roi": str(record.get("roi_pct", 0))}},
                metadata={"user_id": user_id, "type": "memory", **{
                    k: str(v) for k, v in record.items() if isinstance(v, (str, int, float))
                }},
            )
        except Exception:
            pass  # Vector store is optional

    @staticmethod
    def _empty_context() -> Dict[str, Any]:
        return {
            "has_history": False,
            "session_count": 0,
            "recent_sessions": [],
            "past_budgets": [],
            "past_budget_constraints": [],
            "avg_budget_inr": 0,
            "preferred_cities": [],
            "preferred_themes": [],
            "preferred_budget_tiers": [],
            "past_design_preferences": [],
            "past_renovation_goals": [],
            "avg_roi_pct": 0,
            "memory_summary": "",
            "memory_context": "",
        }


# Singleton
agent_memory = AgentMemory()
