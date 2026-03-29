"""
ARKEN — UserGoalAgent v1.1
===========================
v1.1 Changes over v1.0:
  - Fuzzy + multilingual intent parsing: substring matching replaces exact word match
  - ROOM_TYPE_PATTERNS: detects room type from typos and Hindi/Hinglish words
    e.g. "bedrrom" → bedroom, "rasoi" → kitchen, "gusalkhana" → bathroom
  - GOAL_KEYWORDS now uses "kw in intent_text" (substring) not word-boundary match
  - CONSTRAINT_KEYWORDS extended with Hindi/Hinglish: kiraya, bechna, jaldi, etc.
  - _normalize_intent() strips punctuation and extra whitespace before matching
  All output keys and signatures are fully backward compatible.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ── Goal keyword mapping (substring matching — order matters: first match wins) ─

GOAL_KEYWORDS = {
    "sell":       "maximise_resale_value",
    "resale":     "maximise_resale_value",
    "resell":     "maximise_resale_value",
    "bechna":     "maximise_resale_value",   # Hindi: "to sell"
    "value":      "maximise_resale_value",
    "rent":       "maximise_rental_yield",
    "rental":     "maximise_rental_yield",
    "tenant":     "maximise_rental_yield",
    "yield":      "maximise_rental_yield",
    "kiraya":     "maximise_rental_yield",   # Hindi: "rent"
    "live":       "personal_comfort",
    "home":       "personal_comfort",
    "family":     "personal_comfort",
    "comfort":    "personal_comfort",
    "aesthetic":  "aesthetic_refresh",
    "look":       "aesthetic_refresh",
    "style":      "aesthetic_refresh",
    "refresh":    "aesthetic_refresh",
    "cost":       "cost_optimisation",
    "budget":     "cost_optimisation",
    "cheap":      "cost_optimisation",
    "affordable": "cost_optimisation",
    "quick":      "quick_refresh",
    "fast":       "quick_refresh",
    "jaldi":      "quick_refresh",           # Hindi: "quickly/fast"
    "minimal":    "quick_refresh",
    "luxury":     "luxury_upgrade",
    "premium":    "luxury_upgrade",
    "high-end":   "luxury_upgrade",
    "invest":     "investment_optimisation",
    "roi":        "investment_optimisation",
    "return":     "investment_optimisation",
}

CONSTRAINT_KEYWORDS = {
    "no structural": "no_structural_changes",
    "keep wall":     "retain_existing_walls",
    "keep floor":    "retain_existing_flooring",
    "no demolit":    "no_demolition",
    "budget":        "budget_constrained",
    "kam budget":    "budget_constrained",   # Hindi/Hinglish: "low budget"
    "thoda kam":     "budget_constrained",   # Hindi: "a little less"
    "quick":         "time_constrained",
    "jaldi":         "time_constrained",     # Hindi: "quickly"
    "3 week":        "time_constrained",
    "4 week":        "time_constrained",
    "tenant":        "tenant_occupied",
    "occupied":      "tenant_occupied",
    "kiraya":        "maximise_rental_yield", # Hindi: "rent" — also a goal signal
    "bechna":        "maximise_resale_value", # Hindi: "to sell" — also a goal signal
}

# ── Room type patterns (substring matching, handles typos + Hindi) ─────────────

ROOM_TYPE_PATTERNS = {
    "bedroom":     ["bedroom", "bedrm", "bedrrom", "master bed", "sleeping",
                    "kamra", "sone ka", "room"],
    "kitchen":     ["kitchen", "kichen", "cooking", "modular", "rasoi", "chakla"],
    "bathroom":    ["bathroom", "toilet", "washroom", "bath", "gusalkhana", "wc"],
    "living_room": ["living", "hall", "drawing", "baithak", "drawing room", "lounge"],
    "full_home":   ["entire", "whole", "full", "complete", "poora", "ghar", "flat", "house"],
}


def _normalize_intent(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # replace punctuation with space
    text = re.sub(r"\s+", " ", text).strip()
    return text


class UserGoalAgent:
    """
    Extracts structured goals and constraints from free-text user intent.
    v1.1: fuzzy + Hindi/Hinglish substring matching for robust intent parsing.
    Also loads memory context for personalised recommendations.
    """

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        name = "user_goal_agent"

        try:
            raw_intent  = state.get("user_intent") or ""
            intent_text = _normalize_intent(raw_intent)
            theme       = state.get("theme", "Modern Minimalist")
            room_type   = state.get("room_type", "bedroom")
            budget_tier = state.get("budget_tier", "mid")
            budget_inr  = state.get("budget_inr", 750_000)
            user_id     = state.get("user_id", "anonymous")

            # ── Detect room type from intent (override state default if found) ──
            for rt, patterns in ROOM_TYPE_PATTERNS.items():
                if any(pat in intent_text for pat in patterns):
                    room_type = rt
                    break

            # ── Extract primary goal (substring match — first match wins) ──────
            goal = "personal_comfort"
            for kw, g in GOAL_KEYWORDS.items():
                if kw in intent_text:
                    goal = g
                    break

            priority = (
                "roi"
                if goal in ("maximise_resale_value", "maximise_rental_yield", "investment_optimisation")
                else "rental" if goal == "maximise_rental_yield"
                else "aesthetics"
            )

            # ── Extract constraints (substring match) ──────────────────────────
            constraints = []
            for kw, constraint in CONSTRAINT_KEYWORDS.items():
                if kw in intent_text and constraint not in constraints:
                    constraints.append(constraint)

            # ── Extract keywords (stopword filter) ────────────────────────────
            stop_words = {"i", "want", "to", "the", "a", "an", "my", "is", "in",
                          "for", "and", "or", "ka", "ki", "ke", "hai", "hain", "ko"}
            keywords = [
                w for w in intent_text.split()
                if len(w) > 3 and w not in stop_words
            ][:10]

            # ── Load memory ────────────────────────────────────────────────────
            memory_ctx: Dict[str, Any] = {}
            try:
                from memory.agent_memory import agent_memory
                memory_ctx = await agent_memory.recall(user_id, query=raw_intent)
            except Exception as me:
                logger.warning(f"[user_goal_agent] Memory recall failed: {me}")
                memory_ctx = {}

            user_goals = {
                "primary_goal":       goal,
                "priority":           priority,
                "style_preference":   theme,
                "room_type":          room_type,
                "budget_tier":        budget_tier,
                "constraints":        constraints,
                "extracted_keywords": keywords,
                "confidence":         0.85 if intent_text else 0.5,
            }

            parsed_intent = {
                "goal":             goal,
                "priority":         priority,
                "style_preference": theme,
                "room_type":        room_type,
                "budget_tier":      budget_tier,
            }

            updates: Dict[str, Any] = {
                "user_goals":              user_goals,
                "parsed_intent":           parsed_intent,
                "memory_context":          memory_ctx.get("memory_context", ""),
                "past_budget_constraints": memory_ctx.get("past_budget_constraints", []),
                "past_design_preferences": memory_ctx.get("past_design_preferences", []),
                "past_renovation_goals":   memory_ctx.get("past_renovation_goals", []),
            }

            logger.info(
                f"[user_goal_agent] goal={goal} room_type={room_type} priority={priority} "
                f"constraints={constraints} memory_sessions={memory_ctx.get('session_count', 0)}"
            )

        except Exception as e:
            logger.error(f"[user_goal_agent] Error: {e}", exc_info=True)
            updates = {
                "user_goals": {
                    "primary_goal":       "personal_comfort",
                    "priority":           "aesthetics",
                    "style_preference":   state.get("theme", "Modern Minimalist"),
                    "room_type":          state.get("room_type", "bedroom"),
                    "budget_tier":        state.get("budget_tier", "mid"),
                    "constraints":        [],
                    "extracted_keywords": [],
                    "confidence":         0.3,
                },
                "parsed_intent": {"goal": "personal_comfort", "priority": "aesthetics"},
                "memory_context":          "",
                "past_budget_constraints": [],
                "past_design_preferences": [],
                "past_renovation_goals":   [],
                "errors": (state.get("errors") or []) + [f"user_goal_agent: {e}"],
            }

        # Record timing
        elapsed = round(time.perf_counter() - t0, 3)
        timings = dict(state.get("agent_timings") or {})
        timings[name] = elapsed
        updates["agent_timings"] = timings

        completed = list(state.get("completed_agents") or [])
        if name not in completed:
            completed.append(name)
        updates["completed_agents"] = completed

        return updates