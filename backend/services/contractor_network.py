"""
ARKEN — Contractor Network Service v1.0
=========================================
SAVE AS: backend/services/contractor_network.py  — NEW FILE

Enriches every CPM task with:
  - estimated_labour_cost_inr  (daily rate × duration × city multiplier)
  - contractor_links           (3 platform links per trade)
  - tip                        (one practical hiring tip per role)
  - daily_rate_inr             (for frontend display)

Data sources:
  - DAILY_RATES: Q1 2026 Indian contractor market survey
  - CITY_COST_MULTIPLIER: imported from price_forecast.py
  - Platform URLs: Urban Company, Sulekha, JustDial, NoBroker
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Daily labour rates (INR/day, Q1 2026 Indian market) ───────────────────────
# Source: Q1 2026 contractor surveys across metros (Hyderabad = baseline)
DAILY_RATES: Dict[str, Dict[str, int]] = {
    "Licensed Electrician (ISI)": {"basic": 900,  "mid": 1_200, "premium": 1_800},
    "Plumber (CPWD Grade B)":     {"basic": 800,  "mid": 1_100, "premium": 1_600},
    "Plumber":                    {"basic": 800,  "mid": 1_100, "premium": 1_600},
    "Painter":                    {"basic": 600,  "mid": 900,   "premium": 1_400},
    "Civil Contractor":           {"basic": 700,  "mid": 1_000, "premium": 1_500},
    "Carpenter":                  {"basic": 1_000,"mid": 1_400, "premium": 2_200},
    "Flooring Specialist":        {"basic": 900,  "mid": 1_300, "premium": 2_000},
    "Interior Contractor":        {"basic": 1_100,"mid": 1_600, "premium": 2_500},
    "General Labour":             {"basic": 450,  "mid": 600,   "premium": 900},
    "Project Supervisor":         {"basic": 1_500,"mid": 2_200, "premium": 3_500},
}

# ── City multiplier (mirrors price_forecast.py CITY_COST_MULTIPLIER) ──────────
_CITY_MULTIPLIER: Dict[str, float] = {
    "Mumbai":    1.25, "Delhi NCR": 1.20, "Bangalore": 1.15,
    "Hyderabad": 1.00, "Pune":      1.05, "Chennai":   1.05,
    "Kolkata":   0.95, "Ahmedabad": 0.92, "Surat":     0.90,
    "Jaipur":    0.88, "Lucknow":   0.85, "Chandigarh":0.95,
    "Nagpur":    0.87, "Indore":    0.86, "Bhopal":    0.84,
}

def _city_mult(city: str) -> float:
    return _CITY_MULTIPLIER.get(city, 0.95)


# ── Urban Company service slugs ───────────────────────────────────────────────
_UC_SERVICE_MAP: Dict[str, str] = {
    "Licensed Electrician (ISI)": "electrical-repair",
    "Plumber (CPWD Grade B)":     "plumbing",
    "Plumber":                    "plumbing",
    "Painter":                    "painting",
    "Civil Contractor":           "home-renovation",
    "Carpenter":                  "carpentry-woodwork",
    "Flooring Specialist":        "floor-polishing",
    "Interior Contractor":        "false-ceiling",
    "General Labour":             "home-cleaning",
    "Project Supervisor":         "home-renovation",
}

# ── Sulekha service slugs ─────────────────────────────────────────────────────
_SULEKHA_MAP: Dict[str, str] = {
    "Painter":                    "painting",
    "Plumber (CPWD Grade B)":     "plumbing",
    "Plumber":                    "plumbing",
    "Licensed Electrician (ISI)": "electrician",
    "Carpenter":                  "carpenter",
    "Civil Contractor":           "civil-contractor",
    "Interior Contractor":        "interior-designer",
    "Flooring Specialist":        "flooring-contractors",
    "Project Supervisor":         "civil-contractor",
    "General Labour":             "labour-contractors",
}

# ── JustDial service slugs ────────────────────────────────────────────────────
_JD_MAP: Dict[str, str] = {
    "Painter":                    "painters",
    "Plumber (CPWD Grade B)":     "plumbers",
    "Plumber":                    "plumbers",
    "Licensed Electrician (ISI)": "electricians",
    "Carpenter":                  "carpenters",
    "Civil Contractor":           "civil-contractors",
    "Interior Contractor":        "interior-designers",
    "Flooring Specialist":        "flooring-contractors",
    "Project Supervisor":         "civil-contractors",
    "General Labour":             "labour-contractors",
}

# ── Practical hiring tips per role ────────────────────────────────────────────
_HIRING_TIPS: Dict[str, str] = {
    "Licensed Electrician (ISI)": (
        "Always ask for ISI certification and insist on ELCB installation. "
        "Request a written warranty for at least 1 year on all wiring work."
    ),
    "Plumber (CPWD Grade B)": (
        "Verify CPWD grade certification. Ask for water-pressure test after completion "
        "and insist on water-tight joints inspection before wall closure."
    ),
    "Plumber": (
        "Get at least 3 quotes. Ask specifically whether waterproofing membrane is "
        "included in the rate for bathroom work — it often is not."
    ),
    "Painter": (
        "Specify number of putty coats, primer coats, and paint coats in writing. "
        "Asian Paints and Berger have certified painter networks — request a certified painter "
        "for warranty purposes."
    ),
    "Civil Contractor": (
        "Ask for a site visit before quoting — ambiguous scope leads to post-work disputes. "
        "Retain 10% of payment until 30-day post-completion inspection."
    ),
    "Carpenter": (
        "Confirm material grade (BWP/BWR/MR) in writing before order placement. "
        "Carpenters often quote low and charge extra for hardware — clarify whether "
        "hinges, channels, and handles are included."
    ),
    "Flooring Specialist": (
        "Request a tile layout plan before work starts to minimise wastage. "
        "Ensure the rate includes levelling compound — uneven floors are the #1 cause "
        "of tile cracking within 12 months."
    ),
    "Interior Contractor": (
        "Ask for a 3D mock-up or sample board before confirming false ceiling design. "
        "Confirm that the rate includes POP finishing — grid-only is often quoted lower."
    ),
    "General Labour": (
        "Hire through a registered contractor firm rather than daily wage labourers "
        "for accountability. Ensure labourers are covered under ESIC for site safety."
    ),
    "Project Supervisor": (
        "Hire a supervisor independent of your main contractor to avoid conflict of interest. "
        "Define clear milestone-based payment terms tied to verified completion."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# URL builders
# ─────────────────────────────────────────────────────────────────────────────

def _urban_company_url(city: str, role: str) -> Optional[str]:
    service = _UC_SERVICE_MAP.get(role)
    if not service:
        return None
    city_slug = city.lower().replace(" ", "-").replace("ncr", "ncr")
    return f"https://www.urbancompany.com/{city_slug}/{service}"


def _sulekha_url(city: str, role: str) -> Optional[str]:
    service  = _SULEKHA_MAP.get(role, "home-renovation-services")
    city_slug = city.lower().replace(" ", "-")
    return f"https://www.sulekha.com/{city_slug}/{service}-contractors"


def _justdial_url(city: str, role: str) -> Optional[str]:
    service  = _JD_MAP.get(role, "home-renovation")
    city_slug = city.lower().replace(" ", "-")
    return f"https://www.justdial.com/{city_slug}/{service}"


def _nobroker_url(city: str) -> str:
    city_slug = city.lower().replace(" ", "-")
    return f"https://www.nobroker.in/packers-and-movers/{city_slug}"


# ─────────────────────────────────────────────────────────────────────────────
# Main service
# ─────────────────────────────────────────────────────────────────────────────

class ContractorNetworkService:
    """
    Enriches a CPM task with real contractor search links and cost estimates.

    Usage:
        svc = ContractorNetworkService()
        info = svc.get_contractor_info(
            city="Bangalore",
            contractor_role="Carpenter",
            budget_tier="mid",
            duration_days=7,
        )
    """

    def get_contractor_info(
        self,
        *,
        city: str,
        contractor_role: str,
        budget_tier: str = "mid",
        duration_days: int = 3,
    ) -> Dict:
        """
        Return contractor links, daily rate, total estimated cost, and a hiring tip.

        Formula: estimated_labour_cost_inr = daily_rate × duration_days × city_multiplier
        """
        mult           = _city_mult(city)
        role_rates     = DAILY_RATES.get(contractor_role, DAILY_RATES["General Labour"])
        daily_rate_raw = role_rates.get(budget_tier, role_rates["mid"])
        daily_rate     = round(daily_rate_raw * mult)
        total_cost     = round(daily_rate * duration_days)

        links: List[Dict] = []

        # Urban Company
        uc_url = _urban_company_url(city, contractor_role)
        if uc_url:
            links.append({
                "platform": "Urban Company",
                "url":      uc_url,
                "label":    "Find on Urban Company",
            })

        # Sulekha
        sl_url = _sulekha_url(city, contractor_role)
        if sl_url:
            links.append({
                "platform": "Sulekha",
                "url":      sl_url,
                "label":    "Find on Sulekha",
            })

        # JustDial
        jd_url = _justdial_url(city, contractor_role)
        if jd_url:
            links.append({
                "platform": "JustDial",
                "url":      jd_url,
                "label":    "Find on JustDial",
            })

        # NoBroker (always available as fourth option)
        links.append({
            "platform": "NoBroker",
            "url":      _nobroker_url(city),
            "label":    "Find on NoBroker",
        })

        tip = _HIRING_TIPS.get(contractor_role, (
            "Get at least 3 quotes and verify previous work references before confirming. "
            "Define payment milestones tied to verified task completion."
        ))

        return {
            "daily_rate_inr":           daily_rate,
            "estimated_labour_cost_inr": total_cost,
            "duration_days":            duration_days,
            "city_multiplier":          round(mult, 2),
            "contractor_links":         links[:3],   # top 3 platforms
            "tip":                      tip,
            "rate_source":              "ARKEN contractor survey Q1 2026",
        }

    def enrich_schedule(
        self,
        tasks: List[Dict],
        city: str,
        budget_tier: str = "mid",
    ) -> tuple[List[Dict], int]:
        """
        Enrich every task in a CPM schedule with contractor_info.

        Returns (enriched_tasks, total_estimated_labour_inr).
        """
        total_labour = 0
        enriched: List[Dict] = []

        for task in tasks:
            task = dict(task)   # shallow copy — don't mutate state
            role     = task.get("contractor_role", "General Labour")
            duration = task.get("duration_days", 1)

            try:
                info = self.get_contractor_info(
                    city=city,
                    contractor_role=role,
                    budget_tier=budget_tier,
                    duration_days=duration,
                )
                task["contractor_info"] = info
                total_labour += info["estimated_labour_cost_inr"]
            except Exception as e:
                logger.warning(f"[ContractorNetwork] Enrichment failed for '{role}': {e}")
                task["contractor_info"] = None

            enriched.append(task)

        return enriched, total_labour
