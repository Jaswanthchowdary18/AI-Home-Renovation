"""
ARKEN — Project Coordinator Agent v3.0
=======================================
Generates Critical Path Method (CPM) schedules for Indian renovation projects.

v3.0 Changes over v2.0 (FEATURE 2 — Contractor Network integration):
  - generate_schedule() now calls ContractorNetworkService for every task.
  - Each task in the output now contains a "contractor_info" dict with:
      daily_rate_inr, estimated_labour_cost_inr, contractor_links (3 platforms),
      and a practical hiring tip.
  - "total_estimated_labour_inr" added to the schedule summary.
  - ContractorNetworkService import is lazy (graceful degradation if service
    module is missing).

v2.0 features preserved:
  - Area-dependent task durations
  - CITY_LABOR_MULTIPLIER
  - CONTRACTOR_LEAD_TIME_DAYS
  - Risk buffer days (10/15/20% by budget tier)
  - realistic_end_date / best_case_end / worst_case_end
  - Monsoon detection and warning
  - estimate_cost_accuracy_range()
"""

import logging
import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Task library ──────────────────────────────────────────────────────────────
TASK_LIBRARY = {
    "site_assessment": {
        "name": "Site Assessment & Documentation",
        "base_duration": 1, "area_scales": False,
        "dependencies": [], "role": "Project Supervisor",
        "cost_per_sqft": 8, "is_critical": True,
    },
    "demolition": {
        "name": "Selective Demolition",
        "base_duration": 2, "area_scales": False,
        "dependencies": ["site_assessment"], "role": "Civil Contractor",
        "cost_per_sqft": 15, "is_critical": True,
    },
    "electrical_rough": {
        "name": "Electrical Rough-in & Rewiring",
        "base_duration": 3, "area_scales": False,
        "dependencies": ["demolition"], "role": "Licensed Electrician (ISI)",
        "cost_per_sqft": 45, "is_critical": False,
    },
    "plumbing": {
        "name": "Plumbing & Waterproofing",
        "base_duration": 4, "area_scales": False,
        "dependencies": ["demolition"], "role": "Plumber (CPWD Grade B)",
        "cost_per_sqft": 35, "is_critical": False,
    },
    "civil_work": {
        "name": "Masonry & Civil Rectification",
        "base_duration": 3, "area_scales": False,
        "dependencies": ["demolition"], "role": "Civil Contractor",
        "cost_per_sqft": 25, "is_critical": True,
    },
    "false_ceiling": {
        "name": "False Ceiling / POP Work",
        "base_duration": 3, "area_scales": True,
        "dependencies": ["electrical_rough", "civil_work"], "role": "Interior Contractor",
        "cost_per_sqft": 55, "is_critical": False,
    },
    "wall_prep": {
        "name": "Wall Plastering, Putty & Primer",
        "base_duration": 4, "area_scales": True,
        "dependencies": ["civil_work", "plumbing"], "role": "Painter",
        "cost_per_sqft": 30, "is_critical": True,
    },
    "flooring": {
        "name": "Flooring Installation",
        "base_duration": 3, "area_scales": True,
        "dependencies": ["wall_prep"], "role": "Flooring Specialist",
        "cost_per_sqft": 40, "is_critical": True,
    },
    "painting": {
        "name": "Wall & Ceiling Painting (2 coats)",
        "base_duration": 4, "area_scales": True,
        "dependencies": ["false_ceiling", "wall_prep"], "role": "Painter",
        "cost_per_sqft": 28, "is_critical": True,
    },
    "carpentry": {
        "name": "Modular Carpentry & Joinery",
        "base_duration": 5, "area_scales": True,
        "dependencies": ["false_ceiling"], "role": "Carpenter",
        "cost_per_sqft": 120, "is_critical": False,
    },
    "electrical_finish": {
        "name": "Electrical Fixtures & Switches",
        "base_duration": 2, "area_scales": False,
        "dependencies": ["painting", "carpentry"], "role": "Licensed Electrician (ISI)",
        "cost_per_sqft": 20, "is_critical": False,
    },
    "fixtures": {
        "name": "Bathroom / Kitchen Fixtures",
        "base_duration": 2, "area_scales": False,
        "dependencies": ["flooring", "plumbing"], "role": "Plumber",
        "cost_per_sqft": 25, "is_critical": False,
    },
    "cleaning": {
        "name": "Deep Cleaning & Snagging",
        "base_duration": 1, "area_scales": False,
        "dependencies": ["electrical_finish", "fixtures"], "role": "General Labour",
        "cost_per_sqft": 5, "is_critical": True,
    },
    "handover": {
        "name": "Inspection & Handover",
        "base_duration": 1, "area_scales": False,
        "dependencies": ["cleaning"], "role": "Project Supervisor",
        "cost_per_sqft": 5, "is_critical": True,
    },
}

LABOR_EFFICIENCY_BY_MONTH = {
    1: 0.95, 2: 0.95, 3: 0.90,
    4: 0.88, 5: 0.85, 6: 0.80,
    7: 0.75, 8: 0.75, 9: 0.82,
    10: 0.92, 11: 0.90, 12: 0.85,
}

MONSOON_MONTHS = {6, 7, 8, 9}

CITY_LABOR_MULTIPLIER: Dict[str, float] = {
    "Mumbai":    1.30, "Delhi NCR": 1.20, "Bangalore": 1.15,
    "Chennai":   1.05, "Pune":      1.10, "Hyderabad": 1.00,
    "Kolkata":   1.00,
}
_DEFAULT_CITY_MULTIPLIER = 0.95

CONTRACTOR_LEAD_TIME_DAYS: Dict[str, int] = {
    "Licensed Electrician (ISI)": 4,
    "Plumber (CPWD Grade B)":     2,
    "Plumber":                    2,
    "Flooring Specialist":        6,
    "Civil Contractor":           3,
    "Carpenter":                  8,
    "Interior Contractor":        5,
    "Painter":                    1,
    "Project Supervisor":         0,
    "General Labour":             0,
}

RISK_BUFFER_PCT: Dict[str, float] = {
    "basic": 0.10, "mid": 0.15, "premium": 0.20,
}

COST_ACCURACY_RANGE = {"low_mult": 0.85, "high_mult": 1.25}


def _month_name(month: int) -> str:
    return ["", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"][month]


class ProjectCoordinatorAgent:
    """
    Generates full CPM schedule with contractor network enrichment.

    v3.0: every task now carries contractor_info (links + cost estimate + tip).
    """

    def generate_schedule(
        self,
        *,
        area_sqft: float,
        budget_inr: int,
        room_type: str = "bedroom",
        city: str = "Hyderabad",
        budget_tier: str = "mid",
        start_date: Optional[date] = None,
        include_tasks: Optional[List[str]] = None,
    ) -> Dict:
        if start_date is None:
            start_date = date.today() + timedelta(days=7)

        month            = start_date.month
        labor_efficiency = LABOR_EFFICIENCY_BY_MONTH.get(month, 0.90)
        city_multiplier  = CITY_LABOR_MULTIPLIER.get(city, _DEFAULT_CITY_MULTIPLIER)

        task_keys = include_tasks or list(TASK_LIBRARY.keys())
        if room_type == "bedroom":
            task_keys = [t for t in task_keys if t not in ("plumbing", "fixtures")]

        scheduled_tasks, critical_path_days = self._compute_cpm(
            task_keys, area_sqft, labor_efficiency, city_multiplier, start_date,
        )

        base_total_days = max(t["end_day"] for t in scheduled_tasks) if scheduled_tasks else 14

        # ── Monsoon adjustment ────────────────────────────────────────────────
        monsoon_warning: Optional[str] = None
        total_days = base_total_days

        if month in MONSOON_MONTHS:
            eff         = LABOR_EFFICIENCY_BY_MONTH[month]
            monsoon_mult = 1.0 / eff if eff > 0 else 1.25
            total_days   = math.ceil(base_total_days * monsoon_mult)
            monsoon_warning = (
                f"Monsoon season detected (month {month}) — schedule adjusted from "
                f"{base_total_days} to {total_days} days for reduced labor efficiency "
                f"({eff:.0%} efficiency in {_month_name(month)}). "
                "June–September typically adds 20–30% to renovation durations in most Indian cities."
            )
            logger.info(f"[Coordinator] Monsoon adjustment: {base_total_days}d → {total_days}d")

        # ── Risk buffer ───────────────────────────────────────────────────────
        buf_pct          = RISK_BUFFER_PCT.get(budget_tier, 0.15)
        risk_buffer_days = math.ceil(total_days * buf_pct)

        projected_end  = start_date + timedelta(days=total_days)
        realistic_end  = projected_end + timedelta(days=risk_buffer_days)
        worst_case_end = projected_end + timedelta(days=risk_buffer_days * 2)

        schedule_confidence = (
            "The timeline assumes normal contractor availability. "
            "Monsoon season (June–September) typically adds 20–30% to durations in most Indian cities. "
            f"Risk buffer of {risk_buffer_days} days ({buf_pct * 100:.0f}% of base duration) "
            f"applied for {budget_tier}-tier renovation complexity."
        )

        # ── Cost allocation ───────────────────────────────────────────────────
        for task in scheduled_tasks:
            task["estimated_cost_inr"] = int(
                (task["cost_per_sqft"] * area_sqft) *
                (budget_inr / max(self._total_base_cost(area_sqft), 1))
            )

        # ── FEATURE 2: Contractor Network enrichment ──────────────────────────
        total_estimated_labour_inr = 0
        try:
            from services.contractor_network import ContractorNetworkService
            svc = ContractorNetworkService()
            scheduled_tasks, total_estimated_labour_inr = svc.enrich_schedule(
                scheduled_tasks, city=city, budget_tier=budget_tier,
            )
            logger.info(
                f"[Coordinator] Contractor network enriched {len(scheduled_tasks)} tasks; "
                f"total_labour=₹{total_estimated_labour_inr:,}"
            )
        except ImportError:
            logger.warning("[Coordinator] ContractorNetworkService not available — skipping enrichment")
        except Exception as e:
            logger.warning(f"[Coordinator] Contractor enrichment failed: {e}")

        risks           = self._compute_risks(city, month, room_type, area_sqft)
        risk_score      = sum(r["probability"] for r in risks) / len(risks)
        contractor_list = self._build_contractor_list(scheduled_tasks, city)
        cost_range      = self.estimate_cost_accuracy_range(
            budget_inr, room_type, city, month, budget_tier,
        )

        result: Dict = {
            "total_days":              total_days,
            "base_total_days":         base_total_days,
            "critical_path_days":      critical_path_days,
            "start_date":              start_date.isoformat(),
            "projected_end":           projected_end.isoformat(),
            "best_case_end":           projected_end.isoformat(),
            "realistic_end_date":      realistic_end.isoformat(),
            "worst_case_end":          worst_case_end.isoformat(),
            "risk_buffer_days":        risk_buffer_days,
            "schedule_confidence":     schedule_confidence,
            "labor_efficiency":        labor_efficiency,
            "city_labor_multiplier":   city_multiplier,
            "budget_tier":             budget_tier,
            "tasks":                   scheduled_tasks,
            "risk_score":              round(risk_score, 3),
            "risks":                   risks,
            "contractor_list":         contractor_list,
            "cost_accuracy_range":     cost_range,
            # ── FEATURE 2 addition ────────────────────────────────────────────
            "total_estimated_labour_inr": total_estimated_labour_inr,
        }

        if monsoon_warning:
            result["monsoon_warning"] = monsoon_warning

        return result

    # ── CPM ───────────────────────────────────────────────────────────────────

    def _compute_cpm(
        self,
        task_keys: List[str],
        area_sqft: float,
        efficiency: float,
        city_multiplier: float,
        start_date: date,
    ) -> Tuple[List[Dict], int]:
        lib = {k: TASK_LIBRARY[k] for k in task_keys if k in TASK_LIBRARY}

        durations: Dict[str, int] = {}
        for k, t in lib.items():
            base = t["base_duration"]
            if t.get("area_scales", False):
                raw = base + math.ceil(area_sqft / 100)
            else:
                raw = base
            adjusted = max(1, math.ceil(math.ceil(raw / efficiency) * city_multiplier))
            durations[k] = adjusted

        earliest_start: Dict[str, int] = {}
        for task_key in self._topological_sort(lib):
            deps       = lib[task_key]["dependencies"]
            valid_deps = [d for d in deps if d in lib]
            dep_end    = max((earliest_start[d] + durations[d] for d in valid_deps), default=0)
            role       = lib[task_key]["role"]
            lead_days  = CONTRACTOR_LEAD_TIME_DAYS.get(role, 0)
            earliest_start[task_key] = dep_end + lead_days

        scheduled: List[Dict] = []
        for k, t in lib.items():
            es  = earliest_start[k]
            dur = durations[k]
            scheduled.append({
                "id":              k,
                "name":            t["name"],
                "start_day":       es,
                "end_day":         es + dur,
                "duration_days":   dur,
                "dependencies":    [d for d in t["dependencies"] if d in lib],
                "contractor_role": t["role"],
                "contractor_lead_time_days": CONTRACTOR_LEAD_TIME_DAYS.get(t["role"], 0),
                "cost_per_sqft":   t["cost_per_sqft"],
                "is_critical":     t.get("is_critical", False),
                "start_date":      (start_date + timedelta(days=es)).isoformat(),
                "end_date":        (start_date + timedelta(days=es + dur)).isoformat(),
            })

        scheduled.sort(key=lambda x: x["start_day"])
        critical_days = max(
            (t["end_day"] for t in scheduled if t["is_critical"]), default=14
        )
        return scheduled, critical_days

    def _topological_sort(self, lib: Dict) -> List[str]:
        visited: set = set()
        order: List[str] = []

        def dfs(k: str) -> None:
            if k in visited:
                return
            visited.add(k)
            for dep in lib.get(k, {}).get("dependencies", []):
                if dep in lib:
                    dfs(dep)
            order.append(k)

        for key in lib:
            dfs(key)
        return order

    def _total_base_cost(self, area_sqft: float) -> int:
        return int(sum(t["cost_per_sqft"] for t in TASK_LIBRARY.values()) * area_sqft)

    def _compute_risks(
        self, city: str, month: int, room_type: str, area_sqft: float,
    ) -> List[Dict]:
        return [
            {"factor": "Material price volatility (Steel/Cement)", "probability": 0.38, "impact": "Medium", "mitigation": "Lock procurement in first 2 weeks"},
            {"factor": "Skilled labor availability", "probability": 0.52 if month in (7, 8, 10, 11) else 0.35, "impact": "High", "mitigation": "Pre-book contractors 3 weeks ahead"},
            {"factor": "Monsoon weather disruption", "probability": 0.55 if month in MONSOON_MONTHS else 0.10, "impact": "Medium", "mitigation": "Schedule weather-sensitive work in dry months"},
            {"factor": "Municipal permit delay", "probability": 0.44 if city in ("Mumbai", "Delhi NCR") else 0.25, "impact": "Medium", "mitigation": "File NOC application 4 weeks before start"},
            {"factor": "Scope creep / design changes", "probability": 0.40, "impact": "Medium", "mitigation": "Freeze design before contractor mobilization"},
            {"factor": "Contractor no-show / replacement", "probability": 0.30, "impact": "High", "mitigation": "Maintain list of backup contractors per trade"},
        ]

    def _build_contractor_list(self, tasks: List[Dict], city: str) -> List[Dict]:
        seen_roles: Dict[str, Dict] = {}
        for t in tasks:
            role = t["contractor_role"]
            info = t.get("contractor_info") or {}   # populated by enrich_schedule
            if role not in seen_roles:
                seen_roles[role] = {
                    "role":                      role,
                    "required_from":             t["start_date"],
                    "required_until":            t["end_date"],
                    "lead_time_days":            CONTRACTOR_LEAD_TIME_DAYS.get(role, 0),
                    "tasks":                     [t["name"]],
                    "rate_range_inr":            self._rate_range(role, city),
                    # ── from ContractorNetworkService.enrich_schedule ────────
                    "daily_rate_inr":            info.get("daily_rate_inr"),
                    "estimated_labour_cost_inr": info.get("estimated_labour_cost_inr"),
                    "duration_days":             t.get("duration_days", info.get("duration_days")),
                    "contractor_links":          info.get("contractor_links", []),
                    "tip":                       info.get("tip", ""),
                }
            else:
                seen_roles[role]["required_until"] = max(
                    seen_roles[role]["required_until"], t["end_date"]
                )
                seen_roles[role]["tasks"].append(t["name"])
                # Accumulate cost across multiple tasks for same role
                extra_cost = info.get("estimated_labour_cost_inr", 0)
                if extra_cost and seen_roles[role]["estimated_labour_cost_inr"]:
                    seen_roles[role]["estimated_labour_cost_inr"] += extra_cost
                # Accumulate duration
                extra_days = t.get("duration_days", 0)
                if extra_days and seen_roles[role]["duration_days"]:
                    seen_roles[role]["duration_days"] += extra_days
        return list(seen_roles.values())

    @staticmethod
    def _rate_range(role: str, city: str) -> str:
        city_mult = {"Mumbai": 1.4, "Bangalore": 1.3, "Delhi NCR": 1.35,
                     "Hyderabad": 1.0, "Pune": 1.1, "Chennai": 1.05}.get(city, 1.0)
        rates = {
            "Project Supervisor":         (35000, 50000),
            "Civil Contractor":           (18000, 28000),
            "Licensed Electrician (ISI)": (12000, 20000),
            "Plumber (CPWD Grade B)":     (10000, 16000),
            "Interior Contractor":        (22000, 35000),
            "Painter":                    (8000, 14000),
            "Flooring Specialist":        (12000, 18000),
            "Carpenter":                  (15000, 25000),
            "General Labour":             (7000, 10000),
        }
        lo, hi = rates.get(role, (10000, 20000))
        return f"₹{int(lo * city_mult):,} – ₹{int(hi * city_mult):,}/month"

    # ── estimate_cost_accuracy_range (unchanged from v2.0) ────────────────────

    def estimate_cost_accuracy_range(
        self, budget_inr: int, room_type: str, city: str, month: int, budget_tier: str = "mid",
    ) -> Dict:
        low_inr  = int(budget_inr * COST_ACCURACY_RANGE["low_mult"])
        high_inr = int(budget_inr * COST_ACCURACY_RANGE["high_mult"])
        overrun_factors: List[str] = []

        if room_type in ("kitchen", "bathroom"):
            overrun_factors.append(
                "Hidden plumbing defects discovered during demolition — common in properties >10 years old (adds 5–15%)"
            )
        if room_type == "bathroom":
            overrun_factors.append(
                "Waterproofing failures requiring full re-application — occurs in ~30% of bathroom renovations"
            )
        if room_type in ("full_home", "living_room"):
            overrun_factors.append(
                "False ceiling height adjustment after electrical routing — adds 2–3 extra days and 3–5% material cost"
            )
        if city in ("Mumbai", "Delhi NCR"):
            overrun_factors.append(
                f"{city}: society NOC or municipal approval delays (adds 1–2 weeks and compliance costs)"
            )
        if month in MONSOON_MONTHS:
            overrun_factors.append(
                "Monsoon season: material deliveries delayed 3–7 days; drying times extended 30–50%"
            )
        if budget_tier == "premium":
            overrun_factors.append(
                "Premium-tier: imported materials have 4–6 week lead times; customs delays can add 8–12%"
            )
        elif budget_tier == "basic":
            overrun_factors.append(
                "Basic-tier: contractor substitutions more frequent — replacement sourcing adds 3–5 days"
            )
        overrun_factors.append(
            "Scope creep from client change requests — industry average adds 8–12% (CIDC 2024)"
        )

        return {
            "cost_estimate_inr": budget_inr,
            "cost_low_inr":      low_inr,
            "cost_high_inr":     high_inr,
            "low_note":          "-15% assumes smooth execution, no design changes, quick contractor availability",
            "high_note":         "+25% accounts for typical Indian renovation overruns and unforeseen conditions",
            "overrun_factors":   overrun_factors[:5],
            "source":            "CIDC India overrun statistics 2024, ANAROCK contractor survey Q4 2024",
        }