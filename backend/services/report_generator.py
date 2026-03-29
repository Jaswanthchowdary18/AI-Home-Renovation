"""
ARKEN — PDF Renovation Report Generator v2.0
=============================================
Generates a professional, downloadable PDF renovation report from structured pipeline data.

v2.0 upgrades:
  - Section 9: Derived Insights (data-backed facts with source attribution)
  - Section 10: Market Benchmark (city/room percentile analysis)
  - Section 11: Optimal Budget Allocation (per-category ROI-weighted split)
  - Section 12: Material Price Signals (procurement recommendations)
  - Decision scores table (quick wins highlighted)
  - All new sections are backward-compatible — skipped if data not present

Dependencies: reportlab (pure Python, no system deps)
Falls back to HTML if reportlab unavailable.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates a professional PDF renovation report.
    Uses ReportLab with graceful HTML fallback.
    """

    BRAND_COLOUR = (0.18, 0.42, 0.78)
    ACCENT_COLOUR = (0.95, 0.55, 0.10)
    DARK_GREY = (0.15, 0.15, 0.15)
    LIGHT_GREY = (0.94, 0.94, 0.94)
    WHITE = (1.0, 1.0, 1.0)

    def generate(
        self,
        report: Dict[str, Any],
        project_id: str,
        render_image_bytes: Optional[bytes] = None,
    ) -> bytes:
        try:
            return self._generate_pdf(report, project_id, render_image_bytes)
        except Exception as e:
            logger.warning(f"PDF generation failed ({e}), falling back to HTML report")
            return self._generate_html_fallback(report, project_id).encode("utf-8")

    def _generate_pdf(
        self,
        report: Dict[str, Any],
        project_id: str,
        render_image_bytes: Optional[bytes],
    ) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, white, black
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage, KeepTogether,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=18*mm, rightMargin=18*mm,
            topMargin=20*mm, bottomMargin=18*mm,
            title=f"ARKEN Renovation Report — {project_id[:8].upper()}",
        )

        styles = getSampleStyleSheet()
        blue = HexColor("#2e6bc5")
        orange = HexColor("#f28c1a")
        dark = HexColor("#252525")
        light_bg = HexColor("#f5f7fa")
        green = HexColor("#27ae60")
        red = HexColor("#c0392b")
        amber = HexColor("#e67e22")

        h1 = ParagraphStyle("H1", parent=styles["Normal"], fontSize=22, textColor=blue,
                             spaceAfter=4, fontName="Helvetica-Bold", alignment=TA_LEFT)
        h2 = ParagraphStyle("H2", parent=styles["Normal"], fontSize=14, textColor=blue,
                             spaceAfter=3, spaceBefore=10, fontName="Helvetica-Bold")
        h3 = ParagraphStyle("H3", parent=styles["Normal"], fontSize=11, textColor=dark,
                             spaceAfter=2, spaceBefore=5, fontName="Helvetica-Bold")
        body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, textColor=dark,
                               spaceAfter=2, leading=13)
        small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8.5, textColor=HexColor("#555555"),
                                spaceAfter=1, leading=11)
        caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, textColor=HexColor("#777777"),
                                  alignment=TA_CENTER, spaceAfter=2)
        highlight = ParagraphStyle("Highlight", parent=styles["Normal"], fontSize=10,
                                   textColor=orange, fontName="Helvetica-Bold", spaceAfter=2)
        kpi_style = ParagraphStyle("kpi_val", fontSize=16, fontName="Helvetica-Bold",
                                   textColor=blue, alignment=TA_CENTER)
        kpi_label = ParagraphStyle("kpi_lbl", fontSize=7.5, textColor=HexColor("#666666"), alignment=TA_CENTER)

        story = []

        # ── Header ────────────────────────────────────────────────────────
        header_data = [[
            Paragraph("<b>ARKEN</b>", ParagraphStyle("brand", fontSize=28, textColor=blue, fontName="Helvetica-Bold")),
            Paragraph(
                f"<font color='#888888' size='8'>Renovation Intelligence Report</font><br/>"
                f"<font color='#555555' size='8'>Project ID: {project_id[:12].upper()}</font><br/>"
                f"<font color='#555555' size='8'>Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}</font>",
                ParagraphStyle("hdr_right", fontSize=8, alignment=TA_RIGHT)
            ),
        ]]
        header_table = Table(header_data, colWidths=["50%", "50%"])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, -1), 1.5, blue),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 6*mm))

        # ── Headline ──────────────────────────────────────────────────────
        headline = report.get("summary_headline", "Renovation Report")
        story.append(Paragraph(headline, h1))
        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#dddddd")))
        story.append(Spacer(1, 4*mm))

        # ── KPI boxes ─────────────────────────────────────────────────────
        roi = report.get("roi_forecast", {})
        cost = report.get("cost_estimate", {})
        layout = report.get("layout_analysis", {})
        style_info = report.get("style_analysis", {})
        ie = report.get("insight_engine", {})

        kpis = [
            (roi.get("roi_percentage", "—"), "Projected ROI"),
            (cost.get("total", "—"), "Total Cost"),
            (roi.get("equity_gain", "—"), "Equity Gain"),
            (str(report.get("renovation_timeline", {}).get("total_days", "—")) + " days", "Timeline"),
            (layout.get("layout_score", "—"), "Layout Score"),
        ]
        kpi_rows = [
            [Paragraph(str(v), kpi_style) for v, _ in kpis],
            [Paragraph(str(l), kpi_label) for _, l in kpis],
        ]
        kpi_table = Table(kpi_rows, colWidths=["20%"] * 5)
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), light_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
            ("LINEAFTER", (0, 0), (-2, -1), 0.5, HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 6*mm))

        # ── 1. Room Analysis ──────────────────────────────────────────────
        room = report.get("room_analysis", {})
        story.append(Paragraph("1. Room Analysis", h2))
        room_rows = [
            ["Room Type", room.get("room_type", "—").title(), "Condition", room.get("room_condition", "—").title()],
            ["Wall Treatment", room.get("wall_treatment", "—"), "Floor", room.get("floor_material", "—")],
            ["Ceiling", room.get("ceiling_type", "—"), "Natural Light", room.get("natural_light", "—").title()],
            ["Quality Tier", room.get("quality_tier", "—").title(), "Style", style_info.get("detected_style", "—")],
        ]
        story.append(self._two_col_table(room_rows))
        story.append(Spacer(1, 2*mm))

        observations = room.get("specific_observations", [])
        if observations:
            story.append(Paragraph("Key Observations:", h3))
            for obs in observations[:6]:
                story.append(Paragraph(f"• {obs}", body))
        story.append(Spacer(1, 4*mm))

        # ── 2. Style & Layout ─────────────────────────────────────────────
        story.append(Paragraph("2. Style Detection & Layout Analysis", h2))
        layout_rows = [
            ["Detected Style", style_info.get("detected_style", "—"),
             "Confidence", f"{float(style_info.get('confidence', 0)) * 100:.0f}%"],
            ["Layout Score", layout.get("layout_score", "—"),
             "Walkable Space", layout.get("walkable_space", "—")],
            ["Lighting Score", layout.get("lighting_score", "—"),
             "Furniture Density", str(layout.get("furniture_density", "—")).title()],
        ]
        story.append(self._two_col_table(layout_rows))

        for issue in layout.get("issues", [])[:4]:
            story.append(Paragraph(f"⚠ {issue}",
                ParagraphStyle("warn", parent=body, textColor=HexColor("#e67e22"))))
        for s in layout.get("suggestions", [])[:3]:
            story.append(Paragraph(f"✓ {s}",
                ParagraphStyle("sugg", parent=body, textColor=HexColor("#27ae60"))))
        story.append(Spacer(1, 4*mm))

        # ── 3. Design Recommendations ─────────────────────────────────────
        recs = report.get("design_recommendations", [])
        if recs:
            story.append(Paragraph("3. Design Recommendations (Explainable AI)", h2))
            for i, rec in enumerate(recs[:5], 1):
                prio_colour = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}.get(
                    rec.get("priority", "medium"), "#f39c12"
                )
                story.append(Paragraph(
                    f"{i}. {rec.get('title', '—')} "
                    f"<font color='{prio_colour}'>[{rec.get('priority', '').upper()} PRIORITY]</font>",
                    h3
                ))
                for reason in rec.get("reasoning", [])[:3]:
                    story.append(Paragraph(f"  → {reason}", small))
                cost_line = rec.get("estimated_cost", "")
                roi_line = rec.get("roi_impact", "")
                if cost_line:
                    story.append(Paragraph(f"  Cost: {cost_line}  |  ROI Impact: {roi_line}", small))
                story.append(Spacer(1, 2*mm))
            story.append(Spacer(1, 2*mm))

        # ── 4. Cost Estimate ──────────────────────────────────────────────
        story.append(Paragraph("4. Renovation Cost Estimate", h2))
        cost_rows = [
            ["Category", "Amount (INR)"],
            ["Materials", cost.get("materials", "—")],
            ["Labour", cost.get("labour", "—")],
            ["Supervision", cost.get("supervision", "—")],
            ["Contingency (3%)", cost.get("contingency", "—")],
        ]
        cost_table = Table(cost_rows, colWidths=["60%", "40%"])
        cost_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), blue),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, light_bg]),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(cost_table)
        total_table = Table([["TOTAL", cost.get("total", "—")]], colWidths=["60%", "40%"])
        total_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), orange),
            ("TEXTCOLOR", (0, 0), (-1, -1), white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(total_table)
        story.append(Spacer(1, 4*mm))

        # ── 5. ROI Forecast ───────────────────────────────────────────────
        story.append(Paragraph("5. ROI Forecast", h2))
        roi_rows = [
            ["Pre-Renovation Value", roi.get("pre_reno_value", "—"), "Post-Renovation Value", roi.get("post_reno_value", "—")],
            ["Projected ROI", roi.get("roi_percentage", "—"), "Equity Gain", roi.get("equity_gain", "—")],
            ["Payback Period", roi.get("payback_period", "—"), "Rental Yield Δ", roi.get("rental_yield_improvement", "—")],
            ["Model Type", roi.get("model_type", "—"), "Confidence", roi.get("model_confidence", "—")],
        ]
        story.append(self._two_col_table(roi_rows))

        # ROI explanation from v2.0 model
        roi_explanation = roi.get("explanation", {})
        if isinstance(roi_explanation, dict) and roi_explanation.get("roi_narrative"):
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph("ROI Explanation:", h3))
            story.append(Paragraph(roi_explanation["roi_narrative"], body))
            drivers = roi_explanation.get("primary_drivers", [])
            if drivers:
                for d in drivers[:4]:
                    if isinstance(d, dict):
                        story.append(Paragraph(
                            f"  ✦ <b>{d.get('driver', '')}</b> ({d.get('value', '')}) — {d.get('explanation', '')}",
                            small,
                        ))
        story.append(Spacer(1, 4*mm))

        # ── 6. Renovation Timeline ────────────────────────────────────────
        tl = report.get("renovation_timeline", {})
        story.append(Paragraph("6. Renovation Timeline", h2))
        story.append(Paragraph(
            f"Total Duration: <b>{tl.get('total_days', '—')} days</b> "
            f"({tl.get('calendar_weeks', '—')} calendar weeks) | "
            f"Crew Size: {tl.get('workers_required', 3)} workers",
            body
        ))
        story.append(Spacer(1, 2*mm))
        phases = tl.get("phases", [])
        if phases:
            phase_data = [["Phase", "Duration", "Execution"]]
            for phase in phases:
                phase_data.append([
                    phase.get("phase", ""),
                    f"{phase.get('days', 0)} days",
                    "Parallel" if phase.get("parallel") else "Sequential",
                ])
            phase_table = Table(phase_data, colWidths=["55%", "20%", "25%"])
            phase_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), blue),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
            ]))
            story.append(phase_table)
        story.append(Spacer(1, 4*mm))

        # ── 7. Market Intelligence ────────────────────────────────────────
        mkt = report.get("market_intelligence", {})
        loc = mkt.get("location_context", {})
        budget_info = mkt.get("budget_assessment", {})
        if loc or budget_info:
            story.append(Paragraph("7. Market Intelligence", h2))
            city = mkt.get("city", "")
            if loc.get("trend"):
                story.append(Paragraph(f"{city}: {loc.get('trend', '')}", body))
            mkt_rows = []
            if loc.get("appreciation_5yr_pct"):
                mkt_rows.append(["5-Year Appreciation", f"{loc['appreciation_5yr_pct']}%",
                                  "Rental Yield", f"{loc.get('rental_yield_pct', '—')}%"])
            if loc.get("avg_psf_inr"):
                mkt_rows.append(["Avg Price/SqFt", f"₹{loc['avg_psf_inr']:,}",
                                  "Market Tier", f"Tier {loc.get('market_tier', '—')}"])
            if budget_info.get("what_it_covers"):
                mkt_rows.append(["Budget Covers", budget_info.get("what_it_covers", ""),
                                  "ROI Potential", budget_info.get("roi_potential", "")])
            if mkt_rows:
                story.append(self._two_col_table(mkt_rows))

        # ── 8. AI Insight Engine ──────────────────────────────────────────
        if ie:
            story.append(Spacer(1, 4*mm))
            story.append(Paragraph("8. AI Insight Engine", h2))

            ie_scores = [
                (f"{ie.get('renovation_priority_score', '—')}/100", "Renovation Priority"),
                (f"{ie.get('cost_effectiveness_score', '—')}/100", "Cost Effectiveness"),
                (f"{ie.get('roi_score', '—')}", "ROI Score"),
                (f"{ie.get('overall_insight_score', '—')}/100", "Overall Score"),
            ]
            score_rows = [
                [Paragraph(str(v), kpi_style) for v, _ in ie_scores],
                [Paragraph(str(l), kpi_label) for _, l in ie_scores],
            ]
            score_table = Table(score_rows, colWidths=["25%"] * 4)
            score_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), light_bg),
                ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
                ("LINEAFTER", (0, 0), (-2, -1), 0.5, HexColor("#cccccc")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(score_table)
            story.append(Spacer(1, 3*mm))

            if ie.get("summary"):
                story.append(Paragraph(ie["summary"], body))
                story.append(Spacer(1, 2*mm))

            # Priority repairs
            priority_repairs = ie.get("priority_repairs", [])
            if priority_repairs:
                story.append(Paragraph("Priority Repair List:", h3))
                urgency_colours = {
                    "critical": "#e74c3c", "high": "#e67e22",
                    "medium": "#f39c12", "low": "#27ae60",
                }
                for pr in priority_repairs[:6]:
                    if not isinstance(pr, dict):
                        continue
                    urg = pr.get("urgency", "medium")
                    urg_colour = urgency_colours.get(urg, "#f39c12")
                    story.append(Paragraph(
                        f"<b>{pr.get('rank', '')}. {pr.get('action', '')}</b> "
                        f"<font color='{urg_colour}'>[{urg.upper()}]</font> "
                        f"— {pr.get('category', '').title()} | "
                        f"Est. Cost: {pr.get('estimated_cost_label', 'TBD')} | "
                        f"Impact: {pr.get('impact_score', '—')}/10",
                        small,
                    ))
                    if pr.get("reasoning"):
                        story.append(Paragraph(f"  → {pr['reasoning'][:120]}", small))
                story.append(Spacer(1, 2*mm))

            # Budget strategy
            bs = ie.get("budget_strategy", {})
            if isinstance(bs, dict) and bs.get("strategy_name"):
                story.append(Paragraph("Budget Strategy:", h3))
                story.append(Paragraph(
                    f"<b>{bs.get('strategy_name', '')}</b>: {bs.get('description', '')}",
                    body,
                ))
                alloc = bs.get("allocation", {})
                if alloc:
                    alloc_rows = [[k.title(), f"₹{v:,}"] for k, v in alloc.items() if v]
                    if alloc_rows:
                        alloc_table = Table(alloc_rows, colWidths=["50%", "50%"])
                        alloc_table.setStyle(TableStyle([
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, light_bg]),
                            ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
                            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ]))
                        story.append(alloc_table)
                story.append(Spacer(1, 2*mm))

            # ROI insight
            roi_ins = ie.get("roi_insight", {})
            if isinstance(roi_ins, dict) and roi_ins.get("interpretation"):
                story.append(Paragraph("ROI Analysis:", h3))
                story.append(Paragraph(roi_ins["interpretation"], body))
                roi_detail_rows = []
                if roi_ins.get("value_per_rupee"):
                    roi_detail_rows.append([
                        "Value per ₹1 Spent", f"₹{roi_ins['value_per_rupee']:.2f}",
                        "Payback Period", roi_ins.get("payback_period", "—"),
                    ])
                if roi_ins.get("rental_yield_improvement"):
                    roi_detail_rows.append([
                        "Rental Yield Improvement", roi_ins["rental_yield_improvement"],
                        "Confidence", f"{float(roi_ins.get('confidence', 0)) * 100:.0f}%",
                    ])
                if roi_detail_rows:
                    story.append(self._two_col_table(roi_detail_rows))
                story.append(Spacer(1, 2*mm))

            # Renovation sequence
            reno_seq = ie.get("renovation_sequence", [])
            if reno_seq:
                story.append(Paragraph("Recommended Renovation Sequence:", h3))
                seq_data = [["Step", "Phase", "Duration", "Notes"]]
                for phase in reno_seq:
                    if not isinstance(phase, dict):
                        continue
                    seq_data.append([
                        str(phase.get("step", "")),
                        phase.get("phase", ""),
                        f"{phase.get('duration_days', 0)} days",
                        "✓ Parallel" if phase.get("can_parallel") else "→ Sequential",
                    ])
                seq_table = Table(seq_data, colWidths=["8%", "50%", "18%", "24%"])
                seq_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), blue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, light_bg]),
                    ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("ALIGN", (2, 0), (3, -1), "CENTER"),
                ]))
                story.append(seq_table)
                story.append(Spacer(1, 2*mm))

            # Key wins / risk flags
            key_wins = ie.get("key_wins", [])
            risk_flags = ie.get("risk_flags", [])
            if key_wins:
                story.append(Paragraph(
                    "<b>Key Wins:</b><br/>" + "<br/>".join(f"✓ {w}" for w in key_wins[:4]),
                    small,
                ))
            if risk_flags:
                story.append(Paragraph(
                    "<b>Risk Flags:</b><br/>" + "<br/>".join(f"⚠ {f}" for f in risk_flags[:3]),
                    ParagraphStyle("risk", parent=small, textColor=HexColor("#c0392b")),
                ))
            story.append(Spacer(1, 4*mm))

        # ── 9. Derived Insights (v2.0 NEW) ────────────────────────────────
        derived_insights = ie.get("derived_insights", []) if ie else []
        if derived_insights:
            story.append(Paragraph("9. Data-Backed Renovation Insights", h2))
            story.append(Paragraph(
                "The following insights are derived from ARKEN's analytics engine, "
                "backed by Indian real-estate research and market data.",
                body,
            ))
            story.append(Spacer(1, 2*mm))

            for idx, ins in enumerate(derived_insights[:6], 1):
                if not isinstance(ins, dict):
                    continue
                is_positive = ins.get("is_positive", True)
                icon = "✓" if is_positive else "⚠"
                colour = "#27ae60" if is_positive else "#e67e22"
                category_badge = ins.get("category", "").upper()

                story.append(Paragraph(
                    f"<font color='{colour}'><b>{icon} [{category_badge}]</b></font> "
                    f"{ins.get('insight', '')}",
                    body,
                ))
                roi_impact = ins.get("roi_impact", "")
                source = ins.get("source", "")
                confidence = ins.get("confidence", 0)
                if roi_impact or source:
                    story.append(Paragraph(
                        f"  <font color='#888888'>ROI Impact: {roi_impact} | "
                        f"Source: {source} | Confidence: {int(confidence * 100)}%</font>",
                        small,
                    ))
                story.append(Spacer(1, 1*mm))

            story.append(Spacer(1, 4*mm))

        # ── 10. Market Benchmark (v2.0 NEW) ──────────────────────────────
        benchmark = ie.get("market_benchmark", {}) if ie else {}
        if benchmark:
            story.append(Paragraph("10. Market Benchmark Analysis", h2))
            story.append(Paragraph(benchmark.get("benchmark_insight", ""), body))
            story.append(Spacer(1, 2*mm))

            bench_rows = []
            if benchmark.get("city_avg_roi_pct"):
                bench_rows.append([
                    "City Average ROI", f"{benchmark['city_avg_roi_pct']:.1f}%",
                    "Project ROI", f"{benchmark.get('project_roi_pct', '—'):.1f}%",
                ])
            if benchmark.get("roi_vs_city_avg") is not None:
                bench_rows.append([
                    "ROI vs Average", f"{benchmark['roi_vs_city_avg']:+.1f}%",
                    "Performance Tier", benchmark.get("performance_label", "—"),
                ])
            if benchmark.get("city_avg_cost_per_sqft"):
                bench_rows.append([
                    "City Avg Cost/SqFt", f"₹{benchmark['city_avg_cost_per_sqft']:,}",
                    "Project Cost/SqFt", f"₹{int(benchmark.get('project_cost_per_sqft', 0)):,}",
                ])
            if benchmark.get("city_5yr_appreciation_pct"):
                bench_rows.append([
                    "5-Year City Appreciation", f"{benchmark['city_5yr_appreciation_pct']}%",
                    "Room Type Premium", f"+{benchmark.get('room_type_premium_pct', 0)}%",
                ])
            if bench_rows:
                story.append(self._two_col_table(bench_rows))
            story.append(Spacer(1, 4*mm))

        # ── 11. Optimal Budget Allocation (v2.0 NEW) ─────────────────────
        budget_alloc = ie.get("optimal_budget_allocation", []) if ie else []
        if budget_alloc:
            story.append(Paragraph("11. Optimal Budget Allocation (ROI-Weighted)", h2))
            story.append(Paragraph(
                "Budget allocation recommended by ARKEN's ROI-weighted optimiser "
                "for maximum value delivery in this renovation type.",
                body,
            ))
            story.append(Spacer(1, 2*mm))
            alloc_data = [["Category", "Recommended (INR)", "% of Total", "Rationale"]]
            for item in budget_alloc[:8]:
                if not isinstance(item, dict):
                    continue
                alloc_data.append([
                    item.get("category", ""),
                    f"₹{int(item.get('recommended_inr', 0)):,}",
                    f"{item.get('pct_of_total', 0):.1f}%",
                    str(item.get("rationale", ""))[:80],
                ])
            alloc_table = Table(alloc_data, colWidths=["22%", "22%", "12%", "44%"])
            alloc_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), blue),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("ALIGN", (1, 0), (2, -1), "RIGHT"),
            ]))
            story.append(alloc_table)
            story.append(Spacer(1, 4*mm))

        # ── 12. Material Price Signals (v2.0 NEW) ─────────────────────────
        price_signals = ie.get("material_price_signals", []) if ie else []
        if price_signals:
            story.append(Paragraph("12. Material Price Signals", h2))
            story.append(Paragraph(
                "Live procurement signals from ARKEN's price forecasting engine. "
                "Act on high-urgency signals before committing to contractor rates.",
                body,
            ))
            story.append(Spacer(1, 2*mm))
            signal_data = [["Material", "Trend", "90-Day Change", "Action", "Urgency"]]
            for sig in price_signals[:6]:
                if not isinstance(sig, dict):
                    continue
                trend = sig.get("trend", "stable")
                trend_icon = "↑" if trend == "up" else "↓" if trend == "down" else "→"
                urgency = sig.get("urgency", "low")
                urg_colours = {"high": HexColor("#e74c3c"), "medium": HexColor("#e67e22"), "low": HexColor("#27ae60")}
                signal_data.append([
                    sig.get("material", ""),
                    f"{trend_icon} {trend.title()}",
                    f"{sig.get('pct_change_90d', 0):+.1f}%",
                    str(sig.get("procurement_recommendation", ""))[:60],
                    urgency.upper(),
                ])
            sig_table = Table(signal_data, colWidths=["22%", "12%", "14%", "38%", "14%"])
            sig_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), blue),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, light_bg]),
                ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("ALIGN", (4, 0), (4, -1), "CENTER"),
            ]))
            story.append(sig_table)
            story.append(Spacer(1, 4*mm))

        # ── Footer ────────────────────────────────────────────────────────
        story.append(Spacer(1, 8*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#dddddd")))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            f"<font color='#999999' size='7'>"
            f"Generated by ARKEN AI Renovation Intelligence Platform v2.0 | "
            f"{datetime.now().strftime('%d %B %Y')} | "
            f"Insights derived from ARKEN analytics engine, NHB Housing Data, Anarock Research 2024. "
            f"All cost estimates are indicative. Actual costs may vary by contractor and material availability."
            f"</font>",
            ParagraphStyle("footer", fontSize=7, textColor=HexColor("#999999"), alignment=TA_CENTER),
        ))

        doc.build(story)
        return buf.getvalue()

    def _two_col_table(self, rows: List[List[str]]) -> Any:
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib.colors import HexColor, white
        light_bg = HexColor("#f5f7fa")

        table = Table(rows, colWidths=["22%", "28%", "22%", "28%"])
        table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, light_bg]),
            ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
            ("LINEAFTER", (1, 0), (1, -1), 0.5, HexColor("#dddddd")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return table

    def _generate_html_fallback(self, report: Dict[str, Any], project_id: str) -> str:
        roi = report.get("roi_forecast", {})
        cost = report.get("cost_estimate", {})
        tl = report.get("renovation_timeline", {})
        room = report.get("room_analysis", {})
        ie = report.get("insight_engine", {})

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ARKEN Renovation Report — {project_id[:8].upper()}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 24px; color: #252525; }}
  h1 {{ color: #2e6bc5; font-size: 24px; }}
  h2 {{ color: #2e6bc5; font-size: 16px; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 28px; }}
  h3 {{ color: #444; font-size: 13px; margin-top: 14px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin: 16px 0; }}
  .kpi {{ background: #f5f7fa; border-radius: 6px; padding: 12px; text-align: center; border: 1px solid #e0e6ef; }}
  .kpi-val {{ font-size: 20px; font-weight: bold; color: #2e6bc5; }}
  .kpi-lbl {{ font-size: 11px; color: #888; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
  th {{ background: #2e6bc5; color: white; padding: 6px 10px; text-align: left; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .rec {{ background: #fff8f0; border-left: 4px solid #f28c1a; padding: 10px; margin: 8px 0; border-radius: 4px; }}
  .insight-positive {{ background: #f0fff4; border-left: 4px solid #27ae60; padding: 8px 12px; margin: 6px 0; border-radius: 4px; }}
  .insight-caution {{ background: #fff8f0; border-left: 4px solid #e67e22; padding: 8px 12px; margin: 6px 0; border-radius: 4px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; margin-right: 6px; }}
  .badge-market {{ background: #d6eaff; color: #1a5fa3; }}
  .badge-material {{ background: #e6f7ee; color: #1a7a43; }}
  .badge-cost {{ background: #fff0e0; color: #a35a00; }}
  .badge-risk {{ background: #ffe5e5; color: #a30000; }}
  .badge-design {{ background: #f0e6ff; color: #5a00a3; }}
  .score-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 12px 0; }}
  .footer {{ color: #999; font-size: 11px; margin-top: 32px; border-top: 1px solid #ddd; padding-top: 8px; }}
  .signal-high {{ color: #e74c3c; font-weight: bold; }}
  .signal-medium {{ color: #e67e22; font-weight: bold; }}
  .signal-low {{ color: #27ae60; }}
</style>
</head>
<body>
<h1>ARKEN Renovation Report v2.0</h1>
<p style="color:#888;font-size:12px">Project: {project_id[:12].upper()} | Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}</p>
<p><strong>{report.get('summary_headline', '')}</strong></p>

<div class="kpi-grid">
  <div class="kpi"><div class="kpi-val">{roi.get('roi_percentage','—')}</div><div class="kpi-lbl">Projected ROI</div></div>
  <div class="kpi"><div class="kpi-val">{cost.get('total','—')}</div><div class="kpi-lbl">Total Cost</div></div>
  <div class="kpi"><div class="kpi-val">{roi.get('equity_gain','—')}</div><div class="kpi-lbl">Equity Gain</div></div>
  <div class="kpi"><div class="kpi-val">{tl.get('total_days','—')} days</div><div class="kpi-lbl">Timeline</div></div>
  <div class="kpi"><div class="kpi-val">{report.get('layout_analysis',{}).get('layout_score','—')}</div><div class="kpi-lbl">Layout Score</div></div>
</div>

{self._html_insight_engine(report)}
{self._html_derived_insights(report)}
{self._html_market_benchmark(report)}
{self._html_budget_allocation(report)}
{self._html_price_signals(report)}

<div class="footer">Generated by ARKEN AI Renovation Intelligence Platform v2.0 |
Insights derived from ARKEN analytics engine, NHB Housing Data, Anarock Research 2024. |
All cost estimates are indicative.</div>
</body>
</html>"""
        return html

    def _html_insight_engine(self, report: Dict) -> str:
        ie = report.get("insight_engine", {})
        if not ie:
            return "<p><em>Insight Engine data not available.</em></p>"

        parts = [
            "<h2>8. AI Insight Engine</h2>",
            f'<div class="score-grid">'
            f'<div class="kpi"><div class="kpi-val">{ie.get("renovation_priority_score","—")}/100</div><div class="kpi-lbl">Priority Score</div></div>'
            f'<div class="kpi"><div class="kpi-val">{ie.get("cost_effectiveness_score","—")}/100</div><div class="kpi-lbl">Cost Effectiveness</div></div>'
            f'<div class="kpi"><div class="kpi-val">{ie.get("roi_score","—")}</div><div class="kpi-lbl">ROI Score</div></div>'
            f'<div class="kpi"><div class="kpi-val">{ie.get("overall_insight_score","—")}/100</div><div class="kpi-lbl">Overall Score</div></div>'
            f'</div>',
        ]

        if ie.get("summary"):
            parts.append(f'<p>{ie["summary"]}</p>')

        wins = ie.get("key_wins", [])
        if wins:
            parts.append("<h3>Key Wins</h3><ul>" + "".join(f"<li>✓ {w}</li>" for w in wins) + "</ul>")

        flags = ie.get("risk_flags", [])
        if flags:
            parts.append('<h3 style="color:#c0392b">Risk Flags</h3><ul style="color:#c0392b">'
                         + "".join(f"<li>⚠ {f}</li>" for f in flags) + "</ul>")

        return "\n".join(parts)

    def _html_derived_insights(self, report: Dict) -> str:
        ie = report.get("insight_engine", {})
        insights = ie.get("derived_insights", []) if ie else []
        if not insights:
            return ""

        parts = ["<h2>9. Data-Backed Renovation Insights</h2>"]
        for ins in insights[:6]:
            if not isinstance(ins, dict):
                continue
            is_positive = ins.get("is_positive", True)
            cat = ins.get("category", "design")
            css = "insight-positive" if is_positive else "insight-caution"
            badge_css = f"badge-{cat}" if cat in ("market", "material", "cost", "risk", "design") else "badge-design"
            icon = "✓" if is_positive else "⚠"
            parts.append(
                f'<div class="{css}">'
                f'<span class="badge {badge_css}">{cat.upper()}</span>'
                f'<strong>{icon} {ins.get("insight", "")}</strong>'
                f'<br/><small style="color:#888">ROI Impact: {ins.get("roi_impact","")} | '
                f'Source: {ins.get("source","")} | Confidence: {int(ins.get("confidence",0)*100)}%</small>'
                f'</div>'
            )
        return "\n".join(parts)

    def _html_market_benchmark(self, report: Dict) -> str:
        ie = report.get("insight_engine", {})
        benchmark = ie.get("market_benchmark", {}) if ie else {}
        if not benchmark:
            return ""

        parts = [
            "<h2>10. Market Benchmark Analysis</h2>",
            f'<p>{benchmark.get("benchmark_insight","")}</p>',
            "<table><tr><th>Metric</th><th>City Average</th><th>This Project</th></tr>",
            f"<tr><td>ROI</td><td>{benchmark.get('city_avg_roi_pct','—'):.1f}%</td><td>{benchmark.get('project_roi_pct','—'):.1f}%</td></tr>",
            f"<tr><td>Cost/SqFt</td><td>₹{int(benchmark.get('city_avg_cost_per_sqft',0)):,}</td><td>₹{int(benchmark.get('project_cost_per_sqft',0)):,}</td></tr>",
            f"<tr><td>Performance</td><td colspan='2'><strong>{benchmark.get('performance_label','—')}</strong></td></tr>",
            "</table>",
        ]
        return "\n".join(parts)

    def _html_budget_allocation(self, report: Dict) -> str:
        ie = report.get("insight_engine", {})
        allocs = ie.get("optimal_budget_allocation", []) if ie else []
        if not allocs:
            return ""

        parts = [
            "<h2>11. Optimal Budget Allocation</h2>",
            "<table><tr><th>Category</th><th>Recommended</th><th>% of Total</th><th>Rationale</th></tr>",
        ]
        for item in allocs[:8]:
            if not isinstance(item, dict):
                continue
            parts.append(
                f"<tr><td>{item.get('category','')}</td>"
                f"<td>₹{int(item.get('recommended_inr',0)):,}</td>"
                f"<td>{item.get('pct_of_total',0):.1f}%</td>"
                f"<td>{item.get('rationale','')[:100]}</td></tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    def _html_price_signals(self, report: Dict) -> str:
        ie = report.get("insight_engine", {})
        signals = ie.get("material_price_signals", []) if ie else []
        if not signals:
            return ""

        parts = [
            "<h2>12. Material Price Signals</h2>",
            "<table><tr><th>Material</th><th>Trend</th><th>90-Day Change</th><th>Recommendation</th><th>Urgency</th></tr>",
        ]
        for sig in signals[:6]:
            if not isinstance(sig, dict):
                continue
            trend = sig.get("trend", "stable")
            icon = "↑" if trend == "up" else "↓" if trend == "down" else "→"
            urgency = sig.get("urgency", "low")
            urg_class = f"signal-{urgency}"
            parts.append(
                f"<tr><td>{sig.get('material','')}</td>"
                f"<td>{icon} {trend.title()}</td>"
                f"<td class='{urg_class}'>{sig.get('pct_change_90d',0):+.1f}%</td>"
                f"<td>{sig.get('procurement_recommendation','')[:80]}</td>"
                f"<td class='{urg_class}'>{urgency.upper()}</td></tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)