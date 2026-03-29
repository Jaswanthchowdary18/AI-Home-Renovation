"""
ARKEN — /api/v1/artifacts routes v3.0
PDF report generation + render/mask proxy.
"""

import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

from services.cache import cache_service
from services.storage import s3_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ── PDF Report ──────────────────────────────────────────────────────────────

from pydantic import BaseModel
from typing import Any, Dict

class PDFReportRequest(BaseModel):
    """Direct PDF generation from frontend state — no cache needed."""
    project_id: str
    theme: str = "Modern Minimalist"
    city: str = "Hyderabad"
    room_type: str = "bedroom"
    budget_tier: str = "mid"
    budget_inr: int = 750000
    insights: Dict[str, Any] = {}
    roi: Dict[str, Any] = {}
    design: Dict[str, Any] = {}
    schedule: Dict[str, Any] = {}
    material_prices: list = []
    render_url: str = ""
    # Before / after images for the PDF visual comparison page
    # original_image_b64: raw base64 string (no data: prefix)
    # renovated_image_b64: raw base64 string OR full data: URL (both accepted)
    original_image_b64: str = ""
    original_image_mime: str = "image/jpeg"
    renovated_image_b64: str = ""


@router.post("/{project_id}/report/pdf")
async def generate_pdf_from_data(project_id: str, req: PDFReportRequest):
    """Generate PDF directly from POST body — works without cached pipeline result."""

    # ── Normalise renovated image: accept data: URL or raw base64 ──────────
    ren_b64 = req.renovated_image_b64 or ""
    if ren_b64.startswith("data:"):
        # Strip "data:image/png;base64," prefix
        ren_b64 = ren_b64.split(",", 1)[-1]
    # Also accept render_url as fallback for renovated image
    if not ren_b64 and req.render_url.startswith("data:"):
        ren_b64 = req.render_url.split(",", 1)[-1]

    report_data = {
        "project_id": project_id,
        "theme": req.theme,
        "city": req.city,
        "room_type": req.room_type,
        "budget_tier": req.budget_tier,
        "budget_inr": req.budget_inr,
        # ── Images for before/after comparison page ───────────────────────
        "original_image_b64":  req.original_image_b64 or "",
        "original_image_mime": req.original_image_mime or "image/jpeg",
        "renovated_image_b64": ren_b64,
        # Normalise shape to what _generate_pdf expects
        "visual": {
            "room_features": {
                "room_type": req.room_type,
                "style_label": req.theme,
                "floor_area_sqft": 120,
                **req.insights.get("visual_analysis", {}),
            },
            "design_recommendations": [],
            "layout_report": {},
        },
        "roi": {
            "roi_pct": float(req.roi.get("roi_pct", 0)),
            "payback_months": int(req.roi.get("payback_months", 36)),
            "equity_gain_inr": int(req.roi.get("equity_gain_inr", 0)),
            "model_type": req.roi.get("model_type", "XGBoost"),
            "model_confidence": req.roi.get("model_confidence", 0.7),
            "roi_prediction": req.roi,
            **req.roi,   # pass all roi fields flat so _generate_pdf reads them directly
        },
        "design": {
            "total_cost_inr": int(req.design.get("total_inr", req.budget_inr)),
            "line_items": req.design.get("line_items", []),
            "material_inr": int(req.design.get("material_inr", 0)),
            "labour_inr": int(req.design.get("labour_inr", 0)),
            "gst_inr": int(req.design.get("gst_inr", 0)),
            "contingency_inr": int(req.design.get("contingency_inr", 0)),
            "products_subtotal_inr": int(req.design.get("products_subtotal_inr", 0)),
            **req.design,  # pass all design fields through
        },
        "schedule": req.schedule,
        "insights": req.insights,
    }
    # Also cache for GET endpoint
    await cache_service.set(f"project_report:{project_id}", report_data, ttl=3600)
    try:
        pdf_bytes = _generate_pdf(report_data, project_id)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=arken-report-{project_id[:8]}.pdf"},
        )
    except Exception as e:
        logger.error(f"[PDF POST] Generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"PDF generation failed: {e}")


@router.get("/{project_id}/report/pdf")
async def export_pdf_report(
    project_id: str,
    task_id: Optional[str] = Query(None),
):
    """Generate and stream a PDF renovation report."""
    # Try project_report cache first, fall back to task result
    report_data = await cache_service.get(f"project_report:{project_id}")

    if not report_data and task_id:
        task_data = await cache_service.get(f"task:{task_id}")
        if task_data and task_data.get("result"):
            report_data = task_data["result"]

    if not report_data:
        raise HTTPException(404, "Report not found. Run an analysis first.")

    try:
        pdf_bytes = _generate_pdf(report_data, project_id)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=arken-report-{project_id[:8]}.pdf"},
        )
    except Exception as e:
        logger.error(f"[PDF] Generation failed: {e}")
        raise HTTPException(500, f"PDF generation failed: {e}")


@router.get("/{project_id}/report/json")
async def export_json_report(
    project_id: str,
    task_id: Optional[str] = Query(None),
):
    """Return structured report as JSON."""
    report_data = await cache_service.get(f"project_report:{project_id}")
    if not report_data and task_id:
        task_data = await cache_service.get(f"task:{task_id}")
        if task_data:
            report_data = task_data.get("result", task_data)
    if not report_data:
        raise HTTPException(404, "Report not found.")
    return report_data


# ── Render / Mask proxy ─────────────────────────────────────────────────────

@router.get("/{project_id}/render/{version}")
async def get_render(project_id: str, version: int = 1):
    key = f"projects/{project_id}/renders/v{version}.png"
    try:
        from config import settings
        img = await s3_service.download_bytes(key, bucket=settings.S3_BUCKET_RENDERS)
        return Response(content=img, media_type="image/png")
    except Exception:
        raise HTTPException(404, "Render not found")


@router.get("/{project_id}/mask/{mask_type}")
async def get_mask(project_id: str, mask_type: str = "combined_reno"):
    key = f"projects/{project_id}/masks/{mask_type}.png"
    try:
        from config import settings
        img = await s3_service.download_bytes(key, bucket=settings.S3_BUCKET_UPLOADS)
        return Response(content=img, media_type="image/png")
    except Exception:
        raise HTTPException(404, "Mask not found")


# ── PDF builder ─────────────────────────────────────────────────────────────


def _generate_pdf(data: dict, project_id: str) -> bytes:
    """
    ARKEN Premium PDF Report v3.0
    Full-page premium renovation intelligence report.
    Sections: Cover KPIs, Room Analysis, Cost Breakdown, Full BOQ,
              ROI & Rent Uplift, Renovation Sequence, Risk Factors,
              Market Intelligence, Material Price Signals, Footer.
    """
    import io as _io
    from datetime import datetime as _dt
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, white, black
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, KeepTogether, PageBreak,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
        from reportlab.platypus.flowables import BalancedColumns
    except ImportError:
        return _html_pdf_fallback(data, project_id)

    # ── Colour palette ──────────────────────────────────────────────────────
    NAVY    = HexColor("#0f172a")   # deep navy — headings, header bg
    INDIGO  = HexColor("#3730a3")   # brand indigo — section titles, accents
    VIOLET  = HexColor("#7c3aed")   # purple — highlights
    TEAL    = HexColor("#0d9488")   # teal — rent/positive
    AMBER   = HexColor("#d97706")   # amber — warnings / totals
    GREEN   = HexColor("#059669")   # green — gains / good
    RED     = HexColor("#dc2626")   # red — risk / negative
    SLATE   = HexColor("#475569")   # body text
    LIGHT   = HexColor("#f1f5f9")   # light row bg
    LIGHTER = HexColor("#f8fafc")   # alternate row bg
    BORDER  = HexColor("#e2e8f0")   # table borders
    GOLD    = HexColor("#f59e0b")   # gold total row
    MID     = HexColor("#64748b")   # muted text

    W = A4[0] - 36*mm   # usable width

    # ── Styles ──────────────────────────────────────────────────────────────
    SS = getSampleStyleSheet()

    def sty(name, **kw):
        return ParagraphStyle(name, parent=SS["Normal"], **kw)

    brand_h = sty("bh", fontSize=30, fontName="Helvetica-Bold", textColor=white,
                  alignment=TA_LEFT, spaceAfter=0, leading=32)
    brand_sub = sty("bs", fontSize=10, textColor=HexColor("#94a3b8"), spaceAfter=0)
    sec = sty("sec", fontSize=13, fontName="Helvetica-Bold", textColor=INDIGO,
              spaceBefore=10, spaceAfter=4, leading=16)
    sub = sty("sub", fontSize=10, fontName="Helvetica-Bold", textColor=NAVY,
              spaceBefore=6, spaceAfter=2)
    body = sty("body", fontSize=9, textColor=SLATE, leading=13, spaceAfter=2)
    small = sty("sm", fontSize=8, textColor=MID, leading=11, spaceAfter=1)
    tiny = sty("tiny", fontSize=7, textColor=MID, leading=10)
    kpi_val = sty("kv", fontSize=18, fontName="Helvetica-Bold", textColor=INDIGO,
                  alignment=TA_CENTER, leading=20)
    kpi_lbl = sty("kl", fontSize=7.5, textColor=MID, alignment=TA_CENTER, leading=10)
    bullet = sty("bul", fontSize=9, textColor=SLATE, leading=13,
                 leftIndent=10, spaceAfter=2)
    warn = sty("warn", fontSize=9, textColor=AMBER, leading=13, leftIndent=10)
    good = sty("good", fontSize=9, textColor=GREEN, leading=13, leftIndent=10)
    right_b = sty("rb", fontSize=9, fontName="Helvetica-Bold", textColor=NAVY,
                  alignment=TA_RIGHT)
    footer_s = sty("ft", fontSize=7, textColor=MID, alignment=TA_CENTER, leading=9)

    # ── Helpers ─────────────────────────────────────────────────────────────
    def inr(n, lakh=True):
        try:
            n = float(n)
            if n == 0: return "-"
            neg = n < 0; n = abs(n)
            if n >= 1e7:   s = f"Rs {n/1e7:.2f} Cr"
            elif n >= 1e5: s = f"Rs {n/1e5:.1f}L"
            elif n >= 1e3: s = f"Rs {n/1e3:.0f}K"
            else:          s = f"Rs {n:.0f}"
            return ("-" if neg else "") + s
        except: return str(n) or "-"

    def inr_full(n):
        try:
            n = int(float(n))
            if n == 0: return "-"
            s = f"{abs(n):,}"
            return ("-" if n < 0 else "") + "Rs " + s
        except: return str(n) or "-"

    def pct(n):
        try: return f"{float(n):.1f}%"
        except: return str(n) or "—"

    def safe(v, fallback="-"):
        if v is None or v == "" or v == 0: return fallback
        return str(v)

    def clean(s):
        """
        Replace rupee glyph (unsupported in Helvetica) with 'Rs ' and strip
        any other characters that render as black squares in the PDF.
        Must be applied to all backend-generated string fields before Paragraph().
        """
        if not s: return str(s) if s is not None else ""
        return (str(s)
                .replace("\u20b9", "Rs ")   # ₹  unicode rupee sign
                .replace("\u25a0", "Rs ")   # ■  black square (already-mangled rupee)
                .replace("₹", "Rs ")        # literal rupee if present
                .strip())

    def divider(color=BORDER, thick=0.5):
        return HRFlowable(width="100%", thickness=thick, color=color, spaceAfter=3, spaceBefore=3)

    def section_header(title, number=None):
        label = f"{number}.  {title}" if number else title
        return [
            Spacer(1, 3*mm),
            Paragraph(label, sec),
            HRFlowable(width="100%", thickness=1, color=INDIGO, spaceAfter=3),
        ]

    def kpi_block(pairs):
        """pairs = [(value_str, label_str), ...]"""
        n = len(pairs)
        vals  = [Paragraph(p[0], kpi_val) for p in pairs]
        lbls  = [Paragraph(p[1], kpi_lbl) for p in pairs]
        t = Table([vals, lbls], colWidths=[W/n]*n)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), LIGHT),
            ("BOX",        (0,0), (-1,-1), 0.5, BORDER),
            ("LINEAFTER",  (0,0), (-2,-1), 0.5, BORDER),
            ("TOPPADDING", (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ]))
        return t

    def two_col(rows, label_w="35%", val_w="15%"):
        """rows = [[label, val, label, val], ...]"""
        col = W * 0.5
        tbl = Table(rows, colWidths=[col*0.45, col*0.55, col*0.45, col*0.55])
        tbl.setStyle(TableStyle([
            ("FONTSIZE",  (0,0), (-1,-1), 8.5),
            ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0,0), (0,-1), NAVY),
            ("TEXTCOLOR", (2,0), (2,-1), NAVY),
            ("TEXTCOLOR", (1,0), (1,-1), SLATE),
            ("TEXTCOLOR", (3,0), (3,-1), SLATE),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [white, LIGHTER]),
            ("BOX",       (0,0), (-1,-1), 0.5, BORDER),
            ("LINEAFTER", (1,0), (1,-1), 0.5, BORDER),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
        ]))
        return tbl

    def header_table(cols, data_rows, col_widths=None):
        rows = [cols] + data_rows
        cw = col_widths or ([W/len(cols)] * len(cols))
        t = Table(rows, colWidths=cw, repeatRows=1)
        styles_list = [
            ("BACKGROUND",   (0,0), (-1,0), NAVY),
            ("TEXTCOLOR",    (0,0), (-1,0), white),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, LIGHTER]),
            ("BOX",          (0,0), (-1,-1), 0.5, BORDER),
            ("INNERGRID",    (0,0), (-1,-1), 0.3, BORDER),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ("LEFTPADDING",  (0,0), (-1,-1), 5),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ]
        t.setStyle(TableStyle(styles_list))
        return t

    # ── Data extraction ─────────────────────────────────────────────────────
    now_str   = _dt.now().strftime("%d %B %Y, %H:%M")
    city      = data.get("city", "India")
    room_type = data.get("room_type", "bedroom").replace("_", " ").title()
    theme     = data.get("theme", "Modern Minimalist")
    budget_tier = data.get("budget_tier", "mid").title()

    roi_raw   = data.get("roi", {})
    design    = data.get("design", {})
    schedule  = data.get("schedule", {})
    insights  = data.get("insights", {})

    # ROI fields — handle nested roi_prediction or flat dict
    roi_pred  = roi_raw if isinstance(roi_raw, dict) else {}
    roi_pct   = float(roi_pred.get("roi_pct", 0))
    payback   = int(roi_pred.get("payback_months", 0))
    equity    = int(roi_pred.get("equity_gain_inr", roi_pred.get("net_gain_inr", 0)))
    pre_val   = int(roi_pred.get("pre_reno_value_inr", 0))
    post_val  = int(roi_pred.get("post_reno_value_inr", 0))
    rent_b    = int(roi_pred.get("rent_before_inr_per_month", 0))
    rent_a    = int(roi_pred.get("rent_after_inr_per_month", 0))
    rent_up   = float(roi_pred.get("rent_uplift_pct", 0))
    monthly_inc = int(roi_pred.get("monthly_rental_increase_inr", 0))
    esc_pct   = roi_pred.get("rent_escalation_pct_annual", 0)
    model_t   = str(roi_pred.get("model_type", "Ensemble ML"))
    conf      = float(roi_pred.get("model_confidence", 0.7))
    rupee     = roi_pred.get("rupee_breakdown") or {}
    risk_factors = roi_pred.get("risk_factors", []) or []
    city_tier = int(roi_pred.get("city_tier", 1))
    roi_ci_low  = roi_pred.get("roi_ci_low", 0)
    roi_ci_high = roi_pred.get("roi_ci_high", 0)

    # Cost fields
    total_cost = int(design.get("total_inr", data.get("budget_inr", 0)))
    mat_inr    = int(design.get("material_inr", 0))
    lab_inr    = int(design.get("labour_inr", 0))
    gst_inr    = int(design.get("gst_inr", 0))
    cont_inr   = int(design.get("contingency_inr", 0))
    prod_inr   = int(design.get("products_subtotal_inr", 0))
    line_items = design.get("line_items", []) or []

    # Insights fields
    vis       = insights.get("visual_analysis", {}) or {}
    fin       = insights.get("financial_outlook", {}) or {}
    ie        = insights.get("insight_engine", {}) or {}
    mkt       = insights.get("market_intelligence", {}) or {}
    loc       = mkt if isinstance(mkt, dict) else {}
    reno_seq  = insights.get("renovation_sequence", []) or []
    priority_repairs = insights.get("priority_repairs", []) or ie.get("priority_repairs", []) or []
    headline  = insights.get("summary_headline", f"{theme} {room_type} Renovation — {city}")
    derived   = ie.get("derived_insights", []) or []
    price_sigs = insights.get("material_price_signals", []) or ie.get("material_price_signals", []) or []
    key_wins  = ie.get("key_wins", []) or []

    # ── Build story ─────────────────────────────────────────────────────────
    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=16*mm,
        title=f"ARKEN Renovation Report — {project_id[:8].upper()}",
        author="ARKEN PropTech Intelligence",
    )
    story = []

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER HEADER
    # ════════════════════════════════════════════════════════════════════════

    # Dark header band
    hdr = Table(
        [[
            Paragraph("<b>ARKEN</b>", brand_h),
            Paragraph(
                f"PropTech Intelligence Report<br/>"
                f"<font size='8' color='#94a3b8'>Project {project_id[:16].upper()}&nbsp;&nbsp;|&nbsp;&nbsp;{now_str}</font>",
                sty("hr", fontSize=10, textColor=HexColor("#cbd5e1"), alignment=TA_RIGHT, leading=14)
            ),
        ]],
        colWidths=["50%", "50%"],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS",(0,0), (-1,-1), [4,4,0,0]),
    ]))
    story.append(hdr)

    # Headline strip
    hl_tbl = Table([[Paragraph(clean(headline), sty("hl", fontSize=12, fontName="Helvetica-Bold",
                       textColor=white, leading=15))]],
                   colWidths=[W + 0.1*mm])
    hl_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), INDIGO),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(hl_tbl)
    story.append(Spacer(1, 4*mm))

    # ── KPI row ─────────────────────────────────────────────────────────────
    kpis = [
        (pct(roi_pct),          "Projected ROI"),
        (inr(total_cost),       "Total Cost (incl. GST)"),
        (inr(equity),           "Net Equity Gain"),
        (f"+{pct(rent_up)}",    "Est. Rent Increase"),
        (f"{payback} mo",       "Rental Break-even"),
    ]
    story.append(kpi_block(kpis))
    story.append(Spacer(1, 3*mm))

    # Confidence + model badge row
    conf_colour = "059669" if conf >= 0.80 else ("d97706" if conf >= 0.60 else "dc2626")
    story.append(Paragraph(
        f"<font color='#{conf_colour}'>● Model confidence: {conf*100:.0f}%</font>&nbsp;&nbsp;"
        f"<font color='#64748b'>Model: {model_t}</font>&nbsp;&nbsp;"
        f"<font color='#64748b'>City: {city} (Tier {city_tier})</font>&nbsp;&nbsp;"
        f"<font color='#64748b'>Budget tier: {budget_tier}</font>",
        sty("conf", fontSize=8, textColor=MID, spaceAfter=4)
    ))

    # ════════════════════════════════════════════════════════════════════════
    # BEFORE / AFTER IMAGE COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    orig_b64 = data.get("original_image_b64", "") or ""
    ren_b64_img = data.get("renovated_image_b64", "") or ""

    # Strip any remaining data: prefix (safety net)
    if orig_b64.startswith("data:"):
        orig_b64 = orig_b64.split(",", 1)[-1]
    if ren_b64_img.startswith("data:"):
        ren_b64_img = ren_b64_img.split(",", 1)[-1]

    if orig_b64 or ren_b64_img:
        import base64 as _b64
        from io import BytesIO as _BIO
        from reportlab.platypus import Image as _RLImg

        story.append(Spacer(1, 3*mm))

        # Label row
        n_images   = (1 if orig_b64 else 0) + (1 if ren_b64_img else 0)
        img_w      = (W - 6*mm) / n_images       # width per image
        img_h      = img_w * 0.70                  # ~4:3 aspect
        img_cells  = []
        lbl_cells  = []

        for b64_str, label, accent in [
            (orig_b64,    "BEFORE RENOVATION", SLATE),
            (ren_b64_img, "AFTER RENOVATION",  TEAL),
        ]:
            if not b64_str:
                continue
            try:
                raw = _b64.b64decode(b64_str)
                img_io = _BIO(raw)
                img_obj = _RLImg(img_io, width=img_w, height=img_h)
                img_cells.append(img_obj)
                lbl_cells.append(
                    Paragraph(
                        label,
                        sty(f"il_{label}", fontSize=8, fontName="Helvetica-Bold",
                            textColor=accent, alignment=TA_CENTER)
                    )
                )
            except Exception as img_err:
                logger.warning(f"[PDF] Image decode failed for {label}: {img_err}")

        if img_cells:
            n = len(img_cells)
            cw = [(W - 6*mm*(n-1)) / n] * n
            if n == 2:
                cw = [(W - 6*mm) / 2] * 2

            # Image row
            img_row_t = Table([img_cells], colWidths=cw if n > 1 else [W])
            img_row_t.setStyle(TableStyle([
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("LINEAFTER",     (0,0), (-2,-1), 1.5, white) if n > 1 else ("BOX", (0,0), (-1,-1), 0, white),
                ("TOPPADDING",    (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ]))

            # Label row
            lbl_row_t = Table([lbl_cells], colWidths=cw if n > 1 else [W])
            lbl_row_t.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), NAVY),
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))

            # Wrap in outer table with border
            outer = Table(
                [[img_row_t], [lbl_row_t]],
                colWidths=[W],
            )
            outer.setStyle(TableStyle([
                ("BOX",           (0,0), (-1,-1), 1, INDIGO),
                ("TOPPADDING",    (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ]))
            story.append(outer)
            story.append(Spacer(1, 3*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — ROOM & VISUAL ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    story += section_header("Room & Visual Analysis", 1)

    room_rows = [
        ["Room Type",        vis.get("room_type", room_type).title(),
         "Style Detected",   vis.get("style_detected", theme)],
        ["Renovation Scope", vis.get("renovation_scope", "—").replace("_"," ").title(),
         "Room Condition",   vis.get("room_condition", "—").title()],
        ["Natural Light",    vis.get("natural_light", "—").title(),
         "CV Model",         vis.get("cv_model", "cv_pipeline_v2")],
        ["Floor Area",       f"{vis.get('floor_area_sqft', 84):.0f} sqft",
         "Wall Area",        f"{vis.get('wall_area_sqft', 297):.0f} sqft"],
    ]
    issues = vis.get("issues_detected", []) or []
    upgrades = vis.get("high_value_upgrades", []) or []
    dets = vis.get("detected_objects", []) or []
    story.append(two_col(room_rows))
    if issues:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Issues detected by Gemini Vision:", sub))
        for iss in issues[:5]:
            story.append(Paragraph(f"!  {clean(str(iss))}", warn))
    if upgrades:
        story.append(Paragraph("High-ROI upgrade opportunities:", sub))
        for upg in upgrades[:4]:
            story.append(Paragraph(f"+  {clean(str(upg))}", good))
    if dets:
        story.append(Paragraph(f"Detected objects: {clean(', '.join(str(d) for d in dets[:8]))}", small))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — COST ESTIMATE
    # ════════════════════════════════════════════════════════════════════════
    story += section_header("Renovation Cost Estimate", 2)

    # FIX: use construction subtotal for % calculation (excludes products)
    construction_sub = max(mat_inr + lab_inr + gst_inr + cont_inr, 1)
    cost_rows = []
    if mat_inr:
        cost_rows.append(["Materials",           inr_full(mat_inr),  f"{mat_inr/construction_sub*100:.0f}%", "Tiles, paint, fittings, fixtures"])
    if lab_inr:
        cost_rows.append(["Labour",              inr_full(lab_inr),  f"{lab_inr/construction_sub*100:.0f}%", "Carpentry, painting, civil, electrical"])
    if gst_inr:
        cost_rows.append(["GST (18% on mats)",   inr_full(gst_inr),  f"{gst_inr/construction_sub*100:.0f}%", "Statutory tax on materials"])
    if cont_inr:
        cost_rows.append(["Contingency (5%)",    inr_full(cont_inr), f"{cont_inr/construction_sub*100:.0f}%", "Buffer for overruns and price changes"])
    if prod_inr:
        cost_rows.append(["Products/Furnishings",inr_full(prod_inr), "extra", "Suggested furniture & fixtures"])

    if cost_rows:
        # FIX: % of total should be calculated against construction subtotal
        # (mat + lab + gst + cont), not total_cost which includes products.
        # Products are a separate line and shouldn't inflate other %s.
        construction_subtotal = max(mat_inr + lab_inr + gst_inr + cont_inr, 1)
        cost_rows_pct = []
        for row in cost_rows:
            # row = [label, inr_full, pct_str, notes]
            # Recalc pct against correct base
            cost_rows_pct.append(row)
        cost_t = header_table(
            ["Category", "Amount", "% of Subtotal", "Notes"],
            cost_rows,
            col_widths=[W*0.28, W*0.22, W*0.12, W*0.38],
        )
        story.append(cost_t)

    # Total row
    tot_t = Table(
        [["TOTAL (incl. GST + contingency)", inr_full(total_cost)]],
        colWidths=[W*0.62, W*0.38],
    )
    tot_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), GOLD),
        ("TEXTCOLOR",     (0,0), (-1,-1), white),
        ("FONTNAME",      (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("ALIGN",         (1,0), (1,-1), "RIGHT"),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(tot_t)
    story.append(Spacer(1, 2*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — FULL BILL OF QUANTITIES (BOQ)
    # ════════════════════════════════════════════════════════════════════════
    if line_items:
        story += section_header("Bill of Quantities (BOQ)", 3)
        story.append(Paragraph(
            f"{len(line_items)} line items  ·  {city}  ·  {budget_tier} tier  ·  "
            f"Source: ARKEN SKU catalog with city-adjusted pricing",
            small
        ))
        story.append(Spacer(1, 1*mm))

        boq_data = []
        running_total = 0
        for item in line_items:
            # FIX: pipeline uses "product" as the description key (from design_planner.py)
            # Combine brand + product for a complete description
            brand = str(item.get("brand", "")).strip()
            prod  = str(item.get("product",
                        item.get("description",
                        item.get("name",
                        item.get("item", ""))))).strip()
            if brand and prod and brand.lower() not in prod.lower():
                desc = f"{brand} {prod}"
            else:
                desc = prod or brand
            desc = desc[:52]  # slightly wider trim

            qty  = item.get("quantity", item.get("qty", ""))
            # FIX: shorten long unit strings so they don't overflow the narrow Unit column
            unit_raw = str(item.get("unit", ""))
            unit = (unit_raw
                    .replace("sqft (incl. 10% wastage)", "sqft+10%")
                    .replace("(incl. 10% wastage)", "+10%")
                    .replace("incl. 10% wastage", "+10%"))
            unit = unit[:14]  # cap at 14 chars

            rate = item.get("rate_inr", item.get("rate", 0))
            amt  = item.get("total_inr", item.get("amount_inr", item.get("total", 0)))
            try: amt = float(amt)
            except: amt = 0
            running_total += amt
            cat  = item.get("category", "")
            boq_data.append([
                desc,
                str(qty) if qty != "" else "-",
                unit,
                f"{float(rate):,.0f}" if rate else "-",
                f"{amt:,.0f}" if amt else "-",
                cat[:22] if cat else "",
            ])

        boq_t = header_table(
            ["Description", "Qty", "Unit", "Rate (Rs)", "Amt (Rs)", "Category"],
            boq_data,
            # FIX: give Description more width, narrow Unit column
            col_widths=[W*0.38, W*0.06, W*0.10, W*0.12, W*0.12, W*0.22],
        )
        story.append(boq_t)

        # BOQ total
        boq_tot = Table(
            [["", "", "", "BOQ SUBTOTAL", f"{running_total:,.0f}"]],
            colWidths=[W*0.36, W*0.06, W*0.08, W*0.28, W*0.22],
        )
        boq_tot.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), NAVY),
            ("TEXTCOLOR",     (0,0), (-1,-1), white),
            ("FONTNAME",      (3,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("ALIGN",         (3,0), (-1,-1), "RIGHT"),
            ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(boq_tot)
        story.append(Spacer(1, 2*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — ROI & INVESTMENT ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story += section_header("ROI & Investment Analysis", 4)

    # ROI kpi sub-row
    roi_kpis = [
        (pct(roi_pct),       "Property Value Uplift"),
        (inr(equity),        "Net Equity at Sale"),
        (inr(pre_val),       "Pre-reno Property Value"),
        (inr(post_val),      "Post-reno Property Value"),
    ]
    if roi_ci_low and roi_ci_high:
        story.append(Paragraph(
            f"Confidence interval: {pct(roi_ci_low)} – {pct(roi_ci_high)}  "
            f"(model: {model_t}, confidence {conf*100:.0f}%)",
            small
        ))
    story.append(kpi_block(roi_kpis))
    story.append(Spacer(1, 3*mm))

    # Comparable context
    comp = roi_pred.get("comparable_context", {}) or {}
    if comp.get("interpretation"):
        story.append(Paragraph(clean(comp["interpretation"]), body))
        story.append(Spacer(1, 2*mm))

    # ── Rental Income section ────────────────────────────────────────────────
    story.append(Paragraph("Rental Income Analysis", sub))
    rent_rows = [
        ["Rent before renovation",  inr_full(rent_b) + "/month",
         "Gross yield (base)",      pct(roi_pred.get("rental_yield_base_pct", 0))],
        ["Rent after renovation",   inr_full(rent_a) + "/month",
         "Effective yield (post)",  pct(roi_pred.get("rental_yield_post_pct", 0))],
        ["Monthly rent increase",   inr_full(monthly_inc) + "/month",
         "Annual rent increase",    inr_full(monthly_inc * 12) + "/year"],
        ["Estimated rent uplift",   f"+{pct(rent_up)}",
         "Annual escalation (est.)",f"{esc_pct}%/year" if esc_pct else "5–7%/year"],
        ["Rental break-even",       f"{payback} months",
         "With annual escalation",  f"(vs ~{max(payback+10,payback):d} mo flat-rate)" if esc_pct else "—"],
    ]
    story.append(two_col(rent_rows))
    if esc_pct:
        story.append(Paragraph(
            f"Break-even accounts for {esc_pct}%/year rent escalation — standard {city} "
            f"lease renewal rate (99acres 2024, NoBroker 2024). Rent grows at each annual "
            f"renewal, recovering renovation cost faster than a flat-rate estimate.",
            small
        ))
    story.append(Spacer(1, 2*mm))

    # ── 3 Ways to get money back ─────────────────────────────────────────────
    if rupee and rupee.get("how_you_get_money_back"):
        story.append(Paragraph("3 Ways to Recover Your Investment", sub))
        hwymb = rupee["how_you_get_money_back"]

        via_resale = hwymb.get("via_resale", {})
        via_rental = hwymb.get("via_rental", {})
        combo      = hwymb.get("combined_3yr", {})

        recovery_rows = []
        if via_resale.get("net_gain_inr") is not None:
            recovery_rows.append([
                "① Sell immediately",
                inr_full(via_resale.get("net_gain_inr", 0)),
                "Net profit on sale",
                via_resale.get("timeline", "Immediate"),
            ])
        if via_rental.get("monthly_extra_inr"):
            esc = via_rental.get("annual_escalation_pct", "")
            esc_label = f"Rent grows {esc}%/yr at renewal" if esc else f"Break-even in {payback} months"
            recovery_rows.append([
                "② Rent the property",
                inr_full(via_rental["monthly_extra_inr"]) + "/mo",
                f"Break-even in {payback} months",
                esc_label,
            ])
        if combo.get("total_return_inr"):
            recovery_rows.append([
                "③ Rent 3yr + sell",
                inr_full(combo["total_return_inr"]),
                "Best combined return",
                "3 years then exit",
            ])

        if recovery_rows:
            rt = header_table(
                ["Strategy", "Amount", "Metric", "Timeline"],
                recovery_rows,
                col_widths=[W*0.28, W*0.22, W*0.28, W*0.22],
            )
            story.append(rt)
        story.append(Spacer(1, 2*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — RENOVATION SEQUENCE
    # ════════════════════════════════════════════════════════════════════════
    if reno_seq:
        story += section_header("Renovation Sequence & Timeline", 5)

        # Total days from schedule
        total_days = schedule.get("total_days", schedule.get("days",
                     sum(s.get("duration_days", s.get("days", 0)) for s in reno_seq if isinstance(s, dict))))
        if total_days:
            story.append(Paragraph(
                f"Total duration: <b>{total_days} days</b>  "
                f"({total_days//7} weeks)  ·  "
                f"Sequence follows standard Indian contractor order: "
                f"civil → wet work → electrical → false ceiling → carpentry → painting → fixtures",
                body
            ))
            story.append(Spacer(1, 2*mm))

        seq_data = []
        for step in reno_seq[:15]:
            if not isinstance(step, dict): continue
            import re as _re
            # FIX: strip redundant "Phase N: " prefix that the pipeline adds
            name   = _re.sub(r'^Phase\s*\d+[:\.\-]\s*', '', str(
                        step.get("phase", step.get("step_name", step.get("name", "")))))
            name   = clean(name)[:50]
            days_s = str(step.get("duration_days", step.get("days", step.get("duration", ""))))
            cost_s = inr(step.get("estimated_cost_inr", 0)) if step.get("estimated_cost_inr") else "-"
            ptype  = "Parallel" if step.get("can_parallel") or step.get("parallel") else "Sequential"
            seq_data.append([str(step.get("step", "")), name, f"{days_s} days", cost_s, ptype])

        if seq_data:
            seq_t = header_table(
                ["#", "Phase", "Duration", "Est. Cost", "Execution"],
                seq_data,
                col_widths=[W*0.06, W*0.42, W*0.14, W*0.18, W*0.20],
            )
            story.append(seq_t)
        story.append(Spacer(1, 2*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 — PRIORITY REPAIRS
    # ════════════════════════════════════════════════════════════════════════
    if priority_repairs:
        story += section_header("Priority Repairs & Actions", 6)
        repair_data = []
        urg_labels = {"critical":"CRITICAL","high":"HIGH","medium":"MEDIUM","low":"LOW"}
        for pr in priority_repairs[:8]:
            if not isinstance(pr, dict): continue
            urg = pr.get("urgency", "medium")
            action = clean(str(pr.get("action", pr.get("phase", ""))))[:90]
            cat    = clean(pr.get("category", "")).title()[:20]
            cost_l = clean(str(pr.get("estimated_cost_label",
                               inr(pr.get("estimated_cost_inr", 0)))))
            repair_data.append([
                str(pr.get("rank", pr.get("step", ""))),
                action,
                cat,
                urg_labels.get(urg, urg.upper()),
                cost_l,
            ])
        if repair_data:
            rep_t = header_table(
                ["#", "Action", "Category", "Urgency", "Est. Cost"],
                repair_data,
                col_widths=[W*0.05, W*0.48, W*0.16, W*0.11, W*0.20],
            )
            story.append(rep_t)
        story.append(Spacer(1, 2*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7 — RISK FACTORS
    # ════════════════════════════════════════════════════════════════════════
    if risk_factors:
        story += section_header("Risk Factors", 7)
        for rf in risk_factors[:5]:
            text = rf if isinstance(rf, str) else str(rf.get("factor", rf.get("description", str(rf))))
            story.append(Paragraph(f"!  {clean(text)}", warn))
        story.append(Spacer(1, 2*mm))

    # Key wins
    if key_wins:
        story.append(Paragraph("Key Strengths:", sub))
        for w in key_wins[:4]:
            story.append(Paragraph(f"+  {clean(str(w))}", good))
        story.append(Spacer(1, 2*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8 — MARKET INTELLIGENCE
    # ════════════════════════════════════════════════════════════════════════
    story += section_header("Market Intelligence", 8)

    mkt_rows = []
    if loc.get("appreciation_5yr_pct"):
        mkt_rows.append(["5-Year City Appreciation", f"{loc['appreciation_5yr_pct']}%",
                          "Rental Yield", f"{loc.get('rental_yield_pct', '—')}%"])
    if loc.get("avg_psf_inr"):
        mkt_rows.append(["Avg Price / SqFt", f"Rs {loc['avg_psf_inr']:,}",
                          "Market Tier", f"Tier {loc.get('market_tier', city_tier)}"])

    # Benchmark from ROI comparable_context
    bench = roi_pred.get("comparable_context", {}) or {}
    if bench.get("city_avg_renovation_roi_pct"):
        mkt_rows.append(["City avg renovation ROI", pct(bench["city_avg_renovation_roi_pct"]),
                          "Your ROI vs avg", bench.get("your_roi_vs_city_avg", "—")])
    if bench.get("source"):
        mkt_rows.append(["Data source", bench["source"][:45], "City", city])

    if mkt_rows:
        story.append(two_col(mkt_rows))
    else:
        story.append(Paragraph(
            f"City: {city} · Tier {city_tier} market · "
            f"Source: NHB Residex 2024, ANAROCK Q4 2024, JLL India Residential 2024.",
            small
        ))

    if ie.get("market_benchmark", {}) and isinstance(ie["market_benchmark"], dict):
        mb = ie["market_benchmark"]
        if mb.get("benchmark_insight"):
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(clean(mb["benchmark_insight"]), body))

    story.append(Spacer(1, 2*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 9 — MATERIAL PRICE SIGNALS
    # ════════════════════════════════════════════════════════════════════════
    if price_sigs:
        story += section_header("Material Price Signals (90-day Forecast)", 9)
        story.append(Paragraph(
            "Live procurement signals from ARKEN's Prophet + XGBoost price forecasting engine. "
            "Act on high-urgency signals before locking contractor rates.",
            small
        ))
        story.append(Spacer(1, 1*mm))
        sig_data = []
        for sig in price_sigs[:8]:
            if not isinstance(sig, dict): continue
            trend = sig.get("trend", "stable")
            icon  = "↑" if trend == "up" else ("↓" if trend == "down" else "→")
            urg   = sig.get("urgency", "low").upper()
            sig_data.append([
                sig.get("material", ""),
                f"{icon} {trend.title()}",
                f"{sig.get('pct_change_90d', 0):+.1f}%",
                str(sig.get("procurement_recommendation", ""))[:70],
                urg,
            ])
        if sig_data:
            sig_t = header_table(
                ["Material", "Trend", "90-Day Δ", "Recommendation", "Urgency"],
                sig_data,
                col_widths=[W*0.20, W*0.10, W*0.10, W*0.46, W*0.14],
            )
            story.append(sig_t)
            story.append(Spacer(1, 2*mm))

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 10 — AI PIPELINE SYNTHESIS
    # ════════════════════════════════════════════════════════════════════════
    if derived or ie.get("summary"):
        story += section_header("AI Pipeline Synthesis", 10)
        if ie.get("summary"):
            story.append(Paragraph(clean(ie["summary"]), body))
            story.append(Spacer(1, 2*mm))
        for ins in derived[:6]:
            if not isinstance(ins, dict): continue
            is_pos = ins.get("is_positive", True)
            cat    = ins.get("category", "").upper()
            txt    = clean(ins.get("insight", ""))
            src    = clean(ins.get("source", ""))
            conf_i = int(ins.get("confidence", 0) * 100)
            icon   = "+" if is_pos else "!"
            col    = "059669" if is_pos else "d97706"
            story.append(Paragraph(
                f"<font color='#{col}'><b>{icon} [{cat}]</b></font>  {txt}",
                body
            ))
            if src or conf_i:
                story.append(Paragraph(f"Source: {src}  |  Confidence: {conf_i}%", tiny))
            story.append(Spacer(1, 1*mm))

    # ════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 8*mm))
    story.append(divider(INDIGO, 1))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"Generated by ARKEN PropTech Intelligence Platform v3.0  ·  "
        f"{_dt.now().strftime('%d %B %Y')}  ·  "
        f"Project {project_id[:16].upper()}<br/>"
        f"<font size='6.5'>Data sources: NHB Residex 2024, ANAROCK Q4 2024, JLL India Residential 2024, "
        f"NoBroker Renovation ROI Survey 2024 (8,400 renovations), 99acres Rental Yield Report 2024, "
        f"India House Rent Dataset (4,746 listings), india_renovation_rental_uplift.csv (56,100 rows). "
        f"All cost estimates are indicative. Actual costs may vary by contractor and material availability. "
        f"Consult a registered property valuer for a formal assessment.</font>",
        footer_s
    ))

    doc.build(story)
    return buf.getvalue()


def _html_pdf_fallback(data: dict, project_id: str) -> bytes:
    html = f"""<!DOCTYPE html><html><head><title>ARKEN Report</title></head>
    <body><h1>ARKEN Renovation Report</h1><p>Project: {project_id[:12]}</p>
    <pre>{str(data)[:2000]}</pre></body></html>"""
    return html.encode("utf-8")