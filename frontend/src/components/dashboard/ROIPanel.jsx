"use client";
import { useState, memo } from "react";
import PropTypes from "prop-types";

// LOCATION: frontend/src/components/dashboard/ROIPanel.jsx
//
// v7.0 — Real-world ROI panel redesign
// Changes from v6:
//   • HEADLINE metric is now rent_uplift_pct (e.g. "+38% rent increase")
//     instead of roi_pct (property value uplift) — this is what users
//     actually mean when they ask "what's my ROI after renovation?"
//   • roi_pct (property value uplift) moved to a secondary "Resale" section
//     with a clear label distinguishing it from rent increase.
//   • RentUpliftHero: new top card showing before/after rent, % increase,
//     and a colour-coded badge vs market expectation (30%+ = excellent).
//   • DataSourceBadge: shows exactly which dataset powered the numbers —
//     india_renovation_rental_uplift.csv (56K rows) + House_Rent_Dataset.csv
//   • All monetary numbers use real city data via backend yield premium.
//   • Payback period clamp changed from max-18mo to max-12mo (realistic).

// ── Design tokens ─────────────────────────────────────────────────────────────
const F = {
  mono: { fontFamily: "'JetBrains Mono','Fira Code',monospace" },
  sans: { fontFamily: "'Plus Jakarta Sans','DM Sans',system-ui,sans-serif" },
};
const C = {
  bg: "#05080f", surface: "#0a1120", card: "#0d1525",
  border: "#1a2540", borderHi: "#2a3f6b",
  text: "#e8edf5", muted: "#5a7090", dim: "#1e2e48",
  indigo: "#6470f3", indigoLo: "rgba(100,112,243,0.10)",
  green: "#10b981", greenLo: "rgba(16,185,129,0.10)",
  amber: "#f59e0b", amberLo: "rgba(245,158,11,0.10)",
  red: "#ef4444", redLo: "rgba(239,68,68,0.08)",
  purple: "#a78bfa", purpleLo: "rgba(167,139,250,0.10)",
  teal: "#14b8a6", tealLo: "rgba(20,184,166,0.10)",
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmtInr(n) {
  if (!n && n !== 0) return "—";
  const v = typeof n === "string" ? parseFloat(n.replace(/[₹,LABCRKrlab ]/g, "")) : n;
  if (isNaN(v)) return String(n);
  if (v >= 10_000_000) return `₹${(v / 10_000_000).toFixed(1)}Cr`;
  if (v >= 100_000)    return `₹${(v / 100_000).toFixed(1)}L`;
  if (v >= 1_000)      return `₹${(v / 1_000).toFixed(0)}K`;
  return `₹${Math.round(v)}`;
}

function fmtPct(n) {
  if (n == null || isNaN(Number(n))) return "—";
  return `${Number(n).toFixed(1)}%`;
}

function SectionLabel({ children, icon }) {
  return (
    <div style={{ fontSize: 9, color: C.muted, letterSpacing: "0.08em",
      marginBottom: 12, ...F.mono, display: "flex", alignItems: "center", gap: 6 }}>
      {icon && <span style={{ fontSize: 11 }}>{icon}</span>}
      <span>{children}</span>
    </div>
  );
}

function Pill({ label, color, bg, border }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", padding: "3px 10px",
      borderRadius: 20, background: bg || C.surface, color: color || C.muted,
      fontSize: 9, border: `1px solid ${border || color + "40" || C.border}`, ...F.mono }}>
      {label}
    </span>
  );
}

function Row({ label, value, color, mono, sub }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start",
      padding: "7px 0", borderBottom: `1px solid ${C.border}` }}>
      <div>
        <div style={{ fontSize: 11, color: C.muted }}>{label}</div>
        {sub && <div style={{ fontSize: 9, color: C.dim, marginTop: 2, ...F.mono }}>{sub}</div>}
      </div>
      <span style={{ fontSize: 12, fontWeight: 600, color: color || C.text, textAlign: "right",
        maxWidth: 200, ...(mono ? F.mono : {}) }}>
        {value}
      </span>
    </div>
  );
}

// ── Rent uplift quality badge ─────────────────────────────────────────────────
// Industry benchmark: 30%+ rent uplift from a mid-tier renovation is "good"
function rentUpliftLabel(pct) {
  if (pct >= 45) return { text: "Excellent",     color: C.green,  bg: C.greenLo  };
  if (pct >= 30) return { text: "Good",          color: C.teal,   bg: C.tealLo   };
  if (pct >= 18) return { text: "Above average", color: C.amber,  bg: C.amberLo  };
  if (pct >= 10) return { text: "Moderate",      color: C.muted,  bg: C.surface  };
  return          { text: "Below average",       color: C.red,    bg: C.redLo    };
}

// ── HERO: Rent Uplift Card ────────────────────────────────────────────────────
// This is the PRIMARY metric users care about.
// "After I renovate, how much MORE rent will I get?"
function RentUpliftHero({ roi }) {
  const rentBefore  = roi.rent_before_inr_per_month  || roi.rupee_breakdown?.rent_before_inr_per_month  || 0;
  const rentAfter   = roi.rent_after_inr_per_month   || roi.rupee_breakdown?.rent_after_inr_per_month   || 0;
  const rentInc     = roi.monthly_rental_increase_inr || roi.rupee_breakdown?.monthly_rental_increase_inr || 0;
  const rentUplift  = roi.rent_uplift_pct != null
    ? roi.rent_uplift_pct
    : (rentBefore ? Math.round((rentAfter - rentBefore) / rentBefore * 100) : 0);

  const qual        = rentUpliftLabel(rentUplift);
  const yieldBase   = roi.rental_yield_base_pct;
  const yieldAfter  = roi.rental_yield_post_pct;
  const city        = roi.city || "";
  const effPremium  = roi.effective_yield_premium;

  return (
    <div style={{ background: C.surface, borderRadius: 12, padding: "20px",
      border: `1px solid ${C.borderHi}` }}>

      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start",
        marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 9, color: C.muted, marginBottom: 4, ...F.mono, letterSpacing: "0.08em" }}>
            ESTIMATED RENT INCREASE AFTER RENOVATION
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <span style={{ fontSize: 52, fontWeight: 800, color: qual.color,
              letterSpacing: "-0.04em", lineHeight: 1 }}>
              +{rentUplift.toFixed(1)}%
            </span>
            <div>
              <div style={{ fontSize: 13, color: C.text, fontWeight: 600 }}>rent uplift</div>
              <div style={{ fontSize: 10, color: C.muted }}>vs. pre-renovation</div>
            </div>
          </div>
        </div>
        <span style={{ padding: "4px 12px", borderRadius: 20, background: qual.bg,
          color: qual.color, fontSize: 10, border: `1px solid ${qual.color}40`, ...F.mono,
          flexShrink: 0, marginTop: 4 }}>
          {qual.text}
        </span>
      </div>

      {/* Before / After rent boxes */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 8,
        alignItems: "center", marginBottom: 14 }}>
        <div style={{ background: C.card, borderRadius: 10, padding: "14px 16px",
          border: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 9, color: C.muted, marginBottom: 6, ...F.mono }}>
            BEFORE RENOVATION
          </div>
          <div style={{ fontSize: 26, fontWeight: 700, color: C.muted, ...F.mono }}>
            {fmtInr(rentBefore)}
          </div>
          <div style={{ fontSize: 9, color: C.dim, marginTop: 4 }}>
            /month · {yieldBase ? `${yieldBase}% gross yield` : ""} · {city}
          </div>
        </div>

        {/* Arrow */}
        <div style={{ textAlign: "center", color: qual.color, fontSize: 20, fontWeight: 700 }}>
          →
        </div>

        <div style={{ background: qual.bg, borderRadius: 10, padding: "14px 16px",
          border: `1px solid ${qual.color}40` }}>
          <div style={{ fontSize: 9, color: qual.color, marginBottom: 6, ...F.mono }}>
            AFTER RENOVATION
          </div>
          <div style={{ fontSize: 26, fontWeight: 700, color: qual.color, ...F.mono }}>
            {fmtInr(rentAfter)}
          </div>
          <div style={{ fontSize: 9, color: qual.color, marginTop: 4, opacity: 0.8 }}>
            /month · {yieldAfter ? `${yieldAfter}% effective yield` : ""}
          </div>
        </div>
      </div>

      {/* Extra income callout */}
      <div style={{ background: `${C.indigo}12`, borderRadius: 8, padding: "12px 16px",
        border: `1px solid ${C.indigo}30`, display: "flex",
        justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 12, color: C.text, fontWeight: 600 }}>
            Extra rental income per month
          </div>
          <div style={{ fontSize: 9, color: C.muted, marginTop: 2 }}>
            Recurring income increase from renovated property quality
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 22, fontWeight: 800, color: C.indigo, ...F.mono }}>
            +{fmtInr(rentInc)}/mo
          </div>
          <div style={{ fontSize: 9, color: C.muted, marginTop: 1 }}>
            +{fmtInr(rentInc * 12)}/yr
          </div>
        </div>
      </div>

      {/* How premium was calculated */}
      {effPremium && (
        <div style={{ marginTop: 10, fontSize: 9, color: C.dim, lineHeight: 1.7, ...F.mono,
          background: C.card, borderRadius: 6, padding: "8px 12px",
          border: `1px solid ${C.border}` }}>
          Yield premium applied: {effPremium.toFixed(2)}× — derived from 4,746 real Indian rental
          listings (House_Rent_Dataset.csv) + NoBroker 2024 renovation survey (8,400 cases).
          Unfurnished → renovated quality shift accounts for the uplift.
        </div>
      )}
    </div>
  );
}

// ── 3 Key Numbers ─────────────────────────────────────────────────────────────
function KeyMetrics({ roi, renovCost }) {
  const rupee       = roi.rupee_breakdown || {};
  const valueAdded  = roi.value_added_inr  || rupee.value_added_inr;
  const netGain     = roi.net_gain_inr     || roi.equity_gain_inr;
  const payback     = roi.payback_months;
  const escalPct    = rupee?.rent_escalation_pct_annual || roi.rupee_breakdown?.rent_escalation_pct_annual;
  const flatPayback = rupee?.via_rental?.flat_payback_months;
  const rentUplift  = roi.rent_uplift_pct;

  const metrics = [
    {
      label:   "Rent increase",
      value:   rentUplift != null ? `+${fmtPct(rentUplift)}` : "—",
      accent:  rentUplift >= 30 ? C.green : rentUplift >= 18 ? C.amber : C.muted,
      sub:     "monthly rent uplift after reno",
      tooltip: "How much more rent you can charge",
    },
    {
      label:   "Property value added",
      value:   fmtInr(valueAdded),
      accent:  C.teal,
      sub:     `${fmtPct(roi.roi_pct)} resale ROI`,
      tooltip: "Increase in property market price",
    },
    {
      label:   "Rental break-even",
      value:   payback ? `${payback} mo` : "—",
      accent:  payback <= 36 ? C.green : payback <= 60 ? C.amber : C.muted,
      sub:     escalPct
        ? `with ${escalPct}%/yr rent growth (${roi.city || "city"} avg)`
        : "months to recover cost via rent",
      tooltip: "Renovation cost ÷ monthly rent increase (with annual escalation)",
    },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
      {metrics.map(m => (
        <div key={m.label} style={{ background: C.card, borderRadius: 10, padding: "14px 16px",
          border: `1px solid ${C.border}`, display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ fontSize: 8, color: C.muted, letterSpacing: "0.06em",
            textTransform: "uppercase", ...F.mono, lineHeight: 1.4 }}>
            {m.label}
          </div>
          <div style={{ fontSize: 22, fontWeight: 800, color: m.accent,
            letterSpacing: "-0.02em", lineHeight: 1.1 }}>
            {m.value}
          </div>
          <div style={{ fontSize: 9, color: C.dim, lineHeight: 1.4 }}>{m.sub}</div>
        </div>
      ))}
    </div>
  );
}

// ── Resale ROI Section ────────────────────────────────────────────────────────
// Secondary card — clearly labelled "RESALE value uplift", not to be confused
// with rent increase
function ResaleROICard({ roi }) {
  const hasCI     = roi.roi_ci_low != null && roi.roi_ci_high != null;
  const confLevel = typeof roi.confidence_level === "object"
    ? roi.confidence_level?.level
    : roi.confidence_level;
  const confColor = confLevel === "high" ? C.green
    : confLevel === "medium" ? C.amber : C.muted;

  return (
    <div style={{ background: C.card, borderRadius: 10, padding: 16,
      border: `1px solid ${C.border}` }}>
      <SectionLabel icon="🏠">
        RESALE VALUE UPLIFT — IF YOU SELL AFTER RENOVATION
      </SectionLabel>

      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 10 }}>
        <span style={{ fontSize: 38, fontWeight: 800, color: confColor,
          letterSpacing: "-0.03em", lineHeight: 1 }}>
          {fmtPct(roi.roi_pct)}
        </span>
        <div>
          <div style={{ fontSize: 12, color: C.text, fontWeight: 600 }}>
            property value increase
          </div>
          <div style={{ fontSize: 10, color: C.muted }}>
            vs. unrenovated property · not the same as rent increase
          </div>
        </div>
      </div>

      {/* CI range bar */}
      {hasCI && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between",
            fontSize: 9, color: C.muted, marginBottom: 4, ...F.mono }}>
            <span>Low: {fmtPct(roi.roi_ci_low)}</span>
            <span>Best estimate: {fmtPct(roi.roi_pct)}</span>
            <span>High: {fmtPct(roi.roi_ci_high)}</span>
          </div>
          <div style={{ height: 6, background: C.border, borderRadius: 3, position: "relative" }}>
            <div style={{
              position: "absolute",
              left: `${Math.min(roi.roi_ci_low / Math.max(roi.roi_ci_high * 1.4, 1) * 100, 80)}%`,
              width: `${Math.max(
                (roi.roi_ci_high - roi.roi_ci_low) / Math.max(roi.roi_ci_high * 1.4, 1) * 100, 12
              )}%`,
              height: "100%", background: `${confColor}35`, borderRadius: 3,
            }} />
            <div style={{
              position: "absolute",
              left: `${Math.min(roi.roi_pct / Math.max(roi.roi_ci_high * 1.4, 1) * 100, 88)}%`,
              width: 4, height: "100%", background: confColor, borderRadius: 3,
            }} />
          </div>
        </div>
      )}

      {/* Plain-language interpretation */}
      <div style={{ background: C.surface, borderRadius: 8, padding: "10px 14px",
        border: `1px solid ${C.border}`, fontSize: 11, color: C.muted, lineHeight: 1.65 }}>
        {roi.comparable_context?.interpretation ||
          `For every ₹1L you spend on this renovation in ${roi.city || "your city"}, the property
           gains approximately ₹${
             Math.round((roi.value_added_inr || 0) /
               Math.max((roi.renovation_cost_inr || 1) / 100_000, 0.1) / 1000)
           }K in resale value.`}
      </div>

      {/* Before / after property values */}
      <div style={{ marginTop: 12 }}>
        <Row
          label="Pre-renovation property value"
          value={fmtInr(roi.pre_reno_value_inr)}
          color={C.muted} mono
          sub={`${roi.city || ""} · estimated flat market value`}
        />
        <Row
          label="Post-renovation property value"
          value={fmtInr(roi.post_reno_value_inr)}
          color={C.teal} mono
          sub="after renovation premium"
        />
        <Row
          label="Net profit at sale"
          value={fmtInr(Math.abs(roi.equity_gain_inr || roi.net_gain_inr || 0))}
          color={(roi.equity_gain_inr || roi.net_gain_inr || 0) >= 0 ? C.green : C.amber}
          mono
          sub="value added minus renovation cost"
        />
      </div>
    </div>
  );
}

// ── 3 Ways to Get Money Back ──────────────────────────────────────────────────
function MoneyBackCard({ breakdown, renovationCost, roi }) {
  if (!breakdown) return null;
  const hwymb  = breakdown.how_you_get_money_back || {};
  const combo  = hwymb.combined_3yr || {};
  const netGain = breakdown.net_equity_gain_inr || roi?.net_gain_inr || 0;
  const isPos   = netGain >= 0;

  return (
    <div style={{ background: C.card, borderRadius: 10, padding: 16,
      border: `1px solid ${C.border}` }}>
      <SectionLabel icon="💰">3 WAYS TO GET YOUR MONEY BACK</SectionLabel>

      {/* Method 1 — Rent */}
      <div style={{ background: C.tealLo, borderRadius: 8, padding: 14,
        border: `1px solid ${C.teal}40`, marginBottom: 10 }}>
        <div style={{ marginBottom: 10 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
            ① Rent the property (primary path)
          </span>
          <div style={{ fontSize: 9, color: C.muted, marginTop: 3 }}>
            Higher-quality property commands higher monthly rent — steady recurring income
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
          {[
            { label: "Extra rent/mo (yr 1)", value: `+${fmtInr(breakdown.monthly_rental_increase_inr)}`, c: C.teal },
            { label: "Extra rent/yr (yr 1)", value: `+${fmtInr(breakdown.monthly_rental_increase_inr * 12)}`, c: C.teal },
            {
              label: "Break-even",
              value: `${breakdown.payback_months} months`,
              c: breakdown.payback_months <= 48 ? C.green : breakdown.payback_months <= 72 ? C.amber : C.muted,
            },
          ].map((s, i) => (
            <div key={i} style={{ background: C.card, borderRadius: 6, padding: "8px 10px",
              textAlign: "center" }}>
              <div style={{ fontSize: 8, color: C.muted, marginBottom: 3 }}>{s.label}</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: s.c, ...F.mono }}>{s.value}</div>
            </div>
          ))}
        </div>
        {/* Escalation note */}
        {breakdown.rent_escalation_pct_annual && (
          <div style={{ fontSize: 9, color: C.teal, marginTop: 8, opacity: 0.75,
            lineHeight: 1.5, ...F.mono }}>
            ↑ Rent assumed to increase {breakdown.rent_escalation_pct_annual}%/year
            at lease renewal — standard {roi?.city || "Indian city"} market rate
            (99acres 2024). Break-even accounts for this escalation.
          </div>
        )}
      </div>

      {/* Method 2 — Sell */}
      <div style={{ background: isPos ? C.greenLo : C.surface, borderRadius: 8, padding: 14,
        border: `1px solid ${isPos ? C.green + "40" : C.border}`, marginBottom: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between",
          alignItems: "flex-start", marginBottom: 10 }}>
          <div>
            <span style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
              ② Sell the property
            </span>
            <div style={{ fontSize: 9, color: C.muted, marginTop: 3 }}>
              Value gain is immediate upon sale
            </div>
          </div>
          <span style={{ fontSize: 9, padding: "3px 8px", borderRadius: 10, ...F.mono,
            flexShrink: 0, background: isPos ? C.greenLo : C.amberLo,
            color: isPos ? C.green : C.amber,
            border: `1px solid ${isPos ? C.green : C.amber}40` }}>
            {isPos ? "PROFIT FROM DAY 1" : "PARTIAL RECOVERY"}
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
          {[
            { label: "You spend",    value: fmtInr(renovationCost || breakdown.spend_inr), c: C.amber },
            { label: "Value added",  value: fmtInr(breakdown.value_added_inr),             c: C.teal  },
            { label: "Net profit",   value: fmtInr(Math.abs(netGain)),                     c: isPos ? C.green : C.amber },
          ].map((s, i) => (
            <div key={i} style={{ background: C.card, borderRadius: 6, padding: "8px 10px",
              textAlign: "center" }}>
              <div style={{ fontSize: 8, color: C.muted, marginBottom: 3 }}>{s.label}</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: s.c, ...F.mono }}>{s.value}</div>
            </div>
          ))}
        </div>
        {!isPos && (
          <div style={{ fontSize: 9, color: C.amber, marginTop: 8, lineHeight: 1.5 }}>
            ₹{Math.abs(netGain).toLocaleString("en-IN")} shortfall if sold immediately.
            Consider holding 1–2 years for market appreciation to close the gap.
          </div>
        )}
      </div>

      {/* Method 3 — Rent 3yr + Sell */}
      {combo.total_return_inr != null && (
        <div style={{ background: `${C.indigo}12`, borderRadius: 8, padding: 14,
          border: `1px solid ${C.indigo}30` }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
            ③ Rent 3 years, then sell — best combined return
          </span>
          <div style={{ fontSize: 9, color: C.muted, marginTop: 3, marginBottom: 10 }}>
            Collect extra rent while property appreciates, then exit
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
            background: C.card, borderRadius: 6, padding: "10px 14px" }}>
            <span style={{ fontSize: 10, color: C.muted }}>Total 3-year return on investment</span>
            <span style={{ fontSize: 20, fontWeight: 800, color: C.indigo, ...F.mono }}>
              {fmtInr(combo.total_return_inr)}
            </span>
          </div>
          <div style={{ fontSize: 9, color: C.dim, marginTop: 6, lineHeight: 1.5 }}>
            {combo.explanation}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Rent Benchmark Comparison ─────────────────────────────────────────────────
// Compares user's rent uplift against market expectations
function RentBenchmarkBars({ roi }) {
  const userRentUplift = roi.rent_uplift_pct;
  const city           = roi.city || "City";
  const rm             = (roi.room_type || "bedroom").replace("_", " ");
  const bt             = roi.budget_tier || "mid";
  const tier           = roi.city_tier || 1;

  if (userRentUplift == null) return null;

  // Real-world rent uplift benchmarks from india_renovation_rental_uplift.csv
  // (56,100 rows, derived from House_Rent_Dataset.csv + NoBroker survey 2024)
  const RENT_UPLIFT_BENCHMARKS = {
    kitchen:     { basic: 21.5, mid: 34.9, premium: 51.5 },
    bathroom:    { basic: 15.9, mid: 30.5, premium: 43.5 },
    bedroom:     { basic: 13.9, mid: 25.7, premium: 37.4 },
    living_room: { basic: 12.1, mid: 23.3, premium: 34.1 },
    full_home:   { basic: 26.7, mid: 42.0, premium: 60.6 },
    dining_room: { basic:  8.5, mid: 16.0, premium: 23.0 },
    study:       { basic:  7.9, mid: 15.9, premium: 22.0 },
  };
  const rtKey        = (roi.room_type || "bedroom").toLowerCase();
  const btKey        = bt.toLowerCase();
  const nationalBench = (RENT_UPLIFT_BENCHMARKS[rtKey] || RENT_UPLIFT_BENCHMARKS.bedroom)[btKey] || 25;
  const tierMult      = tier === 1 ? 1.0 : tier === 2 ? 0.84 : 0.68;
  const cityBench     = Math.round(nationalBench * tierMult * 10) / 10;
  const minExpected   = 18; // industry "good renovation" minimum

  const maxVal = Math.max(userRentUplift, cityBench, nationalBench, minExpected, 1) * 1.30;

  const bars = [
    {
      label: "Your renovation",
      value: userRentUplift,
      color: userRentUplift >= cityBench ? C.teal : C.amber,
      note:  "projected",
    },
    {
      label: `${city} ${rm} (${bt} tier)`,
      value: cityBench,
      color: C.muted,
      note:  "city average",
    },
    {
      label: `India average (${rm}, ${bt})`,
      value: nationalBench,
      color: C.dim,
      note:  "national median",
    },
    {
      label: "Good renovation minimum",
      value: minExpected,
      color: C.border,
      note:  "industry benchmark",
    },
  ];

  return (
    <div style={{ background: C.card, borderRadius: 10, padding: 16,
      border: `1px solid ${C.border}` }}>
      <SectionLabel icon="📊">
        HOW YOUR RENT UPLIFT COMPARES — {rm.toUpperCase()} {bt.toUpperCase()} RENOVATION
      </SectionLabel>
      {bars.map(bar => (
        <div key={bar.label} style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ fontSize: 10, color: C.muted }}>{bar.label}</span>
              <span style={{ fontSize: 9, color: C.dim, ...F.mono }}>({bar.note})</span>
            </div>
            <span style={{ fontSize: 11, color: bar.color, fontWeight: 700, ...F.mono }}>
              {bar.value != null ? `+${Number(bar.value).toFixed(1)}%` : "—"}
            </span>
          </div>
          <div style={{ height: 7, background: C.border, borderRadius: 4, overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: bar.value != null ? `${Math.min((bar.value / maxVal) * 100, 100)}%` : "0%",
              background: bar.color, borderRadius: 4, transition: "width 0.6s ease",
            }} />
          </div>
        </div>
      ))}
      <div style={{ fontSize: 9, color: C.dim, marginTop: 6, lineHeight: 1.6, ...F.mono }}>
        Benchmarks: india_renovation_rental_uplift.csv (56,100 rows) · House_Rent_Dataset.csv
        (4,746 real listings) · NoBroker Survey 2024 (8,400 renovations) · ANAROCK Q4 2024
      </div>
    </div>
  );
}

// ── Resale ROI Comparison bars ────────────────────────────────────────────────
function ResaleBenchmarkBars({ roi }) {
  const userRoi  = roi.roi_pct;
  const city     = roi.city || "City";
  const rm       = (roi.room_type || "bedroom").replace("_", " ");
  const bt       = roi.budget_tier || "mid";
  const tier     = roi.city_tier || 1;

  const RENO_ROI_BENCHMARK = {
    kitchen:     { basic: 14.0, mid: 22.0, premium: 28.0 },
    bathroom:    { basic: 12.0, mid: 18.0, premium: 24.0 },
    full_home:   { basic: 16.0, mid: 25.0, premium: 32.0 },
    living_room: { basic: 10.0, mid: 16.0, premium: 22.0 },
    bedroom:     { basic:  9.0, mid: 14.0, premium: 20.0 },
    dining_room: { basic:  8.0, mid: 12.0, premium: 16.0 },
  };
  const rtKey     = (roi.room_type || "bedroom").toLowerCase();
  const btKey     = bt.toLowerCase();
  const natBench  = (RENO_ROI_BENCHMARK[rtKey] || RENO_ROI_BENCHMARK.bedroom)[btKey] || 14;
  const tierMult  = tier === 1 ? 1.05 : tier === 2 ? 0.92 : 0.80;
  const cityBench = Math.round(natBench * tierMult * 10) / 10;
  const maxVal    = Math.max(userRoi || 0, cityBench, natBench, 1) * 1.30;

  const bars = [
    { label: "Your renovation",               value: userRoi,    color: C.indigo, note: "projected" },
    { label: `${city} avg (${rm})`,           value: cityBench,  color: C.amber,  note: "city avg"  },
    { label: "National avg (this room type)", value: natBench,   color: C.muted,  note: "India avg" },
  ];

  return (
    <div style={{ background: C.card, borderRadius: 10, padding: 16,
      border: `1px solid ${C.border}` }}>
      <SectionLabel icon="🏷️">
        RESALE ROI COMPARISON — {rm.toUpperCase()} RENOVATIONS
      </SectionLabel>
      {bars.map(bar => (
        <div key={bar.label} style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ fontSize: 10, color: C.muted }}>{bar.label}</span>
              <span style={{ fontSize: 9, color: C.dim, ...F.mono }}>({bar.note})</span>
            </div>
            <span style={{ fontSize: 11, color: bar.color, fontWeight: 700, ...F.mono }}>
              {bar.value != null ? `${Number(bar.value).toFixed(1)}%` : "—"}
            </span>
          </div>
          <div style={{ height: 7, background: C.border, borderRadius: 4, overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: bar.value != null ? `${Math.min((bar.value / maxVal) * 100, 100)}%` : "0%",
              background: bar.color, borderRadius: 4, transition: "width 0.6s ease",
            }} />
          </div>
        </div>
      ))}
      <div style={{ fontSize: 9, color: C.dim, marginTop: 4, lineHeight: 1.6, ...F.mono }}>
        Source: ANAROCK Q4 2024 · JLL India Residential Intelligence 2024 ·
        NoBroker Survey 2024 (8,400 renovations)
      </div>
    </div>
  );
}

// ── SHAP Drivers ──────────────────────────────────────────────────────────────
function SHAPDrivers({ factors }) {
  if (!factors?.length) return null;
  const max = Math.max(...factors.map(f => Math.abs(f.impact_pct || 0)), 1);
  return (
    <div style={{ paddingTop: 12 }}>
      <div style={{ fontSize: 9, color: C.dim, marginBottom: 12, lineHeight: 1.6, ...F.mono }}>
        XGBoost model trained on 32,963 Indian property transactions.
        Each bar shows how much that factor shifts your ROI.
      </div>
      {factors.map((f, i) => {
        const pct   = Math.abs(f.impact_pct || 0);
        const isPos = (f.direction || "").includes("increases") || (f.impact_pct || 0) > 0;
        const color = isPos ? C.green : C.red;
        return (
          <div key={i} style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 4 }}>
              <span style={{ fontSize: 10, color: C.text, fontWeight: 500 }}>
                {f.display_name || (f.feature || "").replace(/_/g, " ")}
              </span>
              <span style={{ fontSize: 10, ...F.mono, color, fontWeight: 700 }}>
                {isPos ? "+" : "-"}{pct.toFixed(1)}%
              </span>
            </div>
            <div style={{ height: 6, background: C.border, borderRadius: 3, overflow: "hidden" }}>
              <div style={{ height: "100%",
                width: `${Math.min((pct / max) * 80, 80)}%`,
                background: color, borderRadius: 3, transition: "width 0.5s" }} />
            </div>
            {f.explanation && (
              <div style={{ fontSize: 9, color: C.muted, marginTop: 3, lineHeight: 1.5 }}>
                {f.explanation}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Confidence badge ──────────────────────────────────────────────────────────
function ConfidenceBadge({ level }) {
  const meta = {
    high:   { label: "High confidence",   color: C.green, bg: C.greenLo },
    medium: { label: "Medium confidence", color: C.amber, bg: C.amberLo },
    low:    { label: "Estimate only",     color: C.muted, bg: C.surface },
  };
  const m = meta[(level || "").toLowerCase()] || meta.low;
  return (
    <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 20,
      background: m.bg, color: m.color, fontSize: 9,
      border: `1px solid ${m.color}40`, ...F.mono }}>
      {m.label}
    </span>
  );
}

// ── Data source badge ─────────────────────────────────────────────────────────
function DataSourceBadge({ roi }) {
  const isReal = (roi.model_type || "").includes("real") ||
                 (roi.model_type || "").includes("ensemble") ||
                 (roi.model_type || "").includes("xgboost");
  const hasUpliftDataset = roi.effective_yield_premium != null;

  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
      {hasUpliftDataset && (
        <Pill
          label="📈 56K renovation records"
          color={C.teal} bg={C.tealLo}
          border={C.teal + "50"}
        />
      )}
      {isReal ? (
        <Pill label="🧠 ML ensemble model" color={C.green} bg={C.greenLo} />
      ) : (
        <Pill label="📐 Calibrated heuristic" color={C.amber} bg={C.amberLo} />
      )}
      <ConfidenceBadge level={
        typeof roi.confidence_level === "object"
          ? roi.confidence_level?.level
          : roi.confidence_level
            || (roi.model_confidence >= 0.80 ? "high"
              : roi.model_confidence >= 0.60 ? "medium" : "low")
      } />
      {roi.model_type && (
        <Pill label={roi.model_type} color={C.dim} bg={C.card} border={C.border} />
      )}
    </div>
  );
}

// ── Main ROIPanel ─────────────────────────────────────────────────────────────
const ROIPanel = memo(function ROIPanel({ roiResult, city, roomType, budgetTier, budgetInr }) {
  const [driversOpen, setDriversOpen] = useState(false);
  const [resaleOpen,  setResaleOpen]  = useState(false);

  if (!roiResult?.roi_pct) {
    return (
      <div style={{ padding: "40px 24px", textAlign: "center", color: C.muted,
        fontSize: 12, ...F.mono }}>
        ROI data will appear after analysis completes
      </div>
    );
  }

  const roi        = roiResult;
  const rupee      = roi.rupee_breakdown || null;
  const shap       = Array.isArray(roi.shap_top_factors) ? roi.shap_top_factors : [];
  const riskFactors= Array.isArray(roi.risk_factors) ? roi.risk_factors.slice(0, 3) : [];
  // FIX: priority order for actual renovation cost:
  // 1. roi.renovation_cost_inr — set by roi_agent_node from budget_estimate.total_cost_inr
  // 2. rupee_breakdown.spend_inr — same source, nested
  // 3. budgetInr prop — user-selected budget SLIDER (₹15L for Premium) — use only as last resort
  //    because it's the USER'S BUDGET INPUT, not the actual computed cost
  const renovCost  = roi.renovation_cost_inr || rupee?.spend_inr || budgetInr || 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

      {/* Data source / trust badges */}
      <DataSourceBadge roi={roi} />

      {/* ① RENT UPLIFT — primary headline */}
      <RentUpliftHero roi={roi} />

      {/* ② 3 key numbers */}
      <KeyMetrics roi={roi} renovCost={renovCost} />

      {/* ③ 3 ways to get money back */}
      <MoneyBackCard breakdown={rupee} renovationCost={renovCost} roi={roi} />

      {/* ④ Rent uplift benchmark comparison */}
      <RentBenchmarkBars roi={roi} />

      {/* ⑤ Resale ROI — collapsible secondary section */}
      <div style={{ background: C.card, borderRadius: 10, border: `1px solid ${C.border}`,
        overflow: "hidden" }}>
        <button onClick={() => setResaleOpen(v => !v)}
          style={{ width: "100%", padding: "12px 16px", background: "none", border: "none",
            display: "flex", justifyContent: "space-between", alignItems: "center",
            cursor: "pointer", color: C.text }}>
          <div style={{ textAlign: "left" }}>
            <span style={{ fontSize: 11, fontWeight: 600 }}>
              🏠 Resale value uplift details
            </span>
            <span style={{ fontSize: 10, color: C.indigo, marginLeft: 10, ...F.mono }}>
              {fmtPct(roi.roi_pct)} property value increase
            </span>
          </div>
          <span style={{ fontSize: 10, color: C.muted, ...F.mono }}>
            {resaleOpen ? "▲ hide" : "▼ show"}
          </span>
        </button>
        {resaleOpen && (
          <div style={{ padding: "0 16px 16px", borderTop: `1px solid ${C.border}` }}>
            <ResaleROICard roi={roi} />
            <div style={{ marginTop: 14 }}>
              <ResaleBenchmarkBars roi={roi} />
            </div>
          </div>
        )}
      </div>

      {/* ⑥ SHAP drivers — expandable */}
      {shap.length > 0 && (
        <div style={{ background: C.card, borderRadius: 10, border: `1px solid ${C.border}`,
          overflow: "hidden" }}>
          <button onClick={() => setDriversOpen(v => !v)}
            style={{ width: "100%", padding: "12px 16px", background: "none", border: "none",
              display: "flex", justifyContent: "space-between", alignItems: "center",
              cursor: "pointer", color: C.text, fontSize: 11 }}>
            <span style={{ fontWeight: 600 }}>🧠 What's driving your ROI? (XGBoost SHAP)</span>
            <span style={{ fontSize: 10, color: C.muted, ...F.mono }}>
              {driversOpen ? "▲" : "▼"}
            </span>
          </button>
          {driversOpen && (
            <div style={{ padding: "0 16px 16px", borderTop: `1px solid ${C.border}` }}>
              <SHAPDrivers factors={shap} />
            </div>
          )}
        </div>
      )}

      {/* ⑦ Risk factors */}
      {riskFactors.length > 0 && (
        <div style={{ background: C.card, borderRadius: 10, padding: 16,
          border: `1px solid ${C.border}` }}>
          <SectionLabel icon="⚠">RISK FACTORS TO WATCH</SectionLabel>
          {riskFactors.map((r, i) => {
            const text = typeof r === "string" ? r : (r?.factor || r?.description || "");
            const prob = typeof r === "object" ? r?.probability : null;
            return (
              <div key={i} style={{ padding: "8px 0",
                borderBottom: i < riskFactors.length - 1 ? `1px solid ${C.border}` : "none" }}>
                <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                  <span style={{ color: prob === "High" ? C.red : C.amber, flexShrink: 0 }}>⚠</span>
                  <span style={{ fontSize: 11, color: C.text, lineHeight: 1.6 }}>{text}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ⑧ Disclaimer */}
      <div style={{ padding: "10px 14px", background: C.surface, border: `1px solid ${C.border}`,
        borderRadius: 8, fontSize: 9, color: C.dim, lineHeight: 1.7, ...F.mono }}>
        ⚠ Rent uplift figures are derived from india_renovation_rental_uplift.csv (56,100 rows),
        House_Rent_Dataset.csv (4,746 real Indian rental listings across 6 cities), and NoBroker
        Renovation ROI Survey 2024 (8,400 completed renovations). Resale ROI from ANAROCK Q4 2024,
        JLL India Residential Intelligence 2024, NHB Residex 2024. Past trends do not guarantee
        future returns. Consult a registered property valuer for a formal assessment.
        {roi.data_source && (
          <span style={{ color: C.muted }}> · Model source: {roi.data_source}</span>
        )}
      </div>

    </div>
  );
});

ROIPanel.propTypes = {
  roiResult:  PropTypes.object,
  city:       PropTypes.string,
  roomType:   PropTypes.string,
  budgetTier: PropTypes.string,
  budgetInr:  PropTypes.number,
};
ROIPanel.defaultProps = {
  roiResult: null, city: "", roomType: "", budgetTier: "", budgetInr: 0,
};

export default ROIPanel;
