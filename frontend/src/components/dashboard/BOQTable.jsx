"use client";
import { useState, useCallback, memo } from "react";
import PropTypes from "prop-types";

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
};

function fmtInr(n) {
  if (!n && n !== 0) return "—";
  const v = typeof n === "string" ? parseFloat(n.replace(/[₹,LABCRKrlab ]/g, "")) : n;
  if (isNaN(v)) return String(n);
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(v);
}

const CAT_COLORS = {
  Paint:    "#6470f3",
  Tiles:    "#10b981",
  Labour:   "#f59e0b",
  Fittings: "#a78bfa",
  Other:    "#5a7090",
};

function categorizeBOQ(items) {
  const cats = { Paint: 0, Tiles: 0, Labour: 0, Fittings: 0, Other: 0 };
  (items || []).forEach(item => {
    const cat = (item.category || "").toLowerCase();
    const name = (item.product || item.brand || "").toLowerCase();
    if (cat.includes("paint") || name.includes("paint") || name.includes("primer")) cats.Paint += item.total_inr || 0;
    else if (cat.includes("tile") || name.includes("tile") || name.includes("floor") || name.includes("marble")) cats.Tiles += item.total_inr || 0;
    else if (cat.startsWith("labour") || cat.includes("labor") || cat.includes("installation") || cat.includes("work")) cats.Labour += item.total_inr || 0;
    else if (cat.includes("fitting") || cat.includes("fixture") || cat.includes("hardware") || name.includes("door") || name.includes("handle")) cats.Fittings += item.total_inr || 0;
    else cats.Other += item.total_inr || 0;
  });
  return cats;
}

function PieChart({ items }) {
  const cats = categorizeBOQ(items);
  const total = Object.values(cats).reduce((a, b) => a + b, 0);
  if (!total) return null;

  const R = 56, CX = 70, CY = 70;
  let cumAngle = -Math.PI / 2;
  const slices = Object.entries(cats)
    .filter(([, v]) => v > 0)
    .map(([label, value]) => {
      const frac = value / total;
      const angle = frac * 2 * Math.PI;
      const x1 = CX + R * Math.cos(cumAngle);
      const y1 = CY + R * Math.sin(cumAngle);
      cumAngle += angle;
      const x2 = CX + R * Math.cos(cumAngle);
      const y2 = CY + R * Math.sin(cumAngle);
      const large = angle > Math.PI ? 1 : 0;
      return { label, value, frac, x1, y1, x2, y2, large };
    });

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
      <svg width={140} height={140} viewBox="0 0 140 140">
        {slices.map(s => (
          <path
            key={s.label}
            d={`M${CX},${CY} L${s.x1},${s.y1} A${R},${R} 0 ${s.large},1 ${s.x2},${s.y2} Z`}
            fill={CAT_COLORS[s.label] || C.muted}
            opacity={0.88}
            stroke={C.bg}
            strokeWidth={1.5}
          />
        ))}
        <circle cx={CX} cy={CY} r={28} fill={C.card} />
        <text x={CX} y={CY - 5} textAnchor="middle" fill={C.text} fontSize={8} fontFamily="JetBrains Mono,monospace">COST</text>
        <text x={CX} y={CY + 7} textAnchor="middle" fill={C.text} fontSize={8} fontFamily="JetBrains Mono,monospace">MIX</text>
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {slices.map(s => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: CAT_COLORS[s.label] || C.muted, flexShrink: 0 }} />
            <span style={{ fontSize: 9, color: C.muted, ...F.mono, minWidth: 52 }}>{s.label}</span>
            <span style={{ fontSize: 9, color: C.text, ...F.mono }}>{(s.frac * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const BOQTable = memo(function BOQTable({ boqItems, totalCost, city, budgetTier, labourInr }) {
  const [sortCol, setSortCol]   = useState("total_inr");
  const [sortDir, setSortDir]   = useState("desc");

  const items = boqItems || [];

  const sorted = [...items].sort((a, b) => {
    const av = a[sortCol] ?? (typeof a[sortCol] === "string" ? a[sortCol] : 0);
    const bv = b[sortCol] ?? (typeof b[sortCol] === "string" ? b[sortCol] : 0);
    if (typeof av === "number") return sortDir === "asc" ? av - bv : bv - av;
    return sortDir === "asc" ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  });

  // Top 3 most expensive by total_inr
  const top3Ids = new Set(
    [...items].sort((a, b) => (b.total_inr || 0) - (a.total_inr || 0)).slice(0, 3).map((_, i) => i)
  );
  // But we need to track in sorted array. Let's use total_inr threshold instead.
  const sortedByPrice = [...items].sort((a, b) => (b.total_inr || 0) - (a.total_inr || 0));
  const top3Threshold = sortedByPrice[2]?.total_inr ?? 0;

  const handleSort = useCallback((col) => {
    if (sortCol === col) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortCol(col); setSortDir("desc"); }
  }, [sortCol]);

  const downloadCSV = useCallback(() => {
    const headers = ["Category","Brand","Product","SKU","Qty","Unit","Unit Price (INR)","Total (INR)"];
    const rows = items.map(item => [
      item.category || "",
      item.brand || "",
      item.product || "",
      item.sku || "",
      item.qty || "",
      item.unit || "",
      item.rate_inr || 0,
      item.total_inr || 0,
    ]);
    const csvContent = [headers, ...rows]
      .map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(","))
      .join("\n");
    const meta = `"ARKEN BOQ","${city || ""}","${budgetTier || ""}","Total: ${totalCost || 0}"\n\n`;
    const blob = new Blob([meta + csvContent], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `arken-boq-${city || "report"}-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [items, city, budgetTier, totalCost]);

  if (!items.length && !totalCost) {
    return (
      <div style={{ padding: "40px 24px", textAlign: "center", color: C.muted, fontSize: 12, ...F.mono }}>
        BOQ will appear after pipeline analysis runs
      </div>
    );
  }

  const COLS = [
    { key: "category", label: "Category" },
    { key: "brand",    label: "Brand" },
    { key: "product",  label: "Product" },
    { key: "qty",      label: "Qty" },
    { key: "unit",     label: "Unit" },
    { key: "rate_inr", label: "Unit Price" },
    { key: "total_inr",label: "Total" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: 9, color: C.muted, ...F.mono, letterSpacing: "0.08em" }}>
          BILL OF QUANTITIES — {items.length} LINE ITEMS · {city} · {budgetTier?.toUpperCase()}
        </div>
        <button
          onClick={downloadCSV}
          style={{
            padding: "5px 12px", borderRadius: 6, border: `1px solid ${C.green}50`,
            background: C.greenLo, color: C.green, fontSize: 9, cursor: "pointer", ...F.mono,
          }}
        >
          ↓ Download CSV
        </button>
      </div>

      {/* Pie chart + total */}
      {items.length > 0 && (
        <div style={{
          background: C.card, borderRadius: 12, padding: 16,
          border: `1px solid ${C.border}`, display: "flex",
          justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 16,
        }}>
          <PieChart items={items} />
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 9, color: C.muted, ...F.mono, marginBottom: 4 }}>TOTAL (incl. GST + contingency)</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: C.green, letterSpacing: "-0.02em" }}>
              {fmtInr(totalCost)}
            </div>
          </div>
        </div>
      )}

      {/* Labour vs Material split + GST note */}
      {items.length > 0 && (() => {
        // Labour - items are now visible BOQ line items (design_planner v3).
        // Sum them from the items array; labourInr prop is a fallback only.
        const labourFromItems = items
          .filter(i => (i.category || "").startsWith("Labour"))
          .reduce((s, item) => s + (item.total_inr || 0), 0);
        const labourTotal = labourFromItems || labourInr || 0;
        const materialTotal = items
          .filter(i => !(i.category || "").startsWith("Labour"))
          .reduce((s, item) => s + (item.total_inr || 0), 0);
        const gstEstimate = Math.round(materialTotal * 0.18 + labourTotal * 0.12);
        const contingency = Math.round((labourTotal + materialTotal) * 0.10);

        const catBreakdown = {};
        items.forEach(item => {
          const cat = item.category || "Other";
          catBreakdown[cat] = (catBreakdown[cat] || 0) + (item.total_inr || 0);
        });
        const sortedCats = Object.entries(catBreakdown).sort((a,b) => b[1]-a[1]);

        return (
          <div style={{ background: C.card, borderRadius: 10, padding: 16, border: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 9, color: C.muted, ...F.mono, letterSpacing: "0.08em", marginBottom: 12 }}>
              COST BREAKDOWN — LABOUR VS MATERIALS
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
              {[
                { label: "Materials",         value: fmtInr(materialTotal),  color: C.indigo, pct: materialTotal && (labourTotal + materialTotal) ? Math.round(materialTotal / (labourTotal + materialTotal) * 100) : null },
                { label: "Labour",            value: fmtInr(labourTotal),    color: C.amber,  pct: labourTotal && (labourTotal + materialTotal) ? Math.round(labourTotal / (labourTotal + materialTotal) * 100) : null },
                { label: "GST est. (18% on materials)", value: fmtInr(gstEstimate), color: C.muted, pct: null },
                { label: "Contingency (5%)", value: fmtInr(contingency), color: C.muted, pct: null },
              ].map(row => (
                <div key={row.label} style={{ background: C.surface, borderRadius: 8, padding: "10px 12px",
                  border: `1px solid ${C.border}` }}>
                  <div style={{ fontSize: 9, color: C.muted, marginBottom: 3 }}>{row.label}</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: row.color, ...F.mono }}>
                    {row.value}
                    {row.pct != null && <span style={{ fontSize: 9, color: C.dim, marginLeft: 4 }}>{row.pct}%</span>}
                  </div>
                </div>
              ))}
            </div>

            {/* Category subtotals */}
            {sortedCats.length > 1 && (
              <>
                <div style={{ fontSize: 9, color: C.muted, ...F.mono, letterSpacing: "0.08em", marginBottom: 8 }}>
                  BY CATEGORY
                </div>
                {sortedCats.map(([cat, val]) => {
                  const pct = totalCost ? Math.round(val / totalCost * 100) : 0;
                  return (
                    <div key={cat} style={{ display: "flex", justifyContent: "space-between",
                      padding: "5px 0", borderBottom: `1px solid ${C.border}`, alignItems: "center" }}>
                      <span style={{ fontSize: 10, color: C.muted }}>{cat}</span>
                      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                        <div style={{ width: 60, height: 4, background: C.border, borderRadius: 2, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${pct}%`, background: C.indigo, borderRadius: 2 }} />
                        </div>
                        <span style={{ fontSize: 10, color: C.text, ...F.mono, minWidth: 48, textAlign: "right" }}>
                          {fmtInr(val)}
                        </span>
                        <span style={{ fontSize: 9, color: C.dim, ...F.mono, minWidth: 28 }}>{pct}%</span>
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </div>
        );
      })()}

      {/* Table */}
      {items.length > 0 && (
        <div style={{ background: C.card, borderRadius: 10, border: `1px solid ${C.border}`, overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: C.surface }}>
                  {COLS.map(col => (
                    <th
                      key={col.key}
                      onClick={() => handleSort(col.key)}
                      style={{
                        padding: "8px 12px", textAlign: "left", fontSize: 9,
                        color: sortCol === col.key ? C.indigo : C.muted,
                        ...F.mono, letterSpacing: "0.06em",
                        borderBottom: `1px solid ${C.border}`,
                        cursor: "pointer", userSelect: "none",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {col.label} {sortCol === col.key ? (sortDir === "asc" ? "↑" : "↓") : ""}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((item, i) => {
                  const isTop3 = (item.total_inr || 0) >= top3Threshold && top3Threshold > 0;
                  return (
                    <tr
                      key={i}
                      style={{
                        borderBottom: `1px solid ${C.border}`,
                        background: i % 2 === 0 ? "transparent" : C.surface,
                        borderLeft: isTop3 ? `3px solid ${C.amber}` : "3px solid transparent",
                      }}
                    >
                      {[
                        item.category,
                        item.brand,
                        item.product,
                        item.qty,
                        item.unit,
                        fmtInr(item.rate_inr),
                        fmtInr(item.total_inr),
                      ].map((v, j) => (
                        <td
                          key={j}
                          style={{
                            padding: "8px 12px", fontSize: 11,
                            color: j === 6 ? C.green : j === 0 ? C.text : C.muted,
                            fontWeight: j === 6 ? 600 : 400,
                            ...(j >= 3 ? F.mono : {}),
                          }}
                        >
                          {v || "—"}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr style={{ background: C.indigoLo, borderTop: `1px solid ${C.border}` }}>
                  <td colSpan={6} style={{ padding: "10px 12px", fontSize: 12, fontWeight: 700, color: C.text }}>
                    TOTAL (incl. GST + contingency)
                  </td>
                  <td style={{ padding: "10px 12px", fontSize: 15, fontWeight: 800, color: C.green, ...F.mono }}>
                    {fmtInr(totalCost)}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      )}

      {!items.length && totalCost && (
        <div style={{
          background: C.indigoLo, borderRadius: 10, padding: 14,
          border: `1px solid ${C.indigo}40`, fontSize: 11, color: C.muted,
        }}>
          Total project cost: <strong style={{ color: C.green, fontSize: 16 }}>{fmtInr(totalCost)}</strong>
          {" "}(detailed BOQ unavailable — check pipeline errors)
        </div>
      )}
    </div>
  );
});

BOQTable.propTypes = {
  boqItems:   PropTypes.arrayOf(PropTypes.object),
  totalCost:  PropTypes.number,
  city:       PropTypes.string,
  budgetTier: PropTypes.string,
  labourInr:  PropTypes.number,
};

BOQTable.defaultProps = {
  boqItems:   [],
  totalCost:  null,
  city:       "",
  budgetTier: "",
  labourInr:  0,
};

export default BOQTable;
