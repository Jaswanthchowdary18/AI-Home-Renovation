"use client";
import { useState, useMemo, memo } from "react";
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

// ── Vastu scoring engine (client-side, no API) ─────────────────────────────
function computeVastu(roomType, styleLabel, detectedObjects) {
  const objects = Array.isArray(detectedObjects)
    ? detectedObjects.map(o => String(typeof o === "string" ? o : (o?.name || o?.label || o?.change || o?.description || "")).toLowerCase()).filter(Boolean)
    : [];
  const style   = (styleLabel || "").toLowerCase();
  const room    = (roomType  || "").toLowerCase().replace(/_/g, " ");

  let score    = 80;
  const issues = [];

  // ── Bedroom rules ──────────────────────────────────────────────────────
  if (room.includes("bedroom")) {
    const hasMirror = objects.some(o => o.includes("mirror"));
    if (hasMirror) {
      score -= 15;
      issues.push({
        icon: "⚠",
        issue: "Mirror opposite bed detected",
        recommendation: "Move mirror to side wall or inside wardrobe door — mirrors facing the bed cause restless sleep per Vastu.",
        severity: "high",
      });
    }
    const hasDarkTheme = style.includes("dark") || style.includes("industrial") || style.includes("goth");
    if (hasDarkTheme) {
      score -= 10;
      issues.push({
        icon: "⚠",
        issue: "Dark/heavy colour palette",
        recommendation: "Prefer soft pastels or earthy tones for the bedroom. Dark reds and blacks are considered inauspicious in sleeping spaces.",
        severity: "medium",
      });
    }
    const hasSouthWindow = objects.some(o => o.includes("south") && o.includes("window"));
    if (hasSouthWindow) {
      score -= 5;
      issues.push({
        icon: "⚠",
        issue: "Large window on south wall",
        recommendation: "Use heavy curtains on south-facing windows to reduce afternoon heat and energy imbalance.",
        severity: "low",
      });
    }
    if (!hasMirror && !hasDarkTheme) {
      issues.push({
        icon: "✓",
        issue: "Bedroom layout is Vastu-compatible",
        recommendation: "Ensure the bed head points South or East for ideal sleep direction.",
        severity: "ok",
      });
    }
  }

  // ── Kitchen rules ──────────────────────────────────────────────────────
  else if (room.includes("kitchen")) {
    const stoveNorth = objects.some(o => (o.includes("stove") || o.includes("hob") || o.includes("range")) && o.includes("north"));
    if (stoveNorth) {
      score -= 15;
      issues.push({
        icon: "⚠",
        issue: "Stove positioned facing north",
        recommendation: "Stove should face East or South-East (Agni corner). North-facing stove disrupts fire energy.",
        severity: "high",
      });
    }
    const sinkAdjacentStove = objects.some(o => o.includes("sink")) && objects.some(o => o.includes("stove") || o.includes("hob"));
    if (sinkAdjacentStove) {
      score -= 10;
      issues.push({
        icon: "⚠",
        issue: "Sink adjacent to stove",
        recommendation: "Keep sink (water) and stove (fire) elements separated by at least one counter unit to avoid elemental conflict.",
        severity: "medium",
      });
    }
    if (!stoveNorth) {
      issues.push({
        icon: "✓",
        issue: "Kitchen stove direction is acceptable",
        recommendation: "Keep the cooking area in the South-East (Agni) corner of the kitchen for best Vastu compliance.",
        severity: "ok",
      });
    }
  }

  // ── Bathroom rules ─────────────────────────────────────────────────────
  else if (room.includes("bathroom")) {
    const isNorthEast = objects.some(o => o.includes("northeast") || (o.includes("north") && o.includes("east")));
    if (isNorthEast) {
      score -= 20;
      issues.push({
        icon: "⚠",
        issue: "Bathroom in North-East corner",
        recommendation: "North-East is the Ishanya (divine) direction. Relocate or install powerful ventilation and keep the door always shut.",
        severity: "high",
      });
    }
    const doorFacesNorth = objects.some(o => o.includes("door") && o.includes("north"));
    if (doorFacesNorth) {
      score -= 5;
      issues.push({
        icon: "⚠",
        issue: "Bathroom door facing north",
        recommendation: "Prefer West or South-facing bathroom doors. Keep the door closed when not in use.",
        severity: "low",
      });
    }
    if (!isNorthEast) {
      issues.push({
        icon: "✓",
        issue: "Bathroom placement is acceptable",
        recommendation: "Ideal bathroom positions: West or South-West of the home. Ensure good ventilation.",
        severity: "ok",
      });
    }
  }

  // ── Living room rules ──────────────────────────────────────────────────
  else if (room.includes("living") || room.includes("lounge")) {
    const southEntrance = objects.some(o => o.includes("entrance") && o.includes("south"));
    if (southEntrance) {
      score -= 15;
      issues.push({
        icon: "⚠",
        issue: "Main entrance facing south",
        recommendation: "South-facing entrances require a Vastu correction — place a Swastika or Om symbol above the door and use bright lighting.",
        severity: "high",
      });
    }
    const beamOverSeating = objects.some(o => o.includes("beam")) && objects.some(o => o.includes("sofa") || o.includes("seating"));
    if (beamOverSeating) {
      score -= 10;
      issues.push({
        icon: "⚠",
        issue: "Exposed beam directly over seating area",
        recommendation: "Beams above seating create downward pressure energy. Use a false ceiling or place seating away from beams.",
        severity: "medium",
      });
    }
    if (!southEntrance && !beamOverSeating) {
      issues.push({
        icon: "✓",
        issue: "Living room layout is Vastu-compatible",
        recommendation: "Place heavy furniture in South or West zones. Keep North-East corner clear or with light décor.",
        severity: "ok",
      });
    }
  }

  // ── Dining room ────────────────────────────────────────────────────────
  else if (room.includes("dining")) {
    issues.push({
      icon: "✓",
      issue: "Dining room baseline",
      recommendation: "Ideal dining direction: West is auspicious. Avoid South-East for dining table placement.",
      severity: "ok",
    });
  }

  // ── Study / Office ─────────────────────────────────────────────────────
  else if (room.includes("study") || room.includes("office")) {
    issues.push({
      icon: "✓",
      issue: "Study room baseline",
      recommendation: "Sit facing North or East while working for better concentration. Avoid sitting with your back to the door.",
      severity: "ok",
    });
  }

  // Clamp score
  score = Math.max(0, Math.min(100, score));

  return { score, issues };
}

// Circular arc gauge (SVG)
function ScoreGauge({ score }) {
  const R = 52;
  const CX = 70, CY = 70;
  const startAngle = -Math.PI * 0.75;
  const endAngle   =  Math.PI * 0.75;
  const totalArc   = endAngle - startAngle;
  const filledArc  = totalArc * (score / 100);

  function polarToXY(angle, r) {
    return [CX + r * Math.cos(angle), CY + r * Math.sin(angle)];
  }

  const [tx1, ty1] = polarToXY(startAngle, R);
  const [tx2, ty2] = polarToXY(endAngle,   R);
  const [fx1, fy1] = polarToXY(startAngle, R);
  const [fx2, fy2] = polarToXY(startAngle + filledArc, R);

  const largeArc     = totalArc  > Math.PI ? 1 : 0;
  const filledLarge  = filledArc > Math.PI ? 1 : 0;

  const color = score >= 70 ? C.green : score >= 50 ? C.amber : C.red;
  const label = score >= 70 ? "Good"  : score >= 50 ? "Fair"  : "Needs attention";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <svg width={140} height={110} viewBox="0 0 140 110">
        {/* Track */}
        <path
          d={`M ${tx1} ${ty1} A ${R} ${R} 0 ${largeArc} 1 ${tx2} ${ty2}`}
          fill="none" stroke={C.border} strokeWidth={10} strokeLinecap="round"
        />
        {/* Filled arc */}
        {score > 0 && (
          <path
            d={`M ${fx1} ${fy1} A ${R} ${R} 0 ${filledLarge} 1 ${fx2} ${fy2}`}
            fill="none" stroke={color} strokeWidth={10} strokeLinecap="round"
          />
        )}
        {/* Score text */}
        <text x={CX} y={CY + 4} textAnchor="middle"
          fill={color} fontSize={22} fontWeight={800}
          fontFamily="JetBrains Mono,monospace">
          {score}
        </text>
        <text x={CX} y={CY + 20} textAnchor="middle"
          fill={C.muted} fontSize={9}
          fontFamily="JetBrains Mono,monospace">
          / 100
        </text>
      </svg>
      <div style={{
        fontSize: 11, fontWeight: 600, color,
        padding: "3px 12px", borderRadius: 20,
        background: `${color}18`, border: `1px solid ${color}40`,
        marginTop: -4,
      }}>
        {label}
      </div>
    </div>
  );
}

const VastuPanel = memo(function VastuPanel({ roomType, styleLabel, detectedObjects }) {
  const [tooltipOpen, setTooltipOpen] = useState(false);

  const { score, issues } = useMemo(
    () => computeVastu(roomType, styleLabel, detectedObjects),
    [roomType, styleLabel, detectedObjects]
  );

  const scoreColor = score >= 70 ? C.green : score >= 50 ? C.amber : C.red;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Panel header with tooltip */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ fontSize: 9, color: C.muted, ...F.mono, letterSpacing: "0.08em" }}>VASTU COMPLIANCE SCORE</div>
        <div style={{ position: "relative", display: "inline-block" }}>
          <span
            onMouseEnter={() => setTooltipOpen(true)}
            onMouseLeave={() => setTooltipOpen(false)}
            style={{ fontSize: 11, color: C.muted, cursor: "help", userSelect: "none" }}
          >ⓘ</span>
          {tooltipOpen && (
            <div style={{
              position: "absolute", bottom: "calc(100% + 6px)", left: "50%",
              transform: "translateX(-50%)", zIndex: 999,
              background: C.card, border: `1px solid ${C.borderHi}`, borderRadius: 8,
              padding: "10px 14px", minWidth: 280, fontSize: 10, color: C.muted,
              lineHeight: 1.7, boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
            }}>
              <strong style={{ color: C.text, display: "block", marginBottom: 4 }}>What is Vastu Shastra?</strong>
              Vastu Shastra is the ancient Indian science of space and architecture — a system of directional alignments,
              elemental placements, and room proportions designed to promote health, prosperity, and harmony in the home.
              It remains highly relevant in modern Indian real estate decisions.
            </div>
          )}
        </div>
      </div>

      {/* Score gauge + quick summary */}
      <div style={{
        background: C.card, borderRadius: 12, padding: 20,
        border: `1px solid ${C.border}`,
        display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap",
      }}>
        <ScoreGauge score={score} />
        <div style={{ flex: 1, minWidth: 180 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: 6 }}>
            {roomType?.replace(/_/g, " ") || "Room"} Vastu Analysis
          </div>
          <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.7 }}>
            {score >= 70
              ? "This room layout is largely Vastu-compliant. Minor optimisations can further enhance energy flow."
              : score >= 50
                ? "Some Vastu concerns detected. Addressing the issues below can improve harmony and energy balance."
                : "Multiple Vastu issues detected. Consider consulting a certified Vastu expert for structural guidance."}
          </div>
          {styleLabel && (
            <div style={{ marginTop: 8, fontSize: 9, color: C.muted, ...F.mono }}>
              Style: {styleLabel} · Analysis based on visual detection
            </div>
          )}
        </div>
      </div>

      {/* Issues list */}
      <div style={{ background: C.card, borderRadius: 12, border: `1px solid ${C.border}`, overflow: "hidden" }}>
        <div style={{
          padding: "10px 16px", background: C.surface, borderBottom: `1px solid ${C.border}`,
          fontSize: 9, color: C.muted, ...F.mono, letterSpacing: "0.08em",
        }}>
          VASTU FINDINGS — {issues.length} ITEM{issues.length !== 1 ? "S" : ""}
        </div>
        <div style={{ padding: "8px 0" }}>
          {issues.map((item, i) => {
            const isOk   = item.severity === "ok";
            const isHigh = item.severity === "high";
            const itemColor = isOk ? C.green : isHigh ? C.red : C.amber;
            return (
              <div key={i} style={{
                padding: "12px 16px",
                borderBottom: i < issues.length - 1 ? `1px solid ${C.border}` : "none",
                display: "flex", gap: 12, alignItems: "flex-start",
                borderLeft: `3px solid ${itemColor}`,
                marginLeft: 0,
              }}>
                <span style={{ fontSize: 16, flexShrink: 0, lineHeight: 1.3 }}>{item.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginBottom: 3 }}>
                    {item.issue}
                  </div>
                  <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.65 }}>
                    {item.recommendation}
                  </div>
                </div>
                {!isOk && (
                  <span style={{
                    fontSize: 8, ...F.mono, padding: "2px 6px", borderRadius: 4,
                    background: `${itemColor}20`, color: itemColor, border: `1px solid ${itemColor}40`,
                    flexShrink: 0, marginTop: 2,
                  }}>
                    {item.severity}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Disclaimer */}
      <div style={{
        padding: "10px 14px", background: C.surface,
        border: `1px solid ${C.border}`, borderRadius: 8,
        fontSize: 9, color: C.dim, lineHeight: 1.7, ...F.mono,
      }}>
        🔮 Vastu analysis is advisory and based on classical Vastu Shastra principles applied to visual object detection.
        Consult a certified Vastu expert before making structural or architectural decisions.
      </div>
    </div>
  );
});

VastuPanel.propTypes = {
  roomType:        PropTypes.string,
  styleLabel:      PropTypes.string,
  detectedObjects: PropTypes.arrayOf(PropTypes.oneOfType([PropTypes.string, PropTypes.object])),
};

VastuPanel.defaultProps = {
  roomType:        "",
  styleLabel:      "",
  detectedObjects: [],
};

export default VastuPanel;
