"use client";
import { useState, useEffect, useRef } from "react";
import PropTypes from "prop-types";

const API = typeof process !== "undefined"
  ? (process.env?.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1")
  : "http://localhost:8000/api/v1";

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
  purple: "#a78bfa",
};

export default function ModelHealthBadge() {
  const [health, setHealth]     = useState(null);
  const [status, setStatus]     = useState("loading"); // loading | good | degraded | critical | unknown
  const [expanded, setExpanded] = useState(false);
  const panelRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);

    (async () => {
      try {
        const r = await fetch(`${API}/health/drift`, { signal: controller.signal });
        clearTimeout(timer);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        setHealth(data);
        // Derive overall status from drift data
        const styleAcc = data.style_accuracy_30d ?? data.style_accuracy ?? null;
        const roiOk    = data.roi_model_status ?? "ok";
        if (styleAcc !== null) {
          if (styleAcc >= 0.80 && roiOk !== "degraded" && roiOk !== "critical") setStatus("good");
          else if (styleAcc >= 0.60) setStatus("degraded");
          else setStatus("critical");
        } else {
          setStatus("good"); // API responded but no accuracy metrics — treat as good
        }
      } catch (e) {
        clearTimeout(timer);
        setStatus("unknown");
      }
    })();

    return () => { clearTimeout(timer); controller.abort(); };
  }, []);

  // Close panel when clicking outside
  useEffect(() => {
    if (!expanded) return;
    const handler = (e) => {
      if (panelRef.current && !panelRef.current.contains(e.target)) setExpanded(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [expanded]);

  const META = {
    loading:  { dot: C.dim,    label: "Models: Checking" },
    good:     { dot: C.green,  label: "Models: Good" },
    degraded: { dot: C.amber,  label: "Models: Degraded" },
    critical: { dot: C.red,    label: "Models: Critical" },
    unknown:  { dot: C.muted,  label: "Models: Unknown" },
  };
  const m = META[status] || META.unknown;

  const styleAccPct = health?.style_accuracy_30d != null
    ? `${(health.style_accuracy_30d * 100).toFixed(1)}%`
    : health?.style_accuracy != null
      ? `${(health.style_accuracy * 100).toFixed(1)}%`
      : "—";

  return (
    <div ref={panelRef} style={{ position: "relative", display: "inline-block" }}>
      {/* Badge */}
      <button
        onClick={() => setExpanded(v => !v)}
        style={{
          display: "flex", alignItems: "center", gap: 6,
          padding: "4px 10px", borderRadius: 6,
          border: `1px solid ${C.border}`, background: C.surface,
          cursor: "pointer", ...F.mono,
        }}
      >
        <span style={{
          width: 7, height: 7, borderRadius: "50%",
          background: m.dot,
          boxShadow: status === "good" ? `0 0 6px ${C.green}80` :
                     status === "degraded" ? `0 0 6px ${C.amber}80` :
                     status === "critical" ? `0 0 6px ${C.red}80` : "none",
          flexShrink: 0,
          animation: status === "loading" ? "pulse 1.2s ease-in-out infinite" : "none",
        }} />
        <span style={{ fontSize: 9, color: m.dot }}>{m.label}</span>
      </button>

      {/* Expanded panel */}
      {expanded && (
        <div style={{
          position: "absolute", top: "calc(100% + 8px)", right: 0, zIndex: 9999,
          background: C.card, border: `1px solid ${C.borderHi}`,
          borderRadius: 10, padding: 14, minWidth: 260,
          boxShadow: "0 12px 40px rgba(0,0,0,0.6)",
        }}>
          <div style={{ fontSize: 9, color: C.muted, ...F.mono, letterSpacing: "0.08em", marginBottom: 10 }}>
            ML MODEL HEALTH
          </div>

          {[
            {
              label: "Style Accuracy (30d)",
              value: styleAccPct,
              color: health?.style_accuracy_30d >= 0.8 ? C.green : C.amber,
            },
            {
              label: "ROI Model",
              value: health?.roi_model_status ? String(health.roi_model_status).toUpperCase() : (status === "unknown" ? "UNKNOWN" : "OK"),
              color: health?.roi_model_status === "degraded" ? C.amber :
                     health?.roi_model_status === "critical" ? C.red : C.green,
            },
            {
              label: "Price Forecast",
              value: health?.price_forecast_status ? String(health.price_forecast_status).toUpperCase() : (status === "unknown" ? "UNKNOWN" : "OK"),
              color: health?.price_forecast_status === "degraded" ? C.amber : C.green,
            },
            {
              label: "Last Evaluation",
              value: health?.last_evaluated_at
                ? new Date(health.last_evaluated_at).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "2-digit" })
                : health?.last_evaluation_date || "—",
              color: C.muted,
            },
          ].map(row => (
            <div key={row.label} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "6px 0", borderBottom: `1px solid ${C.border}`,
            }}>
              <span style={{ fontSize: 10, color: C.muted }}>{row.label}</span>
              <span style={{ fontSize: 10, fontWeight: 600, color: row.color, ...F.mono }}>{row.value}</span>
            </div>
          ))}

          {status === "unknown" && (
            <div style={{ fontSize: 9, color: C.dim, ...F.mono, marginTop: 8, lineHeight: 1.6 }}>
              Health endpoint unavailable — dashboard functions normally.
            </div>
          )}
        </div>
      )}

      <style>{`@keyframes pulse { 0%,100%{opacity:0.4} 50%{opacity:1} }`}</style>
    </div>
  );
}

ModelHealthBadge.propTypes = {};
