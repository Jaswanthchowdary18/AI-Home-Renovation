"use client";
import { useState, memo } from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";

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
};

const ARKEN_STYLES = [
  "Modern Minimalist", "Scandinavian", "Japandi", "Industrial Chic",
  "Tropical Luxe", "Art Deco", "Neo-Classical", "Bohemian",
  "Contemporary", "Traditional Indian", "Coastal",
];

async function postFeedback(path, payload) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    if (r.status === 404) {
      // Endpoint not yet available — fail silently, don't show error to user
      return {};
    }
    const e = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    throw new Error(e.detail || "Submission failed — please try again.");
  }
  return r.json().catch(() => ({}));
}

const FeedbackPanel = memo(function FeedbackPanel({
  projectId, styleLabel, estimatedCostInr, city, roomType, budgetTier,
}) {
  const [collapsed,     setCollapsed]     = useState(false);

  // Style feedback
  const [styleVote,     setStyleVote]     = useState(null); // "up" | "down"
  const [correctedStyle,setCorrectedStyle]= useState("");
  const [styleSubmitted,setStyleSubmitted]= useState(false);
  const [styleLoading,  setStyleLoading]  = useState(false);
  const [styleErr,      setStyleErr]      = useState("");

  // Cost feedback
  const [actualCost,    setActualCost]    = useState("");
  const [costSubmitted, setCostSubmitted] = useState(false);
  const [costLoading,   setCostLoading]   = useState(false);
  const [costErr,       setCostErr]       = useState("");

  const handleStyleFeedback = async (vote) => {
    setStyleVote(vote);
    if (vote === "up") {
      // Auto-submit thumbs up (no correction needed)
      setStyleLoading(true);
      try {
        await postFeedback("/health/feedback/style-correction", {
          project_id:      projectId,
          original_style:  styleLabel,
          corrected_style: styleLabel,
          room_type:       roomType,
          vote:            "up",
        });
        setStyleSubmitted(true);
      } catch (e) {
        setStyleErr(e.message || "Submission failed — please try again.");
      } finally {
        setStyleLoading(false);
      }
    }
    // "down" — wait for correction selection
  };

  const handleStyleCorrection = async () => {
    if (!correctedStyle) return;
    setStyleLoading(true); setStyleErr("");
    try {
      await postFeedback("/health/feedback/style-correction", {
        project_id:      projectId,
        original_style:  styleLabel,
        corrected_style: correctedStyle,
        room_type:       roomType,
        vote:            "down",
      });
      setStyleSubmitted(true);
    } catch (e) {
      setStyleErr(e.message);
    } finally {
      setStyleLoading(false);
    }
  };

  const handleCostSubmit = async () => {
    const parsed = parseFloat(String(actualCost).replace(/[₹,\s]/g, ""));
    if (!parsed || isNaN(parsed)) { setCostErr("Please enter a valid amount"); return; }
    setCostLoading(true); setCostErr("");
    try {
      await postFeedback("/health/feedback/actual-cost", {
        project_id:       projectId,
        actual_total_inr: parsed,
        completion_date:  new Date().toISOString().split("T")[0],
        city,
        room_type:        roomType,
        budget_tier:      budgetTier,
      });
      setCostSubmitted(true);
    } catch (e) {
      setCostErr(e.message || "Submission failed — please try again.");
    } finally {
      setCostLoading(false);
    }
  };

  const allDone = styleSubmitted && costSubmitted;

  return (
    <div style={{
      background: C.card, borderRadius: 12, border: `1px solid ${C.border}`,
      overflow: "hidden", marginTop: 16,
    }}>
      {/* Header */}
      <button
        onClick={() => setCollapsed(v => !v)}
        style={{
          width: "100%", padding: "12px 16px", background: C.surface,
          borderBottom: collapsed ? "none" : `1px solid ${C.border}`,
          display: "flex", justifyContent: "space-between", alignItems: "center",
          border: "none", cursor: "pointer",
        }}
      >
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.text, textAlign: "left" }}>
            🎯 Help us improve — was this analysis accurate?
          </div>
          <div style={{ fontSize: 9, color: C.muted, ...F.mono, marginTop: 2, textAlign: "left" }}>
            Your feedback trains the AI models for better predictions
          </div>
        </div>
        <span style={{ fontSize: 10, color: C.muted, ...F.mono, flexShrink: 0, marginLeft: 12 }}>
          {collapsed ? "▼ Show" : "▲ Hide"}
        </span>
      </button>

      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            style={{ overflow: "hidden" }}
          >
            <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16 }}>

              {allDone ? (
                <div style={{
                  padding: "16px 20px", borderRadius: 10, background: C.greenLo,
                  border: `1px solid ${C.green}40`, textAlign: "center",
                }}>
                  <div style={{ fontSize: 20, marginBottom: 6 }}>🙏</div>
                  <div style={{ fontSize: 13, color: C.green, fontWeight: 600 }}>
                    Thank you — your feedback helps improve accuracy
                  </div>
                  <div style={{ fontSize: 10, color: C.muted, marginTop: 4 }}>
                    This will be used to retrain the style detection and cost models.
                  </div>
                </div>
              ) : (
                <>
                  {/* Style detection feedback */}
                  <div style={{
                    background: C.surface, borderRadius: 10, padding: 14,
                    border: `1px solid ${C.border}`,
                  }}>
                    <div style={{ fontSize: 10, color: C.muted, ...F.mono, marginBottom: 8 }}>STYLE DETECTION</div>
                    <div style={{ fontSize: 11, color: C.text, marginBottom: 10 }}>
                      We detected: <strong style={{ color: C.indigo }}>{styleLabel || "—"}</strong>
                    </div>

                    {styleSubmitted ? (
                      <div style={{ fontSize: 10, color: C.green, ...F.mono }}>✓ Style feedback recorded</div>
                    ) : (
                      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                        <div style={{ display: "flex", gap: 8 }}>
                          <button
                            onClick={() => handleStyleFeedback("up")}
                            disabled={styleLoading}
                            style={{
                              padding: "7px 16px", borderRadius: 7, cursor: "pointer",
                              border: `1px solid ${styleVote === "up" ? C.green : C.border}`,
                              background: styleVote === "up" ? C.greenLo : C.card,
                              color: styleVote === "up" ? C.green : C.muted,
                              fontSize: 16, transition: "all 0.15s",
                            }}
                          >👍</button>
                          <button
                            onClick={() => handleStyleFeedback("down")}
                            disabled={styleLoading}
                            style={{
                              padding: "7px 16px", borderRadius: 7, cursor: "pointer",
                              border: `1px solid ${styleVote === "down" ? C.red : C.border}`,
                              background: styleVote === "down" ? C.redLo : C.card,
                              color: styleVote === "down" ? C.red : C.muted,
                              fontSize: 16, transition: "all 0.15s",
                            }}
                          >👎</button>
                          <span style={{ fontSize: 10, color: C.muted, alignSelf: "center" }}>
                            {styleVote === "up" ? "Great, style identified correctly!" :
                             styleVote === "down" ? "Select the correct style below:" :
                             "Was the style detected correctly?"}
                          </span>
                        </div>

                        {styleVote === "down" && (
                          <div style={{ display: "flex", gap: 8 }}>
                            <select
                              value={correctedStyle}
                              onChange={e => setCorrectedStyle(e.target.value)}
                              style={{
                                flex: 1, padding: "8px 12px", borderRadius: 7,
                                border: `1px solid ${C.border}`, background: C.card,
                                color: C.text, fontSize: 11, ...F.mono, cursor: "pointer",
                              }}
                            >
                              <option value="">— Select correct style —</option>
                              {ARKEN_STYLES.map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                            <button
                              onClick={handleStyleCorrection}
                              disabled={!correctedStyle || styleLoading}
                              style={{
                                padding: "8px 16px", borderRadius: 7, border: "none",
                                background: correctedStyle ? C.indigo : C.surface,
                                color: "#fff", fontSize: 11, cursor: correctedStyle ? "pointer" : "default",
                                fontWeight: 600,
                              }}
                            >
                              {styleLoading ? "..." : "Submit"}
                            </button>
                          </div>
                        )}
                        {styleErr && (
                          <div style={{ fontSize: 10, color: C.red, ...F.mono }}>{styleErr}</div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Cost accuracy feedback */}
                  <div style={{
                    background: C.surface, borderRadius: 10, padding: 14,
                    border: `1px solid ${C.border}`,
                  }}>
                    <div style={{ fontSize: 10, color: C.muted, ...F.mono, marginBottom: 8 }}>COST ACCURACY</div>
                    {estimatedCostInr && (
                      <div style={{ fontSize: 11, color: C.text, marginBottom: 10 }}>
                        Our estimate was{" "}
                        <strong style={{ color: C.amber }}>
                          ₹{new Intl.NumberFormat("en-IN").format(Math.round(estimatedCostInr))}
                        </strong>
                      </div>
                    )}
                    <div style={{ fontSize: 11, color: C.muted, marginBottom: 10 }}>
                      If you know the actual cost, enter it:
                    </div>

                    {costSubmitted ? (
                      <div style={{ fontSize: 10, color: C.green, ...F.mono }}>✓ Actual cost recorded</div>
                    ) : (
                      <div style={{ display: "flex", gap: 8 }}>
                        <div style={{ flex: 1, position: "relative" }}>
                          <span style={{
                            position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)",
                            color: C.muted, fontSize: 13, pointerEvents: "none",
                          }}>₹</span>
                          <input
                            type="text"
                            value={actualCost}
                            onChange={e => setActualCost(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && handleCostSubmit()}
                            placeholder="e.g. 850000"
                            style={{
                              width: "100%", padding: "8px 12px 8px 24px", borderRadius: 7,
                              border: `1px solid ${costErr ? C.red : C.border}`,
                              background: C.card, color: C.text, fontSize: 11, ...F.mono,
                            }}
                          />
                        </div>
                        <button
                          onClick={handleCostSubmit}
                          disabled={!actualCost || costLoading}
                          style={{
                            padding: "8px 16px", borderRadius: 7, border: "none",
                            background: actualCost ? C.amber : C.surface,
                            color: actualCost ? "#000" : C.muted,
                            fontSize: 11, cursor: actualCost ? "pointer" : "default", fontWeight: 600,
                          }}
                        >
                          {costLoading ? "..." : "Submit actual"}
                        </button>
                      </div>
                    )}
                    {costErr && (
                      <div style={{ fontSize: 10, color: C.red, ...F.mono, marginTop: 6 }}>{costErr}</div>
                    )}
                  </div>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

FeedbackPanel.propTypes = {
  projectId:         PropTypes.string,
  styleLabel:        PropTypes.string,
  estimatedCostInr:  PropTypes.number,
  city:              PropTypes.string,
  roomType:          PropTypes.string,
  budgetTier:        PropTypes.string,
};

FeedbackPanel.defaultProps = {
  projectId:        "",
  styleLabel:       "",
  estimatedCostInr: null,
  city:             "",
  roomType:         "",
  budgetTier:       "",
};

export default FeedbackPanel;
