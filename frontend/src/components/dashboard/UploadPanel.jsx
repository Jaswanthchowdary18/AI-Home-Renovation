"use client";
import { useState, useRef, useCallback } from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";

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

const CITIES  = ["Hyderabad","Bangalore","Mumbai","Delhi NCR","Pune","Chennai"];
const THEMES  = ["Modern Minimalist","Scandinavian","Japandi","Industrial Chic","Tropical Luxe","Art Deco","Neo-Classical","Bohemian"];
const ROOMS   = ["Bedroom","Living Room","Kitchen","Bathroom","Dining Room","Study / Home Office"];
const BUDGETS = [
  { label:"Basic",   range:"₹3–5L",  inr:400000,  tier:"basic",   color:C.muted },
  { label:"Mid",     range:"₹5–10L", inr:750000,  tier:"mid",     color:C.amber },
  { label:"Premium", range:"₹10L+",  inr:1500000, tier:"premium", color:C.green },
];

const ACCEPTED_TYPES = ["image/jpeg","image/png","image/webp","image/heic","image/heif"];
const MAX_SIZE_MB = 20;

function b64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(",")[1]);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

export default function UploadPanel({ onSubmit, loading }) {
  const [city,     setCity]     = useState("Hyderabad");
  const [room,     setRoom]     = useState("Bedroom");
  const [theme,    setTheme]    = useState("Modern Minimalist");
  const [budget,   setBudget]   = useState(BUDGETS[1]);
  const [preview,  setPreview]  = useState(null);
  const [fileErr,  setFileErr]  = useState("");
  const [tipsOpen, setTipsOpen] = useState(false);
  const fileRef = useRef(null);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    // Type check
    const typeOk = ACCEPTED_TYPES.includes(file.type) || file.name.toLowerCase().match(/\.(jpg|jpeg|png|webp|heic)$/);
    if (!typeOk) {
      setFileErr("Unsupported format. Please use JPG, PNG, WebP, or HEIC.");
      return;
    }
    // Size check
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setFileErr(`File too large (${(file.size/1024/1024).toFixed(1)} MB). Maximum is ${MAX_SIZE_MB} MB.`);
      return;
    }
    setFileErr("");
    const url = URL.createObjectURL(file);
    const imgB64 = await b64(file);
    setPreview({ url, b64: imgB64, mime: file.type || "image/jpeg", name: file.name });
  }, []);

  const handleInputChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleSubmit = useCallback(() => {
    if (!preview) return;
    onSubmit({ file: preview, city, room, budget, theme });
  }, [preview, city, room, budget, theme, onSubmit]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Drop zone */}
      <div
        onDragOver={e => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => !preview && fileRef.current?.click()}
        style={{
          border: `1.5px dashed ${fileErr ? C.red : preview ? C.green : C.borderHi}`,
          borderRadius: 16, padding: preview ? 0 : "48px 36px",
          cursor: preview ? "default" : "pointer",
          background: C.surface, transition: "all 0.2s", overflow: "hidden",
          position: "relative",
        }}
        onMouseEnter={e => { if (!preview) e.currentTarget.style.borderColor = C.indigo; }}
        onMouseLeave={e => { if (!preview) e.currentTarget.style.borderColor = fileErr ? C.red : C.borderHi; }}
      >
        {preview ? (
          <div style={{ position: "relative" }}>
            <img
              src={preview.url} alt="Room preview"
              style={{ width: "100%", maxHeight: 320, objectFit: "contain", display: "block", borderRadius: 14 }}
            />
            <div style={{
              position: "absolute", top: 10, right: 10,
              display: "flex", gap: 6,
            }}>
              <button
                onClick={e => { e.stopPropagation(); fileRef.current?.click(); }}
                style={{
                  padding: "5px 12px", borderRadius: 6, border: `1px solid ${C.border}`,
                  background: "rgba(10,17,32,0.85)", color: C.muted, fontSize: 10, cursor: "pointer", ...F.mono,
                }}
              >Change photo</button>
            </div>
            <div style={{
              position: "absolute", bottom: 10, left: 10, fontSize: 9, ...F.mono,
              background: "rgba(0,0,0,0.75)", color: C.green, padding: "3px 10px", borderRadius: 5,
            }}>
              ✓ {preview.name}
            </div>
          </div>
        ) : (
          <>
            <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3, textAlign: "center" }}>⬆</div>
            <div style={{ fontSize: 13, color: C.muted, textAlign: "center" }}>Drop or click to upload room photo</div>
            <div style={{ fontSize: 10, color: C.dim, marginTop: 6, ...F.mono, textAlign: "center" }}>JPG · PNG · WebP · HEIC · max {MAX_SIZE_MB}MB</div>
          </>
        )}
      </div>
      <input ref={fileRef} type="file" accept="image/*,.heic" onChange={handleInputChange} style={{ display: "none" }} />

      {/* File error */}
      <AnimatePresence>
        {fileErr && (
          <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{
              background: C.redLo, border: `1px solid ${C.red}60`, borderRadius: 8,
              padding: "10px 14px", fontSize: 11, color: C.red, ...F.mono,
            }}>
            ✗ {fileErr}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Room photo tips */}
      <div style={{ background: C.card, borderRadius: 10, border: `1px solid ${C.border}`, overflow: "hidden" }}>
        <button
          onClick={() => setTipsOpen(v => !v)}
          style={{
            width: "100%", padding: "10px 14px", background: "none", border: "none",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            cursor: "pointer", color: C.muted, fontSize: 11,
          }}
        >
          <span>📸 Room photo tips for best results</span>
          <span style={{ ...F.mono, fontSize: 10 }}>{tipsOpen ? "▲" : "▼"}</span>
        </button>
        <AnimatePresence>
          {tipsOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }} style={{ overflow: "hidden" }}
            >
              <div style={{ padding: "0 14px 14px", borderTop: `1px solid ${C.border}` }}>
                <p style={{ fontSize: 10, color: C.muted, lineHeight: 1.7, margin: "12px 0 10px" }}>
                  For best results: ensure good lighting, capture the full room, avoid extreme angles.
                </p>
                <div style={{ display: "flex", gap: 8 }}>
                  {[
                    { bg: "linear-gradient(135deg,#1e3a5f,#0a1120)", label: "Good lighting" },
                    { bg: "linear-gradient(135deg,#2d1b4e,#0d1525)", label: "Full room view" },
                    { bg: "linear-gradient(135deg,#1a3320,#0d1525)", label: "Natural angle" },
                  ].map(tip => (
                    <div key={tip.label} style={{
                      flex: 1, aspectRatio: "4/3", borderRadius: 8, background: tip.bg,
                      border: `1px solid ${C.border}`, display: "flex", alignItems: "flex-end",
                      padding: "6px 8px",
                    }}>
                      <span style={{ fontSize: 8, color: C.muted, ...F.mono }}>{tip.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* City selector */}
      <div>
        <div style={{ fontSize: 9, color: C.muted, ...F.mono, marginBottom: 8 }}>CITY</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6 }}>
          {CITIES.map(c => (
            <button key={c} onClick={() => setCity(c)}
              style={{
                fontSize: 10, padding: "8px 4px", borderRadius: 7, cursor: "pointer",
                border: `1px solid ${city === c ? C.indigo : C.border}`,
                background: city === c ? C.indigoLo : C.surface,
                color: city === c ? C.indigo : C.muted, ...F.mono, transition: "all 0.15s",
              }}>
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* Room type */}
      <div>
        <div style={{ fontSize: 9, color: C.muted, ...F.mono, marginBottom: 8 }}>ROOM TYPE</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          {ROOMS.map(r => (
            <button key={r} onClick={() => setRoom(r)}
              style={{
                fontSize: 10, padding: "8px 10px", borderRadius: 7, cursor: "pointer",
                border: `1px solid ${room === r ? C.amber : C.border}`,
                background: room === r ? C.amberLo : C.surface,
                color: room === r ? C.amber : C.muted, textAlign: "left", ...F.mono, transition: "all 0.15s",
              }}>
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Theme */}
      <div>
        <div style={{ fontSize: 9, color: C.muted, ...F.mono, marginBottom: 8 }}>DESIGN THEME</div>
        <select value={theme} onChange={e => setTheme(e.target.value)}
          style={{
            width: "100%", padding: "10px 14px", borderRadius: 8, border: `1px solid ${C.border}`,
            background: C.surface, color: C.text, fontSize: 12, cursor: "pointer", ...F.mono,
          }}>
          {THEMES.map(t => <option key={t}>{t}</option>)}
        </select>
      </div>

      {/* Budget */}
      <div>
        <div style={{ fontSize: 9, color: C.muted, ...F.mono, marginBottom: 8 }}>BUDGET TIER</div>
        <div style={{ display: "flex", gap: 8 }}>
          {BUDGETS.map(b => (
            <button key={b.label} onClick={() => setBudget(b)}
              style={{
                flex: 1, padding: "12px 8px", borderRadius: 8, cursor: "pointer",
                border: `1px solid ${budget.label === b.label ? b.color : C.border}`,
                background: budget.label === b.label ? `${b.color}18` : C.surface,
                color: budget.label === b.label ? b.color : C.muted, transition: "all 0.15s",
              }}>
              <div style={{ fontSize: 12, fontWeight: 700 }}>{b.label}</div>
              <div style={{ fontSize: 10, ...F.mono, opacity: 0.7 }}>{b.range}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!preview || loading}
        style={{
          padding: "15px 0", borderRadius: 12, border: "none",
          background: !preview || loading
            ? "linear-gradient(90deg,#1a2540,#2a3f6b)"
            : "linear-gradient(135deg,#6470f3,#a78bfa)",
          color: "#fff", fontSize: 13, fontWeight: 700,
          cursor: !preview || loading ? "default" : "pointer",
          transition: "opacity 0.2s", position: "relative", overflow: "hidden",
        }}
      >
        {loading
          ? <span style={{ ...F.mono, fontSize: 11 }}>◈ Processing — Gemini rendering...</span>
          : !preview
            ? "Upload a photo to begin"
            : "Analyse with ARKEN →"}
        {loading && (
          <motion.div animate={{ x: ["-100%", "100%"] }} transition={{ repeat: Infinity, duration: 1.4, ease: "linear" }}
            style={{ position: "absolute", inset: 0, background: "linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent)" }} />
        )}
      </button>
    </div>
  );
}

UploadPanel.propTypes = {
  onSubmit: PropTypes.func.isRequired,
  loading: PropTypes.bool,
};

UploadPanel.defaultProps = {
  loading: false,
};
