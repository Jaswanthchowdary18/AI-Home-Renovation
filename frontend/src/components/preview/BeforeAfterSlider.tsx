/**
 * ARKEN — Before/After Comparison Slider
 * Drag to compare original room vs Gemini 2.5 Flash renovation.
 */

"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";

interface Props {
  before?: string | null;
  after?: string | null;
}

export function BeforeAfterSlider({ before, after }: Props) {
  const [pos, setPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updatePos = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full select-none overflow-hidden rounded-xl"
      style={{ aspectRatio: "16/9", cursor: "ew-resize", background: "#0a1020", border: "1px solid #1e293b" }}
      onMouseDown={() => (dragging.current = true)}
      onMouseUp={() => (dragging.current = false)}
      onMouseLeave={() => (dragging.current = false)}
      onMouseMove={(e) => dragging.current && updatePos(e.clientX)}
      onTouchStart={() => (dragging.current = true)}
      onTouchEnd={() => (dragging.current = false)}
      onTouchMove={(e) => updatePos(e.touches[0].clientX)}
    >
      {/* BEFORE */}
      <div className="absolute inset-0">
        {before ? (
          <img src={before} alt="Before renovation" className="w-full h-full object-cover" />
        ) : (
          <EmptySlot label="ORIGINAL ROOM" icon="◈" color="#475569" />
        )}
        <SlotBadge label="BEFORE" side="left" color="rgba(15,23,42,0.8)" textColor="#64748b" />
      </div>

      {/* AFTER — clipped */}
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${pos}%` }}>
        <div className="absolute top-0 left-0 bottom-0" style={{ width: `${(100 / pos) * 100}%` }}>
          {after ? (
            <img src={after} alt="After renovation" className="w-full h-full object-cover" />
          ) : (
            <EmptySlot label="AI RENDERED" icon="▣" color="#6d28d9" />
          )}
          <SlotBadge
            label="AFTER"
            side="right"
            color="rgba(99,102,241,0.2)"
            textColor="#a5b4fc"
            border="1px solid #4c1d95"
          />
        </div>
      </div>

      {/* Divider handle */}
      <div
        className="absolute top-0 bottom-0 flex items-center justify-center z-20"
        style={{ left: `${pos}%`, transform: "translateX(-50%)" }}
      >
        <div className="w-px h-full" style={{ background: "#a78bfa" }} />
        <motion.div
          className="absolute w-9 h-9 rounded-full flex items-center justify-center shadow-xl"
          style={{ background: "#7c3aed", border: "2px solid #a78bfa" }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <span style={{ color: "#fff", fontSize: 12, userSelect: "none" }}>⟺</span>
        </motion.div>
      </div>

      {/* Model badge */}
      {after && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="absolute bottom-3 right-3 flex items-center gap-1.5 px-2.5 py-1 rounded-lg"
          style={{
            background: "rgba(99,102,241,0.25)",
            border: "1px solid rgba(99,102,241,0.4)",
            backdropFilter: "blur(8px)",
          }}
        >
          <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
          <span style={{ fontSize: 10, color: "#a5b4fc", fontFamily: "'DM Mono', monospace" }}>
            GEMINI 2.5 FLASH · IMAGE GEN
          </span>
        </motion.div>
      )}
    </div>
  );
}

function EmptySlot({ label, icon, color }: { label: string; icon: string; color: string }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center"
      style={{ background: "linear-gradient(135deg, #0f172a, #0a1020)" }}>
      <span style={{ fontSize: 52, opacity: 0.12, color }}>{icon}</span>
      <span style={{ fontSize: 11, color: "#334155", fontFamily: "'DM Mono', monospace", marginTop: 8 }}>
        {label}
      </span>
    </div>
  );
}

function SlotBadge({ label, side, color, textColor, border }: any) {
  return (
    <div className="absolute top-3 px-2 py-1 rounded text-xs font-medium"
      style={{
        [side === "left" ? "right" : "left"]: 12,
        background: color, color: textColor,
        fontFamily: "'DM Mono', monospace",
        border: border || "1px solid #1e293b",
      }}>
      {label}
    </div>
  );
}
