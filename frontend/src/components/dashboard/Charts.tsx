/**
 * ARKEN — Gantt Chart Component
 */

"use client";

import { motion } from "framer-motion";
import { ScheduleTask } from "@/store/arken";

const COLORS = ["#6366f1","#f59e0b","#10b981","#ec4899","#06b6d4","#8b5cf6","#f97316","#84cc16"];

interface GanttProps {
  tasks: ScheduleTask[];
  totalDays?: number;
}

export function GanttChart({ tasks, totalDays }: GanttProps) {
  const maxDay = totalDays || Math.max(...tasks.map((t) => t.end_day), 22);

  return (
    <div className="space-y-2">
      {tasks.map((task, i) => (
        <motion.div
          key={task.id}
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.05 }}
          className="flex items-center gap-3"
        >
          <div
            className="text-right flex-shrink-0"
            style={{ width: 190, fontSize: 10, color: "#64748b", fontFamily: "'DM Mono', monospace", lineHeight: 1.3 }}
          >
            {task.name}
          </div>
          <div className="flex-1 relative" style={{ height: 22 }}>
            {/* Track */}
            <div className="absolute inset-0 rounded" style={{ background: "#0f172a" }} />
            {/* Grid lines */}
            {Array.from({ length: maxDay }).map((_, d) => (
              <div
                key={d}
                className="absolute top-0 bottom-0 w-px"
                style={{ left: `${(d / maxDay) * 100}%`, background: "rgba(51,65,85,0.3)" }}
              />
            ))}
            {/* Task bar */}
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: `${(task.duration_days / maxDay) * 100}%`, opacity: 0.9 }}
              transition={{ delay: i * 0.05 + 0.25, duration: 0.5, ease: "easeOut" }}
              className="absolute top-1 bottom-1 rounded overflow-hidden"
              style={{ left: `${(task.start_day / maxDay) * 100}%`, background: COLORS[i % COLORS.length] }}
            >
              <div className="absolute inset-0 flex items-center px-1.5 overflow-hidden">
                <span style={{ fontSize: 8.5, color: "rgba(255,255,255,0.9)", fontFamily: "'DM Mono', monospace", whiteSpace: "nowrap" }}>
                  {task.contractor_role}
                </span>
              </div>
            </motion.div>
            {/* Critical path indicator */}
            {task.is_critical && (
              <div
                className="absolute top-0 left-0 w-1 rounded-l"
                style={{
                  left: `${(task.start_day / maxDay) * 100}%`,
                  height: "100%",
                  background: "#ef4444",
                  width: 2,
                }}
              />
            )}
          </div>
          <div style={{ width: 24, fontSize: 9, color: "#475569", fontFamily: "'DM Mono', monospace", flexShrink: 0 }}>
            {task.duration_days}d
          </div>
        </motion.div>
      ))}
      {/* Day labels */}
      <div className="flex items-center gap-3">
        <div style={{ width: 190 }} />
        <div className="flex-1 flex justify-between" style={{ paddingTop: 4 }}>
          {[0, Math.round(maxDay * 0.25), Math.round(maxDay * 0.5), Math.round(maxDay * 0.75), maxDay].map((d) => (
            <span key={d} style={{ fontSize: 9, color: "#334155", fontFamily: "'DM Mono', monospace" }}>
              D{d}
            </span>
          ))}
        </div>
        <div style={{ width: 24 }} />
      </div>
    </div>
  );
}

/**
 * ARKEN — ROI Visualization Card
 */
import { formatInr } from "@/hooks/useApi";
import { ROIResult } from "@/store/arken";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface ROICardProps {
  roi: ROIResult;
  budgetInr: number;
}

export function ROICard({ roi, budgetInr }: ROICardProps) {
  const chartData = [
    { label: "Pre-Reno", value: roi.pre_reno_value_inr },
    { label: "After Reno", value: roi.post_reno_value_inr },
    { label: "+1 Year", value: roi.post_reno_value_inr * 1.085 },
    { label: "+2 Years", value: roi.post_reno_value_inr * 1.175 },
    { label: "+3 Years", value: roi.post_reno_value_inr * 1.27 },
  ];

  return (
    <div className="rounded-xl p-5" style={{ background: "linear-gradient(135deg, #0a1a14, #071310)", border: "1px solid #064e3b" }}>
      <div style={{ fontSize: 10, color: "#475569", fontFamily: "'DM Mono', monospace", marginBottom: 12, letterSpacing: "0.1em" }}>
        ROI FORECAST · {roi.city.toUpperCase()} · TIER {roi.city_tier}
      </div>

      <div className="flex items-end gap-3 mb-4">
        <span style={{ fontSize: 40, fontWeight: 700, color: "#10b981", lineHeight: 1 }}>
          +{roi.roi_pct.toFixed(1)}%
        </span>
        <span style={{ fontSize: 11, color: "#059669", paddingBottom: 6 }}>property value gain</span>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          { label: "Investment", value: formatInr(budgetInr), color: "#94a3b8" },
          { label: "Value Added", value: `+${formatInr(roi.equity_gain_inr)}`, color: "#34d399" },
          { label: "Payback", value: `${roi.payback_months}mo`, color: "#f59e0b" },
        ].map((item) => (
          <div key={item.label} className="p-2.5 rounded-lg" style={{ background: "#0f1f16", border: "1px solid #064e3b" }}>
            <div style={{ fontSize: 9, color: "#475569", fontFamily: "'DM Mono', monospace", marginBottom: 3 }}>{item.label}</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: item.color }}>{item.value}</div>
          </div>
        ))}
      </div>

      {/* Area chart */}
      <div style={{ height: 80 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="roiGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="label" tick={{ fontSize: 8, fill: "#475569", fontFamily: "'DM Mono', monospace" }} axisLine={false} tickLine={false} />
            <YAxis hide />
            <Tooltip
              formatter={(v: number) => [formatInr(v), "Property Value"]}
              contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 10 }}
            />
            <Area type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} fill="url(#roiGrad)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between mt-3 pt-3" style={{ borderTop: "1px solid #064e3b" }}>
        <span style={{ fontSize: 10, color: "#475569", fontFamily: "'DM Mono', monospace" }}>Rental Yield Δ</span>
        <span style={{ fontSize: 11, color: "#34d399", fontFamily: "'DM Mono', monospace" }}>+{roi.rental_yield_delta}%</span>
        <span style={{ fontSize: 9, color: "#334155", fontFamily: "'DM Mono', monospace" }}>
          Model confidence: {Math.round(roi.model_confidence * 100)}%
        </span>
      </div>
    </div>
  );
}
