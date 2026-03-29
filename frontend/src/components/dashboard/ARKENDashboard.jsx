"use client";
/**
 * ARKEN PropTech Engine v5.0 — Refactored Dashboard
 * All API calls, URLs, payload shapes, and business logic are UNCHANGED.
 * New: component split, Vastu tab, Feedback panel, ModelHealthBadge in nav,
 *      error card with retry, mobile slider, persistent chat history,
 *      PDF loading spinner, Share Results, Escape closes chat.
 */

import { useState, useRef, useCallback, useEffect, memo, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ModelHealthBadge from "./ModelHealthBadge";
import FeedbackPanel    from "./FeedbackPanel";
import VastuPanel       from "./VastuPanel";
import BOQTable         from "./BOQTable";
import ROIPanel         from "./ROIPanel";

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

const CITIES  = ["Hyderabad","Bangalore","Mumbai","Delhi NCR","Pune","Chennai"];
const THEMES  = ["Modern Minimalist","Scandinavian","Japandi","Industrial Chic","Tropical Luxe","Art Deco","Neo-Classical","Bohemian"];
const ROOMS   = ["Bedroom","Living Room","Kitchen","Bathroom","Dining Room","Study / Home Office"];
const BUDGETS = [
  { label:"Basic",     inr:400000,  tier:"basic",   color:C.muted },
  { label:"Mid",     inr:750000,  tier:"mid",     color:C.amber },
  { label:"Premium", inr:1500000, tier:"premium", color:C.green },
];
const ROOM_MAP = {
  "Bedroom":"bedroom","Living Room":"living_room","Kitchen":"kitchen",
  "Bathroom":"bathroom","Dining Room":"dining_room","Study / Home Office":"study",
};
const AGENT_META = [
  { id:"image_analysis",    label:"Visual Assessor",     icon:"👁",  desc:"Gemini Vision — walls, floor, ceiling, materials, style, condition score" },
  { id:"planning",          label:"Design Planner",      icon:"📐", desc:"SKU-based BOQ from Indian brand catalog + CPM construction schedule" },
  { id:"roi_prediction",    label:"ROI Engine",          icon:"📈", desc:"XGBoost + SHAP trained on 32,963 Indian property transactions" },
  { id:"budget_location",   label:"Market Intelligence", icon:"🏙", desc:"City-adjusted pricing + Prophet 90-day material price forecasts" },
  { id:"insight_generation",label:"Insight Synthesiser", icon:"✨", desc:"LangGraph orchestration → renovation sequence, priority repairs, DIY tips" },
];
const QUICK = [
  "What materials did you use?",
  "How much would this cost in total?",
  "Make the walls darker",
  "What changed from the original?",
  "Suggest cheaper flooring alternatives",
  "What's the rental yield improvement?",
];

function fmtInr(n) {
  if (!n && n !== 0) return "—";
  const v = typeof n === "string" ? parseFloat(n.replace(/[₹,LABCRKrlab ]/g,"")) : n;
  if (isNaN(v)) return String(n);
  if (v >= 10_000_000) return `₹${(v/10_000_000).toFixed(1)}Cr`;
  if (v >= 100_000)    return `₹${(v/100_000).toFixed(1)}L`;
  if (v >= 1_000)      return `₹${(v/1_000).toFixed(0)}K`;
  return `₹${Math.round(v)}`;
}
function b64(file) {
  return new Promise((res,rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(",")[1]);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}
async function api(path, opts={}) {
  const r = await fetch(`${API}${path}`, opts);
  if (!r.ok) { const e = await r.json().catch(()=>({detail:"Network error"})); throw new Error(e.detail||`HTTP ${r.status}`); }
  return r.json();
}

function Chip({ children, color=C.indigo }) {
  return <span style={{ display:"inline-block", padding:"2px 8px", borderRadius:4,
    background:`${color}20`, color, fontSize:10, ...F.mono, border:`1px solid ${color}40` }}>{children}</span>;
}
function Stat({ label, value, sub, accent=C.indigo }) {
  return (
    <div style={{ background:C.card, borderRadius:10, padding:"14px 16px",
      border:`1px solid ${C.border}`, display:"flex", flexDirection:"column", gap:4 }}>
      <div style={{ fontSize:9, color:C.muted, ...F.mono, letterSpacing:"0.08em", textTransform:"uppercase" }}>{label}</div>
      <div style={{ fontSize:22, fontWeight:700, color:accent, letterSpacing:"-0.02em" }}>{value||"—"}</div>
      {sub && <div style={{ fontSize:10, color:C.muted }}>{sub}</div>}
    </div>
  );
}
function Empty({ msg }) {
  return <div style={{ padding:"40px 24px", textAlign:"center", color:C.muted, fontSize:12, ...F.mono }}>{msg}</div>;
}
function SectionTitle({ children }) {
  return <div style={{ fontSize:9, color:C.muted, ...F.mono, letterSpacing:"0.1em", marginBottom:12 }}>{children}</div>;
}

function DataSourceBadge({ modelType, confidence, dataSource, dataTransparency }) {
  const [show, setShow] = useState(false);
  const isML = modelType && (modelType.includes("ensemble")||modelType.includes("xgboost")||modelType.includes("real"));
  const confPct   = confidence ? Math.round(confidence*100) : null;
  const confLevel = confidence >= 0.80 ? "High" : confidence >= 0.60 ? "Medium" : "Low";
  const confColor = confidence >= 0.80 ? C.green : confidence >= 0.60 ? C.amber : C.red;
  const tooltipText = dataTransparency||dataSource||"Based on 32,963 real Indian property transactions";
  return (
    <div style={{ display:"inline-flex", alignItems:"center", gap:6, position:"relative" }}>
      <span style={{ fontSize:9, ...F.mono, color:C.muted }}>{isML?"🤖":"📊"} {isML?"ML Model":"Heuristic"}</span>
      {confPct && (
        <span style={{ fontSize:9, ...F.mono, padding:"1px 6px", borderRadius:3,
          background:`${confColor}20`, color:confColor, border:`1px solid ${confColor}40` }}>
          {confLevel} {confPct}%
        </span>
      )}
      <span onMouseEnter={()=>setShow(true)} onMouseLeave={()=>setShow(false)}
        style={{ fontSize:10, color:C.muted, cursor:"help", userSelect:"none" }}>ⓘ</span>
      {show && (
        <div style={{ position:"absolute", bottom:"calc(100% + 6px)", left:0, zIndex:999,
          background:C.card, border:`1px solid ${C.borderHi}`, borderRadius:8,
          padding:"10px 14px", minWidth:260, fontSize:10, color:C.muted, lineHeight:1.6,
          boxShadow:"0 8px 24px rgba(0,0,0,0.5)" }}>{tooltipText}</div>
      )}
    </div>
  );
}
function DisclaimerFooter({ dataSource }) {
  return (
    <div style={{ marginTop:10, padding:"10px 14px", background:C.surface,
      border:`1px solid ${C.border}`, borderRadius:8,
      fontSize:9, color:C.dim, lineHeight:1.7, ...F.mono }}>
      ⚠ Financial projections are estimates based on historical Indian property data.
      Past appreciation trends do not guarantee future returns.
      Consult a property valuer for formal assessment.
      {dataSource && <span style={{ color:C.muted }}> · Source: {dataSource}</span>}
    </div>
  );
}
function ScheduleDateBadges({ schedule }) {
  if (!schedule) return null;
  const best=schedule.best_case_end||schedule.projected_end;
  const expected=schedule.realistic_end_date||best;
  const worst=schedule.worst_case_end;
  if (!best) return null;
  return (
    <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:10 }}>
      {best&&<div style={{ padding:"5px 12px", borderRadius:6, background:C.greenLo, border:`1px solid ${C.green}40`, fontSize:10, ...F.mono }}>
        <span style={{ color:C.muted }}>Best case: </span><span style={{ color:C.green }}>{best}</span></div>}
      {expected&&expected!==best&&<div style={{ padding:"5px 12px", borderRadius:6, background:C.amberLo, border:`1px solid ${C.amber}40`, fontSize:10, ...F.mono }}>
        <span style={{ color:C.muted }}>Expected: </span><span style={{ color:C.amber }}>{expected}</span></div>}
      {worst&&<div style={{ padding:"5px 12px", borderRadius:6, background:C.redLo, border:`1px solid ${C.red}40`, fontSize:10, ...F.mono }}>
        <span style={{ color:C.muted }}>Worst case (monsoon/delays): </span><span style={{ color:C.red }}>{worst}</span></div>}
    </div>
  );
}
function DataFreshnessBadge() {
  return (
    <div style={{ display:"flex", gap:10, alignItems:"center", flexWrap:"wrap" }}>
      <span style={{ fontSize:9, color:C.muted, ...F.mono, padding:"2px 8px",
        background:C.surface, border:`1px solid ${C.border}`, borderRadius:4 }}>
        📅 Material prices verified: Q1 2026
      </span>
      <span style={{ fontSize:9, color:C.muted, ...F.mono, padding:"2px 8px",
        background:C.surface, border:`1px solid ${C.border}`, borderRadius:4 }}>
        🏠 Property data: 2024 dataset (32,963 transactions)
      </span>
    </div>
  );
}

const BASlider = memo(function BASlider({ before, after }) {
  const [pos,setPos] = useState(50);
  const ref  = useRef(null);
  const drag = useRef(false);
  const [isMobile, setIsMobile] = useState(false);
  useEffect(()=>{ setIsMobile(window.innerWidth < 768); },[]);
  const move = useCallback((cx) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((cx-rect.left)/rect.width)*100)));
  },[]);
  return (
    <div ref={ref}
      onMouseDown={()=>{drag.current=true;}}
      onMouseUp={()=>{drag.current=false;}}
      onMouseLeave={()=>{drag.current=false;}}
      onMouseMove={e=>drag.current&&move(e.clientX)}
      onTouchMove={e=>{e.preventDefault();move(e.touches[0].clientX);}}
      style={{
        position:"relative",
        width: isMobile ? "100vw" : "100%",
        marginLeft: isMobile ? "calc(-50vw + 50%)" : 0,
        aspectRatio:"16/9", borderRadius: isMobile ? 0 : 14,
        overflow:"hidden", cursor:"ew-resize", background:C.bg,
        userSelect:"none", border:`1px solid ${C.border}`,
      }}>
      <div style={{ position:"absolute", inset:0 }}>
        {after
          ? <img src={after} alt="after" style={{ width:"100%", height:"100%", objectFit:"cover" }} />
          : <div style={{ width:"100%", height:"100%", display:"flex", alignItems:"center",
              justifyContent:"center", color:C.dim, ...F.mono, fontSize:11 }}>RENDER WILL APPEAR HERE</div>}
        <div style={{ position:"absolute", top:10, right:10, ...F.mono, fontSize:9,
          background:"rgba(99,102,241,0.25)", color:C.purple, padding:"3px 10px", borderRadius:5,
          border:`1px solid ${C.indigo}60` }}>AFTER</div>
      </div>
      <div style={{ position:"absolute", inset:0, overflow:"hidden", width:`${pos}%` }}>
        <div style={{ position:"absolute", inset:0, width:`${100/Math.max(pos,0.1)*100}%` }}>
          {before
            ? <img src={before} alt="before" style={{ width:"100%", height:"100%", objectFit:"cover" }} />
            : <div style={{ width:"100%", height:"100%", background:C.surface, display:"flex",
                alignItems:"center", justifyContent:"center", color:C.muted, ...F.mono }}>BEFORE</div>}
          <div style={{ position:"absolute", top:10, left:10, ...F.mono, fontSize:9,
            background:"rgba(0,0,0,0.75)", color:C.muted, padding:"3px 10px", borderRadius:5 }}>BEFORE</div>
        </div>
      </div>
      <div style={{ position:"absolute", top:0, bottom:0, left:`${pos}%`, width:2,
        background:C.purple, transform:"translateX(-50%)",
        display:"flex", alignItems:"center", justifyContent:"center" }}>
        <div style={{ width:34, height:34, borderRadius:"50%", background:C.indigo,
          border:`2px solid ${C.purple}`, display:"flex", alignItems:"center",
          justifyContent:"center", color:"#fff", fontSize:14, flexShrink:0 }}>⇔</div>
      </div>
    </div>
  );
});

const AgentPipeline = memo(function AgentPipeline({ statuses={}, timings={} }) {
  return (
    <div style={{ background:C.surface, borderRadius:12, border:`1px solid ${C.border}`,
      padding:"14px 18px", marginBottom:18 }}>
      <SectionTitle>LANGGRAPH AGENT PIPELINE</SectionTitle>
      <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
        {AGENT_META.map((a) => {
          const st=statuses[a.id]||"idle";
          const t=timings[a.id];
          const col=st==="complete"?C.green:st==="running"?C.amber:st==="error"?C.red:C.dim;
          return (
            <div key={a.id} style={{ display:"flex", alignItems:"center", gap:10 }}>
              <div style={{ width:28, height:28, borderRadius:7, background:`${col}18`,
                border:`1px solid ${col}50`, display:"flex", alignItems:"center",
                justifyContent:"center", fontSize:13, flexShrink:0 }}>{a.icon}</div>
              <div style={{ flex:1, minWidth:0 }}>
                <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                  <span style={{ fontSize:11, fontWeight:600, color:st==="idle"?C.dim:C.text }}>{a.label}</span>
                  {st==="running"&&<motion.span animate={{opacity:[0.4,1,0.4]}} transition={{repeat:Infinity,duration:1}}
                    style={{ fontSize:9, color:C.amber, ...F.mono }}>● running</motion.span>}
                  {st==="complete"&&<span style={{ fontSize:9, color:C.green, ...F.mono }}>✓ {t?`${t.toFixed(1)}s`:"done"}</span>}
                  {st==="error"&&<span style={{ fontSize:9, color:C.red, ...F.mono }}>✗ failed</span>}
                </div>
                <div style={{ fontSize:9, color:C.muted, marginTop:1 }}>{a.desc}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
});

const ROICard = memo(function ROICard({ roi, city, room, budgetTier, budgetInr }) {
  return <ROIPanel
    roiResult={roi}
    city={roi?.city||city}
    roomType={roi?.room_type||room}
    budgetTier={roi?.budget_tier||budgetTier}
    budgetInr={budgetInr}
  />;
});

const BOQCard = memo(function BOQCard({ design, city, budgetTier }) {
  if (!design?.total_inr && !design?.line_items?.length)
    return <Empty msg="BOQ will appear after pipeline analysis runs (click Refresh Insights)" />;
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
      <div style={{ display:"grid", gridTemplateColumns:`repeat(${design.products_subtotal_inr?5:4},1fr)`, gap:10 }}>
        <Stat label="Materials"   value={fmtInr(design.material_inr)}    accent={C.indigo} />
        <Stat label="Labour"      value={fmtInr(design.labour_inr)}      accent={C.amber} />
        <Stat label="GST @18%"    value={fmtInr(design.gst_inr)}         accent={C.muted} />
        <Stat label="Contingency" value={fmtInr(design.contingency_inr)} accent={C.red} />
        {design.products_subtotal_inr>0&&
          <Stat label="Products" value={fmtInr(design.products_subtotal_inr)} accent={C.purple} />}
      </div>
      <BOQTable boqItems={design.line_items||[]} totalCost={design.total_inr} city={city} budgetTier={budgetTier} labourInr={design.labour_inr||0} />
    </div>
  );
});

const ScheduleCard = memo(function ScheduleCard({ schedule }) {
  const [expandTask, setExpandTask] = useState(null);
  if (!schedule?.tasks?.length)
    return <Empty msg="CPM schedule will appear after pipeline analysis runs (click Refresh Insights)" />;
  const tasks=schedule.tasks||[];
  const total=schedule.total_days||18;
  const risks=schedule.risks||[];
  const dependencies=schedule.dependencies||[];
  const start=schedule.start_date?new Date(schedule.start_date):new Date(Date.now()+7*864e5);
  const fmt=d=>new Date(start.getTime()+d*864e5).toLocaleDateString("en-IN",{day:"numeric",month:"short"});
  const criticalTasks=tasks.filter(t=>t.is_critical);
  const parallelGroups=tasks.filter(t=>t.can_parallel||t.parallel_with?.length>0);
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
      <ScheduleDateBadges schedule={schedule} />

      {/* Summary stats */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:10 }}>
        <Stat label="Total Duration" value={`${total} days`} sub="from site handover" accent={C.amber} />
        <Stat label="Critical Path" value={`${schedule.critical_path_days||total} days`} sub={`${criticalTasks.length} critical tasks`} accent={C.red} />
        <Stat label="Start Date" value={fmt(0)} sub={`End: ${fmt(total)}`} accent={C.green} />
        <Stat label="Parallel Phases" value={parallelGroups.length||"—"} sub="can run simultaneously" accent={C.indigo} />
      </div>

      {/* Schedule confidence + risk buffer */}
      <div style={{ background:C.surface, borderRadius:8, padding:"10px 14px", border:`1px solid ${C.border}` }}>
        {schedule.risk_buffer_days>0&&<div style={{ fontSize:10, color:C.muted, marginBottom:6 }}>
          <span style={{ color:C.amber, fontWeight:600 }}>+{schedule.risk_buffer_days} day risk buffer</span>
          {" "}({schedule.budget_tier||"mid"} tier) already included in total
        </div>}
        <div style={{ fontSize:10, color:C.muted, lineHeight:1.65 }}>
          {schedule.schedule_confidence||"Assumes normal contractor availability in your city. Monsoon season (Jun–Sep) adds 20–30% to timeline."}
        </div>
        {schedule.best_buying_window&&<div style={{ marginTop:6, fontSize:10, color:C.green }}>
          🛒 Best material buying window: {schedule.best_buying_window}
        </div>}
      </div>

      {/* Gantt chart with clickable task details */}
      <div style={{ background:C.card, borderRadius:10, border:`1px solid ${C.border}`, overflow:"hidden" }}>
        <div style={{ padding:"10px 16px", background:C.surface, borderBottom:`1px solid ${C.border}`,
          display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <div style={{ ...F.mono, fontSize:9, color:C.muted, letterSpacing:"0.08em" }}>CPM SCHEDULE — GANTT (click task for details)</div>
          <div style={{ display:"flex", gap:10 }}>
            <span style={{ fontSize:9, color:C.red, ...F.mono }}>■ critical</span>
            <span style={{ fontSize:9, color:C.indigo, ...F.mono }}>■ standard</span>
            <span style={{ fontSize:9, color:C.green, ...F.mono }}>■ parallel</span>
          </div>
        </div>
        <div style={{ padding:16 }}>
          {tasks.map((t,i)=>{
            const pct=(t.start_day/total)*100;
            const wPct=((t.end_day-t.start_day)/total)*100;
            const isExpanded=expandTask===i;
            const barColor=t.is_critical?C.red:(t.can_parallel||t.parallel_with?.length>0)?C.green:C.indigo;
            return (
              <div key={t.id||i} style={{ marginBottom:6 }}>
                <div onClick={()=>setExpandTask(isExpanded?null:i)}
                  style={{ display:"grid", gridTemplateColumns:"190px 1fr 80px", gap:8, alignItems:"center",
                    cursor:"pointer", padding:"4px 4px", borderRadius:6,
                    background:isExpanded?C.surface:"transparent" }}>
                  <div>
                    <div style={{ fontSize:10, color:t.is_critical?C.red:C.text, fontWeight:t.is_critical?600:400, display:"flex", gap:5, alignItems:"center" }}>
                      {t.name}
                      {t.is_critical&&<span style={{ fontSize:8, background:`${C.red}20`, color:C.red, padding:"1px 4px", borderRadius:3, ...F.mono }}>CRITICAL</span>}
                      {(t.can_parallel||t.parallel_with?.length>0)&&<span style={{ fontSize:8, background:`${C.green}20`, color:C.green, padding:"1px 4px", borderRadius:3, ...F.mono }}>PARALLEL</span>}
                    </div>
                    <div style={{ fontSize:9, color:C.muted }}>{t.contractor_role}</div>
                  </div>
                  <div style={{ height:20, background:C.border, borderRadius:4, position:"relative" }}>
                    <div style={{ position:"absolute", left:`${pct}%`, width:`${Math.max(wPct,2)}%`,
                      height:"100%", borderRadius:4, background:barColor, opacity:0.85 }} />
                    {t.float_days>0&&<div style={{ position:"absolute", left:`${pct+wPct}%`, width:`${(t.float_days/total)*100}%`,
                      height:"100%", borderRadius:4, background:`${C.muted}40` }} />}
                  </div>
                  <div style={{ fontSize:9, color:C.muted, ...F.mono, textAlign:"right" }}>
                    D{t.start_day}–D{t.end_day}
                    {t.float_days>0&&<span style={{ color:C.muted, opacity:0.6 }}> +{t.float_days}f</span>}
                  </div>
                </div>
                {isExpanded&&<div style={{ margin:"6px 4px 8px", padding:"10px 14px", background:C.surface,
                  borderRadius:8, border:`1px solid ${C.border}`, fontSize:10, color:C.muted, lineHeight:1.75 }}>
                  {t.description&&<div style={{ color:C.text, marginBottom:6 }}>{t.description}</div>}
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                    {t.duration_days&&<div><span style={{ color:C.muted }}>Duration: </span><span style={{ color:C.text }}>{t.duration_days} days</span></div>}
                    {t.estimated_cost_inr&&<div><span style={{ color:C.muted }}>Est. cost: </span><span style={{ color:C.amber, fontWeight:600 }}>{fmtInr(t.estimated_cost_inr)}</span></div>}
                    {t.contractor_role&&<div><span style={{ color:C.muted }}>Contractor: </span><span style={{ color:C.text }}>{t.contractor_role}</span></div>}
                    {t.float_days>0&&<div><span style={{ color:C.muted }}>Float: </span><span style={{ color:C.green }}>{t.float_days} days flexibility</span></div>}
                  </div>
                  {t.materials?.length>0&&<div style={{ marginTop:6 }}>
                    <span style={{ color:C.muted }}>Materials: </span>
                    <span style={{ color:C.text }}>{t.materials.join(", ")}</span>
                  </div>}
                  {t.predecessors?.length>0&&<div style={{ marginTop:4 }}>
                    <span style={{ color:C.muted }}>Depends on: </span>
                    <span style={{ color:C.text }}>{t.predecessors.join(", ")}</span>
                  </div>}
                  {t.parallel_with?.length>0&&<div style={{ marginTop:4 }}>
                    <span style={{ color:C.green }}>Can run parallel with: </span>
                    <span style={{ color:C.text }}>{t.parallel_with.join(", ")}</span>
                  </div>}
                  {t.quality_check&&<div style={{ marginTop:4, color:C.indigo }}>✓ Quality checkpoint: {t.quality_check}</div>}
                  {t.indian_context&&<div style={{ marginTop:4, color:C.amber }}>🇮🇳 {t.indian_context}</div>}
                </div>}
              </div>
            );
          })}
        </div>
      </div>

      {/* Phase summary if available */}
      {schedule.phases?.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>PHASE BREAKDOWN</SectionTitle>
        {schedule.phases.map((ph,i)=>(
          <div key={i} style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start",
            padding:"8px 0", borderBottom:i<schedule.phases.length-1?`1px solid ${C.border}`:"none" }}>
            <div>
              <div style={{ fontSize:11, fontWeight:600, color:C.text }}>{ph.name||ph.phase}</div>
              {ph.description&&<div style={{ fontSize:10, color:C.muted, marginTop:2, lineHeight:1.6 }}>{ph.description}</div>}
              {ph.tasks&&<div style={{ fontSize:9, color:C.indigo, marginTop:2, ...F.mono }}>{ph.tasks.join(" · ")}</div>}
            </div>
            <div style={{ textAlign:"right", flexShrink:0, marginLeft:16 }}>
              {ph.duration_days&&<div style={{ fontSize:11, fontWeight:600, color:C.amber }}>{ph.duration_days}d</div>}
              {ph.cost_inr&&<div style={{ fontSize:9, color:C.muted, ...F.mono }}>{fmtInr(ph.cost_inr)}</div>}
            </div>
          </div>
        ))}
      </div>}

      {/* Cost distribution across tasks */}
      {tasks.some(t=>t.estimated_cost_inr)&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>💰 COST DISTRIBUTION BY TASK</SectionTitle>
        {tasks.filter(t=>t.estimated_cost_inr).map((t,i,arr)=>{
          const maxCost=Math.max(...arr.map(x=>x.estimated_cost_inr));
          const pct=(t.estimated_cost_inr/maxCost)*100;
          return (
            <div key={i} style={{ marginBottom:8 }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                <span style={{ fontSize:10, color:t.is_critical?C.red:C.text }}>{t.name}</span>
                <span style={{ fontSize:10, color:C.amber, ...F.mono, fontWeight:600 }}>{fmtInr(t.estimated_cost_inr)}</span>
              </div>
              <div style={{ height:5, background:C.border, borderRadius:3 }}>
                <div style={{ height:"100%", width:`${pct}%`, background:t.is_critical?C.red:C.indigo, borderRadius:3 }} />
              </div>
            </div>
          );
        })}
      </div>}

      {/* Risk register */}
      {risks.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>⚠ RISK REGISTER</SectionTitle>
        {risks.map((r,i)=>(
          <div key={i} style={{ padding:"9px 0", borderBottom:i<risks.length-1?`1px solid ${C.border}`:"none" }}>
            <div style={{ display:"flex", gap:8, alignItems:"center", marginBottom:4 }}>
              <Chip color={r.probability==="High"?C.red:r.probability==="Medium"?C.amber:C.muted}>{r.probability}</Chip>
              <span style={{ fontSize:11, fontWeight:600, color:C.text }}>{r.factor}</span>
            </div>
            {r.detail&&<div style={{ fontSize:10, color:C.muted, marginBottom:3, lineHeight:1.6 }}>{r.detail}</div>}
            {r.mitigation&&<div style={{ fontSize:10, color:C.green }}>✓ Mitigation: {r.mitigation}</div>}
            {r.cost_impact_inr&&<div style={{ fontSize:10, color:C.amber, marginTop:3 }}>
              Cost impact if triggered: {fmtInr(r.cost_impact_inr)}
            </div>}
          </div>
        ))}
      </div>}

      {/* Dependencies map */}
      {dependencies.length>0&&<div style={{ background:C.surface, borderRadius:8, padding:"10px 14px", border:`1px solid ${C.border}` }}>
        <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:8, letterSpacing:"0.08em" }}>TASK DEPENDENCIES</div>
        {dependencies.map((d,i)=>(
          <div key={i} style={{ fontSize:10, color:C.muted, padding:"3px 0" }}>
            <span style={{ color:C.text }}>{d.task}</span>
            <span style={{ color:C.muted }}> must finish before </span>
            <span style={{ color:C.indigo }}>{d.depends_on||d.successor}</span>
          </div>
        ))}
      </div>}
    </div>
  );
});

const InsightsCard = memo(function InsightsCard({ insights, loading, timings={}, boqTotal=0 }) {
  if (loading) return (
    <div style={{ padding:40, textAlign:"center" }}>
      <motion.div animate={{opacity:[0.3,1,0.3]}} transition={{repeat:Infinity,duration:1.2}}
        style={{ color:C.amber, fontSize:13, ...F.mono }}>
        ◈ Pipeline running — 10-agent LangGraph processing your renovation...
      </motion.div>
      <div style={{ fontSize:10, color:C.muted, marginTop:8 }}>Vision → RAG → BOQ → ROI → Market → Insights → Report (≈20–40s)</div>
    </div>
  );
  if (!insights||!Object.keys(insights).length)
    return <Empty msg="Pipeline insights not yet generated — render your room first, then click Refresh Insights" />;

  const vis   = insights.visual_analysis||{};
  const fin   = insights.financial_outlook||{};
  const mkt   = insights.market_intelligence||{};
  const bud   = insights.budget_assessment||{};
  const recs  = insights.recommendations||[];
  const risks = insights.risk_factors||[];
  const mats  = insights.top_materials||[];
  const dq    = insights.data_quality||{};
  const diy   = insights.diy_renovation_tips||[];
  const actionChecklist = insights.action_checklist||[];
  const marketTiming = insights.market_timing||{};
  const renovSeq = insights.renovation_sequence||[];
  const priorityRepairs = insights.priority_repairs||[];

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

      {/* Data quality / grounding badge */}
      {dq.confidence_tier&&<div style={{ display:"flex", gap:8, alignItems:"center", flexWrap:"wrap",
        padding:"8px 14px", background:C.surface, borderRadius:8, border:`1px solid ${C.border}` }}>
        <span style={{ fontSize:9, ...F.mono, padding:"2px 8px", borderRadius:4,
          background:dq.confidence_tier==="high"?C.greenLo:dq.confidence_tier==="medium"?C.amberLo:C.surface,
          color:dq.confidence_tier==="high"?C.green:dq.confidence_tier==="medium"?C.amber:C.muted,
          border:`1px solid ${dq.confidence_tier==="high"?C.green:dq.confidence_tier==="medium"?C.amber:C.muted}40` }}>
          {dq.confidence_tier==="high"?"● HIGH CONFIDENCE":dq.confidence_tier==="medium"?"● MEDIUM CONFIDENCE":"● ESTIMATE ONLY"}
        </span>
        {insights.image_grounded&&<Chip color={C.indigo}>📷 Image-grounded</Chip>}
        {insights.rag_grounded&&<Chip color={C.purple}>📚 RAG-enriched</Chip>}
        {insights.dataset_grounded&&<Chip color={C.green}>📊 Dataset-grounded</Chip>}
        {dq.user_message&&<span style={{ fontSize:10, color:C.muted, flex:1 }}>{dq.user_message}</span>}
      </div>}

      {/* Summary headline — cost patched to match BOQ total (same as BOQ tab) */}
      {insights.summary_headline&&<div style={{ background:`linear-gradient(135deg,${C.indigoLo},${C.greenLo})`,
        borderRadius:12, padding:16, border:`1px solid ${C.indigo}40` }}>
        <SectionTitle>AI PIPELINE SYNTHESIS</SectionTitle>
        <div style={{ fontSize:14, color:C.text, lineHeight:1.65, fontWeight:500 }}>
          {boqTotal
            ? insights.summary_headline.replace(
                /₹[\d,]+(\s+total investment)/,
                `₹${boqTotal.toLocaleString("en-IN")}$1`
              )
            : insights.summary_headline}
        </div>
      </div>}

      {/* Visual analysis — full CV output */}
      <div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>👁 ROOM ANALYSIS — WHAT WAS DETECTED</SectionTitle>

        {/* Core room fields grid */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8, marginBottom:12 }}>
          {[
            {l:"Room Type",       v:vis.room_type},
            {l:"Style Detected",  v:vis.style_detected,    c:C.indigo},
            {l:"Natural Light",   v:vis.natural_light},
            {l:"Renovation Scope",v:vis.renovation_scope==="not_assessed"?null:vis.renovation_scope, c:C.amber},
            {l:"Room Condition",  v:vis.room_condition==="not_assessed"?null:vis.room_condition},
            {l:"CV Model",        v:vis.cv_model},
          ].filter(i=>i.v).map(item=>(
            <div key={item.l} style={{ padding:"8px 10px", background:C.surface, borderRadius:8, border:`1px solid ${C.border}` }}>
              <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:3 }}>{item.l}</div>
              <div style={{ fontSize:11, color:item.c||C.text, fontWeight:item.c?600:400 }}>{item.v}</div>
            </div>
          ))}
        </div>

        {/* Condition scores — only if vision ran */}
        {vis.vision_assessed&&<div style={{ marginBottom:12 }}>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:8 }}>CONDITION ASSESSMENT</div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8 }}>
            {[
              {l:"Condition Score",  v:vis.condition_score!=null?`${vis.condition_score}/10`:null, c:vis.condition_score>=7?C.green:vis.condition_score>=4?C.amber:C.red},
              {l:"Wall Condition",   v:vis.wall_condition!=="not_assessed"?vis.wall_condition:null},
              {l:"Floor Condition",  v:vis.floor_condition!=="not_assessed"?vis.floor_condition:null},
            ].filter(i=>i.v).map(item=>(
              <div key={item.l} style={{ padding:"8px 10px", background:C.surface, borderRadius:8, border:`1px solid ${C.border}` }}>
                <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:3 }}>{item.l}</div>
                <div style={{ fontSize:12, color:item.c||C.text, fontWeight:600 }}>{item.v}</div>
              </div>
            ))}
          </div>
        </div>}

        {/* High value upgrades */}
        {vis.high_value_upgrades?.length>0&&<div style={{ marginBottom:12 }}>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:6 }}>HIGH-VALUE UPGRADES IDENTIFIED</div>
          <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
            {vis.high_value_upgrades.map((u,i)=><Chip key={i} color={C.green}>{u}</Chip>)}
          </div>
        </div>}

        {/* Issues detected */}
        {vis.issues_detected?.length>0&&<div style={{ marginBottom:12 }}>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:6 }}>ISSUES DETECTED</div>
          {vis.issues_detected.map((issue,i)=>(
            <div key={i} style={{ fontSize:10, color:C.text, padding:"4px 0", display:"flex", gap:6 }}>
              <span style={{ color:C.amber }}>⚠</span>{issue}
            </div>
          ))}
        </div>}

        {/* Detected objects */}
        {vis.detected_objects?.length>0&&<div style={{ marginBottom:12 }}>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:6 }}>OBJECTS / FURNITURE DETECTED (YOLO)</div>
          <div style={{ display:"flex", gap:5, flexWrap:"wrap" }}>
            {vis.detected_objects.map((obj,i)=><Chip key={i} color={C.muted}>{obj}</Chip>)}
          </div>
        </div>}

        {/* Detected materials */}
        {vis.detected_materials?.length>0&&<div style={{ marginBottom:12 }}>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:6 }}>MATERIALS DETECTED</div>
          <div style={{ display:"flex", gap:5, flexWrap:"wrap" }}>
            {vis.detected_materials.map((mat,i)=><Chip key={i} color={C.indigo}>{mat}</Chip>)}
          </div>
        </div>}

        {/* Style confidence */}
        {vis.style_confidence&&<div style={{ marginBottom:12 }}>
          <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:4 }}>
            <div style={{ fontSize:9, color:C.muted, ...F.mono }}>STYLE CONFIDENCE</div>
            <span title={`${Math.round((vis.style_confidence||0)*100)}% — how closely this room matches a single detected style. Rooms with mixed-use areas (e.g. bedroom + wet area) or multiple competing styles naturally score lower. This is an honest signal, not an error.`}
              style={{ fontSize:9, color:C.dim, cursor:"help", border:`1px solid ${C.border}`, borderRadius:"50%",
                width:13, height:13, display:"inline-flex", alignItems:"center", justifyContent:"center",
                lineHeight:1, flexShrink:0 }}>?</span>
          </div>
          <div style={{ height:6, background:C.border, borderRadius:3 }}>
            <div style={{ height:"100%", width:`${Math.round((vis.style_confidence||0)*100)}%`,
              background:vis.style_confidence>0.7?C.green:vis.style_confidence>0.4?C.amber:C.muted, borderRadius:3 }} />
          </div>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginTop:3 }}>{Math.round((vis.style_confidence||0)*100)}%</div>
        </div>}

        {/* Colour palette */}
        {vis.colour_palette?.length>0&&<div style={{ marginBottom:12 }}>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:6 }}>COLOUR PALETTE</div>
          <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
            {vis.colour_palette.map((c,i)=><Chip key={i}>{typeof c==="string"?c:(c?.name||c?.hex||JSON.stringify(c))}</Chip>)}
          </div>
        </div>}

        {/* Specific changes detected */}
        {vis.specific_changes_detected?.length>0&&<div>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:6 }}>SPECIFIC CHANGES FROM ORIGINAL</div>
          {vis.specific_changes_detected.map((c,i)=>(
            <div key={i} style={{ padding:"5px 0", borderBottom:i<vis.specific_changes_detected.length-1?`1px solid ${C.border}`:"none",
              fontSize:11, color:C.text, display:"flex", gap:8 }}>
              <span style={{ color:C.green, flexShrink:0 }}>→</span>{c}
            </div>
          ))}
        </div>}
      </div>

      {/* Priority repairs */}
      {priorityRepairs?.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        {/* Title: if all items are upgrades (good-condition room), say "Top Upgrades" */}
        {priorityRepairs.every(r=>r?.is_upgrade)
          ? <SectionTitle>⬆ TOP UPGRADES FOR YOUR ROOM</SectionTitle>
          : <SectionTitle>🔧 PRIORITY REPAIRS (DO FIRST)</SectionTitle>
        }
        {priorityRepairs.map((r,i)=>{
          // InsightGenerationAgent sends {issue, severity, how_to_fix, estimated_cost_inr,
          //   category, must_fix_first, is_upgrade}
          // InsightEngine (legacy) sends {action, urgency, estimated_cost_inr, category}
          const issue=typeof r==="string"?r:(r?.issue||r?.repair||r?.action||r?.title||"");
          const howToFix=typeof r==="object"?(r?.how_to_fix||r?.reasoning):null;
          const cost=typeof r==="object"?(r?.estimated_cost_inr||r?.cost_inr):null;
          const severity=typeof r==="object"?(r?.severity||r?.urgency||r?.priority):null;
          const category=typeof r==="object"?r?.category:null;
          const mustFixFirst=typeof r==="object"?r?.must_fix_first:false;
          const isUpgrade=typeof r==="object"?r?.is_upgrade:false;
          const sevColor=severity==="critical"||severity==="high"?C.red
            :severity==="medium"?C.amber:C.muted;
          return (
            <div key={i} style={{ padding:"10px 0", borderBottom:i<priorityRepairs.length-1?`1px solid ${C.border}`:"none",
              display:"flex", gap:8, alignItems:"flex-start" }}>
              <span style={{ color:isUpgrade?C.green:C.red, fontWeight:700, flexShrink:0, fontSize:12 }}>{i+1}.</span>
              <div style={{ flex:1 }}>
                <div style={{ fontSize:11, color:C.text, lineHeight:1.65, fontWeight:500 }}>{issue}</div>
                {howToFix&&<div style={{ fontSize:10, color:C.muted, lineHeight:1.6, marginTop:4 }}>
                  <span style={{ color:isUpgrade?C.green:C.amber }}>
                    {isUpgrade?"How to execute: ":"How to fix: "}
                  </span>{howToFix}
                </div>}
                <div style={{ display:"flex", gap:8, marginTop:6, flexWrap:"wrap", alignItems:"center" }}>
                  {severity&&<Chip color={sevColor}>{severity}</Chip>}
                  {category&&category!=="upgrade"&&<Chip color={C.indigo}>{category}</Chip>}
                  {mustFixFirst&&!isUpgrade&&<Chip color={C.red}>do first</Chip>}
                  {cost&&<span style={{ fontSize:9, color:C.amber, ...F.mono }}>Est: {fmtInr(cost)}</span>}
                </div>
              </div>
            </div>
          );
        })}
      </div>}

      {/* Renovation sequence */}
      {renovSeq?.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>📋 RECOMMENDED RENOVATION SEQUENCE</SectionTitle>
        <div style={{ fontSize:10, color:C.muted, marginBottom:10 }}>Follow this order to avoid rework and cost overruns</div>
        {renovSeq.map((step,i)=>{
          // InsightGenerationAgent sends {title, description, duration_days, contractor_type, cost_inr}
          // InsightEngine (fallback) sends {phase, actions:[], duration_days, cost_inr, step:int}
          // Resolve title: prefer "title", then "phase" (string), never fall through to integer "step"
          const rawTitle=typeof step==="string"?step:(step?.title||step?.phase||null);
          const title=rawTitle||`Step ${i+1}`;
          // Description: prefer "description", else join "actions" array if present
          const actionsArr=typeof step==="object"&&Array.isArray(step?.actions)?step.actions:[];
          const desc=typeof step==="object"
            ?(step?.description||step?.rationale||(actionsArr.length>0?actionsArr.slice(0,2).join(" · "):null))
            :null;
          const duration=typeof step==="object"?step?.duration_days:null;
          const contractor=typeof step==="object"?step?.contractor_type:null;
          const cost=typeof step==="object"?(step?.cost_inr||step?.estimated_cost_inr):null;
          const costPct=typeof step==="object"?step?.estimated_cost_pct:null;
          const dryingTime=typeof step==="object"?step?.requires_drying_time:false;
          return (
            <div key={i} style={{ display:"flex", gap:12, padding:"10px 0",
              borderBottom:i<renovSeq.length-1?`1px solid ${C.border}`:"none" }}>
              <div style={{ width:26, height:26, borderRadius:"50%", background:C.indigoLo,
                border:`1px solid ${C.indigo}60`, display:"flex", alignItems:"center", justifyContent:"center",
                flexShrink:0, fontSize:11, fontWeight:700, color:C.indigo, ...F.mono }}>{i+1}</div>
              <div style={{ flex:1 }}>
                <div style={{ fontSize:11, fontWeight:600, color:C.text, marginBottom:3 }}>{title}</div>
                {desc&&<div style={{ fontSize:10, color:C.muted, lineHeight:1.65, marginBottom:4 }}>{desc}</div>}
                <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginTop:3 }}>
                  {duration&&<span style={{ fontSize:9, color:C.amber, ...F.mono }}>⏱ {duration} day{duration>1?"s":""}</span>}
                  {contractor&&<span style={{ fontSize:9, color:C.indigo, ...F.mono }}>👷 {contractor}</span>}
                  {cost&&<span style={{ fontSize:9, color:C.green, ...F.mono }}>💰 {fmtInr(cost)}</span>}
                  {costPct&&!cost&&<span style={{ fontSize:9, color:C.muted, ...F.mono }}>~{costPct}% of budget</span>}
                  {dryingTime&&<span style={{ fontSize:9, color:C.amber, ...F.mono }}>⏳ drying time required</span>}
                </div>
              </div>
            </div>
          );
        })}
      </div>}

      {/* Financial outlook + Market intelligence */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
        {Object.keys(fin).length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
          <SectionTitle>📈 FINANCIAL OUTLOOK</SectionTitle>
          {[
            {k:"total_cost_inr",   label:"Total Cost",       fmt:(v)=>fmtInr(v),                  c:C.text},
            {k:"cost_per_sqft",    label:"Cost / sqft",      fmt:(v)=>v?`₹${Math.round(v)}/sqft`:null, c:C.text},
            {k:"roi_pct",          label:"Projected ROI",    fmt:(v)=>v!=null?`${Number(v).toFixed(1)}%`:null, c:C.green},
            {k:"equity_gain_inr",  label:"Equity Gain",      fmt:(v)=>fmtInr(v),                  c:C.green},
            {k:"payback_months",   label:"Payback Period",   fmt:(v)=>v?`${v} months`:null,        c:C.amber},
            {k:"model_confidence", label:"Model Confidence", fmt:(v)=>v?`${Math.round(v*100)}%`:null, c:C.indigo},
            {k:"within_budget",    label:"Within Budget",    fmt:(v)=>v!=null?(v?"✓ Yes":"✗ Over budget"):null, c:fin.within_budget?C.green:C.red},
          ].filter(r=>fin[r.k]!=null&&r.fmt(fin[r.k])!=null).map(row=>(
            <div key={row.k} style={{ display:"flex", justifyContent:"space-between", padding:"6px 0", borderBottom:`1px solid ${C.border}` }}>
              <span style={{ fontSize:10, color:C.muted }}>{row.label}</span>
              <span style={{ fontSize:11, fontWeight:600, ...F.mono, color:row.c }}>{row.fmt(fin[row.k])}</span>
            </div>
          ))}
        </div>}
        {Object.keys(mkt).length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
          <SectionTitle>🏙 {mkt.city||"MARKET"} INTELLIGENCE</SectionTitle>
          {mkt.market_trend&&<div style={{ fontSize:11, color:"#94a3b8", lineHeight:1.65, marginBottom:10 }}>{mkt.market_trend}</div>}
          {[
            ["5yr Appreciation", mkt.avg_appreciation_5yr],
            ["Rental Yield",     mkt.rental_yield],
            ["Labour Premium",   mkt.labour_premium],
            ["Price per sqft",   mkt.price_per_sqft_inr?`₹${Number(mkt.price_per_sqft_inr).toLocaleString("en-IN")}`:null],
            ["Demand",           mkt.demand_level],
            ["Best Time to Buy", mkt.best_season_to_buy],
          ].filter(([,v])=>v).map(([l,v])=>(
            <div key={l} style={{ display:"flex", justifyContent:"space-between", padding:"5px 0", borderTop:`1px solid ${C.border}` }}>
              <span style={{ fontSize:10, color:C.muted }}>{l}</span>
              <span style={{ fontSize:10, fontWeight:600, color:C.text, ...F.mono }}>{v}</span>
            </div>
          ))}
        </div>}
      </div>

      {/* Market timing */}
      {Object.keys(marketTiming).length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>⏰ MARKET TIMING INSIGHTS</SectionTitle>
        {marketTiming.recommendation&&<div style={{ fontSize:11, color:C.text, lineHeight:1.65, marginBottom:8 }}>{marketTiming.recommendation}</div>}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
          {[
            ["Buy Now Score",   marketTiming.buy_now_score?`${marketTiming.buy_now_score}/10`:null, C.green],
            ["Urgency",         marketTiming.urgency, marketTiming.urgency==="high"?C.red:marketTiming.urgency==="medium"?C.amber:C.muted],
            ["Best Window",     marketTiming.best_window, C.indigo],
            ["Price Direction", marketTiming.price_direction, marketTiming.price_direction?.includes("up")?C.amber:C.green],
          ].filter(([,v])=>v).map(([l,v,col])=>(
            <div key={l} style={{ padding:"8px 10px", background:C.surface, borderRadius:8, border:`1px solid ${C.border}` }}>
              <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:3 }}>{l}</div>
              <div style={{ fontSize:11, fontWeight:600, color:col||C.text }}>{v}</div>
            </div>
          ))}
        </div>
        {marketTiming.reasoning&&<div style={{ fontSize:10, color:C.muted, marginTop:10, lineHeight:1.65 }}>{marketTiming.reasoning}</div>}
      </div>}

      {/* Budget assessment */}
      {Object.keys(bud).length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <div style={{ display:"flex", gap:10, alignItems:"center", marginBottom:10 }}>
          <SectionTitle>💰 BUDGET ASSESSMENT — {(bud.tier||"").toUpperCase()} TIER</SectionTitle>
          {bud.range&&<Chip color={C.amber}>{bud.range}</Chip>}
        </div>
        {bud.covers&&<div style={{ fontSize:11, color:C.text, marginBottom:8, lineHeight:1.65 }}>{bud.covers}</div>}
        {bud.best_for&&<div style={{ fontSize:10, color:C.green, marginBottom:5 }}>✓ Best for: {bud.best_for}</div>}
        {bud.cautions&&bud.cautions!=="None"&&<div style={{ fontSize:10, color:C.amber, marginBottom:8 }}>⚠ Avoid: {bud.cautions}</div>}
        {/* Budget breakdown if available */}
        {bud.breakdown&&Object.keys(bud.breakdown).length>0&&<div style={{ marginTop:10 }}>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:6 }}>COST BREAKDOWN</div>
          {Object.entries(bud.breakdown).map(([k,v])=>(
            <div key={k} style={{ display:"flex", justifyContent:"space-between", padding:"5px 0", borderBottom:`1px solid ${C.border}` }}>
              <span style={{ fontSize:10, color:C.muted }}>{k.replace(/_/g," ").replace(/\b\w/g,l=>l.toUpperCase())}</span>
              <span style={{ fontSize:10, fontWeight:600, color:C.amber, ...F.mono }}>{fmtInr(v)}</span>
            </div>
          ))}
        </div>}
        {bud.recommended_brands?.length>0&&<div style={{ marginTop:10, display:"flex", gap:6, flexWrap:"wrap" }}>
          {bud.recommended_brands.map((b,i)=><Chip key={i} color={C.indigo}>{b}</Chip>)}
        </div>}
      </div>}

      {/* Recommendations — image-grounded */}
      {recs.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>✅ ACTION RECOMMENDATIONS</SectionTitle>
        <div style={{ fontSize:10, color:C.muted, marginBottom:10 }}>
          {insights.image_grounded?"Based on what was detected in your actual room photo":"Based on Indian renovation benchmarks for your city and budget"}
        </div>
        {recs.map((r,i)=>{
          const recText=typeof r==="string"?r:(r?.recommendation||r?.title||r?.action||JSON.stringify(r));
          const recPriority=typeof r==="object"?r?.priority:null;
          const prioColor=recPriority==="high"?C.red:recPriority==="medium"?C.amber:null;
          const reasoning=typeof r==="object"?(r?.reasoning||r?.reason):null;
          const trigger=typeof r==="object"?r?.trigger:null;
          const category=typeof r==="object"?r?.category:null;
          const source=typeof r==="object"?r?.source:null;
          return (
            <div key={i} style={{ padding:"10px 0", borderBottom:i<recs.length-1?`1px solid ${C.border}`:"none" }}>
              <div style={{ fontSize:11, color:C.text, lineHeight:1.65, display:"flex", gap:8, alignItems:"flex-start" }}>
                <span style={{ color:C.indigo, fontWeight:700, flexShrink:0 }}>{i+1}.</span>
                <div style={{ flex:1 }}>
                  <div>{recText}</div>
                  {trigger&&<div style={{ fontSize:10, color:C.muted, marginTop:3 }}>Triggered by: {trigger}</div>}
                  {Array.isArray(reasoning)&&reasoning.length>0&&<div style={{ fontSize:10, color:C.muted, marginTop:3, lineHeight:1.6 }}>
                    {reasoning.join(" · ")}
                  </div>}
                  <div style={{ display:"flex", gap:6, marginTop:5, flexWrap:"wrap" }}>
                    {recPriority&&<Chip color={prioColor||C.muted}>{recPriority} priority</Chip>}
                    {category&&<Chip color={C.muted}>{category}</Chip>}
                    {source==="image_analysis"&&<Chip color={C.indigo}>📷 from your photo</Chip>}
                    {source==="rag"&&<Chip color={C.purple}>📚 from knowledge base</Chip>}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>}


      {/* DIY renovation tips */}
      {diy?.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>🔨 DIY TIPS — SAVE ON CONTRACTOR COSTS</SectionTitle>
        <div style={{ fontSize:10, color:C.muted, marginBottom:10 }}>
          India-specific guidance matched to your room type and detected issues
        </div>
        {diy.map((tip,i)=>{
          // Backend sends {tip (chapter_title), category, guidance (content summary), source, link}
          const tipTitle=typeof tip==="string"?tip:(tip?.tip||tip?.title||tip?.action||"");
          const guidance=typeof tip==="object"?(tip?.guidance||tip?.description):null;
          const category=typeof tip==="object"?(tip?.category):null;
          const source=typeof tip==="object"?tip?.source:null;
          const link=typeof tip==="object"?tip?.link:null;
          if (!tipTitle) return null;
          return (
            <div key={i} style={{ padding:"10px 0", borderBottom:i<diy.length-1?`1px solid ${C.border}`:"none" }}>
              <div style={{ display:"flex", gap:8, alignItems:"flex-start" }}>
                <span style={{ color:C.amber, flexShrink:0, marginTop:1 }}>🔧</span>
                <div style={{ flex:1 }}>
                  <div style={{ fontSize:11, fontWeight:600, color:C.text, marginBottom:4 }}>{tipTitle}</div>
                  {guidance&&<div style={{ fontSize:10, color:C.muted, lineHeight:1.7, marginBottom:5 }}>{guidance}</div>}
                  <div style={{ display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
                    {category&&<Chip color={C.indigo}>{category.replace(/_/g," ")}</Chip>}
                    {source&&<span style={{ fontSize:9, color:C.dim, ...F.mono }}>{source}</span>}
                    {link&&link.startsWith("http")&&<a href={link} target="_blank" rel="noreferrer"
                      style={{ fontSize:9, color:C.indigo, textDecoration:"none" }}>↗ reference</a>}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>}

      {/* Risk factors */}
      {risks.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>⚠ RISK FACTORS</SectionTitle>
        {risks.map((r,i)=>{
          if (typeof r==="string") return <div key={i} style={{ padding:"8px 0", borderBottom:i<risks.length-1?`1px solid ${C.border}`:"none", fontSize:11, color:C.text }}>⚠ {r}</div>;
          return (
            <div key={i} style={{ padding:"8px 0", borderBottom:i<risks.length-1?`1px solid ${C.border}`:"none" }}>
              <div style={{ display:"flex", gap:8, alignItems:"center", marginBottom:4 }}>
                <Chip color={r.probability==="High"?C.red:r.probability==="Medium"?C.amber:C.muted}>{r.probability||"Medium"}</Chip>
                <span style={{ fontSize:11, fontWeight:600, color:C.text }}>{r.factor||r.risk||""}</span>
              </div>
              {r.detail&&<div style={{ fontSize:10, color:C.muted, marginBottom:3, lineHeight:1.6 }}>{r.detail}</div>}
              {r.mitigation&&<div style={{ fontSize:10, color:C.green, marginBottom:3 }}>✓ Mitigation: {r.mitigation}</div>}
              {r.cost_impact_inr&&<div style={{ fontSize:10, color:C.amber, ...F.mono }}>Impact if triggered: {fmtInr(r.cost_impact_inr)}</div>}
            </div>
          );
        })}
      </div>}

      {/* Top materials from RAG */}
      {mats.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>🏗 MATERIAL RECOMMENDATIONS (RAG KNOWLEDGE BASE)</SectionTitle>
        {mats.map((m,i)=>{
          const title=typeof m==="string"?m:(m?.title||m?.material||m?.name||JSON.stringify(m));
          const summary=typeof m==="object"?(m?.summary||m?.description||m?.text):null;
          return (
            <div key={i} style={{ padding:"8px 0", borderBottom:i<mats.length-1?`1px solid ${C.border}`:"none" }}>
              <div style={{ fontSize:11, color:C.text, fontWeight:600, marginBottom:3 }}>
                <span style={{ color:C.indigo }}>→ </span>{title}
              </div>
              {summary&&<div style={{ fontSize:10, color:C.muted, lineHeight:1.65 }}>{summary}</div>}
            </div>
          );
        })}
      </div>}

      {/* Action checklist */}
      {actionChecklist?.length>0&&<div style={{ background:C.card, borderRadius:10, padding:16, border:`1px solid ${C.border}` }}>
        <SectionTitle>☑ ACTION CHECKLIST</SectionTitle>
        {actionChecklist.map((item,i)=>{
          const text=typeof item==="string"?item:(item?.task||item?.action||item?.title||JSON.stringify(item));
          const done=typeof item==="object"?item?.completed:false;
          const when=typeof item==="object"?item?.when:null;
          return (
            <div key={i} style={{ display:"flex", gap:8, padding:"6px 0",
              borderBottom:i<actionChecklist.length-1?`1px solid ${C.border}`:"none", alignItems:"flex-start" }}>
              <span style={{ color:done?C.green:C.border, fontSize:14, flexShrink:0 }}>{done?"☑":"☐"}</span>
              <div>
                <div style={{ fontSize:11, color:done?C.muted:C.text, textDecoration:done?"line-through":"none" }}>{text}</div>
                {when&&<div style={{ fontSize:9, color:C.muted, ...F.mono, marginTop:2 }}>{when}</div>}
              </div>
            </div>
          );
        })}
      </div>}

      {/* Agent timing footer */}
      {Object.keys(timings).length>0&&<div style={{ background:C.surface, borderRadius:8, padding:"8px 14px",
        border:`1px solid ${C.border}`, display:"flex", gap:14, flexWrap:"wrap" }}>
        {Object.entries(timings).map(([k,v])=>(
          <div key={k} style={{ fontSize:9, color:C.muted, ...F.mono }}>{k}: <span style={{ color:C.text }}>{v?.toFixed(1)}s</span></div>
        ))}
      </div>}
    </div>
  );
});

const PriceCard = memo(function PriceCard({ forecasts }) {
  const [expanded, setExpanded] = useState(null);
  if (!forecasts?.length) return <Empty msg="Price forecasts will appear after pipeline analysis" />;

  // Summary: total budget impact across all materials
  const totalDelta = forecasts.reduce((sum, f) => {
    const d = f.budget_impact?.delta_inr || 0;
    return sum + (typeof d === "number" ? d : 0);
  }, 0);
  const highUrgency = forecasts.filter(f => f.budget_impact?.urgency === "high");

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
      {/* Summary banner */}
      {totalDelta !== 0 && (
        <div style={{ background: totalDelta > 0 ? C.amberLo : C.greenLo, borderRadius:10, padding:"12px 16px",
          border:`1px solid ${totalDelta > 0 ? C.amber : C.green}40`,
          display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <div>
            <div style={{ fontSize:11, fontWeight:600, color: totalDelta > 0 ? C.amber : C.green }}>
              {totalDelta > 0 ? "⚠ Budget at risk from rising material prices" : "✓ Prices softening — potential savings"}
            </div>
            <div style={{ fontSize:9, color:C.muted, ...F.mono, marginTop:3 }}>
              {highUrgency.length > 0 ? `${highUrgency.length} material${highUrgency.length > 1 ? "s" : ""} flagged: procure now` : "Monitor prices monthly"}
            </div>
          </div>
          <div style={{ textAlign:"right", flexShrink:0 }}>
            <div style={{ fontSize:9, color:C.muted, ...F.mono }}>90-DAY BUDGET IMPACT</div>
            <div style={{ fontSize:16, fontWeight:800, color: totalDelta > 0 ? C.amber : C.green, ...F.mono }}>
              {totalDelta > 0 ? "+" : ""}{fmtInr(Math.abs(totalDelta))}
            </div>
          </div>
        </div>
      )}

      {/* Per-material cards */}
      {forecasts.map((f, i) => {
        const isOpen = expanded === i;
        const trendColor = f.trend === "up" ? C.red : f.trend === "down" ? C.green : C.muted;
        const buyNow = f.buy_now_signal;
        const impact = f.budget_impact || {};
        const impactUrgency = impact.urgency || "low";
        const impactColor = impactUrgency === "high" ? C.red : impactUrgency === "medium" ? C.amber : C.muted;
        const name = (f.display_name || f.material_key || "Material").replace(/_/g," ").replace(/\b\w/g,l=>l.toUpperCase());
        return (
          <div key={i} style={{ background:C.card, borderRadius:10, border:`1px solid ${buyNow ? C.amber : C.border}`,
            overflow:"hidden" }}>
            {/* Card header */}
            <div style={{ padding:14 }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:10 }}>
                <div style={{ flex:1 }}>
                  <div style={{ display:"flex", gap:8, alignItems:"center", flexWrap:"wrap" }}>
                    <div style={{ fontSize:12, fontWeight:600, color:C.text }}>{name}</div>
                    {buyNow && <Chip color={C.amber}>🔔 Buy Now</Chip>}
                    {f.volatility_label && (
                      <Chip color={f.volatility_label==="High"?C.red:f.volatility_label==="Low"?C.green:C.amber}>
                        {f.volatility_label} volatility
                      </Chip>
                    )}
                  </div>
                  <div style={{ fontSize:9, color:C.muted, ...F.mono, marginTop:3 }}>
                    {f.unit}{f.city_adjusted ? ` · ${f.city_adjusted}` : ""}
                    {f.confidence_label ? ` · ${f.confidence_label} confidence` : ""}
                  </div>
                </div>
                <div style={{ textAlign:"right", flexShrink:0, marginLeft:12 }}>
                  <div style={{ fontSize:18, fontWeight:800, color:C.text, ...F.mono }}>
                    {fmtInr(f.current_price_inr)}
                  </div>
                  <Chip color={trendColor}>
                    {f.trend==="up"?"↑":f.trend==="down"?"↓":"→"} {Math.abs(f.pct_change_90d||0).toFixed(1)}% (90d)
                  </Chip>
                </div>
              </div>

              {/* 4-point price forecast */}
              <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:6, marginBottom:10 }}>
                {[
                  {l:"Now",   v:fmtInr(f.current_price_inr),         pct: null},
                  {l:"30 days", v:fmtInr(f.forecast_30d_inr||f.forecast_30d), pct:f.pct_change_30d},
                  {l:"60 days", v:fmtInr(f.forecast_60d_inr||f.forecast_60d), pct:f.pct_change_60d},
                  {l:"90 days", v:fmtInr(f.forecast_90d_inr||f.forecast_90d), pct:f.pct_change_90d},
                ].map(item=>(
                  <div key={item.l} style={{ textAlign:"center", background:C.surface,
                    borderRadius:6, padding:"6px 4px", border:`1px solid ${C.border}` }}>
                    <div style={{ fontSize:9, color:C.muted, ...F.mono }}>{item.l}</div>
                    <div style={{ fontSize:11, fontWeight:600, color:C.text, marginTop:2 }}>{item.v}</div>
                    {item.pct != null && (
                      <div style={{ fontSize:8, color: item.pct > 0 ? C.red : item.pct < 0 ? C.green : C.muted, ...F.mono }}>
                        {item.pct > 0 ? "+" : ""}{item.pct.toFixed(1)}%
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Procurement recommendation */}
              {f.procurement_recommendation && (
                <div style={{ background:C.surface, borderRadius:6, padding:"8px 12px",
                  border:`1px solid ${impactColor}40`, marginBottom:8 }}>
                  <div style={{ fontSize:10, color:C.text, lineHeight:1.6 }}>
                    <span style={{ color:impactColor, fontWeight:600 }}>
                      {impactUrgency === "high" ? "🔴 " : impactUrgency === "medium" ? "🟡 " : "🟢 "}
                    </span>
                    {f.procurement_recommendation}
                  </div>
                </div>
              )}

              {/* Budget impact row */}
              {impact.delta_inr != null && impact.delta_inr !== 0 && (
                <div style={{ display:"flex", gap:10, flexWrap:"wrap", alignItems:"center" }}>
                  <span style={{ fontSize:9, color:C.muted }}>
                    Budget impact if delayed 90d:
                  </span>
                  <span style={{ fontSize:11, fontWeight:700, color: impact.delta_inr > 0 ? C.red : C.green, ...F.mono }}>
                    {impact.delta_inr > 0 ? "+" : ""}{fmtInr(impact.delta_inr)}
                  </span>
                  {impact.estimated_qty && (
                    <span style={{ fontSize:9, color:C.dim, ...F.mono }}>
                      (est. {impact.estimated_qty} {f.unit})
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Expandable: trend narrative + confidence note */}
            {(f.trend_narrative || f.confidence_note) && (
              <>
                <button onClick={() => setExpanded(isOpen ? null : i)}
                  style={{ width:"100%", padding:"8px 14px", background:C.surface, border:"none",
                    borderTop:`1px solid ${C.border}`, display:"flex", justifyContent:"space-between",
                    alignItems:"center", cursor:"pointer", color:C.muted, fontSize:9, ...F.mono }}>
                  <span>Why is this price moving?</span>
                  <span>{isOpen ? "▲" : "▼"}</span>
                </button>
                {isOpen && (
                  <div style={{ padding:"12px 14px", borderTop:`1px solid ${C.border}`,
                    background:C.surface, display:"flex", flexDirection:"column", gap:8 }}>
                    {f.trend_narrative && (
                      <div style={{ fontSize:10, color:C.muted, lineHeight:1.7 }}>{f.trend_narrative}</div>
                    )}
                    {f.confidence_note && (
                      <div style={{ fontSize:9, color:C.dim, ...F.mono, lineHeight:1.6,
                        padding:"6px 10px", background:C.card, borderRadius:6, border:`1px solid ${C.border}` }}>
                        {f.confidence_note}
                      </div>
                    )}
                    {f.last_verified_date && (
                      <div style={{ fontSize:9, color:C.dim, ...F.mono }}>
                        Last verified: {f.last_verified_date}
                        {f.forecast_method ? ` · Method: ${f.forecast_method}` : ""}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        );
      })}

      <div style={{ fontSize:9, color:C.dim, ...F.mono, textAlign:"center", paddingTop:4 }}>
        Forecasts use Prophet + XGBoost trained on Indian construction market data. Prices are city-adjusted estimates.
        Always obtain fresh supplier quotes before procurement.
      </div>
    </div>
  );
});

const ContractorCard = memo(function ContractorCard({ contractors }) {
  const [openIdx, setOpenIdx] = useState(null);
  if (!contractors?.length) return <Empty msg="Contractor network data will appear after pipeline analysis" />;

  const VERIFY_MAP = {
    "Licensed Electrician (ISI)": {
      verify: ["Ask for ISI mark certificate (IS 732)", "Confirm ELCB/MCB installation is in scope", "Request 1-year written warranty on all wiring"],
      redFlags: ["Quotes without a site visit", "Refuses to show ISI certification", "No written scope of work"],
    },
    "Plumber (CPWD Grade B)": {
      verify: ["Check CPWD Grade B licence card", "Pressure test after pipe closure before wall sealing", "Confirm waterproof membrane is included for bathrooms"],
      redFlags: ["Does not include waterproofing in bathroom quote", "Wants full payment before work starts"],
    },
    "Plumber": {
      verify: ["Ask explicitly if waterproofing membrane is included", "Get written scope for concealed vs exposed pipes"],
      redFlags: ["No site visit before quoting", "Vague daily-rate quote with no milestone terms"],
    },
    "Painter": {
      verify: ["Confirm putty coats, primer coats, finish coats in writing", "Asian Paints / Berger certified painter for warranty compliance"],
      redFlags: ["Quote without specifying number of coats", "Skips priming stage to save cost"],
    },
    "Carpenter": {
      verify: ["Confirm plywood grade: BWP/BWR/MR in writing before order", "Clarify if hardware (hinges, channels, handles) is included in rate"],
      redFlags: ["Very low quote without material specification", "No sample board or mock-up offered before execution"],
    },
    "Flooring Specialist": {
      verify: ["Request tile layout plan before work starts to minimise wastage", "Confirm levelling compound is included — uneven floors are the #1 cause of tile cracking"],
      redFlags: ["No levelling compound in quote", "Refuses to provide a wastage estimate"],
    },
    "Civil Contractor": {
      verify: ["Mandatory site visit before quoting — ambiguous scope causes disputes", "Payment milestones tied to verified completion stages", "Retain 10% of total until 30-day post-completion inspection"],
      redFlags: ["Purely verbal agreement, no written scope", "Demands more than 40% advance payment"],
    },
    "Interior Contractor": {
      verify: ["Ask for 3D mock-up or sample board before confirming false ceiling design", "Confirm POP finishing is included in false ceiling rate"],
      redFlags: ["Grid-only quote without POP finishing specification", "No material samples or finish options shown"],
    },
    "Project Supervisor": {
      verify: ["Hire independently — not through the main contractor to avoid conflict of interest", "Define daily progress reporting and milestone-based payment terms"],
      redFlags: ["Supervisor is employed by or referred by the main contractor", "No structured daily reporting mechanism"],
    },
  };

  const totalLabour = contractors.reduce((s, c) => s + (c.estimated_labour_cost_inr || 0), 0);

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center",
        background:C.surface, borderRadius:8, padding:"10px 14px", border:`1px solid ${C.border}` }}>
        <div style={{ fontSize:9, color:C.muted, ...F.mono }}>
          🏗 {contractors.length} contractor role{contractors.length > 1 ? "s" : ""} · Urban Company, Sulekha, JustDial · Q1 2026 rates
        </div>
        {totalLabour > 0 && (
          <div style={{ textAlign:"right", flexShrink:0, marginLeft:12 }}>
            <div style={{ fontSize:9, color:C.muted, ...F.mono }}>TOTAL LABOUR EST.</div>
            <div style={{ fontSize:15, fontWeight:800, color:C.amber, ...F.mono }}>{fmtInr(totalLabour)}</div>
          </div>
        )}
      </div>

      {contractors.map((c, i) => {
        const meta = VERIFY_MAP[c.role || ""] || {};
        const isOpen = openIdx === i;
        return (
          <div key={i} style={{ background:C.card, borderRadius:10, border:`1px solid ${C.border}`, overflow:"hidden" }}>
            <div style={{ padding:14 }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:8 }}>
                <div>
                  <div style={{ fontSize:12, fontWeight:700, color:C.text }}>{c.role}</div>
                  <div style={{ fontSize:9, color:C.muted, ...F.mono, marginTop:2 }}>
                    {c.duration_days} day{c.duration_days > 1 ? "s" : ""}
                    {c.tasks?.length ? ` · ${c.tasks.slice(0,2).join(", ")}${c.tasks.length > 2 ? " …" : ""}` : ""}
                  </div>
                </div>
                <div style={{ textAlign:"right", flexShrink:0 }}>
                  <div style={{ fontSize:16, fontWeight:700, color:C.amber, ...F.mono }}>{fmtInr(c.estimated_labour_cost_inr)}</div>
                  <div style={{ fontSize:9, color:C.muted, ...F.mono }}>{fmtInr(c.daily_rate_inr)}/day</div>
                  {c.city_multiplier && c.city_multiplier !== 1 && (
                    <div style={{ fontSize:8, color:C.dim, ...F.mono }}>{c.city_multiplier}× city rate</div>
                  )}
                </div>
              </div>
              {c.tip && (
                <div style={{ fontSize:10, color:C.muted, background:C.surface, borderRadius:6,
                  padding:"6px 10px", marginBottom:10, border:`1px solid ${C.border}`, lineHeight:1.6 }}>
                  💡 {c.tip}
                </div>
              )}
              {(c.contractor_links||[]).length > 0 && (
                <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
                  {c.contractor_links.map((lnk, j) => (
                    <a key={j} href={lnk.url} target="_blank" rel="noopener noreferrer"
                      style={{ fontSize:9, padding:"4px 10px", borderRadius:4, textDecoration:"none",
                        background:C.indigoLo, color:C.indigo, border:`1px solid ${C.indigo}40`, ...F.mono }}>
                      {lnk.platform}
                    </a>
                  ))}
                </div>
              )}
            </div>
            {(meta.verify?.length || meta.redFlags?.length) && (
              <>
                <button onClick={() => setOpenIdx(isOpen ? null : i)}
                  style={{ width:"100%", padding:"8px 14px", background:C.surface, border:"none",
                    borderTop:`1px solid ${C.border}`, display:"flex", justifyContent:"space-between",
                    alignItems:"center", cursor:"pointer", color:C.muted, fontSize:9, ...F.mono }}>
                  <span>What to verify before hiring →</span>
                  <span>{isOpen ? "▲" : "▼"}</span>
                </button>
                {isOpen && (
                  <div style={{ padding:"12px 14px", borderTop:`1px solid ${C.border}`,
                    background:C.surface, display:"flex", flexDirection:"column", gap:12 }}>
                    {meta.verify?.length > 0 && (
                      <div>
                        <div style={{ fontSize:9, color:C.green, marginBottom:6, ...F.mono }}>✓ WHAT TO VERIFY</div>
                        {meta.verify.map((v, vi) => (
                          <div key={vi} style={{ display:"flex", gap:8, padding:"4px 0",
                            fontSize:10, color:C.muted, lineHeight:1.6 }}>
                            <span style={{ color:C.green, flexShrink:0 }}>•</span>{v}
                          </div>
                        ))}
                      </div>
                    )}
                    {meta.redFlags?.length > 0 && (
                      <div>
                        <div style={{ fontSize:9, color:C.red, marginBottom:6, ...F.mono }}>⚠ RED FLAGS — WALK AWAY IF YOU SEE THESE</div>
                        {meta.redFlags.map((rf, ri) => (
                          <div key={ri} style={{ display:"flex", gap:8, padding:"4px 0",
                            fontSize:10, color:C.muted, lineHeight:1.6 }}>
                            <span style={{ color:C.red, flexShrink:0 }}>•</span>{rf}
                          </div>
                        ))}
                      </div>
                    )}
                    <div style={{ fontSize:9, color:C.dim, paddingTop:4,
                      borderTop:`1px solid ${C.border}`, ...F.mono }}>
                      Rate source: ARKEN contractor survey Q1 2026 · Always get 3 written quotes
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        );
      })}
      <div style={{ fontSize:9, color:C.dim, ...F.mono, textAlign:"center", paddingTop:4 }}>
        Verify credentials, request written scope, and tie payments to milestone completion.
        Retain 10% until 30-day post-completion inspection.
      </div>
    </div>
  );
});

const ProductCard = memo(function ProductCard({ suggestions }) {
  if (!suggestions) return <Empty msg="Product recommendations will appear after pipeline analysis" />;
  if (!suggestions.shop_this_look?.length) return (
    <div style={{ padding:"32px 24px", textAlign:"center" }}>
      <div style={{ fontSize:28, marginBottom:12 }}>🛋</div>
      <div style={{ fontSize:13, color:C.muted, maxWidth:480, margin:"0 auto", lineHeight:1.7 }}>
        {suggestions.note||"No furniture or decor items were detected in the rendered image."}
      </div>
    </div>
  );
  const items=suggestions.shop_this_look;
  const total=suggestions.total_room_furnishing_estimate_inr||{};
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center",
        background:C.surface, borderRadius:10, padding:"12px 16px", border:`1px solid ${C.border}` }}>
        <div>
          <div style={{ fontSize:12, fontWeight:700, color:C.text }}>🛋 {items.length} items · {suggestions.style_label||"Detected Style"}</div>
          <div style={{ fontSize:9, color:C.muted, ...F.mono, marginTop:3 }}>{suggestions.note||"Links open real product search pages on Indian e-commerce platforms."}</div>
        </div>
        {total.low&&<div style={{ textAlign:"right", flexShrink:0, marginLeft:16 }}>
          <div style={{ fontSize:9, color:C.muted, ...F.mono }}>ROOM TOTAL EST.</div>
          <div style={{ fontSize:15, fontWeight:800, color:C.amber }}>{fmtInr(total.low)} – {fmtInr(total.high)}</div>
        </div>}
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(240px,1fr))", gap:10 }}>
        {items.map((item,i)=>(
          <div key={i} style={{ background:C.card, borderRadius:10, border:`1px solid ${C.border}`, padding:14 }}>
            <div style={{ fontSize:12, fontWeight:600, color:C.text, marginBottom:5 }}>{item.item_name}</div>
            <div style={{ display:"flex", gap:6, alignItems:"center", marginBottom:10 }}>
              <Chip color={C.muted}>{item.category}</Chip>
              {item.price_range_inr&&<span style={{ fontSize:9, color:C.amber, ...F.mono }}>
                {fmtInr(item.price_range_inr.low)} – {fmtInr(item.price_range_inr.high)}
              </span>}
            </div>
            <div style={{ display:"flex", gap:5, flexWrap:"wrap" }}>
              {(item.links||[]).map((link,j)=>(
                <a key={j} href={link.url} target="_blank" rel="noopener noreferrer"
                  style={{ fontSize:9, padding:"3px 8px", borderRadius:4, textDecoration:"none",
                    background:C.indigoLo, color:C.indigo, border:`1px solid ${C.indigo}40`, ...F.mono }}>
                  {link.store.replace(" India","").replace("Urban Ladder","UrbanLadder")}
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
});

// ── Markdown renderer (bold, italic, inline-code, bullet lists, numbered lists, headings) ──
function renderMarkdown(text) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let i = 0;

  const parseInline = (str) => {
    // Process inline formatting: bold, italic, inline code
    const parts = [];
    let remaining = str;
    let key = 0;
    // Regex: **bold**, *italic*, `code`
    const inlineRe = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
    let lastIndex = 0;
    let match;
    while ((match = inlineRe.exec(remaining)) !== null) {
      if (match.index > lastIndex) {
        parts.push(<span key={key++}>{remaining.slice(lastIndex, match.index)}</span>);
      }
      if (match[2] !== undefined) {
        parts.push(<strong key={key++} style={{ fontWeight:600 }}>{match[2]}</strong>);
      } else if (match[3] !== undefined) {
        parts.push(<em key={key++}>{match[3]}</em>);
      } else if (match[4] !== undefined) {
        parts.push(<code key={key++} style={{ background:"rgba(255,255,255,0.08)", borderRadius:3, padding:"1px 5px", fontFamily:"monospace", fontSize:11 }}>{match[4]}</code>);
      }
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < remaining.length) {
      parts.push(<span key={key++}>{remaining.slice(lastIndex)}</span>);
    }
    return parts.length > 0 ? parts : [<span key={0}>{str}</span>];
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Skip empty lines (add small spacer)
    if (!trimmed) { elements.push(<div key={i} style={{ height:6 }} />); i++; continue; }

    // Headings: ### ## #
    const hMatch = trimmed.match(/^(#{1,3})\s+(.*)/);
    if (hMatch) {
      const level = hMatch[1].length;
      const sizes = [15, 13, 12];
      elements.push(
        <div key={i} style={{ fontWeight:700, fontSize:sizes[level-1]||12, marginTop:level===1?10:6, marginBottom:2, color:"#fff" }}>
          {parseInline(hMatch[2])}
        </div>
      );
      i++; continue;
    }

    // Bullet list: *, -, •
    if (/^[-*•]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^[-*•]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*•]\s+/, ""));
        i++;
      }
      elements.push(
        <ul key={`ul-${i}`} style={{ margin:"4px 0", paddingLeft:16, listStyle:"none" }}>
          {items.map((item, idx) => (
            <li key={idx} style={{ display:"flex", gap:6, marginBottom:2 }}>
              <span style={{ color:"#6470f3", fontWeight:700, flexShrink:0 }}>·</span>
              <span>{parseInline(item)}</span>
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Numbered list: 1. 2. etc
    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i++;
      }
      elements.push(
        <ol key={`ol-${i}`} style={{ margin:"4px 0", paddingLeft:0, listStyle:"none" }}>
          {items.map((item, idx) => (
            <li key={idx} style={{ display:"flex", gap:8, marginBottom:3 }}>
              <span style={{ color:"#6470f3", fontWeight:700, minWidth:16, flexShrink:0 }}>{idx+1}.</span>
              <span>{parseInline(item)}</span>
            </li>
          ))}
        </ol>
      );
      continue;
    }

    // Horizontal rule
    if (/^---+$/.test(trimmed)) {
      elements.push(<hr key={i} style={{ border:"none", borderTop:"1px solid rgba(255,255,255,0.1)", margin:"6px 0" }} />);
      i++; continue;
    }

    // Normal paragraph
    elements.push(<p key={i} style={{ margin:"2px 0", lineHeight:1.7 }}>{parseInline(trimmed)}</p>);
    i++;
  }

  return elements;
}

const ChatMsg = memo(function ChatMsg({ msg }) {
  const isUser=msg.role==="user";
  return (
    <div style={{ display:"flex", justifyContent:isUser?"flex-end":"flex-start", marginBottom:8 }}>
      <div style={{ maxWidth:"82%", padding:"10px 14px",
        borderRadius:isUser?"12px 12px 4px 12px":"12px 12px 12px 4px",
        background:isUser?C.indigo:C.card, border:isUser?"none":`1px solid ${C.border}`,
        fontSize:12, color:isUser?"#fff":C.text, lineHeight:1.65 }}>
        {isUser ? msg.content : renderMarkdown(msg.content)}
        {msg.triggers_rerender&&<div style={{ marginTop:6, fontSize:9, color:C.green, ...F.mono }}>✨ Re-render triggered</div>}
      </div>
    </div>
  );
});

// ══════════════════════════════════════════════════════════════════════════
// MAIN APP
// ══════════════════════════════════════════════════════════════════════════
export default function ARKENApp() {
  const [city,   setCity]   = useState("Hyderabad");
  const [room,   setRoom]   = useState("Bedroom");
  const [theme,  setTheme]  = useState("Modern Minimalist");
  const [budget, setBudget] = useState(BUDGETS[1]);

  const [origUrl,  setOrigUrl]  = useState(null);
  const [origB64,  setOrigB64]  = useState(null);
  const [origMime, setOrigMime] = useState("image/jpeg");
  const [renUrl,   setRenUrl]   = useState(null);
  const [renB64,   setRenB64]   = useState(null);
  const [renMime,  setRenMime]  = useState("image/png");

  const [result,          setResult]          = useState(null);
  const [contractorList,  setContractorList]  = useState([]);
  const [insights,        setInsights]        = useState(null);
  const [insLoading,      setInsLoading]      = useState(false);
  const [priceData,       setPriceData]       = useState(null);
  const [pipelineCtx,     setPipelineCtx]     = useState("");
  const [agentStatuses,   setAgentStatuses]   = useState({});
  const [agentTimings,    setAgentTimings]    = useState({});
  const [completedAgents, setCompletedAgents] = useState([]);

  // FIX c: chatHist at top level so it persists when switching tabs
  const [chatInput,   setChatInput]   = useState("");
  const [chatHist,    setChatHist]    = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [sessionId,   setSessionId]   = useState(null);

  const [step,        setStep]        = useState(0);
  const [activeTab,   setActiveTab]   = useState("insights");
  const [rendering,   setRendering]   = useState(false);
  const [renderErr,   setRenderErr]   = useState("");
  const [renderModel, setRenderModel] = useState("");
  const [rerenderVer, setRerenderVer] = useState(1);
  const [pdfLoading,  setPdfLoading]  = useState(false);
  const [shareToast,  setShareToast]  = useState(false);

  const fileRef        = useRef(null);
  const chatEnd        = useRef(null);
  const lastInsightPid = useRef(null);

  useEffect(()=>{ chatEnd.current?.scrollIntoView({behavior:"smooth"}); },[chatHist]);

  // FIX f: Escape closes chat panel
  useEffect(()=>{
    const h=(e)=>{ if(e.key==="Escape"&&step===5) setStep(4); };
    window.addEventListener("keydown",h);
    return ()=>window.removeEventListener("keydown",h);
  },[step]);

  const handleFile = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = await b64(file);
    setOrigUrl(url); setOrigB64(img); setOrigMime(file.type||"image/jpeg");
    setRenUrl(null); setRenB64(null); setRenderErr(""); setRenderModel("");
    setResult(null); setInsights(null); setPipelineCtx(""); setPriceData(null);
    setAgentStatuses({}); setAgentTimings({}); setCompletedAgents([]);
    setChatHist([]); setSessionId(null);
    lastInsightPid.current=null;
    setStep(1);
  },[]);

  const handleRender = useCallback(async (customInstr="", materialOverrides=null, ver=null) => {
    if (!origB64) return;
    const v=ver||rerenderVer;
    setRendering(true); setRenderErr("");
    setInsights(null); setAgentStatuses({}); setAgentTimings({});
    const steps=AGENT_META.map(a=>a.id);
    steps.forEach((s,i)=>{ setTimeout(()=>setAgentStatuses(prev=>({...prev,[s]:i===0?"running":"idle"})),i*100); });
    const roomKey=ROOM_MAP[room]||room.toLowerCase().replace(/ /g,"_");
    try {
      const [roiRes,priceRes,renderRes]=await Promise.allSettled([
        api("/forecast/roi",{method:"POST",headers:{"Content-Type":"application/json"},
          // FIX: area_sqft:120 was a room area causing ₹4L property values.
          // Pass flat area (900 sqft 2BHK default) and explicit property value
          // so the backend doesn't recompute from room area × PSF.
          body:JSON.stringify({
            renovation_cost_inr:budget.inr,
            area_sqft:900,
            city,
            room_type:roomKey,
            budget_tier:budget.tier,
            // Let backend compute property value from flat area × city PSF
            current_property_value_inr:null,
          })}),
        api("/forecast/materials?horizon_days=90"),
        api("/render/",{method:"POST",headers:{"Content-Type":"application/json"},
          body:JSON.stringify({project_id:`proj_${Date.now()}`,original_image_b64:origB64,original_mime:origMime,
            version:v,theme,city,budget_tier:budget.tier,room_type:roomKey,
            custom_instructions:customInstr,material_overrides:materialOverrides})}),
      ]);
      if(renderRes.status==="rejected") throw new Error(renderRes.reason?.message||"Render failed");
      const rd=renderRes.value;
      const imgB64=rd.image_b64; const imgMime=rd.image_mime||"image/png";
      setRenB64(imgB64); setRenMime(imgMime);
      setRenUrl(`data:${imgMime};base64,${imgB64}`);
      setRenderModel(rd.model_used||"gemini-2.5-flash-image");
      setRerenderVer(v+1);
      // FIX: only use the direct /forecast/roi result as a SEED — it will be
      // overwritten by the richer pipeline result once loadInsights() completes.
      // Also verify the result has rent fields (not just roi_pct) before using.
      if(roiRes.status==="fulfilled"&&roiRes.value?.roi_pct) {
        const roiVal=roiRes.value;
        // Attach city/room_type so ROIPanel has context even from direct call
        setResult(prev=>({...(prev||{}),roi:{...roiVal,city:roiVal.city||city,room_type:roiVal.room_type||roomKey,budget_tier:roiVal.budget_tier||budget.tier}}));
      }
      if(priceRes.status==="fulfilled"&&priceRes.value?.forecasts?.length) setPriceData(priceRes.value.forecasts);
      setStep(3);
      setTimeout(()=>loadInsights(imgB64,imgMime),400);
    } catch(err) {
      setRenderErr(err.message||"Unexpected error");
      AGENT_META.forEach(a=>setAgentStatuses(prev=>({...prev,[a.id]:"error"})));
    } finally { setRendering(false); }
  },[origB64,origMime,budget,room,city,theme,rerenderVer]);

  const loadInsights = useCallback(async (imgB64Arg,imgMimeArg) => {
    const ib64=imgB64Arg||renB64; const imime=imgMimeArg||renMime;
    if(!ib64) return;
    setInsLoading(true);
    const init={};
    AGENT_META.forEach((a,i)=>{init[a.id]=i===0?"running":"idle";});
    setAgentStatuses(init);
    const insightPid=`ins_${Date.now()}`;
    lastInsightPid.current=insightPid;
    const roomKey=ROOM_MAP[room]||room.toLowerCase().replace(/ /g,"_");
    try {
      const data=await api("/chat/insights",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({project_id:insightPid,original_image_b64:origB64,original_image_mime:origMime,
          renovated_image_b64:ib64,renovated_image_mime:imime,theme,city,
          budget_tier:budget.tier,budget_inr:budget.inr,room_type:roomKey})});
      setInsights(data.insights||{});
      setPipelineCtx(data.pipeline_summary?.chat_context||"");
      setCompletedAgents(data.completed_agents||[]);
      setAgentTimings(data.agent_timings||{});
      const statusMap={};
      (data.completed_agents||[]).forEach(a=>{statusMap[a]="complete";});
      setAgentStatuses(statusMap);
      const ps=data.pipeline_summary||{};
      // FIX: chat.py now sends both "roi" and "roi_prediction" keys (full dict).
      // Read "roi" first (preferred), then fall back to "roi_prediction".
      const pipelineRoi = ps.roi || ps.roi_prediction || ps.roi_output;
      if(pipelineRoi?.roi_pct) {
        // Pipeline result overwrites the seed from direct /forecast/roi call —
        // it has accurate property values (flat area × city PSF) and all rent fields.
        setResult(prev=>({...(prev||{}),roi:pipelineRoi}));
      }

      // DEFINITIVE COST FIX: cost_estimate.total_inr is the authoritative construction cost.
      // It is set by BudgetEstimatorAgent (city-adjusted, includes GST+contingency).
      // Store it separately so FeedbackPanel (Cost Accuracy card) always shows this number,
      // and never gets inflated by product suggestions or BOQ-recomputation divergence.
      const authCost = ps.cost_estimate?.total_inr;
      if(authCost>0) {
        setResult(prev=>({...(prev||{}),renovationCostInr:authCost}));
      }

      const dp=ps.design_plan;
      if(dp&&(dp.total_inr||dp.line_items?.length)) {
        // Recompute labour_inr from Labour- category line items (design_planner v3)
        const dpItems=dp.line_items||[];
        const labourFromItems=dpItems.filter(i=>(i.category||'').startsWith('Labour'))
          .reduce((s,i)=>s+(i.total_inr||0),0);
        const matFromItems=dpItems.filter(i=>!(i.category||'').startsWith('Labour'))
          .reduce((s,i)=>s+(i.total_inr||0),0);
        const enrichedDp={
          ...dp,
          labour_inr: labourFromItems||dp.labour_inr||0,
          material_inr: matFromItems||dp.material_inr||0,
        };
        setResult(prev=>({...(prev||{}),design:enrichedDp}));
      } else if(ps.boq_line_items?.length) {
        const items=ps.boq_line_items;
        const labTot=items.filter(i=>(i.category||'').startsWith('Labour'))
          .reduce((s,i)=>s+(i.total_inr||0),0)||ps.labour_estimate||0;
        const matTot=items.filter(i=>!(i.category||'').startsWith('Labour'))
          .reduce((s,i)=>s+(i.total_inr||0),0);
        const gst=Math.round((matTot+labTot)*0.18);
        const cont=Math.round((matTot+labTot)*0.10);
        setResult(prev=>({...(prev||{}),design:{total_inr:matTot+labTot+gst+cont,
          material_inr:matTot,labour_inr:labTot,gst_inr:gst,contingency_inr:cont,
          line_items:items}}));
      }
      if(ps.schedule?.tasks?.length) setResult(prev=>({...(prev||{}),schedule:ps.schedule}));
      if(ps.material_prices?.length) setPriceData(ps.material_prices);
      if(ps.product_suggestions!=null) {
        const prodSugg=ps.product_suggestions;
        setResult(prev=>({...(prev||{}),product_suggestions:prodSugg}));
        // Add products furnishing estimate to the design total
        const prodTotal=prodSugg?.total_room_furnishing_estimate_inr;
        const prodMid=prodTotal?.mid||(prodTotal?.low&&prodTotal?.high
          ?Math.round((prodTotal.low+prodTotal.high)/2):0);
        if(prodMid>0){
          setResult(prev=>{
            const d=prev?.design;
            if(!d) return prev;
            const newTotal=(d.total_inr||0)+prodMid;
            return {...prev,design:{...d,
              total_inr:newTotal,
              products_subtotal_inr:prodMid,
            }};
          });
        }
      }
      if(ps.contractor_list?.length) setContractorList(ps.contractor_list);
    } catch(err) {
      console.error("Insights error:",err);
      AGENT_META.forEach(a=>setAgentStatuses(prev=>({...prev,[a.id]:"error"})));
    } finally { setInsLoading(false); }
  },[renB64,renMime,origB64,origMime,room,city,theme,budget]);

  // FIX d: PDF loading spinner
  const downloadPDF = useCallback(async () => {
    if(!insights&&!result){ alert("Run an analysis first to generate a report"); return; }
    setPdfLoading(true);
    try {
      const pid=lastInsightPid.current||`pdf_${Date.now()}`;
      const r=await fetch(`${API}/artifacts/${pid}/report/pdf`,{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({project_id:pid,theme,city,
          room_type:ROOM_MAP[room]||room.toLowerCase().replace(/ /g,"_"),
          budget_tier:budget.tier,budget_inr:budget.inr,
          insights:insights||{},roi:result?.roi||{},design:result?.design||{},
          schedule:result?.schedule||{},material_prices:priceData||{},
          render_url:renUrl||"",
          // Before / after images for the PDF visual comparison
          original_image_b64:origB64||"",
          original_image_mime:origMime||"image/jpeg",
          renovated_image_b64:renB64||"",
        })});
      if(!r.ok){ const err=await r.json().catch(()=>({detail:`HTTP ${r.status}`})); throw new Error(err.detail||`PDF failed (${r.status})`); }
      const blob=await r.blob();
      const a=document.createElement("a");
      a.href=URL.createObjectURL(blob);
      a.download=`arken-report-${pid.slice(-8)}.pdf`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch(err){ alert(`PDF Error: ${err.message}`); }
    finally { setPdfLoading(false); }
  },[insights,result,theme,city,room,budget,renUrl,priceData]);

  // Share Results
  const handleShare = useCallback(()=>{
    const styleLabel=insights?.visual_analysis?.style_detected||theme;
    const cost=result?.design?.total_inr;
    const roi=result?.roi?.roi_pct;
    const summary=`ARKEN Renovation Analysis — ${room} in ${city}`+
      ` | Style: ${styleLabel}`+
      (cost?` | Estimated cost: ${fmtInr(cost)}`:"")+
      (roi?` | ROI: ${roi.toFixed(1)}%`:"")+
      ` | Theme: ${theme} | Generated by ARKEN PropTech`;
    navigator.clipboard?.writeText(summary).then(()=>{
      setShareToast(true); setTimeout(()=>setShareToast(false),2500);
    }).catch(()=>alert(summary));
  },[insights,result,room,city,theme]);

  const handleChat = useCallback(async () => {
    if(!chatInput.trim()||chatLoading) return;
    const msg=chatInput.trim();
    setChatInput("");
    const newHist=[...chatHist,{role:"user",content:msg}];
    setChatHist(newHist);
    setChatLoading(true);
    const roomKey=ROOM_MAP[room]||room.toLowerCase().replace(/ /g,"_");
    try {
      const data=await api("/chat/",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({project_id:`chat_${Date.now()}`,session_id:sessionId,messages:newHist,
          original_image_b64:origB64,original_mime:origMime,renovated_image_b64:renB64,renovated_mime:renMime,
          theme,city,budget_tier:budget.tier,room_type:roomKey,pipeline_context:pipelineCtx})});
      setSessionId(data.session_id);
      const botMsg={role:"assistant",content:data.message,triggers_rerender:data.triggers_rerender,action:data.action};
      setChatHist(prev=>[...prev,botMsg]);
      if(data.triggers_rerender&&data.action){
        const instr=data.action.custom_instructions||"";
        const overrides=data.action.material_overrides||null;
        if(data.action.theme_change) setTheme(data.action.theme_change);
        setTimeout(async()=>{
          await handleRender(instr,overrides,rerenderVer);
          setChatHist(prev=>[...prev,{role:"assistant",content:`Re-render complete (v${rerenderVer}). Updated — check Before/After slider.`}]);
        },600);
      }
    } catch(err){ setChatHist(prev=>[...prev,{role:"assistant",content:`Error: ${err.message}`}]); }
    finally { setChatLoading(false); }
  },[chatInput,chatLoading,chatHist,sessionId,origB64,origMime,renB64,renMime,theme,city,budget,room,pipelineCtx,handleRender,rerenderVer]);

  const _pdfDisabled=pdfLoading||(!insights&&!result);
  const renderPDFBtn=useCallback((extraStyle={})=>(
    <button onClick={downloadPDF} disabled={_pdfDisabled}
      style={{ padding:"6px 14px", borderRadius:7, border:`1px solid ${C.green}60`, background:C.greenLo,
        color:_pdfDisabled?C.muted:C.green, fontSize:11, cursor:_pdfDisabled?"default":"pointer",
        ...F.mono, display:"flex", alignItems:"center", gap:6, ...extraStyle }}>
      {pdfLoading
        ?<><span style={{ display:"inline-block",width:10,height:10,borderRadius:"50%",
            border:`2px solid ${C.green}40`,borderTopColor:C.green,animation:"spin 0.8s linear infinite" }}/>Generating…</>
        :"↓ PDF"}
    </button>
  ),[downloadPDF,_pdfDisabled,pdfLoading]);

  const vastuDetectedObjects=useMemo(()=>{
    const vis=insights?.visual_analysis||{};
    const objs=[];
    if(vis.style_detected) objs.push(vis.style_detected);
    if(vis.wall_treatment) objs.push(vis.wall_treatment);
    if(vis.floor_material) objs.push(vis.floor_material);
    if(vis.ceiling) objs.push(vis.ceiling);
    if(Array.isArray(vis.specific_changes_detected)) objs.push(...vis.specific_changes_detected);
    return objs;
  },[insights]);
  const vastuStyleLabel=insights?.visual_analysis?.style_detected||theme;
  const roomKey=ROOM_MAP[room]||room.toLowerCase().replace(/ /g,"_");

  return (
    <div style={{ minHeight:"100vh", background:C.bg, color:C.text, ...F.sans, overflowX:"hidden" }}>
      <style>{`
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:4px;height:4px;}
        ::-webkit-scrollbar-track{background:${C.surface};}
        ::-webkit-scrollbar-thumb{background:${C.border};border-radius:2px;}
        button:focus,input:focus,select:focus{outline:none;}
        @keyframes spin{to{transform:rotate(360deg)}}
      `}</style>

      {/* Navbar */}
      <nav style={{ height:52, borderBottom:`1px solid ${C.border}`, display:"flex",
        alignItems:"center", justifyContent:"space-between", padding:"0 28px",
        background:"rgba(5,8,15,0.97)", backdropFilter:"blur(16px)",
        position:"sticky", top:0, zIndex:200 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ width:28, height:28, background:"linear-gradient(135deg,#6470f3,#a78bfa)",
            borderRadius:7, display:"flex", alignItems:"center", justifyContent:"center",
            fontWeight:800, fontSize:13, color:"#fff" }}>A</div>
          <span style={{ fontWeight:800, fontSize:15, letterSpacing:"-0.025em" }}>ARKEN</span>
          <span style={{ fontSize:9, color:C.muted, ...F.mono, marginLeft:2 }}>PropTech v5.0</span>
        </div>
        <div style={{ display:"flex", gap:6, alignItems:"center" }}>
          <DataFreshnessBadge />
          {step>0&&["Upload","Configure","Preview","Results","Chat"].map((l,i)=>(
            <div key={i} style={{ display:"flex", alignItems:"center" }}>
              <button onClick={()=>step>i+1&&setStep(i+1)}
                style={{ fontSize:9, ...F.mono, cursor:step>i+1?"pointer":"default",
                  border:"none", background:"none", padding:"3px 8px", borderRadius:5,
                  color:step===i+1?C.indigo:step>i+1?C.green:C.dim }}>
                {step>i+1?"✓ ":""}{l}
              </button>
              {i<4&&<span style={{ color:C.border, fontSize:10 }}>›</span>}
            </div>
          ))}
          {renderModel&&<div style={{ fontSize:9, ...F.mono, color:C.muted, background:C.surface,
            padding:"3px 8px", borderRadius:5, border:`1px solid ${C.border}`, marginLeft:4 }}>
            ⚡ {renderModel}
          </div>}
          {lastInsightPid.current&&renderPDFBtn({marginLeft:4})}
          <ModelHealthBadge />
        </div>
      </nav>

      <AnimatePresence>
        {shareToast&&<motion.div initial={{opacity:0,y:-20}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-20}}
          style={{ position:"fixed",top:62,right:24,zIndex:9999,background:C.greenLo,
            border:`1px solid ${C.green}50`,borderRadius:8,padding:"10px 16px",
            fontSize:11,color:C.green,...F.mono,boxShadow:"0 8px 24px rgba(0,0,0,0.4)" }}>
          ✓ Summary copied to clipboard!
        </motion.div>}
      </AnimatePresence>

      <AnimatePresence mode="sync">

      {step===0&&<motion.div key="s0" initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} exit={{opacity:0}}
        style={{ maxWidth:600, margin:"80px auto", padding:"0 28px", textAlign:"center" }}>
        <div style={{ fontSize:9, color:C.indigo, ...F.mono, letterSpacing:"0.18em", marginBottom:20 }}>
          AI-POWERED RENOVATION INTELLIGENCE FOR INDIA
        </div>
        <h1 style={{ fontSize:42, fontWeight:800, letterSpacing:"-0.03em", lineHeight:1.1,
          marginBottom:16, background:"linear-gradient(145deg,#e8edf5,#7088b0)",
          WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
          Renovate Any Room<br />with Full AI Pipeline
        </h1>
        <p style={{ fontSize:14, color:C.muted, marginBottom:8, lineHeight:1.75 }}>
          Upload a room photo → Gemini renders your renovation → 10-agent LangGraph pipeline generates
          a real SKU-based BOQ from Indian brand catalog, XGBoost + SHAP ROI forecast, CPM construction schedule,
          90-day material price intelligence, and a downloadable PDF report.
        </p>
        <div onClick={()=>fileRef.current?.click()}
          style={{ border:`1.5px dashed ${C.borderHi}`, borderRadius:16, padding:"56px 36px",
            cursor:"pointer", background:C.surface, transition:"all 0.2s", marginBottom:28 }}
          onMouseEnter={e=>e.currentTarget.style.borderColor=C.indigo}
          onMouseLeave={e=>e.currentTarget.style.borderColor=C.borderHi}>
          <div style={{ fontSize:36, marginBottom:12, opacity:0.3 }}>⬆</div>
          <div style={{ fontSize:13, color:C.muted }}>Drop or click to upload room photo</div>
          <div style={{ fontSize:10, color:C.dim, marginTop:6, ...F.mono }}>JPG · PNG · WEBP · max 20MB</div>
        </div>
        <input ref={fileRef} type="file" accept="image/*" onChange={handleFile} style={{display:"none"}} />
        <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:12 }}>
          {[{icon:"🤖",t:"10-Agent Pipeline",d:"Vision → RAG → BOQ → ROI → Market → Insight → Report"},
            {icon:"📊",t:"XGBoost + SHAP ROI",d:"ML on 32,963 Indian property transactions"},
            {icon:"📄",t:"PDF Report",d:"Full BOQ, schedule, ROI forecast — downloadable"}].map(f=>(
            <div key={f.t} style={{ background:C.surface, borderRadius:12, padding:16, border:`1px solid ${C.border}`, textAlign:"left" }}>
              <div style={{ fontSize:22, marginBottom:8 }}>{f.icon}</div>
              <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:4 }}>{f.t}</div>
              <div style={{ fontSize:10, color:C.muted, lineHeight:1.5 }}>{f.d}</div>
            </div>
          ))}
        </div>
      </motion.div>}

      {step===1&&<motion.div key="s1" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} exit={{opacity:0}}
        style={{ maxWidth:980, margin:"32px auto", padding:"0 28px" }}>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1.15fr", gap:28 }}>
          <div>
            <div style={{ fontSize:9, color:C.muted, ...F.mono, letterSpacing:"0.1em", marginBottom:10 }}>YOUR ROOM</div>
            <img src={origUrl} alt="original"
              style={{ width:"100%", borderRadius:14, aspectRatio:"4/3", objectFit:"cover", border:`1px solid ${C.border}` }} />
            <div style={{ marginTop:10, padding:10, background:C.surface, borderRadius:8, border:`1px solid ${C.border}`, fontSize:10, color:C.muted }}>
              Wrong photo?{" "}<span onClick={()=>fileRef.current?.click()} style={{ color:C.indigo, cursor:"pointer" }}>Change</span>
            </div>
            <input ref={fileRef} type="file" accept="image/*" onChange={handleFile} style={{display:"none"}} />
          </div>
          <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
            <div>
              <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:8 }}>CITY</div>
              <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:6 }}>
                {CITIES.map(c=>(
                  <button key={c} onClick={()=>setCity(c)} style={{ fontSize:10, padding:"8px 4px", borderRadius:7, cursor:"pointer",
                    border:`1px solid ${city===c?C.indigo:C.border}`, background:city===c?C.indigoLo:C.surface,
                    color:city===c?C.indigo:C.muted, ...F.mono, transition:"all 0.15s" }}>{c}</button>
                ))}
              </div>
            </div>
            <div>
              <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:8 }}>ROOM TYPE</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6 }}>
                {ROOMS.map(r=>(
                  <button key={r} onClick={()=>setRoom(r)} style={{ fontSize:10, padding:"8px 10px", borderRadius:7, cursor:"pointer",
                    border:`1px solid ${room===r?C.amber:C.border}`, background:room===r?C.amberLo:C.surface,
                    color:room===r?C.amber:C.muted, textAlign:"left", ...F.mono, transition:"all 0.15s" }}>{r}</button>
                ))}
              </div>
            </div>
            <div>
              <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:8 }}>DESIGN THEME</div>
              <select value={theme} onChange={e=>setTheme(e.target.value)}
                style={{ width:"100%", padding:"10px 14px", borderRadius:8, border:`1px solid ${C.border}`,
                  background:C.surface, color:C.text, fontSize:12, cursor:"pointer", ...F.mono }}>
                {THEMES.map(t=><option key={t}>{t}</option>)}
              </select>
            </div>
            <div>
              <div style={{ fontSize:9, color:C.muted, ...F.mono, marginBottom:8 }}>BUDGET TIER</div>
              <div style={{ display:"flex", gap:8 }}>
                {BUDGETS.map(b=>(
                  <button key={b.label} onClick={()=>setBudget(b)} style={{ flex:1, padding:"12px 8px", borderRadius:8, cursor:"pointer",
                    border:`1px solid ${budget.label===b.label?b.color:C.border}`,
                    background:budget.label===b.label?`${b.color}18`:C.surface,
                    color:budget.label===b.label?b.color:C.muted, transition:"all 0.15s" }}>
                    <div style={{ fontSize:12, fontWeight:700 }}>{b.label}</div>
                    <div style={{ fontSize:10, ...F.mono, opacity:0.7 }}>{b.range}</div>
                  </button>
                ))}
              </div>
            </div>
            <button onClick={()=>setStep(2)} style={{ padding:"14px 0", borderRadius:10, border:"none",
              background:"linear-gradient(135deg,#6470f3,#a78bfa)", color:"#fff", fontSize:13, fontWeight:700, cursor:"pointer" }}>
              Confirm & Proceed →
            </button>
          </div>
        </div>
      </motion.div>}

      {step===2&&<motion.div key="s2" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} exit={{opacity:0}}
        style={{ maxWidth:800, margin:"48px auto", padding:"0 28px" }}>
        <AgentPipeline statuses={agentStatuses} timings={agentTimings} />
        <div style={{ background:C.surface, borderRadius:14, padding:24, border:`1px solid ${C.border}`, marginBottom:18 }}>
          <SectionTitle>RENOVATION BRIEF</SectionTitle>
          {[{l:"Room",v:room},{l:"Theme",v:theme},{l:"City",v:city},{l:"Budget",v:`${budget.label} (${budget.range})`}].map(r=>(
            <div key={r.l} style={{ display:"flex", justifyContent:"space-between", padding:"10px 0", borderBottom:`1px solid ${C.border}` }}>
              <span style={{ fontSize:11, color:C.muted, ...F.mono }}>{r.l}</span>
              <span style={{ fontSize:12, color:C.text, fontWeight:500 }}>{r.v}</span>
            </div>
          ))}
        </div>

        {/* FIX a: user-friendly error card */}
        {renderErr&&<div style={{ background:C.redLo, border:`1px solid ${C.red}60`, borderRadius:12, padding:18, marginBottom:16 }}>
          <div style={{ color:C.red, fontSize:13, fontWeight:600, marginBottom:6 }}>✗ Render failed</div>
          <div style={{ color:C.red, fontSize:11, ...F.mono, marginBottom:12 }}>{renderErr}</div>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:12 }}>
            <button onClick={()=>{
              setRenderErr(""); setStep(0); setRenUrl(null); setRenB64(null);
              setInsights(null); setPipelineCtx(""); setChatHist([]);
              setResult(null); setPriceData(null); setContractorList([]);
              setAgentStatuses({}); setCompletedAgents([]);
              lastInsightPid.current=null;
            }} style={{ padding:"7px 16px", borderRadius:7, border:`1px solid ${C.red}60`,
              background:"transparent", color:C.red, fontSize:11, cursor:"pointer", ...F.mono }}>
              ↺ Try again (reset)
            </button>
            <button onClick={()=>handleRender()} disabled={rendering}
              style={{ padding:"7px 16px", borderRadius:7, border:`1px solid ${C.amber}60`,
                background:"transparent", color:C.amber, fontSize:11, cursor:"pointer", ...F.mono }}>
              ↺ Retry same settings
            </button>
          </div>
          <details style={{ fontSize:10, color:C.muted }}>
            <summary style={{ cursor:"pointer", color:C.amber, ...F.mono }}>Common issues</summary>
            <ul style={{ marginTop:8, paddingLeft:16, lineHeight:1.9, ...F.mono }}>
              <li>Ensure good lighting — dark or blurry photos reduce Gemini accuracy</li>
              <li>File size must be under 20MB</li>
              <li>Supported formats: JPG, PNG, WebP</li>
              <li>Try a different room angle if the render fails repeatedly</li>
            </ul>
          </details>
        </div>}

        <button onClick={()=>handleRender()} disabled={rendering}
          style={{ width:"100%", padding:"15px 0", borderRadius:12, border:"none",
            background:rendering?"linear-gradient(90deg,#1a2540,#2a3f6b)":"linear-gradient(135deg,#6470f3,#a78bfa)",
            color:"#fff", fontSize:14, fontWeight:700, cursor:rendering?"default":"pointer",
            position:"relative", overflow:"hidden" }}>
          {rendering?<span style={{ ...F.mono, fontSize:11 }}>◈ Generating renovation — Gemini processing...</span>:"Launch Renovation Pipeline →"}
          {rendering&&<motion.div animate={{x:["-100%","100%"]}} transition={{repeat:Infinity,duration:1.4,ease:"linear"}}
            style={{ position:"absolute",inset:0,background:"linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent)" }}/>}
        </button>
        <button onClick={()=>setStep(1)} style={{ marginTop:10, width:"100%", background:"none", border:"none",
          color:C.muted, fontSize:11, cursor:"pointer", ...F.mono }}>← Back to settings</button>
      </motion.div>}

      {/* STEP 3: Before/After Preview — slider unchanged in size */}
      {step===3&&<motion.div key="s3" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} exit={{opacity:0}}
        style={{ maxWidth:1160, margin:"28px auto", padding:"0 28px" }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16 }}>
          <div>
            <div style={{ fontSize:18, fontWeight:700, letterSpacing:"-0.02em" }}>Renovation Preview</div>
            <div style={{ fontSize:10, color:C.muted, marginTop:2, ...F.mono }}>{theme} · {city} · {budget.label} · Drag slider to compare</div>
          </div>
          <div style={{ display:"flex", gap:8 }}>
            <button onClick={()=>handleRender()} disabled={rendering}
              style={{ padding:"7px 14px", borderRadius:7, border:`1px solid ${C.border}`,
                background:C.surface, color:C.muted, fontSize:11, cursor:"pointer", ...F.mono }}>↺ Re-render</button>
            {renderPDFBtn()}
            <button onClick={()=>setStep(1)} style={{ padding:"7px 14px", borderRadius:7, border:`1px solid ${C.border}`,
              background:C.surface, color:C.muted, fontSize:11, cursor:"pointer" }}>← Settings</button>
          </div>
        </div>
        <BASlider before={origUrl} after={renUrl} />
        <div style={{ marginTop:12, display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:10, marginBottom:20 }}>
          {[
            {l:"ORIGINAL",  d:"Your room before renovation",        c:C.border,        tc:C.muted,  onClick:null},
            {l:"AI RENDER", d:`${theme} — Gemini 2.5 Flash`,        c:`${C.indigo}40`, tc:C.indigo, onClick:null},
            {l:"RESULTS →", d:"BOQ, ROI, Schedule, Insights",       c:`${C.green}40`,  tc:C.green,  onClick:()=>setStep(4)},
            {l:"CHAT →",    d:"Ask about materials, costs, changes", c:`${C.purple}40`, tc:C.purple, onClick:()=>setStep(5)},
          ].map((card,i)=>(
            <div key={i} onClick={card.onClick}
              onMouseEnter={e=>card.onClick&&(e.currentTarget.style.borderColor=card.tc)}
              onMouseLeave={e=>card.onClick&&(e.currentTarget.style.borderColor=card.c)}
              style={{ padding:14, borderRadius:12, background:C.surface, border:`1px solid ${card.c}`,
                cursor:card.onClick?"pointer":"default", transition:"border-color 0.2s" }}>
              <div style={{ fontSize:9, ...F.mono, color:card.tc, marginBottom:5 }}>{card.l}</div>
              <div style={{ fontSize:10, color:C.muted }}>{card.d}</div>
            </div>
          ))}
        </div>
      </motion.div>}

      {/* STEP 4: Results */}
      {step===4&&<motion.div key="s4" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} exit={{opacity:0}}
        style={{ maxWidth:1160, margin:"28px auto", padding:"0 28px 80px" }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:20 }}>
          <div>
            <div style={{ fontSize:18, fontWeight:700, letterSpacing:"-0.02em" }}>Pipeline Results</div>
            <div style={{ fontSize:10, color:C.muted, ...F.mono, marginTop:2 }}>10-agent LangGraph pipeline · {city} · {theme} · {budget.label}</div>
          </div>
          <div style={{ display:"flex", gap:8 }}>
            <button onClick={()=>loadInsights()} disabled={insLoading}
              style={{ padding:"7px 14px", borderRadius:7, border:`1px solid ${C.border}`,
                background:C.surface, color:insLoading?C.amber:C.muted, fontSize:11, cursor:"pointer", ...F.mono }}>
              {insLoading?"◈ Running...":"↺ Refresh Insights"}
            </button>
            <button onClick={handleShare} style={{ padding:"7px 14px", borderRadius:7,
              border:`1px solid ${C.indigo}60`, background:C.indigoLo, color:C.indigo, fontSize:11, cursor:"pointer", ...F.mono }}>
              ⎘ Share
            </button>
            {renderPDFBtn()}
            <button onClick={()=>setStep(3)} style={{ padding:"7px 14px", borderRadius:7, border:`1px solid ${C.border}`,
              background:C.surface, color:C.muted, fontSize:11, cursor:"pointer" }}>← Preview</button>
          </div>
        </div>

        {/* Before/After thumbnails — same 16/9 aspect ratio, unchanged */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:16 }}>
          {[{src:origUrl,lbl:"BEFORE"},{src:renUrl,lbl:"AFTER"}].map(img=>(
            <div key={img.lbl} style={{ position:"relative", borderRadius:10, overflow:"hidden",
              aspectRatio:"16/9", border:`1px solid ${C.border}` }}>
              <img src={img.src} alt={img.lbl} style={{ width:"100%", height:"100%", objectFit:"cover" }} />
              <div style={{ position:"absolute", top:8, left:8, fontSize:9, ...F.mono,
                background:"rgba(0,0,0,0.75)", color:C.muted, padding:"2px 8px", borderRadius:5 }}>{img.lbl}</div>
            </div>
          ))}
        </div>

        {/* Feedback panel — new, below thumbnails */}
        {(insights||result)&&<FeedbackPanel
          projectId={lastInsightPid.current||""}
          styleLabel={insights?.visual_analysis?.style_detected||vastuStyleLabel}
          // COST FIX: use result.design.total_inr — this is the same number shown
          // as TOTAL in the BOQ tab (construction + products, computed from line items).
          // Falls back to renovationCostInr (cost_estimate.total_inr from pipeline)
          // only if the BOQ total hasn't been computed yet.
          estimatedCostInr={result?.design?.total_inr||result?.renovationCostInr}
          city={city} roomType={roomKey} budgetTier={budget.tier}
        />}

        <AgentPipeline statuses={agentStatuses} timings={agentTimings} />

        {/* Tabs — added Vastu */}
        <div style={{ display:"flex", gap:4, marginBottom:16, borderBottom:`1px solid ${C.border}`, overflowX:"auto" }}>
          {[
            {id:"insights",    label:"✨ Pipeline Insights"},
            {id:"roi",         label:"📈 ROI Forecast"},
            {id:"design",      label:"📐 BOQ & Materials"},
            {id:"vastu",       label:"🪷 Vastu"},
            {id:"schedule",    label:"📋 CPM Schedule"},
            {id:"price",       label:"💹 Price Oracle"},
            {id:"products",    label:"🛍 Products"},
            {id:"contractors", label:"🏗 Contractors"},
          ].map(tab=>(
            <button key={tab.id} onClick={()=>setActiveTab(tab.id)}
              style={{ padding:"8px 16px", borderRadius:"8px 8px 0 0", border:"none", fontSize:11,
                cursor:"pointer", whiteSpace:"nowrap", transition:"all 0.15s",
                background:activeTab===tab.id?C.surface:"transparent",
                color:activeTab===tab.id?C.indigo:C.muted,
                borderBottom:activeTab===tab.id?`2px solid ${C.indigo}`:"2px solid transparent" }}>
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab==="insights"    &&<InsightsCard insights={insights} loading={insLoading} timings={agentTimings} boqTotal={result?.design?.total_inr||0}/>}
        {activeTab==="roi"         &&<ROICard roi={result?.roi} city={city} room={room} budgetTier={budget?.tier} budgetInr={budget?.inr}/>}
        {activeTab==="design"      &&<BOQCard design={result?.design} city={city} budgetTier={budget.tier}/>}
        {activeTab==="vastu"       &&<VastuPanel roomType={roomKey} styleLabel={vastuStyleLabel} detectedObjects={vastuDetectedObjects}/>}
        {activeTab==="schedule"    &&<ScheduleCard schedule={result?.schedule}/>}
        {activeTab==="price"       &&<PriceCard forecasts={priceData}/>}
        {activeTab==="products"    &&<ProductCard suggestions={result?.product_suggestions}/>}
        {activeTab==="contractors" &&<ContractorCard contractors={contractorList}/>}
      </motion.div>}

      {/* STEP 5: Chat — chatHist is top-level state, persists across tab switches */}
      {step===5&&<motion.div key="s5" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} exit={{opacity:0}}
        style={{ maxWidth:960, margin:"24px auto", padding:"0 28px",
          height:"calc(100vh - 110px)", display:"flex", flexDirection:"column" }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:12 }}>
          <div>
            <div style={{ fontSize:17, fontWeight:700 }}>Chat with ARKEN</div>
            <div style={{ fontSize:10, color:C.muted, ...F.mono, marginTop:3 }}>
              Gemini 2.5 Flash · sees both images · full pipeline context · {theme} · {city}
              {" · "}Press <kbd style={{ background:C.surface, border:`1px solid ${C.border}`,
                padding:"0 4px", borderRadius:3, fontSize:8 }}>Esc</kbd> to close
            </div>
          </div>
          <div style={{ display:"flex", gap:8 }}>
            <button onClick={()=>setStep(3)} style={{ padding:"6px 14px", borderRadius:7, border:`1px solid ${C.border}`,
              background:C.surface, color:C.muted, fontSize:11, cursor:"pointer" }}>← Preview</button>
            <button onClick={()=>setStep(4)} style={{ padding:"6px 14px", borderRadius:7, border:"none",
              background:"linear-gradient(135deg,#10b981,#059669)", color:"#fff", fontSize:11, cursor:"pointer" }}>Results →</button>
          </div>
        </div>
        <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:10 }}>
          {QUICK.map(q=>(
            <button key={q} onClick={()=>setChatInput(q)} style={{ padding:"5px 12px", borderRadius:6, border:`1px solid ${C.border}`,
              background:C.surface, color:C.muted, fontSize:10, cursor:"pointer", ...F.mono }}>{q}</button>
          ))}
        </div>
        <div style={{ flex:1, overflowY:"auto", display:"flex", flexDirection:"column",
          gap:8, padding:16, background:C.surface, borderRadius:12, border:`1px solid ${C.border}`, marginBottom:10 }}>
          {chatHist.length===0&&<div style={{ textAlign:"center", color:C.dim, fontSize:12, ...F.mono, margin:"auto", lineHeight:1.8 }}>
            I see both your original and renovated images.<br />
            I have full pipeline data: BOQ, ROI, materials, market insights.<br />
            I'll re-render automatically if you ask for design changes.
          </div>}
          {chatHist.map((msg,i)=><ChatMsg key={i} msg={msg}/>)}
          {chatLoading&&<motion.div animate={{opacity:[0.3,1,0.3]}} transition={{repeat:Infinity,duration:1.2}}
            style={{ alignSelf:"flex-start", padding:"10px 14px", background:C.card, borderRadius:12,
              border:`1px solid ${C.border}`, fontSize:12, color:C.muted, ...F.mono }}>
            ARKEN is thinking...
          </motion.div>}
          <div ref={chatEnd}/>
        </div>
        <div style={{ display:"flex", gap:10 }}>
          <input value={chatInput} onChange={e=>setChatInput(e.target.value)}
            onKeyDown={e=>e.key==="Enter"&&!e.shiftKey&&handleChat()}
            placeholder="Ask about materials, costs, or say 'make the walls darker'..."
            style={{ flex:1, padding:"12px 16px", borderRadius:10, border:`1px solid ${C.border}`,
              background:C.surface, color:C.text, fontSize:12, ...F.mono }}/>
          <button onClick={handleChat} disabled={chatLoading}
            style={{ padding:"12px 20px", borderRadius:10, border:"none",
              background:"linear-gradient(135deg,#6470f3,#a78bfa)", color:"#fff",
              fontSize:16, cursor:"pointer", fontWeight:700, opacity:chatLoading?0.5:1 }}>↑</button>
        </div>
        <div style={{ fontSize:9, color:C.dim, ...F.mono, textAlign:"center", marginTop:6 }}>
          Say "make walls darker" or "change to marble tiles" to trigger automatic re-render
        </div>
      </motion.div>}

      </AnimatePresence>

      {step>=3&&<div style={{ position:"fixed", bottom:0, left:0, right:0,
        borderTop:`1px solid ${C.border}`, padding:"8px 28px",
        display:"flex", alignItems:"center", justifyContent:"space-between",
        background:"rgba(5,8,15,0.97)", backdropFilter:"blur(8px)", zIndex:100 }}>
        <button onClick={()=>{
          setStep(0); setRenUrl(null); setRenB64(null);
          setInsights(null); setPipelineCtx(""); setChatHist([]);
          setResult(null); setPriceData(null); setContractorList([]);
          setAgentStatuses({}); setCompletedAgents([]);
          lastInsightPid.current=null;
        }} style={{ background:"none", border:"none", color:C.muted, fontSize:11, ...F.mono, cursor:"pointer" }}>
          ← Start over
        </button>
        <div style={{ display:"flex", gap:8 }}>
          {[{label:"Preview",color:C.indigo,target:3},{label:"Results",color:C.green,target:4},{label:"Chat",color:C.purple,target:5}].map(btn=>(
            <button key={btn.label} onClick={()=>setStep(btn.target)}
              style={{ padding:"6px 16px", borderRadius:7, border:"1px solid", fontSize:11, cursor:"pointer", transition:"all 0.15s",
                borderColor:step===btn.target?btn.color:C.border,
                background:step===btn.target?`${btn.color}18`:C.surface,
                color:step===btn.target?btn.color:C.muted }}>
              {btn.label}
            </button>
          ))}
        </div>
      </div>}
    </div>
  );
}
