/**
 * ARKEN — Zustand Global Store
 * Central state for renovation project, pipeline status, and UI.
 */

import { create } from "zustand";
import { devtools, persist, subscribeWithSelector } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

// ── Types ─────────────────────────────────────────────────────────────────────

export type BudgetTier = "basic" | "mid" | "premium";
export type PipelineStatus = "idle" | "queued" | "running" | "complete" | "failed";

export interface VisualResult {
  spatial_map: {
    dimensions: {
      estimated_length_ft: number;
      estimated_width_ft: number;
      estimated_height_ft: number;
      wall_area_sqft: number;
      floor_area_sqft: number;
    };
    detected_objects: Record<string, number[]>;
    masks_s3: Record<string, string>;
  };
  material_quantities: {
    paint_liters: number;
    floor_tiles_sqft: number;
    plywood_sqft: number;
  };
  style: {
    tags: string[];
  };
}

export interface ROIResult {
  city: string;
  city_tier: number;
  pre_reno_value_inr: number;
  post_reno_value_inr: number;
  equity_gain_inr: number;
  roi_pct: number;
  rental_yield_delta: number;
  payback_months: number;
  model_confidence: number;
}

export interface CostPlan {
  total_inr: number;
  material_inr: number;
  labour_inr: number;
  gst_inr: number;
  contingency_inr: number;
  line_items: LineItem[];
  supplier_recommendations: Supplier[];
}

export interface LineItem {
  category: string;
  brand: string;
  product: string;
  sku: string;
  qty: number;
  unit: string;
  rate_inr: number;
  total_inr: number;
}

export interface Supplier {
  name: string;
  type: string;
  url: string;
}

export interface MaterialForecast {
  material_key: string;
  unit: string;
  current_price_inr: number;
  forecast_30d_inr: number;
  forecast_60d_inr: number;
  forecast_90d_inr: number;
  volatility_score: number;
  volatility_label: "Low" | "Medium" | "High";
  trend: "up" | "down" | "stable";
  pct_change_90d: number;
}

export interface ScheduleTask {
  id: string;
  name: string;
  start_day: number;
  end_day: number;
  duration_days: number;
  contractor_role: string;
  is_critical: boolean;
  start_date: string;
  end_date: string;
}

export interface Risk {
  factor: string;
  probability: number;
  impact: "Low" | "Medium" | "High";
  mitigation: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  triggersRerender?: boolean;
}

export interface PipelineResult {
  visual: VisualResult;
  design: CostPlan;
  roi: ROIResult;
  price_forecast: MaterialForecast[];
  schedule: {
    total_days: number;
    critical_path_days: number;
    tasks: ScheduleTask[];
    risk_score: number;
    risks: Risk[];
    contractor_list: any[];
  };
  render: {
    cdn_url: string;
    model_used: string;
    generation_time_ms: number;
    version: number;
  };
}

// ── Store interface ───────────────────────────────────────────────────────────

interface ARKENStore {
  // Project config
  projectId: string | null;
  projectName: string;
  uploadedImageUrl: string | null;
  uploadedImageFile: File | null;
  renderedImageUrl: string | null;

  // User settings
  budgetInr: number;
  budgetTier: BudgetTier;
  city: string;
  theme: string;
  roomType: string;

  // Pipeline
  pipelineStatus: PipelineStatus;
  pipelineProgress: number;
  pipelineStep: string;
  taskId: string | null;
  result: PipelineResult | null;

  // Chat
  chatMessages: ChatMessage[];
  chatSessionId: string | null;
  isChatLoading: boolean;

  // UI
  activeTab: string;
  sidebarOpen: boolean;

  // Actions
  setProjectName: (name: string) => void;
  setUploadedImage: (file: File, url: string) => void;
  setRenderedImage: (url: string) => void;
  setBudget: (amount: number) => void;
  setBudgetTier: (tier: BudgetTier) => void;
  setCity: (city: string) => void;
  setTheme: (theme: string) => void;
  setRoomType: (type: string) => void;

  setPipelineStatus: (status: PipelineStatus, progress?: number, step?: string) => void;
  setTaskId: (id: string) => void;
  setProjectId: (id: string) => void;
  setResult: (result: PipelineResult) => void;

  addChatMessage: (msg: Omit<ChatMessage, "id" | "timestamp">) => void;
  setChatLoading: (loading: boolean) => void;
  setChatSessionId: (id: string) => void;

  setActiveTab: (tab: string) => void;
  reset: () => void;
}

const initialState = {
  projectId: null,
  projectName: "Untitled Project",
  uploadedImageUrl: null,
  uploadedImageFile: null,
  renderedImageUrl: null,
  budgetInr: 650_000,
  budgetTier: "mid" as BudgetTier,
  city: "Hyderabad",
  theme: "Modern Minimalist",
  roomType: "bedroom",
  pipelineStatus: "idle" as PipelineStatus,
  pipelineProgress: 0,
  pipelineStep: "",
  taskId: null,
  result: null,
  chatMessages: [
    {
      id: "init",
      role: "system" as const,
      content: "ARKEN Supervisor active. Upload a room image to begin intelligent renovation planning.",
      timestamp: new Date(),
    },
  ],
  chatSessionId: null,
  isChatLoading: false,
  activeTab: "overview",
  sidebarOpen: true,
};

export const useARKENStore = create<ARKENStore>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get) => ({
          ...initialState,

          setProjectName: (name) => set((s) => { s.projectName = name; }),
          setUploadedImage: (file, url) => set((s) => {
            s.uploadedImageFile = file;
            s.uploadedImageUrl = url;
            s.renderedImageUrl = null;
            s.result = null;
            s.pipelineStatus = "idle";
          }),
          setRenderedImage: (url) => set((s) => { s.renderedImageUrl = url; }),
          setBudget: (amount) => set((s) => {
            s.budgetInr = amount;
            s.budgetTier = amount < 500_000 ? "basic" : amount < 1_000_000 ? "mid" : "premium";
          }),
          setBudgetTier: (tier) => set((s) => { s.budgetTier = tier; }),
          setCity: (city) => set((s) => { s.city = city; }),
          setTheme: (theme) => set((s) => { s.theme = theme; }),
          setRoomType: (type) => set((s) => { s.roomType = type; }),

          setPipelineStatus: (status, progress, step) => set((s) => {
            s.pipelineStatus = status;
            if (progress !== undefined) s.pipelineProgress = progress;
            if (step !== undefined) s.pipelineStep = step;
          }),
          setTaskId: (id) => set((s) => { s.taskId = id; }),
          setProjectId: (id) => set((s) => { s.projectId = id; }),
          setResult: (result) => set((s) => {
            s.result = result;
            s.pipelineStatus = "complete";
            s.pipelineProgress = 100;
            if (result.render?.cdn_url) {
              s.renderedImageUrl = result.render.cdn_url;
            }
          }),

          addChatMessage: (msg) => set((s) => {
            s.chatMessages.push({
              ...msg,
              id: Date.now().toString(),
              timestamp: new Date(),
            });
          }),
          setChatLoading: (loading) => set((s) => { s.isChatLoading = loading; }),
          setChatSessionId: (id) => set((s) => { s.chatSessionId = id; }),

          setActiveTab: (tab) => set((s) => { s.activeTab = tab; }),
          reset: () => set(() => ({ ...initialState, chatMessages: initialState.chatMessages })),
        }))
      ),
      {
        name: "arken-store",
        partialize: (s) => ({
          city: s.city,
          theme: s.theme,
          budgetInr: s.budgetInr,
          budgetTier: s.budgetTier,
          projectName: s.projectName,
        }),
      }
    )
  )
);
