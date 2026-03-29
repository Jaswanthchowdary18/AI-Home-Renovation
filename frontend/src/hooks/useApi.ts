/**
 * ARKEN — API Client & React Query Hooks
 * Type-safe API calls to the FastAPI backend.
 */

import axios, { AxiosInstance } from "axios";
import { useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useARKENStore } from "@/store/arken";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ── Axios instance ────────────────────────────────────────────────────────────

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 120_000,
  headers: { "Content-Type": "application/json" },
});

// Attach JWT token
apiClient.interceptors.request.use((config) => {
  const token =
    typeof window !== "undefined"
      ? localStorage.getItem("arken_token")
      : null;

  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Auto-logout on 401
apiClient.interceptors.response.use(
  (r) => r,
  async (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("arken_token");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

// ─────────────────────────────────────────────────────────────────────────────
// ANALYSIS API
// ─────────────────────────────────────────────────────────────────────────────

export interface AnalyzeParams {
  file: File;
  budgetInr: number;
  city: string;
  theme: string;
  budgetTier: string;
  roomType: string;
  projectName: string;
}

export const analyzeRoom = async (params: AnalyzeParams) => {
  const form = new FormData();
  form.append("file", params.file);
  form.append("budget_inr", String(params.budgetInr));
  form.append("city", params.city);
  form.append("theme", params.theme);
  form.append("budget_tier", params.budgetTier);
  form.append("room_type", params.roomType);
  form.append("project_name", params.projectName);

  const { data } = await apiClient.post("/analyze/", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return data;
};

export const pollTaskStatus = async (taskId: string) => {
  const { data } = await apiClient.get(`/analyze/status/${taskId}`);
  return data;
};

export const useAnalyze = () => {
  const store = useARKENStore();

  return useMutation({
    mutationFn: analyzeRoom,
    onSuccess: (data) => {
      store.setTaskId(data.task_id);
      store.setProjectId(data.project_id);
      store.setPipelineStatus("queued", 0, "Queued");
    },
    onError: () => {
      store.setPipelineStatus("failed", 0, "Error");
    },
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE POLLER (React Query v5 compatible)
// ─────────────────────────────────────────────────────────────────────────────

export const usePipelinePoller = (taskId: string | null) => {
  const store = useARKENStore();

  const query = useQuery({
    queryKey: ["pipeline", taskId],
    queryFn: () => pollTaskStatus(taskId!),
    enabled: !!taskId && ["queued", "running"].includes(store.pipelineStatus),

    refetchInterval: (query) => {
      const data = query.state.data as any;

      if (!data || ["complete", "failed"].includes(data?.status)) {
        return false;
      }

      return 2000;
    },
  });

  // ✅ React Query v5 way (instead of onSuccess)
  useEffect(() => {
    if (!query.data) return;

    const data: any = query.data;

    store.setPipelineStatus(
      data.status,
      data.progress_pct,
      data.current_step
    );

    if (data.status === "complete" && data.result) {
      store.setResult(data.result);
    }
  }, [query.data, store]);

  return query;
};

// ─────────────────────────────────────────────────────────────────────────────
// FORECAST API
// ─────────────────────────────────────────────────────────────────────────────

export const useMaterialForecasts = (horizonDays = 90) =>
  useQuery({
    queryKey: ["forecasts", horizonDays],
    queryFn: async () => {
      const { data } = await apiClient.get(
        `/forecast/materials?horizon_days=${horizonDays}`
      );
      return data.forecasts;
    },
    staleTime: 1000 * 60 * 60,
  });

export const useROIForecast = () =>
  useMutation({
    mutationFn: async (p: any) => {
      const { data } = await apiClient.post("/forecast/roi", p);
      return data;
    },
  });

// ─────────────────────────────────────────────────────────────────────────────
// CHAT API
// ─────────────────────────────────────────────────────────────────────────────

export const useChat = () => {
  const store = useARKENStore();

  return useMutation({
    mutationFn: async (userMessage: string) => {
      store.addChatMessage({ role: "user", content: userMessage });
      store.setChatLoading(true);

      const messages = store.chatMessages.map((m) => ({
        role: m.role === "system" ? "assistant" : m.role,
        content: m.content,
      }));

      messages.push({ role: "user", content: userMessage });

      const { data } = await apiClient.post("/chat/", {
        project_id: store.projectId || "demo",
        session_id: store.chatSessionId,
        messages,
      });

      return data;
    },
    onSuccess: (data) => {
      store.setChatLoading(false);
      store.addChatMessage({ role: "assistant", content: data.message });
      if (data.session_id) store.setChatSessionId(data.session_id);
    },
    onError: () => {
      store.setChatLoading(false);
      store.addChatMessage({
        role: "assistant",
        content:
          "Connection error. Please check your network and try again.",
      });
    },
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// RENDER API
// ─────────────────────────────────────────────────────────────────────────────

export const useRerender = () => {
  const store = useARKENStore();

  return useMutation({
    mutationFn: async (instructions: string) => {
      const { data } = await apiClient.post("/render/", {
        project_id: store.projectId,
        version: (store.result?.render?.version || 1) + 1,
        instructions,
        theme: store.theme,
        city: store.city,
        budget_tier: store.budgetTier,
      });

      return data;
    },
    onSuccess: (data) => {
      store.setRenderedImage(data.cdn_url);
    },
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// PROJECTS
// ─────────────────────────────────────────────────────────────────────────────

export const useProjects = () =>
  useQuery({
    queryKey: ["projects"],
    queryFn: async () => {
      const { data } = await apiClient.get("/projects/");
      return data;
    },
  });

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

export const formatInr = (value: number): string => {
  if (value >= 10_00_000) return `₹${(value / 10_00_000).toFixed(1)}Cr`;
  if (value >= 1_00_000) return `₹${(value / 1_00_000).toFixed(1)}L`;
  if (value >= 1000) return `₹${(value / 1000).toFixed(0)}K`;
  return `₹${value}`;
};
