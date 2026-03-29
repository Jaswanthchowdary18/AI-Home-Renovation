"""
ARKEN PropTech Engine v6.0 — FastAPI Backend
Multi-Agent AI Renovation Intelligence Platform.

v6.0 additions over v5.0:
  - Feature 4: /api/v1/feedback router (Prediction Accuracy Feedback)
    POST /api/v1/feedback/accuracy — user correction signal → PredictionLogger SQLite
    GET  /api/v1/feedback/accuracy/summary — aggregated accuracy stats
  - Startup: fine-tuned weight file checks with clear warnings + training instructions
  - graph_pipeline.startup_check_weights() called on lifespan boot
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routes import analyze, artifacts, auth, chat, forecast, projects, render
from api.routes import boq_sync
from api.routes import health as _health_router
# ── Feature 1: Product Suggester ──────────────────────────────────────────────
from api.routes import products
# ── Feature 3: Material Price Alerts ─────────────────────────────────────────
from api.routes import alerts
# ── Feature 4: Prediction Accuracy Feedback ──────────────────────────────────
from api.routes import feedback
from db.session import init_db
from services.cache import cache_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("arken")

# ── Module-level health state — set once at startup, read by health endpoint ──
# Prevents build_multi_agent_graph() from being called on every 30s health check.
_LANGGRAPH_OK: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀  ARKEN engine v6.0 starting (LangGraph + Fine-tuned models + Feedback)...")

    await init_db()
    await cache_service.connect()

    # ── Initialise agent memory ────────────────────────────────────────────
    try:
        from memory.agent_memory import agent_memory
        session_count = sum(len(v) for v in agent_memory._cache.values())
        logger.info(f"✅  Agent memory: {len(agent_memory._cache)} users, {session_count} sessions loaded")
    except Exception as e:
        logger.warning(f"Agent memory init skipped: {e}")

    # ── Fine-tuned weight file checks (v6.0) ──────────────────────────────
    # Logs clear warnings with training instructions for any missing weights.
    # Pipeline degrades gracefully to pretrained fallbacks if weights absent.
    try:
        from agents.graph_pipeline import startup_check_weights
        weight_status = startup_check_weights()
        n_present = sum(1 for v in weight_status.values() if v)
        n_total   = len(weight_status)
        if n_present == n_total:
            logger.info(f"✅  Fine-tuned weights: {n_present}/{n_total} present — full accuracy mode")
        else:
            missing = [k for k, v in weight_status.items() if not v]
            logger.warning(
                f"⚠️   Fine-tuned weights: {n_present}/{n_total} present. "
                f"Missing: {missing}. "
                "Pipeline using pretrained fallbacks for these models. "
                "See weight-check logs above for training commands."
            )
    except Exception as e:
        logger.warning(f"Weight check skipped: {e}")

    # ── Pre-compile LangGraph graph ────────────────────────────────────────
    global _LANGGRAPH_OK
    try:
        from agents.multi_agent_pipeline import build_multi_agent_graph
        graph = build_multi_agent_graph()
        if graph is not None:
            _LANGGRAPH_OK = True
            logger.info("✅  LangGraph multi-agent graph compiled (6 nodes)")
        else:
            logger.warning("⚠️   LangGraph unavailable — sequential fallback will be used")
    except Exception as e:
        logger.warning(f"LangGraph pre-compile skipped: {e}")

    # ── Ensure feedback SQLite table exists ───────────────────────────────
    try:
        from api.routes.feedback import _ensure_feedback_table
        _ensure_feedback_table()
        logger.info("✅  Feedback table: user_feedback + style_corrections ready in predictions.db")
    except Exception as e:
        logger.warning(f"Feedback table init skipped: {e}")

    cache_type = "in-memory" if cache_service.is_using_fallback else "redis"
    logger.info(f"✅  Cache: {cache_type}")
    logger.info("✅  All systems nominal (v6.0)")

    yield

    await cache_service.disconnect()
    logger.info("🛑  ARKEN engine stopped.")


app = FastAPI(
    title="ARKEN PropTech Engine",
    description=(
        "Multi-Agent AI Renovation Intelligence Platform — Indian Market (LangGraph v6.0). "
        "New in v6.0: Fine-tuned CV model pipeline (YOLO/CLIP/Style/Room), "
        "SHAP ROI explainability, IndiaMART price scraper, Prediction Accuracy Feedback."
    ),
    version="6.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://arken.in",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.perf_counter()-start)*1000:.1f}ms"
    return response


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth.router,           prefix="/api/v1/auth",      tags=["auth"])
app.include_router(projects.router,       prefix="/api/v1/projects",  tags=["projects"])
app.include_router(analyze.router,        prefix="/api/v1/analyze",   tags=["analyze"])
app.include_router(render.router,         prefix="/api/v1/render",    tags=["render"])
app.include_router(forecast.router,       prefix="/api/v1/forecast",  tags=["forecast"])
app.include_router(artifacts.router,      prefix="/api/v1/artifacts", tags=["artifacts"])
app.include_router(chat.router,           prefix="/api/v1/chat",      tags=["chat"])
app.include_router(_health_router.router, prefix="/api/v1/health",    tags=["health"])

# ── Feature 1: Product Suggester ──────────────────────────────────────────────
# POST  /api/v1/products/suggest       → shop-this-look from rendered image
# DELETE /api/v1/products/suggest/{id} → invalidate cache
app.include_router(products.router, prefix="/api/v1/products", tags=["products"])

# ── Feature 3: Material Price Alerts ─────────────────────────────────────────
# POST   /api/v1/alerts/                          → create alert
# GET    /api/v1/alerts/{user_id}                 → get user's alerts
# DELETE /api/v1/alerts/{alert_id}                → deactivate alert
# GET    /api/v1/alerts/smart-suggestions/{proj}  → AI-recommended alerts
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])

# ── Feature 4: Prediction Accuracy Feedback (v6.0) ───────────────────────────
# POST /api/v1/feedback/accuracy          → record user accuracy feedback
# GET  /api/v1/feedback/accuracy/summary  → aggregated accuracy stats
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])
# Genuine image-derived BOQ endpoint
app.include_router(boq_sync.router, prefix="/api/v1/boq", tags=["boq"])


@app.get("/", tags=["meta"])
async def root():
    return {
        "name":     "ARKEN PropTech Engine",
        "version":  "6.0.0",
        "pipeline": "LangGraph multi-agent (fine-tuned CV models)",
        "features": [
            "Product Suggester — shop-this-look from rendered images",
            "Contractor Network — real contractor links in CPM schedule",
            "Material Price Alerts — subscription alerts for price movements",
            "Prediction Accuracy Feedback — user corrections → drift monitoring",
            "Fine-tuned CV pipeline — YOLO indian rooms + CLIP + EfficientNet style",
            "SHAP ROI explainability — NHB Residex benchmark validation",
            "IndiaMART price scraper — live cement/steel/tiles/paints prices",
        ],
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
async def health():
    cache_type = "in-memory" if cache_service.is_using_fallback else "redis"

    # Use the module-level flag set once at startup — avoids recompiling
    # build_multi_agent_graph() on every 30s Docker health check, which was
    # flooding logs with "✅ LangGraph compiled" on every poll.
    langgraph_ok = _LANGGRAPH_OK

    # Fine-tuned weight status (cheap disk check, no compilation)
    weight_status: dict = {}
    try:
        from agents.graph_pipeline import WEIGHTS_DIR
        from pathlib import Path as _Path
        _wd = WEIGHTS_DIR
        weight_status = {
            "yolo_indian_rooms.pt":  (_wd / "yolo_indian_rooms.pt").exists(),
            "clip_finetuned.pt":     (_wd / "clip_finetuned.pt").exists(),
            "style_classifier.pt":   (_wd / "style_classifier.pt").exists(),
            "room_classifier.pt":    (_wd / "room_classifier.pt").exists(),
        }
    except Exception:
        pass

    return {
        "status":    "ok",
        "version":   "6.0.0",
        "engine":    "ARKEN",
        "cache":     cache_type,
        "degraded":  cache_service.is_using_fallback,
        "langgraph": "active" if langgraph_ok else "sequential_fallback",
        "agents": [
            "UserGoalAgent",
            "VisionAnalyzerAgent",
            "DesignPlannerAgent",
            "BudgetEstimatorAgent",
            "ROIAgent",
            "ReportAgent",
        ],
        "finetuned_weights": weight_status,
        "features": [
            "product_suggester",
            "contractor_network",
            "price_alerts",
            "prediction_feedback",
            "finetuned_cv_pipeline",
            "roi_shap_explainer",
            "indiamart_price_scraper",
        ],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)