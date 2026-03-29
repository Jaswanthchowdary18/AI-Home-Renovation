# ARKEN — AI Renovation Intelligence Platform

> **Production-grade Multi-Agent AI system for the Indian real estate market**  
> Transforms a single room photograph into a complete, data-driven renovation plan — with visual renders, itemised cost breakouts, ROI projections, material price forecasts, and a Vastu compliance report.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [ML Models & Performance](#ml-models--performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Indian Market Configuration](#indian-market-configuration)
- [Datasets](#datasets)
- [Production Deployment](#production-deployment)
- [Security](#security)
- [Monetisation Model](#monetisation-model)
- [Roadmap](#roadmap)

---

## Overview

ARKEN is a full-stack, AI-powered renovation intelligence platform built specifically for the Indian proptech market. A user uploads a photograph of any room — bedroom, kitchen, bathroom, or living area — selects a budget tier and city, and the platform autonomously:

1. Analyses the room's structural elements using computer vision
2. Generates a design plan with India-specific material recommendations
3. Produces a photorealistic rendered preview of the renovated space
4. Delivers a Bill of Quantities (BOQ) in INR with GST included
5. Forecasts ROI impact on property value
6. Predicts future material prices over a 30/60/90-day horizon
7. Provides a Vastu Shastra compliance assessment

The backend is built on **FastAPI v6.0** with a **LangGraph-orchestrated multi-agent pipeline**, a **Next.js 14** frontend, and a suite of custom-trained ML models fine-tuned on Indian property data.

---

## Key Features

**Computer Vision & Room Understanding**
- YOLOv8 segmentation trained on Indian room types to isolate walls, floors, ceilings, doors, and windows
- Fine-tuned CLIP (ViT-B/32, SigLIP loss) for interior style embedding and theme matching
- Perspective estimation for approximate room dimension inference

**Design Planning**
- SKU-level Bill of Quantities referencing real Indian brands: Asian Paints, Kajaria, Greenply, Havells, and more
- GST @ 18% automatically applied across all line items
- Budget-tier-aware recommendations (Basic: ₹3–5L / Mid: ₹5–10L / Premium: ₹10L+)

**AI Rendering**
- Mask-guided room inpainting via Gemini 2.5 Flash (primary) with Stability AI SDXL as fallback
- Geometry-preserving before/after output

**ROI & Price Forecasting**
- XGBoost ensemble (R² = 0.9992) trained on 32,210 Indian property transaction records
- Facebook Prophet time-series models for 11 material categories across 6 major cities
- Mean MAPE of ~4.4% for material price forecasts

**RAG-Powered Chat Agent**
- Retrieval-Augmented Generation over an Indian renovation knowledge base
- Supports conversational design iteration that can trigger re-renders

**Monitoring & Feedback Loop**
- Data drift monitoring, model retraining scheduler, and prediction accuracy feedback
- Structured logging with PII field masking

---

## Architecture
```
┌──────────────────────────────────────────────────────┐
│                    Next.js 14 Frontend                │
│  Upload Panel │ BOQ Table │ ROI Panel │ Vastu Panel  │
│  Before/After Slider │ Feedback Panel │ Chat Sidebar │
└────────────────────────┬─────────────────────────────┘
                         │  REST + SSE
┌────────────────────────▼─────────────────────────────┐
│              FastAPI Backend (v6.0)                   │
│  /analyze  /render  /forecast  /chat  /feedback       │
│  /products  /alerts  /artifacts  /health              │
└───────────┬──────────────┬──────────────┬────────────┘
            │              │              │
     ┌──────▼──────┐ ┌─────▼─────┐ ┌────▼────────┐
     │  LangGraph  │ │  Redis    │ │  PostgreSQL │
     │  Agents     │ │  Cache    │ │  (Async PG) │
     └──────┬──────┘ └───────────┘ └────────────┘
            │
     ┌──────▼──────────────────────────────┐
     │         ML Model Layer              │
     │  YOLOv8 │ CLIP │ XGBoost │ Prophet │
     └──────────────────────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  External APIs       │
                    │  Gemini 2.5 Flash    │
                    │  OpenAI GPT-4o       │
                    │  Stability AI SDXL   │
                    │  AWS S3              │
                    └──────────────────────┘
```

---

## Agent Pipeline

The core analysis is handled by a LangGraph-orchestrated multi-agent pipeline. Agents run in the following sequence, with ROI, price forecasting, and scheduling executing in parallel:
```
User uploads room image
         │
         ▼
┌─────────────────────┐
│   Visual Assessor   │  YOLOv8-seg → structural masks
│                     │  Fine-tuned CLIP → style embedding
│                     │  Perspective → dimension estimates
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Design Planner    │  Indian brand catalog lookup
│                     │  SKU-based BOQ generation (INR + GST)
│                     │  Theme & budget-tier alignment
└────────┬────────────┘
         │
    ─────┼──────────────────────────────
    │              │              │
    ▼              ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│   ROI    │ │  Price   │ │  Scheduler   │  (parallel)
│ XGBoost  │ │ Prophet  │ │  CPM + GANTT │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     │             │              │
     └─────────────┴──────┬───────┘
                          │
                          ▼
               ┌─────────────────────┐
               │   Rendering Agent   │  Gemini 2.5 Flash
               │                     │  Mask-guided inpainting
               └─────────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │   Report Agent      │  Consolidated renovation report
               │   + Vastu Checker   │  Vastu Shastra compliance
               └─────────────────────┘
```

**Supporting agents:** `InsightGenerationAgent`, `ProductSuggesterAgent`, `RAGRetrievalAgent`, `BudgetEstimatorAgent`, `UserGoalAgent`

---

## ML Models & Performance

### Material Price Forecasting — Facebook Prophet

11 material categories × 6 cities = **66 individually trained Prophet models**

Materials covered: Asian Paints Premium, Kajaria Tiles, Granite, Bricks, Cement OPC 53, River Sand, Steel TMT Fe500, Teak Wood, Copper Wire, PVC/UPVC Windows, Modular Kitchen

Average cross-validation MAPE: **~4.4%** (17 folds per model)

---

### Visual Understanding — Fine-tuned CLIP

| Metric | Value |
|---|---|
| Base model | CLIP ViT-B/32 |
| Loss | SigLIP (learnable temperature) |
| Training samples | 2,648 |
| Validation samples | 663 |
| Epochs | 15 |
| Best val loss | **0.1635** |
| Final image→text accuracy | **41.9%** |

---

## Tech Stack

**Backend**
- Python 3.11, FastAPI, Uvicorn
- LangGraph (multi-agent orchestration)
- SQLAlchemy (async) + AsyncPG + PostgreSQL
- Redis (caching, task queue)
- Alembic (database migrations)
- Pydantic v2 (settings & validation)

**ML / AI**
- YOLOv8 (Ultralytics) — room segmentation
- CLIP ViT-B/32 (OpenAI, fine-tuned) — style embedding
- XGBoost + Scikit-learn — ROI regression & quantile models
- Facebook Prophet — material price time-series forecasting
- Gemini 2.5 Flash (Google) — image rendering / inpainting
- GPT-4o (OpenAI) — chat agent
- Stability AI SDXL — rendering fallback
- FAISS / vector store — RAG retrieval

**Frontend**
- Next.js 14, TypeScript, React
- Zustand (global state)
- React Query (API data fetching)
- Tailwind CSS

**Infrastructure**
- Docker + Docker Compose (CUDA 12.1 multi-stage backend build)
- Nginx (rate limiting, SSL termination, reverse proxy)
- AWS S3 (image storage), CloudFront (CDN)
- AWS RDS PostgreSQL, ElastiCache Redis

---

## Project Structure
```
arken/
├── backend/
│   ├── main.py                          # FastAPI entrypoint (v6.0)
│   ├── config.py                        # Pydantic settings
│   ├── requirements.txt
│   ├── .env.example
│   ├── agents/
│   │   ├── orchestrator/
│   │   │   └── langgraph_orchestrator.py
│   │   ├── visual_assessor.py           # YOLOv8 + CLIP perception
│   │   ├── design_planner.py            # Material catalog & BOQ
│   │   ├── design_planner_node.py
│   │   ├── roi_forecast.py              # XGBoost property value model
│   │   ├── price_forecast.py            # Prophet material price model
│   │   ├── rendering.py                 # Gemini mask-guided inpainting
│   │   ├── coordinator.py               # CPM scheduling + risk register
│   │   ├── insight_generation_agent.py
│   │   ├── product_suggester_agent.py
│   │   ├── rag_retrieval_agent.py
│   │   ├── budget_estimator_agent.py
│   │   ├── user_goal_agent.py
│   │   ├── graph_pipeline.py            # LangGraph pipeline definition
│   │   ├── graph_state.py
│   │   ├── image_feature_schema.py
│   │   └── roi_agent_node.py
│   ├── api/routes/
│   │   ├── analyze.py                   # POST /analyze + status polling
│   │   ├── render.py                    # POST /render (re-render)
│   │   ├── forecast.py                  # GET /forecast/materials + ROI
│   │   ├── chat.py                      # POST /chat + SSE stream
│   │   ├── artifacts.py                 # Render & mask delivery
│   │   ├── products.py                  # Product suggester routes
│   │   ├── alerts.py                    # Material price alert routes
│   │   ├── feedback.py                  # Prediction accuracy feedback
│   │   ├── boq_sync.py                  # BOQ sync routes
│   │   ├── trust_report.py              # Trust & confidence report
│   │   ├── health.py                    # Health check endpoint
│   │   ├── auth.py                      # JWT authentication
│   │   └── projects.py                  # Project CRUD
│   ├── analytics/
│   │   ├── drift_monitor.py
│   │   ├── feedback_collector.py
│   │   └── model_evaluator.py
│   ├── services/
│   │   ├── rag/                         # RAG retriever & vector store
│   │   ├── insight_engine/              # Insight generation service
│   │   ├── monitoring/                  # Drift, retraining, prediction logging
│   │   ├── alerts/                      # Price change detector
│   │   ├── live_prices/                 # Live price fetcher
│   │   ├── datasets/                    # Dataset loader & ingestor
│   │   ├── llm.py                       # GPT-4o / Gemini LLM service
│   │   ├── storage.py                   # AWS S3 service
│   │   ├── cache.py                     # Redis service
│   │   ├── price_scraper.py
│   │   ├── report_generator.py
│   │   └── contractor_network.py
│   ├── ml/
│   │   └── weights/
│   │       ├── yolo_indian_rooms.pt     # Fine-tuned YOLOv8 (not in repo)
│   │       ├── clip_finetuned.pt        # Fine-tuned CLIP (not in repo)
│   │       ├── price_xgb.joblib
│   │       ├── roi_gbm.joblib
│   │       └── prophet_models/          # 66 Prophet .pkl files
│   ├── data/datasets/
│   │   ├── House Price India/
│   │   ├── Housing/
│   │   ├── india_housing_prices/        # City-level CSVs
│   │   ├── india_diy_knowledge/
│   │   ├── indian_renovation_knowledge/
│   │   ├── interior_design_images_metadata/
│   │   └── interior_design_material_style/
│   ├── db/
│   │   ├── session.py
│   │   └── models.py                    # 20+ ORM tables
│   ├── tests/
│   └── scripts/
│       ├── build_datasets.py
│       ├── build_rag_corpus.py
│       ├── evaluate_models.py
│       └── verify_cv_pipeline.py
├── frontend/
│   └── src/
│       ├── app/
│       ├── components/dashboard/
│       │   ├── ARKENDashboard.jsx
│       │   ├── BOQTable.jsx
│       │   ├── ROIPanel.jsx
│       │   ├── VastuPanel.jsx
│       │   ├── FeedbackPanel.jsx
│       │   ├── UploadPanel.jsx
│       │   └── ModelHealthBadge.jsx
│       ├── hooks/useApi.ts
│       └── store/arken.ts
├── ml/
│   └── train.py                         # Full training pipeline
├── ml_models_backup/
│   ├── model_report.json
│   ├── clip_training_report.json
│   ├── prophet_cv_report.json
│   ├── price_xgb.joblib
│   ├── roi_gbm.joblib
│   └── prophet_models/
├── infra/
│   ├── docker/
│   │   ├── backend.Dockerfile           # CUDA 12.1 multi-stage build
│   │   └── frontend.Dockerfile
│   └── nginx/nginx.conf
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose v2
- NVIDIA GPU + CUDA 12.1 drivers (required for YOLOv8 and rendering)
- Node.js 20+, Python 3.11+
- API keys (see [Environment Variables](#environment-variables))

### 1. Clone and Configure
```bash
git clone https://github.com/yourorg/arken.git
cd arken
cp backend/.env.example backend/.env
# Fill in your API keys in backend/.env
```

### 2. Place ML Model Weights

The trained model weights are not committed to this repository due to file size. Place them as follows:
```
backend/ml/weights/
├── yolo_indian_rooms.pt       # Fine-tuned YOLOv8 segmentation
├── clip_finetuned.pt          # Fine-tuned CLIP ViT-B/32
├── price_xgb.joblib
├── roi_gbm.joblib
└── prophet_models/            # 66 city × material Prophet .pkl files
```

To retrain from scratch:
```bash
cd backend
pip install -r requirements.txt
python ml/train.py --model all
```

### 3. Launch the Full Stack
```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Redis | localhost:6379 |
| PostgreSQL | localhost:5432 |

### 4. Run Database Migrations
```bash
cd backend
alembic upgrade head
```

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and populate the following:

| Variable | Provider | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Google AI Studio | Gemini 2.5 Flash rendering (primary) |
| `OPENAI_API_KEY` | OpenAI | GPT-4o chat agent |
| `STABILITY_API_KEY` | Stability AI | SDXL inpainting fallback |
| `AWS_ACCESS_KEY_ID` | AWS | S3 image storage |
| `AWS_SECRET_ACCESS_KEY` | AWS | S3 image storage |
| `AWS_S3_BUCKET` | AWS | S3 bucket name |
| `DATABASE_URL` | — | AsyncPG PostgreSQL connection string |
| `REDIS_URL` | — | Redis connection string |
| `JWT_SECRET` | — | RS256 JWT signing key |

---

## API Reference

### Start Analysis
```
POST /api/v1/analyze/
Content-Type: multipart/form-data

Fields:
  file         Room image (JPG / PNG / HEIC, max 20 MB)
  budget_inr   Integer — 300,000 to 5,000,000
  city         String  — e.g. "Hyderabad"
  theme        String  — e.g. "Modern Minimalist"
  budget_tier  "basic" | "mid" | "premium"
  room_type    "bedroom" | "kitchen" | "bathroom" | "living_room"

Response: { project_id, task_id, status: "queued" }
```

### Poll Analysis Status
```
GET /api/v1/analyze/status/{task_id}

Response:
{
  status:       "running" | "complete" | "failed",
  progress_pct: 0–100,
  current_step: "Visual Assessor" | "Design Planner" | ...,
  result:       { visual, design, roi, price_forecast, schedule, render }
}
```

### Material Price Forecasts
```
GET /api/v1/forecast/materials?horizon_days=90

Response:
{
  forecasts: [{
    material_key, current_price_inr,
    forecast_30d_inr, forecast_60d_inr, forecast_90d_inr,
    volatility_label, trend
  }]
}
```

### Chat (Streaming)
```
POST /api/v1/chat/
{ project_id, messages: [{ role, content }] }

Response: { message, action, triggers_rerender }
```

### Submit Prediction Feedback
```
POST /api/v1/feedback/accuracy
{ project_id, predicted_value, actual_value, feedback_type }

GET /api/v1/feedback/accuracy/summary
```

Full interactive docs available at `/docs` when the backend is running.

---

## Indian Market Configuration

### Material Catalog

| Category | Brands |
|---|---|
| Paint | Asian Paints (Royale / Apcolite), Berger, Nerolac, Dulux |
| Tiles | Kajaria, Somany, Nitco, RAK Ceramics, Simpolo |
| Plywood | Greenply, Century, Kitply |
| Hardware & Electrical | Havells, Legrand, Philips, Anchor, Schneider |


## Datasets

| Dataset | Description | Location |
|---|---|---|
| India Housing Prices | City-level property data (BLR, CHN, DEL, HYD, KOL, MUM) | `data/datasets/india_housing_prices/` |
| House Rent Dataset | 32,210+ Indian property transaction records | `data/datasets/House Price India/` |
| Housing (generic) | Supplementary housing features | `data/datasets/Housing/` |
| Indian Renovation Knowledge | RAG corpus — costs, materials, best practices | `data/datasets/indian_renovation_knowledge/` |
| India DIY Knowledge | Consumer-facing renovation guidance | `data/datasets/india_diy_knowledge/` |
| Interior Design Images Metadata | Train/val/test splits for visual model | `data/datasets/interior_design_images_metadata/` |
| Interior Design Material & Style | Material and style classification metadata | `data/datasets/interior_design_material_style/` |

> Large files and trained model weights are not committed to this repository. See [Quick Start](#quick-start) for instructions on obtaining or retraining them.

---

### Deployment Steps
```bash
# Build and push backend image to ECR
aws ecr create-repository --repository-name arken-backend --region ap-south-1
docker build -f infra/docker/backend.Dockerfile -t arken-backend ./backend
docker tag arken-backend:latest <ecr-uri>/arken-backend:latest
docker push <ecr-uri>/arken-backend:latest

# Run database migrations
alembic upgrade head

# Upload trained model weights to S3
aws s3 cp backend/ml/weights/ s3://arken-artifacts/weights/ --recursive
```

### Scalability Targets

- ECS auto-scaling on API layer (CPU threshold: 70%)
- Separate GPU worker pool for rendering (Redis-backed task queue)
- PostgreSQL read replicas for analytics workloads
- CloudFront CDN for rendered image delivery
- SageMaker endpoints for XGBoost and Prophet model serving
- **Target at launch:** 1,000 concurrent users, 500 renders/day

---

## Security

- JWT RS256 tokens with refresh rotation
- AES-256 encryption at rest (S3 SSE)
- TLS 1.3 in transit
- Role-based access control: `user` / `pro` / `enterprise` / `admin`
- GDPR-compliant: data export and deletion endpoints provided
- Indian IT Act 2000, Section 43A compliant
- No PII in logs (structured logging with field masking)

See `backend/SECURITY.md` for the full security policy.

---

## Monetisation Model

| Plan | Price | Features |
|---|---|---|
| Free | ₹0 / month | 2 analyses/month, basic themes |
| Professional | ₹999 / month | 20 analyses, all themes, PDF export |
| Business | ₹2,999 / month | Unlimited analyses, contractor network, API access |
| Enterprise | Custom | White-label, custom supplier integration |

---

## Roadmap

| Phase | Target | Feature |
|---|---|---|
| Phase 1 | Launch | Static Indian material catalog with direct links |
| Phase 2 | Q2 2026 | IndiaMART API integration for live pricing |
| Phase 3 | Q3 2026 | Rajkot tile exchange + BuildSupply partnership |
| Phase 4 | Q4 2026 | Urban Company contractor network API + booking flow |

---

*Built for Bharat's renovation market.*
