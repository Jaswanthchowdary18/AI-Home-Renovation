# ARKEN PropTech Engine — Complete Deployment Guide

> Production-grade Multi-Agent AI Renovation Intelligence Platform  
> Optimized for Indian Real Estate Market | v2.1.0

---

## 📁 Project Structure

```
arken/
├── backend/                    # FastAPI Python backend
│   ├── main.py                 # App entrypoint
│   ├── config.py               # Pydantic settings
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment template
│   ├── agents/
│   │   ├── orchestrator.py     # CrewAI-style multi-agent pipeline
│   │   ├── visual_assessor.py  # YOLOv8-seg + CLIP perception
│   │   ├── design_planner.py   # Indian material catalog + BOQ
│   │   ├── roi_forecast.py     # XGBoost property value regression
│   │   ├── price_forecast.py   # Prophet material price time-series
│   │   ├── rendering.py        # Gemini 2.5 mask-guided inpainting
│   │   └── coordinator.py      # CPM scheduling + risk register
│   ├── api/routes/
│   │   ├── analyze.py          # POST /analyze + status polling
│   │   ├── render.py           # POST /render (re-render)
│   │   ├── forecast.py         # GET /forecast/materials + ROI
│   │   ├── chat.py             # POST /chat + SSE stream
│   │   ├── auth.py             # JWT auth endpoints
│   │   ├── projects.py         # CRUD projects
│   │   └── artifacts.py        # Render + mask delivery
│   ├── db/
│   │   ├── session.py          # AsyncPG SQLAlchemy setup
│   │   └── models.py           # Full ORM schema (20+ tables)
│   └── services/
│       ├── storage.py          # AWS S3 service
│       ├── cache.py            # Redis service
│       └── llm.py              # GPT-4o / Gemini chat service
├── frontend/                   # Next.js 14 TypeScript frontend
│   ├── package.json
│   ├── src/
│   │   ├── app/page.tsx        # Main dashboard page
│   │   ├── store/arken.ts      # Zustand global state
│   │   ├── hooks/useApi.ts     # React Query API hooks
│   │   └── components/
│   │       ├── dashboard/      # Topbar, Sidebar, Canvas, Tabs
│   │       ├── preview/        # Before/After slider
│   │       └── chat/           # Chat sidebar
├── ml/
│   ├── train.py                # XGBoost + Prophet training pipeline
│   └── weights/                # Model artifacts (gitignored)
├── infra/
│   ├── docker/
│   │   ├── backend.Dockerfile  # CUDA 12.1 multi-stage build
│   │   └── frontend.Dockerfile # Next.js standalone build
│   └── nginx/
│       └── nginx.conf          # Rate limiting + SSL + proxy
├── docker-compose.yml          # Full stack compose
└── docs/                       # This file + architecture diagrams
```

---

## ⚡ Quick Start (Local Dev)

### Prerequisites
- Docker + Docker Compose v2
- NVIDIA GPU + CUDA 12.1 drivers (for YOLOv8 / rendering)
- Node.js 20+, Python 3.11+

### 1. Clone & Configure

```bash
git clone https://github.com/yourorg/arken.git
cd arken
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys
```

### 2. Download ML Model Weights

```bash
# YOLOv8 segmentation model
cd backend
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8x-seg.pt')"
mv ~/.ultralytics/assets/yolov8x-seg.pt ml/weights/

# SAM checkpoint (optional, for precise masks)
wget -P ml/weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 3. Train ML Models

```bash
cd backend
pip install -r requirements.txt
python ml/train.py --model all
# Saves: ml/weights/roi_xgb.pkl + ml/weights/prophet/*.pkl
```

### 4. Launch Full Stack

```bash
docker-compose up --build
```

Services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Redis: localhost:6379
- PostgreSQL: localhost:5432

---

## 🔑 Required API Keys

Set in `backend/.env`:

| Variable | Provider | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Google AI Studio | Gemini 2.5 Flash rendering (primary) |
| `OPENAI_API_KEY` | OpenAI | GPT-4o chat agent |
| `STABILITY_API_KEY` | Stability AI | SDXL inpainting fallback |
| `AWS_ACCESS_KEY_ID` | AWS | S3 image storage |
| `AWS_SECRET_ACCESS_KEY` | AWS | S3 image storage |

---

## 🏗 Agent Pipeline Flow

```
User uploads room image
         │
         ▼
┌─────────────────────┐
│  Visual Assessor    │  YOLOv8-seg → wall/floor/ceiling masks
│  (GPU required)     │  CLIP → style embedding
│                     │  Perspective → dimension estimates
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Design Planner     │  Indian catalog: Asian Paints, Kajaria, Greenply
│                     │  SKU-based BOQ in INR
│                     │  GST @18% included
└────────┬────────────┘
         │
    ─────┼─────────────────────────
    │         │           │
    ▼         ▼           ▼
┌───────┐ ┌───────┐ ┌──────────┐
│  ROI  │ │Price  │ │Scheduler │  (parallel)
│XGBoost│ │Prophet│ │CPM/GANTT │
└───────┘ └───────┘ └──────────┘
    │         │           │
    └─────────┴─────┬─────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Rendering Agent    │  Gemini 2.5 Flash
         │                     │  Mask-guided inpainting
         │                     │  Geometry preservation
         └─────────────────────┘
                    │
                    ▼
         Complete Renovation Report
```

---

## 🇮🇳 Indian Market Configuration

### Material Catalog
- **Paint**: Asian Paints (Royale/Apcolite), Berger, Nerolac, Dulux
- **Tiles**: Kajaria, Somany, Nitco, RAK Ceramics, Simpolo
- **Plywood**: Greenply, Century, Kitply
- **Hardware**: Havells, Legrand, Philips, Anchor, Schneider

### City Coverage
Tier 1: Mumbai, Delhi NCR, Bangalore, Hyderabad, Chennai, Pune, Kolkata  
Tier 2: Ahmedabad, Surat, Jaipur, Lucknow, Chandigarh, Nagpur  
Tier 3: Bhopal, Indore, and 50+ more

### Budget Tiers (INR)
- Basic: ₹3–5 Lakh
- Mid: ₹5–10 Lakh  
- Premium: ₹10 Lakh+

---

## 📊 API Reference

### Start Analysis
```
POST /api/v1/analyze/
Content-Type: multipart/form-data

Fields:
  file         - Room image (JPG/PNG/HEIC, max 20MB)
  budget_inr   - Integer, 300000–5000000
  city         - String (e.g. "Hyderabad")
  theme        - String (e.g. "Modern Minimalist")
  budget_tier  - "basic" | "mid" | "premium"
  room_type    - "bedroom" | "kitchen" | "bathroom" | "living_room"

Response: { project_id, task_id, status: "queued" }
```

### Poll Status
```
GET /api/v1/analyze/status/{task_id}

Response: {
  status: "running" | "complete" | "failed",
  progress_pct: 0–100,
  current_step: "Visual Assessor" | ...,
  result: { visual, design, roi, price_forecast, schedule, render }
}
```

### Material Forecasts
```
GET /api/v1/forecast/materials?horizon_days=90

Response: { forecasts: [{ material_key, current_price_inr, forecast_30d_inr, 
            forecast_60d_inr, forecast_90d_inr, volatility_label, trend }] }
```

### Chat
```
POST /api/v1/chat/
{ project_id, messages: [{role, content}] }

Response: { message, action, triggers_rerender }
```

---

## 🚀 Production Deployment (AWS)

### Infrastructure (estimated costs — Indian region ap-south-1)

| Component | Service | Monthly Cost |
|---|---|---|
| Backend API | EC2 g4dn.xlarge (GPU) | ₹28,000 |
| Database | RDS PostgreSQL t3.medium | ₹6,500 |
| Cache | ElastiCache Redis t3.micro | ₹2,200 |
| Storage | S3 + CloudFront | ₹1,500/TB |
| Frontend | Vercel / Amplify | ₹2,000 |
| **Total** | | **~₹40,000/month** |

### Deployment Steps

```bash
# 1. Build and push Docker images to ECR
aws ecr create-repository --repository-name arken-backend --region ap-south-1
docker build -f infra/docker/backend.Dockerfile -t arken-backend ./backend
docker tag arken-backend:latest <ecr-uri>/arken-backend:latest
docker push <ecr-uri>/arken-backend:latest

# 2. Deploy to ECS Fargate with GPU task definition
# See infra/k8s/ for Kubernetes manifests

# 3. Run database migrations
alembic upgrade head

# 4. Train and upload ML models to S3
python ml/train.py --model all
aws s3 cp ml/weights/ s3://arken-artifacts/weights/ --recursive
```

---

## 💰 Monetization Model

| Plan | Price | Features |
|---|---|---|
| Free | ₹0 | 2 analyses/month, basic themes |
| Professional | ₹999/month | 20 analyses, all themes, PDF export |
| Business | ₹2,999/month | Unlimited, contractor network, API access |
| Enterprise | Custom | White-label, custom supplier integration |

---

## 🔐 Security Compliance

- JWT RS256 tokens with refresh rotation
- AES-256 encryption at rest (S3 SSE)
- TLS 1.3 in transit
- RBAC: user / pro / enterprise / admin
- GDPR: data export + deletion endpoints
- Indian IT Act 2000 Section 43A compliant
- No PII in logs (structured logging with field masking)

---

## 📈 Scalability Plan

- **Horizontal**: ECS auto-scaling on API (CPU > 70%)
- **Rendering**: Separate GPU task queue (Redis + worker pool)
- **Database**: Read replicas for analytics queries
- **CDN**: CloudFront for rendered images (global edge)
- **Model serving**: SageMaker endpoints for XGBoost / Prophet
- **Target**: 1,000 concurrent users, 500 renders/day at launch

---

## 🤝 Indian Supplier Integration Roadmap

**Phase 1 (Launch)**: Static catalog with direct links  
**Phase 2 (Q2)**: IndiaMART API integration for live pricing  
**Phase 3 (Q3)**: Rajkot tile exchange, BuildSupply partnership  
**Phase 4 (Q4)**: Urban Company contractor network API + booking flow

---

*Built with ❤️ for Bharat's renovation market.*
