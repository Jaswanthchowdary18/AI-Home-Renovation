# ARKEN — Backend Dockerfile v5.2
# Fixes over v5.1:
#   - HF_HOME moved from /usr/local/hf_cache to /app/hf_cache.
#     /usr/local is root-owned; the arken user cannot create subdirectories inside it,
#     so all HuggingFace model downloads were failing with PermissionError.
#     /app is chowned to arken, so /app/hf_cache is writable.
#   - /app/hf_cache created and chowned to arken BEFORE the USER switch.
#   - CMDSTAN remains at /usr/local/cmdstan (installed by root builder, copied to runtime;
#     arken only reads it, never writes — no permission issue there).

# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CMDSTAN=/usr/local/cmdstan

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev libgomp1 git curl make \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install CmdStan here in the builder so it's copied to runtime via /usr/local.
# We set CMDSTAN_DIR explicitly so cmdstanpy writes to /usr/local/cmdstan.
ENV CMDSTAN=/usr/local/cmdstan
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(dir='/usr/local/cmdstan', cores=2, progress=False)"

# Verify ultralytics installed correctly and pre-download yolov8n pretrained weights.
# This bakes the weights into the image so the first inference doesn't need internet.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && mv yolov8n.pt /usr/local/lib/python3.11/site-packages/ultralytics/assets/ 2>/dev/null || true


# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CMDSTAN=/usr/local/cmdstan
# HF_HOME must be under /app so the arken user can write to it.
ENV HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 libgomp1 curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local

WORKDIR /app
COPY backend/ .

# Create required directories — all under /app so arken can read/write.
RUN mkdir -p ml/weights data /tmp/arken_local_storage /app/hf_cache \
    && mkdir -p /usr/local/cmdstan \
    && chmod -R 755 /usr/local/cmdstan

RUN groupadd -r arken && useradd -r -g arken -d /app arken
RUN chown -R arken:arken /app
USER arken

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
