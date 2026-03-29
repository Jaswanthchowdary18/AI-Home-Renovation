# ARKEN — Frontend Dockerfile v2.4 (Next.js 14)
# context = project root (.), so prefix all host paths with frontend/
# No public/ folder in this project — removed that COPY entirely

# ── Stage 1: Dependencies ────────────────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app

# context is project root, package.json lives in frontend/
# Note: only copying package.json (no lockfile) — npm install generates a fresh one.
# --legacy-peer-deps prevents peer dependency conflicts in the large dep tree.
COPY frontend/package.json ./
RUN npm install --legacy-peer-deps


# ── Stage 2: Builder ─────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ .

ARG NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
ARG NEXT_PUBLIC_GEMINI_KEY

ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_GEMINI_KEY=${NEXT_PUBLIC_GEMINI_KEY}
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build


# ── Stage 3: Runner ──────────────────────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs \
 && adduser --system --uid 1001 nextjs

COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000

CMD ["node", "server.js"]
