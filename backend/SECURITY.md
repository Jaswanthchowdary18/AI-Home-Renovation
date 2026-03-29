# ARKEN — Security Guide

## ⚠️ Immediate Action Required

A Google API key (`AIzaSyBNKEx...`) was **exposed in this repository's git history**.
This key must be revoked and replaced. Follow the steps in [Rotating the Google API Key](#rotating-the-google-api-key) immediately.

---

## 1. Environment Variables That Must Never Be Committed

The following variables contain secrets that must **only** appear in `.env` (which is gitignored), never in `.env.example`, source code, or CI logs.

| Variable | Description | Risk if Exposed |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini Vision + Chat API | Unauthorized AI API usage, billing fraud |
| `SECRET_KEY` | JWT signing key | Session hijacking, auth bypass |
| `AWS_SECRET_ACCESS_KEY` | S3 access | Data exfiltration, storage deletion |
| `DATABASE_URL` | PostgreSQL credentials | Full database access |
| `REDIS_URL` | Redis credentials | Cache poisoning, data access |
| `OPENAI_API_KEY` | OpenAI API | Billing fraud, data leakage |
| `STABILITY_API_KEY` | Stability AI | Billing fraud |
| `PINECONE_API_KEY` | Vector database | Knowledge base exfiltration |
| `AWS_ACCESS_KEY_ID` | AWS identity | Combined with secret = full AWS access |

### `.gitignore` verification

Ensure these patterns are in your root `.gitignore`:
```
.env
.env.local
.env.production
*.env
backend/.env
```

---

## 2. Generating a Secure `SECRET_KEY`

The `SECRET_KEY` is used for JWT signing. A weak or default key allows attackers to forge authentication tokens.

```bash
# Generate a cryptographically secure 256-bit key
openssl rand -hex 32
```

Copy the output to your `.env`:
```bash
SECRET_KEY=<paste output here>
```

**Never** use the placeholder values:
- `arken-dev-secret-key-change-in-production-32ch` ← insecure default
- `your-super-secret-key-change-in-production-min-32-chars` ← placeholder
- `GENERATE_WITH_openssl_rand_hex_32` ← unset placeholder

---

## 3. Rotating the Google API Key

### Why This Is Urgent

The key `AIzaSyBNKExIcDRRsZ0WUgxBVoRBbvegeMs2IHM` was committed to a git repository. Even if the commit was deleted, it may have been indexed by GitHub, scraped by bots, or cached in CI logs. **Assume it is compromised.**

### Rotation Steps

1. **Open Google Cloud Console**
   - Navigate to: https://console.cloud.google.com/apis/credentials
   - Sign in with the account that owns this key.

2. **Revoke the old key**
   - Find the key starting with `AIzaSyBNKEx...`
   - Click **Delete** or **Restrict** → then **Revoke**.

3. **Create a new API key**
   - Click **+ Create Credentials** → **API Key**
   - Under **API restrictions**, restrict to:
     - Generative Language API (Gemini)
     - Cloud Vision API (if used)
   - Under **Application restrictions**, add your server's IP address(es).

4. **Update your `.env`**
   ```bash
   GOOGLE_API_KEY=<your-new-key>
   ```

5. **Update secret management** (if using AWS Secrets Manager, GCP Secret Manager, or Doppler)
   - Update the secret value in your secret manager.
   - Restart the application to pick up the new key.

6. **Verify the old key is dead**
   ```bash
   curl "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyBNKExIcDRRsZ0WUgxBVoRBbvegeMs2IHM"
   # Expected: 400 API_KEY_INVALID or 403 PERMISSION_DENIED
   ```

7. **Run the rotation checker** to confirm your `.env` is clean:
   ```bash
   python scripts/rotate_secrets.py
   ```

---

## 4. Running `rotate_secrets.py` as a Pre-Commit Hook

The `scripts/rotate_secrets.py` script scans for known-compromised keys before every commit.

### Setup

```bash
# From the repository root:
echo '#!/bin/sh
cd backend && python scripts/rotate_secrets.py --pre-commit
' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The hook exits with code `1` (blocking the commit) if any CRITICAL key is found.

### Manual run

```bash
# Check .env
python scripts/rotate_secrets.py

# Check all .env* files
python scripts/rotate_secrets.py --all-files

# Pre-commit mode (exits 1 on critical find)
python scripts/rotate_secrets.py --pre-commit

# Check a specific file
python scripts/rotate_secrets.py --env-file /path/to/.env
```

---

## 5. API Rate Limiting Strategy

ARKEN implements rate limiting at multiple layers.

### Application-Level (FastAPI)

Configured via:
```env
API_RATE_LIMIT_PER_MINUTE=30
RENDER_RATE_LIMIT=5
```

The `RATE_LIMIT_PER_MINUTE=30` setting is applied globally to all `/api/v1/*` endpoints.
The `RENDER_RATE_LIMIT=5` applies specifically to image rendering endpoints (computationally expensive).

### Recommended Production Stack

```
Client → Cloudflare (DDoS + WAF) → Nginx (connection rate limit) → FastAPI (request rate limit) → Backend
```

**Nginx rate limiting** (`/etc/nginx/conf.d/arken.conf`):
```nginx
limit_req_zone $binary_remote_addr zone=arken_api:10m rate=30r/m;
limit_req_zone $binary_remote_addr zone=arken_render:10m rate=5r/m;

location /api/v1/analyze {
    limit_req zone=arken_api burst=10 nodelay;
}

location /api/v1/render {
    limit_req zone=arken_render burst=2 nodelay;
}
```

### Authenticated vs. Unauthenticated Limits

| Endpoint Group | Unauthenticated | Authenticated (JWT) |
|---|---|---|
| `/api/v1/auth/*` | 10/min | N/A |
| `/api/v1/analyze/*` | 5/min | 30/min |
| `/api/v1/render/*` | 2/min | 5/min |
| `/health/*` | 60/min | 60/min |

---

## 6. Secret Management Best Practices

### Development
- Use `.env` file (gitignored).
- Never share `.env` files over chat or email.
- Use `direnv` or `dotenv` for automatic loading.

### Production
- Use a dedicated secret manager:
  - **AWS**: AWS Secrets Manager + IAM roles
  - **GCP**: Secret Manager + Workload Identity
  - **Generic**: HashiCorp Vault or Doppler
- Rotate secrets quarterly or immediately after any suspected exposure.
- Audit secret access logs monthly.

### CI/CD
- Store secrets as **masked environment variables** in your CI platform (GitHub Actions Secrets, GitLab CI Variables).
- Never echo secrets in CI logs (`set -x` exposes them).
- Use OIDC/workload identity instead of long-lived keys where possible.

---

## 7. Reporting Security Issues

If you discover a security vulnerability in ARKEN, please **do not** open a public GitHub issue.

Contact: security@arken.in (or your designated security contact)

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested remediation

We aim to acknowledge reports within 48 hours.

---

*Last updated: 2025-12-01 | ARKEN Security Team*
