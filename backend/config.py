"""
ARKEN — Application Configuration v3.0
All environment variables with Pydantic Settings validation.
Graceful defaults: app starts without any optional service.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "ARKEN PropTech Engine"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY: SecretStr = SecretStr("arken-dev-secret-key-change-in-production-32ch")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    ALGORITHM: str = "HS256"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://arken:arken@localhost:5432/arken"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 5

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 7200

    # ── AWS S3 ────────────────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[SecretStr] = None
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET_UPLOADS: str = "arken-uploads"
    S3_BUCKET_RENDERS: str = "arken-renders"
    S3_BUCKET_ARTIFACTS: str = "arken-artifacts"
    CDN_BASE_URL: Optional[str] = None
    USE_S3: bool = False

    # ── AI APIs ───────────────────────────────────────────────────────────────
    GOOGLE_API_KEY: Optional[SecretStr] = None
    OPENAI_API_KEY: Optional[SecretStr] = None
    STABILITY_API_KEY: Optional[SecretStr] = None

    # ── Model Paths ───────────────────────────────────────────────────────────
    YOLO_MODEL_PATH: str = "ml/weights/yolov8x-seg.pt"
    SAM_CHECKPOINT: str = "ml/weights/sam_vit_h.pth"
    XGBOOST_MODEL_PATH: str = "ml/weights/roi_xgb.joblib"
    PROPHET_MODEL_DIR: str = "ml/weights/prophet/"
    CLIP_MODEL: str = "ViT-L/14"
    CHROMA_PERSIST_DIR: str = "/app/data/chroma"

    # ── Vector DB ─────────────────────────────────────────────────────────────
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX: str = "arken-insights"

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 30
    RENDER_RATE_LIMIT: int = 5

    # ── Indian Market ─────────────────────────────────────────────────────────
    DEFAULT_CURRENCY: str = "INR"
    GST_RATE: float = 0.18
    SUPPORTED_CITIES: List[str] = [
        "Hyderabad", "Bangalore", "Mumbai", "Delhi NCR",
        "Pune", "Chennai", "Kolkata", "Ahmedabad", "Surat",
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
