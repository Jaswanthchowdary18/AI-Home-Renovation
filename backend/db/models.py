"""
ARKEN — Full Database Schema v2.0
All SQLAlchemy ORM models for the renovation platform.

v2.0 additions:
  - PriceAlertModel: user price threshold alerts for construction materials
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base, TimestampMixin, UUIDMixin


# ── Enums ─────────────────────────────────────────────────────────────────────

class BudgetTier(str, Enum):
    BASIC = "basic"
    MID = "mid"
    PREMIUM = "premium"


class ProjectStatus(str, Enum):
    DRAFT = "draft"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    RENDERING = "rendering"
    COMPLETE = "complete"
    ARCHIVED = "archived"


class RenderStatus(str, Enum):
    QUEUED = "queued"
    SEGMENTING = "segmenting"
    INPAINTING = "inpainting"
    COMPLETE = "complete"
    FAILED = "failed"


class UserRole(str, Enum):
    USER = "user"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


# ── Users ─────────────────────────────────────────────────────────────────────

class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(50), default=UserRole.USER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    preferences: Mapped[Optional[Dict]] = mapped_column(JSON)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user")


class RefreshToken(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "refresh_tokens"

    token: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    user = relationship("User", back_populates="refresh_tokens")


# ── Projects ──────────────────────────────────────────────────────────────────

class Project(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(50), default=ProjectStatus.DRAFT)
    city: Mapped[Optional[str]] = mapped_column(String(100))
    budget_inr: Mapped[Optional[int]] = mapped_column(Integer)
    budget_tier: Mapped[Optional[str]] = mapped_column(String(50))
    theme: Mapped[Optional[str]] = mapped_column(String(100))
    room_type: Mapped[Optional[str]] = mapped_column(String(100))
    area_sqft: Mapped[Optional[float]] = mapped_column(Float)
    project_metadata: Mapped[Optional[Dict]] = mapped_column("metadata", JSON)

    owner = relationship("User", back_populates="projects")
    room_images = relationship("RoomImage", back_populates="project", cascade="all, delete-orphan")
    renders = relationship("RenderArtifact", back_populates="project", cascade="all, delete-orphan")
    analysis = relationship("RoomAnalysis", back_populates="project", uselist=False)
    cost_plan = relationship("CostPlan", back_populates="project", uselist=False)
    roi_report = relationship("ROIReport", back_populates="project", uselist=False)
    schedule = relationship("ProjectSchedule", back_populates="project", uselist=False)
    chat_sessions = relationship("ChatSession", back_populates="project")
    versions = relationship("ProjectVersion", back_populates="project")


# ── Room Images ───────────────────────────────────────────────────────────────

class RoomImage(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "room_images"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    cdn_url: Mapped[Optional[str]] = mapped_column(String(1024))
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    clip_embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(768))

    project = relationship("Project", back_populates="room_images")
    analysis = relationship("RoomAnalysis", back_populates="source_image", uselist=False)


# ── Room Analysis ─────────────────────────────────────────────────────────────

class RoomAnalysis(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "room_analyses"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), unique=True)
    source_image_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("room_images.id"))

    wall_mask_s3: Mapped[Optional[str]] = mapped_column(String(512))
    floor_mask_s3: Mapped[Optional[str]] = mapped_column(String(512))
    ceiling_mask_s3: Mapped[Optional[str]] = mapped_column(String(512))
    combined_mask_s3: Mapped[Optional[str]] = mapped_column(String(512))

    estimated_length_ft: Mapped[Optional[float]] = mapped_column(Float)
    estimated_width_ft: Mapped[Optional[float]] = mapped_column(Float)
    estimated_height_ft: Mapped[Optional[float]] = mapped_column(Float)
    wall_area_sqft: Mapped[Optional[float]] = mapped_column(Float)
    floor_area_sqft: Mapped[Optional[float]] = mapped_column(Float)

    detected_objects: Mapped[Optional[Dict]] = mapped_column(JSON)
    room_style_tags: Mapped[Optional[List]] = mapped_column(JSON)
    style_embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(768))

    paint_liters: Mapped[Optional[float]] = mapped_column(Float)
    tiles_sqft: Mapped[Optional[float]] = mapped_column(Float)
    plywood_sqft: Mapped[Optional[float]] = mapped_column(Float)

    project = relationship("Project", back_populates="analysis")
    source_image = relationship("RoomImage", back_populates="analysis")


# ── Render Artifacts ──────────────────────────────────────────────────────────

class RenderArtifact(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "render_artifacts"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(50), default=RenderStatus.QUEUED)
    prompt_used: Mapped[Optional[str]] = mapped_column(Text)
    negative_prompt: Mapped[Optional[str]] = mapped_column(Text)
    model_used: Mapped[Optional[str]] = mapped_column(String(100))
    render_s3: Mapped[Optional[str]] = mapped_column(String(512))
    cdn_url: Mapped[Optional[str]] = mapped_column(String(1024))
    generation_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    render_metadata: Mapped[Optional[Dict]] = mapped_column("metadata", JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    project = relationship("Project", back_populates="renders")


# ── Cost Plan ─────────────────────────────────────────────────────────────────

class CostPlan(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "cost_plans"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), unique=True)
    total_inr: Mapped[Optional[int]] = mapped_column(Integer)
    material_inr: Mapped[Optional[int]] = mapped_column(Integer)
    labour_inr: Mapped[Optional[int]] = mapped_column(Integer)
    contingency_inr: Mapped[Optional[int]] = mapped_column(Integer)
    gst_inr: Mapped[Optional[int]] = mapped_column(Integer)
    line_items: Mapped[Optional[List]] = mapped_column(JSON)
    supplier_recommendations: Mapped[Optional[List]] = mapped_column(JSON)

    project = relationship("Project", back_populates="cost_plan")


# ── ROI Report ────────────────────────────────────────────────────────────────

class ROIReport(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "roi_reports"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), unique=True)
    city: Mapped[Optional[str]] = mapped_column(String(100))
    city_tier: Mapped[Optional[int]] = mapped_column(Integer)
    pre_reno_value_inr: Mapped[Optional[int]] = mapped_column(Integer)
    post_reno_value_inr: Mapped[Optional[int]] = mapped_column(Integer)
    equity_gain_inr: Mapped[Optional[int]] = mapped_column(Integer)
    roi_pct: Mapped[Optional[float]] = mapped_column(Float)
    rental_yield_delta: Mapped[Optional[float]] = mapped_column(Float)
    payback_months: Mapped[Optional[int]] = mapped_column(Integer)
    model_confidence: Mapped[Optional[float]] = mapped_column(Float)
    feature_importances: Mapped[Optional[Dict]] = mapped_column(JSON)

    project = relationship("Project", back_populates="roi_report")


# ── Project Schedule (CPM) ────────────────────────────────────────────────────

class ProjectSchedule(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "project_schedules"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), unique=True)
    total_days: Mapped[Optional[int]] = mapped_column(Integer)
    critical_path_days: Mapped[Optional[int]] = mapped_column(Integer)
    tasks: Mapped[Optional[List]] = mapped_column(JSON)
    risk_score: Mapped[Optional[float]] = mapped_column(Float)
    risks: Mapped[Optional[List]] = mapped_column(JSON)

    project = relationship("Project", back_populates="schedule")


# ── Price Forecasts ───────────────────────────────────────────────────────────

class MaterialForecast(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "material_forecasts"

    material_name: Mapped[str] = mapped_column(String(200), nullable=False)
    unit: Mapped[str] = mapped_column(String(50))
    current_price_inr: Mapped[float] = mapped_column(Float)
    forecast_30d: Mapped[Optional[float]] = mapped_column(Float)
    forecast_60d: Mapped[Optional[float]] = mapped_column(Float)
    forecast_90d: Mapped[Optional[float]] = mapped_column(Float)
    volatility_score: Mapped[Optional[float]] = mapped_column(Float)
    trend: Mapped[Optional[str]] = mapped_column(String(20))
    source: Mapped[Optional[str]] = mapped_column(String(200))
    forecast_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    raw_forecast: Mapped[Optional[Dict]] = mapped_column(JSON)


# ── Price Alerts (NEW — Feature 3) ────────────────────────────────────────────

class PriceAlertModel(Base, UUIDMixin, TimestampMixin):
    """
    User-defined price threshold alerts for construction materials.

    Columns:
      user_id       — authenticated user who owns the alert
      material_key  — key from SEED_DATA (e.g. "steel_tmt_fe500_per_kg")
      threshold_inr — price level that triggers the alert
      direction     — "above" (alert when price rises above threshold)
                    — "below" (alert when price falls below threshold)
      email         — optional notification address
      is_active     — False once triggered or manually deleted
      triggered_at  — timestamp when the threshold was first crossed
    """
    __tablename__ = "price_alerts"

    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    material_key: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    threshold_inr: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)   # "above" | "below"
    email: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


# ── Chat Sessions ─────────────────────────────────────────────────────────────

class ChatSession(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "chat_sessions"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    project = relationship("Project", back_populates="chat_sessions")


class ChatMessage(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "chat_messages"

    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[Optional[str]] = mapped_column(String(100))
    triggered_rerender: Mapped[bool] = mapped_column(Boolean, default=False)
    message_metadata: Mapped[Optional[Dict]] = mapped_column("metadata", JSON)

    session = relationship("ChatSession", back_populates="messages")


# ── Project Versions ──────────────────────────────────────────────────────────

class ProjectVersion(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "project_versions"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[Dict] = mapped_column(JSON)
    triggered_by: Mapped[Optional[str]] = mapped_column(String(200))

    project = relationship("Project", back_populates="versions")