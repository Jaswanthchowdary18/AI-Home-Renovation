"""
ARKEN Test Suite — conftest.py
==============================
Central stub registry + agent loader.

All heavy dependencies (Gemini, XGBoost, YOLOv8, ChromaDB, Redis,
PostgreSQL, torch, cv2, langgraph, shap, and every internal ml.* /
services.* module) are replaced with lightweight stubs before any
agent code is loaded.  Tests run with zero external services.

Each test file loads agents via load_agent(name, rel_path) which loads
the Python file directly, bypassing agents/__init__.py entirely.

Public helpers
--------------
    load_agent(module_name, rel_path) -> module
    make_minimal_state(**overrides)   -> dict
    make_roi_state(**overrides)       -> dict
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from typing import Any, Dict
from unittest.mock import MagicMock


# ── Stub factory ──────────────────────────────────────────────────────────────

def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ── pydantic ──────────────────────────────────────────────────────────────────

class _BaseModel:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
    def model_dump(self, **_):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

_pyd = _stub("pydantic")
_pyd.BaseModel = _BaseModel
_pyd.Field = lambda *a, **kw: None
_pyd.SecretStr = str
_pyd.validator = lambda *a, **kw: (lambda f: f)
_pyd.field_validator = lambda *a, **kw: (lambda f: f)
_pyd.model_validator = lambda *a, **kw: (lambda f: f)
sys.modules.setdefault("pydantic", _pyd)
sys.modules.setdefault("pydantic.v1", _pyd)
sys.modules.setdefault("pydantic_settings", _stub("pydantic_settings", BaseSettings=_BaseModel))


# ── config ────────────────────────────────────────────────────────────────────

class _Settings:
    GOOGLE_API_KEY = None
    ML_WEIGHTS_DIR = "/tmp/arken_test_weights"
    XGBOOST_MODEL_PATH = "/tmp/arken_test_weights/roi_xgb.joblib"
    SECRET_KEY = "test-secret-key-not-real-32chars!!"
    DATABASE_URL = None
    REDIS_URL = None
    USE_S3 = False
    ENVIRONMENT = "test"
    DEBUG = False

sys.modules.setdefault("config", _stub("config", settings=_Settings()))


# ── numpy / pandas (real if installed, else minimal stub) ─────────────────────

try:
    import numpy  # noqa: F401
except ImportError:
    _np = _stub("numpy")
    _np.clip = lambda x, lo, hi: max(lo, min(hi, x))
    _np.mean = lambda a: sum(a) / max(len(a), 1)
    _np.array = lambda x, **_: x
    sys.modules.setdefault("numpy", _np)

try:
    import pandas  # noqa: F401
except ImportError:
    class _DF:
        def __init__(self, data=None): self._d = data or {}
        def __getitem__(self, k): return self
    sys.modules.setdefault("pandas", _stub("pandas", DataFrame=_DF))


# ── heavy ML / CV stubs ───────────────────────────────────────────────────────

class _XGBRegressor:
    def fit(self, X, y, **kw): return self
    def predict(self, X): return [12.0]

sys.modules.setdefault("xgboost", _stub("xgboost", XGBRegressor=_XGBRegressor))
sys.modules.setdefault("sklearn", _stub("sklearn"))
sys.modules.setdefault("sklearn.metrics",
    _stub("sklearn.metrics", mean_absolute_error=lambda y, p: 0.5))
sys.modules.setdefault("sklearn.model_selection",
    _stub("sklearn.model_selection",
          train_test_split=lambda *a, **kw: (a[0], a[0], a[1], a[1])))
sys.modules.setdefault("sklearn.ensemble",
    _stub("sklearn.ensemble",
          RandomForestRegressor=MagicMock,
          GradientBoostingRegressor=MagicMock))
sys.modules.setdefault("joblib",
    _stub("joblib", dump=lambda obj, path: None, load=lambda path: MagicMock()))

for _m in ["torch", "torchvision", "torchvision.transforms",
           "torchvision.models", "PIL", "PIL.Image", "cv2",
           "ultralytics", "transformers"]:
    sys.modules.setdefault(_m, _stub(_m))

_torch_mod = sys.modules["torch"]
_torch_mod.no_grad = contextlib.nullcontext
_torch_mod.load = lambda *a, **kw: {}
_torch_mod.device = lambda x: x

_pil_img = sys.modules["PIL.Image"]
_pil_img.open = lambda x: MagicMock()
_pil_img.fromarray = lambda x: MagicMock()

sys.modules.setdefault("google", _stub("google"))
sys.modules.setdefault("google.genai", _stub("google.genai", Client=MagicMock()))
sys.modules.setdefault("google.genai.types",
    _stub("google.genai.types",
          Content=MagicMock(), Part=MagicMock(), Blob=MagicMock()))

sys.modules.setdefault("chromadb",
    _stub("chromadb", PersistentClient=type("PC", (), {
        "get_or_create_collection": lambda self, *a, **kw: MagicMock(),
        "__init__": lambda self, *a, **kw: None,
    })))
sys.modules.setdefault("redis",
    _stub("redis", Redis=type("R", (), {
        "get": lambda self, k: None,
        "set": lambda self, *a, **kw: True,
        "ping": lambda self: True,
        "__init__": lambda self, *a, **kw: None,
    })))

_lg_graph = _stub("langgraph.graph", END="__end__")
_lg_graph.StateGraph = type("StateGraph", (), {
    "add_node": lambda self, *a, **kw: None,
    "add_edge": lambda self, *a, **kw: None,
    "add_conditional_edges": lambda self, *a, **kw: None,
    "set_entry_point": lambda self, *a, **kw: None,
    "compile": lambda self, *a, **kw: MagicMock(),
    "__init__": lambda self, *a, **kw: None,
})
sys.modules.setdefault("langgraph", _stub("langgraph"))
sys.modules.setdefault("langgraph.graph", _lg_graph)
sys.modules.setdefault("shap", _stub("shap", TreeExplainer=MagicMock()))

for _m in ["aiofiles", "asyncpg", "sqlalchemy",
           "sqlalchemy.ext", "sqlalchemy.ext.asyncio"]:
    sys.modules.setdefault(_m, _stub(_m))


# ── internal ml.* / services.* stubs ─────────────────────────────────────────

for _s in [
    "ml", "ml.cv_feature_extractor", "ml.cv_model_registry",
    "ml.damage_detector", "ml.depth_estimator", "ml.style_classifier",
    "ml.property_models", "ml.housing_preprocessor", "ml.roi_calibration",
    "ml.roi_explainer",
    "services", "services.rag", "services.rag.retriever",
    "services.rag.context_builder", "services.insight_engine",
    "services.insight_engine.engine", "services.datasets",
    "services.datasets.dataset_loader", "services.trust",
    "services.trust.output_validator",
]:
    sys.modules.setdefault(_s, _stub(_s))


class _InsightEngine:
    def generate(self, state: dict) -> dict:
        return {"recommendations": [], "risk_factors": [],
                "market_timing": {}, "action_checklist": []}

sys.modules["services.insight_engine.engine"].InsightEngine = _InsightEngine


# ── Agent loader ──────────────────────────────────────────────────────────────

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_agent(module_name: str, rel_path: str):
    """Load agent by file path, bypassing agents/__init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    full_path = os.path.join(_BACKEND_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load graph_state (dependency of orchestrator)
load_agent("agents.graph_state", "agents/graph_state.py")


# ── Shared state builders ─────────────────────────────────────────────────────

def make_minimal_state(**overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "project_id": "test-proj-001",
        "city": "Hyderabad",
        "room_type": "bedroom",
        "budget_tier": "mid",
        "budget_inr": 750_000,
        "theme": "Modern Minimalist",
        "floor_area_sqft": 120.0,
        "wall_area_sqft": 200.0,
        "style_label": "Modern Minimalist",
        "style_confidence": 0.72,
        "detected_objects": ["bed", "wardrobe", "ceiling light"],
        "material_types": ["vitrified_tile"],
        "issues_detected": [],
        "condition_score": 68,
        "wall_condition": "fair",
        "floor_condition": "fair",
        "renovation_scope": "partial",
        "high_value_upgrades": ["false ceiling with cove lighting", "modular wardrobe"],
        "room_features": {
            "wall_color": "white",
            "floor_type": "vitrified tiles",
            "ceiling_type": "plain plaster",
            "detected_furniture": ["bed", "wardrobe", "ceiling light"],
            "natural_light": "moderate",
            "free_space_percentage": 45.0,
            "room_area_estimate": 120.0,
            "condition": "fair",
            "condition_score": 68,
            "wall_condition": "fair",
            "floor_condition": "fair",
            "issues_detected": [],
            "renovation_scope": "partial",
            "high_value_upgrades": ["false ceiling with cove lighting"],
            "renovation_priority": ["walls", "flooring", "lighting"],
            "layout_issues": [],
            "color_palette": ["white", "grey"],
            "style_label": "Modern Minimalist",
            "style_confidence": 0.72,
        },
        "vision_features": {
            "wall_treatment": "white paint",
            "floor_material": "vitrified tiles",
            "ceiling_treatment": "plain plaster",
            "furniture_items": ["bed", "wardrobe"],
            "lighting_type": "ceiling light",
            "extraction_source": "gemini",
        },
        "damage_assessment": {
            "overall_condition": "fair",
            "severity": "medium",
            "issues_detected": [],
            "renovation_priority": ["walls", "flooring"],
            "condition_score": 68,
        },
        "layout_report": {
            "layout_score": "70/100",
            "walkable_space_pct": 45.0,
            "issues": [],
        },
        "cost_estimate": {"total_inr": 750_000, "within_budget": True},
        "cv_features": {
            "room_type": "bedroom",
            "detected_objects": ["bed", "wardrobe"],
            "style": "Modern Minimalist",
            "style_confidence": 0.65,
            "materials": ["vitrified_tile"],
            "lighting": "moderate",
            "extraction_source": "cv_pipeline",
        },
        "retrieved_knowledge": [],
        "rag_context": "",
        "agent_timings": {},
        "completed_agents": [],
        "errors": [],
    }
    base.update(overrides)
    return base


def make_roi_state(**overrides) -> Dict[str, Any]:
    base = make_minimal_state()
    base.update({
        "budget_estimate": {"total_cost_inr": 750_000},
        "property_age_years": 10,
        "image_features": {"room_condition": "fair"},
    })
    base.update(overrides)
    return base
