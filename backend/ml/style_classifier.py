"""
ARKEN — Style Classifier v4.0
================================
Upgraded to load fine-tuned weights as PRIMARY path.

v4.0 Changes over v3.0:
  - THREE-TIER classification hierarchy (all from YOUR real dataset):

    Tier 1 — Fine-tuned EfficientNet-B0 (PRIMARY when style_classifier.pt present):
      Trained on your 4,139 real interior design images (train_data.csv / val_data.csv).
      5-class style head: boho, industrial, minimalist, modern, scandinavian.
      model_used = "efficientnet_finetuned"

    Tier 2 — Fine-tuned CLIP (when clip_finetuned.pt present, EfficientNet absent):
      Uses your fine-tuned CLIP image encoder for style embedding.
      Computes cosine similarity against room_type-aware text prompt library.
      model_used = "clip_finetuned_v1" / "clip_finetuned_dual_pass"

    Tier 3 — Zero-shot CLIP (fallback when no fine-tuned weights):
      Dual-pass zero-shot CLIP v1 (metadata-derived room prompts) +
      CLIP v2 (evocative feelings prompts). Identical to v3.0 logic.
      model_used = "clip_v1_metadata_prompts" / "clip_dual_pass"

  - Fine-tuned EfficientNet maps 5 dataset styles → 11 ARKEN canonical styles:
      boho        → "Bohemian eclectic"
      industrial  → "Industrial loft"
      minimalist  → "Modern Minimalist"
      modern      → "Modern Minimalist"  (with higher confidence for pure modern)
      scandinavian→ "Scandinavian"

  - Fine-tuned CLIP: loads clip_finetuned.pt visual encoder, replaces
    CLIPModel's visual projection with fine-tuned weights. All zero-shot
    prompt logic (ROOM_STYLE_PROMPTS, _CLIP_V2_FEELINGS_PROMPTS) reused
    with the better visual encoder.

  - validate_on_dataset() updated: evaluates fine-tuned model if present,
    otherwise falls back to zero-shot evaluation. Saves eval JSON with
    model_tier field indicating which tier was used.

  - All v3.0 public API preserved (classify() signature unchanged).

Dependencies (optional):
  pip install torch torchvision transformers Pillow scikit-learn
  pip install clip @ git+https://github.com/openai/CLIP.git
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Style labels (11 ARKEN classes) ─────────────────────────────────────────
STYLE_LABELS: List[str] = [
    "Modern Minimalist interior",
    "Scandinavian interior",
    "Japandi interior",
    "Industrial loft interior",
    "Bohemian eclectic interior",
    "Contemporary Indian interior",
    "Traditional Indian interior",
    "Art Deco interior",
    "Mid-Century Modern interior",
    "Coastal beach house interior",
    "Farmhouse rustic interior",
]

_SHORT_LABELS: List[str] = [s.replace(" interior", "").strip() for s in STYLE_LABELS]

# ── Dataset style labels (5 classes from YOUR training data) ─────────────────
_DATASET_STYLES: List[str] = ["boho", "industrial", "minimalist", "modern", "scandinavian"]

# ── Fine-tuned EfficientNet: dataset style → ARKEN canonical mapping ─────────
# When fine-tuned model predicts a dataset style, map it to ARKEN canonical
# and assign base confidence based on how unambiguous the mapping is.
_DATASET_TO_ARKEN: Dict[str, Tuple[str, float]] = {
    "boho":         ("Bohemian eclectic",  0.92),
    "industrial":   ("Industrial loft",    0.92),
    "minimalist":   ("Modern Minimalist",  0.88),
    "modern":       ("Modern Minimalist",  0.85),
    "scandinavian": ("Scandinavian",       0.92),
}

# ── CSV style -> ARKEN canonical mapping ─────────────────────────────────────
_CSV_TO_ARKEN: Dict[str, str] = {
    "modern":       "Modern Minimalist",
    "minimalist":   "Modern Minimalist",
    "scandinavian": "Scandinavian",
    "boho":         "Bohemian eclectic",
    "industrial":   "Industrial loft",
}
_ARKEN_TO_CSV_STYLE: Dict[str, str] = {
    "Modern Minimalist": "modern",
    "Scandinavian":      "scandinavian",
    "Industrial loft":   "industrial",
    "Bohemian eclectic": "boho",
}

# ── STYLE_PRIORS from REAL metadata (4,139 rows) ─────────────────────────────
# modern=24.8%, boho=22.2%, industrial=20.7%, minimalist=18.0%, scandinavian=14.3%
_RESIDUAL = 0.043 / 6
STYLE_PRIORS: Dict[str, float] = {
    "Modern Minimalist":   0.214,
    "Scandinavian":        0.143,
    "Japandi":             _RESIDUAL,
    "Industrial loft":     0.207,
    "Bohemian eclectic":   0.222,
    "Contemporary Indian": _RESIDUAL,
    "Traditional Indian":  _RESIDUAL,
    "Art Deco":            _RESIDUAL,
    "Mid-Century Modern":  _RESIDUAL,
    "Coastal beach house": _RESIDUAL,
    "Farmhouse rustic":    _RESIDUAL,
}
_prior_sum = sum(STYLE_PRIORS.values())
STYLE_PRIORS = {k: v / _prior_sum for k, v in STYLE_PRIORS.items()}

# ── CLIP thresholds ──────────────────────────────────────────────────────────
_CLIP_THRESHOLD    = 0.40
_CLIP_BLEND_THRESH = 0.35
_PRIOR_WEIGHT      = 0.20
_CLIP_WEIGHT       = 0.80

# ── Room x Style CLIP Prompt Library v1 (100 prompts, from real metadata) ────
ROOM_STYLE_PROMPTS: Dict[str, Dict[str, List[str]]] = {
    "kitchen": {
        "modern": [
            "modern kitchen with clean lines and handleless flat-panel cabinets",
            "contemporary modular kitchen white glossy surfaces integrated appliances",
            "sleek modern kitchen island marble or quartz countertop pendant lighting",
            "minimalist modern kitchen hidden storage under-cabinet LED lighting",
            "high-gloss lacquer modern kitchen with contemporary fixtures",
        ],
        "minimalist": [
            "minimalist kitchen bare essentials no clutter white walls single tone",
            "ultra-minimal kitchen concealed fridge open space negative space",
            "pure minimalist kitchen matte finish cabinets simple hardware",
            "minimal kitchen Japanese-inspired clean surfaces natural light",
            "minimalist kitchen open shelving neutral palette no ornaments",
        ],
        "boho": [
            "bohemian kitchen open shelving displaying colourful ceramics and plants",
            "boho kitchen terracotta tiles hanging plants woven baskets",
            "eclectic boho kitchen mixed tiles rattan bar stools herbs on shelf",
            "bohemian kitchen warm earthy tones macrame pot holder",
            "boho kitchen colourful mismatched items artisan pottery",
        ],
        "industrial": [
            "industrial kitchen exposed brick wall stainless steel counters",
            "urban industrial kitchen black matte cabinetry pipe shelving Edison bulbs",
            "raw industrial kitchen concrete walls metal bar stools open ceiling",
            "industrial loft kitchen dark cabinets exposed ductwork metal fittings",
            "factory-style kitchen heavy hardware brushed metal surfaces",
        ],
        "scandinavian": [
            "Scandinavian kitchen light wood cabinets white walls natural light",
            "Nordic kitchen simple clean lines birch veneer white tiles",
            "Scandi kitchen pale oak countertop minimal clutter plants",
            "hygge-inspired kitchen wooden accents cosy textures pendant lamp",
            "Scandinavian open kitchen white and wood tones",
        ],
    },
    "living_room": {
        "modern": [
            "modern living room sectional sofa large format tiles recessed LED",
            "contemporary living room neutral palette floating TV unit glass table",
            "sleek modern lounge statement wall panel hidden cove lighting",
            "modern living room clean lines brass accents low-profile furniture",
            "contemporary living room geometric rug and modular shelving",
        ],
        "minimalist": [
            "minimalist living room sparse furniture neutral tones bare floor",
            "ultra-minimal lounge single sofa empty white walls natural light",
            "minimal living room Zen-inspired few objects calm atmosphere",
            "minimalist lounge muted palette no decorative clutter",
            "minimal living space single armchair and floor lamp nothing else",
        ],
        "boho": [
            "bohemian living room macrame wall hanging rattan furniture layered rugs",
            "eclectic boho lounge kilim rug floor cushions layered textiles",
            "boho living room hanging plants gallery wall poufs and throws",
            "bohemian eclectic space woven throws vintage trunk patterned cushions",
            "boho living room fringe lampshade terracotta pots pattern mix",
        ],
        "industrial": [
            "industrial living room exposed brick iron-frame sofa Edison pendants",
            "loft-style lounge concrete ceiling dark leather furniture pipe shelving",
            "industrial living room polished concrete floor black metal shelving",
            "urban loft living area factory windows reclaimed wood coffee table",
            "raw industrial lounge exposed beams dark metal furniture",
        ],
        "scandinavian": [
            "Scandinavian living room pale sofa wooden furniture hygge candles",
            "Nordic living room white walls birch legs furniture and plants",
            "Scandi lounge simplicity warm textiles light tones floor lamp",
            "Scandinavian-style lounge knit cushions and natural materials",
            "hygge living room throw blanket light timber floor simple lines",
        ],
    },
    "bedroom": {
        "modern": [
            "modern bedroom platform bed indirect cove lighting built-in wardrobe",
            "contemporary bedroom neutral palette hidden handles floating bedside",
            "sleek modern bedroom minimal decor upholstered headboard dark accent",
            "modern master bedroom ambient LED strip lighting clean lines",
            "contemporary bedroom tufted headboard and simple bedside lamps",
        ],
        "minimalist": [
            "minimalist bedroom low platform bed white linen bare walls no clutter",
            "ultra-minimal bedroom single bedside lamp concealed wardrobe",
            "Zen minimalist bedroom tatami mat simple white palette floor-level",
            "minimal bedroom muted tones hidden storage calm atmosphere",
            "Japanese minimal bedroom natural materials sparse furniture",
        ],
        "boho": [
            "bohemian bedroom tapestry wall hanging macrame rattan pendant plants",
            "boho bedroom layered bedding dreamcatcher fairy lights",
            "eclectic bohemian bedroom mixed patterns fringe cushions woven blanket",
            "boho bedroom earthy tones vintage rug brass accents floor cushions",
            "bohemian bedroom hanging plants and eclectic wall decor",
        ],
        "industrial": [
            "industrial bedroom exposed brick Edison bulb bedside lamp dark palette",
            "loft-style bedroom concrete walls dark metal bed frame pipe shelving",
            "industrial bedroom raw materials exposed beams dark linen",
            "urban industrial bedroom metal clock decor minimal soft furnishings",
            "industrial master bedroom polished cement floor dark textiles",
        ],
        "scandinavian": [
            "Scandinavian bedroom white walls light oak wood bed frame plants",
            "Nordic bedroom cosy duvet simple bedside light wood floors",
            "Scandi bedroom whitewash warm textiles minimalist palette",
            "hygge bedroom layered white bedding candles wooden floors",
            "Scandinavian bedroom pale palette clean lines natural materials",
        ],
    },
    "bathroom": {
        "modern": [
            "modern bathroom wall-hung toilet frameless glass shower LED mirror",
            "contemporary bathroom large format grey tile floating vanity",
            "modern minimal bathroom waterfall faucet concealed storage",
            "sleek modern bathroom backlit mirror rain shower head",
            "contemporary bathroom monochrome tiles and polished chrome fittings",
        ],
        "minimalist": [
            "minimalist bathroom white tiles clean surfaces hidden cistern",
            "ultra-minimal bathroom vessel sink walk-in shower no clutter",
            "Zen minimalist bathroom pebble floor matte white tiles wooden accents",
            "minimal spa bathroom nothing extra clean lines natural light",
            "minimalist Japanese bathroom clean lines wooden bath tray",
        ],
        "boho": [
            "bohemian bathroom patterned encaustic tiles wicker basket plants",
            "boho bathroom rattan mirror macrame wall hanging earthy tones",
            "eclectic boho bathroom mixed tiles colourful towels terracotta",
            "bohemian bathroom hanging pot plants arched mirror dried flowers",
            "boho spa bathroom woven bath mat and artisan ceramics",
        ],
        "industrial": [
            "industrial bathroom exposed pipes concrete walls metal fixtures",
            "loft-style bathroom black iron fittings subway tiles dark grout",
            "industrial bathroom brushed steel faucets poured concrete basin",
            "raw industrial bathroom exposed brick pipe shelving factory mirror",
            "urban industrial bathroom dark palette and heavy hardware",
        ],
        "scandinavian": [
            "Scandinavian bathroom white tiles light wood accents plants",
            "Nordic bathroom simple clean stone white palette birch wood",
            "Scandi bathroom minimalist white with subtle wood details",
            "hygge bathroom warm textures neutral tiles terrazzo accents",
            "Scandinavian-style bathroom bright airy white and natural materials",
        ],
    },
}

_GENERIC_PROMPTS: Dict[str, List[str]] = {
    "Japandi": [
        "japandi interior wabi-sabi natural bamboo minimal neutral tones",
        "japandi room Japanese and Scandinavian fusion natural materials",
        "japandi style interior earthy palette low furniture handmade ceramics",
        "wabi-sabi japandi room linen textiles and bamboo soft light",
        "japandi living space muted tones unfinished textures calm",
    ],
    "Contemporary Indian": [
        "contemporary Indian interior brass accents stone and teak",
        "modern Indian home with jaali screen and warm colours",
        "Indian contemporary design terracotta and carved wood",
        "fusion Indian interior modern furniture traditional accents",
        "contemporary Indian lounge with embroidered cushions and brass lamp",
    ],
    "Traditional Indian": [
        "traditional Indian interior carved teak furniture silk textiles",
        "classic Indian home with brass diyas and intricate woodwork",
        "heritage Indian interior jewel-tone walls and handloom dhurrie",
        "ethnic Indian room with Rajasthani painted furniture",
        "traditional Indian living room jali screen and antique wood",
    ],
    "Art Deco": [
        "art deco interior geometric patterns gold black velvet upholstery",
        "deco-style room with mirrored surfaces and bold contrasts",
        "art deco living room sunburst motif fan-shaped sconces",
        "art deco interior lacquered furniture and rich jewel tones",
        "decadent art deco room deep teal and gold accents",
    ],
    "Mid-Century Modern": [
        "mid-century modern interior tapered leg furniture walnut tones",
        "1950s retro modern lounge Eames-inspired chairs warm wood",
        "mid-century room mustard yellow and warm walnut",
        "retro modern interior sunburst clock and teak sideboard",
        "mid-century modern home avocado green and warm wood tones",
    ],
    "Coastal beach house": [
        "coastal interior whitewashed wood and sea blue accent",
        "beach house room with nautical rope and driftwood accents",
        "coastal living room pale blue linen and wicker furniture",
        "seaside interior whitewash and sea glass colours",
        "coastal bedroom shiplap walls and navy blue accents",
    ],
    "Farmhouse rustic": [
        "farmhouse interior shiplap walls and reclaimed wood",
        "rustic farmhouse room distressed wood and galvanised metal",
        "country farmhouse kitchen apron sink and open shelving",
        "farmhouse living room sliding barn door and linen sofa",
        "rustic farmhouse bedroom wrought iron bed frame warm tones",
    ],
}

_CLIP_V2_FEELINGS_PROMPTS: Dict[str, List[str]] = {
    "Modern Minimalist": [
        "this room interior design feels like clean quiet simplicity",
        "this room interior design feels like modern urban sophistication",
        "this space feels like a calm minimalist zen retreat",
        "this interior feels like architectural restraint and precision",
        "this room feels like contemporary minimalist design sensibility",
    ],
    "Scandinavian": [
        "this room interior design feels like cosy Nordic hygge warmth",
        "this space feels like a bright Scandinavian winter cabin",
        "this interior feels like simple Nordic natural living",
        "this room feels like Swedish design warmth and simplicity",
        "this space feels like a hygge-inspired Scandi home",
    ],
    "Japandi": [
        "this room interior design feels like wabi-sabi imperfect beauty",
        "this space feels like Japanese and Nordic fusion calm",
        "this interior feels like earthy handcrafted Japanese minimalism",
        "this room feels like peaceful Japanese Scandinavian harmony",
        "this space feels like slow living and natural imperfection",
    ],
    "Industrial loft": [
        "this room interior design feels like a raw urban factory loft",
        "this space feels like gritty New York industrial warehouse",
        "this interior feels like exposed concrete and steel masculinity",
        "this room feels like a converted factory apartment",
        "this space feels like urban rough industrial aesthetic",
    ],
    "Bohemian eclectic": [
        "this room interior design feels like free-spirited bohemian eclectic",
        "this space feels like a colourful artist boho studio",
        "this interior feels like global nomadic bohemian wanderer",
        "this room feels like layered bohemian textile warmth",
        "this space feels like eclectic creative boho sanctuary",
    ],
    "Contemporary Indian": [
        "this room interior design feels like modern Indian luxury fusion",
        "this space feels like a contemporary Indian urban home",
        "this interior feels like Indian modernity with brass and stone",
        "this room feels like fusion Indian contemporary elegance",
        "this space feels like modern India blending old and new",
    ],
    "Traditional Indian": [
        "this room interior design feels like rich Indian heritage grandeur",
        "this space feels like a royal Indian palace interior",
        "this interior feels like traditional Indian craftsmanship and silk",
        "this room feels like ethnic Indian cultural richness",
        "this space feels like timeless classic Indian home warmth",
    ],
    "Art Deco": [
        "this room interior design feels like glamorous 1920s art deco opulence",
        "this space feels like jazz age gold and geometric luxury",
        "this interior feels like Hollywood regency glamour and drama",
        "this room feels like decadent art deco grandeur and boldness",
        "this space feels like theatrical golden age luxury",
    ],
    "Mid-Century Modern": [
        "this room interior design feels like optimistic 1950s retro modern",
        "this space feels like mid-century American suburban cool",
        "this interior feels like Eames era warm walnut and mustard",
        "this room feels like cheerful vintage modernist design",
        "this space feels like retro-future atomic age warmth",
    ],
    "Coastal beach house": [
        "this room interior design feels like breezy coastal beach house relaxation",
        "this space feels like a sun-washed seaside cottage retreat",
        "this interior feels like driftwood and sea glass tranquillity",
        "this room feels like fresh coastal ocean breeze and linen",
        "this space feels like a relaxed beachside vacation home",
    ],
    "Farmhouse rustic": [
        "this room interior design feels like warm rustic farmhouse comfort",
        "this space feels like a cosy rural countryside farmhouse",
        "this interior feels like reclaimed wood and simple country charm",
        "this room feels like wholesome farmhouse family warmth",
        "this space feels like a rural farmhouse retreat",
    ],
}

_GEMINI_ALIAS: Dict[str, List[str]] = {
    "Modern Minimalist":   ["modern", "minimalist", "minimal", "contemporary minimalist"],
    "Scandinavian":        ["scandinavian", "nordic", "scandi"],
    "Japandi":             ["japandi", "japanese", "zen"],
    "Industrial loft":     ["industrial", "loft", "urban industrial"],
    "Bohemian eclectic":   ["bohemian", "boho", "eclectic"],
    "Contemporary Indian": ["contemporary indian", "modern indian", "fusion indian"],
    "Traditional Indian":  ["traditional indian", "classic indian", "ethnic"],
    "Art Deco":            ["art deco", "deco"],
    "Mid-Century Modern":  ["mid-century", "mid century", "retro modern"],
    "Coastal beach house": ["coastal", "beach", "nautical"],
    "Farmhouse rustic":    ["farmhouse", "rustic", "country"],
}

# ── Weights directory ────────────────────────────────────────────────────────
_WEIGHTS_DIR = Path(os.getenv("ML_WEIGHTS_DIR", "/app/ml/weights"))

# Local dev fallback
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if not _WEIGHTS_DIR.exists():
    _local_weights = _BACKEND_DIR / "ml" / "weights"
    if _local_weights.exists():
        _WEIGHTS_DIR = _local_weights


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Fine-tuned EfficientNet-B0 Style Classifier
# ─────────────────────────────────────────────────────────────────────────────

class _FineTunedStyleNet:
    """
    EfficientNet-B0 fine-tuned on your real interior design dataset.
    5 classes: boho, industrial, minimalist, modern, scandinavian.
    Loaded from ml/weights/style_classifier.pt.
    """
    _model   = None
    _device  = None
    _ready: bool = False
    _label_to_idx: Dict[str, int] = {
        "boho": 0, "industrial": 1, "minimalist": 2, "modern": 3, "scandinavian": 4
    }

    @classmethod
    def is_available(cls) -> bool:
        weights_path = _WEIGHTS_DIR / "style_classifier.pt"
        return weights_path.exists()

    @classmethod
    def _load(cls) -> bool:
        if cls._ready:
            return True
        weights_path = _WEIGHTS_DIR / "style_classifier.pt"
        if not weights_path.exists():
            return False
        try:
            import torch
            import torchvision.models as tv_models
            import torch.nn as nn

            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            model = tv_models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(_DATASET_STYLES))
            model.load_state_dict(
                torch.load(str(weights_path), map_location=cls._device)
            )
            model.eval().to(cls._device)
            cls._model = model
            cls._ready = True
            logger.info(
                f"[StyleClassifier-v4] Fine-tuned EfficientNet loaded "
                f"from {weights_path} on {cls._device}"
            )
            return True
        except Exception as e:
            logger.warning(f"[StyleClassifier-v4] EfficientNet load failed: {e}")
            return False

    @classmethod
    def classify(cls, image_bytes: bytes, room_type: str = "") -> Dict[str, Any]:
        """
        Returns classification result using fine-tuned EfficientNet.
        Maps 5 dataset styles → 11 ARKEN canonical styles.
        """
        if not cls._load():
            return {}
        try:
            import torch
            import torch.nn.functional as F
            import torchvision.transforms as T
            from PIL import Image as PILImage

            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(cls._device)

            with torch.no_grad():
                logits = cls._model(tensor)
                probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

            # Build result over dataset styles
            dataset_probs = {_DATASET_STYLES[i]: float(probs[i])
                             for i in range(len(_DATASET_STYLES))}
            best_ds_style = max(dataset_probs, key=dataset_probs.get)
            best_ds_conf  = dataset_probs[best_ds_style]

            # Map to ARKEN canonical
            arken_label, base_conf = _DATASET_TO_ARKEN.get(
                best_ds_style, ("Modern Minimalist", 0.70)
            )
            # Scale confidence: fine-tuned model's softmax × base mapping confidence
            final_conf = round(min(best_ds_conf * base_conf * 1.05, 0.97), 4)

            # Build top-3 over ARKEN labels
            arken_scores: Dict[str, float] = {}
            for ds_style, ds_prob in dataset_probs.items():
                a_label, a_base = _DATASET_TO_ARKEN.get(ds_style, ("Modern Minimalist", 0.7))
                arken_scores[a_label] = arken_scores.get(a_label, 0.0) + ds_prob * a_base

            top3 = sorted(arken_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_result = [
                {"style": lbl, "confidence": round(float(sc), 4)}
                for lbl, sc in top3
            ]

            return {
                "style_label":        arken_label,
                "style_confidence":   final_conf,
                "top_3_styles":       top3_result,
                "model_used":         "efficientnet_finetuned",
                "dataset_style":      best_ds_style,
                "dataset_confidence": round(best_ds_conf, 4),
                "gemini_agreement":   False,
                "metadata_trained":   True,
                "metadata_rows_used": 4139,
                "prompt_source":      "finetuned_efficientnet_real_dataset",
            }
        except Exception as e:
            logger.warning(f"[StyleClassifier-v4] EfficientNet inference failed: {e}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Fine-tuned CLIP visual encoder
# ─────────────────────────────────────────────────────────────────────────────

class _FineTunedCLIP:
    """
    CLIP ViT-B/32 with fine-tuned image encoder (clip_finetuned.pt).
    Uses the full zero-shot prompt library from v3.0 but with better
    image representations learned from your dataset.
    """
    _model     = None
    _preprocess = None
    _ready: bool = False

    @classmethod
    def is_available(cls) -> bool:
        return (_WEIGHTS_DIR / "clip_finetuned.pt").exists()

    @classmethod
    def _load(cls) -> bool:
        if cls._ready:
            return True
        ft_path = _WEIGHTS_DIR / "clip_finetuned.pt"
        if not ft_path.exists():
            return False
        try:
            import clip as openai_clip
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = openai_clip.load(
                "ViT-B/32", device=device, download_root=str(_WEIGHTS_DIR)
            )
            # Load fine-tuned visual encoder weights
            ft_state = torch.load(str(ft_path), map_location=device)
            model.visual.load_state_dict(ft_state)
            model.eval()
            cls._model      = model
            cls._preprocess = preprocess
            cls._ready      = True
            logger.info(
                f"[StyleClassifier-v4] Fine-tuned CLIP visual encoder loaded "
                f"from {ft_path} on {device}"
            )
            return True
        except Exception as e:
            logger.warning(f"[StyleClassifier-v4] Fine-tuned CLIP load failed: {e}")
            return False

    @classmethod
    def encode_image(cls, image_bytes: bytes) -> Optional[np.ndarray]:
        """Returns L2-normalised 512-dim embedding using fine-tuned visual encoder."""
        if not cls._load():
            return None
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image as PILImage

            img    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            device = next(cls._model.parameters()).device
            tensor = cls._preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = cls._model.encode_image(tensor).float()
                feats = F.normalize(feats, dim=-1)
            return feats.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"[StyleClassifier-v4] Fine-tuned CLIP encode failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Main StyleClassifier class (public API preserved from v3.0)
# ─────────────────────────────────────────────────────────────────────────────

class StyleClassifier:
    """
    Interior design style classifier v4.0.

    Three-tier classification with automatic fallback:
      Tier 1: Fine-tuned EfficientNet-B0 (when style_classifier.pt present)
              → Trained on 4,139 real images, 5 dataset styles → 11 ARKEN styles
      Tier 2: Fine-tuned CLIP (when clip_finetuned.pt present)
              → Better image embeddings + full ROOM_STYLE_PROMPTS library
      Tier 3: Zero-shot CLIP dual-pass (always available as fallback)
              → Same as v3.0

    Public API (unchanged from v3.0):
        clf = StyleClassifier()
        result = clf.classify(image_bytes, gemini_style_hint="boho", room_type="bedroom")
    """

    # Zero-shot CLIP state (Tier 3 fallback)
    _clip_model_zs     = None
    _clip_processor_zs = None
    _clip_ready_zs: bool = False

    @classmethod
    def _load_clip_zeroshot(cls) -> bool:
        """Load zero-shot CLIP (transformers-based, Tier 3 fallback)."""
        if cls._clip_ready_zs:
            return True
        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("[StyleClassifier-v4] Loading zero-shot CLIP ViT-B/32 …")
            cls._clip_processor_zs = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            cls._clip_model_zs = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            cls._clip_model_zs.eval()
            cls._clip_ready_zs = True
            logger.info("[StyleClassifier-v4] Zero-shot CLIP ready (Tier 3 fallback).")
            return True
        except Exception as e:
            logger.warning(f"[StyleClassifier-v4] Zero-shot CLIP unavailable: {e}")
            return False

    # ── Public API ───────────────────────────────────────────────────────────

    def classify(
        self,
        image_bytes: bytes,
        gemini_style_hint: str = "",
        room_type: str = "",
    ) -> Dict[str, Any]:
        """
        Classify interior design style from raw image bytes.

        Args:
            image_bytes:       JPEG/PNG/WebP raw bytes.
            gemini_style_hint: Optional style label from Gemini (blend source).
            room_type:         "kitchen"|"bedroom"|"bathroom"|"living_room"

        Returns dict with keys:
            style_label, style_confidence, top_3_styles, model_used,
            gemini_agreement, metadata_trained, metadata_rows_used, prompt_source
        """
        try:
            return self._classify_inner(image_bytes, gemini_style_hint, room_type)
        except Exception as e:
            logger.warning(f"[StyleClassifier-v4] classify() failed: {e}. Keyword fallback.")
            return self._keyword_fallback(gemini_style_hint)

    def _classify_inner(
        self,
        image_bytes: bytes,
        gemini_style_hint: str,
        room_type: str,
    ) -> Dict[str, Any]:
        rt = (room_type or "").lower().replace(" ", "_")

        # ── Tier 1: Fine-tuned EfficientNet ──────────────────────────────────
        if _FineTunedStyleNet.is_available():
            result = _FineTunedStyleNet.classify(image_bytes, rt)
            if result:
                if result["style_confidence"] < _CLIP_BLEND_THRESH and gemini_style_hint:
                    result = self._blend_with_gemini(result, gemini_style_hint)
                result["gemini_agreement"] = self._agrees_with_gemini(
                    result["style_label"], gemini_style_hint
                )
                return result

        # ── Tier 2: Fine-tuned CLIP ───────────────────────────────────────────
        if _FineTunedCLIP.is_available() and _FineTunedCLIP._load():
            result = self._clip_classify_finetuned(image_bytes, rt)
            if result.get("style_confidence", 0) >= _CLIP_THRESHOLD:
                result["gemini_agreement"] = self._agrees_with_gemini(
                    result["style_label"], gemini_style_hint
                )
                return result
            # Low confidence: blend with Gemini hint
            if gemini_style_hint:
                result = self._blend_with_gemini(result, gemini_style_hint)
            result["gemini_agreement"] = self._agrees_with_gemini(
                result["style_label"], gemini_style_hint
            )
            return result

        # ── Tier 3: Zero-shot CLIP dual-pass ─────────────────────────────────
        if not self._load_clip_zeroshot():
            return self._keyword_fallback(gemini_style_hint)

        v1_result = self._clip_classify_v1(image_bytes, rt)
        v1_conf   = v1_result["style_confidence"]

        if v1_conf >= _CLIP_THRESHOLD:
            result = {**v1_result, "model_used": "clip_v1_metadata_prompts"}
        else:
            v2_result = self._clip_classify_v2(image_bytes)
            result    = self._average_clip_passes(v1_result, v2_result)

        if result["style_confidence"] < _CLIP_BLEND_THRESH and gemini_style_hint:
            result = self._blend_with_gemini(result, gemini_style_hint)

        result["gemini_agreement"]   = self._agrees_with_gemini(
            result["style_label"], gemini_style_hint
        )
        result["metadata_trained"]   = True
        result["metadata_rows_used"] = 4139
        result["prompt_source"]      = (
            "room_style_metadata_derived" if rt in ROOM_STYLE_PROMPTS
            else "generic_arken_style_prompts"
        )
        return result

    def _clip_classify_finetuned(
        self, image_bytes: bytes, room_type: str = ""
    ) -> Dict[str, Any]:
        """
        Tier 2: Use fine-tuned CLIP visual encoder + zero-shot prompt library.
        The fine-tuned encoder produces better interior design embeddings.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image as PILImage
        import clip as openai_clip

        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        model   = _FineTunedCLIP._model
        device  = next(model.parameters()).device
        preproc = _FineTunedCLIP._preprocess

        all_prompts: List[str] = []
        style_slices: List[Tuple[int, int]] = []

        for arken_style in _SHORT_LABELS:
            csv_style = _ARKEN_TO_CSV_STYLE.get(arken_style)
            if csv_style and room_type in ROOM_STYLE_PROMPTS:
                prompts = ROOM_STYLE_PROMPTS[room_type][csv_style]
            elif arken_style in _GENERIC_PROMPTS:
                prompts = _GENERIC_PROMPTS[arken_style]
            else:
                idx     = _SHORT_LABELS.index(arken_style)
                prompts = [STYLE_LABELS[idx]]

            start = len(all_prompts)
            all_prompts.extend(prompts)
            style_slices.append((start, len(all_prompts)))

        # Encode image using fine-tuned encoder
        img_tensor = preproc(pil_img).unsqueeze(0).to(device)
        text_tokens = openai_clip.tokenize(all_prompts, truncate=True).to(device)

        with torch.no_grad():
            img_feats  = model.encode_image(img_tensor).float()
            txt_feats  = model.encode_text(text_tokens).float()
            img_norm   = F.normalize(img_feats, dim=-1)
            txt_norm   = F.normalize(txt_feats, dim=-1)
            all_sims   = (img_norm @ txt_norm.T).squeeze(0).cpu().numpy()

        clip_scores = np.zeros(len(_SHORT_LABELS))
        for i, (s, e) in enumerate(style_slices):
            clip_scores[i] = float(np.mean(all_sims[s:e]))

        clip_probs  = _softmax(clip_scores * 50.0)
        prior_array = np.array([STYLE_PRIORS.get(sl, 0.01) for sl in _SHORT_LABELS])
        prior_array = prior_array / prior_array.sum()
        blended     = _CLIP_WEIGHT * clip_probs + _PRIOR_WEIGHT * prior_array
        blended     = blended / blended.sum()

        result = self._probs_to_result(blended, model_used="clip_finetuned_v1")
        result["metadata_trained"]   = True
        result["metadata_rows_used"] = 4139
        result["prompt_source"]      = "clip_finetuned_room_style_prompts"
        return result

    def _clip_classify_v1(self, image_bytes: bytes, room_type: str = "") -> Dict[str, Any]:
        """Zero-shot CLIP Pass 1 (Tier 3 fallback)."""
        import torch
        import torch.nn.functional as F
        from PIL import Image as PILImage

        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

        all_prompts: List[str]              = []
        style_slices: List[Tuple[int, int]] = []

        for arken_style in _SHORT_LABELS:
            csv_style = _ARKEN_TO_CSV_STYLE.get(arken_style)
            if csv_style and room_type in ROOM_STYLE_PROMPTS:
                prompts = ROOM_STYLE_PROMPTS[room_type][csv_style]
            elif arken_style in _GENERIC_PROMPTS:
                prompts = _GENERIC_PROMPTS[arken_style]
            else:
                idx     = _SHORT_LABELS.index(arken_style)
                prompts = [STYLE_LABELS[idx]]

            start = len(all_prompts)
            all_prompts.extend(prompts)
            style_slices.append((start, len(all_prompts)))

        inputs = self.__class__._clip_processor_zs(
            text=all_prompts, images=pil_img,
            return_tensors="pt", padding=True, truncation=True,
        )
        with torch.no_grad():
            outputs       = self.__class__._clip_model_zs(**inputs)
            image_norm    = F.normalize(outputs.image_embeds, dim=-1)
            text_norm     = F.normalize(outputs.text_embeds,  dim=-1)
            all_sims      = (image_norm @ text_norm.T).squeeze(0).numpy()

        clip_scores = np.zeros(len(_SHORT_LABELS))
        for i, (s, e) in enumerate(style_slices):
            clip_scores[i] = float(np.mean(all_sims[s:e]))

        clip_probs  = _softmax(clip_scores * 50.0)
        prior_array = np.array([STYLE_PRIORS.get(sl, 0.01) for sl in _SHORT_LABELS])
        prior_array = prior_array / prior_array.sum()
        blended     = _CLIP_WEIGHT * clip_probs + _PRIOR_WEIGHT * prior_array
        blended     = blended / blended.sum()

        return self._probs_to_result(blended, model_used="clip_v1_metadata_prompts")

    def _clip_classify_v2(self, image_bytes: bytes) -> Dict[str, Any]:
        """Zero-shot CLIP Pass 2 — feelings prompts (Tier 3 fallback)."""
        import torch
        import torch.nn.functional as F
        from PIL import Image as PILImage

        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

        all_prompts: List[str]              = []
        style_slices: List[Tuple[int, int]] = []

        for arken_style in _SHORT_LABELS:
            prompts = _CLIP_V2_FEELINGS_PROMPTS.get(arken_style, [
                f"this room interior design feels like {arken_style.lower()}"
            ])
            start = len(all_prompts)
            all_prompts.extend(prompts)
            style_slices.append((start, len(all_prompts)))

        inputs = self.__class__._clip_processor_zs(
            text=all_prompts, images=pil_img,
            return_tensors="pt", padding=True, truncation=True,
        )
        with torch.no_grad():
            outputs    = self.__class__._clip_model_zs(**inputs)
            image_norm = F.normalize(outputs.image_embeds, dim=-1)
            text_norm  = F.normalize(outputs.text_embeds,  dim=-1)
            all_sims   = (image_norm @ text_norm.T).squeeze(0).numpy()

        clip_scores = np.zeros(len(_SHORT_LABELS))
        for i, (s, e) in enumerate(style_slices):
            clip_scores[i] = float(np.mean(all_sims[s:e]))

        clip_probs  = _softmax(clip_scores * 50.0)
        prior_array = np.array([STYLE_PRIORS.get(sl, 0.01) for sl in _SHORT_LABELS])
        prior_array = prior_array / prior_array.sum()
        blended     = 0.85 * clip_probs + 0.15 * prior_array
        blended     = blended / blended.sum()

        return self._probs_to_result(blended, model_used="clip_v2_feelings_prompts")

    def _average_clip_passes(self, v1: Dict, v2: Dict) -> Dict[str, Any]:
        """Average v1 and v2 probability vectors."""
        v1p = np.zeros(len(_SHORT_LABELS))
        v2p = np.zeros(len(_SHORT_LABELS))
        for entry in v1.get("top_3_styles", []):
            try:
                v1p[_SHORT_LABELS.index(entry["style"])] = float(entry["confidence"])
            except ValueError:
                pass
        for entry in v2.get("top_3_styles", []):
            try:
                v2p[_SHORT_LABELS.index(entry["style"])] = float(entry["confidence"])
            except ValueError:
                pass
        if v1p.sum() < 0.01:
            try:
                v1p[_SHORT_LABELS.index(v1["style_label"])] = v1["style_confidence"]
            except ValueError:
                pass
        if v2p.sum() < 0.01:
            try:
                v2p[_SHORT_LABELS.index(v2["style_label"])] = v2["style_confidence"]
            except ValueError:
                pass
        if v1p.sum() > 0:
            v1p /= v1p.sum()
        if v2p.sum() > 0:
            v2p /= v2p.sum()
        avg = (0.5 * v1p + 0.5 * v2p)
        avg /= max(avg.sum(), 1e-8)
        return self._probs_to_result(avg, model_used="clip_dual_pass")

    # ── Static helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _keyword_fallback(gemini_style_hint: str) -> Dict[str, Any]:
        hint_lower = (gemini_style_hint or "").lower()
        matched    = "Modern Minimalist"
        for short_label, aliases in _GEMINI_ALIAS.items():
            if any(a in hint_lower for a in aliases):
                matched = short_label
                break
        top3 = [
            {"style": matched,               "confidence": 0.55},
            {"style": "Contemporary Indian", "confidence": 0.25},
            {"style": "Modern Minimalist",   "confidence": 0.20},
        ]
        seen: set = set()
        top3 = [t for t in top3 if not (t["style"] in seen or seen.add(t["style"]))]
        return {
            "style_label":        matched,
            "style_confidence":   0.45,
            "top_3_styles":       top3[:3],
            "model_used":         "keyword_rules",
            "gemini_agreement":   True,
            "metadata_trained":   True,
            "metadata_rows_used": 4139,
            "prompt_source":      "keyword_fallback",
        }

    @staticmethod
    def _blend_with_gemini(model_result: Dict, gemini_hint: str) -> Dict[str, Any]:
        hint_lower    = (gemini_hint or "").lower()
        matched_label = model_result["style_label"]
        for short_label, aliases in _GEMINI_ALIAS.items():
            if any(a in hint_lower for a in aliases):
                matched_label = short_label
                break
        return {
            **model_result,
            "style_label":      matched_label,
            "style_confidence": min(model_result["style_confidence"] * 0.6 + 0.30, 0.65),
            "model_used":       model_result.get("model_used", "clip") + "+gemini_blend",
        }

    @staticmethod
    def _probs_to_result(probs: np.ndarray, model_used: str) -> Dict[str, Any]:
        top_indices = np.argsort(probs)[::-1]
        top3 = [
            {"style": _SHORT_LABELS[i], "confidence": round(float(probs[i]), 4)}
            for i in top_indices[:3]
        ]
        return {
            "style_label":        _SHORT_LABELS[top_indices[0]],
            "style_confidence":   round(float(probs[top_indices[0]]), 4),
            "top_3_styles":       top3,
            "model_used":         model_used,
            "gemini_agreement":   False,
            "metadata_trained":   True,
            "metadata_rows_used": 4139,
            "prompt_source":      "room_style_metadata_derived",
        }

    @staticmethod
    def _agrees_with_gemini(style_label: str, gemini_hint: str) -> bool:
        if not gemini_hint:
            return False
        hint_lower  = gemini_hint.lower()
        label_lower = style_label.lower()
        if label_lower in hint_lower or hint_lower in label_lower:
            return True
        for short_label, aliases in _GEMINI_ALIAS.items():
            if short_label.lower() in label_lower:
                return any(a in hint_lower for a in aliases)
        return False

    # ── Dataset validation ───────────────────────────────────────────────────

    @classmethod
    def validate_on_dataset(
        cls,
        val_csv_path: Optional[str] = None,
        image_root: Optional[str] = None,
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Evaluate on val_data.csv.
        Uses fine-tuned EfficientNet if available (Tier 1),
        then fine-tuned CLIP (Tier 2), then zero-shot (Tier 3).
        Saves to ml/weights/style_classifier_eval.json.
        """
        import csv as csv_module

        backend_dir = Path(__file__).resolve().parent.parent

        # Locate val_data.csv
        if val_csv_path:
            csv_path = Path(val_csv_path)
        else:
            for candidate in [
                backend_dir / "data" / "datasets" / "interior_design_material_style" / "val_data.csv",
                backend_dir / "data" / "datasets" / "interior_design_images_metadata" / "val_data.csv",
            ]:
                if candidate.exists():
                    csv_path = candidate
                    break
            else:
                return {"error": "val_data.csv not found"}

        # Locate image root
        if image_root:
            img_root = Path(image_root)
        else:
            for candidate in [
                backend_dir / "data" / "datasets" / "interior_design_material_style",
                backend_dir / "data" / "datasets" / "interior_design_images_metadata",
            ]:
                if candidate.exists():
                    img_root = candidate
                    break
            else:
                img_root = csv_path.parent

        records: List[Tuple[Path, str, str]] = []
        with open(csv_path, "r", encoding="utf-8-sig") as fh:
            for row in csv_module.DictReader(fh):
                raw_path  = row.get("image_path", "").replace("\\", "/")
                room_type = row.get("room_type", "").strip()
                style     = row.get("style", "").strip()
                if not style:
                    continue
                filename = Path(raw_path).name
                for root in [img_root,
                              backend_dir / "data" / "datasets" / "interior_design_material_style",
                              backend_dir / "data" / "datasets" / "interior_design_images_metadata"]:
                    c = root / room_type / style / filename
                    if c.exists():
                        records.append((c, room_type, style))
                        break

        if not records:
            return {"error": "No image files resolved from val_data.csv", "csv_path": str(csv_path)}

        records = records[:max_samples]
        clf = cls()

        # Determine which tier will be used
        if _FineTunedStyleNet.is_available():
            model_tier = "tier1_efficientnet_finetuned"
        elif _FineTunedCLIP.is_available():
            model_tier = "tier2_clip_finetuned"
        else:
            model_tier = "tier3_clip_zeroshot"

        true_labels: List[str] = []
        pred_labels: List[str] = []
        model_used_counts: Dict[str, int] = {}
        errors = 0

        for img_path, room_type, true_style in records:
            try:
                result     = clf.classify(img_path.read_bytes(), room_type=room_type)
                pred       = result["style_label"]
                mu         = result.get("model_used", "unknown")
                model_used_counts[mu] = model_used_counts.get(mu, 0) + 1
                arken_true = _CSV_TO_ARKEN.get(true_style.lower(), true_style)
                true_labels.append(arken_true)
                pred_labels.append(pred)
            except Exception as e:
                logger.debug(f"[validate] Skipped {img_path.name}: {e}")
                errors += 1

        if not true_labels:
            return {"error": "No images could be evaluated", "errors": errors}

        try:
            from sklearn.metrics import accuracy_score, f1_score
            accuracy   = float(accuracy_score(true_labels, pred_labels))
            all_labels = sorted(set(true_labels + pred_labels))
            f1_arr     = f1_score(true_labels, pred_labels, labels=all_labels,
                                   average=None, zero_division=0)
            per_class_f1 = {lbl: round(float(f1_arr[i]), 4)
                            for i, lbl in enumerate(all_labels)}
            macro_f1 = float(np.mean(list(per_class_f1.values())))
        except ImportError:
            correct      = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
            accuracy     = correct / len(true_labels)
            per_class_f1 = {}
            macro_f1     = accuracy

        results = {
            "accuracy":          round(accuracy, 4),
            "macro_f1":          round(macro_f1, 4),
            "per_class_f1":      per_class_f1,
            "samples_evaluated": len(true_labels),
            "errors_skipped":    errors,
            "model_tier":        model_tier,
            "model_used_counts": model_used_counts,
            "csv_path":          str(csv_path),
        }

        try:
            out_path = _WEIGHTS_DIR / "style_classifier_eval.json"
            _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(results, fh, indent=2)
            logger.info(
                f"[StyleClassifier-v4.validate] "
                f"accuracy={accuracy:.3f} macro_f1={macro_f1:.3f} "
                f"tier={model_tier} n={len(true_labels)} → {out_path}"
            )
        except Exception as e:
            logger.warning(f"[StyleClassifier-v4.validate] Could not save eval JSON: {e}")

        return results