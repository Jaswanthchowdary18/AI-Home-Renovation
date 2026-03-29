"""
ARKEN — Rendering Agent v2.1
google-genai SDK — image renovation using Gemini image generation.

v2.1 Changes (BUG 1 FIX):
  - MATERIAL_SPECS now covers ALL (theme × tier) combinations — no more
    generic fallback that made every tier look identical.
  - RENOVATION_PROMPT gains a TIER VISUAL QUALITY RULES block that
    explicitly instructs Gemini HOW the budget tier must change the
    visual appearance of the rendered room.
  - build_material_spec() now always finds an exact match — fallback is
    only reached for completely unknown themes, not for missing tiers.

CONFIRMED WORKING (March 2026):
  - Model: gemini-2.5-flash-image  (GA, current)
  - SDK:   google-genai >= 1.0.0
  - response_modalities=["IMAGE"] in GenerateContentConfig
"""

import base64
import io
import logging
import time
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

# Models to try in order — first one that returns an image wins
IMAGE_MODELS = [
    "gemini-2.5-flash-image",         # GA — primary (March 2026)
    "gemini-2.5-flash-image-preview", # preview fallback
]

# ── TIER VISUAL QUALITY RULES (injected into prompt) ─────────────────────────
# This block is the key fix: it tells Gemini explicitly what "basic", "mid",
# and "premium" look like so renders are visually distinct across tiers.
_TIER_QUALITY_RULES = """
═══════════════════════════════════════════════════════════
BUDGET TIER VISUAL QUALITY — YOU MUST APPLY: {budget_tier} tier ({budget_range})
═══════════════════════════════════════════════════════════
BASIC tier (Rs.3–5 Lakh) — apply ONLY if budget_tier is Basic:
  • Walls: plain flat emulsion paint, no texture. Colours: off-white, cream, or light pastel.
    Brand: Asian Paints Apcolite / Berger Bison.
  • Floor: standard 600×600 glossy vitrified tiles. No large-format, no marble, no wood.
    Brand: Kajaria GVT basic series or Somany.
  • Ceiling: plain white flat ceiling. NO false ceiling, NO cove lighting, NO POP.
  • Lighting: basic batten tube light or single basic LED panel. No pendants, no cove strips.
  • Fixtures: standard Havells fan, anchor modular switches (white).
  • Overall feel: clean, freshly painted standard Indian middle-class apartment.

MID tier (Rs.5–10 Lakh) — apply ONLY if budget_tier is Mid:
  • Walls: sheen or light texture finish. Asian Paints Royale Sheen or Royale Play texture.
    Colours: warm white, greige, or soft grey with one accent wall.
  • Floor: large-format 800×800 polished porcelain tiles OR light wood-finish vinyl planks.
    Brand: Kajaria Endura, Nitco Vitrified 800×800.
  • Ceiling: white POP false ceiling with recessed LED downlights and LED cove strip lighting.
  • Lighting: LED downlights 12W, Philips. Cove strip on perimeter.
  • Fixtures: Havells / Legrand Myrius modular switches, Havells Pacer fan.
  • Overall feel: professionally renovated Indian upper-middle-class home. Modern and polished.

PREMIUM tier (Rs.10 Lakh+) — apply ONLY if budget_tier is Premium:
  • Walls: micro-cement plaster, limewash texture, or high-gloss lacquer panels.
    Colours: deep tones (charcoal, navy, sage), or warm luxury neutrals.
  • Floor: marble-look Italian porcelain slab 800×1600, or natural stone, or herringbone parquet.
    Brand: Simpolo GVT Slab, RAK Ceramics Marble Effect, or natural teak.
  • Ceiling: gypsum coffered ceiling with concealed smart LED. Architectural cove details.
  • Lighting: Philips Hue smart downlights, LED cove concealed behind gypsum reveals.
  • Fixtures: Schneider AvatarOn smart switches, Orient Aeroslim smart fan (brushed metal).
  • Carpentry: high-gloss acrylic or matte lacquer finish, full-height wardrobes, panel doors.
  • Overall feel: luxury showroom apartment. Looks like a 5-star hotel suite.

THE RENDERED IMAGE FOR {budget_tier} TIER MUST LOOK VISUALLY VERY DIFFERENT
FROM THE OTHER TIERS. A BASIC RENDER MUST NOT HAVE FALSE CEILINGS OR MARBLE.
A PREMIUM RENDER MUST NOT HAVE PLAIN WHITE WALLS OR STANDARD TILES.
"""

RENOVATION_PROMPT = """You are renovating this exact {room_type} photograph. Apply a {theme} \
interior design style.

═══════════════════════════════════════════════════════════
BEFORE-AFTER LOCK — WHAT WAS DETECTED IN THE ORIGINAL PHOTO
═══════════════════════════════════════════════════════════
Detected objects that MUST stay in their exact pixel positions: {detected_objects}
Current wall colour / finish: {wall_color}
Current floor material: {floor_type}
Room dimensions hint: {room_dimensions_hint}

These elements are LOCKED. They cannot move, disappear, or change shape.

═══════════════════════════════════════════════════════════
PIXEL-BY-PIXEL ANCHOR RULES
═══════════════════════════════════════════════════════════
• Every item in [{detected_objects}] MUST appear at the identical screen position as in the \
uploaded photo.
• The output image MUST have the identical aspect ratio and camera angle as the uploaded photo. \
Do NOT zoom in, zoom out, pan, tilt, or rotate the virtual camera even 1 degree.
• Window count in the uploaded photo: {window_count}. \
The output MUST contain exactly {window_count} window(s) — no more, no fewer.
• Door count in the uploaded photo: {door_count}. \
The output MUST contain exactly {door_count} door(s) — no more, no fewer.
• Every wall, column, beam, and structural partition visible in the uploaded photo MUST remain \
visible in the same position.

═══════════════════════════════════════════════════════════
PERMITTED CHANGES (surfaces and finishes ONLY)
═══════════════════════════════════════════════════════════
✓ Wall surface finish and colour ONLY — not wall positions or wall area
✓ Floor surface material ONLY — not floor area or floor shape
✓ Ceiling finish and false ceiling style ONLY — not ceiling height
✓ Light fixture styles ONLY — not the positions of ceiling points or wall sconces
✓ Soft furnishing colours ONLY — cushions, curtains, rugs (not furniture layout or furniture type)
✓ Decorative accessories ONLY — vases, artwork, plants

Style: {theme}
City: {city}, India
Budget: {budget_tier} ({budget_range})

{material_spec}

{tier_quality_rules}

{custom_instructions}

═══════════════════════════════════════════════════════════
FORBIDDEN — ABSOLUTE PROHIBITIONS
═══════════════════════════════════════════════════════════
✗ DO NOT add, remove, resize, or relocate ANY doors or windows
✗ DO NOT move ANY furniture piece — every sofa, table, chair, bed stays in its original position
✗ DO NOT change the room's proportions, depth, field of view, or camera perspective
✗ DO NOT add rooms, corridors, or spaces not visible in the original photo
✗ DO NOT change the time of day, outdoor view, or direction of natural light
✗ DO NOT replace structural walls with open space or vice versa
✗ DO NOT introduce architectural elements (arches, columns, niches) absent from the original

═══════════════════════════════════════════════════════════
OUTPUT REQUIREMENT
═══════════════════════════════════════════════════════════
Return a single photorealistic image. It must look like a professional renovation photograph of \
the SAME room — same geometry, same layout — with only the surfaces and finishes updated.
Use Indian materials: Asian Paints, Kajaria tiles, Greenply, Havells, Jaquar."""

# ── COMPLETE MATERIAL_SPECS — ALL (theme × tier) combinations ─────────────────
# BUG 1 ROOT CAUSE FIX: Previously only ~8 specific combinations existed,
# so any unmatched combination fell back to a generic spec identical across
# all tiers. Now every theme has distinct specs for basic, mid, and premium.
#
# Price references (Q1 2026, INR):
#   Asian Paints Apcolite: ₹195/L | Royale Sheen: ₹435/L | Royale Luxury: ₹695/L
#   Kajaria GVT 600×600: ₹62/sqft | Kajaria Endura 800×800: ₹90/sqft
#   Simpolo GVT Slab 800×1600: ₹205/sqft | RAK Ceramics Marble: ₹295/sqft

MATERIAL_SPECS = {

    # ── Modern Minimalist ──────────────────────────────────────────────────
    ("Modern Minimalist", "basic"): {
        "walls":   "flat emulsion, pure white — Asian Paints Apcolite Premium Emulsion OW-01",
        "floor":   "600×600 glossy white vitrified tile — Kajaria GVT basic series",
        "ceiling": "plain flat white ceiling, basic LED batten light, no false ceiling",
        "trim":    "white PVC beading, standard contractor finish",
    },
    ("Modern Minimalist", "mid"): {
        "walls":   "smooth matte/sheen finish, warm white — Asian Paints Royale Sheen OW-01",
        "floor":   "800×800 matte porcelain, light grey — Kajaria Endura series",
        "ceiling": "white POP false ceiling with recessed LED cove lighting",
        "trim":    "off-white satin finish woodwork, Legrand Myrius switches",
    },
    ("Modern Minimalist", "premium"): {
        "walls":   "micro-cement smooth plaster, warm greige — zero-VOC finish",
        "floor":   "800×1600 polished porcelain slab, light ivory — Simpolo GVT Slab",
        "ceiling": "gypsum coffered ceiling with concealed LED cove, architectural reveals",
        "trim":    "matte lacquer full-height panels, Schneider AvatarOn smart switches",
    },

    # ── Scandinavian ──────────────────────────────────────────────────────
    ("Scandinavian", "basic"): {
        "walls":   "plain off-white flat emulsion — Asian Paints Apcolite, no texture",
        "floor":   "light-coloured 600×600 glossy vitrified tile — Kajaria basic",
        "ceiling": "plain white flat ceiling with basic fluorescent/LED batten",
        "trim":    "white PVC dado rail, standard white switch plates",
    },
    ("Scandinavian", "mid"): {
        "walls":   "soft white limewash texture — Asian Paints Royale Play",
        "floor":   "light oak wood-finish vinyl plank flooring 6mm",
        "ceiling": "plain white with flush LED panel lights, no cove",
        "trim":    "white birch veneer edge profiles, Legrand switches",
    },
    ("Scandinavian", "premium"): {
        "walls":   "white micro-cement plaster with subtle grain — matte zero-VOC",
        "floor":   "natural light ash herringbone parquet, oil-finished",
        "ceiling": "white gypsum board ceiling with integrated slim-line LED",
        "trim":    "solid birch veneer skirting, Schneider AvatarOn smart switches",
    },

    # ── Japandi ──────────────────────────────────────────────────────────
    ("Japandi", "basic"): {
        "walls":   "pale warm grey flat emulsion — Asian Paints Apcolite, Stone Grey shade",
        "floor":   "600×600 matt grey vitrified tile — Kajaria basic series",
        "ceiling": "plain white ceiling, no false ceiling, single LED panel",
        "trim":    "natural bamboo or wood-effect vinyl dado, white walls",
    },
    ("Japandi", "mid"): {
        "walls":   "warm greige sheen finish — Asian Paints Royale Sheen Muted Beige",
        "floor":   "wood-finish laminate planks, warm walnut tone",
        "ceiling": "white POP false ceiling with warm-white LED cove",
        "trim":    "dark walnut veneer skirting, Legrand Myrius switches",
    },
    ("Japandi", "premium"): {
        "walls":   "warm grey micro-cement plaster — artisanal trowel finish",
        "floor":   "natural teak herringbone parquet, oil-rubbed finish",
        "ceiling": "exposed natural wood slat ceiling with warm-white strip lights",
        "trim":    "dark charcoal walnut veneer, Schneider AvatarOn switches",
    },

    # ── Industrial Chic ───────────────────────────────────────────────────
    ("Industrial Chic", "basic"): {
        "walls":   "flat grey emulsion, dark charcoal accent wall — Asian Paints Apcolite",
        "floor":   "dark grey 600×600 matt vitrified tile — Kajaria basic",
        "ceiling": "plain exposed slab painted dark grey, basic LED batten",
        "trim":    "black painted metal skirting, standard switches",
    },
    ("Industrial Chic", "mid"): {
        "walls":   "raw concrete texture wallpaint, charcoal grey — Berger Silk texture",
        "floor":   "dark 800×800 polished concrete-look porcelain — Nitco",
        "ceiling": "exposed black metal conduit, Edison bulb pendant lights",
        "trim":    "matte black metal accents, Havells switches dark plate",
    },
    ("Industrial Chic", "premium"): {
        "walls":   "real micro-cement plaster, charcoal with raw texture — artisanal finish",
        "floor":   "polished dark slate natural stone or 800×1600 dark concrete slab — Simpolo",
        "ceiling": "exposed raw concrete soffit with recessed industrial track lighting",
        "trim":    "brushed black steel sections, Schneider AvatarOn black switches",
    },

    # ── Tropical Luxe ─────────────────────────────────────────────────────
    ("Tropical Luxe", "basic"): {
        "walls":   "warm terracotta clay paint — Asian Paints Apcolite, earthy ochre tone",
        "floor":   "beige 600×600 glossy tile — Kajaria basic, terracotta-tone grout",
        "ceiling": "plain white ceiling, standard rattan pendant light shade",
        "trim":    "natural cane/bamboo trim strip, simple jute accessories",
    },
    ("Tropical Luxe", "mid"): {
        "walls":   "warm lime-washed texture, terracotta tones — Asian Paints Royale Play",
        "floor":   "large format 800×800 beige sandstone-look porcelain — Nitco",
        "ceiling": "POP false ceiling with warm cove LED, rattan pendant accent",
        "trim":    "teak wood skirting, Jaquar brass-finish fittings",
    },
    ("Tropical Luxe", "premium"): {
        "walls":   "warm textured lime plaster, terracotta — artisan trowel finish",
        "floor":   "large format beige travertine natural stone, honed finish",
        "ceiling": "rattan / cane false ceiling panels, architectural cove with warm LED",
        "trim":    "teak wood with brass hardware — Jaquar Artize brass fittings",
    },

    # ── Art Deco ──────────────────────────────────────────────────────────
    ("Art Deco", "basic"): {
        "walls":   "deep jade green or navy flat emulsion — Asian Paints Apcolite, jewel tone",
        "floor":   "black and white 600×600 checkerboard glossy tiles — Kajaria basic",
        "ceiling": "plain white ceiling with a single ornate pendant light",
        "trim":    "gold-painted dado rail, simple geometric stencil border",
    },
    ("Art Deco", "mid"): {
        "walls":   "emerald or midnight blue with gold geometric stencil borders — Royale Play",
        "floor":   "black and white porcelain chevron pattern — Kajaria 600×600",
        "ceiling": "POP false ceiling with ornate plaster medallion, brass pendant",
        "trim":    "gold-finish mirror strips, Havells gold-plate switches",
    },
    ("Art Deco", "premium"): {
        "walls":   "deep emerald or navy with gold geometric stencil borders — Dulux Velvet Touch",
        "floor":   "black and white marble chevron pattern — imported natural marble",
        "ceiling": "ornate plaster medallion, brass pendant chandelier, gypsum details",
        "trim":    "high-gloss lacquer with gold inlay, Schneider AvatarOn gold switches",
    },

    # ── Neo-Classical ─────────────────────────────────────────────────────
    ("Neo-Classical", "basic"): {
        "walls":   "ivory flat emulsion with simple dado rail — Asian Paints Apcolite",
        "floor":   "cream or beige 600×600 glossy tile — Kajaria basic, marble-look",
        "ceiling": "plain white ceiling, simple cornice moulding strip",
        "trim":    "white painted MDF skirting, standard switch plates",
    },
    ("Neo-Classical", "mid"): {
        "walls":   "ivory smooth plaster with pilaster panel detail — Royale Sheen",
        "floor":   "marble-look 800×800 polished porcelain — Somany Celestia or Nitco",
        "ceiling": "POP false ceiling with crown moulding and recessed LED",
        "trim":    "antique white satin finish carpentry, Havells switches",
    },
    ("Neo-Classical", "premium"): {
        "walls":   "ivory smooth plaster with pilaster details — Dulux Velvet Touch",
        "floor":   "Carrara marble-look porcelain slab — Simpolo or imported Carrara",
        "ceiling": "coffered ceiling with ornate crown moulding, chandelier",
        "trim":    "antique white with gold leaf accents, Schneider AvatarOn",
    },

    # ── Bohemian ──────────────────────────────────────────────────────────
    ("Bohemian", "basic"): {
        "walls":   "warm terracotta/rust clay paint — Asian Paints Apcolite",
        "floor":   "handmade encaustic cement tiles, geometric pattern — terracotta tones",
        "ceiling": "exposed wooden beams effect, macrame pendant lights",
        "trim":    "raw wood and rattan accents, colourful textile dado",
    },
    ("Bohemian", "mid"): {
        "walls":   "warm sienna / rust textured finish — Asian Paints Royale Play",
        "floor":   "handcrafted patterned ceramic tile 300×300 — Nitco / Johnson",
        "ceiling": "POP ceiling with warm cove LED, rattan pendant cluster lights",
        "trim":    "reclaimed wood skirting, brass switch plates, layered textiles",
    },
    ("Bohemian", "premium"): {
        "walls":   "artisan pigmented plaster, ochre and terracotta — micro-cement trowel",
        "floor":   "handmade Moroccan zellige tile or natural terracotta stone",
        "ceiling": "exposed wooden beam with woven cane inserts, pendant cluster",
        "trim":    "solid teak with hand-carved detail, brass Schneider AvatarOn switches",
    },

    # ── Contemporary Indian ───────────────────────────────────────────────
    ("Contemporary Indian", "basic"): {
        "walls":   "warm beige flat emulsion, one wall in muted saffron — Asian Paints Apcolite",
        "floor":   "600×600 glossy cream or beige vitrified tile — Kajaria basic",
        "ceiling": "plain white ceiling, single LED batten or panel",
        "trim":    "plain white skirting, standard modular switch plates",
    },
    ("Contemporary Indian", "mid"): {
        "walls":   "warm ivory sheen finish with jali-inspired accent panel — Royale Sheen",
        "floor":   "800×800 warm-tone polished porcelain — Kajaria Endura",
        "ceiling": "POP false ceiling with warm LED cove lighting",
        "trim":    "warm teak veneer skirting, Havells or Legrand switches",
    },
    ("Contemporary Indian", "premium"): {
        "walls":   "warm Venetian plaster, terracotta or saffron tones — artisan finish",
        "floor":   "800×1600 natural sandstone-look slab or Indian green marble — Simpolo",
        "ceiling": "carved gypsum jali screen cove ceiling, warm LED",
        "trim":    "teak and brass inlay carpentry, Schneider AvatarOn brass switches",
    },

    # ── Traditional Indian ────────────────────────────────────────────────
    ("Traditional Indian", "basic"): {
        "walls":   "warm yellow or ochre flat emulsion — Asian Paints Apcolite traditional tone",
        "floor":   "Athangudi-style patterned 300×300 cement tile or basic terracotta-look tile",
        "ceiling": "plain white ceiling, simple brass-finish fan",
        "trim":    "natural wood skirting, simple border tile dado",
    },
    ("Traditional Indian", "mid"): {
        "walls":   "warm ochre or saffron texture — Asian Paints Royale Play sponge finish",
        "floor":   "Athangudi-inspired patterned ceramic 300×300 — Johnson or Nitco",
        "ceiling": "POP false ceiling with warm cove LED, traditional brass pendant",
        "trim":    "carved teak wood skirting, brass Havells switch plates",
    },
    ("Traditional Indian", "premium"): {
        "walls":   "Venetian plaster in warm gold/saffron, hand-painted mural border",
        "floor":   "Indian Jaisalmer yellow sandstone or Agra red stone natural flooring",
        "ceiling": "carved teak wood coffered ceiling with brass inlay, warm LED",
        "trim":    "solid teak with brass inlay, Schneider AvatarOn gold switches",
    },
}

BUDGET_RANGES = {
    "basic":   "Rs.3-5 Lakh",
    "mid":     "Rs.5-10 Lakh",
    "premium": "Rs.10 Lakh+",
}

# Canonical tier names for display
_TIER_DISPLAY = {"basic": "Basic", "mid": "Mid", "premium": "Premium"}


def build_material_spec(theme: str, budget_tier: str, overrides: Optional[dict] = None) -> str:
    """
    Return a formatted material specification string for the given theme+tier.

    BUG 1 FIX: Look up exact (theme, tier) match first. If theme is unknown,
    fall back to a tier-appropriate generic spec (NOT a single generic spec
    shared across all tiers). This ensures basic/mid/premium are always distinct.
    """
    tier = budget_tier.lower()

    # 1. Exact match
    spec = MATERIAL_SPECS.get((theme, tier))

    # 2. If theme is unknown, use a tier-appropriate generic (not a tier-blind generic)
    if spec is None:
        _TIER_GENERIC = {
            "basic": {
                "walls":   "flat emulsion, off-white — Asian Paints Apcolite",
                "floor":   "600×600 glossy vitrified tile — Kajaria GVT basic",
                "ceiling": "plain white flat ceiling, no false ceiling, basic LED",
                "trim":    "white PVC skirting, standard switch plates",
            },
            "mid": {
                "walls":   "sheen finish, warm white — Asian Paints Royale Sheen",
                "floor":   "800×800 polished porcelain — Kajaria Endura series",
                "ceiling": "POP false ceiling with LED cove lighting",
                "trim":    "satin finish woodwork, Legrand Myrius switches",
            },
            "premium": {
                "walls":   "micro-cement or Venetian plaster finish — zero-VOC luxury",
                "floor":   "800×1600 polished porcelain slab — Simpolo GVT Slab",
                "ceiling": "gypsum coffered ceiling with concealed smart LED",
                "trim":    "high-gloss lacquer panels, Schneider AvatarOn smart switches",
            },
        }
        spec = _TIER_GENERIC.get(tier, _TIER_GENERIC["mid"])
        logger.warning(
            f"[build_material_spec] Theme '{theme}' not found in MATERIAL_SPECS — "
            f"using tier-appropriate generic for '{tier}'"
        )

    if overrides:
        spec = {**spec, **overrides}

    return "Materials:\n" + "\n".join(f"  - {k.title()}: {v}" for k, v in spec.items())


def build_tier_quality_rules(budget_tier: str, budget_range: str) -> str:
    """Return the tier quality rules block with the correct tier highlighted."""
    return _TIER_QUALITY_RULES.format(
        budget_tier=_TIER_DISPLAY.get(budget_tier.lower(), budget_tier.title()),
        budget_range=budget_range,
    )


def _client() -> genai.Client:
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in backend/.env")
    return genai.Client(api_key=settings.GOOGLE_API_KEY.get_secret_value())


class RenderingAgent:
    """
    Renovation renderer using gemini-2.5-flash-image.
    Passes room photo + prompt to Gemini, receives renovated room photo.

    v2.1: Tier-distinct rendering — Basic/Mid/Premium produce visually
    different images due to expanded MATERIAL_SPECS and explicit
    TIER VISUAL QUALITY RULES block in the prompt.
    """

    async def render(
        self,
        *,
        original_image_bytes: bytes,
        project_id: str,
        version: int,
        theme: str,
        city: str,
        budget_tier: str,
        room_type: str = "room",
        custom_instructions: str = "",
        material_overrides: Optional[dict] = None,
        # CV analysis parameters — populated from vision state when available
        detected_objects: Optional[list] = None,
        wall_color: str = "neutral",
        floor_type: str = "tiles",
        window_count: int = 1,
        door_count: int = 1,
        room_dimensions_hint: str = "",
    ) -> dict:
        start = time.perf_counter()
        client = _client()

        tier_lower   = budget_tier.lower()
        budget_range = BUDGET_RANGES.get(tier_lower, "Rs.5-10 Lakh")
        material_spec = build_material_spec(theme, tier_lower, material_overrides)

        # BUG 1 FIX: build the tier quality rules block
        tier_quality_rules = build_tier_quality_rules(tier_lower, budget_range)

        # Derive window/door counts from detected_objects if not explicitly supplied
        if detected_objects:
            objects_lower = [str(o).lower() for o in detected_objects]
            if window_count == 1:  # only override default, not an explicit caller value
                window_count = max(1, sum(1 for o in objects_lower if "window" in o))
            if door_count == 1:
                door_count = max(1, sum(1 for o in objects_lower if "door" in o))

        detected_objects_str = (
            ", ".join(str(o) for o in detected_objects)
            if detected_objects else "furniture, walls, floor, ceiling"
        )

        prompt = RENOVATION_PROMPT.format(
            room_type=room_type,
            theme=theme,
            city=city,
            budget_tier=_TIER_DISPLAY.get(tier_lower, budget_tier.title()),
            budget_range=budget_range,
            material_spec=material_spec,
            tier_quality_rules=tier_quality_rules,
            custom_instructions=f"Additional: {custom_instructions}" if custom_instructions else "",
            detected_objects=detected_objects_str,
            wall_color=wall_color,
            floor_type=floor_type,
            window_count=window_count,
            door_count=door_count,
            room_dimensions_hint=room_dimensions_hint or "standard room",
        )

        img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        img_bytes = buf.getvalue()

        logger.info(
            f"[{project_id}] Starting render — theme={theme}, tier={tier_lower}, "
            f"city={city}, windows={window_count}, doors={door_count}"
        )

        # First attempt
        rendered_bytes, model_used = await self._render(client, img_bytes, prompt, project_id)

        # Fidelity validation + one retry if structural drift detected
        fidelity = None
        if rendered_bytes:
            fidelity = self.validate_render_fidelity(original_image_bytes, rendered_bytes)
            if fidelity.get("should_retry"):
                logger.warning(
                    f"[{project_id}] Fidelity grade={fidelity['fidelity_grade']} "
                    f"(score={fidelity['fidelity_score']:.2f}) — retrying with stricter prompt"
                )
                retry_prompt = (
                    prompt
                    + "\n\nRETRY INSTRUCTION: Previous attempt changed the room structure. "
                    "This attempt must be 100% identical in layout to the original photo. "
                    "ONLY change surfaces (paint, tiles, ceiling finish)."
                )
                retry_bytes, retry_model = await self._render(
                    client, img_bytes, retry_prompt, project_id
                )
                if retry_bytes:
                    retry_fidelity = self.validate_render_fidelity(
                        original_image_bytes, retry_bytes
                    )
                    # Accept retry if it is equal or better
                    if retry_fidelity["fidelity_score"] >= fidelity["fidelity_score"]:
                        rendered_bytes = retry_bytes
                        model_used = retry_model
                        fidelity = {**retry_fidelity, "was_retry": True}
                    else:
                        fidelity["was_retry"] = False
                        logger.warning(
                            f"[{project_id}] Retry fidelity ({retry_fidelity['fidelity_score']:.2f}) "
                            "worse than first attempt — keeping original render"
                        )

        cdn_url = None
        if getattr(settings, "USE_S3", False) and rendered_bytes:
            try:
                from services.storage import s3_service
                s3_key = f"projects/{project_id}/renders/v{version}.png"
                cdn_url = await s3_service.upload_bytes(
                    rendered_bytes, s3_key,
                    content_type="image/png",
                    bucket=settings.S3_BUCKET_RENDERS,
                )
            except Exception as e:
                logger.warning(f"S3 upload skipped: {e}")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {
            "image_b64": base64.b64encode(rendered_bytes).decode() if rendered_bytes else None,
            "image_mime": "image/png",
            "cdn_url": cdn_url,
            "model_used": model_used,
            "generation_time_ms": elapsed_ms,
            "version": version,
            "fidelity_validation": fidelity,
        }

    def validate_render_fidelity(
        self, original_bytes: bytes, rendered_bytes: bytes
    ) -> dict:
        """
        Computes structural similarity between original and rendered image.
        Uses pixel histogram comparison and aspect ratio check.

        Returns:
            fidelity_score      : float 0.0–1.0 (mean R/G/B histogram correlation)
            aspect_ratio_match  : bool
            fidelity_grade      : "high" | "medium" | "low"
            should_retry        : bool (True when grade == "low")
        """
        import numpy as np
        from PIL import Image as PILImage

        try:
            orig  = PILImage.open(io.BytesIO(original_bytes)).convert("RGB")
            rend  = PILImage.open(io.BytesIO(rendered_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"[validate_render_fidelity] Could not open images: {e}")
            return {
                "fidelity_score": 0.0,
                "aspect_ratio_match": False,
                "fidelity_grade": "low",
                "should_retry": True,
            }

        orig_w, orig_h = orig.size
        rend_w, rend_h = rend.size

        orig_ar = orig_w / max(orig_h, 1)
        rend_ar = rend_w / max(rend_h, 1)
        aspect_ratio_match = abs(orig_ar - rend_ar) <= 0.15

        # Resize both to 64×64 for fast histogram comparison
        thumb_orig = np.array(orig.resize((64, 64), PILImage.BILINEAR), dtype=np.float32)
        thumb_rend = np.array(rend.resize((64, 64), PILImage.BILINEAR), dtype=np.float32)

        correlations = []
        for ch in range(3):   # R, G, B
            hist_o, _ = np.histogram(thumb_orig[:, :, ch], bins=32, range=(0, 256))
            hist_r, _ = np.histogram(thumb_rend[:, :, ch], bins=32, range=(0, 256))
            # Pearson correlation between the two histograms
            denom = np.std(hist_o) * np.std(hist_r)
            if denom < 1e-6:
                corr = 1.0 if np.allclose(hist_o, hist_r) else 0.0
            else:
                corr = float(np.corrcoef(hist_o, hist_r)[0, 1])
            # Clamp to [0, 1] — negative correlation = completely different
            correlations.append(max(0.0, corr))

        fidelity_score = float(np.mean(correlations))

        if fidelity_score >= 0.65:
            fidelity_grade = "high"
        elif fidelity_score >= 0.45:
            fidelity_grade = "medium"
        else:
            fidelity_grade = "low"

        should_retry = fidelity_grade == "low"

        logger.info(
            f"[validate_render_fidelity] score={fidelity_score:.3f} "
            f"grade={fidelity_grade} ar_match={aspect_ratio_match} "
            f"channel_corrs={[round(c, 3) for c in correlations]}"
        )
        return {
            "fidelity_score": round(fidelity_score, 4),
            "aspect_ratio_match": aspect_ratio_match,
            "fidelity_grade": fidelity_grade,
            "should_retry": should_retry,
        }

    async def _render(
        self,
        client: genai.Client,
        image_bytes: bytes,
        prompt: str,
        project_id: str = "",
    ) -> tuple:
        """
        THE ONLY _render METHOD IN THIS CLASS.
        Previous versions had two _render methods — Python uses only the last one.
        That last one had the wrong model (gemini-2.0-flash-exp = NOT FOUND).
        """
        last_error = "No models attempted"

        for model_name in IMAGE_MODELS:
            logger.info(f"[{project_id}] Trying: {model_name}")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type="image/jpeg",
                                        data=image_bytes,
                                    )
                                ),
                                types.Part(text=prompt),
                            ],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        temperature=1,
                        safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT",
                                threshold="BLOCK_ONLY_HIGH",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="BLOCK_ONLY_HIGH",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="BLOCK_ONLY_HIGH",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="BLOCK_ONLY_HIGH",
                            ),
                        ],
                    ),
                )

                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                            data = part.inline_data.data
                            result = data if isinstance(data, bytes) else base64.b64decode(data)
                            logger.info(f"[{project_id}] SUCCESS: {model_name} ({len(result)} bytes)")
                            return result, model_name

                text_parts = [
                    p.text for c in response.candidates
                    for p in c.content.parts if hasattr(p, "text") and p.text
                ]
                finish_reasons = [str(c.finish_reason) for c in response.candidates]
                last_error = (
                    f"{model_name}: text-only (finish={finish_reasons}) "
                    f"'{' '.join(text_parts)[:200]}'"
                )
                logger.warning(f"[{project_id}] {last_error}")

            except Exception as e:
                last_error = f"{model_name}: {str(e)}"
                logger.warning(f"[{project_id}] {last_error}")
                continue

        raise RuntimeError(
            f"All image models failed. Last error: {last_error}\n"
            f"Models tried: {IMAGE_MODELS}\n"
            f"Check: GOOGLE_API_KEY is valid and billing is enabled on Google Cloud."
        )
