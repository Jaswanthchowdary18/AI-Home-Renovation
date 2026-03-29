"""
ARKEN CV Pipeline Diagnostic Tool
===================================
Loads a synthetic test image (or a real JPEG supplied via --image), runs
all four CV modules in sequence, prints their structured output, and reports
which modules are using real deep-learning inference vs graceful fallback.

Usage (from repository root)
-----------------------------
    # Smoke-test with a synthetic 400×300 room-like image
    python backend/scripts/verify_cv_pipeline.py

    # Test with a real photo
    python backend/scripts/verify_cv_pipeline.py --image /path/to/room.jpg

    # JSON output (for CI assertions)
    python backend/scripts/verify_cv_pipeline.py --json

    # Test a specific room type hint
    python backend/scripts/verify_cv_pipeline.py --room-type kitchen

Exit codes
----------
    0  All modules ran (even if using fallback mode)
    1  At least one module raised an unhandled exception
    2  Critical import failure (numpy / Pillow missing)

Output columns in the summary table
------------------------------------
    MODULE      Class name
    STATUS      ✓ real_dl | ✓ fallback | ✗ error
    MODEL       Which model/method was actually used
    KEY_FIELDS  A few representative output values for quick sanity-check
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Bootstrap sys.path so script works from any working directory ─────────────
_SCRIPT_DIR  = Path(__file__).parent.resolve()
_BACKEND_DIR = _SCRIPT_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

logging.basicConfig(
    level=logging.WARNING,   # suppress module-level debug during script
    format="%(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger("verify_cv")


# ─────────────────────────────────────────────────────────────────────────────
# Test image generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_room_image(
    width: int = 400,
    height: int = 300,
    room_type: str = "bedroom",
) -> bytes:
    """
    Generate a minimal synthetic JPEG that looks vaguely like an interior
    room (gradient sky-blue walls, warm beige floor, grey ceiling band).
    This is enough to exercise all fallback paths without requiring a real photo.
    """
    try:
        import numpy as np
        from PIL import Image

        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Ceiling band (top 15%) — grey
        ceiling_h = int(height * 0.15)
        img[:ceiling_h, :] = [180, 180, 180]

        # Wall (middle 55%) — soft warm white / beige
        wall_top = ceiling_h
        wall_bot = int(height * 0.70)
        for y in range(wall_top, wall_bot):
            # Horizontal gradient: left lighter, right slightly darker
            t = y / height
            for x in range(width):
                s = x / width
                img[y, x] = [
                    int(220 - t * 20 - s * 10),
                    int(215 - t * 15 - s * 8),
                    int(200 - t * 10),
                ]

        # Floor (bottom 30%) — warm beige / wood-like
        for y in range(wall_bot, height):
            stripe = (y - wall_bot) % 18 < 2   # horizontal plank lines
            base = [160, 130, 100] if not stripe else [140, 110, 80]
            img[y, :] = base

        # Add a minimal rectangular "window" suggestion (left wall)
        wx1, wy1, wx2, wy2 = int(width * 0.05), int(height * 0.25), \
                              int(width * 0.20), int(height * 0.55)
        img[wy1:wy2, wx1:wx2] = [200, 220, 240]   # bright sky-blue

        # Add a very rough "furniture" dark rectangle (right side of floor)
        if room_type == "bedroom":
            fx1, fy1 = int(width * 0.55), int(height * 0.55)
            img[fy1:height - 10, fx1:width - 20] = [110, 90, 70]
        elif room_type == "kitchen":
            # Counter-top band
            img[int(height * 0.65):int(height * 0.70), :] = [200, 200, 195]

        pil_img = Image.fromarray(img, "RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    except ImportError as e:
        raise RuntimeError(
            f"numpy or Pillow not available: {e}. "
            "Install them: pip install numpy Pillow"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Individual module runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_depth_estimator(
    image_bytes: bytes, room_type: str
) -> Tuple[Dict[str, Any], str, float, Optional[str]]:
    """
    Run DepthEstimator.estimate_room_area().

    Returns (result_dict, status_str, elapsed_ms, error_msg_or_None).
    """
    t0 = time.perf_counter()
    try:
        from ml.depth_estimator import DepthEstimator
        estimator = DepthEstimator()
        result    = estimator.estimate_room_area(image_bytes, room_type)
        elapsed   = (time.perf_counter() - t0) * 1000
        method    = result.get("method", "unknown")
        is_real   = result.get("depth_map_available", False)
        status    = "✓ real_dl" if is_real else "✓ fallback"
        return result, status, elapsed, None
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {}, "✗ error", elapsed, str(e)


def _run_damage_detector(
    image_bytes: bytes,
) -> Tuple[Dict[str, Any], str, float, Optional[str]]:
    """Run DamageDetector.detect()."""
    t0 = time.perf_counter()
    try:
        from ml.damage_detector import DamageDetector
        detector = DamageDetector()
        result   = detector.detect(image_bytes)
        elapsed  = (time.perf_counter() - t0) * 1000
        model    = result.get("model_used", "unknown")
        is_real  = "resnet" in model or "clip" in model
        status   = "✓ real_dl" if is_real else "✓ fallback"
        return result, status, elapsed, None
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {}, "✗ error", elapsed, str(e)


def _run_style_classifier(
    image_bytes: bytes, hint: str = ""
) -> Tuple[Dict[str, Any], str, float, Optional[str]]:
    """Run StyleClassifier.classify()."""
    t0 = time.perf_counter()
    try:
        from ml.style_classifier import StyleClassifier
        clf     = StyleClassifier()
        result  = clf.classify(image_bytes, hint)
        elapsed = (time.perf_counter() - t0) * 1000
        model   = result.get("model_used", "unknown")
        is_real = "clip" in model.lower() and "keyword" not in model.lower()
        status  = "✓ real_dl" if is_real else "✓ fallback"
        return result, status, elapsed, None
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {}, "✗ error", elapsed, str(e)


def _run_cv_extractor(
    image_bytes: bytes, room_type: str
) -> Tuple[Dict[str, Any], str, float, Optional[str]]:
    """Run CVFeatureExtractor.extract() (async, run via asyncio.run)."""
    import asyncio
    t0 = time.perf_counter()
    try:
        from ml.cv_feature_extractor import get_extractor

        async def _async_extract():
            ext = get_extractor()
            feat = await ext.extract(
                image_bytes,
                use_cache=False,
                hint_room_type=room_type,
            )
            return feat

        features = asyncio.run(_async_extract())
        elapsed  = (time.perf_counter() - t0) * 1000
        result   = {
            **features.to_vision_agent_format(),
            "cv_available":  features.cv_available,
            "inference_ms":  features.inference_ms,
            "object_details": features.object_details[:3],   # truncate for display
        }
        is_real = features.cv_available
        status  = "✓ real_dl" if is_real else "✓ fallback"
        return result, status, elapsed, None
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"cv_available": False}, "✗ error", elapsed, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _colourise(text: str, use_colour: bool) -> str:
    if not use_colour:
        return text
    if "real_dl" in text:
        return _GREEN + text + _RESET
    if "fallback" in text:
        return _YELLOW + text + _RESET
    if "error" in text:
        return _RED + text + _RESET
    return text


def _key_fields(module: str, result: Dict[str, Any]) -> str:
    """Extract a short representative summary string from result dict."""
    if module == "DepthEstimator":
        floor   = result.get("floor_area_sqft")
        wall    = result.get("wall_area_sqft")
        ceiling = result.get("ceiling_height_ft")
        conf    = result.get("confidence")
        fl_s  = f"{floor:.0f}"    if isinstance(floor,   (int, float)) else "?"
        wa_s  = f"{wall:.0f}"     if isinstance(wall,    (int, float)) else "?"
        ce_s  = f"{ceiling:.1f}"  if isinstance(ceiling, (int, float)) else "?"
        co_s  = f"{conf:.2f}"     if isinstance(conf,    (int, float)) else "?"
        return (
            f"floor={fl_s}sqft  wall={wa_s}sqft  "
            f"ceiling={ce_s}ft  conf={co_s}  "
            f"method={result.get('method', '?')}"
        )
    if module == "DamageDetector":
        issues = result.get("detected_issues", [])
        scope  = result.get("renovation_scope_recommendation", "?")
        sev    = result.get("severity", "?")
        model  = result.get("model_used", "?")
        return (
            f"issues={issues or '[]'}  "
            f"severity={sev}  "
            f"scope={scope}  "
            f"waterproof={result.get('requires_waterproofing', '?')}  "
            f"model={model}"
        )
    if module == "StyleClassifier":
        top3   = result.get("top_3_styles", [])
        top3s  = " | ".join(
            f"{t['style']}({t['confidence']:.2f})" for t in top3[:3]
            if isinstance(t.get("confidence"), (int, float))
        ) if top3 else "?"
        sc = result.get("style_confidence")
        sc_s = f"{sc:.2f}" if isinstance(sc, (int, float)) else "?"
        return (
            f"label={result.get('style_label', '?')}  "
            f"conf={sc_s}  "
            f"top3=[{top3s}]  "
            f"model={result.get('model_used', '?')}"
        )
    if module == "CVFeatureExtractor":
        objs = result.get("detected_objects", [])[:5]
        return (
            f"cv_available={result.get('cv_available', '?')}  "
            f"room={result.get('room_type', '?')}  "
            f"style={result.get('style', '?')}  "
            f"objects={objs}"
        )
    return str(result)[:120]


def _print_section(title: str, result: Dict[str, Any], colour: bool) -> None:
    """Pretty-print the full result dict for one module."""
    sep = "─" * 70
    print(f"\n{sep}")
    if colour:
        print(f"{_BOLD}{title}{_RESET}")
    else:
        print(title)
    print(sep)
    # Pretty JSON, but clip long embedding arrays
    safe_result = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 20:
            safe_result[k] = v[:5] + [f"... ({len(v)} items)"]
        elif isinstance(v, dict) and len(v) > 15:
            safe_result[k] = dict(list(v.items())[:8])
            safe_result[k]["..."] = f"({len(v)} total keys)"
        else:
            safe_result[k] = v
    try:
        print(json.dumps(safe_result, indent=2, default=str))
    except Exception:
        print(safe_result)


def _print_summary_table(
    rows: List[Tuple[str, str, str, float, Optional[str]]],
    colour: bool,
) -> None:
    """
    Print the summary table.
    rows = [(module, status, key_fields_str, elapsed_ms, error_or_None)]
    """
    hdr = f"{'MODULE':<22}  {'STATUS':<14}  {'TIME':>8}  KEY FIELDS"
    sep = "=" * 100
    print(f"\n{sep}")
    if colour:
        print(f"{_BOLD}{hdr}{_RESET}")
    else:
        print(hdr)
    print(sep)
    for module, status, fields, elapsed, err in rows:
        col_status = _colourise(status, colour)
        line = f"{module:<22}  {col_status:<14}  {elapsed:>7.0f}ms  {fields}"
        print(line)
        if err:
            indent = " " * 4
            for chunk in textwrap.wrap(f"ERROR: {err}", width=90):
                print(indent + (_RED if colour else "") + chunk + (_RESET if colour else ""))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARKEN CV Pipeline Diagnostic — verify all 4 CV modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a JPEG/PNG image (default: synthetic room image)"
    )
    parser.add_argument(
        "--room-type", type=str, default="bedroom",
        choices=["bedroom", "kitchen", "bathroom", "living_room", "dining_room", "study"],
        help="Room type hint for DepthEstimator and CVFeatureExtractor (default: bedroom)"
    )
    parser.add_argument(
        "--style-hint", type=str, default="",
        help="Gemini-style hint for StyleClassifier blend (e.g. 'Modern Minimalist')"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit all results as a single JSON object (no colour / tables)"
    )
    parser.add_argument(
        "--no-colour", action="store_true",
        help="Disable ANSI colour output"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full result dict for each module"
    )
    args = parser.parse_args()

    colour = not args.no_colour and not args.json and sys.stdout.isatty()

    # ── Load or generate test image ───────────────────────────────────────────
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"ERROR: image file not found: {img_path}", file=sys.stderr)
            return 1
        image_bytes = img_path.read_bytes()
        img_source  = str(img_path)
        img_size    = len(image_bytes)
    else:
        try:
            image_bytes = _make_synthetic_room_image(room_type=args.room_type)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        img_source = "synthetic (400×300 px)"
        img_size   = len(image_bytes)

    if not args.json:
        print(f"\n{'=' * 70}")
        print("  ARKEN CV Pipeline — Diagnostic Verification")
        print(f"{'=' * 70}")
        print(f"  Image source : {img_source}")
        print(f"  Image size   : {img_size:,} bytes")
        print(f"  Room type    : {args.room_type}")
        print(f"  Style hint   : {args.style_hint or '(none)'}")
        print(f"  Backend dir  : {_BACKEND_DIR}")
        print()

    # ── Run modules ───────────────────────────────────────────────────────────
    all_results: Dict[str, Any] = {}
    table_rows:  List[Tuple]    = []
    has_error = False

    # 1. DepthEstimator
    if not args.json:
        print("  [1/4] Running DepthEstimator …", end=" ", flush=True)
    r_depth, s_depth, t_depth, e_depth = _run_depth_estimator(image_bytes, args.room_type)
    if not args.json:
        print(f"done ({t_depth:.0f}ms)")
    all_results["depth_estimator"] = r_depth
    table_rows.append(("DepthEstimator", s_depth, _key_fields("DepthEstimator", r_depth), t_depth, e_depth))
    if e_depth:
        has_error = True

    # 2. DamageDetector
    if not args.json:
        print("  [2/4] Running DamageDetector …", end=" ", flush=True)
    r_dmg, s_dmg, t_dmg, e_dmg = _run_damage_detector(image_bytes)
    if not args.json:
        print(f"done ({t_dmg:.0f}ms)")
    all_results["damage_detector"] = r_dmg
    table_rows.append(("DamageDetector", s_dmg, _key_fields("DamageDetector", r_dmg), t_dmg, e_dmg))
    if e_dmg:
        has_error = True

    # 3. StyleClassifier
    if not args.json:
        print("  [3/4] Running StyleClassifier …", end=" ", flush=True)
    r_style, s_style, t_style, e_style = _run_style_classifier(image_bytes, args.style_hint)
    if not args.json:
        print(f"done ({t_style:.0f}ms)")
    all_results["style_classifier"] = r_style
    table_rows.append(("StyleClassifier", s_style, _key_fields("StyleClassifier", r_style), t_style, e_style))
    if e_style:
        has_error = True

    # 4. CVFeatureExtractor
    if not args.json:
        print("  [4/4] Running CVFeatureExtractor …", end=" ", flush=True)
    r_cv, s_cv, t_cv, e_cv = _run_cv_extractor(image_bytes, args.room_type)
    if not args.json:
        print(f"done ({t_cv:.0f}ms)")
    all_results["cv_feature_extractor"] = r_cv
    table_rows.append(("CVFeatureExtractor", s_cv, _key_fields("CVFeatureExtractor", r_cv), t_cv, e_cv))
    if e_cv:
        has_error = True

    # ── Render output ─────────────────────────────────────────────────────────
    if args.json:
        # Machine-readable JSON (exit code only)
        output = {
            "image_source":   img_source,
            "image_bytes":    img_size,
            "room_type_hint": args.room_type,
            "modules": {
                "depth_estimator":    {"status": s_depth, "elapsed_ms": round(t_depth, 1), "result": r_depth},
                "damage_detector":    {"status": s_dmg,   "elapsed_ms": round(t_dmg,   1), "result": r_dmg},
                "style_classifier":   {"status": s_style, "elapsed_ms": round(t_style, 1), "result": r_style},
                "cv_feature_extractor": {"status": s_cv,  "elapsed_ms": round(t_cv,    1), "result": r_cv},
            },
            "has_error": has_error,
        }
        print(json.dumps(output, indent=2, default=str))
        return 1 if has_error else 0

    # Human-readable output
    _print_summary_table(table_rows, colour)

    # Module capability legend
    try:
        from ml import capabilities
        caps = capabilities()
        print()
        print("  Dependency status:")
        for dep, avail in caps.items():
            sym  = ("✓" if avail else "✗")
            col  = (_GREEN if avail else _YELLOW) if colour else ""
            rst  = _RESET if colour else ""
            print(f"    {col}{sym}{rst}  {dep}")
    except Exception:
        pass

    # Optional verbose dump
    if args.verbose:
        _print_section("DepthEstimator — full result",   r_depth, colour)
        _print_section("DamageDetector — full result",   r_dmg,   colour)
        _print_section("StyleClassifier — full result",  r_style, colour)
        _print_section("CVFeatureExtractor — full result", r_cv,  colour)

    # Final verdict
    total_ms = t_depth + t_dmg + t_style + t_cv
    print()
    if has_error:
        print(_RED + "  RESULT: One or more modules raised an exception. See errors above." + _RESET if colour
              else "  RESULT: One or more modules raised an exception. See errors above.")
        print(f"  Total time: {total_ms:.0f}ms")
        return 1
    else:
        real_count = sum(1 for _, s, _, _, _ in table_rows if "real_dl" in s)
        fb_count   = sum(1 for _, s, _, _, _ in table_rows if "fallback" in s)
        msg = (
            f"  RESULT: All 4 modules operational.  "
            f"real_dl={real_count}/4  fallback={fb_count}/4  "
            f"total={total_ms:.0f}ms"
        )
        print((_GREEN if colour else "") + msg + (_RESET if colour else ""))

        if real_count < 4:
            tip = (
                "\n  TIP: Install optional DL dependencies to unlock real inference:\n"
                "    pip install torch torchvision transformers ultralytics Pillow\n"
                "  Model weights will be downloaded automatically to /app/ml/weights/ \n"
                "  on first use."
            )
            print((_YELLOW if colour else "") + tip + (_RESET if colour else ""))

        return 0


if __name__ == "__main__":
    sys.exit(main())
