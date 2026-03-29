"""
ARKEN — Design Planner Agent v3.0
SAVE AS: backend/agents/design_planner.py

v3.0 FIX: hidden labour lump sum deleted. Labour is now visible line items.
See full docstring inside.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CITY_LABOUR_MULTIPLIER: Dict[str, float] = {
    "Mumbai": 1.35, "Delhi NCR": 1.28, "Bangalore": 1.18,
    "Chennai": 1.08, "Pune": 1.05, "Hyderabad": 1.00,
    "Kolkata": 0.95, "Ahmedabad": 0.90, "Jaipur": 0.88,
    "Chandigarh": 0.95, "Lucknow": 0.85, "Nagpur": 0.87,
    "Surat": 0.88, "Indore": 0.86, "Bhopal": 0.84,
}

TIER_COST_MULTIPLIER: Dict[str, float] = {"basic": 1.0, "mid": 1.0, "premium": 1.0}
_TIER_QTY_MULTIPLIER: Dict[str, float] = {"basic": 1.0, "mid": 1.0, "premium": 1.0}

BOQ_SCOPE_BY_TIER: Dict[str, List[str]] = {
    "basic": [
        "Civil Materials", "Wall Preparation", "Crack Repair", "Primer", "Paint",
        "Labour - Painting", "Flooring Tiles", "Labour - Tiling", "Skirting",
        "Fan", "Switches", "Lighting", "Labour - Electrical",
    ],
    "mid": [
        "Civil Materials", "Wall Preparation", "Crack Repair", "Primer", "Paint",
        "Wall Texture", "Labour - Painting",
        "Flooring Tiles", "Wall Tiles", "Labour - Tiling", "Skirting",
        "Plywood", "Laminates", "Hardware - Carpentry", "Labour - Carpentry",
        "Fan", "Switches", "Lighting", "Cove Lighting", "Labour - Electrical",
        "False Ceiling - GI Frame", "False Ceiling - Gypsum Board",
        "False Ceiling - LED Strip", "Labour - False Ceiling",
        "WC", "Basin", "Faucet",
        "Waterproofing Material", "Labour - Waterproofing",
        "Chimney", "Sink", "Hardware",
    ],
    "premium": [
        "Civil Materials", "Wall Preparation", "Crack Repair", "Primer", "Paint",
        "Wall Texture", "Labour - Painting",
        "Flooring Tiles", "Wall Tiles", "Labour - Tiling", "Skirting",
        "Plywood", "Laminates", "Hardware - Carpentry", "Labour - Carpentry",
        "Wardrobe Carcass", "Wardrobe Shutters", "Wardrobe Hardware",
        "Fan", "Switches", "Smart Lighting", "Lighting", "Cove Lighting",
        "Labour - Electrical",
        "False Ceiling - GI Frame", "False Ceiling - Gypsum Board",
        "False Ceiling - LED Strip", "Labour - False Ceiling",
        "WC", "Basin", "Shower", "Faucet",
        "Waterproofing Material", "Labour - Waterproofing",
        "Chimney", "Sink", "Hardware",
    ],
}

PAINT_CATALOG = {
    "basic": [
        {"brand": "Asian Paints", "product": "Apcolite Premium Emulsion", "sku": "AP-APG-20L",
         "finish": "Emulsion", "price_per_liter": 195, "coverage_sqft_per_liter": 14},
        {"brand": "Berger", "product": "Bison Emulsion", "sku": "BG-BSN-20L",
         "finish": "Emulsion", "price_per_liter": 178, "coverage_sqft_per_liter": 13},
    ],
    "mid": [
        {"brand": "Asian Paints", "product": "Royale Sheen", "sku": "AP-RS-4L",
         "finish": "Sheen", "price_per_liter": 435, "coverage_sqft_per_liter": 16},
        {"brand": "Asian Paints", "product": "Royale Play Texture", "sku": "AP-RP-4L",
         "finish": "Texture", "price_per_liter": 395, "coverage_sqft_per_liter": 12},
        {"brand": "Nerolac", "product": "Excel Mica Marble", "sku": "NL-EMM-4L",
         "finish": "Sheen", "price_per_liter": 410, "coverage_sqft_per_liter": 15},
        {"brand": "Berger", "product": "Silk Breatheasy", "sku": "BG-SBE-4L",
         "finish": "Sheen", "price_per_liter": 420, "coverage_sqft_per_liter": 15},
    ],
    "premium": [
        {"brand": "Asian Paints", "product": "Royale Luxury Emulsion", "sku": "AP-RLE-4L",
         "finish": "Matt", "price_per_liter": 695, "coverage_sqft_per_liter": 18},
        {"brand": "Dulux", "product": "Velvet Touch Pearl Glo", "sku": "DX-VTP-4L",
         "finish": "Pearl", "price_per_liter": 740, "coverage_sqft_per_liter": 17},
    ],
}
PAINT_BY_ROOM = {
    "kitchen": {"basic": 1, "mid": 3, "premium": 0},
    "bathroom": {"basic": 0, "mid": 3, "premium": 1},
    "bedroom": {"basic": 0, "mid": 0, "premium": 0},
    "living_room": {"basic": 0, "mid": 2, "premium": 1},
    "default": {"basic": 0, "mid": 0, "premium": 0},
}

TILE_CATALOG = {
    "basic": [
        {"brand": "Kajaria", "product": "Glazed Vitrified 600x600", "sku": "KJR-GVT-6060-BG",
         "price_per_sqft": 62, "finish": "Glossy", "size": "600x600"},
        {"brand": "Somany", "product": "Floor Tile 400x400", "sku": "SM-FT-4040",
         "price_per_sqft": 50, "finish": "Matt", "size": "400x400"},
    ],
    "mid": [
        {"brand": "Kajaria", "product": "Endura Anti-Skid 600x600", "sku": "KJR-EAS-6060",
         "price_per_sqft": 90, "finish": "Anti-Skid Matt", "size": "600x600"},
        {"brand": "Kajaria", "product": "Designer Wall Tile 300x600", "sku": "KJR-DWT-3060",
         "price_per_sqft": 98, "finish": "Glossy", "size": "300x600"},
        {"brand": "Nitco", "product": "Vitrified 800x800", "sku": "NT-VT-8080",
         "price_per_sqft": 115, "finish": "Polished", "size": "800x800"},
        {"brand": "Johnson", "product": "Endura Glossy 600x1200", "sku": "JH-EG-6120",
         "price_per_sqft": 105, "finish": "Polished", "size": "600x1200"},
    ],
    "premium": [
        {"brand": "Simpolo", "product": "GVT Slab 800x1600", "sku": "SP-GVT-8160",
         "price_per_sqft": 205, "finish": "Polished", "size": "800x1600"},
        {"brand": "RAK Ceramics", "product": "Marble Effect 600x1200", "sku": "RAK-ME-6120",
         "price_per_sqft": 295, "finish": "Polished Matt", "size": "600x1200"},
        {"brand": "Porcelain House", "product": "Italian Bianco 600x600", "sku": "PH-IBM-6060",
         "price_per_sqft": 360, "finish": "High Gloss", "size": "600x600"},
    ],
}
THEME_TILE_PREFERENCE: Dict[str, Dict] = {
    "Modern Minimalist": {"mid": 2, "premium": 0},
    "Scandinavian": {"mid": 2, "premium": 1},
    "Japandi": {"mid": 3, "premium": 1},
    "Industrial": {"mid": 0, "premium": 1},
    "Contemporary Indian": {"mid": 1, "premium": 2},
    "Traditional Indian": {"mid": 0, "premium": 2},
    "Art Deco": {"mid": 2, "premium": 0},
    "Bohemian": {"mid": 1, "premium": 2},
}

PLYWOOD_CATALOG = {
    "basic": [{"brand": "Greenply", "product": "Club MR Grade 19mm", "sku": "GLP-CMR-19",
               "price_per_sqft": 75, "type": "MR Grade"}],
    "mid": [{"brand": "Greenply", "product": "Club Prime BWR 18mm", "sku": "GLP-CPB-18",
             "price_per_sqft": 98, "type": "BWR Grade"},
            {"brand": "Kitply", "product": "Gold BWP 19mm", "sku": "KTP-GBP-19",
             "price_per_sqft": 110, "type": "BWP Grade"}],
    "premium": [{"brand": "Greenply", "product": "Marine Grade 19mm", "sku": "GLP-MRN-19",
                 "price_per_sqft": 150, "type": "Marine"},
                {"brand": "Century", "product": "Sainik 710 19mm", "sku": "CTY-S710-19",
                 "price_per_sqft": 138, "type": "BWP Grade"}],
}

HARDWARE_CATALOG = {
    "basic": [
        {"brand": "Havells", "product": "Efficiencia Neo Fan 48\"", "sku": "HVL-EN-48",
         "price_inr": 5200, "category": "Fan"},
        {"brand": "Anchor", "product": "Roma Modular Switches", "sku": "ANC-RM-SET",
         "price_inr": 2400, "category": "Switches"},
        {"brand": "Philips", "product": "LED Batten 20W", "sku": "PHL-B20W",
         "price_inr": 680, "category": "Lighting"},
    ],
    "mid": [
        {"brand": "Havells", "product": "Pacer Neo Ceiling Fan 56\"", "sku": "HVL-PN-56",
         "price_inr": 7200, "category": "Fan"},
        {"brand": "Legrand", "product": "Myrius Modular Switches", "sku": "LGD-MY-SET",
         "price_inr": 4800, "category": "Switches"},
    ],
    "premium": [
        {"brand": "Orient", "product": "Aeroslim Smart Fan 48\"", "sku": "ORT-AS-48",
         "price_inr": 15500, "category": "Fan"},
        {"brand": "Schneider", "product": "AvatarOn Modular Switch", "sku": "SCH-AV-SET",
         "price_inr": 9200, "category": "Switches"},
    ],
}

SANITARY_CATALOG = {
    "basic": [
        {"brand": "Hindware", "product": "Rimless EWC Floor-Mount", "sku": "HW-REW-01",
         "price_inr": 9500, "category": "WC"},
        {"brand": "Cera", "product": "Table Top Basin 45cm", "sku": "CR-TTB-45",
         "price_inr": 4200, "category": "Basin"},
        {"brand": "Jaquar", "product": "Single Lever Mixer", "sku": "JQ-SLM-01",
         "price_inr": 3800, "category": "Faucet"},
    ],
    "mid": [
        {"brand": "Hindware", "product": "Wall-Hung WC + Carrier Frame", "sku": "HW-WHW-01",
         "price_inr": 22000, "category": "WC"},
        {"brand": "Roca", "product": "Counter-top Basin 56cm", "sku": "RC-CTB-56",
         "price_inr": 8500, "category": "Basin"},
        {"brand": "Jaquar", "product": "Overhead Rainfall Shower System", "sku": "JQ-ORS-01",
         "price_inr": 12500, "category": "Shower"},
        {"brand": "Jaquar", "product": "Thermostatic Mixer", "sku": "JQ-TM-01",
         "price_inr": 8200, "category": "Faucet"},
    ],
    "premium": [
        {"brand": "Kohler", "product": "Veil Wall-Hung WC", "sku": "KH-VWH-01",
         "price_inr": 55000, "category": "WC"},
        {"brand": "Roca", "product": "Inspira Countertop Basin 60cm", "sku": "RC-INS-60",
         "price_inr": 18000, "category": "Basin"},
        {"brand": "Grohe", "product": "Rainshower System 310mm", "sku": "GR-RS-310",
         "price_inr": 42000, "category": "Shower"},
        {"brand": "Hansgrohe", "product": "Thermostatic Mixer 3 outlets", "sku": "HG-TM-3",
         "price_inr": 35000, "category": "Faucet"},
    ],
}

KITCHEN_CATALOG = {
    "basic": [
        {"brand": "Kutchina", "product": "Chimney 60cm Auto Clean", "sku": "KC-AC-60",
         "price_inr": 12500, "category": "Chimney"},
        {"brand": "Futura", "product": "Stainless Steel Single Bowl Sink", "sku": "FT-SS-SB",
         "price_inr": 4800, "category": "Sink"},
    ],
    "mid": [
        {"brand": "Faber", "product": "Chimney 60cm Filterless", "sku": "FB-FL-60",
         "price_inr": 18500, "category": "Chimney"},
        {"brand": "Franke", "product": "Stainless Steel Double Bowl", "sku": "FR-DB-01",
         "price_inr": 9800, "category": "Sink"},
        {"brand": "Hettich", "product": "Soft-Close Drawer Runners (set 5)", "sku": "HT-SCR-5",
         "price_inr": 6500, "category": "Hardware"},
    ],
    "premium": [
        {"brand": "Elica", "product": "Island Chimney 90cm", "sku": "EL-IC-90",
         "price_inr": 38000, "category": "Chimney"},
        {"brand": "Blanco", "product": "Granite Sink SILGRANIT", "sku": "BL-SG-01",
         "price_inr": 22000, "category": "Sink"},
        {"brand": "Blum", "product": "Tandem Drawer Set Complete", "sku": "BL-TD-SET",
         "price_inr": 15000, "category": "Hardware"},
    ],
}

WATERPROOFING_CATALOG = {
    "basic": [{"brand": "Dr. Fixit", "product": "Pidiproof LW+ 1L",
               "sku": "DF-PP-1L", "price_inr": 680, "coverage_sqft": 12}],
    "mid": [{"brand": "Dr. Fixit", "product": "Bathseal 2-comp 4kg",
             "sku": "DF-BS-4KG", "price_inr": 2200, "coverage_sqft": 20}],
    "premium": [{"brand": "Dr. Fixit", "product": "Bathseal Pro 2-comp 8kg",
                 "sku": "DF-BSP-8KG", "price_inr": 4200, "coverage_sqft": 40}],
}

REPAIR_CATALOG = {
    "crack_filler": [
        {"brand": "Asian Paints", "product": "Wall Fill Crack Filler 1L",
         "sku": "AP-WF-1L", "price_inr": 420},
    ],
    "primer": [
        {"brand": "Asian Paints", "product": "Wall Primer Ultra 20L",
         "sku": "AP-WP-20L", "price_inr": 3200},
        {"brand": "Berger", "product": "Alkali Resist Primer 4L",
         "sku": "BG-ARP-4L", "price_inr": 1100},
    ],
    "putty": [
        {"brand": "Asian Paints", "product": "Trucare Wall Putty 20kg",
         "sku": "AP-WPT-20KG", "price_inr": 980},
    ],
}

LABOUR_RATES = {
    "painting_per_sqft":      {"basic": 30, "mid": 45,  "premium": 60},
    "tiling_per_sqft":        {"basic": 32, "mid": 50,  "premium": 70},
    "carpentry_per_sqft":     {"basic": 650,"mid": 950, "premium": 1500},
    "electrical_per_point":   {"basic": 300,"mid": 450, "premium": 650},
    "plumbing_per_point":     {"basic": 2200,"mid": 3500,"premium": 5500},
    "waterproofing_per_sqft": {"basic": 40, "mid": 60,  "premium": 85},
    "false_ceiling_per_sqft": {"basic": 75, "mid": 110, "premium": 160},
    "wall_texture_per_sqft":  {"basic": 25, "mid": 55,  "premium": 90},
    "wardrobe_per_sqft":      {"basic": 650,"mid": 950, "premium": 1500},
    "repair_per_sqft":        {"basic": 30, "mid": 50,  "premium": 75},
}


def _item(category, brand, product, unit, qty, rate, priority="must_have", note="", sku=""):
    qty   = round(float(qty), 1)
    rate  = int(round(float(rate)))
    total = int(round(qty * rate))
    d = {"category": category, "brand": brand, "product": product,
         "unit": unit, "qty": qty, "rate_inr": rate, "total_inr": total,
         "priority": priority, "note": note}
    if sku:
        d["sku"] = sku
    return d


class DesignPlannerAgent:

    def plan(self, *, theme, budget_inr, budget_tier, area_sqft, room_type, city, quantities,
             wall_condition="fair", floor_condition="fair", issues_detected=None,
             renovation_scope="partial", high_value_upgrades=None, condition_score=None):
        if condition_score is None:
            logger.warning("[DesignPlanner v3] condition_score not supplied, defaulting to 60")
            condition_score = 60

        tier      = budget_tier.lower()
        city_mult = CITY_LABOUR_MULTIPLIER.get(city, 1.0)
        issues    = issues_detected or []
        area      = round(float(area_sqft), 1)

        q: Dict = {}
        for k, v in quantities.items():
            try:
                q[k] = round(float(v), 1)
            except (TypeError, ValueError):
                q[k] = v

        wall_area  = round(q.get("wall_area_sqft",  area * 3.5), 1)
        floor_area = round(q.get("floor_tiles_sqft", area * 1.1), 1)

        line_items: List[Dict] = []

        # 1. Civil materials
        line_items.extend(self._build_civil_materials(tier, room_type, q, floor_area))
        # 2. Wall prep
        line_items.extend(self._build_wall_prep(tier, wall_area, wall_condition, issues))
        # 3. Paint material
        line_items.extend(self._select_paint(tier, theme, room_type, q, wall_area))
        # 4. Wall texture material (mid/premium)
        if tier in ("mid", "premium"):
            line_items.extend(self._build_wall_texture(tier, wall_area))
        # 5. Flooring + wall tiles
        line_items.extend(self._select_tiles(tier, theme, room_type, q, floor_area))
        # 6. Skirting
        line_items.extend(self._build_skirting(tier, area))
        # 7. False ceiling components (mid/premium bedroom/living_room)
        if tier in ("mid", "premium") and room_type in ("living_room", "bedroom", "full_home"):
            line_items.extend(self._build_false_ceiling_items(tier, area, city_mult))
        # 8. Wardrobe (premium) or basic plywood (mid bedroom)
        if room_type in ("bedroom", "full_home"):
            if tier == "premium":
                line_items.extend(self._build_wardrobe_items(tier, area, city_mult))
            elif tier == "mid":
                line_items.extend(self._select_plywood(tier, q))
        # 9. Electrical hardware
        line_items.extend(self._select_hardware(tier, room_type))
        # 10. Room-specific
        if room_type in ("bathroom", "full_home"):
            line_items.extend(self._select_sanitary(tier))
            line_items.extend(self._build_waterproofing(tier, floor_area))
        if room_type in ("kitchen", "full_home"):
            line_items.extend(self._select_kitchen_items(tier))
        # 11. LABOUR — every trade as separate visible line items
        line_items.extend(self._build_labour_line_items(
            tier, area, wall_area, room_type, city_mult, renovation_scope, line_items
        ))

        # Filter to tier scope and tag
        allowed = set(BOQ_SCOPE_BY_TIER.get(tier, BOQ_SCOPE_BY_TIER["mid"]))
        always_include = {"Wall Preparation", "Crack Repair", "Primer", "Civil Materials"}
        final_items: List[Dict] = []
        for it in line_items:
            it = dict(it)
            it["tier_applied"] = tier
            if "qty" in it:
                try:
                    it["qty"] = round(float(it["qty"]), 1)
                except (TypeError, ValueError):
                    pass
            cat = it.get("category", "")
            if cat in allowed or cat in always_include:
                final_items.append(it)

        total_materials = sum(
            i["total_inr"] for i in final_items
            if not i.get("category", "").startswith("Labour")
        )
        total_labour = sum(
            i["total_inr"] for i in final_items
            if i.get("category", "").startswith("Labour")
        )
        subtotal = total_materials + total_labour
        gst = int(subtotal * 0.18)
        contingency_pct = {"basic": 0.15, "mid": 0.12, "premium": 0.10}.get(tier, 0.12)
        contingency = int(subtotal * contingency_pct)
        grand_total = subtotal + gst + contingency

        logger.info(
            f"[DesignPlanner v3] tier={tier} area={area}sqft items={len(final_items)} "
            f"materials=\u20b9{total_materials:,} labour=\u20b9{total_labour:,} total=\u20b9{grand_total:,}"
        )

        return {
            "total_inr": grand_total, "material_inr": total_materials,
            "labour_inr": total_labour, "gst_inr": gst, "contingency_inr": contingency,
            "line_items": final_items, "boq_scope_tier": tier,
            "tier_cost_multiplier": 1.0,
            "supplier_recommendations": self._suggest_suppliers(city),
            "recommendations": {
                "paint":    next((i for i in final_items if i["category"] == "Paint"), {}),
                "tiles":    next((i for i in final_items if i["category"] == "Flooring Tiles"), {}),
                "hardware": [i for i in final_items if i["category"] in
                             ("Fan", "Switches", "Lighting", "Smart Lighting")],
            },
            "condition_repairs_included": any(i["category"] == "Crack Repair" for i in final_items),
            "condition_score_used": condition_score,
            "high_value_upgrades": high_value_upgrades or [],
        }

    def _build_civil_materials(self, tier, room_type, quantities, floor_area):
        fa = floor_area
        items = []
        adhesive_bags = max(1, math.ceil(fa / 50))
        items.append(_item("Civil Materials","MYK Laticrete","Tile Adhesive S1 Non-Slip 20kg",
            "bag (20kg)", adhesive_bags, 480, sku="MYK-TAS1-20KG",
            note=f"{fa} sqft / 50 sqft per bag @ 3mm bed"))
        grout_kg = round(fa * 0.3, 1)
        items.append(_item("Civil Materials","Pidilite Roff","Wallgrout Premium 1kg (colour-matched)",
            "kg", grout_kg, 85, sku="ROFF-WG-1KG",
            note="3mm joint, 600x600 / 800x800 tiles"))
        if room_type not in ("study",):
            cement_bags = max(1, math.ceil(fa / 40))
            items.append(_item("Civil Materials","UltraTech","OPC 53 Grade Cement 50kg",
                "bag (50kg)", cement_bags, 380, sku="UTC-OPC53-50",
                note=f"{fa} sqft / 40 sqft per bag, bedding mortar"))
            sand_cft = round(fa / 15, 1)
            items.append(_item("Civil Materials","Local Supplier","River Sand / M-Sand (sieved)",
                "cft", sand_cft, 50, sku="SAND-SIEVED",
                note=f"{fa} sqft / 15 sqft per cft, bedding mortar"))
        return items

    def _build_wall_prep(self, tier, wall_area, wall_condition, issues):
        items = []
        issues_text = " ".join(issues).lower()
        putty_bags = max(1, math.ceil(wall_area / 80))
        items.append(_item("Wall Preparation","Asian Paints","Trucare Wall Putty 20kg",
            "bag (20kg)", putty_bags, 980, sku="AP-WPT-20KG",
            note=f"{wall_area} sqft / 80 sqft per bag"))
        primer_cans = max(1, math.ceil(wall_area / 200))
        primer = REPAIR_CATALOG["primer"][1 if wall_condition in ("poor","very poor") else 0]
        items.append(_item("Primer", primer["brand"], primer["product"],
            "can", primer_cans, primer["price_inr"], sku=primer["sku"],
            note=f"{wall_area} sqft / 200 sqft per can"))
        if wall_condition in ("poor","very poor","fair") or any(
            kw in issues_text for kw in ("crack","peel","seepage","damp")):
            cf = REPAIR_CATALOG["crack_filler"][0]
            items.append(_item("Crack Repair", cf["brand"], cf["product"],
                "litre", 2, cf["price_inr"], sku="AP-WF-1L",
                note=f"Wall condition: {wall_condition}"))
        return items

    def _select_paint(self, tier, theme, room_type, quantities, wall_area):
        catalog = PAINT_CATALOG.get(tier, PAINT_CATALOG["mid"])
        room_map = PAINT_BY_ROOM.get(room_type, PAINT_BY_ROOM["default"])
        idx = min(room_map.get(tier, 0), len(catalog) - 1)
        p = catalog[idx]
        litres_needed = round((wall_area * 2) / p["coverage_sqft_per_liter"], 1)
        cans = max(1, math.ceil(litres_needed / 4))
        rate = int(p["price_per_liter"] * 4)
        return [_item("Paint", p["brand"], p["product"], "can (4L)", cans, rate, sku=p["sku"],
            note=f"{wall_area} sqft, 2 coats")]

    def _build_wall_texture(self, tier, wall_area):
        if tier == "mid":
            bags = max(1, math.ceil(wall_area / 80))
            return [_item("Wall Texture","Asian Paints","Royale Play Tex Finish 4kg",
                "bag (4kg)", bags, 1850, sku="AP-RPT-4KG",
                note=f"{wall_area} sqft texture finish")]
        else:
            kits = max(2, math.ceil(wall_area / 40))
            return [_item("Wall Texture","Mapei","Mapecoat Micro-Cement 5kg Kit",
                "kit (5kg)", kits, 4800, sku="MAPEI-MC-5KG",
                note=f"{wall_area} sqft, 2-layer micro-cement system")]

    def _select_tiles(self, tier, theme, room_type, quantities, floor_area):
        catalog = TILE_CATALOG.get(tier, TILE_CATALOG["mid"])
        pref_idx = min(THEME_TILE_PREFERENCE.get(theme, {}).get(tier, 0), len(catalog) - 1)
        if room_type == "bathroom" and tier == "mid":
            pref_idx = 0
        t = catalog[pref_idx]
        sqft_ww = round(floor_area, 1)
        items = [_item("Flooring Tiles", t["brand"], f"{t['product']} ({t['size']})",
            "sqft (incl. 10% wastage)", sqft_ww, t["price_per_sqft"], sku=t["sku"],
            note=f"Floor area: {round(floor_area/1.1, 1)} sqft + 10% wastage")]
        if room_type in ("bathroom","kitchen") and tier in ("mid","premium"):
            wt_sqft = round(quantities.get("wall_tiles_sqft", floor_area * 0.6) * 1.1, 1)
            wt = catalog[1] if len(catalog) > 1 else catalog[0]
            items.append(_item("Wall Tiles", wt["brand"], f"{wt['product']} ({wt['size']})",
                "sqft (incl. 10% wastage)", wt_sqft, wt["price_per_sqft"], sku=wt["sku"]))
        return items

    def _build_skirting(self, tier, area_sqft):
        perimeter = round(4 * math.sqrt(area_sqft), 1)
        if tier == "basic":
            return [_item("Skirting","Asian Granito","Ceramic Skirting 4\"x24\"",
                "rft", perimeter, 55, sku="AG-SK-4X24",
                note=f"Room perimeter ~{perimeter} rft")]
        elif tier == "mid":
            return [_item("Skirting","Kajaria","Vitrified Skirting 4\"x24\"",
                "rft", perimeter, 85, sku="KJR-SK-4X24",
                note=f"Room perimeter ~{perimeter} rft")]
        else:
            return [_item("Skirting","Simpolo","Full-Body Porcelain Skirting 4\"x32\"",
                "rft", perimeter, 140, sku="SP-SK-4X32",
                note=f"Room perimeter ~{perimeter} rft, matching floor tile")]

    def _build_false_ceiling_items(self, tier, area_sqft, city_mult):
        items = []
        perimeter = round(4 * math.sqrt(area_sqft), 1)
        if tier == "mid":
            gi_rft = round(area_sqft * 0.8, 1)
            items.append(_item("False Ceiling - GI Frame","Steel Junction",
                "GI Channel 0.55mm (Main + Cross, full set)",
                "rft", gi_rft, 45, sku="GI-FC-SET",
                note="Main runners + cross tees for suspended grid"))
            gyp_sqft = round(area_sqft * 1.05, 1)
            items.append(_item("False Ceiling - Gypsum Board","Saint-Gobain",
                "Gyproc Plasterboard 12.5mm",
                "sqft", gyp_sqft, 55, sku="SG-GPB-12",
                note=f"{area_sqft} sqft + 5% wastage"))
            items.append(_item("False Ceiling - GI Frame","Saint-Gobain",
                "POP / Joint Compound Finishing Coat",
                "sqft", area_sqft, 25, sku="SG-POP-FC",
                note="Skim + tape finish over gypsum board"))
        else:
            gi_rft = round(area_sqft * 1.1, 1)
            items.append(_item("False Ceiling - GI Frame","Steel Junction",
                "GI Channel Heavy 0.8mm (Coffered grid system)",
                "rft", gi_rft, 65, sku="GI-FC-COFF",
                note="Heavy gauge for coffered multi-level design"))
            gyp_sqft = round(area_sqft * 1.15, 1)
            items.append(_item("False Ceiling - Gypsum Board","Saint-Gobain",
                "Gyproc Moisture Resistant Board 12.5mm",
                "sqft", gyp_sqft, 75, sku="SG-GPB-MR",
                note=f"{area_sqft} sqft + 15% wastage (coffered shape)"))
            items.append(_item("False Ceiling - GI Frame","Saint-Gobain",
                "Cornice / Cove Moulding + POP Finish",
                "rft", perimeter, 180, sku="SG-COVE-POP",
                note=f"Coffered reveal edges, {perimeter} rft"))
        led_reels = max(1, math.ceil(perimeter / 5))
        led_brand = "Legrand" if tier == "premium" else "Anchor"
        led_prod  = "LED Strip 5m 24V Warm White" if tier == "premium" else "LED Strip 5m Warm White"
        led_rate  = 3800 if tier == "premium" else 1800
        led_sku   = "LGD-CS-5M" if tier == "premium" else "ANC-LS-5M"
        items.append(_item("False Ceiling - LED Strip", led_brand, led_prod,
            "reel (5m)", led_reels, led_rate, sku=led_sku,
            note=f"Cove, {perimeter} rft perimeter / 5m per reel"))
        dl_count = max(4, int(area_sqft / 25))
        dl_brand = "Philips Hue" if tier == "premium" else "Philips"
        dl_prod  = "Smart Downlight 10W" if tier == "premium" else "LED Downlight 12W 4-pack"
        dl_rate  = 2625 if tier == "premium" else 600
        dl_sku   = "PHL-HUE-DL" if tier == "premium" else "PHL-DL-12W"
        items.append(_item("Lighting", dl_brand, dl_prod,
            "unit", dl_count, dl_rate, sku=dl_sku,
            note=f"Recessed downlights, 1 per ~25 sqft ceiling area"))
        return items

    def _build_wardrobe_items(self, tier, area_sqft, city_mult):
        width_ft  = round(math.sqrt(area_sqft) * 1.2, 1)
        height_ft = 9.0
        depth_ft  = 2.0
        carcass_sqft = round(2 * (width_ft*height_ft + width_ft*depth_ft + height_ft*depth_ft), 1)
        shutter_sqft = round(width_ft * height_ft, 1)
        items = []
        ply = PLYWOOD_CATALOG["premium"][0]
        items.append(_item("Wardrobe Carcass", ply["brand"],
            f"{ply['product']} (Wardrobe carcass)",
            "sqft", carcass_sqft, ply["price_per_sqft"], sku=ply["sku"],
            note=f"Wardrobe: {width_ft}ft W x {height_ft}ft H x {depth_ft}ft D"))
        items.append(_item("Wardrobe Shutters","Greenlam",
            "High-Gloss Acrylic Laminate 1mm",
            "sqft", shutter_sqft, 380, sku="GL-ACS-HG",
            note=f"Shutter area: {width_ft}ft x {height_ft}ft"))
        handle_count = max(4, int(width_ft / 2))
        items.append(_item("Wardrobe Hardware","Hafele",
            "Brass Pull Handle 256mm",
            "unit", handle_count, 850, sku="HF-PH-256",
            note="Recessed brass handles"))
        hinge_count = max(6, handle_count * 2)
        items.append(_item("Wardrobe Hardware","Hettich",
            "Soft-Close Concealed Hinge",
            "unit", hinge_count, 320, sku="HT-SCH-35",
            note=f"Soft-close for {handle_count} doors"))
        edgeband_rft = round((width_ft + height_ft + depth_ft) * 8, 1)
        items.append(_item("Wardrobe Hardware","Rehau",
            "PVC Edgebanding 22mm",
            "rft", edgeband_rft, 18, sku="REHAU-EB-22",
            note="All exposed plywood edges"))
        return items

    def _select_hardware(self, tier, room_type):
        catalog = HARDWARE_CATALOG.get(tier, HARDWARE_CATALOG["mid"])
        items = []
        for item in catalog:
            if item["category"] in ("Cove Lighting", "Lighting", "Smart Lighting") and tier in ("mid","premium"):
                continue
            items.append(_item(item["category"], item["brand"], item["product"],
                "unit", 1, item["price_inr"], sku=item["sku"],
                priority="must_have" if item["category"] in ("Fan","Switches") else "nice_to_have"))
        return items

    def _select_sanitary(self, tier):
        return [_item(i["category"], i["brand"], i["product"], "unit", 1, i["price_inr"], sku=i["sku"])
                for i in SANITARY_CATALOG.get(tier, SANITARY_CATALOG["mid"])]

    def _build_waterproofing(self, tier, floor_area):
        wp = WATERPROOFING_CATALOG.get(tier, WATERPROOFING_CATALOG["mid"])[0]
        kits = max(1, math.ceil(floor_area / wp["coverage_sqft"]) + 1)
        return [_item("Waterproofing Material", wp["brand"], wp["product"],
            "kit", kits, wp["price_inr"], sku=wp["sku"],
            note="Apply before tiling. 24hr flood test required.")]

    def _select_kitchen_items(self, tier):
        return [_item(i["category"], i["brand"], i["product"], "unit", 1, i["price_inr"],
            sku=i["sku"], priority="must_have" if i["category"] in ("Chimney","Sink") else "nice_to_have")
                for i in KITCHEN_CATALOG.get(tier, KITCHEN_CATALOG["mid"])]

    def _select_plywood(self, tier, quantities):
        ply = PLYWOOD_CATALOG.get(tier, PLYWOOD_CATALOG["mid"])[0]
        sqft = round(float(quantities.get("plywood_sqft", 35.0)), 1)
        return [_item("Plywood", ply["brand"], ply["product"], "sqft", sqft,
            ply["price_per_sqft"], sku=ply["sku"])]

    def _build_labour_line_items(self, tier, area_sqft, wall_area, room_type,
                                  city_mult, scope, existing_items):
        scope_mod = {"cosmetic_only": 0.65, "partial": 1.0,
                     "full_room": 1.25, "structural_plus": 1.55}.get(scope, 1.0)
        items = []
        lr = LABOUR_RATES
        cats = {i.get("category","") for i in existing_items}

        def _lab(category, product, unit, qty, base_rate, note=""):
            rate = round(base_rate * city_mult * scope_mod)
            return _item(category, "Contractor", product, unit, round(qty,1), rate, note=note)

        # Painting labour — walls + ceiling area
        total_paint_area = round(wall_area + area_sqft, 1)
        items.append(_lab("Labour - Painting",
            "Wall & Ceiling Painting (putty + primer + 2 coats)",
            "sqft", total_paint_area, lr["painting_per_sqft"][tier],
            note=f"{wall_area} sqft walls + {area_sqft} sqft ceiling"))

        # Texture application labour
        if tier in ("mid","premium") and "Wall Texture" in cats:
            items.append(_lab("Labour - Painting",
                "Wall Texture / Micro-Cement Application",
                "sqft", wall_area, lr["wall_texture_per_sqft"][tier],
                note=f"Specialist application on {wall_area} sqft"))

        # Tiling labour — floor
        items.append(_lab("Labour - Tiling",
            "Floor Tile Laying (incl. adhesive + level check)",
            "sqft", area_sqft, lr["tiling_per_sqft"][tier],
            note=f"{area_sqft} sqft floor"))

        # Wall tile labour
        if any(c == "Wall Tiles" for c in cats):
            wt_area = round(area_sqft * 0.6, 1)
            items.append(_lab("Labour - Tiling",
                "Wall Tile Laying Labour",
                "sqft", wt_area, lr["tiling_per_sqft"][tier],
                note=f"Wall tile area ~{wt_area} sqft"))

        # Skirting installation
        if "Skirting" in cats:
            perimeter = round(4 * math.sqrt(area_sqft), 1)
            items.append(_lab("Labour - Tiling",
                "Skirting Installation Labour",
                "rft", perimeter, 35,
                note=f"Room perimeter ~{perimeter} rft"))

        # False ceiling labour
        if any("False Ceiling" in c for c in cats):
            items.append(_lab("Labour - False Ceiling",
                "False Ceiling Fabrication & Installation (GI + gypsum + POP)",
                "sqft", area_sqft, lr["false_ceiling_per_sqft"][tier],
                note=f"{area_sqft} sqft — GI fixing, boarding, tape-finish"))

        # Wardrobe / carpentry labour
        has_wardrobe = any(c.startswith("Wardrobe") for c in cats)
        has_ply = "Plywood" in cats or "Plywood / Shuttering" in cats
        if has_wardrobe or has_ply:
            width_ft = round(math.sqrt(area_sqft) * 1.2, 1)
            carp_sqft = round(width_ft * 9, 1) if has_wardrobe else round(area_sqft * 0.18, 1)
            items.append(_lab("Labour - Carpentry",
                "Wardrobe / Carpentry Fabrication & Installation",
                "sqft", carp_sqft, lr["carpentry_per_sqft"][tier],
                note=f"{carp_sqft} sqft carpentry work"))

        # Electrical labour
        elec_points = 0
        for it in existing_items:
            c = it.get("category","")
            if c in ("Fan","Lighting","Smart Lighting"):
                elec_points += int(it.get("qty", 1))
            elif c == "Switches":
                elec_points += 4
            elif "LED Strip" in c:
                elec_points += 2
        elec_points = max(6, elec_points)
        items.append(_lab("Labour - Electrical",
            "Electrical Points — Fan, Lights, Switches",
            "point", elec_points, lr["electrical_per_point"][tier],
            note=f"{elec_points} electrical points"))

        # Plumbing labour
        if room_type in ("kitchen","bathroom","full_home"):
            pp = 4 if room_type == "kitchen" else 6
            items.append(_lab("Labour - Plumbing",
                "Plumbing Points — Supply & Drain",
                "point", pp, lr["plumbing_per_point"][tier],
                note=f"{pp} plumbing points"))

        # Waterproofing labour
        if room_type in ("bathroom","full_home"):
            items.append(_lab("Labour - Waterproofing",
                "Waterproofing Application (2 coats)",
                "sqft", area_sqft, lr["waterproofing_per_sqft"][tier],
                note=f"{area_sqft} sqft. Flood test required after."))

        return items

    @staticmethod
    def _suggest_suppliers(city):
        base = [
            {"name": "Asian Paints Dealer Network", "type": "Paint", "url": "asianpaints.com/dealer-locator"},
            {"name": "Kajaria World (Authorised)", "type": "Tiles", "url": "kajaria.com/storelocator"},
            {"name": "Greenply Dealer", "type": "Plywood", "url": "greenply.com/dealer"},
            {"name": "Havells Galaxy", "type": "Electrical", "url": "havells.com/store-locator"},
            {"name": "Jaquar World", "type": "Bathroom Fittings", "url": "jaquar.com/store-locator"},
            {"name": "MYK Laticrete Dealer", "type": "Tile Adhesive & Grout", "url": "myklaticrete.com"},
            {"name": "Saint-Gobain Gyproc", "type": "False Ceiling", "url": "saint-gobain.in"},
        ]
        extras = {
            "Hyderabad": [{"name": "Sriram Agencies, Secunderabad", "type": "Multi-category", "url": ""}],
            "Bangalore":  [{"name": "Hindware Studio, Whitefield", "type": "Bathroom", "url": "hindware.com"}],
        }
        return base + extras.get(city, [])
