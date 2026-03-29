"""
ARKEN — Material Price Scraper v1.0
======================================
Scrapes IndiaMART product pages for live material prices and appends to the
historical CSV without overwriting or duplicating existing rows.

Materials covered:
  - cement_opc53_per_bag_50kg       → OPC 53 Grade Cement (50 kg bag)
  - steel_tmt_fe500_per_kg          → TMT Steel Fe500 (per kg)
  - kajaria_tiles_per_sqft          → Kajaria Floor Tiles (per sqft)
  - asian_paints_premium_per_litre  → Asian Paints Royale (per litre)

IndiaMART structure note:
  IndiaMART product pages render prices inside structured microdata /
  JSON-LD or within well-known CSS selectors.  Because IndiaMART can
  change its markup, this scraper uses a layered extraction strategy:
    1. JSON-LD   (schema.org/Product — most stable)
    2. CSS selectors (data-* attributes, aria-labels)
    3. Regex fallback on raw page text
  If no price is confidently extracted the material is skipped (logged as
  a warning) and no row is written — we never store guesses.

Source tagging:
  source      = "indiamart_scraped"
  source_type = "real_scraped"

Duplicate guard:
  A row is appended ONLY when (date + material_key + city) does not
  already exist in the CSV.  Existing rows are NEVER modified.

Usage:
    from services.price_scraper import MaterialPriceScraper
    scraper = MaterialPriceScraper()
    scraper.run_monthly_update()

    # Or single material + city:
    price = scraper.scrape_price("cement_opc53_per_bag_50kg", "Hyderabad")
"""

from __future__ import annotations

import csv
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_APP_DIR     = Path(os.getenv("ARKEN_APP_DIR", "/app"))

def _resolve(app_rel: str, local_rel: str) -> Path:
    a = _APP_DIR    / app_rel
    b = _BACKEND_DIR / local_rel
    return a if a.exists() else b

_PRICES_CSV = _resolve(
    "data/datasets/material_prices/india_material_prices_historical.csv",
    "data/datasets/material_prices/india_material_prices_historical.csv",
)

_CSV_FIELDNAMES = ["date", "material_key", "price_inr", "city", "source", "source_type"]

# ── HTTP settings ─────────────────────────────────────────────────────────────
_REQUEST_TIMEOUT   = 18          # seconds
_RETRY_COUNT       = 2
_BETWEEN_REQUESTS  = 3.5         # seconds — be polite to IndiaMART

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── IndiaMART search URLs (curated for each material) ───────────────────────
# Each URL is an IndiaMART category/search page that reliably lists products
# with machine-readable pricing for the target material.
_MATERIAL_URLS: Dict[str, str] = {
    "cement_opc53_per_bag_50kg": (
        "https://www.indiamart.com/proddetail/opc-53-grade-cement-50-kg-bag.html"
        # Fallback search if direct page fails:
        # "https://www.indiamart.com/search.mp?ss=opc+53+grade+cement+50+kg"
    ),
    "steel_tmt_fe500_per_kg": (
        "https://www.indiamart.com/proddetail/fe-500-tmt-steel-bar.html"
    ),
    "kajaria_tiles_per_sqft": (
        "https://www.indiamart.com/proddetail/kajaria-ceramic-tiles.html"
    ),
    "asian_paints_premium_per_litre": (
        "https://www.indiamart.com/proddetail/asian-paints-royale-emulsion.html"
    ),
}

# Alternative search URLs used when direct product pages fail
_MATERIAL_SEARCH_URLS: Dict[str, str] = {
    "cement_opc53_per_bag_50kg":
        "https://www.indiamart.com/search.mp?ss=opc+53+grade+cement+50+kg+bag",
    "steel_tmt_fe500_per_kg":
        "https://www.indiamart.com/search.mp?ss=fe500+tmt+steel+bar+per+kg",
    "kajaria_tiles_per_sqft":
        "https://www.indiamart.com/search.mp?ss=kajaria+floor+tiles+per+sqft",
    "asian_paints_premium_per_litre":
        "https://www.indiamart.com/search.mp?ss=asian+paints+royale+per+litre",
}

# ── Plausible price ranges (INR) for sanity-check ────────────────────────────
# Based on 2020–2024 real data ranges in india_material_prices_historical.csv
_PLAUSIBLE_RANGES: Dict[str, Tuple[float, float]] = {
    "cement_opc53_per_bag_50kg":        (260.0,  800.0),   # ₹/bag
    "steel_tmt_fe500_per_kg":           ( 40.0,  120.0),   # ₹/kg
    "kajaria_tiles_per_sqft":           ( 25.0,  180.0),   # ₹/sqft
    "asian_paints_premium_per_litre":   (200.0,  700.0),   # ₹/litre
}

# ── Cities to scrape ──────────────────────────────────────────────────────────
_TARGET_CITIES = ["Hyderabad", "Bangalore", "Mumbai", "Delhi NCR", "Chennai"]

# City → IndiaMART location query fragment (used in search URL refinement)
_CITY_LOCATION: Dict[str, str] = {
    "Hyderabad":  "Hyderabad",
    "Bangalore":  "Bangalore",
    "Mumbai":     "Mumbai",
    "Delhi NCR":  "Delhi",
    "Chennai":    "Chennai",
}


class MaterialPriceScraper:
    """
    Scrapes IndiaMART for live construction material prices and appends
    new rows to india_material_prices_historical.csv.

    Thread-safety: not designed for concurrent use; run_monthly_update()
    acquires a file-level lock via CSV append semantics (last-write-wins
    for concurrent processes is acceptable since the duplicate guard is
    checked immediately before each write).
    """

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else _PRICES_CSV
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()
        self._existing: set = self._load_existing_keys()
        logger.info(
            f"[PriceScraper] Initialised. CSV: {self._csv_path} "
            f"({len(self._existing)} existing rows)"
        )

    # ── CSV helpers ───────────────────────────────────────────────────────────

    def _ensure_csv_header(self) -> None:
        """Write header row if CSV is empty or missing."""
        if not self._csv_path.exists() or self._csv_path.stat().st_size == 0:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
                writer.writeheader()
            logger.info(f"[PriceScraper] Created CSV with header: {self._csv_path}")

    def _load_existing_keys(self) -> set:
        """Load (date, material_key, city) tuples already in the CSV."""
        keys: set = set()
        try:
            with open(self._csv_path, "r", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    date = (row.get("date") or "").strip()[:10]  # YYYY-MM-DD
                    mat  = (row.get("material_key") or "").strip()
                    city = (row.get("city") or "").strip()
                    if date and mat and city:
                        keys.add((date, mat, city))
        except Exception as e:
            logger.warning(f"[PriceScraper] Could not load existing keys: {e}")
        return keys

    def _append_row(self, row: Dict[str, Any]) -> bool:
        """
        Append a single row to the CSV — ONLY if (date+material_key+city)
        doesn't already exist.  Returns True if written, False if skipped.
        """
        date = str(row.get("date", ""))[:10]
        mat  = str(row.get("material_key", ""))
        city = str(row.get("city", ""))
        key  = (date, mat, city)

        if key in self._existing:
            logger.debug(f"[PriceScraper] Skipping duplicate: {key}")
            return False

        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
                writer.writerow({
                    "date":        date,
                    "material_key": mat,
                    "price_inr":   round(float(row["price_inr"]), 2),
                    "city":        city,
                    "source":      "indiamart_scraped",
                    "source_type": "real_scraped",
                })
            self._existing.add(key)
            return True
        except Exception as e:
            logger.error(f"[PriceScraper] Failed to append row {key}: {e}")
            return False

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_page(url: str) -> Optional[str]:
        """
        Fetch a URL with retry logic.
        Returns HTML text or None on persistent failure.
        """
        try:
            import requests
        except ImportError:
            logger.error("[PriceScraper] 'requests' not installed. pip install requests")
            return None

        for attempt in range(1, _RETRY_COUNT + 1):
            try:
                resp = requests.get(
                    url,
                    headers=_HEADERS,
                    timeout=_REQUEST_TIMEOUT,
                    allow_redirects=True,
                )
                if resp.status_code == 200:
                    return resp.text
                logger.warning(
                    f"[PriceScraper] HTTP {resp.status_code} on attempt {attempt}: {url}"
                )
            except Exception as e:
                logger.warning(f"[PriceScraper] Request error attempt {attempt}: {e}")
            if attempt < _RETRY_COUNT:
                time.sleep(2.0 * attempt)

        return None

    # ── Price extraction ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_price_from_json_ld(html: str) -> Optional[float]:
        """
        Extract price from schema.org JSON-LD embedded in page.
        IndiaMART embeds <script type="application/ld+json"> with
        Product schema when a product page is directly loaded.
        """
        try:
            import json
            pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
            for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
                try:
                    data = json.loads(match.group(1))
                    # Handle both single object and @graph array
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get("@type") in ("Product", "Offer"):
                            # Direct Offer
                            offer = item.get("offers") or item
                            if isinstance(offer, dict):
                                price = offer.get("price") or offer.get("lowPrice")
                                if price:
                                    return float(str(price).replace(",", ""))
                            elif isinstance(offer, list) and offer:
                                price = offer[0].get("price")
                                if price:
                                    return float(str(price).replace(",", ""))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_price_from_css(html: str) -> Optional[float]:
        """
        Extract price from known IndiaMART CSS patterns.
        IndiaMART uses several price display selectors over time.
        We match multiple patterns for resilience.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("[PriceScraper] 'beautifulsoup4' not installed. pip install beautifulsoup4")
            return None

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Pattern 1: price in data-price attribute (IndiaMART product cards)
            for el in soup.find_all(attrs={"data-price": True}):
                try:
                    val = float(str(el["data-price"]).replace(",", "").strip())
                    if val > 0:
                        return val
                except (ValueError, TypeError):
                    pass

            # Pattern 2: <span class="prc"> or <span class="price">
            for cls in ("prc", "price", "price-unit", "selling-price", "product-price"):
                el = soup.find("span", class_=re.compile(cls, re.IGNORECASE))
                if el:
                    text = el.get_text(strip=True)
                    nums = re.findall(r"[\d,]+(?:\.\d+)?", text.replace("₹", "").replace("Rs", ""))
                    for n in nums:
                        try:
                            val = float(n.replace(",", ""))
                            if val > 1:
                                return val
                        except ValueError:
                            pass

            # Pattern 3: aria-label containing price
            for el in soup.find_all(attrs={"aria-label": True}):
                label = str(el.get("aria-label", ""))
                if "price" in label.lower() or "₹" in label or "rs" in label.lower():
                    nums = re.findall(r"[\d,]+(?:\.\d+)?", label.replace("₹", ""))
                    for n in nums:
                        try:
                            val = float(n.replace(",", ""))
                            if val > 1:
                                return val
                        except ValueError:
                            pass

            # Pattern 4: meta itemprop="price"
            el = soup.find("meta", itemprop="price")
            if el and el.get("content"):
                try:
                    return float(str(el["content"]).replace(",", ""))
                except (ValueError, TypeError):
                    pass

        except Exception as e:
            logger.debug(f"[PriceScraper] CSS extraction error: {e}")

        return None

    @staticmethod
    def _extract_price_regex(html: str) -> Optional[float]:
        """
        Last-resort regex extraction: find ₹ or Rs followed by a number
        in the page body text.  Returns the first plausible price found.
        """
        # Match ₹1,234 / Rs. 1,234 / INR 1,234 patterns
        patterns = [
            r"₹\s*([\d,]+(?:\.\d+)?)",
            r"Rs\.?\s*([\d,]+(?:\.\d+)?)",
            r"INR\s*([\d,]+(?:\.\d+)?)",
            r'"price"\s*:\s*"?([\d,]+(?:\.\d+)?)"?',
            r'"lowPrice"\s*:\s*"?([\d,]+(?:\.\d+)?)"?',
        ]
        candidates: List[float] = []
        for pat in patterns:
            for match in re.finditer(pat, html):
                try:
                    val = float(match.group(1).replace(",", ""))
                    if val > 1:
                        candidates.append(val)
                except ValueError:
                    pass

        if not candidates:
            return None

        # Return the median of first 10 candidates (avoid outlier prices
        # from footer/nav elements that often appear before product content)
        from statistics import median
        return median(candidates[:10])

    def _extract_price(self, html: str, material_key: str) -> Optional[float]:
        """
        Run extraction chain: JSON-LD → CSS → Regex.
        Validates result against plausible range for the material.
        """
        price: Optional[float] = None

        for extractor in [
            self._extract_price_from_json_ld,
            self._extract_price_from_css,
            self._extract_price_regex,
        ]:
            result = extractor(html)
            if result is not None and result > 0:
                price = result
                break

        if price is None:
            return None

        lo, hi = _PLAUSIBLE_RANGES.get(material_key, (1.0, 1_000_000.0))
        if not (lo <= price <= hi):
            logger.warning(
                f"[PriceScraper] Price ₹{price:.2f} for {material_key} is outside "
                f"plausible range [{lo}, {hi}] — discarding"
            )
            return None

        return round(price, 2)

    # ── Public scraping API ───────────────────────────────────────────────────

    def scrape_price(
        self, material_key: str, city: str
    ) -> Optional[float]:
        """
        Scrape current price for one material in one city from IndiaMART.

        Strategy:
          1. Fetch the curated direct product page URL.
          2. If price extraction fails, fall back to city-qualified search URL.
          3. Validate price against plausible range.

        Returns:
            Price in INR (float) or None if scraping fails.
        """
        if material_key not in _MATERIAL_URLS:
            logger.error(f"[PriceScraper] Unknown material_key: {material_key}")
            return None

        city_loc = _CITY_LOCATION.get(city, city)

        # Attempt 1: curated direct URL
        url = _MATERIAL_URLS[material_key]
        html = self._fetch_page(url)
        if html:
            price = self._extract_price(html, material_key)
            if price is not None:
                logger.info(
                    f"[PriceScraper] ✓ {material_key} @ {city}: ₹{price:.2f} "
                    f"(direct page)"
                )
                return price

        time.sleep(_BETWEEN_REQUESTS)

        # Attempt 2: city-qualified search URL
        search_base = _MATERIAL_SEARCH_URLS.get(material_key, "")
        if search_base:
            search_url = f"{search_base}+{city_loc.replace(' ', '+')}"
            html = self._fetch_page(search_url)
            if html:
                price = self._extract_price(html, material_key)
                if price is not None:
                    logger.info(
                        f"[PriceScraper] ✓ {material_key} @ {city}: ₹{price:.2f} "
                        f"(search fallback)"
                    )
                    return price

        logger.warning(
            f"[PriceScraper] ✗ Could not extract price for {material_key} @ {city}"
        )
        return None

    def scrape_and_append(
        self,
        material_key: str,
        city: str,
        date: Optional[str] = None,
    ) -> bool:
        """
        Scrape price and append to CSV if not already present.

        Args:
            material_key: e.g. "cement_opc53_per_bag_50kg"
            city:         e.g. "Hyderabad"
            date:         ISO date string YYYY-MM-DD (defaults to today)

        Returns:
            True if a new row was appended, False otherwise.
        """
        if date is None:
            date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        # Normalise to first of month for monthly time-series consistency
        date = date[:8] + "01"

        # Skip if already exists
        if (date, material_key, city) in self._existing:
            logger.debug(
                f"[PriceScraper] Row already exists — skipping: "
                f"{material_key} @ {city} on {date}"
            )
            return False

        price = self.scrape_price(material_key, city)
        if price is None:
            return False

        written = self._append_row({
            "date":         date,
            "material_key": material_key,
            "price_inr":    price,
            "city":         city,
        })
        return written

    def run_monthly_update(
        self,
        cities: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a full monthly scrape for all 4 materials × N cities.

        Args:
            cities: List of city names (defaults to _TARGET_CITIES).
            date:   ISO date YYYY-MM-DD (defaults to first of current month).

        Returns:
            Summary dict: {
                total_attempted, total_written, total_skipped, total_failed,
                per_material: {material_key: {city: "written"|"skipped"|"failed"}}
            }
        """
        if cities is None:
            cities = _TARGET_CITIES
        if date is None:
            now = datetime.now(tz=timezone.utc)
            date = now.strftime("%Y-%m-01")

        materials = list(_MATERIAL_URLS.keys())
        logger.info(
            f"[PriceScraper] Starting monthly update: {len(materials)} materials × "
            f"{len(cities)} cities → date={date}"
        )

        results: Dict[str, Dict[str, str]] = {m: {} for m in materials}
        total_attempted = total_written = total_skipped = total_failed = 0

        for mat in materials:
            for city in cities:
                total_attempted += 1
                try:
                    key = (date[:8] + "01", mat, city)
                    if key in self._existing:
                        results[mat][city] = "skipped"
                        total_skipped += 1
                        logger.debug(
                            f"[PriceScraper] Skip (exists): {mat} @ {city} {date}"
                        )
                        continue

                    written = self.scrape_and_append(mat, city, date)
                    if written:
                        results[mat][city] = "written"
                        total_written += 1
                    else:
                        # scrape_and_append returns False either because
                        # scraping failed OR row already existed (checked again)
                        if (date[:8] + "01", mat, city) in self._existing:
                            results[mat][city] = "skipped"
                            total_skipped += 1
                        else:
                            results[mat][city] = "failed"
                            total_failed += 1

                except Exception as e:
                    logger.error(
                        f"[PriceScraper] Unexpected error for {mat} @ {city}: {e}",
                        exc_info=True,
                    )
                    results[mat][city] = "failed"
                    total_failed += 1

                # Polite delay between requests
                time.sleep(_BETWEEN_REQUESTS)

        summary = {
            "date":             date,
            "total_attempted":  total_attempted,
            "total_written":    total_written,
            "total_skipped":    total_skipped,
            "total_failed":     total_failed,
            "per_material":     results,
        }

        logger.info(
            f"[PriceScraper] Monthly update complete: "
            f"attempted={total_attempted} written={total_written} "
            f"skipped={total_skipped} failed={total_failed}"
        )
        return summary


# ── Module-level convenience ──────────────────────────────────────────────────

_scraper_instance: Optional[MaterialPriceScraper] = None


def get_price_scraper() -> MaterialPriceScraper:
    """Return the singleton MaterialPriceScraper."""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = MaterialPriceScraper()
    return _scraper_instance


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    scraper = MaterialPriceScraper()
    summary = scraper.run_monthly_update()
    print(f"\n✅ Monthly scrape complete:")
    print(f"   Written:  {summary['total_written']}")
    print(f"   Skipped:  {summary['total_skipped']}")
    print(f"   Failed:   {summary['total_failed']}")
