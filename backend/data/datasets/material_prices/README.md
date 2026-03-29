# India Material Prices — Historical Data

This directory holds **real historical construction material price data** for
India. When `india_material_prices_historical.csv` is present here, the
ARKEN `PriceForecastAgent` will use it for genuine time-series forecasting
instead of the seed-based estimated fallback.

---

## Expected CSV format

**Filename:** `india_material_prices_historical.csv`

| Column | Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` | Date of the price observation |
| `material_key` | string | Matches a key in `SEED_DATA` (see below) |
| `price_inr` | float | Price in Indian Rupees, in the material's native unit |
| `source` | string | Data source identifier (e.g., `indiamart`, `cmie`, `constructionworld`) |
| `city` | string | City name, or `national` for pan-India index (optional but recommended) |

### Example rows

```csv
date,material_key,price_inr,source,city
2025-01-15,cement_opc53_per_bag_50kg,385.00,indiamart,Mumbai
2025-01-15,cement_opc53_per_bag_50kg,370.00,indiamart,Hyderabad
2025-02-01,steel_tmt_fe500_per_kg,63.50,constructionworld,national
2025-03-10,kajaria_tiles_per_sqft,88.50,cmie,Bangalore
```

---

## Valid `material_key` values

These must exactly match the keys in `SEED_DATA` in `price_forecast.py`:

| material_key | Unit | Description |
|---|---|---|
| `cement_opc53_per_bag_50kg` | per 50 kg bag | OPC 53 grade cement |
| `steel_tmt_fe500_per_kg` | per kg | TMT Fe500 reinforcement bar |
| `teak_wood_per_cft` | per cubic foot | Grade A teak timber |
| `kajaria_tiles_per_sqft` | per sq ft | 600×600 glazed vitrified tiles |
| `copper_wire_per_kg` | per kg | Electrical-grade copper wire |
| `sand_river_per_brass` | per brass (100 cft) | River sand |
| `bricks_per_1000` | per 1000 units | Standard red clay bricks |
| `granite_per_sqft` | per sq ft | Black Galaxy granite slab |
| `asian_paints_premium_per_litre` | per litre | Asian Paints Royale Aspira |
| `pvc_upvc_window_per_sqft` | per sq ft | UPVC double-glazed window |
| `modular_kitchen_per_sqft` | per sq ft | Mid-range laminate modular kitchen |
| `bathroom_sanitary_set` | per set | Hindware/Cera standard suite |

---

## Where to obtain real data

### 1. IndiaMART
- URL: https://www.indiamart.com
- Navigate to each product category and record listed prices with dates.
- Prices vary by city and seller — record `city` for each row.
- Free to access; manual data collection required unless you have API access.

### 2. ConstructionWorld
- URL: https://www.constructionworld.in/price-trends
- Publishes monthly price indices and commodity reports.
- Suitable for national-level monthly data.

### 3. CMIE (Centre for Monitoring Indian Economy)
- URL: https://cmie.com — requires a subscription
- The **Commodities & Prices** module provides verified weekly/monthly
  construction input prices across 40+ cities.
- Recommended as the most authoritative source if budget permits.

### 4. NHB Residex + RBI data
- NHB publishes a quarterly Housing Price Index: https://nhb.org.in/research/residex
- Useful for property value benchmarking but not raw material prices.

### 5. Manufacturer price lists
- Asian Paints: https://www.asianpaints.com (annual price revision list)
- Kajaria: https://www.kajariaceramics.com (dealer price list)
- JSW/TATA Steel: TMT bar price cards updated monthly

---

## How to run the price update script

Once you have collected new price data and appended it to the CSV:

```bash
# From the backend directory
python -c "
from agents.price_forecast import PriceForecastAgent
agent = PriceForecastAgent()
print(agent.forecast_material('cement_opc53_per_bag_50kg', city='Mumbai'))
"
```

The agent will automatically detect the updated CSV on the next
`PriceForecastAgent()` instantiation (the `RealPriceDataManager` is a
singleton — restart the backend process to reload).

To force a reload without restarting:

```python
from agents.price_forecast import _price_manager
_price_manager = None  # clear singleton
from agents.price_forecast import _get_price_manager
mgr = _get_price_manager()  # reloads CSV
```

---

## Data quality indicators in API output

Every forecast result from `PriceForecastAgent` includes:

| Field | Meaning |
|---|---|
| `data_quality` | `"historical"` (real CSV used) or `"estimated_seed_based"` (fallback) |
| `confidence_note` | Human-readable explanation of how confident the forecast is and why |
| `last_verified_date` | Date the seed prices were last manually verified (`2026-01-01` baseline) |
| `buy_now_signal` | `true` if 90-day price increase exceeds 5%, `false` otherwise |

The `portfolio_summary.data_reliability` field reports `"high"` / `"medium"` /
`"estimated"` at the project level so the frontend can display a clear
reliability badge to end users.
