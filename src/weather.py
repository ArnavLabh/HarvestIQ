"""Weather context via Open-Meteo (free, no API key required).

We use:
  - geocoding-api.open-meteo.com  → district name → lat/lon
  - api.open-meteo.com/v1/forecast → current + 7-day forecast

Cached on disk so a demo doesn't re-hit the API every reload, and so it
still works offline once warmed up (matches the low-connectivity
constraint in the brief).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "weather"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass
class WeatherSnapshot:
    district: str
    lat: float
    lon: float
    temp_c: float
    humidity_pct: float
    rain_mm_24h: float
    rain_forecast_7d_mm: float
    rainy_days_next_7: int
    summary: str
    source: str  # "live" or "cache" or "fallback"

    def disease_pressure(self) -> str:
        """Heuristic: warm + humid → high fungal disease risk."""
        if self.humidity_pct >= 70 and 15 <= self.temp_c <= 28:
            return "high"
        if self.humidity_pct >= 55:
            return "moderate"
        return "low"

    def spray_window(self) -> str:
        """Whether the next few days are good for foliar spray."""
        if self.rainy_days_next_7 >= 3:
            return "poor"  # rain will wash off
        if self.rainy_days_next_7 == 0 and self.humidity_pct < 30:
            return "marginal"  # too dry, drift risk
        return "good"


def _cache_path(district: str) -> Path:
    safe = district.replace(" ", "_").lower()
    return CACHE_DIR / f"{safe}.json"


def _load_cache(district: str) -> dict | None:
    p = _cache_path(district)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if data.get("cached_on") == date.today().isoformat():
            return data
    except Exception:  # noqa: BLE001
        return None
    return None


def _save_cache(district: str, payload: dict) -> None:
    payload["cached_on"] = date.today().isoformat()
    _cache_path(district).write_text(json.dumps(payload))


def _fallback(district: str) -> WeatherSnapshot:
    return WeatherSnapshot(
        district=district, lat=0.0, lon=0.0,
        temp_c=22.0, humidity_pct=55.0, rain_mm_24h=0.0,
        rain_forecast_7d_mm=0.0, rainy_days_next_7=0,
        summary="Weather unavailable — using seasonal Rabi defaults.",
        source="fallback",
    )


@lru_cache(maxsize=128)
def get_weather(district: str, state: str = "India") -> WeatherSnapshot:
    cached = _load_cache(district)
    if cached:
        return WeatherSnapshot(**{k: v for k, v in cached.items() if k != "cached_on"})

    try:
        geo = requests.get(GEOCODE_URL, params={
            "name": district, "country": "India", "count": 1,
        }, timeout=10).json()
        if not geo.get("results"):
            return _fallback(district)
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        fc = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "daily": "precipitation_sum",
            "forecast_days": 7,
            "timezone": "Asia/Kolkata",
        }, timeout=10).json()

        cur = fc.get("current", {})
        daily = fc.get("daily", {})
        rains = daily.get("precipitation_sum", []) or []
        snap = WeatherSnapshot(
            district=district, lat=lat, lon=lon,
            temp_c=float(cur.get("temperature_2m", 22.0)),
            humidity_pct=float(cur.get("relative_humidity_2m", 55.0)),
            rain_mm_24h=float(cur.get("precipitation", 0.0)),
            rain_forecast_7d_mm=float(sum(rains)),
            rainy_days_next_7=sum(1 for r in rains if r and r > 1.0),
            summary=(
                f"{float(cur.get('temperature_2m',22.0)):.0f}°C, "
                f"{float(cur.get('relative_humidity_2m',55.0)):.0f}% RH, "
                f"{sum(rains):.0f} mm rain forecast over next 7 days"
            ),
            source="live",
        )
        _save_cache(district, snap.__dict__.copy())
        return snap
    except Exception:  # noqa: BLE001
        return _fallback(district)
