"""Auto-pick the right Syngenta product for a grower.

Track 1 only — uses crop + stage + weather. No POS or visit data
(those belong to Track 2). Output includes a transparent reasoning
trail so any judge can challenge the pick.
"""
from __future__ import annotations

from dataclasses import dataclass

from .weather import WeatherSnapshot

# (crop, stage) → list of (product, reason) candidates.
CROP_STAGE_CATALOG: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("wheat", "sowing"):        [("Cruiser 350 FS", "Seed treatment protects against early-stage pests.")],
    ("wheat", "tillering"):     [("Topik 15 WP", "Narrow-leaf weed control during the tillering window."),
                                  ("Axial 50 EC", "Alternative grass-weed herbicide for tillering.")],
    ("wheat", "flowering"):     [("Tilt 250 EC", "Controls rust and powdery mildew at flowering."),
                                  ("Amistar 250 SC", "Broad-spectrum fungicide for flowering-stage disease.")],
    ("mustard", "tillering"):   [("Score 250 EC", "Preventive fungicide ahead of flowering disease pressure.")],
    ("mustard", "flowering"):   [("Score 250 EC", "Alternaria blight protection at flowering.")],
    ("chickpea", "vegetative"): [("Actara 25 WG", "Controls aphids and jassids during vegetative growth.")],
    ("chickpea", "flowering"):  [("Actara 25 WG", "Pod-borer protection at flowering / podding stage.")],
    ("potato", "tuber_initiation"): [("Kavach 75 WP", "Late-blight protection at tuber initiation — critical window.")],
    ("potato", "flowering"):    [("Kavach 75 WP", "Late-blight protection during flowering — humid weather raises risk."),
                                  ("Amistar 250 SC", "Systemic alternative for late blight.")],
}

GENERIC_BY_CROP: dict[str, tuple[str, str]] = {
    "wheat":     ("Tilt 250 EC", "General-purpose fungicide for wheat."),
    "mustard":   ("Score 250 EC", "General-purpose fungicide for mustard."),
    "chickpea":  ("Actara 25 WG", "General-purpose insecticide for chickpea."),
    "potato":    ("Kavach 75 WP", "General-purpose fungicide for potato."),
    "barley":    ("Amistar 250 SC", "Broad-spectrum fungicide for cereals."),
    "lentil":    ("Amistar 250 SC", "Broad-spectrum fungicide for pulses."),
    "maize":     ("Amistar 250 SC", "Broad-spectrum fungicide for cereals."),
    "cumin":     ("Score 250 EC", "Disease protection for cumin."),
    "safflower": ("Score 250 EC", "Disease protection for safflower."),
}

FUNGICIDES = {"Tilt 250 EC", "Score 250 EC", "Kavach 75 WP", "Amistar 250 SC"}


@dataclass
class ProductRecommendation:
    product: str
    score: float
    reasons: list[str]
    weather_modifier: str


def recommend(*, crop: str, current_stage: str,
              weather: WeatherSnapshot) -> ProductRecommendation:
    candidates = CROP_STAGE_CATALOG.get((crop, current_stage), [])
    if not candidates and crop in GENERIC_BY_CROP:
        candidates = [GENERIC_BY_CROP[crop]]
    if not candidates:
        return ProductRecommendation(
            product="Amistar 250 SC", score=0.3,
            reasons=["No specific match — defaulting to broad-spectrum fungicide."],
            weather_modifier="none",
        )

    pressure = weather.disease_pressure()
    spray = weather.spray_window()

    best_score = -1.0
    best_product = candidates[0][0]
    best_reasons: list[str] = []
    weather_note = "neutral"

    for product, base_reason in candidates:
        score = 1.0
        reasons = [base_reason]

        is_fungicide = product in FUNGICIDES
        if is_fungicide and pressure == "high":
            score += 0.6
            reasons.append(
                f"Weather favours disease ({weather.temp_c:.0f}°C, "
                f"{weather.humidity_pct:.0f}% RH) — fungicide is urgent."
            )
            weather_note = "high disease pressure → boost"
        elif is_fungicide and pressure == "moderate":
            score += 0.2
            weather_note = "moderate disease pressure"
        elif is_fungicide and pressure == "low":
            score -= 0.2
            weather_note = "low disease pressure"

        if spray == "poor":
            score -= 0.3
            reasons.append("Heavy rain forecast — delay spray; lead with advisory tone.")
            weather_note = f"{weather_note} / poor spray window".strip(" /")

        if score > best_score:
            best_score = score
            best_product = product
            best_reasons = reasons

    return ProductRecommendation(
        product=best_product, score=round(best_score, 2),
        reasons=best_reasons, weather_modifier=weather_note,
    )
