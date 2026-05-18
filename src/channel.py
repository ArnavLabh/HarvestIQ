"""Channel & timing recommender.

Maps (device_type, language, crop, current stage) onto:
  - the best channel to reach the grower,
  - the best week to send the message relative to a stage,
  - the recommended creative format.

Channel priority is a transparent rules engine — easy to defend to
judges who'll ask "why this channel?". The optimal-week mapping is
derived from agronomic stage-to-product timing windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

# Product → ideal application stage(s). Drives both the message moment
# and how urgent the call-to-action should be.
PRODUCT_STAGE_WINDOW: dict[str, list[str]] = {
    "Topik 15 WP":      ["tillering"],                  # wheat herbicide
    "Tilt 250 EC":      ["tillering", "flowering"],     # wheat fungicide
    "Score 250 EC":     ["flowering"],                  # mustard fungicide
    "Actara 25 WG":     ["vegetative", "flowering"],    # chickpea insecticide
    "Kavach 75 WP":     ["tuber_initiation", "flowering"],  # potato fungicide
    "Amistar 250 SC":   ["flowering"],                  # broad-spectrum fungicide
}

# Lead-time in days before the ideal stage to send the message.
LEAD_DAYS = 7

TONE_BY_LANGUAGE = {
    "Hindi": "warm, respectful, agronomically precise",
    "Punjabi": "warm, direct, action-oriented",
    "Marathi": "respectful, technical clarity",
    "Gujarati": "warm, prosperity-oriented framing",
    "Telugu": "respectful, formal, agronomically precise",
    "Tamil": "respectful, formal, agronomically precise",
    "Kannada": "respectful, formal, agronomically precise",
    "Bengali": "warm, story-driven",
    "Odia": "warm, respectful",
    "Malayalam": "respectful, formal, agronomically precise",
}


@dataclass
class ChannelRecommendation:
    primary_channel: str
    secondary_channel: str
    format: str
    rationale: str


def recommend_channel(device_type: str, language: str, offline_attended: bool) -> ChannelRecommendation:
    device = (device_type or "unknown").lower()

    if device == "smartphone":
        return ChannelRecommendation(
            primary_channel="WhatsApp",
            secondary_channel="SMS",
            format="60-sec vertical video + 1-line caption + product image",
            rationale="Smartphone user — rich-media WhatsApp drives the strongest click-through.",
        )
    if device == "keypad":
        return ChannelRecommendation(
            primary_channel="IVR voice call",
            secondary_channel="SMS",
            format="45-sec voice message in local language + retailer follow-up",
            rationale="Feature-phone user — voice respects low literacy and works on 2G.",
        )
    if offline_attended:
        return ChannelRecommendation(
            primary_channel="Retailer-led conversation",
            secondary_channel="SMS",
            format="One-page printed leaflet + retailer talking points",
            rationale="Grower already engages offline — retailer touch is highest-trust channel.",
        )
    return ChannelRecommendation(
        primary_channel="SMS",
        secondary_channel="Retailer-led conversation",
        format="Short SMS + retailer follow-up",
        rationale="Unknown device — SMS guarantees delivery; retailer adds context.",
    )


def optimal_send_date(crop_calendar: dict, product: str,
                      as_of: date | None = None) -> date | None:
    """Pick the date to send: LEAD_DAYS before the next applicable stage."""
    if not crop_calendar:
        return None
    as_of = as_of or date(2026, 2, 15)
    target_stages = PRODUCT_STAGE_WINDOW.get(product, [])
    if not target_stages:
        return as_of  # generic: send today

    for stage in crop_calendar.get("stages", []):
        if stage.get("stage") in target_stages:
            try:
                approx = date.fromisoformat(stage["approx"])
            except (KeyError, ValueError):
                continue
            ideal = approx - timedelta(days=LEAD_DAYS)
            if ideal >= as_of:
                return ideal
            return as_of  # window already started — send now
    return as_of


def tone_for_language(language: str) -> str:
    return TONE_BY_LANGUAGE.get(language, "warm, respectful, agronomically precise")
