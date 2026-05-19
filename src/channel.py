"""Channel and timing recommender + receptivity benchmarks.

Channel selection covers the five channels in the brief:
  WhatsApp, voice call (IVR), SMS, social media, retailer network.

Receptivity helpers compute historical engagement by segment so the
UI can show 'expected click rate for this segment' and suggest the
best creative variant.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

# Product → ideal stage windows (drives send timing)
PRODUCT_STAGE_WINDOW: dict[str, list[str]] = {
    "Topik 15 WP":      ["tillering"],
    "Tilt 250 EC":      ["tillering", "flowering"],
    "Score 250 EC":     ["flowering"],
    "Actara 25 WG":     ["vegetative", "flowering"],
    "Kavach 75 WP":     ["tuber_initiation", "flowering"],
    "Amistar 250 SC":   ["flowering"],
}

LEAD_DAYS = 7

TONE_BY_LANGUAGE = {
    "Hindi":     "warm, respectful, agronomically precise",
    "Punjabi":   "warm, direct, action-oriented",
    "Marathi":   "respectful, technical clarity",
    "Gujarati":  "warm, prosperity-oriented framing",
    "Telugu":    "respectful, formal, agronomically precise",
    "Tamil":     "respectful, formal, agronomically precise",
    "Kannada":   "respectful, formal, agronomically precise",
    "Bengali":   "warm, story-driven",
    "Odia":      "warm, respectful",
    "Malayalam": "respectful, formal, agronomically precise",
}


@dataclass
class ChannelRecommendation:
    primary_channel: str
    secondary_channel: str
    social_amplification: str
    format: str
    rationale: str


def recommend_channel(device_type: str, language: str,
                      offline_attended: bool) -> ChannelRecommendation:
    device = (device_type or "unknown").lower()

    if device == "smartphone":
        return ChannelRecommendation(
            primary_channel="WhatsApp",
            secondary_channel="SMS",
            social_amplification="Instagram Reels + Facebook district page in " + language,
            format="60-second vertical video + 1-line caption + product image",
            rationale="Smartphone user. WhatsApp rich media drives the strongest click-through; social amplification builds peer proof.",
        )
    if device == "keypad":
        return ChannelRecommendation(
            primary_channel="IVR voice call",
            secondary_channel="SMS",
            social_amplification="Retailer network (in-shop poster + counter talk)",
            format="45-second voice message in local language + retailer follow-up",
            rationale="Feature-phone user. Voice respects low literacy and works on 2G; retailer network reinforces in-person.",
        )
    if offline_attended:
        return ChannelRecommendation(
            primary_channel="Retailer network",
            secondary_channel="SMS",
            social_amplification="Field-day photo shared via WhatsApp groups",
            format="One-page printed leaflet + retailer talking points",
            rationale="Grower already engages offline. Retailer touch is the highest-trust channel.",
        )
    return ChannelRecommendation(
        primary_channel="SMS",
        secondary_channel="Retailer network",
        social_amplification="Local social media in " + language,
        format="Short SMS + retailer follow-up",
        rationale="Unknown device. SMS guarantees delivery; retailer adds context.",
    )


def optimal_send_date(crop_calendar: dict, product: str,
                      as_of: date | None = None) -> date | None:
    if not crop_calendar:
        return None
    as_of = as_of or date(2026, 2, 15)
    target_stages = PRODUCT_STAGE_WINDOW.get(product, [])
    if not target_stages:
        return as_of
    for stage in crop_calendar.get("stages", []):
        if stage.get("stage") in target_stages:
            try:
                approx = date.fromisoformat(stage["approx"])
            except (KeyError, ValueError):
                continue
            ideal = approx - timedelta(days=LEAD_DAYS)
            return ideal if ideal >= as_of else as_of
    return as_of


def tone_for_language(language: str) -> str:
    return TONE_BY_LANGUAGE.get(language, "warm, respectful, agronomically precise")


# ── Receptivity prediction ────────────────────────────────────────────

def segment_click_rate(whatsapp: pd.DataFrame, growers: pd.DataFrame,
                       *, device_type: str | None = None,
                       language: str | None = None,
                       crop: str | None = None) -> dict:
    """Historical click rate for a segment from WhatsApp engagement log.

    The WhatsApp log only covers smartphone users. For keypad / unknown
    growers we benchmark against the smartphone segment that matches the
    farmer's language and crop, with a note explaining the proxy.
    Falls back through ever-broader segments if the narrowest one is
    too small (< 30 delivered messages).
    """
    g = growers[["grower_id", "device_type", "language"]].copy()
    base = whatsapp.merge(g, on="grower_id", how="inner")
    proxy_note = ""

    # WhatsApp log excludes non-smartphone users — proxy via smartphone segment
    effective_device = device_type
    if device_type in (None, "keypad", "unknown"):
        effective_device = "smartphone"
        if device_type and device_type != "smartphone":
            proxy_note = ("WhatsApp log only covers smartphone users; "
                          f"benchmarking via smartphone segment as proxy.")

    levels = [
        {"device": effective_device, "lang": language, "crop": crop},
        {"device": effective_device, "lang": None,     "crop": crop},
        {"device": effective_device, "lang": language, "crop": None},
        {"device": None,             "lang": None,     "crop": crop},
        {"device": None,             "lang": None,     "crop": None},
    ]

    for level in levels:
        df = base
        if level["device"]: df = df[df["device_type"] == level["device"]]
        if level["lang"]:   df = df[df["language"] == level["lang"]]
        if level["crop"]:   df = df[df["campaign_crop"] == level["crop"]]
        delivered = int(df["delivered_status"].sum())
        if delivered >= 30:
            opened = int(df["opened_status"].sum())
            clicked = int(df["clicked_status"].sum())
            level_desc = " · ".join(
                v for v in [level["device"], level["lang"], level["crop"]] if v
            ) or "all segments"
            return {
                "sent": int(len(df)),
                "delivered": delivered,
                "opened": opened,
                "clicked": clicked,
                "open_rate": opened / delivered,
                "click_rate": clicked / delivered,
                "level": level_desc,
                "proxy_note": proxy_note,
            }

    # Last resort
    delivered = int(base["delivered_status"].sum())
    opened = int(base["opened_status"].sum())
    clicked = int(base["clicked_status"].sum())
    return {
        "sent": int(len(base)),
        "delivered": delivered,
        "opened": opened,
        "clicked": clicked,
        "open_rate": opened / delivered if delivered else 0.0,
        "click_rate": clicked / delivered if delivered else 0.0,
        "level": "overall (fallback)",
        "proxy_note": proxy_note,
    }


def predict_best_variant(*, grower: dict, click_score: float) -> tuple[str, str]:
    """Heuristic: which creative variant will this farmer respond to best?

    Trained on the click model's score and known farmer behaviour patterns.
    Returns (variant_name, reason).
    """
    prior_engaged = (grower.get("wa_clicked", 0) or 0) > 0
    smartphone = grower.get("device_type") == "smartphone"
    age = grower.get("grower_age") or 45

    if prior_engaged and click_score > 0.5:
        return ("urgent",
                "Farmer has clicked before and ranks in the top tier. "
                "Urgent, action-first copy converts highest for engaged segments.")
    if not smartphone or age >= 55:
        return ("educational",
                "Older or feature-phone user. Educational tone with agronomic "
                "context builds trust and works on voice formats.")
    return ("social_proof",
            "New or lightly engaged smartphone farmer. Peer-driven social "
            "proof is the highest-converting cold-open format.")
