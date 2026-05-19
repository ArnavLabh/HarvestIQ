"""Single entry point: grower_id → full campaign plan (no content yet).

Content generation is a separate step triggered by an explicit user
action in the UI, so it is not auto-run here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .channel import (
    ChannelRecommendation,
    optimal_send_date,
    recommend_channel,
    tone_for_language,
)
from .content import ContentBrief, ContentResult
from .data import HarvestData
from .product_recommender import ProductRecommendation, recommend
from .weather import WeatherSnapshot, get_weather


@dataclass
class GrowerContext:
    grower: dict[str, Any]
    weather: WeatherSnapshot
    recommendation: ProductRecommendation
    channel: ChannelRecommendation
    send_date: str
    tone: str
    active_threat: str = "none"
    content: ContentResult | None = None

    def to_brief(self, *, variant: str = "default") -> ContentBrief:
        g = self.grower
        return ContentBrief(
            crop=g["crop"], product=self.recommendation.product,
            state=g["state"], district=g["district"],
            language=g["language"], current_stage=g["current_stage"],
            channel=self.channel.primary_channel, format=self.channel.format,
            tone=self.tone, variant=variant,
            active_threat=self.active_threat or "none",
            weather_summary=self.weather.summary,
            farmer_age=int(g["grower_age"]) if g.get("grower_age") else None,
            farm_size_acres=float(g["grower_farm_size"]) if g.get("grower_farm_size") else None,
        )


def build_context(data: HarvestData, grower_row: pd.Series, *,
                  active_threat: str = "none",
                  as_of: date | None = None) -> GrowerContext:
    g = grower_row.to_dict()
    weather = get_weather(g["district"], g["state"])
    rec = recommend(crop=g["crop"], current_stage=g["current_stage"], weather=weather)
    ch = recommend_channel(g["device_type"], g["language"], g["offline_campaign_attended"])
    send = optimal_send_date(g.get("crop_calendar", {}), rec.product, as_of=as_of)
    return GrowerContext(
        grower=g, weather=weather, recommendation=rec, channel=ch,
        send_date=send.isoformat() if send else "",
        tone=tone_for_language(g["language"]),
        active_threat=active_threat,
    )
