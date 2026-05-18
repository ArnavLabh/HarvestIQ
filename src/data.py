"""Data loading & joining — Track 1 (Marketing) datasets only.

Track 1 data signals (per the brief):
  - growers.csv               → farmer profile, language, device, crop calendar
  - whatsapp_campaign.csv     → historical engagement (delivered/opened/clicked)
  - digital_funnel_weekly.csv → 4 flagship Rabi campaigns (impressions→leads)

All retailer / rep / POS / inventory tables belong to Track 2 and are
intentionally NOT loaded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Syngenta_IITM_Hackathon_2026_dataset (1)"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def _parse_crop_calendar(raw: str) -> dict:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _current_stage(calendar: dict, today: date) -> str:
    if not calendar:
        return "unknown"
    stages = calendar.get("stages", [])
    sowing = calendar.get("sowing", {})
    harvest = calendar.get("harvest", {})

    def to_date(s: str) -> date | None:
        try:
            return date.fromisoformat(s)
        except (TypeError, ValueError):
            return None

    sowing_start = to_date(sowing.get("start", ""))
    harvest_end = to_date(harvest.get("end", ""))

    if sowing_start and today < sowing_start:
        return "pre_sowing"
    if harvest_end and today > harvest_end:
        return "post_harvest"

    current = "sowing"
    for stage in stages:
        approx = to_date(stage.get("approx", ""))
        if approx and approx <= today:
            current = stage.get("stage", current)
    return current


@dataclass
class HarvestData:
    growers: pd.DataFrame
    funnel: pd.DataFrame
    whatsapp: pd.DataFrame

    def grower_features(self, as_of: date | None = None) -> pd.DataFrame:
        as_of = as_of or date(2026, 2, 15)
        df = self.growers.copy()
        df["crop_calendar"] = df["grower_crop_calendar"].apply(_parse_crop_calendar)
        df["crop"] = df["crop_calendar"].apply(lambda c: c.get("crop", "unknown"))
        df["current_stage"] = df["crop_calendar"].apply(lambda c: _current_stage(c, as_of))

        agg = self.whatsapp.groupby("grower_id").agg(
            wa_sent=("id", "count"),
            wa_delivered=("delivered_status", "sum"),
            wa_opened=("opened_status", "sum"),
            wa_clicked=("clicked_status", "sum"),
        )
        df = df.merge(agg, on="grower_id", how="left").fillna({
            "wa_sent": 0, "wa_delivered": 0, "wa_opened": 0, "wa_clicked": 0,
        })
        df["open_rate"] = np.where(df["wa_delivered"] > 0,
                                   df["wa_opened"] / df["wa_delivered"], 0.0)
        df["click_rate"] = np.where(df["wa_opened"] > 0,
                                    df["wa_clicked"] / df["wa_opened"], 0.0)
        df["engaged"] = (df["wa_clicked"] > 0).astype(int)
        return df

    def campaign_funnel_summary(self) -> pd.DataFrame:
        f = self.funnel.copy()
        return f.groupby(["campaign_id", "campaign_crop", "campaign_product"]).agg(
            impressions=("social_post_impression", "sum"),
            visits=("landing_page_visits", "sum"),
            leads=("lead_form_submission", "sum"),
        ).reset_index().assign(
            visit_rate=lambda d: d["visits"] / d["impressions"],
            lead_rate=lambda d: d["leads"] / d["visits"],
        )

    def whatsapp_funnel_by_crop(self) -> pd.DataFrame:
        w = self.whatsapp.copy()
        return w.groupby(["campaign_crop", "campaign_product"]).agg(
            sent=("id", "count"),
            delivered=("delivered_status", "sum"),
            opened=("opened_status", "sum"),
            clicked=("clicked_status", "sum"),
        ).reset_index().assign(
            delivery_rate=lambda d: d["delivered"] / d["sent"],
            open_rate=lambda d: d["opened"] / d["delivered"],
            click_rate=lambda d: d["clicked"] / d["opened"].replace(0, np.nan),
        )


@lru_cache(maxsize=1)
def load_all() -> HarvestData:
    return HarvestData(
        growers=_read("growers.csv"),
        funnel=_read("digital_funnel_weekly.csv"),
        whatsapp=_read("whatsapp_campaign.csv"),
    )
