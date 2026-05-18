"""Targeting model: predict P(click) for a WhatsApp campaign message.

Trains a logistic regression on the WhatsApp engagement log joined with
grower features. Returns calibrated click probabilities used to rank
growers for a given campaign product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import HarvestData

CATEGORICAL = ["state", "language", "device_type", "gender", "crop", "current_stage"]
NUMERIC = ["grower_age", "grower_farm_size"]


@dataclass
class TargetingModel:
    pipeline: Pipeline
    auc: float
    feature_cols: list[str]

    def score(self, growers: pd.DataFrame) -> pd.Series:
        X = growers[self.feature_cols].copy()
        for col in CATEGORICAL:
            X[col] = X[col].fillna("unknown").astype(str)
        for col in NUMERIC:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        proba = self.pipeline.predict_proba(X)[:, 1]
        return pd.Series(proba, index=growers.index, name="click_score")


def _build_training_frame(data: HarvestData) -> pd.DataFrame:
    """One row per delivered WhatsApp message with grower features + click label."""
    wa = data.whatsapp[data.whatsapp["delivered_status"]].copy()
    g = data.grower_features()
    df = wa.merge(g, on="grower_id", how="inner", suffixes=("", "_g"))
    df["clicked"] = df["clicked_status"].astype(int)
    return df


def train(data: HarvestData) -> TargetingModel:
    df = _build_training_frame(data)
    feature_cols = CATEGORICAL + NUMERIC
    X = df[feature_cols].copy()
    for col in CATEGORICAL:
        X[col] = X[col].fillna("unknown").astype(str)
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    y = df["clicked"].values

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else float("nan")

    pipe.fit(X, y)
    return TargetingModel(pipeline=pipe, auc=auc, feature_cols=feature_cols)


def rank_growers(model: TargetingModel, growers: pd.DataFrame,
                 *, crop: str | None = None, top_k: int | None = None) -> pd.DataFrame:
    pool = growers if crop is None else growers[growers["crop"] == crop]
    scored = pool.assign(click_score=model.score(pool))
    scored = scored.sort_values("click_score", ascending=False)
    if top_k:
        scored = scored.head(top_k)
    return scored
