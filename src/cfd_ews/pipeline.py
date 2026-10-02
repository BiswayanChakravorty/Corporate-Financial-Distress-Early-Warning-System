from __future__ import annotations

import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from .features import build_financial_features
from .models import BASELINE_FEATURES, build_model_suite


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = build_financial_features(df)
    if "distress_event" in out:
        out["target"] = (
            out.groupby("company_id", sort=False)["distress_event"]
            .shift(-1)
            .astype("Float64")
        )
    return out.dropna(subset=["target"]).copy()


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    periods = sorted(df["period"].unique())
    cutoff = periods[max(1, int(len(periods) * (1 - test_fraction))) - 1]
    train = df[df["period"] <= cutoff].copy()
    test = df[df["period"] > cutoff].copy()
    return train, test


def fit_suite(df: pd.DataFrame) -> dict:
    data = prepare_panel(df)
    train, test = chronological_split(data)
    X_train, y_train = train[BASELINE_FEATURES], train["target"].astype(int)
    X_test, y_test = test[BASELINE_FEATURES], test["target"].astype(int)
    results = {}
    for name, model in build_model_suite().items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "brier_score": float(brier_score_loss(y_test, probability)),
            "test_rows": len(test),
        }
    return results
