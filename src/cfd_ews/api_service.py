from __future__ import annotations

from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cfd_ews.models import BASELINE_FEATURES, build_model_suite
from cfd_ews.pipeline import chronological_split, prepare_panel
from cfd_ews.synthetic import make_demo_panel

APP_VERSION = "1.1.0-demo"

app = FastAPI(
    title="Corporate Financial Distress Early-Warning System API",
    version=APP_VERSION,
    description="Functional research API. Current model is deterministic and trained only on synthetic data.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoreRequest(BaseModel):
    model: Literal["logistic", "random_forest", "hist_gradient_boosting"] = "logistic"
    current_ratio: float = Field(1.25, ge=0)
    debt_to_assets: float = Field(0.58, ge=0)
    debt_to_equity: float = Field(1.35, ge=0)
    gross_margin: float = Field(0.30)
    operating_margin: float = Field(0.09)
    roa: float = Field(0.04)
    fcf_margin: float = Field(0.02)
    revenue_yoy: float = Field(0.05)
    operating_income_yoy: float = Field(0.03)
    market_volatility: float = Field(0.25)

class ScoreResponse(BaseModel):
    mode: str
    model: str
    probability: float
    risk_band: str
    feature_snapshot: dict[str, float]
    model_version: str

@lru_cache(maxsize=1)
def trained_models():
    panel = prepare_panel(make_demo_panel())
    train, test = chronological_split(panel)
    models = build_model_suite()
    for model in models.values():
        model.fit(train[BASELINE_FEATURES], train["target"].astype(int))
    return models, len(train), len(test)

def _band(p: float) -> str:
    if p < 0.33:
        return "Lower"
    if p < 0.66:
        return "Intermediate"
    return "Higher"

@app.get("/")
def root():
    return {
        "service": "Corporate Financial Distress Early-Warning System",
        "version": APP_VERSION,
        "mode": "synthetic_demo",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    models, train_rows, test_rows = trained_models()
    return {
        "status": "ok",
        "mode": "synthetic_demo",
        "model_version": APP_VERSION,
        "models": list(models),
        "train_rows": train_rows,
        "test_rows": test_rows,
    }

@app.get("/models")
def models():
    return {
        "mode": "synthetic_demo",
        "models": [
            {"id": "logistic", "label": "Logistic Regression", "role": "interpretable baseline"},
            {"id": "random_forest", "label": "Random Forest", "role": "non-linear benchmark"},
            {"id": "hist_gradient_boosting", "label": "Histogram Gradient Boosting", "role": "boosting benchmark"},
        ],
        "features": BASELINE_FEATURES,
    }

@app.get("/demo/companies")
def demo_companies():
    panel = prepare_panel(make_demo_panel())
    models, _, _ = trained_models()
    latest = panel.sort_values("period").groupby("company_id").tail(1).copy()
    model = models["logistic"]
    latest["probability"] = model.predict_proba(latest[BASELINE_FEATURES])[:, 1]
    rows = []
    for _, r in latest.nlargest(12, "probability").iterrows():
        p = float(r["probability"])
        rows.append({
            "company_id": str(r["company_id"]),
            "period": str(r["period"]),
            "probability": round(p, 4),
            "risk_band": _band(p),
        })
    return {"mode": "synthetic_demo", "companies": rows}

@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    models, _, _ = trained_models()
    values = request.model_dump()
    selected_model = values.pop("model")
    frame = pd.DataFrame([{k: values[k] for k in BASELINE_FEATURES}])
    probability = float(models[selected_model].predict_proba(frame)[0, 1])
    return ScoreResponse(
        mode="synthetic_demo",
        model=selected_model,
        probability=round(probability, 6),
        risk_band=_band(probability),
        feature_snapshot={k: float(values[k]) for k in BASELINE_FEATURES},
        model_version=APP_VERSION,
    )
