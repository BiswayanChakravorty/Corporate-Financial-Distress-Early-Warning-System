from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cfd_ews.models import BASELINE_FEATURES, build_model_suite

app = FastAPI(
    title="Corporate Financial Distress EWS",
    version="1.0.0",
    description="Research API for company distress-risk experiments.",
)

class CompanyObservation(BaseModel):
    current_ratio: float | None = None
    debt_to_assets: float | None = None
    debt_to_equity: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    roa: float | None = None
    fcf_margin: float | None = None
    revenue_yoy: float | None = None
    operating_income_yoy: float | None = None
    market_volatility: float | None = None

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/score")
def score(observation: CompanyObservation):
    # A fitted production model should be loaded from a versioned artifact.
    # This endpoint deliberately refuses to imply empirical validity without training data.
    raise HTTPException(
        status_code=503,
        detail="No empirical model artifact is bundled. Run the training pipeline on point-in-time data first."
    )
