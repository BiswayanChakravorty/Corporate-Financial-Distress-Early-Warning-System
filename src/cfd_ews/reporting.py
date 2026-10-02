from __future__ import annotations
import pandas as pd
from sklearn.inspection import permutation_importance
from .models import BASELINE_FEATURES

def risk_band(probability: float) -> str:
    if probability < 0.33: return "Lower"
    if probability < 0.66: return "Intermediate"
    return "Higher"

def permutation_drivers(model, X: pd.DataFrame, y: pd.Series, repeats: int = 8) -> pd.DataFrame:
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=repeats, random_state=42)
    return pd.DataFrame({
        "feature": BASELINE_FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

def company_risk_report(model, row: pd.Series) -> dict:
    frame = pd.DataFrame([row[BASELINE_FEATURES].to_dict()])
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "risk_probability": probability,
        "risk_band": risk_band(probability),
        "observed_features": {
            k: (None if pd.isna(row[k]) else float(row[k])) for k in BASELINE_FEATURES
        },
    }
