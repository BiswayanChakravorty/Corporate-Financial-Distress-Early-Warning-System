import numpy as np
import pandas as pd
from cfd_ews.evaluation import binary_metrics
from cfd_ews.features import build_financial_features
from cfd_ews.models import BASELINE_FEATURES, build_logistic_baseline

def sample_data(n=40):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "company_id": np.repeat(["A","B"], n//2),
        "period": list(range(n//2))*2,
        "current_assets": rng.uniform(80,180,n),
        "current_liabilities": rng.uniform(60,150,n),
        "total_debt": rng.uniform(20,160,n),
        "total_assets": rng.uniform(180,500,n),
        "shareholders_equity": rng.uniform(80,300,n),
        "revenue": rng.uniform(120,800,n),
        "gross_profit": rng.uniform(40,350,n),
        "operating_income": rng.uniform(5,180,n),
        "net_income": rng.uniform(-20,120,n),
        "free_cash_flow": rng.uniform(-40,150,n),
        "market_volatility": rng.uniform(.10,.80,n),
    })

def test_feature_engineering():
    df = build_financial_features(sample_data())
    assert set(BASELINE_FEATURES).issubset(df.columns)

def test_logistic_baseline():
    df = build_financial_features(sample_data())
    y = pd.Series(np.tile([0,1],20))
    model = build_logistic_baseline()
    model.fit(df[BASELINE_FEATURES], y)
    p = model.predict_proba(df[BASELINE_FEATURES])[:,1]
    metrics = binary_metrics(y,p)
    assert len(p) == len(df)
    assert 0 <= metrics["brier_score"] <= 1
