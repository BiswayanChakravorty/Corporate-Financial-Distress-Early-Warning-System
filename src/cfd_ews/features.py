from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "current_assets","current_liabilities","total_debt","total_assets",
    "shareholders_equity","revenue","gross_profit","operating_income",
    "net_income","free_cash_flow","market_volatility",
}

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)

def build_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable financial and market features."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["current_ratio"] = _safe_divide(out["current_assets"], out["current_liabilities"])
    out["debt_to_assets"] = _safe_divide(out["total_debt"], out["total_assets"])
    out["debt_to_equity"] = _safe_divide(out["total_debt"], out["shareholders_equity"])
    out["gross_margin"] = _safe_divide(out["gross_profit"], out["revenue"])
    out["operating_margin"] = _safe_divide(out["operating_income"], out["revenue"])
    out["roa"] = _safe_divide(out["net_income"], out["total_assets"])
    out["fcf_margin"] = _safe_divide(out["free_cash_flow"], out["revenue"])
    if "company_id" in out.columns and "period" in out.columns:
        out = out.sort_values(["company_id","period"]).copy()
        grouped = out.groupby("company_id", sort=False)
        out["revenue_yoy"] = grouped["revenue"].pct_change()
        out["operating_income_yoy"] = grouped["operating_income"].pct_change()
    else:
        out["revenue_yoy"] = np.nan
        out["operating_income_yoy"] = np.nan
    return out
