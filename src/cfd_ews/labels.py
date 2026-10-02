from __future__ import annotations
import pandas as pd

def forward_distress_label(
    df: pd.DataFrame,
    distress_col: str = "distress_event",
    horizon: int = 1,
    company_col: str = "company_id",
) -> pd.Series:
    """Shift historical distress events backward to create a forward label."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    missing = {distress_col, company_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    ordered = df.sort_values([company_col, "period"]) if "period" in df.columns else df
    label = (
        ordered.groupby(company_col, sort=False)[distress_col]
        .shift(-horizon)
        .astype("Float64")
    )
    return label.reindex(df.index)
