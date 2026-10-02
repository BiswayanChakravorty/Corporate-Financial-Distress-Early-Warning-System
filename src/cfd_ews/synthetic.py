from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_panel(
    companies: int = 80,
    periods: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic synthetic panel for end-to-end demos.

    This data is intentionally synthetic and must never be presented as
    empirical evidence about real companies.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(companies):
        debt_pressure = rng.normal(0, 0.15)
        quality = rng.normal(0, 0.5)
        for t in range(periods):
            revenue = max(50, 300 * np.exp(0.025 * t + quality * 0.05 + rng.normal(0, 0.08)))
            assets = max(100, revenue * rng.uniform(0.7, 1.5))
            debt = max(5, assets * (0.25 + debt_pressure + rng.normal(0, 0.04)))
            equity = max(10, assets - debt - rng.uniform(0, 30))
            gross_margin = np.clip(0.35 + quality * 0.03 + rng.normal(0, 0.03), 0.05, 0.65)
            op_margin = np.clip(gross_margin - 0.12 + rng.normal(0, 0.03), -0.2, 0.4)
            gross_profit = revenue * gross_margin
            operating_income = revenue * op_margin
            net_income = operating_income - debt * 0.035 + rng.normal(0, 4)
            fcf = net_income + rng.normal(0, 15)
            current_assets = revenue * rng.uniform(0.25, 0.65)
            current_liabilities = revenue * rng.uniform(0.20, 0.60)
            volatility = np.clip(0.25 + debt_pressure * 0.4 + rng.normal(0, 0.07), 0.05, 1.2)
            distress_score = (
                1.7 * max(0, debt / assets - 0.65)
                + 1.3 * max(0, 1.0 - current_assets / current_liabilities)
                + 1.4 * max(0, -op_margin)
                + 0.7 * volatility
                - 0.5 * max(0, fcf / revenue)
            )
            probability = 1 / (1 + np.exp(-(distress_score - 0.65) * 4))
            event = int(rng.random() < probability * 0.10)
            rows.append({
                "company_id": f"C{c:04d}",
                "period": t,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "total_debt": debt,
                "total_assets": assets,
                "shareholders_equity": equity,
                "revenue": revenue,
                "gross_profit": gross_profit,
                "operating_income": operating_income,
                "net_income": net_income,
                "free_cash_flow": fcf,
                "market_volatility": volatility,
                "distress_event": event,
            })
    return pd.DataFrame(rows)
