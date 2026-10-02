# Data contract

The modeling table uses one row per company × reporting period.

## Required raw fields
company_id, period, current_assets, current_liabilities, total_debt, total_assets, shareholders_equity, revenue, gross_profit, operating_income, net_income, free_cash_flow, market_volatility, distress_event.

## Point-in-time requirement
Every feature must be timestamped to the earliest time it became publicly available. A statement with a fiscal period ending June is not automatically usable on June 30; the filing/publication date controls availability.

## Distress event
The event taxonomy should be documented and frozen before evaluation. Candidate events include bankruptcy, default, restructuring, or another externally verifiable distress state.

## Missingness
Missing financial line items remain missing until a documented imputation rule is applied. The pipeline must never silently replace unavailable information with zero.
