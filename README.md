# Corporate Financial Distress Early-Warning System

A research-oriented system for detecting early signs of corporate financial distress from public financial and market data.

## Architecture
```
Public Data Sources
  ├─ Financial statements
  ├─ Market data
  ├─ Macro indicators
  └─ Company metadata
          ↓
      Data contracts
          ↓
      Quality checks
          ↓
    Feature engineering
          ↓
     Time-aware dataset
          ↓
  Logistic baseline model
          ↓
 Calibration + metrics
          ↓
 Explainability / risk drivers
          ↓
 Consulting-style diagnosis
```

## Initial baseline
We begin with interpretable **logistic regression** before adding tree ensembles, anomaly detection, survival analysis, and NLP.

Features:
- current ratio
- debt-to-assets
- debt-to-equity
- gross margin
- operating margin
- return on assets
- free-cash-flow margin
- revenue growth
- operating-income growth
- market volatility

## Research discipline
Financial ML will use chronological validation and point-in-time feature construction to reduce look-ahead bias. The production label must be defined from authoritative historical distress events and a fixed prediction horizon.

## Repository
```
data/{raw,interim,processed}/
notebooks/
src/cfd_ews/
tests/
docs/
```

This is a research/educational system, not investment, credit, or lending advice.
