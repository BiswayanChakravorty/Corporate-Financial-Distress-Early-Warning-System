# Corporate Financial Distress Early-Warning System

Research-grade prototype for identifying early deterioration in corporate financial condition using public financial and market information.

## Implemented
- Point-in-time-aware data-contract design
- Financial ratio and growth feature engineering
- Forward distress-label framework
- Chronological out-of-time split
- Logistic Regression baseline
- Random Forest benchmark
- Histogram Gradient Boosting benchmark
- ROC-AUC, PR-AUC, Brier score and log loss
- Permutation-based driver analysis
- FastAPI service skeleton
- Streamlit research dashboard
- Deterministic synthetic panel for software validation
- Pytest + GitHub Actions CI
- Docker deployment scaffold

## Architecture

    Public financial / market / macro data
                    |
             Raw data snapshots
                    |
               Data contract
                    |
             Quality validation
                    |
           Point-in-time features
                    |
           Forward distress label
                    |
            Chronological split
                    |
          +---------+---------+
          |         |         |
       Logistic   Random     HGB
                  Forest
          |         |         |
          +---------+---------+
                    |
            Probability metrics
                    |
             Driver analysis
                    |
          Company risk workflow
                    |
          Consulting-style memo

## Run locally

    python -m venv .venv
    pip install -r requirements.txt
    pytest -q
    python scripts/train_demo.py
    streamlit run dashboard.py
    uvicorn api.main:app --reload

## Demo vs empirical mode

The repository contains synthetic data only so the software can be exercised without presenting generated observations as real financial evidence.

For an empirical study:
1. Obtain a documented public-company dataset.
2. Persist raw responses with retrieval/publication timestamps.
3. Normalize into the data contract.
4. Construct labels from an independently defined distress-event source.
5. Enforce point-in-time availability.
6. Run chronological and walk-forward evaluation.
7. Report calibration, class prevalence, confidence intervals and robustness checks.

The project intentionally does not ship fabricated empirical accuracy numbers.

## Research methodology

### Prediction target

For company i at prediction time t, the forward target is 1 when the company enters the predefined distress state within the chosen horizon.

### Leakage control

A feature at time t may only use information publicly available at or before t. This is critical for filings, restatements, market data and event labels.

### Feature families

| Family | Examples |
|---|---|
| Liquidity | current ratio |
| Leverage | debt/assets, debt/equity |
| Profitability | gross margin, operating margin, ROA |
| Cash generation | FCF margin |
| Growth | revenue and operating-income YoY |
| Market risk | volatility |

### Evaluation

The current pipeline uses an out-of-time split. Future empirical work should add rolling walk-forward validation, confidence intervals and probability calibration.

## API

`GET /health` provides a health check.

`POST /score` is intentionally disabled until an empirical, versioned model artifact has been trained. This prevents synthetic/demo probabilities from being presented as real-world credit or investment evidence.

## Dashboard

The Streamlit dashboard compares the three baseline models on the synthetic panel and provides company-level screening output. Synthetic probabilities are explicitly labeled as non-empirical.

## Roadmap

### v1 — implemented
- research architecture
- data contract
- feature engine
- supervised baseline suite
- dashboard/API/CI

### v2
- real SEC/public-company ingestion
- point-in-time filing database
- independently sourced event labels
- walk-forward validation
- probability calibration
- confidence intervals

### v3
- SHAP explanations
- Isolation Forest anomaly layer
- survival analysis
- sector and macro controls
- historical peer benchmarking

### v4
- analyst workflow
- automated company risk memo
- monitoring and drift detection
- model registry and reproducible experiments

## Disclaimer

This is a research and educational system. Model outputs should not be treated as investment, lending, credit, or other financial advice.
