# Corporate Financial Distress Early-Warning System

Research-grade prototype for identifying early deterioration in corporate financial condition using financial and market information.

## Live research interface

The GitHub Pages site is the static research cockpit:
https://biswayanchakravorty.github.io/Corporate-Financial-Distress-Early-Warning-System/

It contains the research overview, pipeline, risk screener, cohort monitor, methodology controls and API status.

## Architecture

**GitHub Pages → static frontend → REST → FastAPI backend → model service**

GitHub Pages cannot execute Python. The repository therefore keeps the browser application and Python model service separate.

### Frontend
- `docs/index.html` — application shell
- `docs/styles.css` — responsive research UI
- `docs/app.js` — API client, screener and cohort visualization
- `docs/config.js` — backend URL configuration

### Backend
- `api/index.py` — Vercel/serverless FastAPI entrypoint
- `api/main.py` — local FastAPI entrypoint
- `src/cfd_ews/api_service.py` — REST service
- `vercel.json` — serverless configuration

API routes:
- `GET /health`
- `GET /models`
- `GET /demo/companies`
- `POST /score`

The current backend trains the existing model suite on deterministic synthetic data at runtime. It is functional for software validation but **not empirical evidence about real companies**.

## Local development

1. Create a virtual environment.
2. Install `requirements.txt`.
3. Run `pytest -q`.
4. Run `uvicorn api.main:app --reload`.
5. Open the FastAPI documentation at `http://127.0.0.1:8000/docs`.

To connect the Pages frontend to a deployed backend, edit `docs/config.js` and set `API_BASE_URL` to the backend origin. Without an API URL, the frontend remains usable through its explicitly labelled browser-only synthetic fallback.

## Modeling stack

- Point-in-time-aware data contract
- Financial ratio and growth feature engineering
- Forward distress-label framework
- Chronological out-of-time split
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting
- ROC-AUC, PR-AUC, Brier score and log loss
- Permutation-based driver analysis

## Research integrity

The project does not fabricate empirical accuracy, company distress probabilities or historical performance.

For an empirical release:
1. Acquire a documented public-company dataset.
2. Persist raw observations with retrieval and publication timestamps.
3. Normalize them into the data contract.
4. Independently define and freeze distress events.
5. Enforce point-in-time feature availability.
6. Run chronological and rolling walk-forward evaluation.
7. Calibrate probabilities and report uncertainty.
8. Add robustness, sector and macro controls.

## Roadmap

**v1 — implemented:** research architecture, model suite, synthetic validation, API and Pages cockpit.

**v2:** real public-company ingestion, point-in-time filing store, independently sourced distress events, walk-forward validation and calibration.

**v3:** SHAP explanations, anomaly detection, survival analysis, sector/macro controls and historical peer benchmarking.

**v4:** analyst workflow, risk memos, monitoring, drift detection and model registry.

## Disclaimer

This is research and educational software. Current model outputs use synthetic training data and must not be interpreted as investment, lending, credit or financial advice.
