# Methodology — v0.1

## Research question
Can public financial and market information identify companies whose financial condition is deteriorating early enough to support intervention or deeper diligence?

## Unit of analysis
One company × one reporting period. Every feature at time t must be computable using information available by the prediction timestamp.

## Leakage controls
1. Features use only contemporaneously available information.
2. Labels refer to a future period.
3. Train/validation/test splits respect chronology.

Point-in-time correctness is mandatory once live/public-company feeds are connected.

## Baseline
Logistic regression is the reference model because it provides a transparent log-odds mapping and probabilistic output.

## Planned progression
Logistic Regression → Tree Ensemble → Calibrated Ensemble → Isolation Forest → Survival Model → Hybrid risk engine.

## Decision output
signal → supporting financial evidence → model contribution → historical context → analyst investigation path
