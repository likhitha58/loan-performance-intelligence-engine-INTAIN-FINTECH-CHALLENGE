# Loan Performance Intelligence Engine

## Intain Campus FinTech Challenge 2026 — AI Track

An ML-first loan analytics and reviewer intelligence system for predicting future loan events, detecting anomalies, explaining model behavior, and supporting portfolio-level risk decisions.

## Key Capabilities

- Data profiling and quality intelligence
- Leakage-safe historical feature engineering
- Time-aware model validation
- Multi-target loan-event prediction
- Histogram Gradient Boosting models
- Probability calibration and threshold optimization
- Global feature importance
- Next-state transition prediction
- Anomaly detection
- Risk evidence and recommended actions
- Macro scenario simulation
- Portfolio and segment-level intelligence
- Streamlit reviewer dashboard
- Automated test coverage

## Prediction Targets

The engine generates probabilities for:

- `next_3m_delinquency_flag`
- `next_6m_delinquency_flag`
- `next_12m_default_flag`
- `next_12m_prepayment_flag`

It also generates:

- Next predicted loan state
- Anomaly score
- Exception type
- Top risk drivers
- Recommended action
- Confidence score

## Project Structure

```text
src/
  data_intelligence/     Data loading, profiling, features and quality
  modeling/              Prediction, anomaly, explainability and risk intelligence
  reporting/             Portfolio reporting

scripts/
  generate_submission.py End-to-end submission generation
  run_pipeline.py        Pipeline execution

app.py                   Streamlit dashboard
tests/                   Automated test suite
submission/              Generated submission artifacts
docs/                    Project plan and development documentation