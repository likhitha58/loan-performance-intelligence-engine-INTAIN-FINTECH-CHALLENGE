# Loan Performance Intelligence Engine

## Intain Campus FinTech Challenge 2026 — AI Track

An ML-first loan analytics and reviewer intelligence system developed for the Intain Campus FinTech Challenge 2026.

The system combines data intelligence, leakage-safe machine learning, risk evidence, portfolio analytics, and reproducible reporting into an end-to-end loan performance intelligence pipeline.

---

## Problem

The challenge asks participants to build a serious AI engine for loan-level data covering:

- Data profiling
- Feature engineering
- Supervised prediction
- Time-aware validation
- Anomaly detection
- Explainability
- Model calibration
- Scenario simulation
- LLM-assisted reviewer explanations
- Agentic coding evidence

The predictive system uses data science and machine learning. The intelligence layer is designed to provide transparent, grounded reviewer assistance rather than replacing predictive models.

---

## Current Implementation

The project currently implements the core data intelligence, predictive modeling, risk intelligence, portfolio analytics, and reporting pipeline.

### Data Intelligence

Implemented capabilities include:

- Data pack ingestion
- Automated dataset profiling
- Missingness intelligence
- Statistical outlier analysis
- Loan relationship validation
- Record quality scoring
- Synthetic anomaly detection
- Leakage-safe historical feature engineering

### Predictive Modeling

The modeling pipeline includes:

- Chronological train/validation splitting
- Leakage-safe feature selection
- Baseline target evaluation
- Multi-target evaluation
- Histogram Gradient Boosting
- ROC-AUC and PR-AUC evaluation
- Threshold optimization
- Prediction probability generation

The current default prediction target is:

```text
next_12m_default_flag