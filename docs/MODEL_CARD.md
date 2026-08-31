
# Model Card — Loan Performance Intelligence Engine

## 1. Model Overview

The Loan Performance Intelligence Engine is an ML-first loan analytics system developed for the Intain Campus FinTech Challenge 2026 — AI Track.

The primary predictive model is a Histogram Gradient Boosting Classifier used to estimate the probability of a loan experiencing default within the next 12 months.

The model is designed to support reviewer risk assessment. It does not replace human judgment or make autonomous lending decisions.

## 2. Primary Prediction Task

Primary target:

`next_12m_default_flag`

The target represents whether a loan experiences a default event within the next 12 months.

The system also evaluates:

- `next_3m_delinquency_flag`
- `next_6m_delinquency_flag`
- `next_12m_default_flag`
- `next_12m_prepayment_flag`

## 3. Data and Validation Strategy

The project uses loan-month panel data.

A chronological split is used to reduce temporal leakage:

- Training: January 2019 — June 2023
- Validation: July 2023 — June 2024

No randomization is used in the temporal split.

The validation period is strictly later than the training period.

## 4. Feature Engineering

The production feature set contains 37 model features.

Feature groups include:

- Balance and repayment features
- Historical delinquency features
- Loan lifecycle features
- Loan status transition features
- Calendar/time features

Examples include:

- `balance_ratio`
- `balance_change_1m`
- `balance_change_pct_1m`
- `balance_reduction`
- `balance_remaining_ratio`
- `dpd_lag_1m`
- `delinquent_flag`
- `delinquent_lag_1m`
- `age_ratio`
- `maturity_proximity`
- `previous_status`
- `status_change_flag`
- Reporting year/month
- Origination year/month

Historical features use information from the same loan and the immediately preceding calendar month when that month is present. Missing calendar months are not treated as valid one-month lags.

## 5. Leakage Prevention

The feature pipeline explicitly excludes:

- Loan identifiers
- Future target columns
- Data-quality ground-truth columns
- `next_state`
- `loss_severity_band`
- Raw date columns used only for feature construction

`next_state` and `loss_severity_band` are excluded because they can contain forward-looking or outcome-derived information.

The chronological train/validation split further reduces the risk of using future observations to predict earlier observations.

## 6. Preprocessing

Numeric features:

- Missing values are replaced using median imputation.

Categorical features:

- Missing values are replaced using most-frequent imputation.
- Categories are converted using OrdinalEncoder.
- Previously unseen categories are encoded as `-1`.

Preprocessing and the model are implemented as a single scikit-learn Pipeline.

## 7. Model Configuration

Model:

`HistGradientBoostingClassifier`

Configuration:

- `max_iter = 200`
- `learning_rate = 0.08`
- `max_leaf_nodes = 31`
- `l2_regularization = 1.0`
- `random_state = 42`

Histogram Gradient Boosting was selected because it improved PR-AUC over the baseline for all four evaluated prediction targets.

## 8. Model Performance

For the primary 12-month default prediction task:

- ROC-AUC: 0.9037
- PR-AUC: 0.5803
- Validation prevalence: 6.39%

Gradient Boosting PR-AUC results:

| Target | Baseline PR-AUC | Gradient Boosting PR-AUC |
|---|---:|---:|
| next_3m_delinquency_flag | 0.3311 | 0.3588 |
| next_6m_delinquency_flag | 0.3104 | 0.3435 |
| next_12m_default_flag | 0.3964 | 0.5803 |
| next_12m_prepayment_flag | 0.1649 | 0.5051 |

The largest PR-AUC improvements are observed for:

- 12-month default: +0.1839
- 12-month prepayment: +0.3402

## 9. Classification Threshold

The model produces a probability score.

Threshold analysis was performed separately from model training.

A threshold of 0.20 was selected as the balanced/F1-oriented operating point.

At threshold 0.20:

- Precision: 58.68%
- Recall: 60.44%
- F1: 59.54%
- Predicted positive rate: 6.59%

At threshold 0.50:

- Precision: 77.40%
- Recall: 41.02%
- F1: 53.62%
- Predicted positive rate: 3.39%

The appropriate threshold ultimately depends on the business cost of missed defaults versus unnecessary interventions.

## 10. Calibration

The project includes probability calibration analysis using:

- Calibration tables
- Brier score
- Expected Calibration Error (ECE)

These metrics are used to assess whether predicted probabilities are consistent with observed event rates.

## 11. Outputs

The predictive layer produces:

- Loan-level default probabilities
- Binary risk predictions based on an operating threshold
- Model evaluation metrics
- Risk evidence used by downstream intelligence components

These outputs feed the risk intelligence, explainability, portfolio intelligence, scenario analysis, reviewer copilot, dashboard, and submission pipeline.

## 12. Explainability and Human Review

The system provides evidence-based explanations linked to observable model inputs and risk signals.

The explainability layer is intended to help reviewers understand why a loan received a particular risk assessment.

The system does not claim that a model feature proves a borrower-specific cause.

## 13. Limitations

Important limitations include:

- Model performance depends on the quality and representativeness of the available loan data.
- Historical patterns may not fully represent future market conditions.
- A high probability is a model risk signal, not proof that a borrower will default.
- Threshold selection involves a business trade-off between precision and recall.
- Anomaly scores indicate unusual observations but do not independently establish the cause of an anomaly.
- Scenario analysis represents simulated conditions rather than guaranteed future outcomes.
- LLM-generated reviewer assistance must remain grounded in model evidence and should be reviewed by a human.

## 14. Intended Use

The model is intended for:

- Loan portfolio risk analysis
- Reviewer prioritization
- Risk monitoring
- Portfolio-level intelligence
- Decision-support workflows
- Explainable risk review

It should not be treated as an autonomous replacement for financial or credit decision-making.

## 15. Reproducibility

The project uses:

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- pytest

The model uses a fixed random seed of 42.

The repository contains the feature engineering, modeling, evaluation, testing, reporting, dashboard, and submission-generation code required to reproduce the implemented workflow.

## 16. Validation Status

The project test suite currently passes:

`161 tests passed`

The generated submission was validated with:

- 17,392 rows
- 12 columns
- 0 null values
- 0 duplicate `(loan_id, reporting_month)` rows

These checks are part of final submission validation.