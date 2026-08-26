# Model Comparison Report

## Evaluation Setup

Models were evaluated using a chronological train/validation split.

- Training period: 2019-01 through 2023-06
- Validation period: 2023-07 through 2024-06
- Evaluation metric: PR-AUC
- Primary reason for PR-AUC: the prediction targets are imbalanced binary outcomes.

The production feature set contains 37 features after excluding:

- `next_state`
- `loss_severity_band`

These fields were excluded because they present forward-looking or outcome-derived leakage risk.

## Model Comparison

| Target | Baseline PR-AUC | Gradient Boosting PR-AUC | Improvement |
|---|---:|---:|---:|
| next_3m_delinquency_flag | 0.3311 | 0.3588 | +0.0277 |
| next_6m_delinquency_flag | 0.3104 | 0.3435 | +0.0331 |
| next_12m_default_flag | 0.3964 | 0.5803 | +0.1839 |
| next_12m_prepayment_flag | 0.1649 | 0.5051 | +0.3402 |

## Model Selection

Gradient Boosting was selected as the primary predictive model because it improves PR-AUC over the baseline for all four prediction targets.

The strongest improvements occur for:

- 12-month default: +0.1839 PR-AUC
- 12-month prepayment: +0.3402 PR-AUC

## 12-Month Default Model

The leakage-safe gradient boosting model achieved:

- ROC-AUC: 0.9037
- PR-AUC: 0.5803
- Validation prevalence: 6.39%

## Threshold Analysis

Threshold optimization was performed on the 12-month default model.

### F1-oriented threshold

At a threshold of 0.20:

- Precision: 58.68%
- Recall: 60.44%
- F1: 59.54%
- Predicted positive rate: 6.59%

### Precision-oriented threshold

At a threshold of 0.50:

- Precision: 77.40%
- Recall: 41.02%
- F1: 53.62%
- Predicted positive rate: 3.39%

Therefore, 0.20 is treated as the balanced/F1-oriented operating point, while 0.50 represents a more precision-oriented operating point.

The threshold should ultimately be selected according to the business cost of missed defaults versus unnecessary interventions rather than purely maximizing F1.
