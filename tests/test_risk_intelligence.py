import pandas as pd
import pytest

from src.modeling.risk_intelligence import (
    add_risk_evidence,
    generate_risk_evidence,
)


def test_generate_risk_evidence_identifies_direct_risk():
    row = pd.Series(
        {
            "current_status": "DELINQUENT",
            "dpd_lag_1m": 30,
            "status_change_flag": 1,
            "ltv_band": "91-95",
            "dti_band": ">50",
            "credit_score_band": "<620",
            "modification_flag": 1,
        }
    )

    evidence = generate_risk_evidence(row)

    assert evidence == [
        "CURRENT_DELINQUENCY",
        "RECENT_DELINQUENCY",
        "STATUS_CHANGE",
        "HIGH_LTV",
        "HIGH_DTI",
        "LOW_CREDIT_SCORE",
        "MODIFIED_LOAN",
    ]


def test_current_loan_can_have_contextual_evidence():
    row = pd.Series(
        {
            "current_status": "CURRENT",
            "dpd_lag_1m": 0,
            "status_change_flag": 0,
            "ltv_band": "81-90",
            "dti_band": "44-50",
            "credit_score_band": "620-659",
            "modification_flag": 0,
        }
    )

    evidence = generate_risk_evidence(row)

    assert evidence == [
        "HIGH_LTV",
        "HIGH_DTI",
        "LOW_CREDIT_SCORE",
    ]


def test_add_risk_evidence_preserves_prediction_rows():
    predictions = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": ["2024-01-01", "2024-01-01"],
            "predicted_probability": [0.8, 0.3],
            "risk_flag": [1, 1],
            "threshold": [0.2, 0.2],
        }
    )

    features = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": ["2024-01-01", "2024-01-01"],
            "current_status": ["DEFAULT", "CURRENT"],
            "dpd_lag_1m": [30, 0],
            "status_change_flag": [1, 0],
            "ltv_band": ["91-95", "61-70"],
            "dti_band": [">50", "21-30"],
            "credit_score_band": ["<620", "740-779"],
            "modification_flag": [1, 0],
        }
    )

    result = add_risk_evidence(
        predictions,
        features,
    )

    assert len(result) == 2
    assert result.loc[0, "risk_evidence"] == [
        "CURRENT_DEFAULT",
        "RECENT_DELINQUENCY",
        "STATUS_CHANGE",
        "HIGH_LTV",
        "HIGH_DTI",
        "LOW_CREDIT_SCORE",
        "MODIFIED_LOAN",
    ]
    assert result.loc[1, "risk_evidence"] == []


def test_missing_keys_raise():
    predictions = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "predicted_probability": [0.8],
        }
    )

    features = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "reporting_month": ["2024-01-01"],
        }
    )

    with pytest.raises(ValueError):
        add_risk_evidence(predictions, features)
        
def test_evidence_category_distinguishes_model_only_flags():
    predictions = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": ["2024-01-01", "2024-01-01"],
            "predicted_probability": [0.8, 0.1],
            "risk_flag": [1, 0],
            "threshold": [0.2, 0.2],
        }
    )

    features = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": ["2024-01-01", "2024-01-01"],
            "current_status": ["CURRENT", "CURRENT"],
            "dpd_lag_1m": [0, 0],
            "status_change_flag": [0, 0],
            "ltv_band": ["61-70", "61-70"],
            "dti_band": ["21-30", "21-30"],
            "credit_score_band": ["740-779", "740-779"],
            "modification_flag": [0, 0],
        }
    )

    result = add_risk_evidence(
        predictions,
        features,
    )

    assert result.loc[0, "evidence_category"] == "MODEL_SIGNAL_ONLY"
    assert result.loc[1, "evidence_category"] == "NOT_FLAGGED"