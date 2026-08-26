import pandas as pd
import pytest

from src.modeling.gradient_boosting import fit_gradient_boosting
from src.modeling.prediction import generate_predictions


def test_generate_predictions_returns_expected_columns():
    df = _sample_data()

    pipeline, feature_columns = fit_gradient_boosting(
        df,
        target="next_12m_default_flag",
    )

    result = generate_predictions(
        df,
        pipeline,
        feature_columns,
        threshold=0.20,
    )

    assert list(result.columns) == [
        "loan_id",
        "reporting_month",
        "predicted_probability",
        "risk_flag",
        "threshold",
    ]


def test_prediction_probabilities_are_valid():
    df = _sample_data()

    pipeline, feature_columns = fit_gradient_boosting(
        df,
        target="next_12m_default_flag",
    )

    result = generate_predictions(
        df,
        pipeline,
        feature_columns,
        threshold=0.20,
    )

    assert result["predicted_probability"].between(0, 1).all()
    assert result["risk_flag"].isin([0, 1]).all()
    assert (result["threshold"] == 0.20).all()


def test_threshold_controls_risk_flag():
    df = _sample_data()

    pipeline, feature_columns = fit_gradient_boosting(
        df,
        target="next_12m_default_flag",
    )

    low_threshold = generate_predictions(
        df,
        pipeline,
        feature_columns,
        threshold=0.20,
    )

    high_threshold = generate_predictions(
        df,
        pipeline,
        feature_columns,
        threshold=0.80,
    )

    assert high_threshold["risk_flag"].sum() <= low_threshold["risk_flag"].sum()


def test_invalid_threshold_raises():
    df = _sample_data()

    pipeline, feature_columns = fit_gradient_boosting(
        df,
        target="next_12m_default_flag",
    )

    with pytest.raises(ValueError):
        generate_predictions(
            df,
            pipeline,
            feature_columns,
            threshold=1.0,
        )


def _sample_data() -> pd.DataFrame:
    rows = []

    for loan_id in ["L1", "L2", "L3", "L4"]:
        for month in range(1, 9):
            rows.append(
                {
                    "loan_id": loan_id,
                    "month_index": month,
                    "reporting_month": f"2023-{month:02d}-01",
                    "origination_month": "2023-01-01",
                    "loan_age_months": month,
                    "remaining_term_months": 24 - month,
                    "original_balance": 100000.0,
                    "current_balance": 100000.0 - month * 1000,
                    "interest_rate": 0.05,
                    "credit_score_band": "GOOD",
                    "ltv_band": "LOW",
                    "dti_band": "LOW",
                    "state": "CA",
                    "loan_purpose": "PURCHASE",
                    "occupancy_type": "OWNER",
                    "property_type": "SFR",
                    "servicer_name": "SERVICER_A",
                    "current_status": "CURRENT",
                    "days_past_due": 0,
                    "modification_flag": 0,
                    "prepayment_flag": 0,
                    "default_flag": 0,
                    "loss_severity_band": None,
                    "last_updated_at": f"2023-{month:02d}-15",
                    "source_system": "SYSTEM_A",
                    "document_status": "COMPLETE",
                    "next_3m_delinquency_flag": int(
                        loan_id in ["L3", "L4"] and month in [3, 8]
                    ),
                    "next_6m_delinquency_flag": int(
                        loan_id == "L4" and month in [3, 8]
                    ),
                    "next_12m_default_flag": int(
                        loan_id == "L4" and month in [3, 8]
                    ),
                    "next_12m_prepayment_flag": int(
                        loan_id == "L3" and month in [3, 8]
                    ),
                    "next_state": "CURRENT",
                    "exception_required": 0,
                    "exception_type": "",
                }
            )

    return pd.DataFrame(rows)