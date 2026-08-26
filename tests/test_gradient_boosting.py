import pandas as pd
import pytest

from src.modeling.gradient_boosting import run_gradient_boosting


def test_gradient_boosting_returns_metrics():
    df = _sample_data()

    result = run_gradient_boosting(
        df,
        target="next_12m_default_flag",
    )

    assert result.target == "next_12m_default_flag"
    assert result.train_rows > 0
    assert result.validation_rows > 0
    assert 0 <= result.positive_rate <= 1
    assert 0 <= result.roc_auc <= 1
    assert 0 <= result.pr_auc <= 1
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1
    assert 0 <= result.f1 <= 1


def test_gradient_boosting_rejects_invalid_target():
    df = _sample_data()

    with pytest.raises(ValueError):
        run_gradient_boosting(
            df,
            target="not_a_target",
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