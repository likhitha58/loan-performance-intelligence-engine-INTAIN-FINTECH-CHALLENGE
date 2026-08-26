import pandas as pd
import pytest

from src.modeling.evaluation import (
    DEFAULT_TARGETS,
    evaluate_target,
    evaluate_targets,
)


def _sample_data():
    rows = []

    for loan_id in ["L1", "L2", "L3", "L4"]:
        for month in range(1, 9):
            rows.append(
                {
                    "loan_id": loan_id,
                    "month_index": month,
                    "reporting_month": f"2023-{month:02d}-01",
                    "origination_month": "2022-01-01",
                    "loan_age_months": month + 12,
                    "remaining_term_months": 348 - month,
                    "original_term_months": 360,
                    "original_balance": 100000.0,
                    "current_balance": 100000.0 - month * 1000,
                    "interest_rate": 0.05,
                    "credit_score_band": "HIGH",
                    "ltv_band": "LOW",
                    "dti_band": "LOW",
                    "state": "CA",
                    "loan_purpose": "PURCHASE",
                    "occupancy_type": "PRIMARY",
                    "property_type": "SFR",
                    "servicer_name": "SERVICER_A",
                    "current_status": "CURRENT",
                    "days_past_due": 0,
                    "modification_flag": 0,
                    "prepayment_flag": 0,
                    "default_flag": 0,
                    "loss_severity_band": None,
                    "last_updated_at": f"2023-0{month}-15",
                    "source_system": "CORE",
                    "document_status": "COMPLETE",
                    "next_state": "CURRENT",
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
                }
            )

    return pd.DataFrame(rows)


def test_default_targets_are_defined():
    assert len(DEFAULT_TARGETS) == 4
    assert "next_12m_default_flag" in DEFAULT_TARGETS


def test_evaluate_target_returns_expected_metrics():
    df = _sample_data()

    result = evaluate_target(
        df,
        "next_12m_default_flag",
    )

    assert result["target"] == "next_12m_default_flag"
    assert result["train_rows"] > 0
    assert result["validation_rows"] > 0
    assert 0 <= result["positive_rate"] <= 1
    assert 0 <= result["roc_auc"] <= 1
    assert 0 <= result["pr_auc"] <= 1
    assert 0 <= result["naive_pr_auc"] <= 1
    assert result["pr_auc_lift"] >= 0


def test_evaluate_targets_returns_one_row_per_target():
    df = _sample_data()

    results = evaluate_targets(
        df,
        targets=[
            "next_3m_delinquency_flag",
            "next_12m_default_flag",
        ],
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert set(results["target"]) == {
        "next_3m_delinquency_flag",
        "next_12m_default_flag",
    }


def test_evaluate_targets_defaults_to_four_targets():
    df = _sample_data()

    results = evaluate_targets(df)

    assert len(results) == 4
    assert set(results["target"]) == set(DEFAULT_TARGETS)


def test_pr_auc_lift_is_relative_to_prevalence():
    df = _sample_data()

    result = evaluate_target(
        df,
        "next_12m_default_flag",
    )

    assert result["naive_pr_auc"] == pytest.approx(
        result["positive_rate"]
    )

    assert result["pr_auc_lift"] == pytest.approx(
        result["pr_auc"] / result["positive_rate"]
    )