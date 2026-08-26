import numpy as np
import pandas as pd
import pytest

from src.modeling.baseline import (
    build_baseline_pipeline,
    evaluate_binary_predictions,
    prepare_model_data,
)


def _sample_data():
    return pd.DataFrame(
        {
            "loan_id": [
                "L1",
                "L1",
                "L2",
                "L2",
            ],
            "month_index": [
                1,
                2,
                1,
                2,
            ],
            "reporting_month": [
                "2023-01-01",
                "2023-02-01",
                "2023-01-01",
                "2023-02-01",
            ],
            "origination_month": [
                "2022-01-01",
                "2022-01-01",
                "2022-06-01",
                "2022-06-01",
            ],
            "loan_age_months": [
                13,
                14,
                8,
                9,
            ],
            "remaining_term_months": [
                347,
                346,
                351,
                350,
            ],
            "original_term_months": [
                360,
                360,
                360,
                360,
            ],
            "original_balance": [
                110000.0,
                110000.0,
                90000.0,
                90000.0,
            ],
            "current_balance": [
                100000.0,
                98000.0,
                80000.0,
                78000.0,
            ],
            "interest_rate": [
                0.05,
                0.05,
                0.055,
                0.055,
            ],
            "credit_score_band": [
                "HIGH",
                "HIGH",
                "MEDIUM",
                "MEDIUM",
            ],
            "ltv_band": [
                "LOW",
                "LOW",
                "MEDIUM",
                "MEDIUM",
            ],
            "dti_band": [
                "LOW",
                "LOW",
                "MEDIUM",
                "MEDIUM",
            ],
            "state": [
                "CA",
                "CA",
                "TX",
                "TX",
            ],
            "loan_purpose": [
                "PURCHASE",
                "PURCHASE",
                "REFINANCE",
                "REFINANCE",
            ],
            "occupancy_type": [
                "PRIMARY",
                "PRIMARY",
                "PRIMARY",
                "PRIMARY",
            ],
            "property_type": [
                "SFR",
                "SFR",
                "SFR",
                "SFR",
            ],
            "servicer_name": [
                "SERVICER_A",
                "SERVICER_A",
                "SERVICER_B",
                "SERVICER_B",
            ],
            "current_status": [
                "CURRENT",
                "CURRENT",
                "DELINQUENT",
                "DELINQUENT",
            ],
            "days_past_due": [
                0,
                0,
                30,
                45,
            ],
            "modification_flag": [
                0,
                0,
                0,
                0,
            ],
            "prepayment_flag": [
                0,
                0,
                0,
                0,
            ],
            "default_flag": [
                0,
                0,
                0,
                0,
            ],
            "loss_severity_band": [
                None,
                None,
                None,
                None,
            ],
            "last_updated_at": [
                "2023-01-15",
                "2023-02-15",
                "2023-01-15",
                "2023-02-15",
            ],
            "source_system": [
                "CORE",
                "CORE",
                "CORE",
                "CORE",
            ],
            "document_status": [
                "COMPLETE",
                "COMPLETE",
                "COMPLETE",
                "COMPLETE",
            ],
            "next_state": [
                "CURRENT",
                "CURRENT",
                "DELINQUENT",
                "CURRENT",
            ],
            "next_3m_delinquency_flag": [
                0,
                0,
                1,
                1,
            ],
            "next_6m_delinquency_flag": [
                0,
                1,
                1,
                1,
            ],
            "next_12m_default_flag": [
                0,
                0,
                1,
                0,
            ],
            "next_12m_prepayment_flag": [
                0,
                1,
                0,
                0,
            ],
        }
    )


def test_prepare_model_data_excludes_target():
    df = _sample_data()

    X, y = prepare_model_data(
        df,
        "next_12m_default_flag",
    )

    assert "next_12m_default_flag" not in X.columns
    assert len(X) == len(y)


def test_prepare_model_data_excludes_future_state():
    df = _sample_data()

    X, _ = prepare_model_data(
        df,
        "next_12m_default_flag",
    )

    assert "next_state" not in X.columns


def test_baseline_pipeline_can_fit():
    df = _sample_data()

    X, y = prepare_model_data(
        df,
        "next_12m_default_flag",
    )

    pipeline = build_baseline_pipeline(X)

    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(
        probabilities >= 0
    )
    assert np.all(
        probabilities <= 1
    )


def test_evaluate_binary_predictions_returns_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([
        0.1,
        0.2,
        0.8,
        0.9,
    ])
    
    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )
    metrics = evaluate_binary_predictions(
        y_true,
        probabilities,
    )

    assert set(metrics) == {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
    }

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_evaluate_predictions_validates_probability_length():
    y_true = pd.Series([0, 1, 1])

    with pytest.raises(ValueError):
        evaluate_binary_predictions(
            y_true,
            np.array([0.2, 0.8]),
        )


def test_invalid_target_raises():
    df = _sample_data()

    with pytest.raises(ValueError, match="Target column"):
        prepare_model_data(
            df,
            "does_not_exist",
        )