import pandas as pd
import pytest

from src.submission import (
    SUBMISSION_COLUMNS,
    build_submission,
    write_submission,
)


def _prediction_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
            ],
            "pred_next_3m_delinquency_prob": [
                0.10,
                0.20,
            ],
            "pred_next_6m_delinquency_prob": [
                0.15,
                0.25,
            ],
            "pred_next_12m_default_prob": [
                0.20,
                0.40,
            ],
            "pred_next_12m_prepayment_prob": [
                0.05,
                0.10,
            ],
        }
    )


def _transition_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
            ],
            "pred_next_state": [
                "CURRENT",
                "DELINQUENT",
            ],
            "next_state_confidence": [
                0.90,
                0.75,
            ],
        }
    )


def _anomaly_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
            ],
            "anomaly_score": [
                0.10,
                0.85,
            ],
            "exception_type": [
                "",
                "HIGH_LTV",
            ],
        }
    )


def _explanation_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
            ],
            "reasons": [
                ["HIGH_LTV"],
                ["CURRENT_DELINQUENCY", "HIGH_DTI"],
            ],
            "action_priority": [
                "MONITOR",
                "IMMEDIATE_REVIEW",
            ],
        }
    )


def test_build_submission_has_required_columns():
    result = build_submission(
        _prediction_data(),
        _transition_data(),
        _anomaly_data(),
        _explanation_data(),
    )

    assert list(result.columns) == SUBMISSION_COLUMNS


def test_build_submission_preserves_rows():
    result = build_submission(
        _prediction_data(),
        _transition_data(),
        _anomaly_data(),
        _explanation_data(),
    )

    assert len(result) == 2
    assert result["loan_id"].tolist() == [
        "L1",
        "L2",
    ]


def test_build_submission_contains_transition_output():
    result = build_submission(
        _prediction_data(),
        _transition_data(),
    )

    assert result.loc[0, "pred_next_state"] == "CURRENT"
    assert result.loc[1, "pred_next_state"] == "DELINQUENT"

    assert result.loc[0, "confidence"] == pytest.approx(
        0.90
    )


def test_build_submission_contains_anomaly_output():
    result = build_submission(
        _prediction_data(),
        anomaly_data=_anomaly_data(),
    )

    assert result.loc[1, "anomaly_score"] == pytest.approx(
        0.85
    )
    assert result.loc[1, "exception_type"] == "HIGH_LTV"


def test_build_submission_contains_explanation_output():
    result = build_submission(
        _prediction_data(),
        explanation_data=_explanation_data(),
    )

    assert "HIGH_LTV" in result.loc[0, "top_drivers"]
    assert "HIGH_DTI" in result.loc[1, "top_drivers"]

    assert (
        result.loc[1, "recommended_action"]
        == "IMMEDIATE_REVIEW"
    )


def test_submission_defaults_without_optional_outputs():
    result = build_submission(
        _prediction_data()
    )

    assert result["pred_next_state"].isna().all()
    assert result["anomaly_score"].eq(0.0).all()
    assert result["exception_type"].eq("").all()
    assert result["top_drivers"].eq("").all()


def test_missing_prediction_column_raises():
    data = _prediction_data().drop(
        columns=["pred_next_12m_default_prob"]
    )

    with pytest.raises(ValueError):
        build_submission(data)


def test_write_submission(tmp_path):
    submission = build_submission(
        _prediction_data(),
        _transition_data(),
        _anomaly_data(),
        _explanation_data(),
    )

    path = tmp_path / "submission.csv"

    result_path = write_submission(
        submission,
        path,
    )

    assert result_path.exists()

    saved = pd.read_csv(result_path)

    assert list(saved.columns) == SUBMISSION_COLUMNS
    assert len(saved) == 2