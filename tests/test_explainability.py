import pandas as pd
import pytest

from src.modeling.explainability import (
    build_explanation,
    build_explanation_summary,
)


def _sample_row(**overrides) -> pd.Series:
    data = {
        "loan_id": "L1",
        "reporting_month": "2024-01-01",
        "predicted_probability": 0.75,
        "risk_flag": 1,
        "threshold": 0.20,
        "risk_tier": "CRITICAL",
        "action_priority": "IMMEDIATE_REVIEW",
        "risk_evidence": [
            "CURRENT_DELINQUENCY",
            "HIGH_LTV",
            "LOW_CREDIT_SCORE",
        ],
        "evidence_count": 3,
        "evidence_category": "OBSERVABLE_RISK_SIGNALS",
        "current_status": "DELINQUENT",
        "credit_score_band": "<620",
        "ltv_band": ">95",
        "dti_band": "37-43",
        "dpd_lag_1m": 15,
        "status_change_flag": 0,
        "modification_flag": 0,
    }

    data.update(overrides)
    return pd.Series(data)


def test_build_explanation_contains_risk_assessment():
    result = build_explanation(_sample_row())

    assert result["loan_id"] == "L1"
    assert result["risk_tier"] == "CRITICAL"
    assert result["action_priority"] == "IMMEDIATE_REVIEW"
    assert result["predicted_probability"] == pytest.approx(0.75)


def test_build_explanation_contains_evidence():
    result = build_explanation(_sample_row())

    assert len(result["reasons"]) == 3
    assert any("delinquent" in reason.lower() for reason in result["reasons"])
    assert any("ltv" in reason.lower() for reason in result["reasons"])
    assert any("credit" in reason.lower() for reason in result["reasons"])


def test_explanation_is_grounded_in_evidence():
    result = build_explanation(
        _sample_row(
            risk_evidence=["HIGH_LTV"],
            evidence_count=1,
        )
    )

    assert len(result["reasons"]) == 1
    assert "ltv" in result["reasons"][0].lower()


def test_model_only_flag_is_explicit():
    result = build_explanation(
        _sample_row(
            risk_evidence=[],
            evidence_count=0,
            evidence_category="MODEL_SIGNAL_ONLY",
            risk_tier="HIGH",
            action_priority="REVIEW",
        )
    )

    assert result["evidence_category"] == "MODEL_SIGNAL_ONLY"
    assert result["reasons"]
    assert "model" in result["reasons"][0].lower()


def test_missing_required_column_raises():
    row = _sample_row().drop("risk_tier")

    with pytest.raises(ValueError):
        build_explanation(row)


def test_build_explanation_summary():
    data = pd.DataFrame(
        [
            _sample_row(),
            _sample_row(
                loan_id="L2",
                predicted_probability=0.25,
                risk_tier="MEDIUM",
                action_priority="MONITOR",
                risk_evidence=["HIGH_DTI"],
                evidence_count=1,
                evidence_category="OBSERVABLE_RISK_SIGNALS",
            ),
        ]
    )

    result = build_explanation_summary(data)

    assert len(result) == 2
    assert set(result["loan_id"]) == {"L1", "L2"}
    assert "explanation" in result.columns