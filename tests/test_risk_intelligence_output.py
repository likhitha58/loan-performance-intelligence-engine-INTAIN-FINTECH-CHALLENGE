import pandas as pd
import pytest

from src.modeling.risk_intelligence_output import (
    assign_action_priority,
    assign_risk_tier,
    build_risk_intelligence_output,
)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "LOW"),
        (0.1999, "LOW"),
        (0.20, "MEDIUM"),
        (0.3999, "MEDIUM"),
        (0.40, "HIGH"),
        (0.6999, "HIGH"),
        (0.70, "CRITICAL"),
        (1.0, "CRITICAL"),
    ],
)
def test_assign_risk_tier(probability, expected):
    assert assign_risk_tier(probability) == expected


def test_invalid_probability_raises():
    with pytest.raises(ValueError):
        assign_risk_tier(-0.01)

    with pytest.raises(ValueError):
        assign_risk_tier(1.01)


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("LOW", "ROUTINE"),
        ("MEDIUM", "MONITOR"),
        ("HIGH", "REVIEW"),
        ("CRITICAL", "IMMEDIATE_REVIEW"),
    ],
)
def test_assign_action_priority(tier, expected):
    assert assign_action_priority(tier) == expected


def test_unknown_risk_tier_raises():
    with pytest.raises(ValueError):
        assign_action_priority("UNKNOWN")


def test_build_risk_intelligence_output():
    evidence = pd.DataFrame(
        {
            "loan_id": ["L1", "L2", "L3"],
            "reporting_month": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
            ],
            "predicted_probability": [
                0.10,
                0.35,
                0.85,
            ],
            "risk_flag": [0, 1, 1],
            "threshold": [0.20, 0.20, 0.20],
            "risk_evidence": [
                [],
                ["HIGH_LTV"],
                ["CURRENT_DELINQUENCY"],
            ],
            "evidence_count": [0, 1, 1],
            "evidence_category": [
                "NOT_FLAGGED",
                "OBSERVABLE_RISK_SIGNALS",
                "OBSERVABLE_RISK_SIGNALS",
            ],
        }
    )

    result = build_risk_intelligence_output(evidence)

    assert len(result) == 3

    assert result.loc[0, "risk_tier"] == "LOW"
    assert result.loc[1, "risk_tier"] == "MEDIUM"
    assert result.loc[2, "risk_tier"] == "CRITICAL"

    assert result.loc[0, "action_priority"] == "ROUTINE"
    assert result.loc[1, "action_priority"] == "MONITOR"
    assert result.loc[2, "action_priority"] == "IMMEDIATE_REVIEW"

    assert list(result.columns) == [
        "loan_id",
        "reporting_month",
        "predicted_probability",
        "risk_flag",
        "threshold",
        "risk_tier",
        "action_priority",
        "risk_evidence",
        "evidence_count",
        "evidence_category",
    ]