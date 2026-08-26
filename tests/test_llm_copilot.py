import pandas as pd
import pytest

from src.modeling.llm_copilot import (
    build_reviewer_context,
    generate_reviewer_prompt,
)


def _sample_row() -> pd.Series:
    return pd.Series(
        {
            "loan_id": "L1",
            "reporting_month": "2024-01-01",
            "predicted_probability": 0.75,
            "risk_tier": "CRITICAL",
            "action_priority": "IMMEDIATE_REVIEW",
            "risk_evidence": [
                "CURRENT_DELINQUENCY",
                "HIGH_LTV",
            ],
            "evidence_category": "OBSERVABLE_RISK_SIGNALS",
        }
    )


def test_build_reviewer_context():
    result = build_reviewer_context(_sample_row())

    assert result["loan_id"] == "L1"
    assert result["risk_tier"] == "CRITICAL"
    assert result["action_priority"] == "IMMEDIATE_REVIEW"
    assert result["predicted_probability"] == pytest.approx(0.75)
    assert result["risk_evidence"] == [
        "CURRENT_DELINQUENCY",
        "HIGH_LTV",
    ]


def test_reviewer_prompt_contains_grounded_context():
    context = build_reviewer_context(_sample_row())
    prompt = generate_reviewer_prompt(context)

    assert "L1" in prompt
    assert "CRITICAL" in prompt
    assert "IMMEDIATE_REVIEW" in prompt
    assert "CURRENT_DELINQUENCY" in prompt
    assert "HIGH_LTV" in prompt


def test_prompt_requires_grounded_response():
    context = build_reviewer_context(_sample_row())
    prompt = generate_reviewer_prompt(context)

    assert "Do not invent" in prompt
    assert "Do not override" in prompt


def test_missing_required_column_raises():
    row = _sample_row().drop("risk_tier")

    with pytest.raises(ValueError):
        build_reviewer_context(row)