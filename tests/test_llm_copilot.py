import pandas as pd
import pytest

from src.modeling.llm_copilot import (
    build_copilot_batch,
    build_copilot_record,
    build_deterministic_reviewer_summary,
    build_reviewer_context,
    generate_reviewer_prompt,
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
    }

    data.update(overrides)

    return pd.Series(data)


def test_build_reviewer_context():
    result = build_reviewer_context(
        _sample_row()
    )

    assert result["loan_id"] == "L1"
    assert result["risk_tier"] == "CRITICAL"
    assert result["predicted_probability"] == pytest.approx(
        0.75
    )
    assert len(result["risk_evidence"]) == 3


def test_context_rejects_invalid_probability():
    with pytest.raises(ValueError):
        build_reviewer_context(
            _sample_row(
                predicted_probability=1.5
            )
        )


def test_context_handles_empty_evidence():
    result = build_reviewer_context(
        _sample_row(
            risk_evidence=[]
        )
    )

    assert result["risk_evidence"] == []


def test_prompt_contains_grounding_rules():
    context = build_reviewer_context(
        _sample_row()
    )

    prompt = generate_reviewer_prompt(
        context
    )

    assert "Do not invent" in prompt
    assert "observable evidence" in prompt
    assert "model signal" in prompt


def test_deterministic_summary_is_grounded():
    context = build_reviewer_context(
        _sample_row()
    )

    result = build_deterministic_reviewer_summary(
        context
    )

    assert result["grounding_status"] == "GROUNDED"
    assert result["evidence_summary"]
    assert result["follow_up_questions"]


def test_model_only_signal_is_explicit():
    context = build_reviewer_context(
        _sample_row(
            risk_evidence=[],
            evidence_category="MODEL_SIGNAL_ONLY",
        )
    )

    result = build_deterministic_reviewer_summary(
        context
    )

    assert result["grounding_status"] == (
        "MODEL_SIGNAL_ONLY"
    )

    assert any(
        "model signal" in item.lower()
        for item in result["evidence_summary"]
    )


def test_build_copilot_record():
    result = build_copilot_record(
        _sample_row()
    )

    assert "context" in result
    assert "prompt" in result
    assert "fallback" in result


def test_batch_generation():
    data = pd.DataFrame(
        [
            _sample_row(),
            _sample_row(
                loan_id="L2",
                predicted_probability=0.30,
                risk_tier="MEDIUM",
                action_priority="MONITOR",
                risk_evidence=["HIGH_DTI"],
                evidence_count=1,
            ),
        ]
    )

    result = build_copilot_batch(data)

    assert len(result) == 2
    assert set(result["loan_id"]) == {
        "L1",
        "L2",
    }

    assert "copilot_prompt" in result.columns
    assert "grounding_status" in result.columns


def test_empty_batch():
    data = pd.DataFrame(
        columns=[
            "loan_id",
            "reporting_month",
            "predicted_probability",
            "risk_tier",
            "action_priority",
            "risk_evidence",
            "evidence_category",
        ]
    )

    result = build_copilot_batch(data)

    assert result.empty