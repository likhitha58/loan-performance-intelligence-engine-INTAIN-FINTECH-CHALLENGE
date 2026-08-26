import pandas as pd
import pytest

from src.modeling.transition import (
    STATE_CLASSES,
    evaluate_transition_model,
    fit_transition_model,
    predict_next_state,
)


def _sample_data() -> pd.DataFrame:
    rows = []

    states = [
        "CURRENT",
        "DELINQUENT",
        "DEFAULT",
        "PREPAID",
        "CLOSED",
    ]

    for i in range(50):
        rows.append(
            {
                "loan_id": f"L{i:04d}",
                "reporting_month": pd.Timestamp(
                    "2024-01-01"
                ) + pd.DateOffset(months=i),
                "current_status": states[i % 5],
                "credit_score": 600 + (i % 200),
                "ltv": 50 + (i % 50),
                "dti": 20 + (i % 30),
                "next_state": states[(i + 1) % 5],
            }
        )

    return pd.DataFrame(rows)


def test_fit_transition_model():
    data = _sample_data()

    pipeline, features, metrics = (
        fit_transition_model(data)
    )

    assert pipeline is not None
    assert features
    assert "accuracy" in metrics
    assert "macro_f1" in metrics


def test_predictions_contain_valid_states():
    data = _sample_data()

    pipeline, features, _ = fit_transition_model(
        data
    )

    result = predict_next_state(
        data,
        pipeline,
        features,
    )

    assert "pred_next_state" in result.columns

    assert set(
        result["pred_next_state"]
    ).issubset(set(STATE_CLASSES))


def test_probabilities_sum_to_one():
    data = _sample_data()

    pipeline, features, _ = fit_transition_model(
        data
    )

    result = predict_next_state(
        data,
        pipeline,
        features,
    )

    probability_columns = [
        column
        for column in result.columns
        if column.startswith("prob_next_state_")
    ]

    totals = result[
        probability_columns
    ].sum(axis=1)

    assert totals.round(6).eq(1.0).all()


def test_confidence_is_valid_probability():
    data = _sample_data()

    pipeline, features, _ = fit_transition_model(
        data
    )

    result = predict_next_state(
        data,
        pipeline,
        features,
    )

    assert result[
        "next_state_confidence"
    ].between(0, 1).all()


def test_evaluate_transition_model():
    data = _sample_data()

    pipeline, features, _ = fit_transition_model(
        data
    )

    metrics = evaluate_transition_model(
        data,
        pipeline,
        features,
    )

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_missing_next_state_raises():
    data = _sample_data().drop(
        columns=["next_state"]
    )

    with pytest.raises(ValueError):
        fit_transition_model(data)


def test_unexpected_state_raises():
    data = _sample_data()
    data.loc[0, "next_state"] = "UNKNOWN"

    with pytest.raises(ValueError):
        fit_transition_model(data)