import numpy as np
import pandas as pd
import pytest

from src.modeling.scenario_simulation import (
    run_scenario,
    summarize_scenario,
)


class FakePipeline:

    def predict_proba(self, X):
        result = np.full(
            len(X),
            0.25,
        )

        return np.column_stack(
            [1 - result, result]
        )


def _sample_data():
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
            ],
            "ltv_band": [
                "71-80",
                "61-70",
            ],
            "dti_band": [
                "37-43",
                "21-30",
            ],
            "credit_score_band": [
                "700-739",
                "780+",
            ],
        }
    )


def test_invalid_scenario_raises():
    with pytest.raises(ValueError):
        run_scenario(
            _sample_data(),
            FakePipeline(),
            [],
            "UNKNOWN",
        )


def test_scenario_summary():
    data = pd.DataFrame(
        {
            "scenario": [
                "TEST",
                "TEST",
            ],
            "baseline_probability": [
                0.10,
                0.20,
            ],
            "scenario_probability": [
                0.20,
                0.40,
            ],
            "baseline_risk_flag": [
                0,
                1,
            ],
            "scenario_risk_flag": [
                1,
                1,
            ],
        }
    )

    result = summarize_scenario(data)

    assert result["observations"] == 2
    assert result[
        "baseline_average_probability"
    ] == pytest.approx(0.15)

    assert result[
        "scenario_average_probability"
    ] == pytest.approx(0.30)

    assert result[
        "average_probability_change"
    ] == pytest.approx(0.15)

    assert result[
        "baseline_flagged_observations"
    ] == 1

    assert result[
        "scenario_flagged_observations"
    ] == 2

    assert result[
        "flagged_observation_change"
    ] == 1


def _sample_predictions():
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
                0.05,
                0.10,
            ],
            "pred_next_12m_prepayment_prob": [
                0.20,
                0.30,
            ],
            "credit_score_band": [
                "620-659",
                "700-739",
            ],
            "servicer": [
                "A",
                "B",
            ],
        }
    )


def test_macro_scenario_names():
    from src.modeling.scenario_simulation import SCENARIOS

    assert "BASE" in SCENARIOS
    assert "ADVERSE_CREDIT" in SCENARIOS
    assert "HIGH_PREPAYMENT" in SCENARIOS


def test_adverse_credit_increases_risk():
    from src.modeling.scenario_simulation import run_macro_scenario

    result = run_macro_scenario(
        _sample_predictions(),
        "ADVERSE_CREDIT",
    )

    assert (
        result["scenario_default_probability"]
        > result["baseline_default_probability"]
    ).all()

    assert (
        result["scenario_delinquency_probability"]
        > result["baseline_delinquency_probability"]
    ).all()

    assert (
        result["scenario_prepayment_probability"]
        < result["baseline_prepayment_probability"]
    ).all()


def test_high_prepayment_increases_prepayment():
    from src.modeling.scenario_simulation import run_macro_scenario

    result = run_macro_scenario(
        _sample_predictions(),
        "HIGH_PREPAYMENT",
    )

    assert (
        result["scenario_prepayment_probability"]
        > result["baseline_prepayment_probability"]
    ).all()


def test_macro_summary():
    from src.modeling.scenario_simulation import (
        run_macro_scenario,
        summarize_macro_scenario,
    )

    output = run_macro_scenario(
        _sample_predictions(),
        "ADVERSE_CREDIT",
    )

    summary = summarize_macro_scenario(output)

    assert summary["observations"] == 2
    assert summary["scenario"] == "ADVERSE_CREDIT"
    assert summary["default_change"] > 0
    assert summary["delinquency_change"] > 0
    assert summary["prepayment_change"] < 0


def test_segment_impacts():
    from src.modeling.scenario_simulation import (
        run_macro_scenario,
        summarize_segment_impacts,
    )

    output = run_macro_scenario(
        _sample_predictions(),
        "ADVERSE_CREDIT",
        segment_columns=["credit_score_band"],
    )

    result = summarize_segment_impacts(
        output,
        "credit_score_band",
    )

    assert len(result) == 2
    assert "default_change" in result.columns
    assert "delinquency_change" in result.columns
    assert "prepayment_change" in result.columns