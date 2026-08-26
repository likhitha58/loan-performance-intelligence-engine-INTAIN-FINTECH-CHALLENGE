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