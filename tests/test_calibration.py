import numpy as np
import pandas as pd
import pytest

from src.modeling.calibration import (
    build_calibration_table,
    calculate_brier_score,
    calculate_expected_calibration_error,
)


def test_build_calibration_table():
    y_true = pd.Series(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    result = build_calibration_table(
        y_true,
        probabilities,
        bins=2,
    )

    assert len(result) == 2
    assert result["observations"].sum() == 4
    assert result["average_predicted_probability"].tolist() == [
        pytest.approx(0.15),
        pytest.approx(0.85),
    ]
    assert result["observed_positive_rate"].tolist() == [
        pytest.approx(0.0),
        pytest.approx(1.0),
    ]


def test_calibration_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_calibration_table(
            pd.Series([0, 1]),
            np.array([0.2]),
        )


def test_calibration_rejects_invalid_probability():
    with pytest.raises(ValueError):
        build_calibration_table(
            pd.Series([0, 1]),
            np.array([-0.1, 0.8]),
        )


def test_brier_score():
    y_true = pd.Series([0, 1])
    probabilities = np.array([0.0, 1.0])

    assert calculate_brier_score(
        y_true,
        probabilities,
    ) == pytest.approx(0.0)


def test_expected_calibration_error():
    table = pd.DataFrame(
        {
            "observations": [2, 2],
            "absolute_calibration_error": [
                0.10,
                0.20,
            ],
        }
    )

    assert calculate_expected_calibration_error(
        table
    ) == pytest.approx(0.15)