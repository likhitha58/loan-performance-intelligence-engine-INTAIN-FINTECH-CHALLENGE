import numpy as np
import pandas as pd
import pytest

from src.modeling.thresholds import (
    evaluate_thresholds,
    find_best_f1_threshold,
)


def test_evaluate_thresholds_returns_expected_columns():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    result = evaluate_thresholds(
        y_true,
        probabilities,
        thresholds=np.array([0.5, 0.7]),
    )

    assert len(result) == 2
    assert list(result.columns) == [
        "threshold",
        "precision",
        "recall",
        "f1",
        "predicted_positive_rate",
    ]


def test_threshold_predictions_change_with_threshold():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    result = evaluate_thresholds(
        y_true,
        probabilities,
        thresholds=np.array([0.5, 0.8]),
    )

    assert (
        result.loc[
            result["threshold"] == 0.5,
            "predicted_positive_rate",
        ].iloc[0]
        >
        result.loc[
            result["threshold"] == 0.8,
            "predicted_positive_rate",
        ].iloc[0]
    )


def test_find_best_f1_threshold():
    results = pd.DataFrame(
        {
            "threshold": [0.3, 0.5, 0.7],
            "precision": [0.4, 0.6, 0.8],
            "recall": [0.9, 0.7, 0.4],
            "f1": [0.55, 0.65, 0.53],
            "predicted_positive_rate": [0.4, 0.3, 0.1],
        }
    )

    best = find_best_f1_threshold(results)

    assert best["threshold"] == 0.5
    assert best["f1"] == 0.65


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        evaluate_thresholds(
            pd.Series([0, 1]),
            np.array([0.2]),
        )


def test_empty_results_raise():
    with pytest.raises(ValueError):
        find_best_f1_threshold(
            pd.DataFrame(
                columns=[
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "predicted_positive_rate",
                ]
            )
        )