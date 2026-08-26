from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate binary classification performance across thresholds."""

    y_true = pd.Series(y_true).astype(int)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if thresholds is None:
        thresholds = np.arange(
            0.05,
            0.96,
            0.05,
        )

    rows = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "predicted_positive_rate": float(
                    predictions.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def find_best_f1_threshold(
    threshold_results: pd.DataFrame,
) -> dict[str, float]:
    """Return the threshold with the highest F1 score."""

    if threshold_results.empty:
        raise ValueError(
            "threshold_results cannot be empty."
        )

    row = threshold_results.loc[
        threshold_results["f1"].idxmax()
    ]

    return {
        "threshold": float(row["threshold"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "predicted_positive_rate": float(
            row["predicted_positive_rate"]
        ),
    }