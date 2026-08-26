from __future__ import annotations

import numpy as np
import pandas as pd


def build_calibration_table(
    y_true: pd.Series,
    probabilities: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Build a probability-bin calibration table."""

    if bins <= 1:
        raise ValueError("bins must be greater than 1.")

    y_true = pd.Series(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(
            "probabilities must be between 0 and 1."
        )

    edges = np.linspace(0.0, 1.0, bins + 1)

    rows = []

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]

        if index == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        observed_rate = float(
            y_true[mask].mean()
        )

        predicted_rate = float(
            probabilities[mask].mean()
        )

        rows.append(
            {
                "bin": index + 1,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "observations": count,
                "average_predicted_probability": predicted_rate,
                "observed_positive_rate": observed_rate,
                "absolute_calibration_error": abs(
                    predicted_rate - observed_rate
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_brier_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """Calculate the Brier score for binary predictions."""

    y_true = pd.Series(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(
            "probabilities must be between 0 and 1."
        )

    return float(
        np.mean(
            (probabilities - y_true.to_numpy()) ** 2
        )
    )


def calculate_expected_calibration_error(
    calibration_table: pd.DataFrame,
) -> float:
    """Calculate observation-weighted expected calibration error."""

    required_columns = {
        "observations",
        "absolute_calibration_error",
    }

    missing = required_columns - set(
        calibration_table.columns
    )

    if missing:
        raise ValueError(
            f"Missing calibration columns: {sorted(missing)}"
        )

    total = calibration_table["observations"].sum()

    if total <= 0:
        raise ValueError(
            "calibration_table must contain observations."
        )

    return float(
        (
            calibration_table["observations"]
            * calibration_table["absolute_calibration_error"]
        ).sum()
        / total
    )