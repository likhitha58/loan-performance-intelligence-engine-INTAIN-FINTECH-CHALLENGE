from __future__ import annotations

import pandas as pd


DEFAULT_VALIDATION_START = "2023-07-01"


def temporal_split(
    df: pd.DataFrame,
    validation_start: str = DEFAULT_VALIDATION_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a loan-month panel chronologically.

    All observations before validation_start are used for training.
    Observations from validation_start onward are used for validation.

    No randomization is performed.
    """
    if "reporting_month" not in df.columns:
        raise ValueError(
            "Input dataframe must contain 'reporting_month'."
        )

    result = df.copy()

    result["reporting_month"] = pd.to_datetime(
        result["reporting_month"],
        errors="coerce",
    )

    if result["reporting_month"].isna().any():
        raise ValueError(
            "'reporting_month' contains invalid or missing dates."
        )

    cutoff = pd.Timestamp(validation_start)

    train = result[
        result["reporting_month"] < cutoff
    ].copy()

    validation = result[
        result["reporting_month"] >= cutoff
    ].copy()

    if train.empty:
        raise ValueError(
            "Training split is empty. Move validation_start later."
        )

    if validation.empty:
        raise ValueError(
            "Validation split is empty. Move validation_start earlier."
        )

    if train["reporting_month"].max() >= validation["reporting_month"].min():
        raise AssertionError(
            "Temporal leakage detected: training period overlaps validation."
        )

    return train, validation