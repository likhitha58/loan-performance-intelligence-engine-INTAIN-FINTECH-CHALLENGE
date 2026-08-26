from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMNS = {
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
}

QUALITY_COLUMNS = {
    "exception_required",
    "exception_type",
}

IDENTIFIER_COLUMNS = {
    "loan_id",
}

LEAKAGE_RISK_COLUMNS = {
    "next_state",
    "loss_severity_band",
}

RAW_DATE_COLUMNS = {
    "reporting_month",
    "origination_month",
    "last_updated_at",
}

def _validate_columns(df: pd.DataFrame) -> None:
    """Validate that the minimum columns required for feature engineering exist."""
    required = {
        "loan_id",
        "month_index",
        "reporting_month",
        "original_balance",
        "current_balance",
        "loan_age_months",
        "remaining_term_months",
        "days_past_due",
        "current_status",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns for feature engineering: {missing}"
        )


def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with temporal columns normalized."""
    result = df.copy()

    result["reporting_month"] = pd.to_datetime(
        result["reporting_month"],
        errors="coerce",
    )

    if "origination_month" in result.columns:
        result["origination_month"] = pd.to_datetime(
            result["origination_month"],
            errors="coerce",
        )

    if "last_updated_at" in result.columns:
        result["last_updated_at"] = pd.to_datetime(
            result["last_updated_at"],
            errors="coerce",
        )

    return result


def _sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records chronologically within each loan."""
    return df.sort_values(
        ["loan_id", "reporting_month", "month_index"],
        kind="mergesort",
    ).copy()


def _previous_calendar_month_available(
    reporting_month: pd.Series,
) -> pd.Series:
    """
    Identify whether the immediately previous calendar month exists.

    This prevents a record at month 3 from treating month 1 as a
    one-month lag when month 2 is absent.
    """
    previous_month = reporting_month.groupby(
        reporting_month.index
    ).transform(lambda x: x)

    return previous_month


def _build_balance_features(result: pd.DataFrame) -> None:
    """Create current and historical balance features."""

    result["balance_ratio"] = np.where(
        result["original_balance"].notna()
        & (result["original_balance"] != 0),
        result["current_balance"] / result["original_balance"],
        np.nan,
    )

    result["_previous_reporting_month"] = result.groupby(
        "loan_id"
    )["reporting_month"].shift(1)

    result["_previous_balance"] = result.groupby(
        "loan_id"
    )["current_balance"].shift(1)

    result["_month_gap"] = (
        (
            result["reporting_month"].dt.year
            - result["_previous_reporting_month"].dt.year
        )
        * 12
        + (
            result["reporting_month"].dt.month
            - result["_previous_reporting_month"].dt.month
        )
    )

    valid_previous_month = result["_month_gap"] == 1

    result["balance_change_1m"] = np.where(
        valid_previous_month,
        result["current_balance"] - result["_previous_balance"],
        np.nan,
    )

    previous_balance = result["_previous_balance"]

    result["balance_change_pct_1m"] = np.where(
        valid_previous_month
        & previous_balance.notna()
        & (previous_balance != 0),
        (
            result["current_balance"] - previous_balance
        ) / previous_balance,
        np.nan,
    )

    result["balance_reduction"] = (
        result["original_balance"] - result["current_balance"]
    )

    result["balance_remaining_ratio"] = np.where(
        result["original_balance"].notna()
        & (result["original_balance"] != 0),
        result["current_balance"] / result["original_balance"],
        np.nan,
    )


def _build_delinquency_features(result: pd.DataFrame) -> None:
    """Create historical delinquency features using true calendar lags."""

    grouped_dpd = result.groupby("loan_id")["days_past_due"]

    result["_dpd_lag_1_observed"] = grouped_dpd.shift(1)

    result["_previous_reporting_month"] = result.groupby(
        "loan_id"
    )["reporting_month"].shift(1)

    result["_month_gap"] = (
        (
            result["reporting_month"].dt.year
            - result["_previous_reporting_month"].dt.year
        )
        * 12
        + (
            result["reporting_month"].dt.month
            - result["_previous_reporting_month"].dt.month
        )
    )

    valid_previous_month = result["_month_gap"] == 1

    result["dpd_lag_1m"] = np.where(
        valid_previous_month,
        result["_dpd_lag_1_observed"],
        np.nan,
    )

    result["delinquent_flag"] = (
        result["days_past_due"].fillna(0) > 0
    ).astype(int)

    result["delinquent_lag_1m"] = np.where(
        valid_previous_month,
        result.groupby("loan_id")["delinquent_flag"].shift(1),
        np.nan,
    )


def _build_lifecycle_features(result: pd.DataFrame) -> None:
    """Create loan lifecycle and maturity features."""

    result["age_ratio"] = np.where(
        (
            result["loan_age_months"].notna()
            & result["remaining_term_months"].notna()
            & (
                result["loan_age_months"]
                + result["remaining_term_months"]
            > 0
            )
        ),
        result["loan_age_months"]
        / (
            result["loan_age_months"]
            + result["remaining_term_months"]
        ),
        np.nan,
    )

    result["maturity_proximity"] = np.where(
        result["remaining_term_months"].notna(),
        result["remaining_term_months"] <= 12,
        np.nan,
    ).astype(float)


def _build_status_features(result: pd.DataFrame) -> None:
    """Create current status transition features."""

    previous_status = result.groupby(
        "loan_id"
    )["current_status"].shift(1)

    result["previous_status"] = previous_status

    result["status_change_flag"] = (
        previous_status.notna()
        & result["current_status"].ne(previous_status)
    ).astype(int)

    result["status_change_flag"] = np.where(
        previous_status.notna(),
        result["status_change_flag"],
        np.nan,
    )


def _build_temporal_features(result: pd.DataFrame) -> None:
    """Create calendar features without using future information."""

    result["reporting_year"] = result["reporting_month"].dt.year
    result["reporting_month_number"] = result["reporting_month"].dt.month

    if "origination_month" in result.columns:
        result["origination_year"] = (
            result["origination_month"].dt.year
        )
        result["origination_month_number"] = (
            result["origination_month"].dt.month
        )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe loan performance features.

    The function preserves one output row per input row and computes
    historical features only from observations belonging to the same
    loan and the immediately preceding calendar month.
    """
    _validate_columns(df)

    result = _prepare_dates(df)
    result = _sort_panel(result)

    _build_balance_features(result)
    _build_delinquency_features(result)
    _build_lifecycle_features(result)
    _build_status_features(result)
    _build_temporal_features(result)

    temporary_columns = [
        "_previous_reporting_month",
        "_previous_balance",
        "_month_gap",
        "_dpd_lag_1_observed",
    ]

    result = result.drop(
        columns=[
            column
            for column in temporary_columns
            if column in result.columns
        ]
    )

    return result


def get_feature_columns(
    df: pd.DataFrame,
    additional_exclusions: Iterable[str] | None = None,
) -> list[str]:
    """
    Return columns eligible for model features.

    Identifiers, forward-looking targets, and data-quality ground truth
    are explicitly excluded.
    """
    exclusions = (
        TARGET_COLUMNS
        | QUALITY_COLUMNS
        | IDENTIFIER_COLUMNS
        | LEAKAGE_RISK_COLUMNS
        | RAW_DATE_COLUMNS
    )

    if additional_exclusions:
        exclusions |= set(additional_exclusions)

    return [
        column
        for column in df.columns
        if column not in exclusions
    ]