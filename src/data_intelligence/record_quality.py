from __future__ import annotations

import pandas as pd


def calculate_record_missingness(
    df: pd.DataFrame,
    exclude_columns: set[str] | None = None,
) -> pd.Series:
    """
    Calculate percentage of missing fields for each record.

    Target columns and identifiers should normally be excluded.
    """

    exclude_columns = exclude_columns or {
        "loan_id",
        "reporting_month",
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
    }

    columns = [
        c for c in df.columns
        if c not in exclude_columns
    ]

    if not columns:
        return pd.Series(
            0.0,
            index=df.index,
        )

    return (
        df[columns]
        .isna()
        .mean(axis=1)
        .mul(100)
    )


def calculate_relationship_violation_rate(
    relationship_violations: pd.DataFrame,
    total_rules: int,
    record_count: int,
) -> pd.Series:
    """
    Convert relationship validation violations into
    a per-record percentage.

    Each rule can contribute at most one violation to a
    given record.
    """

    if record_count == 0:
        return pd.Series(dtype=float)

    if relationship_violations.empty:
        return pd.Series(
            0.0,
            index=range(record_count),
        )

    counts = (
        relationship_violations
        .groupby("record_index")
        .size()
    )

    result = pd.Series(
        0.0,
        index=range(record_count),
    )

    if total_rules > 0:
        result.loc[counts.index] = (
            counts / total_rules * 100
        )

    return result


def calculate_conflict_rate(
    conflict_flags: pd.Series,
) -> pd.Series:
    """
    Convert a boolean source-conflict flag into
    a percentage-style metric.

    For a single record this is either 0 or 100.
    """

    return (
        conflict_flags
        .fillna(False)
        .astype(bool)
        .astype(float)
        .mul(100)
    )


def calculate_outlier_rate(
    outlier_flags: pd.DataFrame,
) -> pd.Series:
    """
    Calculate percentage of monitored numeric fields
    flagged as statistical outliers for each record.
    """

    if outlier_flags.shape[1] == 0:
        return pd.Series(
            0.0,
            index=outlier_flags.index,
        )

    return (
        outlier_flags
        .fillna(False)
        .mean(axis=1)
        .mul(100)
    )


def build_record_quality_metrics(
    df: pd.DataFrame,
    relationship_flags: pd.DataFrame | None = None,
    outlier_flags: pd.DataFrame | None = None,
    conflict_flags: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build the four metrics consumed by the quality-score engine.
    """

    result = pd.DataFrame(
        {
            "loan_id": df["loan_id"].values,
            "reporting_month": df[
                "reporting_month"
            ].values,
        }
    )

    result["missingness_rate"] = (
        calculate_record_missingness(df)
        .reset_index(drop=True)
    )

    if relationship_flags is None:
        result["relationship_violation_rate"] = 0.0
    else:
        result["relationship_violation_rate"] = (
            relationship_flags
            .fillna(False)
            .mean(axis=1)
            .mul(100)
            .reset_index(drop=True)
        )

    if outlier_flags is None:
        result["outlier_rate"] = 0.0
    else:
        result["outlier_rate"] = (
            calculate_outlier_rate(
                outlier_flags
            )
            .reset_index(drop=True)
        )

    if conflict_flags is None:
        result["conflict_rate"] = 0.0
    else:
        result["conflict_rate"] = (
            calculate_conflict_rate(
                conflict_flags
            )
            .reset_index(drop=True)
        )

    return result