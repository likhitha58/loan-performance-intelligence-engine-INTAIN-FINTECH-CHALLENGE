from __future__ import annotations

import pandas as pd

from src.data_intelligence.outliers import calculate_iqr_outliers
from src.data_intelligence.quality_score import score_dataframe
from src.data_intelligence.record_quality import (
    build_record_quality_metrics,
)


def build_relationship_flags(
    train: pd.DataFrame,
    static: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one boolean column per relationship rule.
    """

    df = train.copy()

    static_subset = static[
        ["loan_id", "original_term_months"]
    ]

    df = df.merge(
        static_subset,
        on="loan_id",
        how="left",
        validate="many_to_one",
    )

    df["reporting_month"] = pd.to_datetime(
        df["reporting_month"],
        errors="coerce",
    )

    df["origination_month"] = pd.to_datetime(
        df["origination_month"],
        errors="coerce",
    )

    df["last_updated_at"] = pd.to_datetime(
        df["last_updated_at"],
        errors="coerce",
    )

    month_difference = (
        (df["reporting_month"].dt.year
         - df["origination_month"].dt.year) * 12
        + (
            df["reporting_month"].dt.month
            - df["origination_month"].dt.month
        )
    )

    expected_age = month_difference + 1

    expected_remaining_term = (
        df["original_term_months"]
        - df["loan_age_months"]
    ).clip(lower=0)

    flags = pd.DataFrame(index=train.index)

    flags["R002"] = (
        df["origination_month"]
        > df["reporting_month"]
    )

    flags["R003"] = (
        df["loan_age_months"]
        != expected_age
    )

    flags["R004"] = (
        df["remaining_term_months"]
        != expected_remaining_term
    )

    flags["R005"] = (
        df["remaining_term_months"] < 0
    )

    flags["R006"] = (
        df["current_balance"] < 0
    )

    flags["R007"] = (
        (df["current_status"] == "CURRENT")
        & (df["days_past_due"] != 0)
    )

    flags["R008"] = (
        (df["current_status"] == "DELINQUENT")
        & (df["days_past_due"] <= 0)
    )

    flags["R009"] = (
        (df["prepayment_flag"] == 1)
        & (df["current_status"] != "PREPAID")
    )

    flags["R010"] = (
        (df["default_flag"] == 1)
        & (df["current_status"] != "DEFAULT")
    )

    flags["R011"] = (
        (df["current_status"] == "DEFAULT")
        & df["loss_severity_band"].isna()
    )

    flags["R023"] = (
        df["last_updated_at"]
        < df["reporting_month"]
    )

    return flags.fillna(False)


def build_outlier_flags(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build row-level IQR outlier flags.

    IQR bounds are calculated independently for each
    numeric feature.
    """

    excluded = {
        "loan_id",
        "month_index",
        "reporting_month",
        "origination_month",
        "loan_age_months",
        "remaining_term_months",
        "modification_flag",
        "prepayment_flag",
        "default_flag",
        "exception_required",
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
    }

    numeric_columns = [
        c
        for c in train.select_dtypes(
            include="number"
        ).columns
        if c not in excluded
    ]

    flags = pd.DataFrame(
        False,
        index=train.index,
        columns=numeric_columns,
    )

    for column in numeric_columns:

        series = train[column]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flags[column] = (
            (series < lower)
            | (series > upper)
        ).fillna(False)

    return flags


def build_servicer_conflict_flags(
    train: pd.DataFrame,
    servicer_updates: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Map servicer conflicts onto exact loan-month records.

    Returns
    -------
    conflict_flags
        Boolean conflict flag per monthly record.

    conflict_details
        Servicer details associated with the record.
    """

    core = train[
        ["loan_id", "reporting_month"]
    ].copy()

    updates = servicer_updates.copy()

    core["reporting_month"] = pd.to_datetime(
        core["reporting_month"]
    )

    updates["reporting_month"] = pd.to_datetime(
        updates["reporting_month"]
    )

    conflicts = updates[
        updates["conflict_with_core_record"] == 1
    ].copy()

    conflicts = conflicts.sort_values(
        "update_date"
    )

    # Keep one conflict record per loan-month.
    conflicts = conflicts.drop_duplicates(
        ["loan_id", "reporting_month"],
        keep="first",
    )

    merged = core.merge(
        conflicts[
            [
                "loan_id",
                "reporting_month",
                "servicer_name",
                "reported_status",
                "reported_balance",
                "reported_days_past_due",
            ]
        ],
        on=["loan_id", "reporting_month"],
        how="left",
    )

    conflict_flags = (
        merged["reported_status"]
        .notna()
    )

    details = pd.DataFrame(
        {
            "servicer_conflict": conflict_flags,
            "servicer_name": merged["servicer_name"],
            "reported_status": merged[
                "reported_status"
            ],
            "reported_balance": merged[
                "reported_balance"
            ],
            "reported_days_past_due": merged[
                "reported_days_past_due"
            ],
        }
    )

    return conflict_flags, details

def build_quality_report(
    train: pd.DataFrame,
    static: pd.DataFrame,
    servicer_updates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the complete record-level quality pipeline.
    """

    relationship_flags = build_relationship_flags(
        train,
        static,
    )

    outlier_flags = build_outlier_flags(
        train,
    )

    conflict_flags, conflict_details = (
        build_servicer_conflict_flags(
            train,
            servicer_updates,
        )
    )

    metrics = build_record_quality_metrics(
        train,
        relationship_flags=relationship_flags,
        outlier_flags=outlier_flags,
        conflict_flags=conflict_flags,
    )

    scored = score_dataframe(metrics)

    result = pd.concat(
        [
            train[
                [
                    "loan_id",
                    "reporting_month",
                    "exception_required",
                    "exception_type",
                ]
            ].reset_index(drop=True),

            metrics.drop(
                columns=[
                    "loan_id",
                    "reporting_month",
                ]
            ).reset_index(drop=True),

            scored.drop(
                columns=["loan_id"]
            ).reset_index(drop=True),

            conflict_details.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return result