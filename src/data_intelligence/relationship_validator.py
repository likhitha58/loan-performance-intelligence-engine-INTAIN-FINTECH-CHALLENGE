from __future__ import annotations

from typing import Dict

import pandas as pd


def _months_between(
    start: pd.Series,
    end: pd.Series,
) -> pd.Series:
    """
    Calculate calendar-month difference between two dates.
    """

    return (
        (end.dt.year - start.dt.year) * 12
        + (end.dt.month - start.dt.month)
    )


def validate_relationships(
    train: pd.DataFrame,
    static: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run deterministic cross-field and cross-file
    consistency checks on the monthly loan panel.

    Parameters
    ----------
    train:
        Monthly performance dataset.

    static:
        Origination/static loan attributes.

    Returns
    -------
    pd.DataFrame
        One row per validation rule with violation
        counts and percentages.
    """

    df = train.copy()

    static_subset = static[
        [
            "loan_id",
            "original_term_months",
        ]
    ].copy()

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

    results = []

    def add_result(
        rule_id: str,
        description: str,
        violations: pd.Series,
    ) -> None:

        violations = violations.fillna(False)

        violation_count = int(violations.sum())
        total = len(violations)

        results.append(
            {
                "rule_id": rule_id,
                "description": description,
                "rows_checked": total,
                "violation_count": violation_count,
                "violation_pct": round(
                    violation_count / total * 100,
                    4,
                )
                if total
                else 0.0,
            }
        )

    # ---------------------------------------------------------
    # R002 — Origination cannot be after reporting month
    # ---------------------------------------------------------

    add_result(
        "R002",
        "origination_month <= reporting_month",
        df["origination_month"]
        > df["reporting_month"],
    )

    # ---------------------------------------------------------
    # R003 — Loan age consistency
    #
    # Convention:
    # origination month = loan_age_months 1
    # ---------------------------------------------------------

    month_difference = _months_between(
        df["origination_month"],
        df["reporting_month"],
    )

    expected_age = month_difference + 1

    add_result(
        "R003",
        "loan_age_months matches origination/reporting dates",
        df["loan_age_months"] != expected_age,
    )

    # ---------------------------------------------------------
    # R004 — Remaining term consistency
    # ---------------------------------------------------------

    expected_remaining_term = (
        df["original_term_months"]
        - df["loan_age_months"]
    ).clip(lower=0)

    add_result(
        "R004",
        "remaining_term_months matches original_term_months - loan_age_months",
        df["remaining_term_months"]
        != expected_remaining_term,
    )

    # ---------------------------------------------------------
    # R005 — Remaining term cannot be negative
    # ---------------------------------------------------------

    add_result(
        "R005",
        "remaining_term_months >= 0",
        df["remaining_term_months"] < 0,
    )

    # ---------------------------------------------------------
    # R006 — Current balance cannot be negative
    # ---------------------------------------------------------

    add_result(
        "R006",
        "current_balance >= 0",
        df["current_balance"] < 0,
    )

    # ---------------------------------------------------------
    # R007 — CURRENT loans must have 0 DPD
    # ---------------------------------------------------------

    add_result(
        "R007",
        "CURRENT loans should report 0 days past due",
        (
            (df["current_status"] == "CURRENT")
            & (df["days_past_due"] != 0)
        ),
    )

    # ---------------------------------------------------------
    # R008 — DELINQUENT loans must have positive DPD
    # ---------------------------------------------------------

    add_result(
        "R008",
        "DELINQUENT loans should report positive days past due",
        (
            (df["current_status"] == "DELINQUENT")
            & (df["days_past_due"] <= 0)
        ),
    )

    # ---------------------------------------------------------
    # R009 — Prepayment consistency
    # ---------------------------------------------------------

    add_result(
        "R009",
        "prepayment_flag == 1 implies PREPAID status",
        (
            (df["prepayment_flag"] == 1)
            & (df["current_status"] != "PREPAID")
        ),
    )

    # ---------------------------------------------------------
    # R010 — Default consistency
    # ---------------------------------------------------------

    add_result(
        "R010",
        "default_flag == 1 implies DEFAULT status",
        (
            (df["default_flag"] == 1)
            & (df["current_status"] != "DEFAULT")
        ),
    )

    # ---------------------------------------------------------
    # R011 — DEFAULT requires loss severity
    # ---------------------------------------------------------

    add_result(
        "R011",
        "DEFAULT rows should have loss_severity_band",
        (
            (df["current_status"] == "DEFAULT")
            & df["loss_severity_band"].isna()
        ),
    )

    # ---------------------------------------------------------
    # R023 — last_updated_at >= reporting_month
    # ---------------------------------------------------------

    reporting_timestamp = df["reporting_month"]

    add_result(
        "R023",
        "last_updated_at >= reporting_month",
        df["last_updated_at"] < reporting_timestamp,
    )

    return pd.DataFrame(results)


def get_violation_rows(
    train: pd.DataFrame,
    static: pd.DataFrame,
    rule_id: str,
) -> pd.DataFrame:
    """
    Return the actual records violating a specific rule.
    """

    df = train.copy()

    df = df.merge(
        static[
            [
                "loan_id",
                "original_term_months",
            ]
        ],
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

    month_difference = _months_between(
        df["origination_month"],
        df["reporting_month"],
    )

    expected_age = month_difference + 1

    expected_remaining_term = (
        df["original_term_months"]
        - df["loan_age_months"]
    ).clip(lower=0)

    conditions = {
        "R002": (
            df["origination_month"]
            > df["reporting_month"]
        ),
        "R003": (
            df["loan_age_months"]
            != expected_age
        ),
        "R004": (
            df["remaining_term_months"]
            != expected_remaining_term
        ),
        "R005": (
            df["remaining_term_months"] < 0
        ),
        "R006": (
            df["current_balance"] < 0
        ),
        "R007": (
            (df["current_status"] == "CURRENT")
            & (df["days_past_due"] != 0)
        ),
        "R008": (
            (df["current_status"] == "DELINQUENT")
            & (df["days_past_due"] <= 0)
        ),
        "R009": (
            (df["prepayment_flag"] == 1)
            & (df["current_status"] != "PREPAID")
        ),
        "R010": (
            (df["default_flag"] == 1)
            & (df["current_status"] != "DEFAULT")
        ),
        "R011": (
            (df["current_status"] == "DEFAULT")
            & df["loss_severity_band"].isna()
        ),
        "R023": (
            df["last_updated_at"]
            < df["reporting_month"]
        ),
    }

    if rule_id not in conditions:
        raise ValueError(
            f"Unsupported relationship rule: {rule_id}"
        )

    mask = conditions[rule_id].fillna(False)

    return df.loc[mask].copy()