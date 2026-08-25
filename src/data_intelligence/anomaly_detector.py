from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data_intelligence.relationship_validator import (
    _months_between,
)


@dataclass(frozen=True)
class AnomalyRule:
    rule_id: str
    anomaly_type: str
    severity: str
    description: str


ANOMALY_RULES = [
    AnomalyRule(
        "R002",
        "date_inconsistency",
        "HIGH",
        "Origination month occurs after reporting month.",
    ),
    AnomalyRule(
        "R003",
        "impossible_loan_age",
        "HIGH",
        "Loan age does not match origination and reporting dates.",
    ),
    AnomalyRule(
        "R004",
        "invalid_remaining_term",
        "HIGH",
        "Remaining term does not match original term and loan age.",
    ),
    AnomalyRule(
        "R005",
        "invalid_remaining_term",
        "HIGH",
        "Remaining term is negative.",
    ),
    AnomalyRule(
    "R006",
    "balance_inconsistency",
    "HIGH",
    "Current balance exceeds original balance.",
    ),
    AnomalyRule(
        "R007",
        "delinquency_status_inconsistency",
        "MEDIUM",
        "CURRENT status has non-zero days past due.",
    ),
    AnomalyRule(
        "R008",
        "delinquency_status_inconsistency",
        "MEDIUM",
        "DELINQUENT status has non-positive days past due.",
    ),
    AnomalyRule(
        "R009",
        "prepayment_status_inconsistency",
        "HIGH",
        "Prepayment flag is inconsistent with PREPAID status.",
    ),
    AnomalyRule(
        "R010",
        "default_status_inconsistency",
        "CRITICAL",
        "Default flag is inconsistent with DEFAULT status.",
    ),
    AnomalyRule(
        "R011",
        "missing_loss_severity",
        "HIGH",
        "DEFAULT record has no loss severity band.",
    ),
    AnomalyRule(
        "R012",
        "missing_document_status",
        "MEDIUM",
        "Document status is missing.",
    ),
    AnomalyRule(
    "R013",
    "unexpected_missing_value",
    "MEDIUM",
    "A required record attribute is unexpectedly missing.",
    ),
    AnomalyRule(
    "R014",
    "delinquency_status_inconsistency",
    "HIGH",
    "Loan status is inconsistent with the reported delinquency state.",
    ),
    AnomalyRule(
        "R023",
        "date_inconsistency",
        "HIGH",
        "Last updated timestamp precedes reporting month.",
    ),
]


SEVERITY_WEIGHT = {
    "LOW": 0.20,
    "MEDIUM": 0.40,
    "HIGH": 0.70,
    "CRITICAL": 1.00,
}


def _prepare_data(
    train: pd.DataFrame,
    static: pd.DataFrame,
) -> pd.DataFrame:

    df = train.copy()

    static_subset = static[
        [
            "loan_id",
            "original_term_months",
        ]
    ]

    df = df.merge(
        static_subset,
        on="loan_id",
        how="left",
        validate="many_to_one",
    )

    for column in [
        "reporting_month",
        "origination_month",
        "last_updated_at",
    ]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    return df


def detect_rule_violations(
    train: pd.DataFrame,
    static: pd.DataFrame,
) -> dict[str, pd.Series]:

    df = _prepare_data(train, static)

    month_difference = _months_between(
        df["origination_month"],
        df["reporting_month"],
    )

    expected_age = month_difference + 1

    expected_remaining_term = (
        df["original_term_months"]
        - df["loan_age_months"]
    ).clip(lower=0)

    violations = {
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
        df["current_balance"] > df["original_balance"]
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

        "R012": (
            df["document_status"].isna()
        ),
        "R013": (
        df[
            [
                "current_balance",
                "interest_rate",
                "credit_score_band",
                "servicer_name",
            ]
        ].isna().any(axis=1)
        ),
        "R014": (
        (
            df["current_status"].isin(
                ["DELINQUENT", "DEFAULT"]
            )
            & (df["days_past_due"] <= 0)
        )
        |
        (
            df["current_status"].isin(
                ["PREPAID", "CLOSED"]
            )
            & (df["days_past_due"] > 0)
        )
        ),

        "R023": (
            df["last_updated_at"]
            < df["reporting_month"]
        ),
    }

    return {
        rule_id: mask.fillna(False)
        for rule_id, mask in violations.items()
    }


def _combine_types(types: list[str]) -> str:
    return "|".join(sorted(set(types)))


def _severity(types: list[str]) -> str:

    if not types:
        return "NONE"

    severities = [
        rule.severity
        for rule in ANOMALY_RULES
        if rule.anomaly_type in types
    ]

    if "CRITICAL" in severities:
        return "CRITICAL"

    if "HIGH" in severities:
        return "HIGH"

    if "MEDIUM" in severities:
        return "MEDIUM"

    return "LOW"


def build_anomaly_report(
    train: pd.DataFrame,
    static: pd.DataFrame,
) -> pd.DataFrame:

    violations = detect_rule_violations(
        train,
        static,
    )

    result = train[
        [
            "loan_id",
            "reporting_month",
        ]
    ].copy()

    rule_lookup = {
        rule.rule_id: rule
        for rule in ANOMALY_RULES
    }

    anomaly_types = []
    violated_rules = []
    severities = []
    anomaly_scores = []

    for index in train.index:

        row_rules = [
            rule_id
            for rule_id, mask in violations.items()
            if bool(mask.loc[index])
        ]

        row_types = [
            rule_lookup[rule_id].anomaly_type
            for rule_id in row_rules
        ]

        row_severity = _severity(row_types)

        if not row_rules:
            score = 0.0
        else:
            weights = [
                SEVERITY_WEIGHT[
                    rule_lookup[rule_id].severity
                ]
                for rule_id in row_rules
            ]

            # Multiple independent violations increase
            # the anomaly score, but the score remains <= 1.
            score = min(
                1.0,
                sum(weights) / 2.0,
            )

        anomaly_types.append(
            _combine_types(row_types)
        )

        violated_rules.append(
            "|".join(row_rules)
        )

        severities.append(
            row_severity
        )

        anomaly_scores.append(
            round(score, 4)
        )

    result["anomaly_detected"] = (
        result.index.map(
            lambda i: bool(violated_rules[i])
        )
    )

    result["violated_rules"] = violated_rules
    result["anomaly_types"] = anomaly_types
    result["severity"] = severities
    result["anomaly_score"] = anomaly_scores

    return result