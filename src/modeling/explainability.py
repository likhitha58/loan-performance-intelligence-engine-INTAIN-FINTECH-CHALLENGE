from __future__ import annotations

import pandas as pd


_REQUIRED_COLUMNS = {
    "loan_id",
    "risk_tier",
    "action_priority",
    "predicted_probability",
    "risk_evidence",
    "evidence_category",
}


_EVIDENCE_MESSAGES = {
    "CURRENT_DEFAULT": "The loan is currently in default.",
    "CURRENT_DELINQUENCY": "The loan is currently delinquent.",
    "RECENT_DELINQUENCY": "Recent delinquency was observed.",
    "STATUS_CHANGE": "A recent loan status change was observed.",
    "HIGH_LTV": "The loan has elevated LTV exposure.",
    "HIGH_DTI": "The loan has elevated debt-to-income exposure.",
    "LOW_CREDIT_SCORE": "The loan has a lower credit-score profile.",
    "MODIFIED_LOAN": "The loan has a modification indicator.",
}


def _validate_row(row: pd.Series) -> None:
    missing = _REQUIRED_COLUMNS - set(row.index)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def _build_reasons(row: pd.Series) -> list[str]:
    evidence = row["risk_evidence"]

    if not evidence:
        if row["evidence_category"] == "MODEL_SIGNAL_ONLY":
            return [
                "The risk flag is based on the model signal without "
                "a matching observable risk-evidence rule."
            ]

        return [
            "No supporting risk evidence was identified."
        ]

    reasons = []

    for code in evidence:
        message = _EVIDENCE_MESSAGES.get(code)

        if message is not None:
            reasons.append(message)
        else:
            reasons.append(
                f"Risk evidence identified: {code}."
            )

    return reasons


def build_explanation(row: pd.Series) -> dict[str, object]:
    """Build a deterministic, evidence-grounded explanation."""

    _validate_row(row)

    reasons = _build_reasons(row)

    explanation = (
        f"Risk tier: {row['risk_tier']}. "
        f"Action priority: {row['action_priority']}. "
        f"Predicted probability: "
        f"{float(row['predicted_probability']):.2%}."
    )

    return {
        "loan_id": row["loan_id"],
        "risk_tier": row["risk_tier"],
        "action_priority": row["action_priority"],
        "predicted_probability": float(
            row["predicted_probability"]
        ),
        "evidence_category": row["evidence_category"],
        "reasons": reasons,
        "explanation": explanation,
    }


def build_explanation_summary(
    risk_output: pd.DataFrame,
) -> pd.DataFrame:
    """Build explanations for all loan-level risk observations."""

    missing = _REQUIRED_COLUMNS - set(risk_output.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    rows = [
        build_explanation(row)
        for _, row in risk_output.iterrows()
    ]

    return pd.DataFrame(rows)