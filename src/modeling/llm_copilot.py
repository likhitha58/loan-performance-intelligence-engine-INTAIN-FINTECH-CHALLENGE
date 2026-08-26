from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "loan_id",
    "reporting_month",
    "predicted_probability",
    "risk_tier",
    "action_priority",
    "risk_evidence",
    "evidence_category",
}


def build_reviewer_context(row: pd.Series) -> dict[str, object]:
    """Build a structured, grounded context for reviewer assistance."""

    missing = REQUIRED_COLUMNS - set(row.index)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return {
        "loan_id": row["loan_id"],
        "reporting_month": row["reporting_month"],
        "predicted_probability": float(
            row["predicted_probability"]
        ),
        "risk_tier": row["risk_tier"],
        "action_priority": row["action_priority"],
        "risk_evidence": list(row["risk_evidence"]),
        "evidence_category": row["evidence_category"],
    }


def generate_reviewer_prompt(
    context: dict[str, object],
) -> str:
    """Generate a grounded prompt for an LLM reviewer assistant."""

    evidence = ", ".join(
        str(item) for item in context["risk_evidence"]
    )

    return f"""
You are a loan risk reviewer assistant.

Review the following model-generated risk context.

Loan ID: {context["loan_id"]}
Reporting Month: {context["reporting_month"]}
Predicted Default Probability: {context["predicted_probability"]:.4f}
Risk Tier: {context["risk_tier"]}
Action Priority: {context["action_priority"]}
Evidence Category: {context["evidence_category"]}
Risk Evidence: {evidence}

Provide a concise reviewer-oriented explanation.

Use only the information provided above.
Do not invent borrower information, causes, events, or financial details.
Do not override the model's risk decision.
Clearly distinguish model signals from observable evidence.
If the evidence is insufficient, explicitly state that.
""".strip()