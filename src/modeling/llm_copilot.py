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


EVIDENCE_MESSAGES = {
    "CURRENT_DEFAULT": "The loan is currently in default.",
    "CURRENT_DELINQUENCY": "The loan is currently delinquent.",
    "RECENT_DELINQUENCY": "Recent delinquency was observed.",
    "STATUS_CHANGE": "A recent loan status change was observed.",
    "HIGH_LTV": "The loan has elevated LTV exposure.",
    "HIGH_DTI": "The loan has elevated debt-to-income exposure.",
    "LOW_CREDIT_SCORE": "The loan has a lower credit-score profile.",
    "MODIFIED_LOAN": "The loan has a modification indicator.",
}


def _validate_context_columns(row: pd.Series) -> None:
    """Validate that all fields required by the Copilot are present."""

    missing = REQUIRED_COLUMNS - set(row.index)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def _normalise_evidence(value: object) -> list[str]:
    """Convert evidence into a safe deterministic list."""

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        return [value]

    return [str(value)]


def build_reviewer_context(row: pd.Series) -> dict[str, object]:
    """Build structured, grounded context for an LLM reviewer."""

    _validate_context_columns(row)

    evidence = _normalise_evidence(
        row["risk_evidence"]
    )

    probability = float(
        row["predicted_probability"]
    )

    if not 0 <= probability <= 1:
        raise ValueError(
            "predicted_probability must be between 0 and 1."
        )

    return {
        "loan_id": row["loan_id"],
        "reporting_month": row["reporting_month"],
        "predicted_probability": probability,
        "risk_tier": str(row["risk_tier"]),
        "action_priority": str(row["action_priority"]),
        "risk_evidence": evidence,
        "evidence_category": str(
            row["evidence_category"]
        ),
    }


def _evidence_descriptions(
    evidence: list[str],
) -> list[str]:
    """Translate evidence codes into reviewer-readable statements."""

    descriptions = []

    for code in evidence:
        descriptions.append(
            EVIDENCE_MESSAGES.get(
                code,
                f"Risk evidence identified: {code}.",
            )
        )

    return descriptions


def generate_reviewer_prompt(
    context: dict[str, object],
) -> str:
    """Generate a grounded prompt for an LLM reviewer assistant."""

    evidence = _normalise_evidence(
        context.get("risk_evidence")
    )

    evidence_text = (
        ", ".join(evidence)
        if evidence
        else "None identified"
    )

    return f"""
You are a loan risk reviewer assistant.

Review ONLY the model-generated context provided below.

Loan ID: {context["loan_id"]}
Reporting Month: {context["reporting_month"]}
Predicted Default Probability: {context["predicted_probability"]:.4f}
Risk Tier: {context["risk_tier"]}
Action Priority: {context["action_priority"]}
Evidence Category: {context["evidence_category"]}
Risk Evidence: {evidence}

Your response must:

1. Explain the model signal concisely.
2. Identify the observable evidence supporting the signal.
3. Clearly distinguish model-derived information from observable evidence.
4. State when supporting evidence is insufficient.
5. Suggest appropriate reviewer follow-up questions.
6. Do not invent borrower information, financial details,
   causes, events, or missing observations.
7. Do not override or replace the model's risk decision.
8. Do not claim certainty about future borrower behaviour.

Return a concise reviewer-oriented assessment.
""".strip()


def build_deterministic_reviewer_summary(
    context: dict[str, object],
) -> dict[str, object]:
    """
    Produce an offline-safe reviewer summary.

    This acts as the deterministic fallback when no external
    LLM service is configured.
    """

    evidence = _normalise_evidence(
        context.get("risk_evidence")
    )

    evidence_descriptions = _evidence_descriptions(
        evidence
    )

    if evidence_descriptions:
        evidence_summary = evidence_descriptions
    else:
        evidence_summary = [
            "No supporting observable risk evidence was identified."
        ]

    if context["evidence_category"] == "MODEL_SIGNAL_ONLY":
        evidence_summary.append(
            "The current risk classification is primarily a "
            "model signal and should receive additional reviewer scrutiny."
        )

    follow_up_questions = [
        "What recent account activity should be reviewed?",
        "Has the observed risk signal changed materially over time?",
    ]

    if not evidence:
        follow_up_questions.append(
            "What additional borrower or loan information is available "
            "to validate the model signal?"
        )

    return {
        "loan_id": context["loan_id"],
        "risk_tier": context["risk_tier"],
        "action_priority": context["action_priority"],
        "predicted_probability": context[
            "predicted_probability"
        ],
        "evidence_category": context[
            "evidence_category"
        ],
        "evidence_summary": evidence_summary,
        "follow_up_questions": follow_up_questions,
        "llm_required": True,
        "grounding_status": (
            "GROUNDED"
            if evidence
            else "MODEL_SIGNAL_ONLY"
        ),
    }


def build_copilot_record(
    row: pd.Series,
) -> dict[str, object]:
    """
    Build the complete Copilot payload for one observation.

    The returned structure can be passed to an external LLM
    service later without changing the underlying risk logic.
    """

    context = build_reviewer_context(row)

    prompt = generate_reviewer_prompt(
        context
    )

    fallback = build_deterministic_reviewer_summary(
        context
    )

    return {
        "context": context,
        "prompt": prompt,
        "fallback": fallback,
    }


def build_copilot_batch(
    risk_output: pd.DataFrame,
) -> pd.DataFrame:
    """Build Copilot records for an entire risk-output table."""

    missing = REQUIRED_COLUMNS - set(
        risk_output.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if risk_output.empty:
        return pd.DataFrame(
            columns=[
                "loan_id",
                "reporting_month",
                "risk_tier",
                "action_priority",
                "predicted_probability",
                "evidence_category",
                "copilot_prompt",
                "grounding_status",
            ]
        )

    records = []

    for _, row in risk_output.iterrows():
        record = build_copilot_record(row)

        context = record["context"]
        fallback = record["fallback"]

        records.append(
            {
                "loan_id": context["loan_id"],
                "reporting_month": context[
                    "reporting_month"
                ],
                "risk_tier": context["risk_tier"],
                "action_priority": context[
                    "action_priority"
                ],
                "predicted_probability": context[
                    "predicted_probability"
                ],
                "evidence_category": context[
                    "evidence_category"
                ],
                "copilot_prompt": record["prompt"],
                "grounding_status": fallback[
                    "grounding_status"
                ],
            }
        )

    return pd.DataFrame(records)