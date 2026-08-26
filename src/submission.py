from __future__ import annotations

from pathlib import Path

import pandas as pd


SUBMISSION_COLUMNS = [
    "loan_id",
    "reporting_month",
    "pred_next_3m_delinquency_prob",
    "pred_next_6m_delinquency_prob",
    "pred_next_12m_default_prob",
    "pred_next_12m_prepayment_prob",
    "pred_next_state",
    "anomaly_score",
    "exception_type",
    "top_drivers",
    "recommended_action",
    "confidence",
]


def _validate_columns(
    data: pd.DataFrame,
    required: set[str],
) -> None:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required submission columns: {sorted(missing)}"
        )


def _join_driver_values(values: object) -> str:
    if isinstance(values, (list, tuple, set)):
        return "; ".join(str(value) for value in values)

    if values is None:
        return ""

    try:
        if pd.isna(values):
            return ""
    except (TypeError, ValueError):
        pass

    return str(values)


def _normalise_keys(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["reporting_month"] = pd.to_datetime(
        result["reporting_month"]
    ).dt.strftime("%Y-%m-%d")
    return result


def _merge_optional(
    result: pd.DataFrame,
    optional: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    optional = _normalise_keys(
        optional[columns].copy()
    )

    keys = ["loan_id", "reporting_month"]
    value_columns = [
        column for column in columns if column not in keys
    ]

    # Remove any existing versions before merging.
    result = result.drop(
        columns=[
            column
            for column in value_columns
            if column in result.columns
        ],
        errors="ignore",
    )

    return result.merge(
        optional,
        on=keys,
        how="left",
    )


def build_submission(
    prediction_data: pd.DataFrame,
    transition_data: pd.DataFrame | None = None,
    anomaly_data: pd.DataFrame | None = None,
    explanation_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combine existing model outputs into the challenge
    submission schema.
    """

    required = {
        "loan_id",
        "reporting_month",
        "pred_next_3m_delinquency_prob",
        "pred_next_6m_delinquency_prob",
        "pred_next_12m_default_prob",
        "pred_next_12m_prepayment_prob",
    }

    _validate_columns(prediction_data, required)

    result = _normalise_keys(prediction_data)

    # Existing state output can be preserved if already present.
    if "pred_next_state" not in result.columns:
        result["pred_next_state"] = pd.NA

    if "confidence" not in result.columns:
        result["confidence"] = pd.NA

    # ---------------------------------------------------------
    # Transition model
    # ---------------------------------------------------------

    if transition_data is not None:
        _validate_columns(
            transition_data,
            {
                "loan_id",
                "reporting_month",
                "pred_next_state",
                "next_state_confidence",
            },
        )

        transition = transition_data[
            [
                "loan_id",
                "reporting_month",
                "pred_next_state",
                "next_state_confidence",
            ]
        ].copy()

        transition = transition.rename(
            columns={
                "next_state_confidence": "confidence"
            }
        )

        result = _merge_optional(
            result,
            transition,
            [
                "loan_id",
                "reporting_month",
                "pred_next_state",
                "confidence",
            ],
        )

    # ---------------------------------------------------------
    # Anomaly output
    # ---------------------------------------------------------

    if "anomaly_score" not in result.columns:
        result["anomaly_score"] = 0.0

    if "exception_type" not in result.columns:
        result["exception_type"] = ""

    if anomaly_data is not None:
        _validate_columns(
            anomaly_data,
            {
                "loan_id",
                "reporting_month",
                "anomaly_score",
                "exception_type",
            },
        )

        result = _merge_optional(
            result,
            anomaly_data,
            [
                "loan_id",
                "reporting_month",
                "anomaly_score",
                "exception_type",
            ],
        )

    # ---------------------------------------------------------
    # Explainability output
    # ---------------------------------------------------------

    if "top_drivers" not in result.columns:
        result["top_drivers"] = ""

    if "recommended_action" not in result.columns:
        result["recommended_action"] = ""

    if explanation_data is not None:
        _validate_columns(
            explanation_data,
            {
                "loan_id",
                "reporting_month",
                "reasons",
                "action_priority",
            },
        )

        explanations = explanation_data[
            [
                "loan_id",
                "reporting_month",
                "reasons",
                "action_priority",
            ]
        ].copy()

        explanations["top_drivers"] = explanations[
            "reasons"
        ].apply(_join_driver_values)

        explanations = explanations.rename(
            columns={
                "action_priority": "recommended_action"
            }
        )

        result = _merge_optional(
            result,
            explanations[
                [
                    "loan_id",
                    "reporting_month",
                    "top_drivers",
                    "recommended_action",
                ]
            ],
            [
                "loan_id",
                "reporting_month",
                "top_drivers",
                "recommended_action",
            ],
        )

    # ---------------------------------------------------------
    # Final cleanup
    # ---------------------------------------------------------

    result["anomaly_score"] = pd.to_numeric(
        result["anomaly_score"],
        errors="coerce",
    ).fillna(0.0)

    result["confidence"] = pd.to_numeric(
        result["confidence"],
        errors="coerce",
    )

    result["exception_type"] = result[
        "exception_type"
    ].fillna("")

    result["top_drivers"] = result[
        "top_drivers"
    ].fillna("")

    result["recommended_action"] = result[
        "recommended_action"
    ].fillna("")

    return result[SUBMISSION_COLUMNS]


def write_submission(
    submission: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Validate and write the submission CSV."""

    missing = set(SUBMISSION_COLUMNS) - set(
        submission.columns
    )

    if missing:
        raise ValueError(
            f"Submission is missing columns: {sorted(missing)}"
        )

    output = submission[SUBMISSION_COLUMNS].copy()

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        destination,
        index=False,
    )

    return destination