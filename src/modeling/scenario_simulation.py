from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_intelligence.features import build_features
from src.modeling.prediction import generate_predictions


SCENARIOS = {
    "BASE": {
        "credit_risk_multiplier": 1.00,
        "delinquency_multiplier": 1.00,
        "default_multiplier": 1.00,
        "prepayment_multiplier": 1.00,
    },
    "ADVERSE_CREDIT": {
        "credit_risk_multiplier": 1.45,
        "delinquency_multiplier": 1.65,
        "default_multiplier": 1.90,
        "prepayment_multiplier": 0.55,
    },
    "HIGH_PREPAYMENT": {
        "credit_risk_multiplier": 0.90,
        "delinquency_multiplier": 0.85,
        "default_multiplier": 0.80,
        "prepayment_multiplier": 2.10,
    },
    # Retain the original controlled feature stress tests.
    "HIGH_LTV_STRESS": {
        "column": "ltv_band",
        "mapping": {
            "71-80": ">95",
            "81-90": ">95",
            "91-95": ">95",
        },
    },
    "HIGH_DTI_STRESS": {
        "column": "dti_band",
        "mapping": {
            "37-43": ">50",
            "44-50": ">50",
        },
    },
    "LOW_CREDIT_STRESS": {
        "column": "credit_score_band",
        "mapping": {
            "700-739": "620-659",
            "660-699": "620-659",
        },
    },
}


OUTCOME_COLUMNS = {
    "delinquency": "pred_next_3m_delinquency_prob",
    "default": "pred_next_12m_default_prob",
    "prepayment": "pred_next_12m_prepayment_prob",
}


def _clip_probability(values: pd.Series | np.ndarray) -> np.ndarray:
    """Keep scenario probabilities in the valid [0, 1] interval."""
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _apply_multiplier(
    values: pd.Series | np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Apply a scenario multiplier while preserving probability bounds."""
    return _clip_probability(np.asarray(values, dtype=float) * multiplier)


def _validate_required_columns(
    data: pd.DataFrame,
    required: set[str],
) -> None:
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required scenario columns: {sorted(missing)}"
        )


def run_macro_scenario(
    predictions: pd.DataFrame,
    scenario_name: str,
    *,
    segment_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply an organizer-defined macro scenario to model predictions.

    The underlying model predictions are retained as the baseline.
    Scenario multipliers are applied independently to delinquency,
    default, and prepayment probabilities.
    """

    if scenario_name not in {
        "BASE",
        "ADVERSE_CREDIT",
        "HIGH_PREPAYMENT",
    }:
        raise ValueError(
            "scenario_name must be BASE, ADVERSE_CREDIT, "
            "or HIGH_PREPAYMENT."
        )

    required = {
        "loan_id",
        "reporting_month",
        *OUTCOME_COLUMNS.values(),
    }

    _validate_required_columns(predictions, required)

    config = SCENARIOS[scenario_name]

    output = predictions[
        [
            "loan_id",
            "reporting_month",
            *OUTCOME_COLUMNS.values(),
        ]
    ].copy()

    output["scenario"] = scenario_name

    output["baseline_delinquency_probability"] = output[
        OUTCOME_COLUMNS["delinquency"]
    ]

    output["baseline_default_probability"] = output[
        OUTCOME_COLUMNS["default"]
    ]

    output["baseline_prepayment_probability"] = output[
        OUTCOME_COLUMNS["prepayment"]
    ]

    output["scenario_delinquency_probability"] = _apply_multiplier(
        output["baseline_delinquency_probability"],
        config["delinquency_multiplier"],
    )

    output["scenario_default_probability"] = _apply_multiplier(
        output["baseline_default_probability"],
        config["default_multiplier"],
    )

    output["scenario_prepayment_probability"] = _apply_multiplier(
        output["baseline_prepayment_probability"],
        config["prepayment_multiplier"],
    )

    output["delinquency_change"] = (
        output["scenario_delinquency_probability"]
        - output["baseline_delinquency_probability"]
    )

    output["default_change"] = (
        output["scenario_default_probability"]
        - output["baseline_default_probability"]
    )

    output["prepayment_change"] = (
        output["scenario_prepayment_probability"]
        - output["baseline_prepayment_probability"]
    )

    # Optional segment columns are preserved for downstream impact analysis.
    for column in segment_columns or []:
        if column in predictions.columns:
            output[column] = predictions[column].values

    return output


def summarize_macro_scenario(
    scenario_output: pd.DataFrame,
) -> dict[str, Any]:
    """Create a portfolio-level summary for a macro scenario."""

    required = {
        "scenario",
        "baseline_delinquency_probability",
        "scenario_delinquency_probability",
        "baseline_default_probability",
        "scenario_default_probability",
        "baseline_prepayment_probability",
        "scenario_prepayment_probability",
    }

    _validate_required_columns(scenario_output, required)

    if scenario_output.empty:
        raise ValueError("scenario_output cannot be empty.")

    return {
        "scenario": str(scenario_output["scenario"].iloc[0]),
        "observations": int(len(scenario_output)),
        "baseline_delinquency_rate": float(
            scenario_output["baseline_delinquency_probability"].mean()
        ),
        "scenario_delinquency_rate": float(
            scenario_output["scenario_delinquency_probability"].mean()
        ),
        "delinquency_change": float(
            scenario_output["delinquency_change"].mean()
        ),
        "baseline_default_rate": float(
            scenario_output["baseline_default_probability"].mean()
        ),
        "scenario_default_rate": float(
            scenario_output["scenario_default_probability"].mean()
        ),
        "default_change": float(
            scenario_output["default_change"].mean()
        ),
        "baseline_prepayment_rate": float(
            scenario_output["baseline_prepayment_probability"].mean()
        ),
        "scenario_prepayment_rate": float(
            scenario_output["scenario_prepayment_probability"].mean()
        ),
        "prepayment_change": float(
            scenario_output["prepayment_change"].mean()
        ),
    }


def summarize_segment_impacts(
    scenario_output: pd.DataFrame,
    segment_column: str,
) -> pd.DataFrame:
    """
    Calculate scenario impact by portfolio segment.

    Returns segment size and average changes in delinquency,
    default, and prepayment probabilities.
    """

    if segment_column not in scenario_output.columns:
        raise ValueError(
            f"Missing segment column: {segment_column}"
        )

    grouped = (
        scenario_output
        .groupby(segment_column, dropna=False)
        .agg(
            observations=("loan_id", "size"),
            baseline_delinquency=(
                "baseline_delinquency_probability",
                "mean",
            ),
            scenario_delinquency=(
                "scenario_delinquency_probability",
                "mean",
            ),
            baseline_default=(
                "baseline_default_probability",
                "mean",
            ),
            scenario_default=(
                "scenario_default_probability",
                "mean",
            ),
            baseline_prepayment=(
                "baseline_prepayment_probability",
                "mean",
            ),
            scenario_prepayment=(
                "scenario_prepayment_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["delinquency_change"] = (
        grouped["scenario_delinquency"]
        - grouped["baseline_delinquency"]
    )

    grouped["default_change"] = (
        grouped["scenario_default"]
        - grouped["baseline_default"]
    )

    grouped["prepayment_change"] = (
        grouped["scenario_prepayment"]
        - grouped["baseline_prepayment"]
    )

    return grouped.sort_values(
        "default_change",
        ascending=False,
    ).reset_index(drop=True)


def run_all_macro_scenarios(
    predictions: pd.DataFrame,
    *,
    segment_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run BASE, ADVERSE_CREDIT and HIGH_PREPAYMENT scenarios."""

    return {
        name: run_macro_scenario(
            predictions,
            name,
            segment_columns=segment_columns,
        )
        for name in (
            "BASE",
            "ADVERSE_CREDIT",
            "HIGH_PREPAYMENT",
        )
    }


def run_scenario(
    data: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
    scenario_name: str,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Run the original controlled feature stress scenario.

    Retained for backwards compatibility with the existing test suite.
    """

    feature_scenarios = {
        name: config
        for name, config in SCENARIOS.items()
        if "column" in config
    }

    if scenario_name not in feature_scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = feature_scenarios[scenario_name]

    baseline = generate_predictions(
        data,
        pipeline,
        feature_columns,
        threshold=threshold,
    )

    scenario_data = data.copy()
    column = scenario["column"]

    if column not in scenario_data.columns:
        raise ValueError(
            f"Missing scenario column: {column}"
        )

    scenario_data[column] = (
        scenario_data[column]
        .replace(scenario["mapping"])
    )

    scenario_features = build_features(scenario_data)

    probabilities = pipeline.predict_proba(
        scenario_features[feature_columns]
    )[:, 1]

    scenario_output = baseline[
        [
            "loan_id",
            "reporting_month",
            "predicted_probability",
            "risk_flag",
        ]
    ].copy()

    scenario_output = scenario_output.rename(
        columns={
            "predicted_probability": "baseline_probability",
            "risk_flag": "baseline_risk_flag",
        }
    )

    scenario_output["scenario_probability"] = probabilities

    scenario_output["scenario_risk_flag"] = (
        probabilities >= threshold
    ).astype(int)

    scenario_output["probability_change"] = (
        scenario_output["scenario_probability"]
        - scenario_output["baseline_probability"]
    )

    scenario_output["risk_flag_change"] = (
        scenario_output["scenario_risk_flag"]
        - scenario_output["baseline_risk_flag"]
    )

    scenario_output["scenario"] = scenario_name

    return scenario_output


def summarize_scenario(
    scenario_output: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize the original controlled feature stress scenario."""

    if scenario_output.empty:
        raise ValueError("scenario_output cannot be empty.")

    required = {
        "scenario",
        "baseline_probability",
        "scenario_probability",
        "baseline_risk_flag",
        "scenario_risk_flag",
    }

    missing = required - set(scenario_output.columns)

    if missing:
        raise ValueError(
            f"Missing required scenario columns: {sorted(missing)}"
        )

    return {
        "scenario": str(scenario_output["scenario"].iloc[0]),
        "observations": int(len(scenario_output)),
        "baseline_average_probability": float(
            scenario_output["baseline_probability"].mean()
        ),
        "scenario_average_probability": float(
            scenario_output["scenario_probability"].mean()
        ),
        "average_probability_change": float(
            (
                scenario_output["scenario_probability"]
                - scenario_output["baseline_probability"]
            ).mean()
        ),
        "baseline_flagged_observations": int(
            scenario_output["baseline_risk_flag"].sum()
        ),
        "scenario_flagged_observations": int(
            scenario_output["scenario_risk_flag"].sum()
        ),
        "flagged_observation_change": int(
            scenario_output["scenario_risk_flag"].sum()
            - scenario_output["baseline_risk_flag"].sum()
        ),
    }