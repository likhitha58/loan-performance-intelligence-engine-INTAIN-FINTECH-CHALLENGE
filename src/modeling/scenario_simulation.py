from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_intelligence.features import build_features
from src.modeling.prediction import generate_predictions


SCENARIOS = {
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


def run_scenario(
    data: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
    scenario_name: str,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Run a controlled stress scenario and return baseline/scenario comparison."""

    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {scenario_name}"
        )

    scenario = SCENARIOS[scenario_name]

    baseline = generate_predictions(
        data,
        pipeline,
        feature_columns,
        threshold=threshold,
    )

    scenario_data = data.copy()

    column = scenario["column"]

    scenario_data[column] = (
        scenario_data[column]
        .replace(scenario["mapping"])
    )

    scenario_features = build_features(
        scenario_data
    )

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
    """Summarize the portfolio impact of a scenario."""

    if scenario_output.empty:
        raise ValueError(
            "scenario_output cannot be empty."
        )

    required = {
        "scenario",
        "baseline_probability",
        "scenario_probability",
        "baseline_risk_flag",
        "scenario_risk_flag",
    }

    missing = required - set(
        scenario_output.columns
    )

    if missing:
        raise ValueError(
            f"Missing required scenario columns: "
            f"{sorted(missing)}"
        )

    return {
        "scenario": str(
            scenario_output["scenario"].iloc[0]
        ),
        "observations": int(
            len(scenario_output)
        ),
        "baseline_average_probability": float(
            scenario_output[
                "baseline_probability"
            ].mean()
        ),
        "scenario_average_probability": float(
            scenario_output[
                "scenario_probability"
            ].mean()
        ),
        "average_probability_change": float(
            (
                scenario_output[
                    "scenario_probability"
                ]
                - scenario_output[
                    "baseline_probability"
                ]
            ).mean()
        ),
        "baseline_flagged_observations": int(
            scenario_output[
                "baseline_risk_flag"
            ].sum()
        ),
        "scenario_flagged_observations": int(
            scenario_output[
                "scenario_risk_flag"
            ].sum()
        ),
        "flagged_observation_change": int(
            scenario_output[
                "scenario_risk_flag"
            ].sum()
            - scenario_output[
                "baseline_risk_flag"
            ].sum()
        ),
    }