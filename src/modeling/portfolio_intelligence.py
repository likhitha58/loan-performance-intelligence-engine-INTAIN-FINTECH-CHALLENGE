from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "loan_id",
    "reporting_month",
    "predicted_probability",
    "risk_flag",
    "risk_tier",
    "action_priority",
}


def build_portfolio_summary(
    risk_output: pd.DataFrame,
) -> dict[str, float | int]:
    """Build high-level portfolio risk statistics."""

    missing = REQUIRED_COLUMNS - set(risk_output.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return {
        "total_observations": int(len(risk_output)),
        "unique_loans": int(
            risk_output["loan_id"].nunique()
        ),
        "flagged_observations": int(
            risk_output["risk_flag"].sum()
        ),
        "flagged_rate": float(
            risk_output["risk_flag"].mean()
        ),
        "average_predicted_probability": float(
            risk_output["predicted_probability"].mean()
        ),
    }


def build_risk_tier_summary(
    risk_output: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize observations by risk tier."""

    summary = (
        risk_output
        .groupby("risk_tier")
        .agg(
            observations=("loan_id", "size"),
            unique_loans=("loan_id", "nunique"),
            average_probability=(
                "predicted_probability",
                "mean",
            ),
            flagged_observations=(
                "risk_flag",
                "sum",
            ),
        )
        .reset_index()
    )

    total = len(risk_output)

    summary["observation_share"] = (
        summary["observations"] / total
    )

    return summary.sort_values(
        "average_probability",
        ascending=False,
    ).reset_index(drop=True)


def build_monthly_risk_trend(
    risk_output: pd.DataFrame,
) -> pd.DataFrame:
    """Build monthly portfolio risk trend."""

    result = risk_output.copy()

    result["reporting_month"] = pd.to_datetime(
        result["reporting_month"]
    )

    trend = (
        result
        .groupby("reporting_month")
        .agg(
            observations=("loan_id", "size"),
            unique_loans=("loan_id", "nunique"),
            flagged_observations=(
                "risk_flag",
                "sum",
            ),
            average_probability=(
                "predicted_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    trend["flagged_rate"] = (
        trend["flagged_observations"]
        / trend["observations"]
    )

    return trend.sort_values(
        "reporting_month"
    ).reset_index(drop=True)


def build_dimension_risk_summary(
    risk_output: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    """
    Summarize risk concentration across a categorical dimension.
    """

    if dimension not in risk_output.columns:
        raise ValueError(
            f"Unknown dimension: {dimension}"
        )

    result = (
        risk_output
        .groupby(dimension, dropna=False)
        .agg(
            observations=("loan_id", "size"),
            flagged_observations=(
                "risk_flag",
                "sum",
            ),
            average_probability=(
                "predicted_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    result["flagged_rate"] = (
        result["flagged_observations"]
        / result["observations"]
    )

    return result.sort_values(
        "flagged_rate",
        ascending=False,
    ).reset_index(drop=True)
    
    
def build_portfolio_intelligence_report(
    risk_output: pd.DataFrame,
    feature_data: pd.DataFrame,
    dimensions: tuple[str, ...] = (
        "current_status",
        "credit_score_band",
        "ltv_band",
        "dti_band",
    ),
) -> dict[str, object]:
    """
    Build the complete portfolio-level risk intelligence report.

    Risk predictions and portfolio dimensions are kept separate:
    risk_output contains model/risk decisions, while feature_data
    provides observable borrower dimensions for concentration analysis.
    """

    required_keys = {
        "loan_id",
        "reporting_month",
    }

    missing_risk = required_keys - set(risk_output.columns)

    if missing_risk:
        raise ValueError(
            f"Missing required risk columns: "
            f"{sorted(missing_risk)}"
        )

    missing_features = set(dimensions) - set(feature_data.columns)

    if missing_features:
        raise ValueError(
            f"Missing required feature dimensions: "
            f"{sorted(missing_features)}"
        )

    if len(risk_output) != len(feature_data):
        raise ValueError(
            "risk_output and feature_data must have "
            "the same number of rows."
        )

    combined = risk_output.copy()

    for dimension in dimensions:
        combined[dimension] = feature_data[
            dimension
        ].to_numpy()

    report = {
        "portfolio_summary": build_portfolio_summary(
            risk_output
        ),
        "risk_tier_summary": build_risk_tier_summary(
            risk_output
        ),
        "monthly_risk_trend": build_monthly_risk_trend(
            risk_output
        ),
        "dimension_summaries": {},
    }

    for dimension in dimensions:
        report["dimension_summaries"][dimension] = (
            build_dimension_risk_summary(
                combined,
                dimension,
            )
        )

    return report