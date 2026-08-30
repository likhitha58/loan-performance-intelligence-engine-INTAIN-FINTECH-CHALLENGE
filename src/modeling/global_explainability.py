from __future__ import annotations

import pandas as pd


REQUIRED_IMPORTANCE_COLUMNS = {
    "feature",
    "importance",
}


def build_feature_importance_table(
    pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Extract global feature importance from a fitted tree-based pipeline.

    The function supports pipelines whose final estimator exposes
    feature_importances_.
    """

    if pipeline is None:
        raise ValueError("pipeline cannot be None.")

    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")

    if not hasattr(pipeline, "named_steps"):
        raise ValueError(
            "pipeline must be a fitted sklearn Pipeline."
        )

    estimator = pipeline.steps[-1][1]

    if not hasattr(estimator, "feature_importances_"):
        raise ValueError(
            "The final estimator does not expose feature_importances_."
        )

    importances = estimator.feature_importances_

    if len(importances) != len(feature_columns):
        raise ValueError(
            "Number of feature importances does not match "
            "feature_columns."
        )

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )

    result["importance"] = pd.to_numeric(
        result["importance"],
        errors="coerce",
    )

    if result["importance"].isna().any():
        raise ValueError(
            "Feature importance contains invalid values."
        )

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    total = result["importance"].sum()

    if total > 0:
        result["importance_share"] = (
            result["importance"] / total
        )
    else:
        result["importance_share"] = 0.0

    result["rank"] = (
        result.index + 1
    )

    return result[
        [
            "rank",
            "feature",
            "importance",
            "importance_share",
        ]
    ]


def summarize_global_importance(
    importance_table: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top globally important model features."""

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    missing = (
        REQUIRED_IMPORTANCE_COLUMNS
        - set(importance_table.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if importance_table.empty:
        return importance_table.copy()

    return (
        importance_table
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def build_global_explainability_summary(
    model_name: str,
    importance_table: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Create a report-ready global explainability table.
    """

    if not model_name:
        raise ValueError(
            "model_name cannot be empty."
        )

    top_features = summarize_global_importance(
        importance_table,
        top_n=top_n,
    ).copy()

    top_features.insert(
        0,
        "model",
        model_name,
    )

    return top_features