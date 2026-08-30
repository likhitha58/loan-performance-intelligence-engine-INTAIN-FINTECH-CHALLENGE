from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


REQUIRED_IMPORTANCE_COLUMNS = {
    "feature",
    "importance",
}


def build_feature_importance_table(
    pipeline,
    feature_columns: list[str],
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    *,
    n_repeats: int = 3,
    sample_size: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a global feature-importance table for a fitted pipeline.

    Native feature_importances_ are used when available.

    For estimators such as HistGradientBoostingClassifier that do not
    expose native feature_importances_, permutation importance is
    calculated on a deterministic sample of the training data.

    Sampling keeps global explainability computationally practical
    without changing the fitted model or its predictions.
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

    # --------------------------------------------------------------
    # Native feature importance
    # --------------------------------------------------------------

    if hasattr(estimator, "feature_importances_"):
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

    # --------------------------------------------------------------
    # Permutation importance fallback
    # --------------------------------------------------------------

    else:
        if X is None or y is None:
            raise ValueError(
                "X and y are required when the final estimator "
                "does not expose feature_importances_."
            )

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than zero."
            )

        if n_repeats <= 0:
            raise ValueError(
                "n_repeats must be greater than zero."
            )

        missing = set(feature_columns) - set(X.columns)

        if missing:
            raise ValueError(
                "X is missing required feature columns: "
                f"{sorted(missing)}"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of observations."
            )

        X_used = X[feature_columns].copy()
        y_used = y.copy()

        # Deterministic sampling keeps runtime bounded while
        # preserving a representative global importance estimate.
        if len(X_used) > sample_size:
            sample_indices = (
                X_used.sample(
                    n=sample_size,
                    random_state=random_state,
                ).index
            )

            X_used = X_used.loc[sample_indices]
            y_used = y_used.loc[sample_indices]

        permutation = permutation_importance(
            pipeline,
            X_used,
            y_used,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="roc_auc",
            n_jobs=-1,
        )

        result = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": permutation.importances_mean,
            }
        )

    # --------------------------------------------------------------
    # Validate and format
    # --------------------------------------------------------------

    result["importance"] = pd.to_numeric(
        result["importance"],
        errors="coerce",
    )

    if result["importance"].isna().any():
        raise ValueError(
            "Feature importance contains invalid values."
        )

    result = result.sort_values(
        ["importance", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    total = result["importance"].sum()

    if total > 0:
        result["importance_share"] = (
            result["importance"] / total
        )
    else:
        result["importance_share"] = 0.0

    result["rank"] = result.index + 1

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