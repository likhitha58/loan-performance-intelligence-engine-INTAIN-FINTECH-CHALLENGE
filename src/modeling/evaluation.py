from __future__ import annotations

import pandas as pd

from src.modeling.baseline import run_baseline
from src.modeling.split import temporal_split


DEFAULT_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]


def evaluate_target(
    train: pd.DataFrame,
    target: str,
) -> dict[str, float | str | int]:
    """Evaluate the baseline model for one binary target."""

    result = run_baseline(
        train,
        target=target,
    )

    # Use the same chronological validation period as the baseline.
    _, validation = temporal_split(train)

    validation_rate = (
        validation[target]
        .astype(int)
        .mean()
    )

    # Constant-probability classifier has PR-AUC equal
    # to the positive prevalence.
    naive_pr_auc = validation_rate

    lift = (
        result.pr_auc / naive_pr_auc
        if naive_pr_auc > 0
        else float("nan")
    )

    return {
        "target": result.target,
        "train_rows": result.train_rows,
        "validation_rows": result.validation_rows,
        "positive_rate": validation_rate,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "naive_pr_auc": naive_pr_auc,
        "pr_auc_lift": lift,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
    }


def evaluate_targets(
    train: pd.DataFrame,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate the baseline model across multiple targets."""

    if targets is None:
        targets = DEFAULT_TARGETS

    results = [
        evaluate_target(
            train,
            target,
        )
        for target in targets
    ]

    return (
        pd.DataFrame(results)
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )