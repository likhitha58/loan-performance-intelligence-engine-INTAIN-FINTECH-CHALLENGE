from __future__ import annotations

from typing import Dict

import pandas as pd


def calculate_missingness(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate missing-value statistics for every column
    in every dataset.

    Returns
    -------
    pd.DataFrame
        One row per dataset/column.
    """

    records = []

    for dataset_name, df in datasets.items():

        total_rows = len(df)

        for column in df.columns:

            missing_count = int(df[column].isna().sum())

            missing_pct = (
                missing_count / total_rows * 100
                if total_rows
                else 0.0
            )

            records.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "row_count": total_rows,
                    "missing_count": missing_count,
                    "missing_pct": round(
                        missing_pct,
                        4,
                    ),
                }
            )

    return pd.DataFrame(records)


def identify_missing_columns(
    missingness: pd.DataFrame,
    threshold_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Return columns whose missingness exceeds the
    specified percentage threshold.
    """

    return (
        missingness[
            missingness["missing_pct"] > threshold_pct
        ]
        .sort_values(
            ["dataset", "missing_pct"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def summarize_missingness(
    missingness: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one summary row per dataset.
    """

    summary = (
        missingness
        .groupby("dataset")
        .agg(
            columns=("column", "count"),
            columns_with_missing=(
                "missing_count",
                lambda x: int((x > 0).sum()),
            ),
            total_missing_values=(
                "missing_count",
                "sum",
            ),
        )
        .reset_index()
    )

    return summary


def compare_missingness(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str = "train",
    right_name: str = "test",
) -> pd.DataFrame:
    """
    Compare missingness percentages for columns shared
    between two datasets.

    Columns present in only one dataset are excluded because
    they cannot be compared as equivalent features.
    """

    shared_columns = sorted(
        set(left.columns).intersection(right.columns)
    )

    left_missing = (
        left[shared_columns]
        .isna()
        .mean()
        .mul(100)
        .rename(left_name)
    )

    right_missing = (
        right[shared_columns]
        .isna()
        .mean()
        .mul(100)
        .rename(right_name)
    )

    comparison = pd.concat(
        [left_missing, right_missing],
        axis=1,
    )

    comparison["difference_pct_points"] = (
        comparison[left_name]
        - comparison[right_name]
    ).abs()

    return (
        comparison
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values(
            "difference_pct_points",
            ascending=False,
        )
    )