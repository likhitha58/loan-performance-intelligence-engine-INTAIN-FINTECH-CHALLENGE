from __future__ import annotations

from typing import Dict

import pandas as pd


DEFAULT_EXCLUDED_COLUMNS = {
    "loan_id",
    "month_index",
    "exception_required",
    "modification_flag",
    "prepayment_flag",
    "default_flag",
}


def calculate_iqr_outliers(
    datasets: Dict[str, pd.DataFrame],
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    """
    Detect statistical outliers in numeric columns using
    the 1.5 * IQR rule.

    The function reports outliers but does not modify data.
    """

    excluded = (
        DEFAULT_EXCLUDED_COLUMNS
        if excluded_columns is None
        else excluded_columns
    )

    records = []

    for dataset_name, df in datasets.items():

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:

            if column in excluded:
                continue

            series = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            if iqr == 0:
                outlier_count = 0
                lower_bound = q1
                upper_bound = q3
            else:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_count = int(
                    (
                        (series < lower_bound)
                        | (series > upper_bound)
                    ).sum()
                )

            total_count = len(series)

            records.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "non_missing_count": total_count,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_count": outlier_count,
                    "outlier_pct": round(
                        outlier_count / total_count * 100,
                        4,
                    ),
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["dataset", "outlier_pct"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def summarize_outliers(
    outliers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce a dataset-level outlier summary.
    """

    if outliers.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "numeric_columns",
                "columns_with_outliers",
                "total_outlier_values",
            ]
        )

    return (
        outliers
        .groupby("dataset")
        .agg(
            numeric_columns=(
                "column",
                "count",
            ),
            columns_with_outliers=(
                "outlier_count",
                lambda x: int((x > 0).sum()),
            ),
            total_outlier_values=(
                "outlier_count",
                "sum",
            ),
        )
        .reset_index()
    )


def get_top_outliers(
    outliers: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the columns with the highest outlier rates.
    """

    return (
        outliers
        .sort_values(
            "outlier_pct",
            ascending=False,
        )
        .head(n)
        .reset_index(drop=True)
    )