from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


DATE_COLUMN_CANDIDATES = {
    "reporting_month",
    "origination_month",
    "last_updated_at",
    "update_date",
}


def profile_dataset(
    name: str,
    df: pd.DataFrame,
) -> dict:
    """
    Generate a dataset-level profile.

    Parameters
    ----------
    name:
        Logical dataset name.

    df:
        Dataset to profile.

    Returns
    -------
    dict
        Structured dataset profile.
    """

    date_ranges = {}

    for column in DATE_COLUMN_CANDIDATES:
        if column not in df.columns:
            continue

        values = pd.to_datetime(df[column], errors="coerce")

        valid = values.dropna()

        if not valid.empty:
            date_ranges[column] = {
                "min": str(valid.min()),
                "max": str(valid.max()),
            }

    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(
            df.memory_usage(deep=True).sum() / (1024**2),
            3,
        ),
        "date_ranges": date_ranges,
    }


def profile_columns(
    name: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate column-level profiling statistics.
    """

    records = []

    row_count = len(df)

    for column in df.columns:

        series = df[column]

        missing_count = int(series.isna().sum())

        missing_pct = (
            (missing_count / row_count) * 100
            if row_count
            else 0.0
        )

        unique_count = int(series.nunique(dropna=True))

        record = {
            "dataset": name,
            "column": column,
            "dtype": str(series.dtype),
            "rows": int(row_count),
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "unique_count": unique_count,
            "unique_pct": round(
                (unique_count / row_count) * 100
                if row_count
                else 0.0,
                4,
            ),
        }

        if pd.api.types.is_numeric_dtype(series):

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            record.update(
                {
                    "min": _safe_float(numeric.min()),
                    "max": _safe_float(numeric.max()),
                    "mean": _safe_float(numeric.mean()),
                    "median": _safe_float(numeric.median()),
                    "std": _safe_float(numeric.std()),
                }
            )

        elif pd.api.types.is_object_dtype(series):

            top_values = (
                series
                .value_counts(dropna=False)
                .head(10)
                .to_dict()
            )

            record["top_values"] = str(top_values)

        records.append(record)

    return pd.DataFrame(records)


def profile_datasets(
    datasets: Dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Profile all datasets.

    Returns
    -------
    dataset_summary:
        One row per dataset.

    column_summary:
        One row per column.
    """

    dataset_records = []
    column_records = []

    for name, df in datasets.items():

        dataset_records.append(
            profile_dataset(name, df)
        )

        column_profile = profile_columns(
            name,
            df,
        )

        column_records.append(column_profile)

    dataset_summary = pd.DataFrame(dataset_records)

    column_summary = pd.concat(
        column_records,
        ignore_index=True,
    )

    return dataset_summary, column_summary


def _safe_float(value):
    """
    Convert pandas/numpy numeric values into
    JSON-friendly Python floats.
    """

    if pd.isna(value):
        return None

    return float(value)