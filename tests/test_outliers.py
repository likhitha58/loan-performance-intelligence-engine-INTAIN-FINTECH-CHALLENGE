import pandas as pd

from src.data_intelligence.outliers import (
    calculate_iqr_outliers,
    get_top_outliers,
    summarize_outliers,
)


def test_calculate_iqr_outliers():

    datasets = {
        "train": pd.DataFrame(
            {
                "loan_id": [
                    "L1",
                    "L2",
                    "L3",
                    "L4",
                    "L5",
                ],
                "balance": [
                    100.0,
                    110.0,
                    105.0,
                    108.0,
                    1000.0,
                ],
                "default_flag": [
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
            }
        )
    }

    result = calculate_iqr_outliers(datasets)

    balance = result[
        result["column"] == "balance"
    ].iloc[0]

    assert balance["outlier_count"] >= 1


def test_excluded_columns_are_not_profiled():

    datasets = {
        "train": pd.DataFrame(
            {
                "loan_id": ["L1", "L2"],
                "month_index": [1, 2],
                "default_flag": [0, 1],
                "balance": [100.0, 200.0],
            }
        )
    }

    result = calculate_iqr_outliers(datasets)

    assert "loan_id" not in set(result["column"])
    assert "month_index" not in set(result["column"])
    assert "default_flag" not in set(result["column"])


def test_summarize_outliers():

    outliers = pd.DataFrame(
        {
            "dataset": ["train", "train"],
            "column": ["balance", "rate"],
            "outlier_count": [10, 5],
            "outlier_pct": [10.0, 5.0],
        }
    )

    result = summarize_outliers(outliers)

    row = result.iloc[0]

    assert row["numeric_columns"] == 2
    assert row["columns_with_outliers"] == 2
    assert row["total_outlier_values"] == 15


def test_get_top_outliers():

    outliers = pd.DataFrame(
        {
            "dataset": ["train", "train", "train"],
            "column": ["a", "b", "c"],
            "outlier_count": [1, 20, 5],
            "outlier_pct": [1.0, 20.0, 5.0],
        }
    )

    result = get_top_outliers(outliers, n=2)

    assert list(result["column"]) == ["b", "c"]