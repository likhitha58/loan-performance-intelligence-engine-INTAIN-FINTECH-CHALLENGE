import pandas as pd

from src.data_intelligence.missingness import (
    calculate_missingness,
    compare_missingness,
    identify_missing_columns,
    summarize_missingness,
)


def test_calculate_missingness():

    datasets = {
        "train": pd.DataFrame(
            {
                "loan_id": ["L1", "L2", "L3"],
                "balance": [100.0, None, 300.0],
                "status": ["CURRENT", None, "DEFAULT"],
            }
        )
    }

    result = calculate_missingness(datasets)

    assert len(result) == 3

    balance = result[
        result["column"] == "balance"
    ].iloc[0]

    assert balance["missing_count"] == 1
    assert balance["missing_pct"] == 33.3333


def test_identify_missing_columns():

    missingness = pd.DataFrame(
        {
            "dataset": ["train", "train"],
            "column": ["balance", "status"],
            "missing_count": [10, 0],
            "missing_pct": [10.0, 0.0],
        }
    )

    result = identify_missing_columns(
        missingness,
        threshold_pct=0,
    )

    assert len(result) == 1
    assert result.iloc[0]["column"] == "balance"


def test_summarize_missingness():

    missingness = pd.DataFrame(
        {
            "dataset": ["train", "train", "train"],
            "column": ["a", "b", "c"],
            "missing_count": [10, 0, 5],
        }
    )

    result = summarize_missingness(missingness)

    row = result.iloc[0]

    assert row["columns"] == 3
    assert row["columns_with_missing"] == 2
    assert row["total_missing_values"] == 15


def test_compare_missingness():

    train = pd.DataFrame(
        {
            "balance": [100, None, 300],
            "status": ["CURRENT", "CURRENT", "DEFAULT"],
        }
    )

    test = pd.DataFrame(
        {
            "balance": [100, 200],
            "status": ["CURRENT", None],
        }
    )

    result = compare_missingness(
        train,
        test,
    )

    assert set(result["column"]) == {
        "balance",
        "status",
    }

    balance = result[
        result["column"] == "balance"
    ].iloc[0]

    assert balance["train"] > balance["test"]