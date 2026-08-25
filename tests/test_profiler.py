import pandas as pd

from src.data_intelligence.profiler import (
    profile_columns,
    profile_dataset,
    profile_datasets,
)


def test_profile_dataset():

    df = pd.DataFrame(
        {
            "loan_id": ["L1", "L2", "L3"],
            "current_balance": [100.0, 200.0, 300.0],
            "status": ["CURRENT", "CURRENT", "DELINQUENT"],
        }
    )

    result = profile_dataset("test", df)

    assert result["dataset"] == "test"
    assert result["rows"] == 3
    assert result["columns"] == 3
    assert result["duplicate_rows"] == 0


def test_profile_columns():

    df = pd.DataFrame(
        {
            "balance": [100.0, 200.0, None],
            "status": ["CURRENT", "CURRENT", "DELINQUENT"],
        }
    )

    result = profile_columns("test", df)

    assert len(result) == 2

    balance = result[
        result["column"] == "balance"
    ].iloc[0]

    assert balance["missing_count"] == 1
    assert balance["missing_pct"] == 33.3333


def test_profile_datasets():

    datasets = {
        "train": pd.DataFrame(
            {
                "loan_id": ["L1", "L2"],
                "balance": [100.0, 200.0],
            }
        ),
        "test": pd.DataFrame(
            {
                "loan_id": ["L3"],
                "balance": [300.0],
            }
        ),
    }

    dataset_summary, column_summary = profile_datasets(
        datasets
    )

    assert len(dataset_summary) == 2
    assert len(column_summary) == 4