import pandas as pd
import pytest

from src.modeling.split import temporal_split


def _panel():
    return pd.DataFrame(
        {
            "loan_id": [
                "L1",
                "L1",
                "L1",
                "L2",
                "L2",
                "L2",
            ],
            "reporting_month": [
                "2023-05-01",
                "2023-06-01",
                "2023-07-01",
                "2023-06-01",
                "2023-07-01",
                "2023-08-01",
            ],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )


def test_temporal_split_respects_cutoff():
    df = _panel()

    train, validation = temporal_split(
        df,
        validation_start="2023-07-01",
    )

    assert train["reporting_month"].max() == pd.Timestamp(
        "2023-06-01"
    )

    assert validation["reporting_month"].min() == pd.Timestamp(
        "2023-07-01"
    )


def test_temporal_split_preserves_all_rows():
    df = _panel()

    train, validation = temporal_split(
        df,
        validation_start="2023-07-01",
    )

    assert len(train) + len(validation) == len(df)


def test_temporal_split_does_not_randomize():
    df = _panel()

    train, validation = temporal_split(
        df,
        validation_start="2023-07-01",
    )

    assert train["value"].tolist() == [1, 2, 4]
    assert validation["value"].tolist() == [3, 5, 6]


def test_temporal_split_can_handle_string_dates():
    df = _panel()

    train, validation = temporal_split(
        df,
        validation_start="2023-07-01",
    )

    assert pd.api.types.is_datetime64_any_dtype(
        train["reporting_month"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        validation["reporting_month"]
    )


def test_missing_reporting_month_raises():
    df = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "value": [1],
        }
    )

    with pytest.raises(ValueError, match="reporting_month"):
        temporal_split(df)


def test_invalid_reporting_month_raises():
    df = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "reporting_month": ["not-a-date"],
        }
    )

    with pytest.raises(ValueError, match="invalid"):
        temporal_split(df)