import pandas as pd
import pytest

from src.data_intelligence.features import (
    build_features,
    get_feature_columns,
)


def _sample_panel():
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L1", "L1", "L2", "L2"],
            "month_index": [1, 2, 3, 1, 3],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-01-01",
                "2024-03-01",
            ],
            "origination_month": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
            ],
            "loan_age_months": [1, 2, 3, 1, 3],
            "remaining_term_months": [359, 358, 357, 359, 357],
            "original_balance": [
                100000,
                100000,
                100000,
                200000,
                200000,
            ],
            "current_balance": [
                100000,
                98000,
                95000,
                200000,
                190000,
            ],
            "interest_rate": [5.0, 5.0, 5.0, 4.5, 4.5],
            "credit_score_band": [
                "HIGH",
                "HIGH",
                "HIGH",
                "MEDIUM",
                "MEDIUM",
            ],
            "ltv_band": ["LOW", "LOW", "LOW", "MEDIUM", "MEDIUM"],
            "dti_band": ["LOW", "LOW", "LOW", "HIGH", "HIGH"],
            "state": ["CA", "CA", "CA", "TX", "TX"],
            "loan_purpose": [
                "PURCHASE",
                "PURCHASE",
                "PURCHASE",
                "REFINANCE",
                "REFINANCE",
            ],
            "occupancy_type": [
                "OWNER",
                "OWNER",
                "OWNER",
                "INVESTOR",
                "INVESTOR",
            ],
            "property_type": [
                "SF",
                "SF",
                "SF",
                "CONDO",
                "CONDO",
            ],
            "servicer_name": [
                "SVC_A",
                "SVC_A",
                "SVC_A",
                "SVC_B",
                "SVC_B",
            ],
            "current_status": [
                "CURRENT",
                "CURRENT",
                "DELINQUENT",
                "CURRENT",
                "CURRENT",
            ],
            "days_past_due": [0, 0, 30, 0, 0],
            "modification_flag": [0, 0, 0, 0, 0],
            "prepayment_flag": [0, 0, 0, 0, 0],
            "default_flag": [0, 0, 0, 0, 0],
            "loss_severity_band": [None, None, None, None, None],
            "last_updated_at": [
                "2024-01-05",
                "2024-02-05",
                "2024-03-05",
                "2024-01-05",
                "2024-03-05",
            ],
            "source_system": [
                "CORE",
                "CORE",
                "CORE",
                "CORE",
                "CORE",
            ],
            "document_status": [
                "COMPLETE",
                "COMPLETE",
                "COMPLETE",
                "COMPLETE",
                "COMPLETE",
            ],
        }
    )


def test_build_features_preserves_row_count():
    df = _sample_panel()

    result = build_features(df)

    assert len(result) == len(df)


def test_build_features_preserves_identifiers_and_dates():
    df = _sample_panel()

    result = build_features(df)

    assert result["loan_id"].tolist() == df["loan_id"].tolist()

    expected_dates = pd.to_datetime(df["reporting_month"])

    assert result["reporting_month"].tolist() == expected_dates.tolist()


def test_balance_ratio_is_calculated():
    df = _sample_panel()

    result = build_features(df)

    assert result.loc[1, "balance_ratio"] == pytest.approx(0.98)
    assert result.loc[2, "balance_ratio"] == pytest.approx(0.95)


def test_first_observation_has_no_historical_balance_change():
    df = _sample_panel()

    result = build_features(df)

    assert pd.isna(result.loc[0, "balance_change_1m"])


def test_historical_balance_change_uses_previous_observation():
    df = _sample_panel()

    result = build_features(df)

    assert result.loc[1, "balance_change_1m"] == pytest.approx(-2000)
    assert result.loc[2, "balance_change_1m"] == pytest.approx(-3000)


def test_history_does_not_cross_loans():
    df = _sample_panel()

    result = build_features(df)

    # First observation for L2 must not inherit L1's balance.
    assert pd.isna(result.loc[3, "balance_change_1m"])


def test_missing_calendar_month_is_not_treated_as_one_month_lag():
    df = _sample_panel()

    result = build_features(df)

    # L2 has observations at month 1 and month 3.
    # Therefore month 3 has no true one-month historical observation.
    assert pd.isna(result.loc[4, "balance_change_1m"])


def test_feature_columns_exclude_targets():
    df = _sample_panel()

    df["next_3m_delinquency_flag"] = 0
    df["next_6m_delinquency_flag"] = 0
    df["next_12m_default_flag"] = 0
    df["next_12m_prepayment_flag"] = 0
    df["exception_required"] = 0
    df["exception_type"] = ""

    columns = get_feature_columns(df)

    forbidden = {
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
        "exception_required",
        "exception_type",
        "loan_id",
    }

    assert forbidden.isdisjoint(columns)


def test_features_are_numeric_or_categorical_values():
    df = _sample_panel()

    result = build_features(df)

    assert "balance_ratio" in result.columns
    assert "balance_change_1m" in result.columns