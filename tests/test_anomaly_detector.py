import pandas as pd

from src.data_intelligence.anomaly_detector import (
    build_anomaly_report,
)


def _train_record(**overrides):
    """Create a minimal valid monthly-performance record."""
    record = {
        "loan_id": "L100001",
        "reporting_month": "2024-01-01",
        "origination_month": "2023-01-01",
        "original_balance": 100000.0,
        "current_balance": 95000.0,
        "interest_rate": 5.0,
        "loan_age_months": 13,
        "remaining_term_months": 347,
        "current_status": "CURRENT",
        "days_past_due": 0,
        "document_status": "COMPLETE",
        "servicer_name": "SVC_ALPHA",
        "credit_score_band": "700_PLUS",
        "loss_severity_band": None,
        "prepayment_flag": 0,
        "default_flag": 0,
        "last_updated_at": "2024-01-05",
    }

    record.update(overrides)

    return pd.DataFrame([record])


def _static():
    """Create the corresponding static loan record."""
    return pd.DataFrame(
        [
            {
                "loan_id": "L100001",
                "origination_month": "2023-01-01",
                "original_balance": 100000.0,
                "original_term_months": 360,
                "interest_rate": 5.0,
                "credit_score_band": "700_PLUS",
                "servicer_name": "SVC_ALPHA",
                "property_type": "SF",
                "occupancy_status": "OWNER",
                "original_ltv": 80.0,
                "original_dti": 30.0,
                "state": "CA",
                "documentation_type": "FULL",
            }
        ]
    )


def test_clean_record_has_no_anomaly():
    train = _train_record()

    report = build_anomaly_report(
        train,
        _static(),
    )

    assert len(report) == 1
    assert bool(report.loc[0, "anomaly_detected"]) is False
    assert report.loc[0, "severity"] == "NONE"
    assert report.loc[0, "anomaly_score"] == 0.0
    assert report.loc[0, "anomaly_types"] == ""


def test_detects_impossible_loan_age():
    train = _train_record(
        loan_age_months=-5,
    )

    report = build_anomaly_report(
        train,
        _static(),
    )

    assert bool(report.loc[0, "anomaly_detected"]) is True
    assert "impossible_loan_age" in report.loc[
        0,
        "anomaly_types",
    ]


def test_detects_missing_document_status():
    train = _train_record(
        document_status=None,
    )

    report = build_anomaly_report(
        train,
        _static(),
    )

    assert bool(report.loc[0, "anomaly_detected"]) is True
    assert "missing_document_status" in report.loc[
        0,
        "anomaly_types",
    ]


def test_detects_multiple_anomalies():
    train = _train_record(
        loan_age_months=-5,
        remaining_term_months=-10,
        document_status=None,
    )

    report = build_anomaly_report(
        train,
        _static(),
    )

    assert bool(report.loc[0, "anomaly_detected"]) is True

    anomaly_types = report.loc[
        0,
        "anomaly_types",
    ]

    assert "impossible_loan_age" in anomaly_types
    assert "invalid_remaining_term" in anomaly_types
    assert "missing_document_status" in anomaly_types


def test_detects_balance_inconsistency():
    train = _train_record(
        original_balance=100000.0,
        current_balance=120000.0,
    )

    report = build_anomaly_report(
        train,
        _static(),
    )

    assert bool(report.loc[0, "anomaly_detected"]) is True
    assert "balance_inconsistency" in report.loc[
        0,
        "anomaly_types",
    ]


def test_detects_unexpected_missing_value():
    train = _train_record(
        current_balance=None,
    )

    report = build_anomaly_report(
        train,
        _static(),
    )

    assert bool(report.loc[0, "anomaly_detected"]) is True
    assert "unexpected_missing_value" in report.loc[
        0,
        "anomaly_types",
    ]


def test_detects_delinquency_status_inconsistency():
    train = _train_record(
        current_status="DELINQUENT",
        days_past_due=0,
    )

    report = build_anomaly_report(
        train,
        _static(),
    )

    assert bool(report.loc[0, "anomaly_detected"]) is True
    assert "delinquency_status_inconsistency" in report.loc[
        0,
        "anomaly_types",
    ]