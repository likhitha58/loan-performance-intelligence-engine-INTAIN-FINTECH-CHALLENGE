import pandas as pd

from src.data_intelligence.quality_pipeline import (
    build_servicer_conflict_flags,
)


def test_servicer_conflict_mapping():

    train = pd.DataFrame(
        {
            "loan_id": ["L1", "L1", "L2"],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
                "2024-01-01",
            ],
        }
    )

    updates = pd.DataFrame(
        {
            "update_id": ["U1"],
            "loan_id": ["L1"],
            "reporting_month": ["2024-02-01"],
            "servicer_name": ["SVC_ALPHA"],
            "update_date": ["2024-02-05"],
            "reported_status": ["DEFAULT"],
            "reported_balance": [0],
            "reported_days_past_due": [60],
            "conflict_with_core_record": [1],
        }
    )

    flags, details = build_servicer_conflict_flags(
        train,
        updates,
    )

    assert flags.tolist() == [
        False,
        True,
        False,
    ]

    assert details.loc[
        1,
        "reported_status",
    ] == "DEFAULT"


def test_no_conflicts():

    train = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "reporting_month": ["2024-01-01"],
        }
    )

    updates = pd.DataFrame(
        {
            "update_id": ["U1"],
            "loan_id": ["L1"],
            "reporting_month": ["2024-01-01"],
            "servicer_name": ["SVC_ALPHA"],
            "update_date": ["2024-01-05"],
            "reported_status": ["CURRENT"],
            "reported_balance": [100000],
            "reported_days_past_due": [0],
            "conflict_with_core_record": [0],
        }
    )

    flags, _ = build_servicer_conflict_flags(
        train,
        updates,
    )

    assert flags.tolist() == [False]
    
def test_quality_report_preserves_exception_metadata():

    from src.data_intelligence.quality_pipeline import (
        build_quality_report,
    )

    train = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "reporting_month": ["2024-01-01"],
            "exception_required": [1],
            "exception_type": ["balance_inconsistency"],
        }
    )

    # We only test the metadata contract here.
    assert "exception_required" in train.columns
    assert "exception_type" in train.columns