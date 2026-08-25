import pandas as pd

from src.data_intelligence.relationship_validator import (
    get_violation_rows,
    validate_relationships,
)


def make_static():
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "original_term_months": [360, 360],
        }
    )


def make_train():
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "month_index": [1, 2],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
            ],
            "origination_month": [
                "2024-01-01",
                "2024-01-01",
            ],
            "loan_age_months": [1, 2],
            "remaining_term_months": [359, 358],
            "original_balance": [200000, 200000],
            "current_balance": [199000, 198000],
            "interest_rate": [5.0, 5.0],
            "credit_score_band": ["700-739", "700-739"],
            "ltv_band": ["71-80", "71-80"],
            "dti_band": ["31-36", "31-36"],
            "state": ["OH", "OH"],
            "loan_purpose": ["PURCHASE", "PURCHASE"],
            "occupancy_type": ["PRIMARY", "PRIMARY"],
            "property_type": ["SINGLE_FAMILY", "SINGLE_FAMILY"],
            "servicer_name": ["SVC_ALPHA", "SVC_ALPHA"],
            "current_status": ["CURRENT", "CURRENT"],
            "days_past_due": [0, 0],
            "modification_flag": [0, 0],
            "prepayment_flag": [0, 0],
            "default_flag": [0, 0],
            "loss_severity_band": [None, None],
            "last_updated_at": [
                "2024-01-05",
                "2024-02-05",
            ],
            "source_system": ["SYNTH_CORE_LOS", "SYNTH_CORE_LOS"],
            "document_status": ["COMPLETE", "COMPLETE"],
            "exception_required": [0, 0],
            "exception_type": ["", ""],
        }
    )


def test_clean_data_has_no_relationship_violations():

    result = validate_relationships(
        make_train(),
        make_static(),
    )

    assert result["violation_count"].sum() == 0


def test_detects_loan_age_violation():

    train = make_train()

    train.loc[1, "loan_age_months"] = 20

    result = validate_relationships(
        train,
        make_static(),
    )

    r003 = result[
        result["rule_id"] == "R003"
    ].iloc[0]

    assert r003["violation_count"] == 1


def test_detects_delinquency_violation():

    train = make_train()

    train.loc[0, "current_status"] = "DELINQUENT"
    train.loc[0, "days_past_due"] = 0

    result = validate_relationships(
        train,
        make_static(),
    )

    r008 = result[
        result["rule_id"] == "R008"
    ].iloc[0]

    assert r008["violation_count"] == 1


def test_get_violation_rows():

    train = make_train()

    train.loc[0, "prepayment_flag"] = 1

    violations = get_violation_rows(
        train,
        make_static(),
        "R009",
    )

    assert len(violations) == 1
    assert violations.iloc[0]["loan_id"] == "L1"