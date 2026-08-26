from __future__ import annotations

import pandas as pd


def generate_risk_evidence(row: pd.Series) -> list[str]:
    """Generate transparent supporting risk signals for one observation."""

    evidence: list[str] = []

    if row.get("current_status") == "DEFAULT":
        evidence.append("CURRENT_DEFAULT")

    if row.get("current_status") == "DELINQUENT":
        evidence.append("CURRENT_DELINQUENCY")

    dpd_lag = row.get("dpd_lag_1m")
    if pd.notna(dpd_lag) and dpd_lag > 0:
        evidence.append("RECENT_DELINQUENCY")

    if row.get("status_change_flag") == 1:
        evidence.append("STATUS_CHANGE")

    if row.get("ltv_band") in {"81-90", "91-95", ">95"}:
        evidence.append("HIGH_LTV")

    if row.get("dti_band") in {"44-50", ">50"}:
        evidence.append("HIGH_DTI")

    if row.get("credit_score_band") in {"<620", "620-659"}:
        evidence.append("LOW_CREDIT_SCORE")

    if row.get("modification_flag") == 1:
        evidence.append("MODIFIED_LOAN")

    return evidence


def add_risk_evidence(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach transparent supporting risk evidence to predictions."""

    required = {
        "loan_id",
        "reporting_month",
    }

    missing_predictions = required - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing_predictions)}"
        )

    missing_features = required - set(features.columns)
    if missing_features:
        raise ValueError(
            f"Missing feature columns: {sorted(missing_features)}"
        )

    result = predictions.copy()

    feature_data = features.copy()

    result["_reporting_month_key"] = pd.to_datetime(
        result["reporting_month"]
    )
    feature_data["_reporting_month_key"] = pd.to_datetime(
        feature_data["reporting_month"]
    )

    evidence_columns = [
        "loan_id",
        "_reporting_month_key",
        "current_status",
        "dpd_lag_1m",
        "status_change_flag",
        "ltv_band",
        "dti_band",
        "credit_score_band",
        "modification_flag",
    ]

    result = result.merge(
        feature_data[evidence_columns],
        on=["loan_id", "_reporting_month_key"],
        how="left",
        validate="one_to_one",
    )

    result["risk_evidence"] = result.apply(
        generate_risk_evidence,
        axis=1,
    )

    result["evidence_count"] = result["risk_evidence"].str.len()
    result["evidence_category"] = result.apply(
    lambda row: (
        "MODEL_SIGNAL_ONLY"
        if row["risk_flag"] == 1 and row["evidence_count"] == 0
        else (
            "OBSERVABLE_RISK_SIGNALS"
            if row["risk_flag"] == 1
            else "NOT_FLAGGED"
        )
    ),
    axis=1,
    )

    return result.drop(
        columns=["_reporting_month_key"]
    )