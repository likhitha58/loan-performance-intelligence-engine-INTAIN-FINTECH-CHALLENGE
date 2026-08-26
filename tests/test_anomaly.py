import pandas as pd
import pytest

from src.modeling.anomaly import (
    detect_anomalies,
    fit_anomaly_model,
    score_anomalies,
)


def _sample_data() -> pd.DataFrame:
    rows = []

    for i in range(50):
        rows.append(
            {
                "loan_id": f"L{i:04d}",
                "reporting_month": "2024-01-01",
                "balance": 100000 + i * 100,
                "ltv": 60 + (i % 20),
                "dti": 20 + (i % 15),
                "credit_score": 700 + (i % 50),
            }
        )

    # Deliberate extreme observation.
    rows[-1]["balance"] = 10000000
    rows[-1]["ltv"] = 500
    rows[-1]["dti"] = 200
    rows[-1]["credit_score"] = 300

    return pd.DataFrame(rows)


def test_fit_anomaly_model():
    data = _sample_data()

    model, features = fit_anomaly_model(data)

    assert model is not None
    assert features


def test_score_anomalies_output():
    data = _sample_data()

    model, features = fit_anomaly_model(data)

    result = score_anomalies(
        data,
        model,
        features,
    )

    assert len(result) == len(data)
    assert "anomaly_score" in result.columns
    assert "exception_type" in result.columns

    assert result["anomaly_score"].between(
        0,
        1,
    ).all()


def test_detect_anomalies():
    result = detect_anomalies(
        _sample_data()
    )

    assert len(result) == 50
    assert result["anomaly_score"].between(
        0,
        1,
    ).all()


def test_invalid_contamination():
    with pytest.raises(ValueError):
        fit_anomaly_model(
            _sample_data(),
            contamination=0.8,
        )


def test_missing_required_column():
    data = _sample_data().drop(
        columns=["loan_id"]
    )

    with pytest.raises(ValueError):
        fit_anomaly_model(data)