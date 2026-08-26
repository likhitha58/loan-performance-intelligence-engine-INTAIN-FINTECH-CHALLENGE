from pathlib import Path

from src.data_intelligence.features import build_features
from src.data_intelligence.loader import load_data_pack
from src.modeling.anomaly import detect_anomalies
from src.modeling.gradient_boosting import fit_gradient_boosting
from src.modeling.prediction import generate_predictions
from src.modeling.risk_intelligence import add_risk_evidence
from src.modeling.risk_intelligence_output import (
    build_risk_intelligence_output,
)
from src.modeling.transition import (
    fit_transition_model,
    predict_next_state,
)
from src.submission import build_submission, write_submission


TARGETS = {
    "next_3m_delinquency_flag":
        "pred_next_3m_delinquency_prob",
    "next_6m_delinquency_flag":
        "pred_next_6m_delinquency_prob",
    "next_12m_default_flag":
        "pred_next_12m_default_prob",
    "next_12m_prepayment_flag":
        "pred_next_12m_prepayment_prob",
}


def main():
    data = load_data_pack()
    train = data["train"]
    test = data["test"]

    prediction_data = test[
        ["loan_id", "reporting_month"]
    ].copy()

    for target, output_column in TARGETS.items():
        print(f"Training {target}...")

        pipeline, features = fit_gradient_boosting(
            train,
            target=target,
        )

        predictions = generate_predictions(
            test,
            pipeline,
            features,
            threshold=0.20,
        )

        prediction_data[output_column] = (
            predictions["predicted_probability"].to_numpy()
        )

    print("Training next-state model...")

    transition_pipeline, transition_features, metrics = (
        fit_transition_model(train)
    )

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")

    transition_predictions = predict_next_state(
        test,
        transition_pipeline,
        transition_features,
    )

    transition_data = test[
        ["loan_id", "reporting_month"]
    ].copy()

    transition_data["pred_next_state"] = (
        transition_predictions[
            "pred_next_state"
        ].to_numpy()
    )

    transition_data["next_state_confidence"] = (
        transition_predictions[
            "next_state_confidence"
        ].to_numpy()
    )

    # Anomaly detection.
    print("Detecting anomalies...")

    anomaly_data = detect_anomalies(test)

    print(
        "  Anomalies:",
        (anomaly_data["exception_type"] != "").sum(),
    )

    # Grounded risk evidence.
    print("Building risk evidence...")

    test_features = build_features(test)

    risk_predictions = prediction_data[
        [
            "loan_id",
            "reporting_month",
            "pred_next_12m_default_prob",
        ]
    ].copy()

    risk_predictions = risk_predictions.rename(
        columns={
            "pred_next_12m_default_prob":
                "predicted_probability"
        }
    )

    risk_predictions["risk_flag"] = (
        risk_predictions["predicted_probability"]
        >= 0.20
    )

    risk_predictions["threshold"] = 0.20

    risk_evidence = add_risk_evidence(
        risk_predictions,
        test_features,
    )

    risk_output = build_risk_intelligence_output(
        risk_evidence
    )

    risk_output["reasons"] = risk_output[
        "risk_evidence"
    ].apply(
        lambda values: (
            values
            if isinstance(values, list)
            else []
        )
    )

    risk_output = risk_output[
        [
            "loan_id",
            "reporting_month",
            "reasons",
            "action_priority",
        ]
    ]

    submission = build_submission(
        prediction_data,
        transition_data=transition_data,
        anomaly_data=anomaly_data,
        explanation_data=risk_output,
    )

    output_path = Path(
        "submission/submission.csv"
    )

    write_submission(
        submission,
        output_path,
    )

    print()
    print("Submission generated successfully.")
    print(f"Path: {output_path}")
    print(f"Rows: {len(submission)}")
    print(f"Columns: {len(submission.columns)}")


if __name__ == "__main__":
    main()