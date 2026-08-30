import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.modeling.global_explainability import (
    build_feature_importance_table,
    summarize_global_importance,
    build_global_explainability_summary,
)


def _fitted_pipeline():
    X = pd.DataFrame(
        {
            "feature_a": [0, 1, 0, 1, 1, 0],
            "feature_b": [1, 1, 0, 0, 1, 0],
            "feature_c": [0, 0, 1, 1, 0, 1],
        }
    )

    y = np.array([0, 1, 0, 1, 1, 0])

    pipeline = Pipeline(
        [
            (
                "model",
                GradientBoostingClassifier(
                    random_state=42,
                ),
            )
        ]
    )

    pipeline.fit(X, y)

    return pipeline


def test_build_feature_importance_table():
    pipeline = _fitted_pipeline()

    result = build_feature_importance_table(
        pipeline,
        [
            "feature_a",
            "feature_b",
            "feature_c",
        ],
    )

    assert len(result) == 3
    assert list(result.columns) == [
        "rank",
        "feature",
        "importance",
        "importance_share",
    ]

    assert result["rank"].tolist() == [1, 2, 3]

    assert result["importance"].sum() == pytest.approx(
        1.0
    )


def test_importance_is_sorted():
    pipeline = _fitted_pipeline()

    result = build_feature_importance_table(
        pipeline,
        [
            "feature_a",
            "feature_b",
            "feature_c",
        ],
    )

    assert result["importance"].tolist() == sorted(
        result["importance"].tolist(),
        reverse=True,
    )


def test_top_n_summary():
    pipeline = _fitted_pipeline()

    importance = build_feature_importance_table(
        pipeline,
        [
            "feature_a",
            "feature_b",
            "feature_c",
        ],
    )

    result = summarize_global_importance(
        importance,
        top_n=2,
    )

    assert len(result) == 2


def test_global_summary_contains_model():
    pipeline = _fitted_pipeline()

    importance = build_feature_importance_table(
        pipeline,
        [
            "feature_a",
            "feature_b",
            "feature_c",
        ],
    )

    result = build_global_explainability_summary(
        "next_12m_default_flag",
        importance,
        top_n=2,
    )

    assert len(result) == 2
    assert set(result["model"]) == {
        "next_12m_default_flag"
    }


def test_empty_feature_columns_raises():
    pipeline = _fitted_pipeline()

    with pytest.raises(ValueError):
        build_feature_importance_table(
            pipeline,
            [],
        )


def test_missing_feature_importances_raises():
    class FakePipeline:
        named_steps = {}

        steps = [
            (
                "model",
                object(),
            )
        ]

    with pytest.raises(ValueError):
        build_feature_importance_table(
            FakePipeline(),
            ["feature_a"],
        )