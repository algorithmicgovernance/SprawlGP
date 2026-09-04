"""Tests for binary conformal prediction-set coverage diagnostics."""

import numpy as np
import pandas as pd
from src.models.evaluation.prediction_sets import (
    build_mondrian_prediction_sets,
    build_prediction_sets,
    conformal_quantile,
    nonconformity_scores,
    temporal_mondrian_prediction_set_coverage,
    temporal_prediction_set_coverage,
)


def test_nonconformity_scores_follow_binary_definition() -> None:
    """s(x,0)=p and s(x,1)=1-p for binary labels."""
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.6, 0.3])
    scores = nonconformity_scores(y, p)
    np.testing.assert_allclose(scores, np.array([0.1, 0.2, 0.6, 0.7]))


def test_prediction_sets_include_expected_labels() -> None:
    """Prediction sets should reflect thresholded class nonconformity."""
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    sets = build_prediction_sets(probability, q_hat=0.35)

    assert list(sets["prediction_set"]) == ["{0}", "{}", "{}", "{1}"]
    assert list(sets["set_size"]) == [1.0, 0.0, 0.0, 1.0]


def test_temporal_coverage_skips_first_fold_and_returns_metrics() -> None:
    """Fold 1 has no prior calibration fold and must be skipped."""
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "probability_raw": [0.2, 0.7, 0.1, 0.9, 0.3, 0.8],
            "fold": [1, 1, 2, 2, 3, 3],
        }
    )

    coverage = temporal_prediction_set_coverage(
        frame=frame,
        target_column="target",
        probability_column="probability_raw",
        fold_column="fold",
        target_coverage=0.80,
    )

    assert list(coverage["fold"]) == [2, 3]
    assert list(coverage["calibration_rows"]) == [2, 4]
    assert list(coverage["evaluation_rows"]) == [2, 2]
    assert np.isfinite(coverage["empirical_coverage"].to_numpy()).all()


def test_conformal_quantile_matches_rank_rule() -> None:
    """Finite-sample conformal quantile should use ceil((n+1)*(1-alpha))."""
    scores = np.array([0.1, 0.4, 0.2, 0.5, 0.3])
    q_hat = conformal_quantile(scores, alpha=0.2)
    assert np.isclose(q_hat, 0.5)


def test_mondrian_sets_handle_singleton_empty_and_both_labels() -> None:
    sets_with_both = build_mondrian_prediction_sets(
        probability=np.array([0.1, 0.5, 0.9]),
        q_hat_0=0.6,
        q_hat_1=0.6,
    )
    sets_with_empty = build_mondrian_prediction_sets(
        probability=np.array([0.1, 0.5, 0.9]),
        q_hat_0=0.4,
        q_hat_1=0.4,
    )

    assert list(sets_with_both["prediction_set"]) == ["{0}", "{0,1}", "{1}"]
    assert list(sets_with_both["set_size"]) == [1.0, 2.0, 1.0]

    assert list(sets_with_empty["prediction_set"]) == ["{0}", "{}", "{1}"]
    assert list(sets_with_empty["set_size"]) == [1.0, 0.0, 1.0]


def test_temporal_mondrian_uses_true_class_past_fold_scores_only() -> None:
    frame = pd.DataFrame(
        {
            "target_transition_1y": [0, 0, 1, 1, 0, 1, 0, 1],
            "probability_raw": [0.1, 0.2, 0.6, 0.7, 0.3, 0.8, 0.4, 0.9],
            "fold": [1, 1, 1, 1, 2, 2, 3, 3],
        }
    )

    coverage = temporal_mondrian_prediction_set_coverage(
        frame=frame,
        target_column="target_transition_1y",
        probability_column="probability_raw",
        target_coverage=0.80,
    )

    assert list(coverage["fold"]) == [2, 3]
    assert list(coverage["calibration_rows"]) == [4, 6]
    assert list(coverage["calibration_negative_rows"]) == [2, 3]
    assert list(coverage["calibration_positive_rows"]) == [2, 3]
    assert list(coverage["evaluation_rows"]) == [2, 2]
    assert list(coverage["q_hat_0"]) == [0.2, 0.3]
    np.testing.assert_allclose(coverage["q_hat_1"], [0.4, 0.4])
    assert list(coverage["positive_class_coverage"]) == [1.0, 1.0]
    assert list(coverage["negative_class_coverage"]) == [0.0, 0.0]


def test_future_fold_does_not_affect_earlier_mondrian_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "probability_raw": [0.1, 0.6, 0.2, 0.7, 0.99, 0.01],
            "fold": [1, 1, 2, 2, 3, 3],
        }
    )

    original = temporal_mondrian_prediction_set_coverage(
        frame, "target", "probability_raw"
    )
    changed = frame.copy()
    changed.loc[changed["fold"] == 3, "probability_raw"] = [0.01, 0.99]
    perturbed = temporal_mondrian_prediction_set_coverage(
        changed, "target", "probability_raw"
    )

    fold_2_columns = ["q_hat_0", "q_hat_1", "calibration_rows"]
    pd.testing.assert_series_equal(
        original.loc[original["fold"] == 2, fold_2_columns].iloc[0],
        perturbed.loc[perturbed["fold"] == 2, fold_2_columns].iloc[0],
    )
