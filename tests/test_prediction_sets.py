"""Tests for binary conformal prediction-set coverage diagnostics."""

import numpy as np
import pandas as pd

from src.models.evaluation.prediction_sets import (
    build_prediction_sets,
    conformal_quantile,
    nonconformity_scores,
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
    assert (coverage["calibration_rows"].to_numpy() > 0).all()
    assert np.isfinite(coverage["empirical_coverage"].to_numpy()).all()


def test_conformal_quantile_matches_rank_rule() -> None:
    """Finite-sample conformal quantile should use ceil((n+1)*(1-alpha))."""
    scores = np.array([0.1, 0.4, 0.2, 0.5, 0.3])
    q_hat = conformal_quantile(scores, alpha=0.2)
    assert np.isclose(q_hat, 0.5)
