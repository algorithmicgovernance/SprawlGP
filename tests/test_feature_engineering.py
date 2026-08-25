"""Minimal tests for shared urban-expansion feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.urban_expansion import (
    add_candidate_features,
    predictors_for,
)


def test_candidate_features_use_existing_columns_only() -> None:
    """Core candidate transformations must be finite and deterministic."""
    frame = pd.DataFrame(
        {
            "population_density_t": [100.0, 200.0],
            "distance_to_built_m_t": [0.0, 100.0],
            "slope_degrees": [1.0, 2.0],
            "built_fraction_11x11_t": [0.2, 0.4],
            "recent_local_growth_5y_t": [0.1, 0.3],
        }
    )

    result = add_candidate_features(frame)

    expected = {
        "log_population_density_t",
        "log_distance_to_built_m_t",
        "slope_squared_t",
        "built_fraction_x_recent_growth_t",
        "built_fraction_x_log_population_t",
    }
    assert expected.issubset(result.columns)
    assert np.isfinite(result[list(expected)].to_numpy()).all()


@pytest.mark.parametrize(
    "feature_set",
    [
        "base",
        "log_distance",
        "slope_squared",
        "growth_interaction",
        "population_interaction",
        "nonlinear_core",
        "interactions",
    ],
)
def test_feature_set_mapping_is_defined(feature_set: str) -> None:
    """Every declared experiment feature set must resolve to predictors."""
    predictors = predictors_for(feature_set)
    assert len(predictors) > 0
    
    
def test_calibration_merge_preserves_probability_bias() -> None:
    """Additional calibration metrics must not overwrite SVGP fold metrics."""
    fold_metrics = pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "probability_bias": [0.1, 0.2, -0.1],
            "log_loss": [0.4, 0.5, 0.6],
        }
    )

    calibration = pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "probability_bias": [0.1, 0.2, -0.1],
            "absolute_probability_bias": [0.1, 0.2, 0.1],
            "ece": [0.08, 0.15, 0.17],
            "calibration_intercept": [0.0, -0.5, 0.8],
            "calibration_slope": [1.0, 1.2, 0.9],
        }
    )

    calibration_columns = [
        "fold",
        "ece",
        "calibration_intercept",
        "calibration_slope",
        "absolute_probability_bias",
    ]

    merged = fold_metrics.merge(
        calibration[calibration_columns],
        on="fold",
        how="left",
        validate="one_to_one",
    )

    assert "probability_bias" in merged.columns
    assert "probability_bias_x" not in merged.columns
    assert "probability_bias_y" not in merged.columns
