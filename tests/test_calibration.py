"""Minimal tests for shared binary calibration diagnostics."""

import numpy as np
from scipy.special import expit
from src.models.evaluation.calibration import (
    PROBABILITY_EPSILON,
    FittedBinaryCalibrator,
    calibration_metrics,
    reliability_table,
)


def test_calibration_outputs_are_finite_for_non_degenerate_predictions() -> None:
    """Shared calibration diagnostics must return usable finite values."""
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.05, 0.10, 0.25, 0.55, 0.70, 0.80, 0.90, 0.35])

    table = reliability_table(y, p, n_bins=4, strategy="quantile")
    metrics = calibration_metrics(y, p, n_bins=4, strategy="quantile")

    assert int(table["cell_count"].sum()) == len(y)
    assert all(np.isfinite(list(metrics.values())))


def test_platt_formula_is_applied_exactly() -> None:
    probability = np.array([0.0, 0.2, 0.8, 1.0])
    fitted = FittedBinaryCalibrator(
        method="platt",
        coefficient_a=-0.4,
        coefficient_b=1.3,
    )
    clipped = np.clip(
        probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    expected = expit(-0.4 + 1.3 * np.log(clipped / (1.0 - clipped)))

    np.testing.assert_allclose(fitted.predict(probability), expected)
    assert fitted.monotonic_direction() == "increasing"


def test_beta_formula_is_applied_exactly() -> None:
    probability = np.array([0.0, 0.2, 0.8, 1.0])
    fitted = FittedBinaryCalibrator(
        method="beta",
        coefficient_a=0.7,
        coefficient_b=-1.1,
        coefficient_c=0.2,
    )
    clipped = np.clip(
        probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    expected = expit(
        0.7 * np.log(clipped) - 1.1 * np.log1p(-clipped) + 0.2
    )

    np.testing.assert_allclose(fitted.predict(probability), expected)
    assert fitted.monotonic_direction() == "increasing"


def test_beta_non_monotonic_mapping_is_flagged() -> None:
    fitted = FittedBinaryCalibrator(
        method="beta",
        coefficient_a=1.0,
        coefficient_b=1.0,
        coefficient_c=0.0,
    )

    assert fitted.monotonic_direction() == "non_monotonic"
