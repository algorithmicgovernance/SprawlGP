"""Minimal tests for shared binary calibration diagnostics."""

import numpy as np

from src.models.evaluation.calibration import (
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
