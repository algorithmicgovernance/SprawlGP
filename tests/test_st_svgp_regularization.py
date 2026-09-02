"""Focused tests for optional ST-SVGP kernel regularization."""

from __future__ import annotations

import copy

import numpy as np
import tensorflow as tf
from src.models.train_st_svgp import (
    _evaluate_early_stopping_window,
    _update_early_stopping_state,
    spatial_kernel_regularization,
    temporal_lengthscale_regularization,
)

DTYPE = tf.float64
REGULARIZATION = {
    "spatial_log_lengthscale": {
        "enabled": True,
        "center_km": 1.0,
        "sigma_log": 0.75,
        "weight": 1.0,
    },
    "kernel_log_variance": {
        "enabled": True,
        "center": 1.0,
        "sigma_log": 0.75,
        "weight": 1.0,
    },
}

EARLY_STOPPING = {
    "min_iterations": 1500,
    "max_iterations": 4000,
    "checkpoint_interval": 50,
    "stability_window_iterations": 250,
    "patience_checkpoints": 3,
    "epsilon": 1.0e-12,
    "diagnostic_elbo": {
        "endpoint_relative_change_max": 0.001,
        "relative_range_max": 0.0025,
    },
    "parameter_stability": {
        "spatial_lengthscale_relative_change_max": 0.01,
        "kernel_variance_relative_change_max": 0.01,
        "temporal_lengthscale_relative_change_max": 0.02,
        "temporal_trend_absolute_change_max": 0.01,
    },
    "site_stability": {
        "median_relative_change_max": 0.01,
        "maximum_relative_change_max": 0.02,
    },
}


def test_pre_specified_early_stopping_boundaries_and_state() -> None:
    def records(last_completed: int = 1500) -> list[dict[str, object]]:
        return [
            {
                "completed_iterations": completed,
                "diagnostic_loss_fixed": 1000.0,
                "site_relative_change_since_checkpoint": 0.0,
                "spatial_lengthscale_x_km": 100.0,
                "spatial_lengthscale_y_km": 100.0,
                "kernel_variance": 100.0,
                "temporal_lengthscale_steps": 100.0,
                "temporal_trend_coefficient": 0.0,
            }
            for completed in range(
                last_completed - 250,
                last_completed + 50,
                50,
            )
        ]

    before_minimum = _evaluate_early_stopping_window(
        records(1450), EARLY_STOPPING
    )
    assert before_minimum["early_stop_eligible"] is False
    assert before_minimum["early_stop_convergence_gate_pass"] is False

    passing = _evaluate_early_stopping_window(records(), EARLY_STOPPING)
    assert passing["window_observation_count"] == 6
    assert passing["site_interval_count"] == 5
    assert passing["early_stop_convergence_gate_pass"] is True

    def evaluate(
        updates: dict[int, dict[str, float]],
    ) -> dict[str, object]:
        candidate = copy.deepcopy(records())
        for index, values in updates.items():
            candidate[index].update(values)
        return _evaluate_early_stopping_window(candidate, EARLY_STOPPING)

    endpoint_boundary = evaluate(
        {-1: {"diagnostic_loss_fixed": 1001.0}}
    )
    assert endpoint_boundary[
        "diagnostic_elbo_endpoint_relative_change"
    ] == EARLY_STOPPING["diagnostic_elbo"][
        "endpoint_relative_change_max"
    ]
    assert endpoint_boundary["early_stop_elbo_plateau_pass"] is True
    assert evaluate(
        {
            -1: {
                "diagnostic_loss_fixed": np.nextafter(1001.0, np.inf)
            }
        }
    )["early_stop_elbo_plateau_pass"] is False

    range_boundary = evaluate(
        {-2: {"diagnostic_loss_fixed": 1002.5}}
    )
    assert range_boundary["diagnostic_elbo_relative_range"] == (
        EARLY_STOPPING["diagnostic_elbo"]["relative_range_max"]
    )
    assert range_boundary["early_stop_elbo_plateau_pass"] is True
    assert evaluate(
        {
            -2: {
                "diagnostic_loss_fixed": np.nextafter(1002.5, np.inf)
            }
        }
    )["early_stop_elbo_plateau_pass"] is False

    for field in (
        "spatial_lengthscale_x_km",
        "spatial_lengthscale_y_km",
        "kernel_variance",
    ):
        boundary = evaluate({-1: {field: 101.0}})
        assert boundary[
            {
                "spatial_lengthscale_x_km": (
                    "spatial_lengthscale_x_relative_change_window"
                ),
                "spatial_lengthscale_y_km": (
                    "spatial_lengthscale_y_relative_change_window"
                ),
                "kernel_variance": "kernel_variance_relative_change_window",
            }[field]
        ] == 0.01
        assert boundary["early_stop_parameter_stability_pass"] is True
        assert evaluate(
            {-1: {field: np.nextafter(101.0, np.inf)}}
        )["early_stop_parameter_stability_pass"] is False

    temporal_boundary = evaluate(
        {-1: {"temporal_lengthscale_steps": 102.0}}
    )
    assert temporal_boundary[
        "temporal_lengthscale_relative_change_window"
    ] == 0.02
    assert temporal_boundary["early_stop_parameter_stability_pass"] is True
    assert evaluate(
        {-1: {"temporal_lengthscale_steps": np.nextafter(102.0, np.inf)}}
    )["early_stop_parameter_stability_pass"] is False

    trend_boundary = evaluate(
        {-1: {"temporal_trend_coefficient": 0.01}}
    )
    assert trend_boundary["temporal_trend_absolute_change_window"] == 0.01
    assert trend_boundary["early_stop_parameter_stability_pass"] is True
    assert evaluate(
        {-1: {"temporal_trend_coefficient": np.nextafter(0.01, np.inf)}}
    )["early_stop_parameter_stability_pass"] is False

    median_boundary = evaluate(
        {
            index: {"site_relative_change_since_checkpoint": 0.01}
            for index in range(1, 6)
        }
    )
    assert median_boundary["site_relative_change_median_window"] == 0.01
    assert median_boundary["early_stop_site_stability_pass"] is True
    assert evaluate(
        {
            index: {
                "site_relative_change_since_checkpoint": np.nextafter(
                    0.01, np.inf
                )
            }
            for index in range(1, 6)
        }
    )["early_stop_site_stability_pass"] is False

    maximum_boundary = evaluate(
        {-1: {"site_relative_change_since_checkpoint": 0.02}}
    )
    assert maximum_boundary["site_relative_change_maximum_window"] == 0.02
    assert maximum_boundary["early_stop_site_stability_pass"] is True
    assert evaluate(
        {
            -1: {
                "site_relative_change_since_checkpoint": np.nextafter(
                    0.02, np.inf
                )
            }
        }
    )["early_stop_site_stability_pass"] is False

    consecutive = 0
    for completed in (1500, 1550):
        consecutive, reason, demonstrated = _update_early_stopping_state(
            completed_iterations=completed,
            eligible=True,
            convergence_gate_pass=True,
            consecutive_passes=consecutive,
            settings=EARLY_STOPPING,
        )
        assert reason is None
        assert demonstrated is False
    consecutive, reason, demonstrated = _update_early_stopping_state(
        completed_iterations=1600,
        eligible=True,
        convergence_gate_pass=True,
        consecutive_passes=consecutive,
        settings=EARLY_STOPPING,
    )
    assert (consecutive, reason, demonstrated) == (
        3,
        "CONVERGENCE_RULE",
        True,
    )

    consecutive = 0
    for gate_pass in (True, True, False):
        consecutive, reason, demonstrated = _update_early_stopping_state(
            completed_iterations=1500,
            eligible=True,
            convergence_gate_pass=gate_pass,
            consecutive_passes=consecutive,
            settings=EARLY_STOPPING,
        )
    assert (consecutive, reason, demonstrated) == (0, None, False)

    consecutive, reason, demonstrated = _update_early_stopping_state(
        completed_iterations=4000,
        eligible=True,
        convergence_gate_pass=False,
        consecutive_passes=0,
        settings=EARLY_STOPPING,
    )
    assert (consecutive, reason, demonstrated) == (
        0,
        "MAX_ITERATIONS",
        False,
    )


def test_spatial_kernel_regularization_values_and_gradients() -> None:
    lengthscales = tf.constant([1.0, 1.0], dtype=DTYPE)
    variance = tf.constant(1.0, dtype=DTYPE)

    missing = spatial_kernel_regularization(
        spatial_lengthscales_km=lengthscales,
        kernel_variance=variance,
        regularization=None,
    )
    disabled = spatial_kernel_regularization(
        spatial_lengthscales_km=lengthscales,
        kernel_variance=variance,
        regularization={
            "spatial_log_lengthscale": {
                **REGULARIZATION["spatial_log_lengthscale"],
                "enabled": False,
            },
            "kernel_log_variance": {
                **REGULARIZATION["kernel_log_variance"],
                "enabled": False,
            },
        },
    )
    assert all(float(value.numpy()) == 0.0 for value in missing)
    assert all(float(value.numpy()) == 0.0 for value in disabled)

    centered = spatial_kernel_regularization(
        spatial_lengthscales_km=lengthscales,
        kernel_variance=variance,
        regularization=REGULARIZATION,
    )
    assert all(float(value.numpy()) == 0.0 for value in centered)

    _, moved_lengthscale, centered_variance = (
        spatial_kernel_regularization(
            spatial_lengthscales_km=tf.constant(
                [2.0, 2.0],
                dtype=DTYPE,
            ),
            kernel_variance=variance,
            regularization=REGULARIZATION,
        )
    )
    _, centered_lengthscale, moved_variance = (
        spatial_kernel_regularization(
            spatial_lengthscales_km=lengthscales,
            kernel_variance=tf.constant(0.6, dtype=DTYPE),
            regularization=REGULARIZATION,
        )
    )
    assert float(moved_lengthscale.numpy()) > 0.0
    assert float(centered_variance.numpy()) == 0.0
    assert float(centered_lengthscale.numpy()) == 0.0
    assert float(moved_variance.numpy()) > 0.0

    log_lengthscales = tf.Variable(
        tf.math.log(tf.constant([2.0, 2.0], dtype=DTYPE))
    )
    log_variance = tf.Variable(
        tf.math.log(tf.constant(0.6, dtype=DTYPE))
    )
    with tf.GradientTape() as tape:
        total, _, _ = spatial_kernel_regularization(
            spatial_lengthscales_km=tf.exp(log_lengthscales),
            kernel_variance=tf.exp(log_variance),
            regularization=REGULARIZATION,
        )
    gradients = tape.gradient(
        total,
        [log_lengthscales, log_variance],
    )

    assert bool(tf.math.is_finite(total).numpy())
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
        for gradient in gradients
    )


def test_temporal_lengthscale_regularization_values_and_gradient() -> None:
    settings = {
        "temporal_log_lengthscale": {
            "enabled": True,
            "center_years": 1.5,
            "weight": 1.0,
        }
    }

    missing = temporal_lengthscale_regularization(
        tf.constant(0.3, dtype=DTYPE),
        step_years=5.0,
        regularization=None,
    )
    disabled = temporal_lengthscale_regularization(
        tf.constant(0.3, dtype=DTYPE),
        step_years=5.0,
        regularization={
            "temporal_log_lengthscale": {
                **settings["temporal_log_lengthscale"],
                "enabled": False,
            }
        },
    )
    centered = temporal_lengthscale_regularization(
        tf.constant(0.3, dtype=DTYPE),
        step_years=5.0,
        regularization=settings,
    )
    moved = temporal_lengthscale_regularization(
        tf.constant(0.78, dtype=DTYPE),
        step_years=5.0,
        regularization=settings,
    )
    initialized = temporal_lengthscale_regularization(
        tf.constant(1.5, dtype=DTYPE),
        step_years=5.0,
        regularization=settings,
    )

    assert float(missing.numpy()) == 0.0
    assert float(disabled.numpy()) == 0.0
    assert float(centered.numpy()) == 0.0
    assert bool(tf.math.is_finite(moved).numpy())
    assert 0.0 < float(moved.numpy()) < float(initialized.numpy())

    log_lengthscale = tf.Variable(
        tf.math.log(tf.constant(0.78, dtype=DTYPE))
    )
    with tf.GradientTape() as tape:
        penalty = temporal_lengthscale_regularization(
            tf.exp(log_lengthscale),
            step_years=5.0,
            regularization=settings,
        )
    gradient = tape.gradient(penalty, log_lengthscale)

    assert gradient is not None
    assert bool(tf.math.is_finite(gradient).numpy())
    assert float(gradient.numpy()) != 0.0