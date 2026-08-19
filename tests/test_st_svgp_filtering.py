"""Gaussian validation tests for ST-SVGP filtering and smoothing."""

from __future__ import annotations

import tensorflow as tf

from src.models.st_svgp.filtering import (
    function_marginals_from_states,
    kalman_filter_matern32,
    rts_smoother_matern32,
)
from src.models.st_svgp.validate_filter_smoother import (
    exact_gp_posterior,
)


DTYPE = tf.float64


def _run_case(
    *,
    times: list[float],
    observations: list[float],
    lengthscale: float = 1.5,
    variance: float = 1.0,
    noise_variance: float = 0.15,
) -> tuple:
    """Run state-space and exact-GP inference for one synthetic case."""
    time_tensor = tf.constant(times, dtype=DTYPE)
    observation_tensor = tf.constant(
        observations,
        dtype=DTYPE,
    )

    filtered = kalman_filter_matern32(
        times=time_tensor,
        observations=observation_tensor,
        observation_noise_variance=noise_variance,
        lengthscale=lengthscale,
        variance=variance,
        dtype=DTYPE,
    )

    smoothed = rts_smoother_matern32(
        times=time_tensor,
        filter_result=filtered,
        lengthscale=lengthscale,
        dtype=DTYPE,
    )

    smooth_mean, smooth_variance = (
        function_marginals_from_states(
            state_means=smoothed.smoothed_means,
            state_covariances=smoothed.smoothed_covariances,
        )
    )

    exact_mean, exact_covariance, exact_log_marginal = (
        exact_gp_posterior(
            times=time_tensor,
            observations=observation_tensor,
            observation_noise_variance=noise_variance,
            lengthscale=lengthscale,
            variance=variance,
            dtype=DTYPE,
        )
    )

    return (
        filtered,
        smoothed,
        smooth_mean,
        smooth_variance,
        exact_mean,
        exact_covariance,
        exact_log_marginal,
    )


def test_rts_smoothed_marginals_match_exact_gp_regular_times() -> None:
    """Regular five-year-step analogue must match exact GP regression."""
    (
        _,
        _,
        smooth_mean,
        smooth_variance,
        exact_mean,
        exact_covariance,
        _,
    ) = _run_case(
        times=[0.0, 1.0, 2.0, 3.0],
        observations=[0.2, -0.1, 0.35, 0.05],
    )

    tf.debugging.assert_near(
        smooth_mean,
        exact_mean,
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    tf.debugging.assert_near(
        smooth_variance,
        tf.linalg.diag_part(exact_covariance),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_rts_smoothed_marginals_match_exact_gp_irregular_times() -> None:
    """Irregular deltas verify that A(delta) and Q(delta) are actually used."""
    (
        _,
        _,
        smooth_mean,
        smooth_variance,
        exact_mean,
        exact_covariance,
        _,
    ) = _run_case(
        times=[0.0, 0.4, 1.0, 2.25, 4.0],
        observations=[0.20, -0.10, 0.35, 0.05, -0.25],
    )

    tf.debugging.assert_near(
        smooth_mean,
        exact_mean,
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    tf.debugging.assert_near(
        smooth_variance,
        tf.linalg.diag_part(exact_covariance),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_filter_log_marginal_matches_exact_gp() -> None:
    """Forward-filter evidence must equal exact GP Gaussian evidence."""
    (
        filtered,
        _,
        _,
        _,
        _,
        _,
        exact_log_marginal,
    ) = _run_case(
        times=[0.0, 0.4, 1.0, 2.25, 4.0],
        observations=[0.20, -0.10, 0.35, 0.05, -0.25],
    )

    tf.debugging.assert_near(
        filtered.log_marginal_likelihood,
        exact_log_marginal,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_smoothing_does_not_increase_function_variance() -> None:
    """Conditioning on future data should not increase marginal uncertainty."""
    (
        filtered,
        smoothed,
        _,
        _,
        _,
        _,
        _,
    ) = _run_case(
        times=[0.0, 0.4, 1.0, 2.25, 4.0],
        observations=[0.20, -0.10, 0.35, 0.05, -0.25],
    )

    _, filtered_variance = function_marginals_from_states(
        state_means=filtered.filtered_means,
        state_covariances=filtered.filtered_covariances,
    )
    _, smoothed_variance = function_marginals_from_states(
        state_means=smoothed.smoothed_means,
        state_covariances=smoothed.smoothed_covariances,
    )

    assert bool(
        tf.reduce_all(
            smoothed_variance
            <= filtered_variance + 1.0e-10
        ).numpy()
    )


def test_filtered_and_smoothed_covariances_are_psd() -> None:
    """All state covariance matrices must remain numerically PSD."""
    (
        filtered,
        smoothed,
        _,
        _,
        _,
        _,
        _,
    ) = _run_case(
        times=[0.0, 0.4, 1.0, 2.25, 4.0],
        observations=[0.20, -0.10, 0.35, 0.05, -0.25],
    )

    for covariance in filtered.filtered_covariances:
        assert float(
            tf.reduce_min(
                tf.linalg.eigvalsh(covariance)
            ).numpy()
        ) >= -1.0e-10

    for covariance in smoothed.smoothed_covariances:
        assert float(
            tf.reduce_min(
                tf.linalg.eigvalsh(covariance)
            ).numpy()
        ) >= -1.0e-10
