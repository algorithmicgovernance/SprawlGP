"""Mathematical tests for the ST-SVGP Matérn-3/2 temporal prior."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import pytest
import tensorflow as tf

from src.models.st_svgp.state_space import (
    matern32_continuous_matrices,
    matern32_covariance,
    matern32_process_noise,
    matern32_state_space_covariance,
    matern32_stationary_covariance,
    matern32_transition_matrix,
)


DTYPE = tf.float64


@pytest.mark.parametrize(
    ("lengthscale", "variance"),
    [
        (0.5, 0.3),
        (1.0, 1.0),
        (1.5, 1.0),
        (3.0, 2.0),
    ],
)
def test_state_space_covariance_matches_direct_matern32(
    lengthscale: float,
    variance: float,
) -> None:
    """The Markov representation must reproduce the Matérn-3/2 kernel."""
    times = tf.constant(
        [0.0, 0.25, 1.0, 2.0, 4.0],
        dtype=DTYPE,
    )

    direct = matern32_covariance(
        times,
        lengthscale=lengthscale,
        variance=variance,
        dtype=DTYPE,
    )

    state_space = matern32_state_space_covariance(
        times,
        lengthscale=lengthscale,
        variance=variance,
        dtype=DTYPE,
    )

    tf.debugging.assert_near(
        state_space,
        direct,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


@pytest.mark.parametrize(
    "delta",
    [0.0, 0.1, 0.25, 1.0, 2.5],
)
def test_closed_form_transition_matches_matrix_exponential(
    delta: float,
) -> None:
    """The analytic A(delta) must equal expm(F * delta)."""
    lengthscale = 1.5
    variance = 1.0

    feedback, _, _, _ = (
        matern32_continuous_matrices(
            lengthscale=lengthscale,
            variance=variance,
            dtype=DTYPE,
        )
    )

    analytic = matern32_transition_matrix(
        delta,
        lengthscale=lengthscale,
        dtype=DTYPE,
    )

    # matrix_exponential = tf.linalg.expm(
    #     feedback
    #     * tf.cast(delta, DTYPE)
    # )

    # tf.debugging.assert_near(
    #     analytic,
    #     matrix_exponential,
    #     atol=1.0e-10,
    #     rtol=1.0e-10,
    # )
    
    matrix_exponential = scipy.linalg.expm(
        feedback.numpy() * float(delta)
    )

    np.testing.assert_allclose(
        analytic.numpy(),
        matrix_exponential,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize(
    "delta",
    [0.0, 0.25, 1.0, 2.5],
)
def test_process_noise_preserves_stationary_covariance(
    delta: float,
) -> None:
    """P_inf must satisfy P_inf = A P_inf A^T + Q."""
    lengthscale = 1.5
    variance = 1.0

    stationary = matern32_stationary_covariance(
        lengthscale=lengthscale,
        variance=variance,
        dtype=DTYPE,
    )

    transition = matern32_transition_matrix(
        delta,
        lengthscale=lengthscale,
        dtype=DTYPE,
    )

    process_noise = matern32_process_noise(
        delta,
        lengthscale=lengthscale,
        variance=variance,
        dtype=DTYPE,
    )

    reconstructed = (
        transition
        @ stationary
        @ tf.transpose(transition)
        + process_noise
    )

    tf.debugging.assert_near(
        reconstructed,
        stationary,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


@pytest.mark.parametrize(
    "delta",
    [0.0, 0.01, 0.25, 1.0, 2.5, 10.0],
)
def test_process_noise_is_positive_semidefinite(
    delta: float,
) -> None:
    """Every valid time increment must produce a PSD Q(delta)."""
    process_noise = matern32_process_noise(
        delta,
        lengthscale=1.5,
        variance=1.0,
        dtype=DTYPE,
    )

    minimum_eigenvalue = tf.reduce_min(
        tf.linalg.eigvalsh(process_noise)
    )

    assert float(minimum_eigenvalue.numpy()) >= -1.0e-10


def test_gpflow_matern32_uses_same_covariance_convention() -> None:
    """Cross-check the direct covariance against the project's GPflow stack."""
    gpflow = pytest.importorskip("gpflow")

    gpflow.config.set_default_float(DTYPE)

    times = tf.constant(
        [[0.0], [1.0], [2.0], [3.0], [4.0]],
        dtype=DTYPE,
    )

    lengthscale = 1.5
    variance = 1.0

    expected = matern32_covariance(
        tf.reshape(times, [-1]),
        lengthscale=lengthscale,
        variance=variance,
        dtype=DTYPE,
    )

    kernel = gpflow.kernels.Matern32(
        variance=variance,
        lengthscales=lengthscale,
    )

    actual = kernel(times)

    tf.debugging.assert_near(
        actual,
        expected,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
