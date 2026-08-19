"""Tests for Bernoulli-probit CVI / natural-gradient inference."""

from __future__ import annotations

import numpy as np
import scipy.special
import tensorflow as tf

from src.models.st_svgp.cvi import (
    cvi_moment_gradient_targets,
    cvi_state_space_elbo,
    dense_gaussian_site_posterior,
    dense_standard_elbo,
    expected_log_bernoulli_probit,
    initialise_cvi_sites,
    natural_gradient_site_update,
    posterior_from_cvi_sites,
    probit_predictive_probability,
)


DTYPE = tf.float64
TIMES = tf.constant(
    [0.0, 0.5, 1.2, 2.0, 3.3],
    dtype=DTYPE,
)
TARGETS = tf.constant(
    [0.0, 1.0, 1.0, 0.0, 1.0],
    dtype=DTYPE,
)
LENGTHSCALE = 1.5
VARIANCE = 1.0
QUADRATURE_DEGREE = 40


def test_expected_log_probit_is_finite() -> None:
    """Bernoulli-probit variational expectations must be finite."""
    mean = tf.constant(
        [-1.0, -0.2, 0.0, 0.5, 1.5],
        dtype=DTYPE,
    )
    variance = tf.constant(
        [0.2, 0.5, 1.0, 0.3, 0.8],
        dtype=DTYPE,
    )

    expected = expected_log_bernoulli_probit(
        targets=TARGETS,
        marginal_mean=mean,
        marginal_variance=variance,
        quadrature_degree=QUADRATURE_DEGREE,
        dtype=DTYPE,
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(expected)
        ).numpy()
    )


def test_cvi_moment_gradients_match_finite_differences() -> None:
    """Autodiff moment gradients must agree with finite differences."""
    mean = tf.constant(
        [-0.3, 0.2, 0.7, -0.1, 0.4],
        dtype=DTYPE,
    )
    variance = tf.constant(
        [0.5, 0.6, 0.4, 0.8, 0.7],
        dtype=DTYPE,
    )

    target_lambda1, target_lambda2, _ = (
        cvi_moment_gradient_targets(
            targets=TARGETS,
            marginal_mean=mean,
            marginal_variance=variance,
            quadrature_degree=QUADRATURE_DEGREE,
            dtype=DTYPE,
        )
    )

    # Recover dE/dm and dE/dv from the moment-gradient targets.
    autodiff_dv = target_lambda2
    autodiff_dm = (
        target_lambda1
        + 2.0 * mean * target_lambda2
    )

    epsilon = 1.0e-6

    def total_expectation(
        current_mean: np.ndarray,
        current_variance: np.ndarray,
    ) -> float:
        return float(
            tf.reduce_sum(
                expected_log_bernoulli_probit(
                    targets=TARGETS,
                    marginal_mean=tf.constant(
                        current_mean,
                        dtype=DTYPE,
                    ),
                    marginal_variance=tf.constant(
                        current_variance,
                        dtype=DTYPE,
                    ),
                    quadrature_degree=QUADRATURE_DEGREE,
                    dtype=DTYPE,
                )
            ).numpy()
        )

    mean_np = mean.numpy()
    variance_np = variance.numpy()

    finite_dm = np.zeros_like(mean_np)
    finite_dv = np.zeros_like(variance_np)

    for index in range(len(mean_np)):
        plus = mean_np.copy()
        minus = mean_np.copy()
        plus[index] += epsilon
        minus[index] -= epsilon

        finite_dm[index] = (
            total_expectation(plus, variance_np)
            - total_expectation(minus, variance_np)
        ) / (2.0 * epsilon)

        plus_v = variance_np.copy()
        minus_v = variance_np.copy()
        plus_v[index] += epsilon
        minus_v[index] -= epsilon

        finite_dv[index] = (
            total_expectation(mean_np, plus_v)
            - total_expectation(mean_np, minus_v)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        autodiff_dm.numpy(),
        finite_dm,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    np.testing.assert_allclose(
        autodiff_dv.numpy(),
        finite_dv,
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_cvi_filter_posterior_matches_dense_gaussian_site_posterior() -> None:
    """Filtering/smoothing and dense Gaussian site inference must agree."""
    sites = initialise_cvi_sites(
        5,
        initial_precision=0.25,
        dtype=DTYPE,
    )

    # Use non-zero site means so the posterior mean is also tested.
    sites = type(sites)(
        lambda1=tf.constant(
            [-0.2, 0.3, 0.4, -0.1, 0.25],
            dtype=DTYPE,
        ),
        lambda2=sites.lambda2,
    )

    posterior = posterior_from_cvi_sites(
        times=TIMES,
        sites=sites,
        lengthscale=LENGTHSCALE,
        variance=VARIANCE,
        dtype=DTYPE,
    )

    dense_mean, dense_covariance, dense_log_marginal = (
        dense_gaussian_site_posterior(
            times=TIMES,
            sites=sites,
            lengthscale=LENGTHSCALE,
            variance=VARIANCE,
            dtype=DTYPE,
        )
    )

    tf.debugging.assert_near(
        posterior.marginal_mean,
        dense_mean,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    tf.debugging.assert_near(
        posterior.marginal_variance,
        tf.linalg.diag_part(dense_covariance),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    tf.debugging.assert_near(
        posterior.filter_result.log_marginal_likelihood,
        dense_log_marginal,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_cvi_elbo_identity_matches_standard_variational_elbo() -> None:
    """CVI ELBO form must equal ELL - KL(q||p)."""
    sites = initialise_cvi_sites(
        5,
        initial_precision=0.4,
        dtype=DTYPE,
    )
    sites = type(sites)(
        lambda1=tf.constant(
            [-0.5, 0.4, 0.6, -0.25, 0.5],
            dtype=DTYPE,
        ),
        lambda2=sites.lambda2,
    )

    posterior = posterior_from_cvi_sites(
        times=TIMES,
        sites=sites,
        lengthscale=LENGTHSCALE,
        variance=VARIANCE,
        dtype=DTYPE,
    )

    cvi_elbo = cvi_state_space_elbo(
        targets=TARGETS,
        posterior=posterior,
        quadrature_degree=QUADRATURE_DEGREE,
        dtype=DTYPE,
    )

    standard_elbo = dense_standard_elbo(
        targets=TARGETS,
        times=TIMES,
        sites=sites,
        lengthscale=LENGTHSCALE,
        variance=VARIANCE,
        quadrature_degree=QUADRATURE_DEGREE,
        dtype=DTYPE,
    )

    tf.debugging.assert_near(
        cvi_elbo,
        standard_elbo,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_natural_gradient_update_keeps_positive_site_precision() -> None:
    """A damped CVI update must keep valid Gaussian pseudo-likelihoods."""
    sites = initialise_cvi_sites(
        5,
        initial_precision=1.0e-3,
        dtype=DTYPE,
    )

    posterior = posterior_from_cvi_sites(
        times=TIMES,
        sites=sites,
        lengthscale=LENGTHSCALE,
        variance=VARIANCE,
        dtype=DTYPE,
    )

    target_lambda1, target_lambda2, _ = (
        cvi_moment_gradient_targets(
            targets=TARGETS,
            marginal_mean=posterior.marginal_mean,
            marginal_variance=posterior.marginal_variance,
            quadrature_degree=QUADRATURE_DEGREE,
            dtype=DTYPE,
        )
    )

    updated = natural_gradient_site_update(
        sites=sites,
        target_lambda1=target_lambda1,
        target_lambda2=target_lambda2,
        gamma=0.2,
        dtype=DTYPE,
    )

    assert bool(
        tf.reduce_all(
            -2.0 * updated.lambda2 > 0.0
        ).numpy()
    )


def test_natural_gradient_iterations_improve_validation_elbo() -> None:
    """The fixed synthetic case should improve under repeated CVI updates."""
    sites = initialise_cvi_sites(
        5,
        initial_precision=1.0e-3,
        dtype=DTYPE,
    )

    elbos: list[float] = []

    for _ in range(15):
        posterior = posterior_from_cvi_sites(
            times=TIMES,
            sites=sites,
            lengthscale=LENGTHSCALE,
            variance=VARIANCE,
            dtype=DTYPE,
        )

        elbo = cvi_state_space_elbo(
            targets=TARGETS,
            posterior=posterior,
            quadrature_degree=QUADRATURE_DEGREE,
            dtype=DTYPE,
        )
        elbos.append(float(elbo.numpy()))

        target_lambda1, target_lambda2, _ = (
            cvi_moment_gradient_targets(
                targets=TARGETS,
                marginal_mean=posterior.marginal_mean,
                marginal_variance=posterior.marginal_variance,
                quadrature_degree=QUADRATURE_DEGREE,
                dtype=DTYPE,
            )
        )

        sites = natural_gradient_site_update(
            sites=sites,
            target_lambda1=target_lambda1,
            target_lambda2=target_lambda2,
            gamma=0.2,
            dtype=DTYPE,
        )

    assert elbos[-1] > elbos[0]
    assert min(np.diff(elbos)) >= -1.0e-10


def test_probit_predictive_probabilities_are_strictly_valid() -> None:
    """Integrated Bernoulli-probit probabilities must lie strictly in (0,1)."""
    probability = probit_predictive_probability(
        marginal_mean=tf.constant(
            [-2.0, -0.5, 0.0, 0.7, 2.0],
            dtype=DTYPE,
        ),
        marginal_variance=tf.constant(
            [0.2, 0.5, 1.0, 0.4, 0.8],
            dtype=DTYPE,
        ),
        dtype=DTYPE,
    )

    assert bool(
        tf.reduce_all(
            (probability > 0.0)
            & (probability < 1.0)
        ).numpy()
    )
