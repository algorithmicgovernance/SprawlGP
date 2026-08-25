"""CVI / natural-gradient components for Bernoulli-probit ST-SVGP.

This module validates the non-conjugate inference mechanism before spatial
inducing states and the real-data training loop are introduced.

The true likelihood remains Bernoulli-probit. CVI represents its contribution
through Gaussian pseudo-likelihood factors whose natural parameters are
updated with natural gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.st_svgp.filtering import (
    FilterResult,
    SmootherResult,
    function_marginals_from_states,
    kalman_filter_matern32,
    rts_smoother_matern32,
)
from src.models.st_svgp.state_space import matern32_covariance


@dataclass(frozen=True)
class CviSites:
    """Gaussian pseudo-likelihood natural parameters."""

    lambda1: tf.Tensor
    lambda2: tf.Tensor


@dataclass(frozen=True)
class CviPosterior:
    """Posterior induced by the GP prior and CVI Gaussian sites."""

    filter_result: FilterResult
    smoother_result: SmootherResult
    marginal_mean: tf.Tensor
    marginal_variance: tf.Tensor
    pseudo_observation: tf.Tensor
    pseudo_variance: tf.Tensor


def initialise_cvi_sites(
    n_sites: int,
    *,
    initial_precision: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> CviSites:
    """Initialise weak, zero-centred Gaussian CVI factors."""
    if n_sites < 1:
        raise ValueError("n_sites must be positive.")
    if initial_precision <= 0.0:
        raise ValueError("initial_precision must be positive.")

    precision = tf.fill(
        [n_sites],
        tf.cast(initial_precision, dtype),
    )

    return CviSites(
        lambda1=tf.zeros([n_sites], dtype=dtype),
        lambda2=-0.5 * precision,
    )


def sites_to_pseudo_observations(
    sites: CviSites,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Convert Gaussian natural parameters to pseudo mean and variance.

    For a scalar Gaussian factor,

        exp(lambda1 * f + lambda2 * f^2)

    the precision is -2 * lambda2, the variance is its inverse, and the
    pseudo observation is lambda1 / precision.
    """
    precision = -2.0 * sites.lambda2

    tf.debugging.assert_positive(
        precision,
        message="CVI site precision must remain positive.",
    )

    pseudo_variance = 1.0 / precision
    pseudo_observation = sites.lambda1 / precision

    return pseudo_observation, pseudo_variance


def expected_log_bernoulli_probit(
    *,
    targets: tf.Tensor,
    marginal_mean: tf.Tensor,
    marginal_variance: tf.Tensor,
    quadrature_degree: int = 40,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Gauss-Hermite approximation of E_q[log p(y|f)].

    The likelihood is

        y | f ~ Bernoulli(Phi(f)).

    Returns one expected log-likelihood value per observation.
    """
    y = tf.reshape(
        tf.convert_to_tensor(targets, dtype=dtype),
        [-1],
    )
    mean = tf.reshape(
        tf.convert_to_tensor(marginal_mean, dtype=dtype),
        [-1],
    )
    variance = tf.reshape(
        tf.convert_to_tensor(marginal_variance, dtype=dtype),
        [-1],
    )

    tf.debugging.assert_equal(
        tf.shape(y),
        tf.shape(mean),
        message="targets and marginal_mean must have matching shapes.",
    )
    tf.debugging.assert_equal(
        tf.shape(mean),
        tf.shape(variance),
        message="marginal mean/variance must have matching shapes.",
    )
    tf.debugging.assert_greater_equal(y, tf.cast(0.0, dtype))
    tf.debugging.assert_less_equal(y, tf.cast(1.0, dtype))
    tf.debugging.assert_positive(
        variance,
        message="Marginal variances must be positive.",
    )

    nodes_np, weights_np = np.polynomial.hermite.hermgauss(
        quadrature_degree
    )
    nodes = tf.constant(nodes_np, dtype=dtype)
    weights = tf.constant(weights_np, dtype=dtype)

    samples = (
        mean[:, None]
        + tf.sqrt(
            tf.cast(2.0, dtype)
            * variance[:, None]
        )
        * nodes[None, :]
    )

    signed_targets = (
        tf.cast(2.0, dtype) * y
        - tf.cast(1.0, dtype)
    )

    standard_normal = tfp.distributions.Normal(
        loc=tf.cast(0.0, dtype),
        scale=tf.cast(1.0, dtype),
    )

    log_likelihood = standard_normal.log_cdf(
        signed_targets[:, None] * samples
    )

    pi = tf.constant(3.141592653589793, dtype=dtype)

    return (
        tf.reduce_sum(
            weights[None, :] * log_likelihood,
            axis=1,
        )
        / tf.sqrt(pi)
    )


def probit_predictive_probability(
    *,
    marginal_mean: tf.Tensor,
    marginal_variance: tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return E_q[Phi(f)] for Gaussian q(f)."""
    mean = tf.convert_to_tensor(
        marginal_mean,
        dtype=dtype,
    )
    variance = tf.convert_to_tensor(
        marginal_variance,
        dtype=dtype,
    )

    tf.debugging.assert_greater_equal(
        variance,
        tf.cast(0.0, dtype),
    )

    standard_normal = tfp.distributions.Normal(
        loc=tf.cast(0.0, dtype),
        scale=tf.cast(1.0, dtype),
    )

    return standard_normal.cdf(
        mean
        / tf.sqrt(
            tf.cast(1.0, dtype)
            + variance
        )
    )


def cvi_moment_gradient_targets(
    *,
    targets: tf.Tensor,
    marginal_mean: tf.Tensor,
    marginal_variance: tf.Tensor,
    quadrature_degree: int,
    dtype: tf.dtypes.DType = tf.float64,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute the natural-parameter target of one CVI update.

    If the Gaussian expectation parameters are

        mu1 = m,
        mu2 = m^2 + v,

    then

        dE/dmu2 = dE/dv,
        dE/dmu1 = dE/dm - 2 m dE/dv.

    These are the Gaussian likelihood-site natural-parameter targets used by
    the natural-gradient CVI update.
    """
    mean = tf.identity(
        tf.convert_to_tensor(
            marginal_mean,
            dtype=dtype,
        )
    )
    variance = tf.identity(
        tf.convert_to_tensor(
            marginal_variance,
            dtype=dtype,
        )
    )

    with tf.GradientTape() as tape:
        tape.watch([mean, variance])

        expected_log_likelihood = (
            expected_log_bernoulli_probit(
                targets=targets,
                marginal_mean=mean,
                marginal_variance=variance,
                quadrature_degree=quadrature_degree,
                dtype=dtype,
            )
        )

        objective = tf.reduce_sum(
            expected_log_likelihood
        )

    gradient_mean, gradient_variance = tape.gradient(
        objective,
        [mean, variance],
    )

    if gradient_mean is None or gradient_variance is None:
        raise RuntimeError(
            "Could not differentiate Bernoulli-probit expectation."
        )

    target_lambda2 = gradient_variance
    target_lambda1 = (
        gradient_mean
        - 2.0 * mean * gradient_variance
    )

    return (
        target_lambda1,
        target_lambda2,
        expected_log_likelihood,
    )


def natural_gradient_site_update(
    *,
    sites: CviSites,
    target_lambda1: tf.Tensor,
    target_lambda2: tf.Tensor,
    gamma: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> CviSites:
    """Apply the damped CVI natural-gradient update."""
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must lie in (0, 1].")

    step = tf.cast(gamma, dtype)
    one = tf.cast(1.0, dtype)

    updated = CviSites(
        lambda1=(
            (one - step) * sites.lambda1
            + step * target_lambda1
        ),
        lambda2=(
            (one - step) * sites.lambda2
            + step * target_lambda2
        ),
    )

    # A valid Gaussian pseudo likelihood requires positive precision.
    precision = -2.0 * updated.lambda2
    tf.debugging.assert_positive(
        precision,
        message=(
            "Natural-gradient update produced a non-positive "
            "Gaussian site precision."
        ),
    )

    return updated


def posterior_from_cvi_sites(
    *,
    times: tf.Tensor,
    sites: CviSites,
    lengthscale: float,
    variance: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> CviPosterior:
    """Compute q(f) by filtering/smoothing the Gaussian CVI factors."""
    pseudo_observation, pseudo_variance = (
        sites_to_pseudo_observations(sites)
    )

    filtered = kalman_filter_matern32(
        times=times,
        observations=pseudo_observation,
        observation_noise_variance=pseudo_variance,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    smoothed = rts_smoother_matern32(
        times=times,
        filter_result=filtered,
        lengthscale=lengthscale,
        dtype=dtype,
    )

    mean, marginal_variance = (
        function_marginals_from_states(
            state_means=smoothed.smoothed_means,
            state_covariances=smoothed.smoothed_covariances,
        )
    )

    return CviPosterior(
        filter_result=filtered,
        smoother_result=smoothed,
        marginal_mean=mean,
        marginal_variance=marginal_variance,
        pseudo_observation=pseudo_observation,
        pseudo_variance=pseudo_variance,
    )


def expected_log_gaussian_sites(
    *,
    posterior: CviPosterior,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return E_q[log N(y_tilde | f, V_tilde)] per site."""
    mean = posterior.marginal_mean
    variance = posterior.marginal_variance
    pseudo_y = posterior.pseudo_observation
    pseudo_variance = posterior.pseudo_variance

    pi = tf.constant(3.141592653589793, dtype=dtype)

    return -0.5 * (
        tf.math.log(
            tf.cast(2.0, dtype)
            * pi
            * pseudo_variance
        )
        + (
            tf.square(pseudo_y - mean)
            + variance
        )
        / pseudo_variance
    )


def cvi_state_space_elbo(
    *,
    targets: tf.Tensor,
    posterior: CviPosterior,
    quadrature_degree: int,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Compute the CVI form of the variational ELBO.

    ELBO =
        E_q log p(y|f)
        - E_q log N(y_tilde|f,V_tilde)
        + log p(y_tilde)

    where the final term is the Gaussian marginal likelihood returned by the
    forward Kalman filter.
    """
    expected_true_likelihood = (
        expected_log_bernoulli_probit(
            targets=targets,
            marginal_mean=posterior.marginal_mean,
            marginal_variance=posterior.marginal_variance,
            quadrature_degree=quadrature_degree,
            dtype=dtype,
        )
    )

    expected_sites = expected_log_gaussian_sites(
        posterior=posterior,
        dtype=dtype,
    )

    return (
        tf.reduce_sum(expected_true_likelihood)
        - tf.reduce_sum(expected_sites)
        + posterior.filter_result.log_marginal_likelihood
    )


def dense_gaussian_site_posterior(
    *,
    times: tf.Tensor,
    sites: CviSites,
    lengthscale: float,
    variance: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Direct dense Gaussian posterior used only as a validation oracle."""
    pseudo_y, pseudo_variance = (
        sites_to_pseudo_observations(sites)
    )

    covariance = matern32_covariance(
        times,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    n = tf.shape(covariance)[0]
    system = (
        covariance
        + tf.linalg.diag(pseudo_variance)
    )

    chol = tf.linalg.cholesky(
        0.5 * (system + tf.transpose(system))
    )

    alpha = tf.linalg.cholesky_solve(
        chol,
        pseudo_y[:, None],
    )

    posterior_mean = tf.reshape(
        covariance @ alpha,
        [-1],
    )

    solved = tf.linalg.cholesky_solve(
        chol,
        covariance,
    )

    posterior_covariance = (
        covariance
        - covariance @ solved
    )
    posterior_covariance = 0.5 * (
        posterior_covariance
        + tf.transpose(posterior_covariance)
    )

    pi = tf.constant(3.141592653589793, dtype=dtype)
    log_marginal = (
        -0.5
        * tf.reshape(
            pseudo_y[None, :]
            @ alpha,
            [],
        )
        - tf.reduce_sum(
            tf.math.log(
                tf.linalg.diag_part(chol)
            )
        )
        - 0.5
        * tf.cast(n, dtype)
        * tf.math.log(
            tf.cast(2.0, dtype) * pi
        )
    )

    return (
        posterior_mean,
        posterior_covariance,
        log_marginal,
    )


def gaussian_kl_to_prior(
    *,
    times: tf.Tensor,
    posterior_mean: tf.Tensor,
    posterior_covariance: tf.Tensor,
    lengthscale: float,
    variance: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """KL[N(m,S) || N(0,K)] for validation of the CVI ELBO identity."""
    prior_covariance = matern32_covariance(
        times,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    prior_chol = tf.linalg.cholesky(prior_covariance)
    posterior_chol = tf.linalg.cholesky(
        0.5
        * (
            posterior_covariance
            + tf.transpose(posterior_covariance)
        )
    )

    prior_inverse_posterior = tf.linalg.cholesky_solve(
        prior_chol,
        posterior_covariance,
    )
    prior_inverse_mean = tf.linalg.cholesky_solve(
        prior_chol,
        posterior_mean[:, None],
    )

    dimension = tf.cast(
        tf.shape(prior_covariance)[0],
        dtype,
    )

    logdet_prior = 2.0 * tf.reduce_sum(
        tf.math.log(
            tf.linalg.diag_part(prior_chol)
        )
    )
    logdet_posterior = 2.0 * tf.reduce_sum(
        tf.math.log(
            tf.linalg.diag_part(posterior_chol)
        )
    )

    quadratic = tf.reshape(
        posterior_mean[None, :]
        @ prior_inverse_mean,
        [],
    )

    return 0.5 * (
        tf.linalg.trace(prior_inverse_posterior)
        + quadratic
        - dimension
        + logdet_prior
        - logdet_posterior
    )


def dense_standard_elbo(
    *,
    targets: tf.Tensor,
    times: tf.Tensor,
    sites: CviSites,
    lengthscale: float,
    variance: float,
    quadrature_degree: int,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Standard E_q log p(y|f) - KL(q||p) validation oracle."""
    mean, covariance, _ = dense_gaussian_site_posterior(
        times=times,
        sites=sites,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    expected_log_likelihood = (
        expected_log_bernoulli_probit(
            targets=targets,
            marginal_mean=mean,
            marginal_variance=tf.linalg.diag_part(covariance),
            quadrature_degree=quadrature_degree,
            dtype=dtype,
        )
    )

    kl = gaussian_kl_to_prior(
        times=times,
        posterior_mean=mean,
        posterior_covariance=covariance,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    return (
        tf.reduce_sum(expected_log_likelihood)
        - kl
    )
