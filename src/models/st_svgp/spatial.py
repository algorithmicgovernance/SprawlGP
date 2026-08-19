"""Spatial Matérn-3/2 components for ST-SVGP."""

from __future__ import annotations

import tensorflow as tf


def matern32_spatial_covariance(
    points_a: tf.Tensor,
    points_b: tf.Tensor,
    *,
    lengthscales_km: tf.Tensor,
    variance: tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return anisotropic Matérn-3/2 covariance in kilometres."""
    a = tf.convert_to_tensor(points_a, dtype=dtype)
    b = tf.convert_to_tensor(points_b, dtype=dtype)
    lengthscales = tf.reshape(
        tf.convert_to_tensor(lengthscales_km, dtype=dtype),
        [1, 1, 2],
    )
    kernel_variance = tf.reshape(
        tf.convert_to_tensor(variance, dtype=dtype),
        [],
    )

    tf.debugging.assert_positive(
        lengthscales,
        message="Spatial lengthscales must be positive.",
    )
    tf.debugging.assert_positive(
        kernel_variance,
        message="Kernel variance must be positive.",
    )

    scaled_difference = (
        a[:, None, :] - b[None, :, :]
    ) / lengthscales

    radius = tf.sqrt(
        tf.maximum(
            tf.reduce_sum(
                tf.square(scaled_difference),
                axis=-1,
            ),
            tf.cast(0.0, dtype),
        )
    )

    sqrt_three = tf.sqrt(tf.cast(3.0, dtype))
    scaled_radius = sqrt_three * radius

    return (
        kernel_variance
        * (1.0 + scaled_radius)
        * tf.exp(-scaled_radius)
    )


def cholesky_with_jitter(
    matrix: tf.Tensor,
    *,
    jitter: float,
) -> tf.Tensor:
    """Cholesky factorisation with fixed diagonal numerical jitter."""
    matrix = 0.5 * (
        matrix + tf.transpose(matrix)
    )
    size = tf.shape(matrix)[0]
    return tf.linalg.cholesky(
        matrix
        + tf.cast(jitter, matrix.dtype)
        * tf.eye(size, dtype=matrix.dtype)
    )


def spatial_conditional(
    *,
    points_km: tf.Tensor,
    inducing_km: tf.Tensor,
    inducing_covariance: tf.Tensor,
    inducing_mean: tf.Tensor,
    inducing_posterior_covariance: tf.Tensor,
    lengthscales_km: tf.Tensor,
    variance: tf.Tensor,
    jitter: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Map q(u_t) to q(g(s,t)) at arbitrary spatial points.

    Returns
    -------
    mean
        Sparse posterior residual mean at each point.
    variance
        Sparse posterior residual marginal variance at each point.
    weights
        K_xZ K_ZZ^{-1}; useful for CVI derivatives.
    """
    cross_covariance = matern32_spatial_covariance(
        points_km,
        inducing_km,
        lengthscales_km=lengthscales_km,
        variance=variance,
        dtype=dtype,
    )

    chol = cholesky_with_jitter(
        inducing_covariance,
        jitter=jitter,
    )

    solved = tf.linalg.cholesky_solve(
        chol,
        tf.transpose(cross_covariance),
    )
    weights = tf.transpose(solved)

    mean = tf.linalg.matvec(
        weights,
        inducing_mean,
    )

    posterior_projection = tf.reduce_sum(
        tf.linalg.matmul(
            weights,
            inducing_posterior_covariance,
        )
        * weights,
        axis=1,
    )

    prior_projection = tf.reduce_sum(
        weights * cross_covariance,
        axis=1,
    )

    conditional_residual = (
        tf.reshape(
            tf.convert_to_tensor(variance, dtype=dtype),
            [],
        )
        - prior_projection
    )

    marginal_variance = (
        posterior_projection
        + conditional_residual
    )

    # Only suppress numerical negatives at the level of machine/jitter noise.
    marginal_variance = tf.maximum(
        marginal_variance,
        tf.cast(jitter, dtype),
    )

    return mean, marginal_variance, weights
