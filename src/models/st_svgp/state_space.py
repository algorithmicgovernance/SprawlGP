"""TensorFlow state-space representation of a Matérn-3/2 temporal GP.

This module contains no Bernoulli likelihood, CVI update, filtering, smoothing,
or training loop. Its only purpose is to define and validate the temporal
Matérn-3/2 Markov prior that the ST-SVGP will later use.

For

    k(τ) = variance * (1 + λ |τ|) exp(-λ |τ|),
    λ = sqrt(3) / lengthscale,

the state is x(t) = [f(t), df(t)/dt]^T with

    dx/dt = F x + L w,

where the white-noise spectral density is chosen so that the stationary
variance of f(t) is exactly ``variance``.
"""

from __future__ import annotations

import tensorflow as tf


def _as_scalar(
    value: float | tf.Tensor,
    *,
    dtype: tf.dtypes.DType,
    name: str,
) -> tf.Tensor:
    """Convert a scalar-like value to the requested TensorFlow dtype."""
    tensor = tf.convert_to_tensor(value, dtype=dtype)
    tf.debugging.assert_rank(
        tensor,
        0,
        message=f"{name} must be a scalar.",
    )
    return tensor


def matern32_decay_rate(
    lengthscale: float | tf.Tensor,
    *,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return λ = sqrt(3) / lengthscale."""
    ell = _as_scalar(
        lengthscale,
        dtype=dtype,
        name="lengthscale",
    )
    tf.debugging.assert_positive(
        ell,
        message="lengthscale must be positive.",
    )
    return tf.sqrt(tf.cast(3.0, dtype)) / ell


def matern32_covariance(
    times_a: tf.Tensor,
    times_b: tf.Tensor | None = None,
    *,
    lengthscale: float | tf.Tensor,
    variance: float | tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Direct Matérn-3/2 covariance matrix.

    Parameters
    ----------
    times_a, times_b
        One-dimensional temporal coordinates expressed in model time units.
        For the current project, one unit is one five-year step.
    lengthscale
        Temporal lengthscale in the same units.
    variance
        Marginal variance of f(t).
    """
    a = tf.reshape(
        tf.convert_to_tensor(times_a, dtype=dtype),
        [-1],
    )
    b = (
        a
        if times_b is None
        else tf.reshape(
            tf.convert_to_tensor(times_b, dtype=dtype),
            [-1],
        )
    )

    var = _as_scalar(
        variance,
        dtype=dtype,
        name="variance",
    )
    tf.debugging.assert_positive(
        var,
        message="variance must be positive.",
    )

    lam = matern32_decay_rate(
        lengthscale,
        dtype=dtype,
    )
    distance = tf.abs(a[:, None] - b[None, :])
    scaled = lam * distance
    return var * (1.0 + scaled) * tf.exp(-scaled)


def matern32_continuous_matrices(
    *,
    lengthscale: float | tf.Tensor,
    variance: float | tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return the continuous-time Matérn-3/2 state-space matrices.

    Returns
    -------
    F
        Feedback matrix with shape (2, 2).
    L
        Noise-effect matrix with shape (2, 1).
    Qc
        Scalar white-noise spectral density.
    H
        Observation matrix with shape (1, 2).
    """
    var = _as_scalar(
        variance,
        dtype=dtype,
        name="variance",
    )
    tf.debugging.assert_positive(
        var,
        message="variance must be positive.",
    )

    lam = matern32_decay_rate(
        lengthscale,
        dtype=dtype,
    )

    zero = tf.cast(0.0, dtype)
    one = tf.cast(1.0, dtype)

    feedback = tf.stack(
        [
            tf.stack([zero, one]),
            tf.stack([-tf.square(lam), -2.0 * lam]),
        ]
    )

    noise_effect = tf.reshape(
        tf.stack([zero, one]),
        [2, 1],
    )

    # This choice makes Var[f(t)] = variance at stationarity.
    spectral_density = (
        4.0
        * tf.pow(lam, 3)
        * var
    )

    observation = tf.reshape(
        tf.stack([one, zero]),
        [1, 2],
    )

    return (
        feedback,
        noise_effect,
        spectral_density,
        observation,
    )


def matern32_stationary_covariance(
    *,
    lengthscale: float | tf.Tensor,
    variance: float | tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return the stationary covariance of [f, df/dt]."""
    var = _as_scalar(
        variance,
        dtype=dtype,
        name="variance",
    )
    tf.debugging.assert_positive(
        var,
        message="variance must be positive.",
    )

    lam = matern32_decay_rate(
        lengthscale,
        dtype=dtype,
    )

    zero = tf.cast(0.0, dtype)

    return tf.stack(
        [
            tf.stack([var, zero]),
            tf.stack([zero, var * tf.square(lam)]),
        ]
    )


def matern32_transition_matrix(
    delta: float | tf.Tensor,
    *,
    lengthscale: float | tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return the exact discrete transition matrix A(delta).

    The closed form is equivalent to expm(F * delta) for the Matérn-3/2 SDE.
    """
    dt = _as_scalar(
        delta,
        dtype=dtype,
        name="delta",
    )
    tf.debugging.assert_greater_equal(
        dt,
        tf.cast(0.0, dtype),
        message="delta must be non-negative.",
    )

    lam = matern32_decay_rate(
        lengthscale,
        dtype=dtype,
    )

    one = tf.cast(1.0, dtype)
    decay = tf.exp(-lam * dt)

    transition = tf.stack(
        [
            tf.stack(
                [
                    one + lam * dt,
                    dt,
                ]
            ),
            tf.stack(
                [
                    -tf.square(lam) * dt,
                    one - lam * dt,
                ]
            ),
        ]
    )

    return decay * transition


def matern32_process_noise(
    delta: float | tf.Tensor,
    *,
    lengthscale: float | tf.Tensor,
    variance: float | tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Return the exact discrete process-noise covariance Q(delta).

    For a stationary linear SDE,

        P_inf = A P_inf A^T + Q,

    so Q can be obtained without numerical integration.
    """
    transition = matern32_transition_matrix(
        delta,
        lengthscale=lengthscale,
        dtype=dtype,
    )
    stationary = matern32_stationary_covariance(
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    process_noise = (
        stationary
        - transition
        @ stationary
        @ tf.transpose(transition)
    )

    # Remove tiny floating-point asymmetry before Cholesky/eigendecomposition.
    return 0.5 * (
        process_noise
        + tf.transpose(process_noise)
    )


def matern32_state_space_covariance(
    times: tf.Tensor,
    *,
    lengthscale: float | tf.Tensor,
    variance: float | tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> tf.Tensor:
    """Recover Cov[f(t_i), f(t_j)] through the Markov representation."""
    temporal_coordinates = tf.reshape(
        tf.convert_to_tensor(times, dtype=dtype),
        [-1],
    )

    stationary = matern32_stationary_covariance(
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )
    _, _, _, observation = matern32_continuous_matrices(
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    pairwise_deltas = tf.abs(
        temporal_coordinates[:, None]
        - temporal_coordinates[None, :]
    )
    flat_deltas = tf.reshape(
        pairwise_deltas,
        [-1],
    )

    def covariance_for_delta(delta: tf.Tensor) -> tf.Tensor:
        transition = matern32_transition_matrix(
            delta,
            lengthscale=lengthscale,
            dtype=dtype,
        )
        value = (
            observation
            @ transition
            @ stationary
            @ tf.transpose(observation)
        )
        return tf.reshape(value, [])

    flat_covariance = tf.map_fn(
        covariance_for_delta,
        flat_deltas,
        fn_output_signature=dtype,
    )

    size = tf.shape(temporal_coordinates)[0]
    covariance = tf.reshape(
        flat_covariance,
        [size, size],
    )

    return 0.5 * (
        covariance
        + tf.transpose(covariance)
    )
