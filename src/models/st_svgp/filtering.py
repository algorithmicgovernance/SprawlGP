"""Sequential Kalman filtering and RTS smoothing for the ST-SVGP prior.

This module is Gaussian-only. It is shared by:
- the Gaussian filter/smoother validation experiment;
- the CVI pseudo-likelihood posterior used by ST-SVGP.

The observation noise may be either a scalar or one value per time step.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from src.models.st_svgp.state_space import (
    matern32_continuous_matrices,
    matern32_process_noise,
    matern32_stationary_covariance,
    matern32_transition_matrix,
)


@dataclass(frozen=True)
class FilterResult:
    """Outputs of the sequential Kalman filter."""

    predicted_means: tf.Tensor
    predicted_covariances: tf.Tensor
    filtered_means: tf.Tensor
    filtered_covariances: tf.Tensor
    log_marginal_likelihood: tf.Tensor


@dataclass(frozen=True)
class SmootherResult:
    """Outputs of the backward Rauch-Tung-Striebel smoother."""

    smoothed_means: tf.Tensor
    smoothed_covariances: tf.Tensor


def _symmetrise(matrix: tf.Tensor) -> tf.Tensor:
    """Remove tiny floating-point asymmetry."""
    return 0.5 * (matrix + tf.transpose(matrix))


def _solve_spd(
    matrix: tf.Tensor,
    right_hand_side: tf.Tensor,
) -> tf.Tensor:
    """Solve an SPD linear system through Cholesky factorisation."""
    chol = tf.linalg.cholesky(_symmetrise(matrix))
    return tf.linalg.cholesky_solve(chol, right_hand_side)


def _observation_noise_vector(
    observation_noise_variance: float | tf.Tensor,
    *,
    n_times: tf.Tensor,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Return one positive observation-noise variance per time step."""
    noise = tf.convert_to_tensor(
        observation_noise_variance,
        dtype=dtype,
    )

    rank = noise.shape.rank

    if rank == 0:
        result = tf.fill(
            [n_times],
            noise,
        )
    elif rank == 1:
        tf.debugging.assert_equal(
            tf.shape(noise)[0],
            n_times,
            message=(
                "Vector observation_noise_variance must match "
                "the number of time points."
            ),
        )
        result = noise
    else:
        raise ValueError(
            "observation_noise_variance must be scalar or one-dimensional."
        )

    tf.debugging.assert_positive(
        result,
        message="Observation-noise variances must be positive.",
    )

    return result


def kalman_filter_matern32(
    *,
    times: tf.Tensor,
    observations: tf.Tensor,
    observation_noise_variance: float | tf.Tensor,
    lengthscale: float | tf.Tensor,
    variance: float | tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> FilterResult:
    """Run a Kalman filter for a Matérn-3/2 temporal GP.

    State
    -----
    x(t) = [f(t), df(t)/dt]^T

    Observation
    -----------
    y_t = H x_t + epsilon_t

    ``observation_noise_variance`` may be a scalar or a vector with one
    variance per time step. The vector case is required by CVI, where every
    pseudo-likelihood factor has its own Gaussian variance.
    """
    temporal_coordinates = tf.reshape(
        tf.convert_to_tensor(times, dtype=dtype),
        [-1],
    )
    y = tf.reshape(
        tf.convert_to_tensor(observations, dtype=dtype),
        [-1],
    )

    n_times_tensor = tf.shape(temporal_coordinates)[0]

    tf.debugging.assert_equal(
        n_times_tensor,
        tf.shape(y)[0],
        message="times and observations must have the same length.",
    )
    tf.debugging.assert_greater_equal(
        n_times_tensor,
        1,
        message="At least one observation is required.",
    )

    n_times = int(n_times_tensor.numpy())

    if n_times > 1:
        deltas = temporal_coordinates[1:] - temporal_coordinates[:-1]
        tf.debugging.assert_positive(
            deltas,
            message="times must be strictly increasing.",
        )

    observation_noise = _observation_noise_vector(
        observation_noise_variance,
        n_times=n_times_tensor,
        dtype=dtype,
    )

    _, _, _, observation_matrix = matern32_continuous_matrices(
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    prior_mean = tf.zeros([2], dtype=dtype)
    prior_covariance = matern32_stationary_covariance(
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    predicted_means: list[tf.Tensor] = []
    predicted_covariances: list[tf.Tensor] = []
    filtered_means: list[tf.Tensor] = []
    filtered_covariances: list[tf.Tensor] = []

    total_log_marginal = tf.cast(0.0, dtype)

    previous_filtered_mean = prior_mean
    previous_filtered_covariance = prior_covariance

    observation_vector = tf.reshape(observation_matrix, [2])
    identity = tf.eye(2, dtype=dtype)
    pi = tf.constant(3.141592653589793, dtype=dtype)
    log_two_pi = tf.math.log(tf.cast(2.0, dtype) * pi)

    for index in range(n_times):
        if index == 0:
            predicted_mean = prior_mean
            predicted_covariance = prior_covariance
        else:
            delta = (
                temporal_coordinates[index]
                - temporal_coordinates[index - 1]
            )

            transition = matern32_transition_matrix(
                delta,
                lengthscale=lengthscale,
                dtype=dtype,
            )
            process_noise = matern32_process_noise(
                delta,
                lengthscale=lengthscale,
                variance=variance,
                dtype=dtype,
            )

            predicted_mean = tf.linalg.matvec(
                transition,
                previous_filtered_mean,
            )
            predicted_covariance = (
                transition
                @ previous_filtered_covariance
                @ tf.transpose(transition)
                + process_noise
            )
            predicted_covariance = _symmetrise(
                predicted_covariance
            )

        predicted_observation = tf.tensordot(
            observation_vector,
            predicted_mean,
            axes=1,
        )
        innovation = y[index] - predicted_observation

        current_noise = observation_noise[index]

        projected_variance = (
            tf.tensordot(
                observation_vector,
                tf.linalg.matvec(
                    predicted_covariance,
                    observation_vector,
                ),
                axes=1,
            )
            + current_noise
        )

        tf.debugging.assert_positive(
            projected_variance,
            message="Innovation variance must be positive.",
        )

        kalman_gain = (
            tf.linalg.matvec(
                predicted_covariance,
                observation_vector,
            )
            / projected_variance
        )

        filtered_mean = (
            predicted_mean
            + kalman_gain * innovation
        )

        # Joseph form.
        kh = tf.tensordot(
            kalman_gain,
            observation_vector,
            axes=0,
        )
        update = identity - kh

        filtered_covariance = (
            update
            @ predicted_covariance
            @ tf.transpose(update)
            + current_noise
            * tf.tensordot(
                kalman_gain,
                kalman_gain,
                axes=0,
            )
        )
        filtered_covariance = _symmetrise(
            filtered_covariance
        )

        log_increment = -0.5 * (
            log_two_pi
            + tf.math.log(projected_variance)
            + tf.square(innovation) / projected_variance
        )
        total_log_marginal = (
            total_log_marginal
            + log_increment
        )

        predicted_means.append(predicted_mean)
        predicted_covariances.append(predicted_covariance)
        filtered_means.append(filtered_mean)
        filtered_covariances.append(filtered_covariance)

        previous_filtered_mean = filtered_mean
        previous_filtered_covariance = filtered_covariance

    return FilterResult(
        predicted_means=tf.stack(predicted_means, axis=0),
        predicted_covariances=tf.stack(
            predicted_covariances,
            axis=0,
        ),
        filtered_means=tf.stack(filtered_means, axis=0),
        filtered_covariances=tf.stack(
            filtered_covariances,
            axis=0,
        ),
        log_marginal_likelihood=total_log_marginal,
    )


def rts_smoother_matern32(
    *,
    times: tf.Tensor,
    filter_result: FilterResult,
    lengthscale: float | tf.Tensor,
    dtype: tf.dtypes.DType = tf.float64,
) -> SmootherResult:
    """Run the backward Rauch-Tung-Striebel smoother."""
    temporal_coordinates = tf.reshape(
        tf.convert_to_tensor(times, dtype=dtype),
        [-1],
    )

    filtered_means = filter_result.filtered_means
    filtered_covariances = filter_result.filtered_covariances
    predicted_means = filter_result.predicted_means
    predicted_covariances = filter_result.predicted_covariances

    n_times = int(tf.shape(temporal_coordinates)[0].numpy())

    smoothed_means: list[tf.Tensor | None] = [None] * n_times
    smoothed_covariances: list[tf.Tensor | None] = [None] * n_times

    smoothed_means[-1] = filtered_means[-1]
    smoothed_covariances[-1] = filtered_covariances[-1]

    for index in range(n_times - 2, -1, -1):
        delta = (
            temporal_coordinates[index + 1]
            - temporal_coordinates[index]
        )

        transition = matern32_transition_matrix(
            delta,
            lengthscale=lengthscale,
            dtype=dtype,
        )

        filtered_covariance = filtered_covariances[index]
        predicted_next_covariance = (
            predicted_covariances[index + 1]
        )

        cross_covariance = (
            filtered_covariance
            @ tf.transpose(transition)
        )

        smoother_gain = tf.transpose(
            _solve_spd(
                predicted_next_covariance,
                tf.transpose(cross_covariance),
            )
        )

        next_smoothed_mean = smoothed_means[index + 1]
        next_smoothed_covariance = smoothed_covariances[index + 1]

        assert next_smoothed_mean is not None
        assert next_smoothed_covariance is not None

        smoothed_mean = (
            filtered_means[index]
            + tf.linalg.matvec(
                smoother_gain,
                (
                    next_smoothed_mean
                    - predicted_means[index + 1]
                ),
            )
        )

        smoothed_covariance = (
            filtered_covariance
            + smoother_gain
            @ (
                next_smoothed_covariance
                - predicted_next_covariance
            )
            @ tf.transpose(smoother_gain)
        )
        smoothed_covariance = _symmetrise(
            smoothed_covariance
        )

        smoothed_means[index] = smoothed_mean
        smoothed_covariances[index] = smoothed_covariance

    return SmootherResult(
        smoothed_means=tf.stack(
            [
                value
                for value in smoothed_means
                if value is not None
            ],
            axis=0,
        ),
        smoothed_covariances=tf.stack(
            [
                value
                for value in smoothed_covariances
                if value is not None
            ],
            axis=0,
        ),
    )


def function_marginals_from_states(
    *,
    state_means: tf.Tensor,
    state_covariances: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Extract marginal mean and variance of f(t)."""
    return (
        state_means[:, 0],
        state_covariances[:, 0, 0],
    )
