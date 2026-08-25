"""Dense-block CVI filtering/smoothing for spatial inducing states.

At each time t, u_t contains the M_s spatial inducing function values.
The temporal Matérn-3/2 prior is represented as a Markov state with two
components per inducing location. CVI contributes one dense Gaussian
pseudo-likelihood block per time.
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
class DenseCviSites:
    """Natural parameters of time-factorised dense Gaussian CVI sites."""

    lambda1: tf.Tensor   # [T, M]
    lambda2: tf.Tensor   # [T, M, M]


@dataclass(frozen=True)
class BlockFilterResult:
    """Sequential block Kalman-filter outputs."""

    predicted_means: tf.Tensor       # [T, 2M]
    predicted_covariances: tf.Tensor # [T, 2M, 2M]
    filtered_means: tf.Tensor        # [T, 2M]
    filtered_covariances: tf.Tensor  # [T, 2M, 2M]
    log_marginal_likelihood: tf.Tensor


@dataclass(frozen=True)
class BlockSmootherResult:
    """RTS smoother outputs."""

    smoothed_means: tf.Tensor
    smoothed_covariances: tf.Tensor


def _symmetrise(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (
        matrix + tf.transpose(matrix)
    )


def _solve_spd(
    matrix: tf.Tensor,
    rhs: tf.Tensor,
    *,
    jitter: float,
) -> tf.Tensor:
    matrix = _symmetrise(matrix)
    size = tf.shape(matrix)[0]
    chol = tf.linalg.cholesky(
        matrix
        + tf.cast(jitter, matrix.dtype)
        * tf.eye(size, dtype=matrix.dtype)
    )
    return tf.linalg.cholesky_solve(
        chol,
        rhs,
    )


def initialise_dense_sites(
    n_times: int,
    n_spatial: int,
    *,
    initial_precision: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> DenseCviSites:
    """Initialise weak zero-centred dense Gaussian sites."""
    if n_times < 1 or n_spatial < 1:
        raise ValueError(
            "n_times and n_spatial must be positive."
        )
    if initial_precision <= 0.0:
        raise ValueError(
            "initial_precision must be positive."
        )

    identity = tf.eye(
        n_spatial,
        batch_shape=[n_times],
        dtype=dtype,
    )

    return DenseCviSites(
        lambda1=tf.zeros(
            [n_times, n_spatial],
            dtype=dtype,
        ),
        lambda2=(
            -0.5
            * tf.cast(initial_precision, dtype)
            * identity
        ),
    )


def site_precision(
    sites: DenseCviSites,
) -> tf.Tensor:
    """Return symmetric positive precision blocks -2 lambda2."""
    precision = -2.0 * sites.lambda2
    return 0.5 * (
        precision
        + tf.transpose(
            precision,
            perm=[0, 2, 1],
        )
    )


def dense_sites_to_gaussian(
    sites: DenseCviSites,
    *,
    jitter: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert natural parameters to pseudo observations/covariances."""
    precision = site_precision(sites)

    pseudo_means: list[tf.Tensor] = []
    pseudo_covariances: list[tf.Tensor] = []

    n_times = int(tf.shape(precision)[0].numpy())
    n_spatial = int(tf.shape(precision)[1].numpy())

    identity = tf.eye(
        n_spatial,
        dtype=precision.dtype,
    )

    for index in range(n_times):
        current_precision = precision[index]
        chol = tf.linalg.cholesky(
            _symmetrise(current_precision)
            + tf.cast(jitter, precision.dtype)
            * identity
        )

        covariance = tf.linalg.cholesky_solve(
            chol,
            identity,
        )
        mean = tf.linalg.cholesky_solve(
            chol,
            sites.lambda1[index][:, None],
        )[:, 0]

        pseudo_means.append(mean)
        pseudo_covariances.append(
            _symmetrise(covariance)
        )

    return (
        tf.stack(pseudo_means, axis=0),
        tf.stack(pseudo_covariances, axis=0),
        precision,
    )


def temporal_block_matrices(
    *,
    spatial_covariance: tf.Tensor,
    delta: tf.Tensor,
    temporal_lengthscale: tf.Tensor,
    dtype: tf.dtypes.DType,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Build A, Q and H for M_s coupled Matérn-3/2 temporal states."""
    n_spatial = tf.shape(spatial_covariance)[0]

    temporal_a = matern32_transition_matrix(
        delta,
        lengthscale=temporal_lengthscale,
        dtype=dtype,
    )
    temporal_q = matern32_process_noise(
        delta,
        lengthscale=temporal_lengthscale,
        variance=tf.cast(1.0, dtype),
        dtype=dtype,
    )
    _, _, _, temporal_h = matern32_continuous_matrices(
        lengthscale=temporal_lengthscale,
        variance=tf.cast(1.0, dtype),
        dtype=dtype,
    )

    identity = tf.eye(
        n_spatial,
        dtype=dtype,
    )

    transition = tf.experimental.numpy.kron(
        identity,
        temporal_a,
    )
    process_noise = tf.experimental.numpy.kron(
        spatial_covariance,
        temporal_q,
    )
    observation = tf.experimental.numpy.kron(
        identity,
        temporal_h,
    )

    return (
        transition,
        _symmetrise(process_noise),
        observation,
    )


def initial_block_covariance(
    *,
    spatial_covariance: tf.Tensor,
    temporal_lengthscale: tf.Tensor,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Stationary covariance of the 2M-dimensional inducing state."""
    temporal_stationary = matern32_stationary_covariance(
        lengthscale=temporal_lengthscale,
        variance=tf.cast(1.0, dtype),
        dtype=dtype,
    )
    return _symmetrise(
        tf.experimental.numpy.kron(
            spatial_covariance,
            temporal_stationary,
        )
    )


def block_kalman_filter(
    *,
    times: tf.Tensor,
    pseudo_observations: tf.Tensor,
    pseudo_covariances: tf.Tensor,
    spatial_covariance: tf.Tensor,
    temporal_lengthscale: tf.Tensor,
    jitter: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> BlockFilterResult:
    """Run the sequential filter for dense M_s-dimensional CVI blocks."""
    times = tf.reshape(
        tf.convert_to_tensor(times, dtype=dtype),
        [-1],
    )
    pseudo_y = tf.convert_to_tensor(
        pseudo_observations,
        dtype=dtype,
    )
    pseudo_v = tf.convert_to_tensor(
        pseudo_covariances,
        dtype=dtype,
    )

    n_times = int(tf.shape(times)[0].numpy())
    n_spatial = int(
        tf.shape(spatial_covariance)[0].numpy()
    )

    tf.debugging.assert_equal(
        tf.shape(pseudo_y),
        [n_times, n_spatial],
    )
    tf.debugging.assert_equal(
        tf.shape(pseudo_v),
        [n_times, n_spatial, n_spatial],
    )

    if n_times > 1:
        tf.debugging.assert_positive(
            times[1:] - times[:-1],
            message="Training times must be strictly increasing.",
        )

    # H is time invariant. Delta=0 is sufficient to construct it.
    _, _, observation = temporal_block_matrices(
        spatial_covariance=spatial_covariance,
        delta=tf.cast(0.0, dtype),
        temporal_lengthscale=temporal_lengthscale,
        dtype=dtype,
    )

    state_dimension = 2 * n_spatial
    prior_mean = tf.zeros(
        [state_dimension],
        dtype=dtype,
    )
    prior_covariance = initial_block_covariance(
        spatial_covariance=spatial_covariance,
        temporal_lengthscale=temporal_lengthscale,
        dtype=dtype,
    )

    predicted_means: list[tf.Tensor] = []
    predicted_covariances: list[tf.Tensor] = []
    filtered_means: list[tf.Tensor] = []
    filtered_covariances: list[tf.Tensor] = []

    previous_mean = prior_mean
    previous_covariance = prior_covariance

    total_log_marginal = tf.cast(0.0, dtype)
    pi = tf.constant(3.141592653589793, dtype=dtype)
    log_two_pi = tf.math.log(
        tf.cast(2.0, dtype) * pi
    )

    for index in range(n_times):
        if index == 0:
            predicted_mean = prior_mean
            predicted_covariance = prior_covariance
        else:
            delta = times[index] - times[index - 1]
            transition, process_noise, _ = (
                temporal_block_matrices(
                    spatial_covariance=spatial_covariance,
                    delta=delta,
                    temporal_lengthscale=temporal_lengthscale,
                    dtype=dtype,
                )
            )

            predicted_mean = tf.linalg.matvec(
                transition,
                previous_mean,
            )
            predicted_covariance = (
                transition
                @ previous_covariance
                @ tf.transpose(transition)
                + process_noise
            )
            predicted_covariance = _symmetrise(
                predicted_covariance
            )

        predicted_observation = tf.linalg.matvec(
            observation,
            predicted_mean,
        )
        innovation = (
            pseudo_y[index]
            - predicted_observation
        )

        innovation_covariance = (
            observation
            @ predicted_covariance
            @ tf.transpose(observation)
            + pseudo_v[index]
        )
        innovation_covariance = _symmetrise(
            innovation_covariance
        )

        cross_covariance = (
            predicted_covariance
            @ tf.transpose(observation)
        )

        gain = tf.transpose(
            _solve_spd(
                innovation_covariance,
                tf.transpose(cross_covariance),
                jitter=jitter,
            )
        )

        filtered_mean = (
            predicted_mean
            + tf.linalg.matvec(
                gain,
                innovation,
            )
        )

        filtered_covariance = (
            predicted_covariance
            - gain
            @ innovation_covariance
            @ tf.transpose(gain)
        )
        filtered_covariance = _symmetrise(
            filtered_covariance
        )

        chol_s = tf.linalg.cholesky(
            innovation_covariance
            + tf.cast(jitter, dtype)
            * tf.eye(n_spatial, dtype=dtype)
        )
        solved_innovation = tf.linalg.cholesky_solve(
            chol_s,
            innovation[:, None],
        )[:, 0]

        logdet_s = (
            2.0
            * tf.reduce_sum(
                tf.math.log(
                    tf.linalg.diag_part(chol_s)
                )
            )
        )

        total_log_marginal = (
            total_log_marginal
            - 0.5
            * (
                tf.cast(n_spatial, dtype)
                * log_two_pi
                + logdet_s
                + tf.tensordot(
                    innovation,
                    solved_innovation,
                    axes=1,
                )
            )
        )

        predicted_means.append(predicted_mean)
        predicted_covariances.append(
            predicted_covariance
        )
        filtered_means.append(filtered_mean)
        filtered_covariances.append(
            filtered_covariance
        )

        previous_mean = filtered_mean
        previous_covariance = filtered_covariance

    return BlockFilterResult(
        predicted_means=tf.stack(
            predicted_means,
            axis=0,
        ),
        predicted_covariances=tf.stack(
            predicted_covariances,
            axis=0,
        ),
        filtered_means=tf.stack(
            filtered_means,
            axis=0,
        ),
        filtered_covariances=tf.stack(
            filtered_covariances,
            axis=0,
        ),
        log_marginal_likelihood=total_log_marginal,
    )


def block_rts_smoother(
    *,
    times: tf.Tensor,
    filter_result: BlockFilterResult,
    spatial_covariance: tf.Tensor,
    temporal_lengthscale: tf.Tensor,
    jitter: float,
    dtype: tf.dtypes.DType = tf.float64,
) -> BlockSmootherResult:
    """Run the backward RTS smoother on the dense inducing state."""
    times = tf.reshape(
        tf.convert_to_tensor(times, dtype=dtype),
        [-1],
    )
    n_times = int(tf.shape(times)[0].numpy())

    smoothed_means: list[tf.Tensor | None] = (
        [None] * n_times
    )
    smoothed_covariances: list[tf.Tensor | None] = (
        [None] * n_times
    )

    smoothed_means[-1] = (
        filter_result.filtered_means[-1]
    )
    smoothed_covariances[-1] = (
        filter_result.filtered_covariances[-1]
    )

    for index in range(n_times - 2, -1, -1):
        delta = times[index + 1] - times[index]
        transition, _, _ = temporal_block_matrices(
            spatial_covariance=spatial_covariance,
            delta=delta,
            temporal_lengthscale=temporal_lengthscale,
            dtype=dtype,
        )

        filtered_covariance = (
            filter_result.filtered_covariances[index]
        )
        predicted_next = (
            filter_result.predicted_covariances[index + 1]
        )

        cross = (
            filtered_covariance
            @ tf.transpose(transition)
        )

        smoother_gain = tf.transpose(
            _solve_spd(
                predicted_next,
                tf.transpose(cross),
                jitter=jitter,
            )
        )

        next_mean = smoothed_means[index + 1]
        next_covariance = (
            smoothed_covariances[index + 1]
        )

        assert next_mean is not None
        assert next_covariance is not None

        smoothed_mean = (
            filter_result.filtered_means[index]
            + tf.linalg.matvec(
                smoother_gain,
                (
                    next_mean
                    - filter_result.predicted_means[
                        index + 1
                    ]
                ),
            )
        )

        smoothed_covariance = (
            filtered_covariance
            + smoother_gain
            @ (
                next_covariance
                - predicted_next
            )
            @ tf.transpose(smoother_gain)
        )

        smoothed_means[index] = smoothed_mean
        smoothed_covariances[index] = (
            _symmetrise(smoothed_covariance)
        )

    return BlockSmootherResult(
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


def inducing_function_marginals(
    *,
    state_means: tf.Tensor,
    state_covariances: tf.Tensor,
    n_spatial: int,
    temporal_lengthscale: tf.Tensor,
    dtype: tf.dtypes.DType,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Extract q(u_t) function-value marginals from 2M states."""
    identity = tf.eye(
        n_spatial,
        dtype=dtype,
    )
    _, _, _, temporal_h = matern32_continuous_matrices(
        lengthscale=temporal_lengthscale,
        variance=tf.cast(1.0, dtype),
        dtype=dtype,
    )
    observation = tf.experimental.numpy.kron(
        identity,
        temporal_h,
    )

    means: list[tf.Tensor] = []
    covariances: list[tf.Tensor] = []

    n_times = int(
        tf.shape(state_means)[0].numpy()
    )

    for index in range(n_times):
        mean = tf.linalg.matvec(
            observation,
            state_means[index],
        )
        covariance = (
            observation
            @ state_covariances[index]
            @ tf.transpose(observation)
        )
        means.append(mean)
        covariances.append(
            _symmetrise(covariance)
        )

    return (
        tf.stack(means, axis=0),
        tf.stack(covariances, axis=0),
    )


def expected_log_dense_sites(
    *,
    sites: DenseCviSites,
    inducing_means: tf.Tensor,
    inducing_covariances: tf.Tensor,
    jitter: float,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """Compute sum_t E_q log N(y_tilde_t | u_t, V_tilde_t)."""
    pseudo_y, _, precision = (
        dense_sites_to_gaussian(
            sites,
            jitter=jitter,
        )
    )

    n_times = int(
        tf.shape(inducing_means)[0].numpy()
    )
    n_spatial = int(
        tf.shape(inducing_means)[1].numpy()
    )

    pi = tf.constant(3.141592653589793, dtype=dtype)
    total = tf.cast(0.0, dtype)

    for index in range(n_times):
        current_precision = precision[index]
        chol_precision = tf.linalg.cholesky(
            _symmetrise(current_precision)
            + tf.cast(jitter, dtype)
            * tf.eye(n_spatial, dtype=dtype)
        )

        logdet_precision = (
            2.0
            * tf.reduce_sum(
                tf.math.log(
                    tf.linalg.diag_part(
                        chol_precision
                    )
                )
            )
        )

        difference = (
            pseudo_y[index]
            - inducing_means[index]
        )

        quadratic = tf.tensordot(
            difference,
            tf.linalg.matvec(
                current_precision,
                difference,
            ),
            axes=1,
        )

        trace_term = tf.linalg.trace(
            current_precision
            @ inducing_covariances[index]
        )

        # logdet(V) = -logdet(precision)
        total = total - 0.5 * (
            tf.cast(n_spatial, dtype)
            * tf.math.log(
                tf.cast(2.0, dtype) * pi
            )
            - logdet_precision
            + quadratic
            + trace_term
        )

    return total


def damped_dense_natural_gradient_update(
    *,
    sites: DenseCviSites,
    target_lambda1: tf.Tensor,
    target_lambda2: tf.Tensor,
    requested_gamma: float,
    minimum_gamma: float,
    maximum_retries: int,
    minimum_eigenvalue: float,
) -> tuple[DenseCviSites, float]:
    """Apply a CVI natural-gradient update with SPD-preserving backtracking."""
    gamma = float(requested_gamma)

    for _ in range(maximum_retries + 1):
        step = tf.cast(gamma, sites.lambda1.dtype)

        candidate_lambda1 = (
            (1.0 - step) * sites.lambda1
            + step * target_lambda1
        )
        candidate_lambda2 = (
            (1.0 - step) * sites.lambda2
            + step * target_lambda2
        )

        candidate = DenseCviSites(
            lambda1=candidate_lambda1,
            lambda2=0.5 * (
                candidate_lambda2
                + tf.transpose(
                    candidate_lambda2,
                    perm=[0, 2, 1],
                )
            ),
        )

        precision = site_precision(candidate)

        minimum = float(
            tf.reduce_min(
                tf.linalg.eigvalsh(precision)
            ).numpy()
        )

        finite = bool(
            tf.reduce_all(
                tf.math.is_finite(precision)
            ).numpy()
        )

        if (
            finite
            and minimum >= minimum_eigenvalue
        ):
            return candidate, gamma

        gamma *= 0.5

        if gamma < minimum_gamma:
            break

    raise RuntimeError(
        "CVI Natural-Gradient update could not preserve "
        "positive-definite dense site precision."
    )
