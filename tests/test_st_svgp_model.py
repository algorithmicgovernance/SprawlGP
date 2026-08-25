"""Focused tests for the real spatial ST-SVGP implementation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.models.st_svgp.block_inference import (
    DenseCviSites,
    block_kalman_filter,
    block_rts_smoother,
    dense_sites_to_gaussian,
    inducing_function_marginals,
    initialise_dense_sites,
)
from src.models.st_svgp.spatial import (
    matern32_spatial_covariance,
    spatial_conditional,
)
from src.models.st_svgp.state_space import (
    matern32_covariance,
)


DTYPE = tf.float64


def test_spatial_matern32_covariance_is_symmetric_psd() -> None:
    points = tf.constant(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, -0.5],
            [0.25, 1.5],
        ],
        dtype=DTYPE,
    )

    covariance = matern32_spatial_covariance(
        points,
        points,
        lengthscales_km=tf.constant(
            [2.0, 1.5],
            dtype=DTYPE,
        ),
        variance=tf.constant(
            1.2,
            dtype=DTYPE,
        ),
        dtype=DTYPE,
    )

    tf.debugging.assert_near(
        covariance,
        tf.transpose(covariance),
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    minimum = tf.reduce_min(
        tf.linalg.eigvalsh(covariance)
    )
    assert float(minimum.numpy()) >= -1.0e-10


def test_spatial_sparse_conditional_variance_is_positive() -> None:
    inducing = tf.constant(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [0.5, 1.5],
        ],
        dtype=DTYPE,
    )
    points = tf.constant(
        [
            [0.25, 0.10],
            [1.0, 0.8],
        ],
        dtype=DTYPE,
    )

    kzz = matern32_spatial_covariance(
        inducing,
        inducing,
        lengthscales_km=tf.constant(
            [2.0, 2.0],
            dtype=DTYPE,
        ),
        variance=tf.constant(
            1.0,
            dtype=DTYPE,
        ),
        dtype=DTYPE,
    )

    _, variance, _ = spatial_conditional(
        points_km=points,
        inducing_km=inducing,
        inducing_covariance=kzz,
        inducing_mean=tf.zeros(
            [3],
            dtype=DTYPE,
        ),
        inducing_posterior_covariance=(
            0.5 * kzz
        ),
        lengthscales_km=tf.constant(
            [2.0, 2.0],
            dtype=DTYPE,
        ),
        variance=tf.constant(
            1.0,
            dtype=DTYPE,
        ),
        jitter=1.0e-9,
        dtype=DTYPE,
    )

    assert bool(
        tf.reduce_all(
            variance > 0.0
        ).numpy()
    )


def test_block_filter_smoother_matches_dense_gaussian_inducing_posterior() -> None:
    """Structured state-space inference must match a dense GP oracle."""
    inducing = tf.constant(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ],
        dtype=DTYPE,
    )
    times = tf.constant(
        [0.0, 0.7, 1.8],
        dtype=DTYPE,
    )

    spatial_covariance = matern32_spatial_covariance(
        inducing,
        inducing,
        lengthscales_km=tf.constant(
            [1.5, 1.5],
            dtype=DTYPE,
        ),
        variance=tf.constant(
            1.2,
            dtype=DTYPE,
        ),
        dtype=DTYPE,
    )

    sites = initialise_dense_sites(
        3,
        3,
        initial_precision=0.4,
        dtype=DTYPE,
    )
    sites = DenseCviSites(
        lambda1=tf.constant(
            [
                [0.2, -0.1, 0.3],
                [0.1, 0.25, -0.2],
                [-0.2, 0.15, 0.35],
            ],
            dtype=DTYPE,
        ),
        lambda2=sites.lambda2,
    )

    pseudo_y, pseudo_v, _ = (
        dense_sites_to_gaussian(
            sites,
            jitter=1.0e-12,
        )
    )

    filtered = block_kalman_filter(
        times=times,
        pseudo_observations=pseudo_y,
        pseudo_covariances=pseudo_v,
        spatial_covariance=spatial_covariance,
        temporal_lengthscale=tf.constant(
            1.4,
            dtype=DTYPE,
        ),
        jitter=1.0e-12,
        dtype=DTYPE,
    )

    smoothed = block_rts_smoother(
        times=times,
        filter_result=filtered,
        spatial_covariance=spatial_covariance,
        temporal_lengthscale=tf.constant(
            1.4,
            dtype=DTYPE,
        ),
        jitter=1.0e-12,
        dtype=DTYPE,
    )

    state_mean, state_covariance = (
        inducing_function_marginals(
            state_means=(
                smoothed.smoothed_means
            ),
            state_covariances=(
                smoothed.smoothed_covariances
            ),
            n_spatial=3,
            temporal_lengthscale=tf.constant(
                1.4,
                dtype=DTYPE,
            ),
            dtype=DTYPE,
        )
    )

    temporal_covariance = matern32_covariance(
        times,
        lengthscale=1.4,
        variance=1.0,
        dtype=DTYPE,
    )

    # Time-major stack: [u_t1, u_t2, ...].
    prior = tf.experimental.numpy.kron(
        temporal_covariance,
        spatial_covariance,
    )
    noise = tf.linalg.LinearOperatorBlockDiag(
        [
            tf.linalg.LinearOperatorFullMatrix(
                pseudo_v[index]
            )
            for index in range(3)
        ]
    ).to_dense()

    system = prior + noise
    chol = tf.linalg.cholesky(system)

    y_flat = tf.reshape(
        pseudo_y,
        [-1, 1],
    )
    alpha = tf.linalg.cholesky_solve(
        chol,
        y_flat,
    )

    dense_mean = tf.reshape(
        prior @ alpha,
        [3, 3],
    )

    solved = tf.linalg.cholesky_solve(
        chol,
        prior,
    )
    dense_covariance = (
        prior
        - prior @ solved
    )

    dense_marginal_covariances = tf.stack(
        [
            dense_covariance[
                index * 3 : (index + 1) * 3,
                index * 3 : (index + 1) * 3,
            ]
            for index in range(3)
        ],
        axis=0,
    )

    tf.debugging.assert_near(
        state_mean,
        dense_mean,
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    tf.debugging.assert_near(
        state_covariance,
        dense_marginal_covariances,
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_real_config_contract_keeps_2020_locked() -> None:
    """The production model must not expose the final test during development."""
    import yaml
    from pathlib import Path

    path = Path(
        "configs/modeling/st_svgp.yaml" 
    )
    config = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert config["final_test"]["origins"] == [2020]
    assert config["final_test"]["evaluate"] is False
    assert config["inducing"]["spatial_points"] == 64
    assert (
        config["inference"]["method"]
        == "cvi_natural_gradient"
    )
    assert (
        len(config["linear_predictors"])
        == 9
    )


def test_temporal_lengthscale_can_be_frozen() -> None:
    """Frozen ell_t must be non-trainable and excluded from Adam variables."""
    from src.models.st_svgp.model import STSVGPModel

    model = STSVGPModel(
        inducing_locations_km=tf.constant(
            [[0.0, 0.0], [1.0, 0.0]],
            dtype=DTYPE,
        ),
        n_features=9,
        spatial_initial_lengthscale_km=2.0,
        temporal_initial_lengthscale_steps=1.5,
        variance_initial=1.0,
        jitter=1.0e-6,
        quadrature_degree=20,
        temporal_lengthscale_trainable=False,
        dtype=DTYPE,
    )

    assert model.temporal_lengthscale_trainable is False
    assert model.log_temporal_lengthscale.trainable is False
    assert all(
        variable is not model.log_temporal_lengthscale
        for variable in model.trainable_variables
    )
    tf.debugging.assert_near(
        model.temporal_lengthscale,
        tf.constant(1.5, dtype=DTYPE),
        atol=1.0e-12,
        rtol=1.0e-12,
    )

