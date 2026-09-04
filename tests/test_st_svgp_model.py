"""Focused tests for the real spatial ST-SVGP implementation."""

from __future__ import annotations

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
    from pathlib import Path

    import yaml

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


def test_inducing_locations_can_be_trainable_through_spatial_path() -> None:
    from src.models.st_svgp.model import STSVGPModel

    initial_locations = tf.constant(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        dtype=DTYPE,
    )

    def make_model(*, trainable: bool = False) -> STSVGPModel:
        with tf.device("/CPU:0"):
            return STSVGPModel(
                inducing_locations_km=initial_locations,
                n_features=1,
                spatial_initial_lengthscale_km=2.0,
                temporal_initial_lengthscale_steps=1.5,
                variance_initial=1.0,
                jitter=1.0e-6,
                quadrature_degree=20,
                inducing_locations_trainable=trainable,
                dtype=DTYPE,
            )

    fixed_model = make_model()
    assert not isinstance(fixed_model.inducing_locations_km, tf.Variable)
    assert all(
        variable is not fixed_model.inducing_locations_km
        for variable in fixed_model.trainable_variables
    )
    fixed_covariance = fixed_model.spatial_covariance()
    assert tuple(fixed_covariance.shape) == (3, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(fixed_covariance)).numpy())

    model = make_model(trainable=True)
    assert isinstance(model.inducing_locations_km, tf.Variable)
    assert model.inducing_locations_km.trainable
    assert sum(
        variable is model.inducing_locations_km
        for variable in model.trainable_variables
    ) == 1

    def spatial_loss() -> tf.Tensor:
        spatial_covariance = model.spatial_covariance()
        latent_mean, latent_variance = model.latent_marginals(
            features=tf.zeros([1, 1], dtype=DTYPE),
            coordinates_km=tf.constant([[0.2, 0.3]], dtype=DTYPE),
            inducing_mean=tf.constant([0.4, -0.2, 0.1], dtype=DTYPE),
            inducing_covariance=0.5 * spatial_covariance,
            spatial_covariance=spatial_covariance,
        )
        return tf.reduce_sum(latent_mean + latent_variance)

    with tf.GradientTape() as tape:
        loss = spatial_loss()
    gradient = tape.gradient(loss, model.inducing_locations_km)

    assert gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
    assert bool(tf.reduce_any(tf.not_equal(gradient, 0.0)).numpy())

    before = model.inducing_locations_km.numpy().copy()
    model.inducing_locations_km.assign_sub(
        tf.cast(1.0e-4, DTYPE) * gradient
    )
    after = model.inducing_locations_km.numpy()
    assert bool(tf.reduce_all(tf.math.is_finite(after)).numpy())
    assert bool(tf.reduce_any(tf.not_equal(before, after)).numpy())


def test_linear_temporal_trend_mean_equivalence_effect_and_gradient() -> None:
    from src.models.st_svgp.model import STSVGPModel

    def make_model(*, enabled: bool) -> STSVGPModel:
        return STSVGPModel(
            inducing_locations_km=tf.constant(
                [[0.0, 0.0], [1.0, 0.0]],
                dtype=DTYPE,
            ),
            n_features=2,
            spatial_initial_lengthscale_km=2.0,
            temporal_initial_lengthscale_steps=1.5,
            variance_initial=1.0,
            jitter=1.0e-6,
            quadrature_degree=20,
            temporal_trend_enabled=enabled,
            temporal_trend_initial_coefficient=0.0,
            dtype=DTYPE,
        )

    features = tf.constant(
        [[1.0, 2.0], [-1.0, 0.5]],
        dtype=DTYPE,
    )
    disabled = make_model(enabled=False)
    enabled = make_model(enabled=True)
    disabled.beta0.assign(0.4)
    disabled.beta.assign([0.2, -0.3])
    enabled.beta0.assign(disabled.beta0)
    enabled.beta.assign(disabled.beta)

    baseline_mean = disabled.linear_mean(features)
    tf.debugging.assert_near(
        disabled.linear_mean(
            features,
            temporal_trend_time=1.0,
        ),
        baseline_mean,
    )
    tf.debugging.assert_near(
        enabled.linear_mean(
            features,
            temporal_trend_time=1.0,
        ),
        baseline_mean,
    )

    assert enabled.beta_time is not None
    enabled.beta_time.assign(0.25)
    trended_mean = enabled.linear_mean(
        features,
        temporal_trend_time=1.0,
    )
    tf.debugging.assert_near(
        trended_mean - baseline_mean,
        tf.fill(
            tf.shape(baseline_mean),
            tf.constant(0.25, dtype=DTYPE),
        ),
    )

    with tf.GradientTape() as tape:
        objective = tf.reduce_sum(
            enabled.linear_mean(
                features,
                temporal_trend_time=1.0,
            )
        )
    gradient = tape.gradient(objective, enabled.beta_time)

    assert gradient is not None
    assert bool(tf.math.is_finite(gradient).numpy())
    tf.debugging.assert_near(
        gradient,
        tf.constant(2.0, dtype=DTYPE),
    )

