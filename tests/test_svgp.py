"""Essential tests for the first SVGP implementation."""

from __future__ import annotations

import numpy as np
import pytest

gpflow = pytest.importorskip("gpflow")
# from gpflow.optimizers.natgrad import XiSqrtMeanVar
from gpflow.keras import tf_keras

import tensorflow as tf

# Metal GPU backend (tensorflow-metal) when running GPflow variational
# parameter updates in the test harness. This specific test context triggers
# an abort on Apple Silicon, while the full training pipeline handles the GPU
# correctly and will use the 'auto' device setting (float32) specified in the
# experiment configuration.
tf.config.set_visible_devices([], "GPU")

from src.models.train_svgp import (  # noqa: E402
    build_inducing_grid,
    build_model,
    validate_temporal_contract,
)


def _config() -> dict:
    return {
        "kernel": {
            "family": "matern32_separable",
            "spatial_initial_lengthscale_km": 2.0,
            "temporal_initial_lengthscale_steps": 1.0,
            "variance": 1.0,
        },
        "inducing": {
            "spatial_points": 2,
            "train_locations": False,
            "kmeans_n_init": 2,
        },
        "training": {
            "random_state": 7,
        },
        "rolling_validation": {
            "folds": [
                {"train_origins": [2000], "validation_origin": 2005},
                {
                    "train_origins": [2000, 2005],
                    "validation_origin": 2010,
                },
            ]
        },
        "final_fit": {
            "origins": [2000, 2005, 2010, 2015],
        },
        "final_test": {
            "origins": [2020],
            "evaluate": False,
        },
    }


def test_inducing_grid_repeats_spatial_support_over_time() -> None:
    """SVGP inducing points must use the same spatial support at each time."""
    # two covariates + x + y + time
    x = np.array(
        [
            [0, 0, 0.0, 0.0, 0.0],
            [0, 0, 1.0, 0.0, 0.0],
            [0, 0, 0.0, 1.0, 1.0],
            [0, 0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    z = build_inducing_grid(
        x,
        n_covariates=2,
        config=_config(),
    )

    assert z.shape == (4, 5)
    np.testing.assert_allclose(
        np.sort(z[:2, 2:4], axis=0),
        np.sort(z[2:, 2:4], axis=0),
    )
    assert set(z[:, 4]) == {0.0, 1.0}


# def test_model_matches_required_svgp_architecture() -> None:
#     """The model must be Bernoulli-probit with separable Matérn-3/2 kernel."""
#     z = np.zeros((4, 5), dtype=float)

#     model = build_model(
#         inducing_points=z,
#         n_covariates=2,
#         num_data=10,
#         config=_config(),
#     )

#     assert isinstance(model, gpflow.models.SVGP)
#     assert isinstance(model.likelihood, gpflow.likelihoods.Bernoulli)

#     spatial, temporal = model.kernel.kernels
#     assert isinstance(spatial, gpflow.kernels.Matern32)
#     assert isinstance(temporal, gpflow.kernels.Matern32)

#     # Full q(u)=N(m,S) is required for NaturalGradient.
#     # GPflow stores a full single-output covariance Cholesky factor as
#     # q_sqrt[P, M, M]. Here P=1 and M=4, hence (1, 4, 4).
#     assert model.q_sqrt.shape == (1, 4, 4)

#     # Fixed Z preserves the repeated space-time inducing grid for the later
#     # controlled comparison with ST-SVGP.
#     assert model.inducing_variable.Z.trainable is False

def test_model_matches_required_svgp_architecture() -> None:
    """The model must match the selected SVGP architecture."""
    z = np.zeros(
        (4, 5),
        dtype=float,
    )

    model = build_model(
        inducing_points=z,
        n_covariates=2,
        num_data=10,
        config=_config(),
    )

    assert isinstance(
        model,
        gpflow.models.SVGP,
    )

    assert isinstance(
        model.likelihood,
        gpflow.likelihoods.Bernoulli,
    )

    spatial, temporal = (
        model.kernel.kernels
    )

    assert isinstance(
        spatial,
        gpflow.kernels.Matern32,
    )

    assert isinstance(
        temporal,
        gpflow.kernels.Matern32,
    )

    # Full single-output variational covariance:
    # q_sqrt[P, M, M] with P=1 and M=4.
    assert model.q_sqrt.shape == (1, 4, 4)

    # Adam must be able to optimise q(u).
    assert model.q_mu.trainable is True
    assert model.q_sqrt.trainable is True

    # Spatial inducing support remains fixed for the future
    # controlled SVGP/ST-SVGP comparison.
    assert (
        model.inducing_variable.Z.trainable
        is False
    )



# def test_natural_gradient_step_remains_finite() -> None:
#     """One Bernoulli natural-gradient update must keep q(u) valid."""
#     z = np.array(
#         [
#             [0.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 1.0, 1.0],
#             [0.0, 0.0, 1.0, 1.0, 1.0],
#         ],
#         dtype=float,
#     )
#     x = np.array(
#         [
#             [0.2, -0.1, 0.1, 0.1, 0.0],
#             [-0.3, 0.4, 0.8, 0.2, 0.0],
#             [0.1, 0.2, 0.2, 0.8, 1.0],
#             [-0.2, -0.2, 0.9, 0.9, 1.0],
#         ],
#         dtype=float,
#     )
#     y = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=float)

#     model = build_model(
#         inducing_points=z,
#         n_covariates=2,
#         num_data=len(x),
#         config=_config(),
#     )

#     loss = model.training_loss_closure((x, y))
#     natural_gradient = gpflow.optimizers.NaturalGradient(gamma=0.01)
#     natural_gradient.minimize(
#         loss,
#         var_list=[
#             (
#                 model.q_mu,
#                 model.q_sqrt,
#                 XiSqrtMeanVar(),
#             )
#         ],
#     )

#     assert np.isfinite(model.q_mu.numpy()).all()
#     assert np.isfinite(model.q_sqrt.numpy()).all()

def test_adam_step_keeps_variational_state_finite() -> None:
    """One Adam step must keep the variational posterior finite."""
    z = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    x = np.array(
        [
            [0.2, -0.1, 0.1, 0.1, 0.0],
            [-0.3, 0.4, 0.8, 0.2, 0.0],
            [0.1, 0.2, 0.2, 0.8, 1.0],
            [-0.2, -0.2, 0.9, 0.9, 1.0],
        ],
        dtype=float,
    )

    y = np.array(
        [
            [0.0],
            [1.0],
            [0.0],
            [1.0],
        ],
        dtype=float,
    )

    model = build_model(
        inducing_points=z,
        n_covariates=2,
        num_data=len(x),
        config=_config(),
    )

    loss = model.training_loss_closure(
        (
            x,
            y,
        )
    )

    optimizer = tf_keras.optimizers.Adam(
        learning_rate=0.001
    )

    optimizer.minimize(
        loss,
        var_list=model.trainable_variables,
    )

    assert np.isfinite(
        model.q_mu.numpy()
    ).all()

    assert np.isfinite(
        model.q_sqrt.numpy()
    ).all()

def test_temporal_contract_keeps_final_test_locked() -> None:
    """The implementation must never train on or inspect the 2020 test origin."""
    config = _config()
    validate_temporal_contract(config)

    config["final_test"]["evaluate"] = True
    with pytest.raises(ValueError):
        validate_temporal_contract(config)
