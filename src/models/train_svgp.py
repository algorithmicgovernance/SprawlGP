"""Train the Sparse Variational Gaussian Process classifier.

Scientific model
----------------
For an eligible cell i at forecast origin t,

    f(i, t) = beta_0 + x(i, t)^T beta + g(s_i, t)

with

    g(s, t) ~ GP(0, k_s(s, s') k_t(t, t'))
    Y(i, t) | f(i, t) ~ Bernoulli(Phi(f(i, t))).

The implementation follows the scalable GP-classification construction of
Hensman, Matthews & Ghahramani (2015), especially Section 4, Eqs. 17--21:
q(u)=N(m,S), q(f,u)=p(f|u)q(u), and the ELBO is an expected Bernoulli
log-likelihood minus KL[q(u)||p(u)]. That KL term is the variational
"complexity penalty": it prevents the approximate posterior at the inducing
variables from moving away from the GP prior unless the data support it.

Hamelijnck et al. (2021), Section 2.2 Eq. 5, reviews the same SVGP objective
and its O(N M^2 + M^3) structure before deriving ST-SVGP. To keep the later
comparison clean, this code already uses a separable Matérn-3/2 spatial ×
temporal kernel and repeats one set of spatial inducing locations over the
training times (their Figure 1).

This module does not evaluate the locked 2020→2025 final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from gpflow.keras import tf_keras
# from gpflow.optimizers.natgrad import XiSqrtMeanVar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .metrics import probabilistic_metrics, validate_probabilities


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


@dataclass
class Preprocessor:
    """Training-only transforms needed by the semiparametric GP.

    `scaler` acts only on the observed linear predictors x in x^T beta.
    Space is expressed in kilometres and centred for numerical stability.
    Time is represented in five-year steps. Space/time are *not* linear
    predictors; they define the GP covariance inputs.
    """

    scaler: StandardScaler
    spatial_center_km: np.ndarray
    linear_predictors: tuple[str, ...]
    x_coordinate: str
    y_coordinate: str
    forecast_origin: str
    time_origin_year: int
    time_step_years: int


class CovariateLinearMean(gpflow.functions.MeanFunction):
    """Learn beta_0 + x^T beta while the GP models residual space-time structure.

    This separation is deliberate: the remote-sensing, demographic and terrain
    variables retain an interpretable linear contribution, while g(s,t) captures
    residual dependence through the kernel. It also lets SVGP and the future
    ST-SVGP share the same scientific model and differ mainly in inference.
    """

    def __init__(self, n_covariates: int) -> None:
        super().__init__()
        dtype = gpflow.config.default_float()
        self.n_covariates = int(n_covariates)
        self.beta = gpflow.Parameter(
            np.zeros((n_covariates, 1), dtype=dtype)
        )
        self.intercept = gpflow.Parameter(
            np.zeros((1,), dtype=dtype)
        )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return (
            tf.linalg.matmul(x[:, : self.n_covariates], self.beta)
            + self.intercept
        )


def load_config(path: Path) -> dict[str, Any]:
    """Load one SVGP YAML configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The SVGP configuration must be a mapping.")
    return config


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_runtime(config: dict[str, Any]) -> dict[str, object]:
    """Configure GPflow precision and TensorFlow device visibility.

    The reference configuration uses float64 on CPU for numerical stability.
    TensorFlow does not call Apple's backend "MPS"; when the optional
    tensorflow-metal plugin is installed it exposes the Mac GPU as a TensorFlow
    GPU device.

    Returns:
        Runtime metadata written to the model card.
    """
    compute = config["compute"]
    float_name = str(compute["float_type"]).lower()

    if float_name == "float64":
        gpflow.config.set_default_float(np.float64)
    elif float_name == "float32":
        gpflow.config.set_default_float(np.float32)
    else:
        raise ValueError("compute.float_type must be float32 or float64.")

    gpflow.config.set_default_jitter(float(compute["jitter"]))

    requested_device = str(compute["device"]).lower()
    physical_gpus = tf.config.list_physical_devices("GPU")

    if requested_device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        selected_device = "CPU"
    elif requested_device == "gpu":
        if not physical_gpus:
            raise RuntimeError(
                "compute.device=gpu was requested but TensorFlow found no GPU."
            )
        selected_device = "GPU"
    elif requested_device == "auto":
        selected_device = "GPU" if physical_gpus else "CPU"
    else:
        raise ValueError("compute.device must be cpu, gpu or auto.")

    return {
        "requested_device": requested_device,
        "selected_device": selected_device,
        "physical_gpus_detected": len(physical_gpus),
        "float_type": float_name,
        "jitter": float(compute["jitter"]),
    }


def validate_temporal_contract(config: dict[str, Any]) -> None:
    """Protect chronology and keep the final 2020 origin untouched."""
    for fold in config["rolling_validation"]["folds"]:
        train_origins = [int(v) for v in fold["train_origins"]]
        validation_origin = int(fold["validation_origin"])

        if not train_origins or max(train_origins) >= validation_origin:
            raise ValueError(
                "Every rolling-validation origin must follow its training origins."
            )

    final_fit = [int(v) for v in config["final_fit"]["origins"]]
    final_test = [int(v) for v in config["final_test"]["origins"]]

    if max(final_fit) >= min(final_test):
        raise ValueError("Final-fit origins must precede the final test.")

    if bool(config["final_test"].get("evaluate", False)):
        raise ValueError("The final test must remain locked.")


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load the frozen cell-time table and validate only required invariants."""
    dataset = config["dataset"]
    path = Path(dataset["path"])

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_parquet(path)

    raw_linear = [
        feature
        for feature in config["linear_predictors"]
        if feature != "log_population_density_t"
    ]
    population_source = config["derived_features"][
        "log_population_density_t"
    ]["source"]

    required = {
        dataset["target"],
        dataset["forecast_origin"],
        dataset["target_year"],
        dataset["cell_id"],
        dataset["x_coordinate"],
        dataset["y_coordinate"],
        population_source,
        *raw_linear,
    }

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing columns required by SVGP: " + ", ".join(missing)
        )

    if frame.duplicated(
        [dataset["cell_id"], dataset["forecast_origin"]]
    ).any():
        raise ValueError("Duplicate cell-time rows were detected.")

    target = pd.to_numeric(frame[dataset["target"]], errors="raise")
    if set(target.astype(int).unique()) != {0, 1}:
        raise ValueError("SVGP requires a binary target containing 0 and 1.")

    origin = pd.to_numeric(
        frame[dataset["forecast_origin"]], errors="raise"
    ).astype(int)
    target_year = pd.to_numeric(
        frame[dataset["target_year"]], errors="raise"
    ).astype(int)

    if not target_year.eq(origin + 5).all():
        raise ValueError("target_year must equal forecast_origin + 5.")

    expected_origins = {
        *[
            int(v)
            for fold in config["rolling_validation"]["folds"]
            for v in fold["train_origins"]
        ],
        *[
            int(fold["validation_origin"])
            for fold in config["rolling_validation"]["folds"]
        ],
        *[int(v) for v in config["final_fit"]["origins"]],
        *[int(v) for v in config["final_test"]["origins"]],
    }
    missing_origins = sorted(expected_origins.difference(set(origin.unique())))
    if missing_origins:
        raise ValueError(
            f"Configured forecast origins are absent: {missing_origins}"
        )

    return frame


def add_log_population(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create log1p population density, matching the retained LR specification."""
    source = config["derived_features"][
        "log_population_density_t"
    ]["source"]
    population = pd.to_numeric(frame[source], errors="raise").astype(float)

    if population.lt(0).any() or not np.isfinite(population).all():
        raise ValueError("Population density must be finite and non-negative.")

    result = frame.copy()
    result["log_population_density_t"] = np.log1p(population)
    return result


def select_rows(
    frame: pd.DataFrame,
    origin_column: str,
    origins: list[int],
) -> pd.DataFrame:
    """Return one chronological modelling subset."""
    return frame.loc[
        frame[origin_column].astype(int).isin(origins)
    ].copy()


def fit_preprocessor(
    train: pd.DataFrame,
    config: dict[str, Any],
) -> Preprocessor:
    """Fit transforms on training rows only to avoid temporal leakage."""
    dataset = config["dataset"]
    predictors = tuple(str(v) for v in config["linear_predictors"])

    linear_values = train.loc[:, predictors].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(linear_values).all():
        raise ValueError("Linear predictors contain non-finite values.")

    scaler = StandardScaler().fit(linear_values)

    spatial_km = (
        train.loc[
            :,
            [dataset["x_coordinate"], dataset["y_coordinate"]],
        ]
        .to_numpy(dtype=float)
        / 1000.0
    )
    spatial_center_km = spatial_km.mean(axis=0)

    return Preprocessor(
        scaler=scaler,
        spatial_center_km=spatial_center_km,
        linear_predictors=predictors,
        x_coordinate=str(dataset["x_coordinate"]),
        y_coordinate=str(dataset["y_coordinate"]),
        forecast_origin=str(dataset["forecast_origin"]),
        time_origin_year=int(config["time"]["origin_year"]),
        time_step_years=int(config["time"]["step_years"]),
    )


def transform_inputs(
    frame: pd.DataFrame,
    preprocessor: Preprocessor,
) -> np.ndarray:
    """Build [standardised covariates | x_km | y_km | time_step].

    Hensman et al. allow generic inducing inputs X. Here we deliberately reserve
    the final three dimensions for the separable space-time covariance so that
    the later ST-SVGP can use the same scientific inputs.
    """
    covariates = preprocessor.scaler.transform(
        frame.loc[:, preprocessor.linear_predictors].to_numpy(dtype=float)
    )

    spatial = (
        frame.loc[
            :,
            [preprocessor.x_coordinate, preprocessor.y_coordinate],
        ].to_numpy(dtype=float)
        / 1000.0
        - preprocessor.spatial_center_km
    )

    time = (
        (
            frame[preprocessor.forecast_origin].to_numpy(dtype=float)
            - preprocessor.time_origin_year
        )
        / preprocessor.time_step_years
    )[:, None]

    dtype = gpflow.config.default_float()
    result = np.hstack([covariates, spatial, time]).astype(dtype)

    if not np.isfinite(result).all():
        raise ValueError("SVGP input matrix contains non-finite values.")

    return result


# ---------------------------------------------------------------------------
# GP construction
# ---------------------------------------------------------------------------


def build_inducing_grid(
    train_inputs: np.ndarray,
    n_covariates: int,
    config: dict[str, Any],
) -> np.ndarray:
    """Create spatial inducing points repeated at every training time.

    Hensman et al. (2015) use inducing variables u=f(Z) to replace dependence
    on all N latent values by M << N support variables.

    Hamelijnck et al. (2021), Figure 1, explains the bridge to ST-SVGP:
    standard SVGP can place the same spatial inducing locations at each time,
    whereas ST-SVGP tracks those spatial locations through a state-space model.
    We construct exactly that repeated space-time grid now.

    The inducing locations are fixed after k-means in this first implementation
    so that the later SVGP/ST-SVGP comparison can share the same spatial support.
    """
    spatial = train_inputs[:, n_covariates : n_covariates + 2]
    times = np.unique(train_inputs[:, n_covariates + 2])

    requested = int(config["inducing"]["spatial_points"])
    unique_spatial = np.unique(spatial, axis=0)

    if len(unique_spatial) < requested:
        raise ValueError(
            f"Only {len(unique_spatial)} unique spatial points are available "
            f"for {requested} requested inducing locations."
        )

    kmeans = KMeans(
        n_clusters=requested,
        random_state=int(config["training"]["random_state"]),
        n_init=int(config["inducing"]["kmeans_n_init"]),
    ).fit(unique_spatial)

    spatial_centres = kmeans.cluster_centers_
    total_dimension = train_inputs.shape[1]
    inducing_blocks = []

    for time_value in np.sort(times):
        block = np.zeros(
            (requested, total_dimension),
            dtype=gpflow.config.default_float(),
        )
        block[:, n_covariates : n_covariates + 2] = spatial_centres
        block[:, n_covariates + 2] = time_value
        inducing_blocks.append(block)

    return np.vstack(inducing_blocks)


def build_model(
    inducing_points: np.ndarray,
    n_covariates: int,
    num_data: int,
    config: dict[str, Any],
) -> gpflow.models.SVGP:
    """Construct the Bernoulli-probit SVGP used before ST-SVGP.

    Literature links
    ----------------
    Hensman et al. (2015), Section 4, Eq. 21:
        ELBO = sum_n E_q(f_n)[log p(y_n|f_n)] - KL[q(u)||p(u)].

    Hamelijnck et al. (2021), Section 2.2, Eq. 5:
        the same SVGP objective costs O(N M^2 + M^3) and supports
        non-Gaussian likelihoods through variational expectations.

    The product kernel k_s * k_t is chosen now because the later ST-SVGP
    state-space construction requires an explicit separable space-time kernel.
    """
    if config["kernel"]["family"] != "matern32_separable":
        raise ValueError("Only matern32_separable is implemented.")

    spatial_dims = [n_covariates, n_covariates + 1]
    temporal_dim = [n_covariates + 2]

    spatial_kernel = gpflow.kernels.Matern32(
        variance=float(config["kernel"]["variance"]),
        lengthscales=[
            float(config["kernel"]["spatial_initial_lengthscale_km"]),
            float(config["kernel"]["spatial_initial_lengthscale_km"]),
        ],
        active_dims=spatial_dims,
    )
    temporal_kernel = gpflow.kernels.Matern32(
        variance=1.0,
        lengthscales=float(
            config["kernel"]["temporal_initial_lengthscale_steps"]
        ),
        active_dims=temporal_dim,
    )

    # The product already has one overall amplitude. Fixing the temporal
    # variance avoids a redundant variance × variance parameterisation.
    gpflow.set_trainable(temporal_kernel.variance, False)

    likelihood = gpflow.likelihoods.Bernoulli()
    mean_function = CovariateLinearMean(n_covariates)

    model = gpflow.models.SVGP(
        kernel=spatial_kernel * temporal_kernel,
        likelihood=likelihood,
        inducing_variable=inducing_points,
        mean_function=mean_function,
        num_data=int(num_data),
        q_diag=False,
        whiten=True,
    )

    # GPflow's NaturalGradient works on the full Gaussian q(u), represented by
    # q_mu and q_sqrt. Keep them out of Adam so the two optimisers have distinct
    # responsibilities, as in the natural-gradient variational literature.
    # gpflow.set_trainable(model.q_mu, False)
    # gpflow.set_trainable(model.q_sqrt, False)

    # if not bool(config["inducing"]["train_locations"]):
    #     gpflow.set_trainable(model.inducing_variable.Z, False)
    
    # q_mu and q_sqrt define the variational posterior q(u)=N(m,S).
    # They remain trainable because the reference SVGP optimizes the complete
    # Hensman ELBO with Adam.
    #
    # q_sqrt parameterizes S through a matrix factor, so we retain the full
    # variational covariance while avoiding a separate Natural Gradient update.
    if not bool(config["inducing"]["train_locations"]):
        gpflow.set_trainable(
            model.inducing_variable.Z,
            False,
        )

    return model


# ---------------------------------------------------------------------------
# Optimisation and prediction
# ---------------------------------------------------------------------------


# def train_model(
#     model: gpflow.models.SVGP,
#     x_train: np.ndarray,
#     y_train: np.ndarray,
#     iterations: int,
#     config: dict[str, Any],
#     run_name: str,
# ) -> pd.DataFrame:
#     """Optimise the stochastic SVGP ELBO with NatGrad + Adam.

#     The expected log-likelihood term in the Hensman ELBO factorises over data
#     points, which is what makes unbiased mini-batch optimisation possible.
#     The KL[q(u)||p(u)] complexity penalty remains global and is handled by the
#     SVGP objective through `num_data`.

#     NaturalGradient updates q(u)=N(m,S); Adam updates kernel hyperparameters and
#     beta in the parametric mean. Hamelijnck et al. use the same conceptual
#     separation before replacing generic temporal GP algebra by filtering and
#     smoothing in ST-SVGP.
#     """
#     training = config["training"]
#     seed = int(training["random_state"])

#     tf.random.set_seed(seed)
#     np.random.seed(seed)

#     dataset = (
#         tf.data.Dataset.from_tensor_slices((x_train, y_train))
#         .shuffle(
#             buffer_size=min(
#                 len(x_train),
#                 int(training["shuffle_buffer"]),
#             ),
#             seed=seed,
#             reshuffle_each_iteration=True,
#         )
#         .repeat()
#         .batch(int(training["batch_size"]))
#         .prefetch(tf.data.AUTOTUNE)
#     )
#     iterator = iter(dataset)
#     loss = model.training_loss_closure(iterator, compile=True)

#     adam = tf_keras.optimizers.Adam(
#         learning_rate=float(training["adam_learning_rate"])
#     )
#     # Hensman-style natural gradients update the Gaussian q(u).
#     #
#     # GPflow's default natural-parameter transform can become numerically
#     # aggressive for a non-conjugate Bernoulli likelihood with stochastic
#     # mini-batches. GPflow's natural-gradient documentation explicitly shows
#     # XiSqrtMeanVar for Bernoulli models and notes that the parameterisation can
#     # matter for finite step sizes. We therefore update through the
#     # (mean, Cholesky-factor) representation and use a conservative gamma.
#     natural_gradient = gpflow.optimizers.NaturalGradient(
#         gamma=float(training["natural_gradient_gamma"])
#     )
#     variational_parameters = [
#         (
#             model.q_mu,
#             model.q_sqrt,
#             XiSqrtMeanVar(),
#         )
#     ]

#     records = []
#     log_every = int(training["log_every"])

#     for step in range(1, int(iterations) + 1):
#         # GPflow documents this interleaving for SVGP: standard gradients for
#         # hyperparameters, natural gradients for the variational Gaussian.
#         adam.minimize(loss, var_list=model.trainable_variables)
#         natural_gradient.minimize(
#             loss,
#             var_list=variational_parameters,
#         )

#         # q_sqrt is the Cholesky factor of the variational covariance.
#         # Non-finite values here mean q(u)=N(m,S) is no longer a valid
#         # variational Gaussian, so continuing would invalidate the ELBO.
#         tf.debugging.assert_all_finite(
#             model.q_mu,
#             "Natural-gradient update produced non-finite q_mu.",
#         )
#         tf.debugging.assert_all_finite(
#             model.q_sqrt,
#             "Natural-gradient update produced non-finite q_sqrt.",
#         )

#         if step == 1 or step % log_every == 0 or step == iterations:
#             batch = next(iterator)
#             elbo = float(model.elbo(batch).numpy())
#             records.append(
#                 {
#                     "run": run_name,
#                     "iteration": step,
#                     "minibatch_elbo": elbo,
#                 }
#             )
#             print(
#                 f"[{run_name}] iteration={step} "
#                 f"minibatch_elbo={elbo:.6f}"
#             )

#     return pd.DataFrame(records)


def train_model(
    model: gpflow.models.SVGP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    iterations: int,
    config: dict[str, Any],
    run_name: str,
) -> pd.DataFrame:
    """Optimise the stochastic SVGP ELBO with Adam.

    Hensman et al. (2015), Section 4, Eqs. 17--21 define the
    classification SVGP objective as:

        expected log likelihood - KL[q(u) || p(u)]

    The expected likelihood factorises across observations, which enables
    stochastic mini-batch optimisation.

    Adam jointly updates:
      - q_mu and q_sqrt, defining q(u)=N(m,S);
      - kernel hyperparameters;
      - coefficients of the parametric mean.

    Natural gradients are not required to define SVGP and are intentionally
    reserved for ST-SVGP, where Hamelijnck et al. use natural-gradient/CVI
    updates together with temporal filtering and smoothing.

    Args:
        model: GPflow SVGP model.
        x_train: Training model inputs.
        y_train: Binary target column.
        iterations: Number of stochastic optimisation steps.
        config: Experiment configuration.
        run_name: Fold or final-fit identifier.

    Returns:
        Training-history table containing ELBO diagnostics.
    """
    training = config["training"]
    seed = int(training["random_state"])

    tf.random.set_seed(seed)
    np.random.seed(seed)

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                x_train,
                y_train,
            )
        )
        .shuffle(
            buffer_size=min(
                len(x_train),
                int(training["shuffle_buffer"]),
            ),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .repeat()
        .batch(
            int(training["batch_size"])
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    iterator = iter(dataset)

    loss = model.training_loss_closure(
        iterator,
        compile=True,
    )

    # GPflow's large-data SVGP workflow uses Adam directly on the
    # stochastic ELBO. All SVGP parameters except the fixed inducing
    # locations are optimized jointly.
    optimizer = tf_keras.optimizers.Adam(
        learning_rate=float(
            training["adam_learning_rate"]
        )
    )

    records = []
    log_every = int(
        training["log_every"]
    )

    for step in range(
        1,
        int(iterations) + 1,
    ):
        optimizer.minimize(
            loss,
            var_list=model.trainable_variables,
        )

        # q(u)=N(m,S) must remain numerically finite throughout training.
        tf.debugging.assert_all_finite(
            model.q_mu,
            "Adam produced non-finite q_mu.",
        )

        tf.debugging.assert_all_finite(
            model.q_sqrt,
            "Adam produced non-finite q_sqrt.",
        )

        if (
            step == 1
            or step % log_every == 0
            or step == iterations
        ):
            batch = next(iterator)

            elbo = float(
                model.elbo(batch).numpy()
            )

            state = model_state(model)

            records.append(
                {
                    "run": run_name,
                    "iteration": step,
                    "minibatch_elbo": elbo,
                    "spatial_lengthscale_x_km": (
                        state[
                            "spatial_lengthscales_km"
                        ][0]
                    ),
                    "spatial_lengthscale_y_km": (
                        state[
                            "spatial_lengthscales_km"
                        ][1]
                    ),
                    "temporal_lengthscale_steps": (
                        state[
                            "temporal_lengthscale_steps"
                        ]
                    ),
                    "kernel_variance": (
                        state[
                            "kernel_variance"
                        ]
                    ),
                }
            )

            print(
                f"[{run_name}] "
                f"iteration={step} "
                f"minibatch_elbo={elbo:.6f}"
            )

    return pd.DataFrame(records)

def predict_in_batches(
    model: gpflow.models.SVGP,
    inputs: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predictive probability and latent mean/variance.

    For Bernoulli classification GPflow's default inverse link is probit,
    matching Hensman et al. (2015). `predict_y` integrates the latent Gaussian
    uncertainty through the likelihood; the returned mean is P(Y=1|data).
    """
    probabilities = []
    latent_means = []
    latent_variances = []

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        latent_mean, latent_variance = model.predict_f(batch)
        probability, _ = model.predict_y(batch)

        probabilities.append(probability.numpy()[:, 0])
        latent_means.append(latent_mean.numpy()[:, 0])
        latent_variances.append(latent_variance.numpy()[:, 0])

    probability = np.concatenate(probabilities)
    latent_mean = np.concatenate(latent_means)
    latent_variance = np.concatenate(latent_variances)

    validate_probabilities(probability)
    return probability, latent_mean, latent_variance


def model_state(
    model: gpflow.models.SVGP,
) -> dict[str, object]:
    """Extract the small set of scientifically interpretable fitted parameters."""
    spatial_kernel, temporal_kernel = model.kernel.kernels
    mean_function = model.mean_function

    return {
        "spatial_lengthscales_km": np.asarray(
            spatial_kernel.lengthscales.numpy()
        ).tolist(),
        "temporal_lengthscale_steps": float(
            temporal_kernel.lengthscales.numpy()
        ),
        "kernel_variance": float(spatial_kernel.variance.numpy()),
        "linear_intercept": float(mean_function.intercept.numpy()[0]),
        "linear_coefficients_standardised": np.asarray(
            mean_function.beta.numpy()
        )[:, 0].tolist(),
        "inducing_count": int(model.inducing_variable.Z.shape[0]),
    }


def save_preprocessor(
    preprocessor: Preprocessor,
    path: Path,
) -> None:
    """Persist transforms required to reconstruct final-model inputs."""
    payload = {
        "linear_predictors": list(preprocessor.linear_predictors),
        "linear_mean": preprocessor.scaler.mean_.tolist(),
        "linear_scale": preprocessor.scaler.scale_.tolist(),
        "spatial_center_km": preprocessor.spatial_center_km.tolist(),
        "x_coordinate": preprocessor.x_coordinate,
        "y_coordinate": preprocessor.y_coordinate,
        "forecast_origin": preprocessor.forecast_origin,
        "time_origin_year": preprocessor.time_origin_year,
        "time_step_years": preprocessor.time_step_years,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def evaluate_fold(
    frame: pd.DataFrame,
    config: dict[str, Any],
    fold_index: int,
    fold: dict[str, Any],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Fit one historical fold and return metrics, predictions and ELBO history."""
    dataset = config["dataset"]
    origin_column = str(dataset["forecast_origin"])
    target_column = str(dataset["target"])

    train_origins = [int(v) for v in fold["train_origins"]]
    validation_origin = int(fold["validation_origin"])

    train = select_rows(frame, origin_column, train_origins)
    validation = select_rows(frame, origin_column, [validation_origin])

    preprocessor = fit_preprocessor(train, config)
    x_train = transform_inputs(train, preprocessor)
    x_validation = transform_inputs(validation, preprocessor)

    y_train = train[target_column].to_numpy(
        dtype=gpflow.config.default_float()
    )[:, None]
    y_validation = validation[target_column].to_numpy(dtype=int)

    n_covariates = len(preprocessor.linear_predictors)
    inducing = build_inducing_grid(
        x_train,
        n_covariates,
        config,
    )
    model = build_model(
        inducing,
        n_covariates,
        len(train),
        config,
    )

    history = train_model(
        model,
        x_train,
        y_train,
        int(config["training"]["iterations"]),
        config,
        run_name=f"fold_{fold_index}",
    )

    probability, latent_mean, latent_variance = predict_in_batches(
        model,
        x_validation,
        int(config["training"]["prediction_batch_size"]),
    )
    metrics = probabilistic_metrics(y_validation, probability)
    state = model_state(model)

    observed_prevalence = float(y_validation.mean())
    record = {
        "fold": fold_index,
        "train_origins": ",".join(map(str, train_origins)),
        "validation_origin": validation_origin,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "observed_prevalence": observed_prevalence,
        "mean_predicted_probability": float(probability.mean()),
        "probability_bias": float(
            probability.mean() - observed_prevalence
        ),
        **metrics,
        "spatial_lengthscale_x_km": float(
            state["spatial_lengthscales_km"][0]
        ),
        "spatial_lengthscale_y_km": float(
            state["spatial_lengthscales_km"][1]
        ),
        "temporal_lengthscale_steps": float(
            state["temporal_lengthscale_steps"]
        ),
        "kernel_variance": float(state["kernel_variance"]),
        "inducing_count": int(state["inducing_count"]),
    }

    predictions = validation[
        [
            dataset["cell_id"],
            origin_column,
            dataset["target_year"],
            target_column,
        ]
    ].copy()
    predictions["probability_raw"] = probability
    predictions["latent_mean"] = latent_mean
    predictions["latent_variance"] = latent_variance
    predictions["fold"] = fold_index

    return record, predictions, history


def summarise_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average historical metrics with equal weight per temporal fold."""
    return pd.DataFrame(
        [
            {
                "folds": int(metrics["fold"].nunique()),
                "mean_log_loss": float(metrics["log_loss"].mean()),
                "std_log_loss": float(metrics["log_loss"].std()),
                "mean_brier_score": float(metrics["brier_score"].mean()),
                "std_brier_score": float(metrics["brier_score"].std()),
                "mean_pr_auc": float(metrics["pr_auc"].mean()),
                "std_pr_auc": float(metrics["pr_auc"].std()),
                "mean_roc_auc": float(metrics["roc_auc"].mean()),
                "std_roc_auc": float(metrics["roc_auc"].std()),
                "mean_probability_bias": float(
                    metrics["probability_bias"].mean()
                ),
            }
        ]
    )


def fit_final(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[gpflow.models.SVGP, Preprocessor, pd.DataFrame]:
    """Fit the fixed SVGP on all four pre-test forecast origins."""
    dataset = config["dataset"]
    origins = [int(v) for v in config["final_fit"]["origins"]]
    train = select_rows(
        frame,
        str(dataset["forecast_origin"]),
        origins,
    )

    preprocessor = fit_preprocessor(train, config)
    x_train = transform_inputs(train, preprocessor)
    y_train = train[dataset["target"]].to_numpy(
        dtype=gpflow.config.default_float()
    )[:, None]

    n_covariates = len(preprocessor.linear_predictors)
    inducing = build_inducing_grid(
        x_train,
        n_covariates,
        config,
    )
    model = build_model(
        inducing,
        n_covariates,
        len(train),
        config,
    )

    history = train_model(
        model,
        x_train,
        y_train,
        int(config["training"]["final_iterations"]),
        config,
        run_name="final_fit",
    )

    return model, preprocessor, history


def write_preflight(
    frame: pd.DataFrame,
    config: dict[str, Any],
    config_path: Path,
    runtime: dict[str, object],
) -> dict[str, object]:
    """Write the minimal scientific/runtime contract before training."""
    output = config["outputs"]
    metadata_directory = Path(output["metadata_directory"])
    metadata_directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "PASS",
        "model": "svgp",
        "dataset_path": config["dataset"]["path"],
        "dataset_sha256": sha256(Path(config["dataset"]["path"])),
        "dataset_rows": int(len(frame)),
        "linear_predictors": list(config["linear_predictors"]),
        "gp_inputs": [
            config["dataset"]["x_coordinate"],
            config["dataset"]["y_coordinate"],
            config["dataset"]["forecast_origin"],
        ],
        "kernel": config["kernel"]["family"],
        "spatial_inducing_points": int(
            config["inducing"]["spatial_points"]
        ),
        "rolling_validation_folds": config[
            "rolling_validation"
        ]["folds"],
        "final_fit_origins": config["final_fit"]["origins"],
        "final_test_locked_origins": config["final_test"]["origins"],
        "final_test_evaluated": False,
        "runtime": runtime,
        "config_sha256": sha256(config_path),
    }

    (
        metadata_directory / "preflight.json"
    ).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def run(
    config_path: Path,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Run historical SVGP evaluation and the final four-origin fit."""
    config = load_config(config_path)
    runtime = configure_runtime(config)
    validate_temporal_contract(config)

    frame = add_log_population(
        load_dataset(config),
        config,
    )

    preflight = write_preflight(
        frame,
        config,
        config_path,
        runtime,
    )
    if preflight_only:
        return preflight

    outputs = config["outputs"]
    model_directory = Path(outputs["model_directory"])
    metrics_directory = Path(outputs["metrics_directory"])
    predictions_directory = Path(outputs["predictions_directory"])

    for directory in [
        model_directory,
        metrics_directory,
        predictions_directory,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    metric_records = []
    prediction_frames = []
    history_frames = []

    for fold_index, fold in enumerate(
        config["rolling_validation"]["folds"],
        start=1,
    ):
        record, predictions, history = evaluate_fold(
            frame,
            config,
            fold_index,
            fold,
        )
        metric_records.append(record)
        prediction_frames.append(predictions)
        history_frames.append(history)

    fold_metrics = pd.DataFrame(metric_records)
    fold_metrics_path = (
        metrics_directory / "svgp_rolling_validation_metrics.csv"
    )
    fold_metrics.to_csv(fold_metrics_path, index=False)

    summary = summarise_folds(fold_metrics)
    summary_path = (
        metrics_directory / "svgp_rolling_validation_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    oof_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    oof_path = predictions_directory / "svgp_oof_predictions.parquet"
    oof_predictions.to_parquet(oof_path, index=False)

    final_model, final_preprocessor, final_history = fit_final(
        frame,
        config,
    )
    history_frames.append(final_history)

    training_history_path = metrics_directory / "svgp_training_history.csv"
    pd.concat(
        history_frames,
        ignore_index=True,
    ).to_csv(training_history_path, index=False)

    preprocessor_path = model_directory / "preprocessing.json"
    save_preprocessor(final_preprocessor, preprocessor_path)

    # TensorFlow checkpointing is intentionally used instead of pickling the
    # GPflow object. Reconstructing the model from code/config and restoring
    # parameters is the stable route for later forecasting and ST-SVGP work.
    checkpoint_prefix = str(model_directory / "checkpoint")
    checkpoint = tf.train.Checkpoint(model=final_model)
    checkpoint.write(checkpoint_prefix)

    state = model_state(final_model)
    mean_coefficients_path = (
        metrics_directory / "svgp_linear_mean_coefficients.csv"
    )
    pd.DataFrame(
        {
            "feature": config["linear_predictors"],
            "coefficient_standardised": state[
                "linear_coefficients_standardised"
            ],
        }
    ).to_csv(mean_coefficients_path, index=False)

    model_card = {
        "status": "PASS",
        "model": "svgp",
        "likelihood": "bernoulli_probit",
        "kernel": "matern32_space_x_matern32_time",
        "linear_mean": "beta_0 + x^T beta",
        "linear_predictors": list(config["linear_predictors"]),
        "C_or_class_weight": "not_applicable",
        "spatial_inducing_points": int(
            config["inducing"]["spatial_points"]
        ),
        "inducing_locations_trainable": bool(
            config["inducing"]["train_locations"]
        ),
        "historical_validation": summary.iloc[0].to_dict(),
        "final_fit_origins": [
            int(v) for v in config["final_fit"]["origins"]
        ],
        "final_test_locked_origins": [
            int(v) for v in config["final_test"]["origins"]
        ],
        "final_test_evaluated": False,
        "runtime": runtime,
        "fitted_state": state,
        "checkpoint_prefix": checkpoint_prefix,
        "preprocessing": str(preprocessor_path),
        "fold_metrics": str(fold_metrics_path),
        "rolling_summary": str(summary_path),
        "oof_predictions": str(oof_path),
        "training_history": str(training_history_path),
        "linear_mean_coefficients": str(mean_coefficients_path),
    }

    (
        model_directory / "model_card.json"
    ).write_text(
        json.dumps(model_card, indent=2) + "\n",
        encoding="utf-8",
    )

    return model_card


def main() -> None:
    """Run the SVGP preflight or complete experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    result = run(
        config_path=args.config,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
