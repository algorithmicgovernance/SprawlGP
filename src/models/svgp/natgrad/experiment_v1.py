"""Historical SVGP Natural-Gradient experiment V1.

This file freezes the natgrad_schedule_0025 implementation that was kept as an informative alternative to the selected Adam SVGP.

Train and evaluate the Sparse Variational Gaussian Process classifier.

Scientific model
----------------
For eligible cell i at forecast origin t:

    f(i,t) = beta_0 + x(i,t)^T beta + g(s_i,t)

    g(s,t) ~ GP(0, k_space(s,s') * k_time(t,t'))

    Y(i,t) | f(i,t) ~ Bernoulli(Phi(f(i,t))).

Literature mapping
------------------
Hensman, Matthews & Ghahramani (2015), Section 4, Eqs. 17--21:
the scalable classifier uses q(u)=N(m,S) and optimises the ELBO

    E_q(f)[log p(y|f)] - KL[q(u)||p(u)].

The KL term is the variational complexity penalty: the approximate posterior
must remain close to the GP prior unless the data justify moving away.

Hamelijnck et al. (2021), Section 2.2, Eqs. 5--8:
the same variational objective motivates natural-gradient updates. Their CVI
view rewrites those updates as Gaussian approximate-likelihood sites. In this
SVGP implementation we use GPflow NaturalGradient directly; explicit CVI sites
are deferred to ST-SVGP, where filtering/smoothing needs that representation.

The final 2020→2025 test remains locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from gpflow.keras import tf_keras
from gpflow.optimizers.natgrad import XiNat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.models.evaluation.calibration import calibration_metrics, reliability_table
from src.models.metrics import probabilistic_metrics, validate_probabilities


BASE_FEATURES = (
    "ndbi_t",
    "savi_t",
    "distance_to_built_m_t",
    "built_fraction_11x11_t",
    "recent_local_growth_5y_t",
    "elevation_m",
    "slope_degrees",
    "log_population_density_t",
)

NONLINEAR_CORE_FEATURES = (
    "ndbi_t",
    "savi_t",
    "log_distance_to_built_m_t",
    "built_fraction_11x11_t",
    "recent_local_growth_5y_t",
    "elevation_m",
    "slope_degrees",
    "slope_squared_t",
    "log_population_density_t",
)

NONLINEAR_INTERACTION_FEATURES = (
    *NONLINEAR_CORE_FEATURES,
    "built_fraction_x_recent_growth_t",
    "built_fraction_x_log_population_t",
)


@dataclass
class Preprocessor:
    """Training-only transforms for the semiparametric GP."""
    scaler: StandardScaler
    spatial_center_km: np.ndarray
    linear_predictors: tuple[str, ...]
    x_coordinate: str
    y_coordinate: str
    forecast_origin: str
    time_origin_year: int
    time_step_years: int


class CovariateLinearMean(gpflow.functions.MeanFunction):
    """Parametric beta_0 + x^T beta component of the latent function."""

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
            tf.linalg.matmul(
                x[:, : self.n_covariates],
                self.beta,
            )
            + self.intercept
        )


def load_config(path: Path) -> dict[str, Any]:
    """Load one YAML experiment configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("SVGP config must be a YAML mapping.")
    return config


def sha256(path: Path) -> str:
    """Return SHA-256 for reproducibility metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_runtime(config: dict[str, Any]) -> dict[str, object]:
    """Configure precision and TensorFlow GPU visibility."""
    compute = config["compute"]
    float_name = str(compute["float_type"]).lower()

    if float_name == "float64":
        gpflow.config.set_default_float(np.float64)
    elif float_name == "float32":
        gpflow.config.set_default_float(np.float32)
    else:
        raise ValueError("compute.float_type must be float32 or float64.")

    gpflow.config.set_default_jitter(float(compute["jitter"]))

    requested = str(compute["device"]).lower()
    physical_gpus = tf.config.list_physical_devices("GPU")

    if requested == "cpu":
        tf.config.set_visible_devices([], "GPU")
        selected = "CPU"
    elif requested == "gpu":
        if not physical_gpus:
            raise RuntimeError("GPU requested but TensorFlow found no GPU.")
        selected = "GPU"
    elif requested == "auto":
        selected = "GPU" if physical_gpus else "CPU"
    else:
        raise ValueError("compute.device must be cpu, gpu or auto.")

    return {
        "requested_device": requested,
        "selected_device": selected,
        "physical_gpus_detected": len(physical_gpus),
        "float_type": float_name,
        "jitter": float(compute["jitter"]),
    }


def validate_temporal_contract(config: dict[str, Any]) -> None:
    """Protect chronological validation and the locked 2020 final test."""
    for fold in config["rolling_validation"]["folds"]:
        train_origins = [int(v) for v in fold["train_origins"]]
        validation_origin = int(fold["validation_origin"])
        if not train_origins or max(train_origins) >= validation_origin:
            raise ValueError(
                "Validation origin must follow all training origins."
            )

    final_fit = [int(v) for v in config["final_fit"]["origins"]]
    final_test = [int(v) for v in config["final_test"]["origins"]]

    if max(final_fit) >= min(final_test):
        raise ValueError("Final-fit origins must precede final test.")
    if bool(config["final_test"].get("evaluate", False)):
        raise ValueError("Final test must remain locked.")


def feature_names(feature_set: str) -> tuple[str, ...]:
    """Return the linear-mean predictors for one small feature experiment."""
    if feature_set == "base":
        return BASE_FEATURES
    if feature_set == "nonlinear_core":
        return NONLINEAR_CORE_FEATURES
    if feature_set == "nonlinear_interactions":
        return NONLINEAR_INTERACTION_FEATURES
    raise ValueError(
        "feature_set must be base, nonlinear_core or "
        "nonlinear_interactions."
    )


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create candidate predictors using only variables already in the table.

    These transformations introduce non-linearity without adding external
    datasets:
      - log-distance: diminishing proximity effect;
      - slope squared: non-linear terrain constraint;
      - neighbourhood × recent growth: local momentum interaction;
      - neighbourhood × log population: density-context interaction.

    No x/y/time feature is added to the parametric mean; space and time remain
    represented through the GP kernel.
    """
    result = frame.copy()

    population = pd.to_numeric(
        result["population_density_t"],
        errors="raise",
    ).astype(float)
    distance = pd.to_numeric(
        result["distance_to_built_m_t"],
        errors="raise",
    ).astype(float)
    slope = pd.to_numeric(
        result["slope_degrees"],
        errors="raise",
    ).astype(float)

    if (
        population.lt(0).any()
        or distance.lt(0).any()
        or not np.isfinite(population).all()
        or not np.isfinite(distance).all()
        or not np.isfinite(slope).all()
    ):
        raise ValueError(
            "Population, distance and slope must be finite; "
            "population/distance must be non-negative."
        )

    result["log_population_density_t"] = np.log1p(population)
    result["log_distance_to_built_m_t"] = np.log1p(distance)
    result["slope_squared_t"] = slope ** 2
    result["built_fraction_x_recent_growth_t"] = (
        result["built_fraction_11x11_t"].astype(float)
        * result["recent_local_growth_5y_t"].astype(float)
    )
    result["built_fraction_x_log_population_t"] = (
        result["built_fraction_11x11_t"].astype(float)
        * result["log_population_density_t"].astype(float)
    )

    return result


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load frozen cell-time data and validate essential modelling invariants."""
    dataset = config["dataset"]
    path = Path(dataset["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_parquet(path)

    required = {
        dataset["target"],
        dataset["forecast_origin"],
        dataset["target_year"],
        dataset["cell_id"],
        dataset["x_coordinate"],
        dataset["y_coordinate"],
        "population_density_t",
        "ndbi_t",
        "savi_t",
        "distance_to_built_m_t",
        "built_fraction_11x11_t",
        "recent_local_growth_5y_t",
        "elevation_m",
        "slope_degrees",
    }

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing columns required by SVGP: " + ", ".join(missing)
        )

    if frame.duplicated(
        [dataset["cell_id"], dataset["forecast_origin"]]
    ).any():
        raise ValueError("Duplicate cell-time rows detected.")

    target = pd.to_numeric(frame[dataset["target"]], errors="raise")
    if set(target.astype(int).unique()) != {0, 1}:
        raise ValueError("SVGP requires binary target containing 0 and 1.")

    origin = pd.to_numeric(
        frame[dataset["forecast_origin"]],
        errors="raise",
    ).astype(int)
    target_year = pd.to_numeric(
        frame[dataset["target_year"]],
        errors="raise",
    ).astype(int)

    if not target_year.eq(origin + 5).all():
        raise ValueError("target_year must equal forecast_origin + 5.")

    return engineer_features(frame)


def select_rows(
    frame: pd.DataFrame,
    origin_column: str,
    origins: list[int],
) -> pd.DataFrame:
    """Select one chronological modelling subset."""
    return frame.loc[
        frame[origin_column].astype(int).isin(origins)
    ].copy()


def fit_preprocessor(
    train: pd.DataFrame,
    config: dict[str, Any],
) -> Preprocessor:
    """Fit scaling and spatial centring on training rows only."""
    dataset = config["dataset"]
    predictors = feature_names(str(config["feature_set"]))

    values = train.loc[:, predictors].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Linear predictors contain NaN or Inf.")

    scaler = StandardScaler().fit(values)

    spatial_km = (
        train.loc[
            :,
            [dataset["x_coordinate"], dataset["y_coordinate"]],
        ].to_numpy(dtype=float)
        / 1000.0
    )

    return Preprocessor(
        scaler=scaler,
        spatial_center_km=spatial_km.mean(axis=0),
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
    """Build [standardised covariates | x_km | y_km | time_step]."""
    covariates = preprocessor.scaler.transform(
        frame.loc[
            :,
            preprocessor.linear_predictors,
        ].to_numpy(dtype=float)
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

    result = np.hstack(
        [covariates, spatial, time]
    ).astype(gpflow.config.default_float())

    if not np.isfinite(result).all():
        raise ValueError("SVGP input matrix contains NaN or Inf.")

    return result


def build_inducing_grid(
    train_inputs: np.ndarray,
    n_covariates: int,
    config: dict[str, Any],
) -> np.ndarray:
    """Repeat one spatial inducing support at every training time.

    Hensman et al. use u=f(Z) to replace N latent values by M inducing
    variables. Hamelijnck et al., Figure 1, provides the bridge to ST-SVGP:
    repeated space-time inducing locations in standard SVGP correspond to
    spatial inducing locations tracked through time in ST-SVGP.
    """
    spatial = train_inputs[:, n_covariates : n_covariates + 2]
    times = np.unique(train_inputs[:, n_covariates + 2])

    requested = int(config["inducing"]["spatial_points"])
    unique_spatial = np.unique(spatial, axis=0)

    if len(unique_spatial) < requested:
        raise ValueError(
            f"Only {len(unique_spatial)} unique locations for "
            f"{requested} requested spatial inducing points."
        )

    kmeans = KMeans(
        n_clusters=requested,
        random_state=int(config["training"]["random_state"]),
        n_init=int(config["inducing"]["kmeans_n_init"]),
    ).fit(unique_spatial)

    centres = kmeans.cluster_centers_
    dimension = train_inputs.shape[1]
    blocks = []

    for time_value in np.sort(times):
        block = np.zeros(
            (requested, dimension),
            dtype=gpflow.config.default_float(),
        )
        block[:, n_covariates : n_covariates + 2] = centres
        block[:, n_covariates + 2] = time_value
        blocks.append(block)

    return np.vstack(blocks)


def _temporal_kernel(
    family: str,
    active_dim: list[int],
    lengthscale: float,
) -> gpflow.kernels.Kernel:
    """Construct the Markov-compatible temporal kernel under test."""
    if family == "matern32":
        return gpflow.kernels.Matern32(
            variance=1.0,
            lengthscales=float(lengthscale),
            active_dims=active_dim,
        )
    if family == "matern12":
        return gpflow.kernels.Matern12(
            variance=1.0,
            lengthscales=float(lengthscale),
            active_dims=active_dim,
        )
    raise ValueError("temporal_family must be matern32 or matern12.")


def build_model(
    inducing_points: np.ndarray,
    n_covariates: int,
    num_data: int,
    config: dict[str, Any],
) -> gpflow.models.SVGP:
    """Construct Bernoulli-probit SVGP with separable space-time kernel."""
    kernel_config = config["kernel"]

    spatial_dims = [n_covariates, n_covariates + 1]
    temporal_dim = [n_covariates + 2]

    spatial = gpflow.kernels.Matern32(
        variance=float(kernel_config["variance"]),
        lengthscales=[
            float(kernel_config["spatial_initial_lengthscale_km"]),
            float(kernel_config["spatial_initial_lengthscale_km"]),
        ],
        active_dims=spatial_dims,
    )
    temporal = _temporal_kernel(
        str(kernel_config["temporal_family"]),
        temporal_dim,
        float(kernel_config["temporal_initial_lengthscale_steps"]),
    )

    # Only one product amplitude is identifiable, so temporal variance is fixed.
    gpflow.set_trainable(temporal.variance, False)

    model = gpflow.models.SVGP(
        kernel=spatial * temporal,
        likelihood=gpflow.likelihoods.Bernoulli(),
        inducing_variable=inducing_points,
        mean_function=CovariateLinearMean(n_covariates),
        num_data=int(num_data),
        q_diag=False,
        whiten=True,
    )

    if not bool(config["inducing"]["train_locations"]):
        gpflow.set_trainable(model.inducing_variable.Z, False)

    return model


def _make_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
) -> tf.data.Dataset:
    """Create prevalence-preserving stochastic mini-batches."""
    training = config["training"]
    seed = int(training["random_state"])

    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(
            buffer_size=min(
                len(x_train),
                int(training["shuffle_buffer"]),
            ),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .repeat()
        .batch(int(training["batch_size"]))
        .prefetch(tf.data.AUTOTUNE)
    )


def _adam_step(
    optimizer: Any,
    loss: Any,
    model: gpflow.models.SVGP,
) -> None:
    """Apply one stochastic Adam step to currently trainable variables."""
    optimizer.minimize(
        loss,
        var_list=model.trainable_variables,
    )


def _natgrad_context(device: str):
    """Optionally keep NaturalGradient algebra on CPU on Apple Metal systems."""
    if str(device).lower() == "cpu":
        return tf.device("/CPU:0")
    return nullcontext()



def natural_gradient_gamma(
    step: int,
    initial_gamma: float,
    target_gamma: float,
    ramp_iterations: int,
) -> float:
    """Return a log-linear NaturalGradient step-size schedule.

    Salimbeni et al. (2018) use very small initial natural-gradient steps and
    increase gamma during optimisation. This avoids an abrupt switch to a
    large fixed natural-gradient step.

    Args:
        step: One-based training iteration.
        initial_gamma: Positive starting step size.
        target_gamma: Positive maximum step size.
        ramp_iterations: Iteration at which target_gamma is reached.

    Returns:
        Scheduled gamma in [initial_gamma, target_gamma].
    """
    if initial_gamma <= 0.0 or target_gamma <= 0.0:
        raise ValueError("Natural-gradient gammas must be positive.")
    if target_gamma < initial_gamma:
        raise ValueError("target_gamma must be >= initial_gamma.")
    if ramp_iterations < 1:
        raise ValueError("ramp_iterations must be >= 1.")

    fraction = min(
        max((int(step) - 1) / max(ramp_iterations - 1, 1), 0.0),
        1.0,
    )
    log_gamma = (
        np.log(initial_gamma)
        + fraction * (np.log(target_gamma) - np.log(initial_gamma))
    )
    return float(np.exp(log_gamma))


def _safe_natural_gradient_step(
    model: gpflow.models.SVGP,
    loss: Any,
    optimizer: gpflow.optimizers.NaturalGradient,
    gamma: float,
    config: dict[str, Any],
) -> tuple[float, int]:
    """Apply one XiNat update with same-batch backtracking.

    The retry must use exactly the same mini-batch. Otherwise halving gamma
    while changing the stochastic gradient is not a genuine stability check.

    GPflow's default XiNat parameterisation updates the natural parameters of
    q(u) directly and is the closest implementation-level match to the natural
    parameter update underlying the CVI interpretation.

    Returns:
        Applied gamma and number of backtracking reductions.
    """
    ng = config["inference"]["natural_gradient"]
    minimum = float(ng["minimum_gamma"])
    maximum_retries = int(ng["maximum_retries"])

    current_gamma = float(gamma)
    q_mu_before = model.q_mu.numpy().copy()
    q_sqrt_before = model.q_sqrt.numpy().copy()

    for attempt in range(maximum_retries + 1):
        # Reuse one optimizer instance for the whole training run. GPflow's
        # NaturalGradient constructor subclasses tf_keras Optimizer, and
        # constructing it at every step triggers repeated Apple/Keras warnings.
        optimizer.gamma = current_gamma

        try:
            optimizer.minimize(
                loss,
                var_list=[(model.q_mu, model.q_sqrt)],
            )

            tf.debugging.assert_all_finite(
                model.q_mu,
                "NaturalGradient produced non-finite q_mu.",
            )
            tf.debugging.assert_all_finite(
                model.q_sqrt,
                "NaturalGradient produced non-finite q_sqrt.",
            )
            return current_gamma, attempt

        except (tf.errors.InvalidArgumentError, FloatingPointError):
            model.q_mu.assign(q_mu_before)
            model.q_sqrt.assign(q_sqrt_before)

            if attempt >= maximum_retries:
                raise RuntimeError(
                    "NaturalGradient remained unstable on the same mini-batch "
                    f"after {maximum_retries} backtracking retries. "
                    f"Last gamma={current_gamma:g}."
                )

            next_gamma = current_gamma * 0.5
            if next_gamma < minimum:
                raise RuntimeError(
                    "NaturalGradient requires gamma below the configured "
                    f"minimum_gamma={minimum:g} on this mini-batch."
                )

            current_gamma = next_gamma

    raise RuntimeError("NaturalGradient backtracking failed unexpectedly.")


def model_state(
    model: gpflow.models.SVGP,
) -> dict[str, object]:
    """Extract fitted kernel and mean parameters for diagnostics."""
    spatial, temporal = model.kernel.kernels
    mean = model.mean_function

    return {
        "spatial_lengthscales_km": np.asarray(
            spatial.lengthscales.numpy()
        ).tolist(),
        "temporal_lengthscale_steps": float(
            temporal.lengthscales.numpy()
        ),
        "kernel_variance": float(spatial.variance.numpy()),
        "linear_intercept": float(mean.intercept.numpy()[0]),
        "linear_coefficients_standardised": np.asarray(
            mean.beta.numpy()
        )[:, 0].tolist(),
        "inducing_count": int(model.inducing_variable.Z.shape[0]),
    }


def train_model(
    model: gpflow.models.SVGP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    iterations: int,
    config: dict[str, Any],
    run_name: str,
) -> pd.DataFrame:
    """Optimise the stochastic SVGP ELBO.

    `method=adam` preserves the existing working reference.

    `method=natgrad` follows the standard interleaving pattern:
      - q_mu/q_sqrt are excluded from Adam from the beginning;
      - Adam updates kernel and parametric-mean parameters first;
      - XiNat NaturalGradient then updates q(u);
      - both optimisers see the same mini-batch at each iteration;
      - gamma grows log-linearly from a very small initial value.
    """
    tf.random.set_seed(int(config["training"]["random_state"]))
    np.random.seed(int(config["training"]["random_state"]))

    dataset = _make_dataset(x_train, y_train, config)
    iterator = iter(dataset)

    method = str(config["inference"]["method"]).lower()
    adam_learning_rate = float(
        config["inference"]["adam"]["learning_rate"]
    )

    if method == "natgrad":
        gpflow.set_trainable(model.q_mu, False)
        gpflow.set_trainable(model.q_sqrt, False)

    adam = tf_keras.optimizers.Adam(
        learning_rate=adam_learning_rate
    )

    ng = config["inference"]["natural_gradient"]
    natural_gradient_optimizer = None
    if method == "natgrad":
        # Construct exactly once per fold/final fit. The gamma attribute is
        # updated by the schedule/backtracking without recreating the object.
        natural_gradient_optimizer = gpflow.optimizers.NaturalGradient(
            gamma=float(ng["initial_gamma"]),
            xi_transform=XiNat(),
        )
    records: list[dict[str, object]] = []
    log_every = int(config["training"]["log_every"])

    for step in range(1, int(iterations) + 1):
        batch = next(iterator)

        # A concrete batch is reused by NatGrad, Adam and any NatGrad retry.
        loss = model.training_loss_closure(
            batch,
            compile=False,
        )

        scheduled_gamma = float("nan")
        applied_gamma = float("nan")
        backtracks = 0

        if method == "adam":
            _adam_step(adam, loss, model)
            phase = "adam"

        elif method == "natgrad":
            scheduled_gamma = natural_gradient_gamma(
                step=step,
                initial_gamma=float(ng["initial_gamma"]),
                target_gamma=float(ng["target_gamma"]),
                ramp_iterations=int(ng["ramp_iterations"]),
            )

            # Salimbeni et al. and the GPflow examples update
            # hyperparameters with Adam first, then q(u) with NaturalGradient.
            # q_mu/q_sqrt are non-trainable for Adam, so this step changes only
            # kernel and parametric-mean parameters.
            _adam_step(adam, loss, model)

            assert natural_gradient_optimizer is not None
            applied_gamma, backtracks = _safe_natural_gradient_step(
                model=model,
                loss=loss,
                optimizer=natural_gradient_optimizer,
                gamma=scheduled_gamma,
                config=config,
            )
            phase = "adam+natgrad"

        else:
            raise ValueError("inference.method must be adam or natgrad.")

        tf.debugging.assert_all_finite(
            model.q_mu,
            "SVGP produced non-finite q_mu.",
        )
        tf.debugging.assert_all_finite(
            model.q_sqrt,
            "SVGP produced non-finite q_sqrt.",
        )

        if step == 1 or step % log_every == 0 or step == iterations:
            elbo = float(model.elbo(batch).numpy())
            state = model_state(model)

            records.append(
                {
                    "run": run_name,
                    "iteration": step,
                    "phase": phase,
                    "minibatch_elbo": elbo,
                    "scheduled_natural_gradient_gamma": scheduled_gamma,
                    "applied_natural_gradient_gamma": applied_gamma,
                    "natural_gradient_backtracks": int(backtracks),
                    "spatial_lengthscale_x_km": state[
                        "spatial_lengthscales_km"
                    ][0],
                    "spatial_lengthscale_y_km": state[
                        "spatial_lengthscales_km"
                    ][1],
                    "temporal_lengthscale_steps": state[
                        "temporal_lengthscale_steps"
                    ],
                    "kernel_variance": state["kernel_variance"],
                }
            )

            gamma_text = (
                f" gamma={applied_gamma:.3e}"
                if method == "natgrad"
                else ""
            )
            print(
                f"[{run_name}] iteration={step} phase={phase} "
                f"minibatch_elbo={elbo:.6f}{gamma_text} "
                f"backtracks={backtracks}"
            )

    gpflow.set_trainable(model.q_mu, True)
    gpflow.set_trainable(model.q_sqrt, True)

    return pd.DataFrame(records)


def predict_in_batches(
    model: gpflow.models.SVGP,
    inputs: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Bernoulli probability and latent mean/variance."""
    probabilities = []
    means = []
    variances = []

    for start in range(0, len(inputs), int(batch_size)):
        batch = inputs[start : start + int(batch_size)]
        latent_mean, latent_variance = model.predict_f(batch)
        probability, _ = model.predict_y(batch)

        probabilities.append(probability.numpy()[:, 0])
        means.append(latent_mean.numpy()[:, 0])
        variances.append(latent_variance.numpy()[:, 0])

    probability = np.concatenate(probabilities)
    validate_probabilities(probability)

    return (
        probability,
        np.concatenate(means),
        np.concatenate(variances),
    )


def evaluate_fold(
    frame: pd.DataFrame,
    config: dict[str, Any],
    fold_index: int,
    fold: dict[str, Any],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit/evaluate one chronological fold."""
    dataset = config["dataset"]
    origin_column = str(dataset["forecast_origin"])
    target_column = str(dataset["target"])

    train_origins = [int(v) for v in fold["train_origins"]]
    validation_origin = int(fold["validation_origin"])

    train = select_rows(frame, origin_column, train_origins)
    validation = select_rows(
        frame,
        origin_column,
        [validation_origin],
    )

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

    metrics = probabilistic_metrics(
        y_validation,
        probability,
    )
    calibration = calibration_metrics(
        y_validation,
        probability,
        n_bins=int(config["calibration"]["reliability_bins"]),
        strategy=str(config["calibration"]["binning"]),
    )
    state = model_state(model)

    observed = float(y_validation.mean())
    record = {
        "fold": fold_index,
        "train_origins": ",".join(map(str, train_origins)),
        "validation_origin": validation_origin,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "observed_prevalence": observed,
        "mean_predicted_probability": float(probability.mean()),
        **metrics,
        **calibration,
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

    reliability = reliability_table(
        y_validation,
        probability,
        n_bins=int(config["calibration"]["reliability_bins"]),
        strategy=str(config["calibration"]["binning"]),
    )
    reliability["fold"] = fold_index
    reliability["validation_origin"] = validation_origin

    return record, predictions, history, reliability


def summarise_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise probability quality, ranking, calibration and robustness."""
    return pd.DataFrame(
        [
            {
                "folds": int(metrics["fold"].nunique()),
                "mean_log_loss": float(metrics["log_loss"].mean()),
                "std_log_loss": float(metrics["log_loss"].std()),
                "worst_log_loss": float(metrics["log_loss"].max()),
                "mean_brier_score": float(metrics["brier_score"].mean()),
                "std_brier_score": float(metrics["brier_score"].std()),
                "worst_brier_score": float(metrics["brier_score"].max()),
                "mean_pr_auc": float(metrics["pr_auc"].mean()),
                "std_pr_auc": float(metrics["pr_auc"].std()),
                "worst_pr_auc": float(metrics["pr_auc"].min()),
                "mean_roc_auc": float(metrics["roc_auc"].mean()),
                "std_roc_auc": float(metrics["roc_auc"].std()),
                "mean_probability_bias": float(
                    metrics["probability_bias"].mean()
                ),
                "mean_absolute_probability_bias": float(
                    metrics["absolute_probability_bias"].mean()
                ),
                "mean_ece": float(metrics["ece"].mean()),
                "std_ece": float(metrics["ece"].std()),
                "mean_calibration_intercept": float(
                    metrics["calibration_intercept"].mean()
                ),
                "mean_calibration_slope": float(
                    metrics["calibration_slope"].mean()
                ),
            }
        ]
    )


def save_preprocessor(
    preprocessor: Preprocessor,
    path: Path,
) -> None:
    """Persist final-fit transforms needed for later prediction."""
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
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def fit_final(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[gpflow.models.SVGP, Preprocessor, pd.DataFrame]:
    """Fit selected configuration on all four pre-test forecast origins."""
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


def output_directories(
    config: dict[str, Any],
    output_tag: str | None,
) -> dict[str, Path]:
    """Resolve normal or tuning-specific output directories."""
    outputs = config["outputs"]
    directories = {
        "metadata": Path(outputs["metadata_directory"]),
        "model": Path(outputs["model_directory"]),
        "metrics": Path(outputs["metrics_directory"]),
        "predictions": Path(outputs["predictions_directory"]),
    }

    if output_tag:
        directories = {
            "metadata": directories["metadata"]
            / "tuning"
            / output_tag,
            "model": directories["model"]
            / "tuning"
            / output_tag,
            "metrics": directories["metrics"]
            / "tuning"
            / output_tag,
            "predictions": directories["predictions"]
            / "tuning"
            / output_tag,
        }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def run(
    config_path: Path,
    preflight_only: bool = False,
    validation_only: bool = False,
    output_tag: str | None = None,
) -> dict[str, object]:
    """Run SVGP preflight, rolling validation and optional final fit."""
    config = load_config(config_path)
    runtime = configure_runtime(config)
    validate_temporal_contract(config)
    frame = load_dataset(config)
    directories = output_directories(config, output_tag)

    preflight = {
        "status": "PASS",
        "model": "svgp",
        "dataset_path": config["dataset"]["path"],
        "dataset_sha256": sha256(Path(config["dataset"]["path"])),
        "dataset_rows": int(len(frame)),
        "feature_set": config["feature_set"],
        "linear_predictors": list(
            feature_names(str(config["feature_set"]))
        ),
        "inference_method": config["inference"]["method"],
        "kernel": {
            "spatial": config["kernel"]["spatial_family"],
            "temporal": config["kernel"]["temporal_family"],
        },
        "spatial_inducing_points": int(
            config["inducing"]["spatial_points"]
        ),
        "final_test_locked_origins": config["final_test"]["origins"],
        "final_test_evaluated": False,
        "runtime": runtime,
        "config_sha256": sha256(config_path),
    }
    (
        directories["metadata"] / "preflight.json"
    ).write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )

    if preflight_only:
        return preflight

    metric_records = []
    prediction_frames = []
    history_frames = []
    reliability_frames = []

    for fold_index, fold in enumerate(
        config["rolling_validation"]["folds"],
        start=1,
    ):
        record, predictions, history, reliability = evaluate_fold(
            frame,
            config,
            fold_index,
            fold,
        )
        metric_records.append(record)
        prediction_frames.append(predictions)
        history_frames.append(history)
        reliability_frames.append(reliability)

    fold_metrics = pd.DataFrame(metric_records)
    summary = summarise_folds(fold_metrics)

    fold_metrics.to_csv(
        directories["metrics"]
        / "svgp_rolling_validation_metrics.csv",
        index=False,
    )
    summary.to_csv(
        directories["metrics"]
        / "svgp_rolling_validation_summary.csv",
        index=False,
    )
    pd.concat(
        history_frames,
        ignore_index=True,
    ).to_csv(
        directories["metrics"] / "svgp_training_history.csv",
        index=False,
    )
    pd.concat(
        reliability_frames,
        ignore_index=True,
    ).to_csv(
        directories["metrics"] / "svgp_reliability_bins.csv",
        index=False,
    )

    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_parquet(
        directories["predictions"]
        / "svgp_oof_predictions.parquet",
        index=False,
    )

    if validation_only:
        result = {
            "status": "PASS",
            "model": "svgp",
            "validation_only": True,
            "feature_set": config["feature_set"],
            "inference_method": config["inference"]["method"],
            "historical_validation": summary.iloc[0].to_dict(),
            "metrics_directory": str(directories["metrics"]),
        }
        (
            directories["metadata"] / "validation_result.json"
        ).write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    final_model, preprocessor, final_history = fit_final(
        frame,
        config,
    )
    final_history.to_csv(
        directories["metrics"] / "svgp_final_training_history.csv",
        index=False,
    )

    save_preprocessor(
        preprocessor,
        directories["model"] / "preprocessing.json",
    )

    checkpoint_prefix = str(
        directories["model"] / "checkpoint"
    )
    tf.train.Checkpoint(model=final_model).write(
        checkpoint_prefix
    )

    state = model_state(final_model)
    pd.DataFrame(
        {
            "feature": list(preprocessor.linear_predictors),
            "coefficient_standardised": state[
                "linear_coefficients_standardised"
            ],
        }
    ).to_csv(
        directories["metrics"]
        / "svgp_linear_mean_coefficients.csv",
        index=False,
    )

    model_card = {
        "status": "PASS",
        "model": "svgp",
        "likelihood": "bernoulli_probit",
        "feature_set": config["feature_set"],
        "linear_predictors": list(preprocessor.linear_predictors),
        "inference_method": config["inference"]["method"],
        "natural_gradient": (
            config["inference"]["natural_gradient"]
            if config["inference"]["method"] == "natgrad"
            else None
        ),
        "kernel": {
            "spatial": config["kernel"]["spatial_family"],
            "temporal": config["kernel"]["temporal_family"],
        },
        "spatial_inducing_points": int(
            config["inducing"]["spatial_points"]
        ),
        "historical_validation": summary.iloc[0].to_dict(),
        "final_fit_origins": config["final_fit"]["origins"],
        "final_test_locked_origins": config["final_test"]["origins"],
        "final_test_evaluated": False,
        "runtime": runtime,
        "fitted_state": state,
        "checkpoint_prefix": checkpoint_prefix,
    }
    (
        directories["model"] / "model_card.json"
    ).write_text(
        json.dumps(model_card, indent=2) + "\n",
        encoding="utf-8",
    )

    return model_card


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    result = run(
        config_path=args.config,
        preflight_only=args.preflight_only,
        validation_only=args.validation_only,
        output_tag=args.output_tag,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
