"""Train the full ST-SVGP urban-expansion model.

Model
-----
    f_i,t = beta_0 + x_i,t^T beta + g(s_i, t)

    g ~ GP(0, k_M32_space * k_M32_time)

    Y_i,t | f_i,t ~ Bernoulli(Phi(f_i,t))

Inference
---------
- 64 fixed spatial inducing locations;
- Matérn-3/2 temporal Markov state;
- one dense Gaussian CVI site block per training time;
- sequential filtering + RTS smoothing;
- Natural Gradient for CVI site natural parameters;
- Adam for beta and kernel hyperparameters;
- chronological rolling validation;
- locked 2020 final test.

This file intentionally does not evaluate the 2020 -> 2025 transition.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from src.feature_engineering.urban_expansion import (
    add_candidate_features,
)
from src.models.evaluation.calibration import (
    calibration_metrics,
)
from src.models.st_svgp.block_inference import (
    DenseCviSites,
    damped_dense_natural_gradient_update,
    initialise_dense_sites,
    site_precision,
)
from src.models.st_svgp.cvi import (
    probit_predictive_probability,
)
from src.models.st_svgp.model import (
    STPosterior,
    STSVGPModel,
)


@dataclass(frozen=True)
class FoldPreprocessing:
    """Training-only preprocessing frozen for one temporal fold."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coordinate_center_km: np.ndarray
    inducing_locations_km: np.ndarray


@dataclass(frozen=True)
class TimeData:
    """Prepared arrays for one forecast origin."""

    origin: int
    time_step: float
    temporal_trend_time: float | None
    features: np.ndarray
    coordinates_km: np.ndarray
    targets: np.ndarray
    row_index: np.ndarray


@dataclass(frozen=True)
class FittedFoldState:
    """Objects still in memory when one rolling fold finishes fitting."""

    model: STSVGPModel
    sites: DenseCviSites
    preprocessing: FoldPreprocessing
    train_origins: tuple[int, ...]
    validation_origin: int


FoldStateCallback = Callable[[FittedFoldState], None]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            "ST-SVGP configuration must be a mapping."
        )

    if config.get("model") != "st_svgp":
        raise ValueError(
            "Expected model: st_svgp."
        )

    locked_block = config.get(
        "locked_block",
        config.get("final_test"),
    )
    if not isinstance(locked_block, dict):
        raise ValueError(
            "A locked_block or final_test mapping is required."
        )

    if bool(
        locked_block.get(
            "evaluate",
            False,
        )
    ):
        raise ValueError(
            "The locked temporal block must not be evaluated."
        )

    validate_development_splits(config)

    return config


def validate_development_splits(
    config: dict[str, Any],
) -> None:
    """Reject configured development labels beyond an optional cutoff."""
    development = config.get("development")
    if development is None:
        return
    if not isinstance(development, dict):
        raise ValueError(
            "development must be a mapping."
        )

    maximum_target_year = int(
        development["maximum_target_year"]
    )
    horizon_years = int(
        config["time"]["step_years"]
    )

    configured_origins: list[tuple[str, int]] = []
    for fold_number, fold in enumerate(
        config["rolling_validation"]["folds"],
        start=1,
    ):
        configured_origins.extend(
            (
                f"fold {fold_number} training",
                int(origin),
            )
            for origin in fold["train_origins"]
        )
        configured_origins.append(
            (
                f"fold {fold_number} validation",
                int(fold["validation_origin"]),
            )
        )

    configured_origins.extend(
        ("final fit", int(origin))
        for origin in config["final_fit"]["origins"]
    )

    violations = [
        (
            label,
            origin,
            origin + horizon_years,
        )
        for label, origin in configured_origins
        if origin + horizon_years
        > maximum_target_year
    ]
    if violations:
        details = ", ".join(
            f"{label} {origin}->{target_year}"
            for label, origin, target_year in violations
        )
        raise ValueError(
            "Development target years must be at most "
            f"{maximum_target_year}: {details}."
        )

    locked_block = config.get("locked_block")
    if locked_block is not None:
        locked_origins = [
            int(value)
            for value in locked_block["origins"]
        ]
        locked_target_years = [
            int(value)
            for value in locked_block[
                "target_years"
            ]
        ]
        expected_target_years = [
            origin + horizon_years
            for origin in locked_origins
        ]
        if locked_target_years != expected_target_years:
            raise ValueError(
                "Locked target years must match locked origins "
                "plus time.step_years."
            )
        if any(
            target_year <= maximum_target_year
            for target_year in locked_target_years
        ):
            raise ValueError(
                "Locked target years must follow the development period."
            )


def configure_runtime(
    config: dict[str, Any],
) -> tf.dtypes.DType:
    """Configure the reference TensorFlow runtime."""
    compute = config["compute"]

    if str(compute["float_type"]) != "float64":
        raise ValueError(
            "The first real ST-SVGP implementation "
            "is intentionally frozen to float64."
        )

    if str(compute["device"]).lower() == "cpu":
        try:
            tf.config.set_visible_devices(
                [],
                "GPU",
            )
        except RuntimeError:
            # TensorFlow may already have initialised the runtime.
            pass

    tf.keras.backend.set_floatx("float64")
    return tf.float64


def validate_dataset(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create promoted features and validate the temporal contract."""
    dataset = config["dataset"]
    feature_engineering = config.get(
        "feature_engineering",
        {},
    )

    target = str(dataset["target"])
    origin = str(dataset["forecast_origin"])
    target_year = str(dataset["target_year"])

    frame = add_candidate_features(
        frame,
        recent_growth_column=str(
            feature_engineering.get(
                "recent_growth_column",
                "recent_local_growth_5y_t",
            )
        ),
    )

    required = {
        target,
        origin,
        target_year,
        str(dataset["cell_id"]),
        str(dataset["x_coordinate"]),
        str(dataset["y_coordinate"]),
        *[
            str(value)
            for value in config[
                "linear_predictors"
            ]
        ],
    }

    missing = sorted(
        required.difference(frame.columns)
    )
    if missing:
        raise ValueError(
            "Missing ST-SVGP columns: "
            + ", ".join(missing)
        )

    origins = pd.to_numeric(
        frame[origin],
        errors="raise",
    ).astype(int)
    target_years = pd.to_numeric(
        frame[target_year],
        errors="raise",
    ).astype(int)

    horizon_years = int(
        config["time"]["step_years"]
    )
    if horizon_years < 1:
        raise ValueError(
            "time.step_years must be positive."
        )

    if not target_years.eq(
        origins + horizon_years
    ).all():
        raise ValueError(
            "target_year must equal forecast_origin + "
            f"{horizon_years}."
        )

    y = pd.to_numeric(
        frame[target],
        errors="raise",
    ).astype(int)

    if not set(y.unique()).issubset(
        {0, 1}
    ):
        raise ValueError(
            "ST-SVGP target must be binary."
        )

    return frame


def load_dataset(
    config: dict[str, Any],
) -> pd.DataFrame:
    path = Path(
        config["dataset"]["path"]
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    return validate_dataset(
        pd.read_parquet(path),
        config,
    )


def time_step(
    origin: int,
    config: dict[str, Any],
) -> float:
    time_config = config["time"]
    return (
        int(origin)
        - int(time_config["origin_year"])
    ) / float(time_config["step_years"])


def temporal_trend_time(
    origin: int,
    config: dict[str, Any],
) -> float | None:
    trend = config.get("mean", {}).get(
        "temporal_trend",
        {},
    )
    if not bool(trend.get("enabled", False)):
        return None

    scale_years = float(trend["scale_years"])
    if scale_years <= 0.0:
        raise ValueError(
            "mean.temporal_trend.scale_years must be positive."
        )

    return (
        int(origin)
        - int(trend["reference_year"])
    ) / scale_years


def fit_preprocessing(
    train_frame: pd.DataFrame,
    config: dict[str, Any],
) -> FoldPreprocessing:
    """Fit feature scaling and inducing locations on training data only."""
    features = [
        str(value)
        for value in config[
            "linear_predictors"
        ]
    ]

    x = train_frame[
        features
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(x).all():
        raise ValueError(
            "Non-finite linear predictors."
        )

    # feature_mean = x.mean(axis=0)
    # feature_scale = x.std(
    #     axis=0,
    #     ddof=0,
    # )

    # if (
    #     (feature_scale <= 0.0).any()
    #     or not np.isfinite(
    #         feature_scale
    #     ).all()
    # ):
    #     raise ValueError(
    #         "All linear predictors require positive finite scale."
    #     )
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(
        axis=0,
        ddof=0,
    )

    if not np.isfinite(feature_mean).all():
        raise ValueError(
            "Linear predictor means must be finite."
        )

    if not np.isfinite(feature_scale).all():
        raise ValueError(
            "Linear predictor scales must be finite."
        )

    # A predictor may legitimately be constant in an early temporal fold.
    # In particular, lagged-growth features can have zero variance at the
    # earliest forecast origin. Following standard scaling behaviour, assign
    # unit scale so the centred training values remain zero rather than
    # dropping the predictor or leaking information from future origins.
    zero_variance = feature_scale <= 0.0

    if zero_variance.any():
        constant_features = [
            features[index]
            for index in np.flatnonzero(
                zero_variance
            )
        ]

        print(
            "Constant predictors in this training fold; "
            "using unit scaling: "
            + ", ".join(constant_features)
        )

        feature_scale = feature_scale.copy()
        feature_scale[zero_variance] = 1.0
    # 

    dataset = config["dataset"]
    coordinate_columns = [
        str(dataset["x_coordinate"]),
        str(dataset["y_coordinate"]),
    ]

    # Coordinates are represented in km; only translation is applied.
    coordinates_km = (
        train_frame[
            coordinate_columns
        ].to_numpy(dtype=np.float64)
        / 1000.0
    )
    coordinate_center = (
        coordinates_km.mean(axis=0)
    )
    centred = (
        coordinates_km
        - coordinate_center
    )

    # Repeated cell-time rows are unnecessary for inducing-location k-means.
    unique_coordinates = np.unique(
        centred,
        axis=0,
    )

    n_spatial = int(
        config["inducing"][
            "spatial_points"
        ]
    )

    if len(unique_coordinates) < n_spatial:
        raise ValueError(
            "Fewer unique spatial locations than inducing points."
        )

    kmeans = KMeans(
        n_clusters=n_spatial,
        n_init=int(
            config["inducing"][
                "kmeans_n_init"
            ]
        ),
        random_state=int(
            config["training"][
                "random_state"
            ]
        ),
    )
    kmeans.fit(unique_coordinates)

    return FoldPreprocessing(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coordinate_center_km=(
            coordinate_center
        ),
        inducing_locations_km=(
            kmeans.cluster_centers_
            .astype(np.float64)
        ),
    )


def prepare_time_data(
    frame: pd.DataFrame,
    *,
    origins: list[int],
    preprocessing: FoldPreprocessing,
    config: dict[str, Any],
) -> list[TimeData]:
    """Prepare one array bundle per requested forecast origin."""
    dataset = config["dataset"]
    origin_column = str(
        dataset["forecast_origin"]
    )
    target_column = str(
        dataset["target"]
    )
    features = [
        str(value)
        for value in config[
            "linear_predictors"
        ]
    ]
    coordinate_columns = [
        str(dataset["x_coordinate"]),
        str(dataset["y_coordinate"]),
    ]

    result: list[TimeData] = []

    for origin in origins:
        part = frame.loc[
            frame[origin_column]
            .astype(int)
            .eq(int(origin))
        ]

        if part.empty:
            raise ValueError(
                f"No rows for origin {origin}."
            )

        raw_features = part[
            features
        ].to_numpy(dtype=np.float64)

        scaled_features = (
            raw_features
            - preprocessing.feature_mean
        ) / preprocessing.feature_scale

        coordinates = (
            part[
                coordinate_columns
            ].to_numpy(dtype=np.float64)
            / 1000.0
            - preprocessing.coordinate_center_km
        )

        targets = part[
            target_column
        ].to_numpy(dtype=np.float64)

        result.append(
            TimeData(
                origin=int(origin),
                time_step=time_step(
                    int(origin),
                    config,
                ),
                temporal_trend_time=temporal_trend_time(
                    int(origin),
                    config,
                ),
                features=scaled_features,
                coordinates_km=coordinates,
                targets=targets,
                row_index=part.index.to_numpy(),
            )
        )

    return result


def sample_batches(
    time_data: list[TimeData],
    *,
    total_batch_size: int,
    rng: np.random.Generator,
    dtype: tf.dtypes.DType,
) -> list[
    tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        float,
        float | None,
    ]
]:
    """Sample approximately equal-size real-prevalence batches across time."""
    per_time = max(
        1,
        int(total_batch_size)
        // len(time_data),
    )

    batches = []

    for data in time_data:
        n_rows = len(data.targets)
        batch_size = min(
            per_time,
            n_rows,
        )

        indices = rng.choice(
            n_rows,
            size=batch_size,
            replace=False,
        )

        likelihood_scale = (
            n_rows
            / float(batch_size)
        )

        batches.append(
            (
                tf.constant(
                    data.features[indices],
                    dtype=dtype,
                ),
                tf.constant(
                    data.coordinates_km[
                        indices
                    ],
                    dtype=dtype,
                ),
                tf.constant(
                    data.targets[indices],
                    dtype=dtype,
                ),
                likelihood_scale,
                data.temporal_trend_time,
            )
        )

    return batches


def model_from_preprocessing(
    preprocessing: FoldPreprocessing,
    config: dict[str, Any],
    *,
    dtype: tf.dtypes.DType,
) -> STSVGPModel:
    kernel = config["kernel"]
    temporal_trend = config.get("mean", {}).get(
        "temporal_trend",
        {},
    )
    return STSVGPModel(
        inducing_locations_km=tf.constant(
            preprocessing.inducing_locations_km,
            dtype=dtype,
        ),
        n_features=len(
            config["linear_predictors"]
        ),
        spatial_initial_lengthscale_km=float(
            kernel[
                "spatial_initial_lengthscale_km"
            ]
        ),
        temporal_initial_lengthscale_steps=float(
            kernel[
                "temporal_initial_lengthscale_steps"
            ]
        ),
        temporal_lengthscale_trainable=bool(
            kernel.get(
                "temporal_lengthscale_trainable",
                True,
            )
        ),
        inducing_locations_trainable=bool(
            config["inducing"]["train_locations"]
        ),
        variance_initial=float(
            kernel["variance"]
        ),
        jitter=float(
            config["compute"]["jitter"]
        ),
        quadrature_degree=int(
            config["inference"][
                "quadrature_degree"
            ]
        ),
        temporal_trend_enabled=bool(
            temporal_trend.get(
                "enabled",
                False,
            )
        ),
        temporal_trend_initial_coefficient=float(
            temporal_trend.get(
                "initial_coefficient",
                0.0,
            )
        ),
        dtype=dtype,
    )


def natural_gradient_step(
    *,
    model: STSVGPModel,
    posterior: STPosterior,
    sites: DenseCviSites,
    batches: list[
        tuple[
            tf.Tensor,
            tf.Tensor,
            tf.Tensor,
            float,
            float | None,
        ]
    ],
    config: dict[str, Any],
) -> tuple[DenseCviSites, float]:
    """Update all dense time-specific CVI sites through Natural Gradient."""
    lambda1_targets = []
    lambda2_targets = []

    for index, (
        features,
        coordinates,
        targets,
        likelihood_scale,
        temporal_trend_value,
    ) in enumerate(batches):
        target1, target2 = (
            model.cvi_natural_targets(
                targets=targets,
                features=features,
                temporal_trend_time=(
                    temporal_trend_value
                ),
                coordinates_km=coordinates,
                inducing_mean=tf.stop_gradient(
                    posterior.inducing_means[
                        index
                    ]
                ),
                inducing_covariance=(
                    tf.stop_gradient(
                        posterior
                        .inducing_covariances[
                            index
                        ]
                    )
                ),
                spatial_covariance=(
                    tf.stop_gradient(
                        posterior
                        .spatial_covariance
                    )
                ),
                likelihood_scale=(
                    likelihood_scale
                ),
            )
        )
        lambda1_targets.append(target1)
        lambda2_targets.append(target2)

    inference = config["inference"]

    return damped_dense_natural_gradient_update(
        sites=sites,
        target_lambda1=tf.stack(
            lambda1_targets,
            axis=0,
        ),
        target_lambda2=tf.stack(
            lambda2_targets,
            axis=0,
        ),
        requested_gamma=float(
            inference[
                "natural_gradient_gamma"
            ]
        ),
        minimum_gamma=float(
            inference["minimum_gamma"]
        ),
        maximum_retries=int(
            inference["maximum_retries"]
        ),
        minimum_eigenvalue=float(
            inference[
                "minimum_site_eigenvalue"
            ]
        ),
    )


def spatial_kernel_regularization(
    *,
    spatial_lengthscales_km: tf.Tensor,
    kernel_variance: tf.Tensor,
    regularization: dict[str, Any] | None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return total, spatial-lengthscale, and kernel-variance penalties."""
    lengthscales = tf.convert_to_tensor(
        spatial_lengthscales_km
    )
    variance = tf.convert_to_tensor(
        kernel_variance,
        dtype=lengthscales.dtype,
    )
    zero = tf.zeros([], dtype=lengthscales.dtype)
    settings = regularization or {}

    lengthscale_penalty = zero
    lengthscale_settings = settings.get(
        "spatial_log_lengthscale",
        {},
    )
    if bool(lengthscale_settings.get("enabled", False)):
        center = tf.cast(
            lengthscale_settings["center_km"],
            lengthscales.dtype,
        )
        sigma = tf.cast(
            lengthscale_settings["sigma_log"],
            lengthscales.dtype,
        )
        weight = tf.cast(
            lengthscale_settings["weight"],
            lengthscales.dtype,
        )
        standardized = (
            tf.math.log(lengthscales)
            - tf.math.log(center)
        ) / sigma
        lengthscale_penalty = (
            tf.cast(0.5, lengthscales.dtype)
            * weight
            * tf.reduce_sum(tf.square(standardized))
        )

    variance_penalty = zero
    variance_settings = settings.get(
        "kernel_log_variance",
        {},
    )
    if bool(variance_settings.get("enabled", False)):
        center = tf.cast(
            variance_settings["center"],
            lengthscales.dtype,
        )
        sigma = tf.cast(
            variance_settings["sigma_log"],
            lengthscales.dtype,
        )
        weight = tf.cast(
            variance_settings["weight"],
            lengthscales.dtype,
        )
        standardized = (
            tf.math.log(variance)
            - tf.math.log(center)
        ) / sigma
        variance_penalty = (
            tf.cast(0.5, lengthscales.dtype)
            * weight
            * tf.square(standardized)
        )

    return (
        lengthscale_penalty + variance_penalty,
        lengthscale_penalty,
        variance_penalty,
    )


def temporal_lengthscale_regularization(
    temporal_lengthscale_steps: tf.Tensor,
    *,
    step_years: float,
    regularization: dict[str, Any] | None,
) -> tf.Tensor:
    """Return the physical-year temporal log-lengthscale penalty."""
    lengthscale_steps = tf.convert_to_tensor(
        temporal_lengthscale_steps
    )
    zero = tf.zeros([], dtype=lengthscale_steps.dtype)
    settings = (regularization or {}).get(
        "temporal_log_lengthscale",
        {},
    )
    if not bool(settings.get("enabled", False)):
        return zero

    lengthscale_years = lengthscale_steps * tf.cast(
        step_years,
        lengthscale_steps.dtype,
    )
    center_years = tf.cast(
        settings["center_years"],
        lengthscale_steps.dtype,
    )
    weight = tf.cast(
        settings["weight"],
        lengthscale_steps.dtype,
    )
    return weight * tf.square(
        tf.math.log(lengthscale_years)
        - tf.math.log(center_years)
    )


def _site_relative_change(
    previous: DenseCviSites,
    current: DenseCviSites,
    *,
    epsilon: float,
) -> float:
    numerator = tf.sqrt(
        tf.reduce_sum(
            tf.square(current.lambda1 - previous.lambda1)
        )
        + tf.reduce_sum(
            tf.square(current.lambda2 - previous.lambda2)
        )
    )
    denominator = tf.maximum(
        tf.sqrt(
            tf.reduce_sum(tf.square(previous.lambda1))
            + tf.reduce_sum(tf.square(previous.lambda2))
        ),
        tf.cast(epsilon, previous.lambda1.dtype),
    )
    return float((numerator / denominator).numpy())


def _evaluate_early_stopping_window(
    records: list[dict[str, object]],
    settings: dict[str, Any],
) -> dict[str, object]:
    checkpoint_interval = int(settings["checkpoint_interval"])
    stability_window = int(settings["stability_window_iterations"])
    window_intervals = stability_window // checkpoint_interval
    window_observations = window_intervals + 1
    result: dict[str, object] = {
        "diagnostic_elbo_endpoint_relative_change": float("nan"),
        "diagnostic_elbo_relative_range": float("nan"),
        "spatial_lengthscale_x_relative_change_window": float("nan"),
        "spatial_lengthscale_y_relative_change_window": float("nan"),
        "kernel_variance_relative_change_window": float("nan"),
        "temporal_lengthscale_relative_change_window": float("nan"),
        "temporal_trend_absolute_change_window": float("nan"),
        "early_stop_elbo_plateau_pass": False,
        "early_stop_parameter_stability_pass": False,
        "early_stop_site_stability_pass": False,
        "early_stop_convergence_gate_pass": False,
        "early_stop_eligible": False,
        "window_observation_count": 0,
        "site_interval_count": 0,
        "site_relative_change_median_window": float("nan"),
        "site_relative_change_maximum_window": float("nan"),
    }
    if len(records) < window_observations:
        return result

    window = records[-window_observations:]
    completed = [int(record["completed_iterations"]) for record in window]
    expected = list(
        range(
            completed[-1] - stability_window,
            completed[-1] + checkpoint_interval,
            checkpoint_interval,
        )
    )
    site_changes = np.asarray(
        [
            float(record["site_relative_change_since_checkpoint"])
            for record in window[1:]
        ],
        dtype=np.float64,
    )
    result["window_observation_count"] = len(window)
    result["site_interval_count"] = len(site_changes)
    eligible = bool(
        completed == expected
        and completed[-1] >= int(settings["min_iterations"])
        and np.isfinite(site_changes).all()
    )
    result["early_stop_eligible"] = eligible
    if not eligible:
        return result

    epsilon = float(settings["epsilon"])
    first = window[0]
    last = window[-1]
    diagnostic = np.asarray(
        [float(record["diagnostic_loss_fixed"]) for record in window],
        dtype=np.float64,
    )
    endpoint_change = abs(diagnostic[-1] - diagnostic[0]) / max(
        abs(diagnostic[0]), epsilon
    )
    relative_range = (diagnostic.max() - diagnostic.min()) / max(
        abs(float(np.median(diagnostic))), epsilon
    )

    def relative_change(name: str) -> float:
        first_value = float(first[name])
        return abs(float(last[name]) - first_value) / max(
            abs(first_value), epsilon
        )

    spatial_x_change = relative_change("spatial_lengthscale_x_km")
    spatial_y_change = relative_change("spatial_lengthscale_y_km")
    variance_change = relative_change("kernel_variance")
    temporal_change = relative_change("temporal_lengthscale_steps")
    trend_change = abs(
        float(last["temporal_trend_coefficient"])
        - float(first["temporal_trend_coefficient"])
    )
    site_median = float(np.median(site_changes))
    site_maximum = float(np.max(site_changes))

    diagnostic_thresholds = settings["diagnostic_elbo"]
    parameter_thresholds = settings["parameter_stability"]
    site_thresholds = settings["site_stability"]
    elbo_pass = bool(
        endpoint_change
        <= float(diagnostic_thresholds["endpoint_relative_change_max"])
        and relative_range
        <= float(diagnostic_thresholds["relative_range_max"])
    )
    parameter_pass = bool(
        spatial_x_change
        <= float(
            parameter_thresholds[
                "spatial_lengthscale_relative_change_max"
            ]
        )
        and spatial_y_change
        <= float(
            parameter_thresholds[
                "spatial_lengthscale_relative_change_max"
            ]
        )
        and variance_change
        <= float(
            parameter_thresholds[
                "kernel_variance_relative_change_max"
            ]
        )
        and temporal_change
        <= float(
            parameter_thresholds[
                "temporal_lengthscale_relative_change_max"
            ]
        )
        and trend_change
        <= float(
            parameter_thresholds[
                "temporal_trend_absolute_change_max"
            ]
        )
    )
    site_pass = bool(
        site_median
        <= float(site_thresholds["median_relative_change_max"])
        and site_maximum
        <= float(site_thresholds["maximum_relative_change_max"])
    )
    result.update(
        {
            "diagnostic_elbo_endpoint_relative_change": endpoint_change,
            "diagnostic_elbo_relative_range": relative_range,
            "spatial_lengthscale_x_relative_change_window": spatial_x_change,
            "spatial_lengthscale_y_relative_change_window": spatial_y_change,
            "kernel_variance_relative_change_window": variance_change,
            "temporal_lengthscale_relative_change_window": temporal_change,
            "temporal_trend_absolute_change_window": trend_change,
            "early_stop_elbo_plateau_pass": elbo_pass,
            "early_stop_parameter_stability_pass": parameter_pass,
            "early_stop_site_stability_pass": site_pass,
            "early_stop_convergence_gate_pass": (
                elbo_pass and parameter_pass and site_pass
            ),
            "site_relative_change_median_window": site_median,
            "site_relative_change_maximum_window": site_maximum,
        }
    )
    return result


def _update_early_stopping_state(
    *,
    completed_iterations: int,
    eligible: bool,
    convergence_gate_pass: bool,
    consecutive_passes: int,
    settings: dict[str, Any],
) -> tuple[int, str | None, bool]:
    if eligible:
        consecutive_passes = (
            consecutive_passes + 1 if convergence_gate_pass else 0
        )
    else:
        consecutive_passes = 0

    if consecutive_passes >= int(settings["patience_checkpoints"]):
        return consecutive_passes, "CONVERGENCE_RULE", True
    if completed_iterations >= int(settings["max_iterations"]):
        return consecutive_passes, "MAX_ITERATIONS", False
    return consecutive_passes, None, False


def train_model(
    *,
    model: STSVGPModel,
    time_data: list[TimeData],
    iterations: int,
    config: dict[str, Any],
    dtype: tf.dtypes.DType,
    seed_offset: int,
) -> tuple[
    DenseCviSites,
    STPosterior,
    pd.DataFrame,
]:
    """Alternate CVI Natural Gradient and Adam hyperparameter steps."""
    training = config["training"]
    inference = config["inference"]
    early_stopping = config.get("early_stopping", {})
    early_stopping_enabled = bool(
        early_stopping.get("enabled", False)
    )

    rng = np.random.default_rng(
        int(training["random_state"])
        + int(seed_offset)
    )

    diagnostic_batch_sets = []
    n_train = sum(len(data.targets) for data in time_data)
    if early_stopping_enabled:
        diagnostic_rng = np.random.default_rng(
            int(training["random_state"])
            + int(early_stopping["diagnostic_seed_offset"])
            + int(seed_offset)
        )
        diagnostic_batch_sets = [
            sample_batches(
                time_data,
                total_batch_size=int(training["batch_size"]),
                rng=diagnostic_rng,
                dtype=dtype,
            )
            for _ in range(
                int(early_stopping["fixed_diagnostic_batch_sets"])
            )
        ]

    sites = initialise_dense_sites(
        len(time_data),
        model.n_spatial,
        initial_precision=float(
            inference[
                "initial_site_precision"
            ]
        ),
        dtype=dtype,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(
            training[
                "adam_learning_rate"
            ]
        )
    )

    times = tf.constant(
        [
            data.time_step
            for data in time_data
        ],
        dtype=dtype,
    )

    history: list[dict[str, object]] = []
    log_every = int(
        training["log_every"]
    )
    initial_inducing_locations = tf.identity(
        model.inducing_locations_km
    )

    start = time.perf_counter()
    training_iterations = (
        int(early_stopping["max_iterations"])
        if early_stopping_enabled
        else int(iterations)
    )
    previous_site_snapshot: DenseCviSites | None = None
    consecutive_passes = 0

    for iteration in range(training_iterations):
        batches = sample_batches(
            time_data,
            total_batch_size=int(
                training["batch_size"]
            ),
            rng=rng,
            dtype=dtype,
        )

        # 1. Current structured posterior.
        posterior = model.posterior(
            times=times,
            sites=sites,
        )

        # 2. CVI Natural-Gradient site update.
        sites, applied_gamma = (
            natural_gradient_step(
                model=model,
                posterior=posterior,
                sites=sites,
                batches=batches,
                config=config,
            )
        )

        # 3. Adam step on beta and kernel hyperparameters.
        with tf.GradientTape() as tape:
            elbo, posterior_after_sites = (
                model.stochastic_elbo(
                    times=times,
                    sites=sites,
                    batches=batches,
                )
            )
            (
                regularization_penalty,
                lengthscale_penalty,
                variance_penalty,
            ) = spatial_kernel_regularization(
                spatial_lengthscales_km=(
                    model.spatial_lengthscales
                ),
                kernel_variance=model.variance,
                regularization=config.get(
                    "regularization"
                ),
            )
            temporal_penalty = temporal_lengthscale_regularization(
                model.temporal_lengthscale,
                step_years=float(
                    config["time"]["step_years"]
                ),
                regularization=config.get(
                    "regularization"
                ),
            )
            total_regularization_penalty = (
                regularization_penalty
                + temporal_penalty
            )
            loss = -elbo + total_regularization_penalty

        gradients = tape.gradient(
            loss,
            model.trainable_variables,
        )

        if any(
            gradient is None
            for gradient in gradients
        ):
            raise RuntimeError(
                "Missing Adam gradient in ST-SVGP."
            )

        finite = all(
            bool(
                tf.reduce_all(
                    tf.math.is_finite(
                        gradient
                    )
                ).numpy()
            )
            for gradient in gradients
        )
        if not finite:
            raise RuntimeError(
                "Non-finite Adam gradient in ST-SVGP."
            )

        gradients, gradient_norm = (
            tf.clip_by_global_norm(
                gradients,
                float(
                    training[
                        "gradient_clip_norm"
                    ]
                ),
            )
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables,
                strict=True,
            )
        )

        completed_iterations = iteration + 1
        early_stop_checkpoint = (
            early_stopping_enabled
            and completed_iterations
            % int(early_stopping["checkpoint_interval"])
            == 0
        )
        fixed_iteration_log = (
            not early_stopping_enabled
            and (
                iteration % log_every == 0
                or iteration == training_iterations - 1
            )
        )
        if early_stop_checkpoint or fixed_iteration_log:
            diagnostic_loss_fixed = float("nan")
            site_relative_change = float("nan")
            current_site_snapshot: DenseCviSites | None = None
            if early_stopping_enabled:
                diagnostic_losses = []
                for diagnostic_batches in diagnostic_batch_sets:
                    diagnostic_elbo, _ = model.stochastic_elbo(
                        times=times,
                        sites=sites,
                        batches=diagnostic_batches,
                    )
                    diagnostic_losses.append(
                        -float(diagnostic_elbo.numpy()) / n_train
                    )
                diagnostic_loss_fixed = float(
                    np.mean(diagnostic_losses)
                )
                current_site_snapshot = DenseCviSites(
                    lambda1=tf.identity(sites.lambda1),
                    lambda2=tf.identity(sites.lambda2),
                )
                if previous_site_snapshot is not None:
                    site_relative_change = _site_relative_change(
                        previous_site_snapshot,
                        current_site_snapshot,
                        epsilon=float(early_stopping["epsilon"]),
                    )

            precision = site_precision(
                sites
            )
            inducing_displacements_m = (
                tf.linalg.norm(
                    model.inducing_locations_km
                    - initial_inducing_locations,
                    axis=1,
                )
                * tf.cast(1000.0, dtype)
            )
            pairwise_distances_m = (
                tf.linalg.norm(
                    model.inducing_locations_km[:, None, :]
                    - model.inducing_locations_km[None, :, :],
                    axis=2,
                )
                * tf.cast(1000.0, dtype)
            )
            off_diagonal = tf.logical_not(
                tf.eye(model.n_spatial, dtype=tf.bool)
            )
            minimum_precision = float(
                tf.reduce_min(
                    tf.linalg.eigvalsh(
                        precision
                    )
                ).numpy()
            )

            record = {
                "iteration": int(
                    iteration
                ),
                "elbo_stochastic": float(
                    elbo.numpy()
                ),
                "regularization_spatial_log_lengthscale": float(
                    lengthscale_penalty.numpy()
                ),
                "regularization_kernel_log_variance": float(
                    variance_penalty.numpy()
                ),
                "regularization_total": float(
                    total_regularization_penalty.numpy()
                ),
                "natural_gradient_gamma": float(
                    applied_gamma
                ),
                "minimum_site_precision_eigenvalue": (
                    minimum_precision
                ),
                "gradient_global_norm": float(
                    gradient_norm.numpy()
                ),
                "spatial_lengthscale_x_km": float(
                    model.spatial_lengthscales[
                        0
                    ].numpy()
                ),
                "spatial_lengthscale_y_km": float(
                    model.spatial_lengthscales[
                        1
                    ].numpy()
                ),
                "temporal_lengthscale_steps": float(
                    model.temporal_lengthscale
                    .numpy()
                ),
                "temporal_lengthscale_trainable": bool(
                    model.temporal_lengthscale_trainable
                ),
                "kernel_variance": float(
                    model.variance.numpy()
                ),
                "linear_intercept": float(
                    model.beta0.numpy()
                ),
                "inducing_mean_displacement_m": float(
                    tf.reduce_mean(
                        inducing_displacements_m
                    ).numpy()
                ),
                "inducing_max_displacement_m": float(
                    tf.reduce_max(
                        inducing_displacements_m
                    ).numpy()
                ),
                "inducing_min_pairwise_distance_m": float(
                    tf.reduce_min(
                        tf.boolean_mask(
                            pairwise_distances_m,
                            off_diagonal,
                        )
                    ).numpy()
                ),
                "elapsed_seconds": float(
                    time.perf_counter()
                    - start
                ),
            }
            temporal_settings = config.get(
                "regularization",
                {},
            ).get(
                "temporal_log_lengthscale",
                {},
            )
            if bool(temporal_settings.get("enabled", False)):
                record[
                    "regularization_temporal_log_lengthscale"
                ] = float(temporal_penalty.numpy())
            if model.beta_time is not None:
                record["temporal_trend_coefficient"] = float(
                    model.beta_time.numpy()
                )

            stop_reason: str | None = None
            if early_stopping_enabled:
                record.update(
                    {
                        "completed_iterations": completed_iterations,
                        "diagnostic_loss_fixed": diagnostic_loss_fixed,
                        "site_relative_change_since_checkpoint": (
                            site_relative_change
                        ),
                    }
                )
                window = _evaluate_early_stopping_window(
                    [*history, record],
                    early_stopping,
                )
                audit_fields = (
                    "diagnostic_elbo_endpoint_relative_change",
                    "diagnostic_elbo_relative_range",
                    "spatial_lengthscale_x_relative_change_window",
                    "spatial_lengthscale_y_relative_change_window",
                    "kernel_variance_relative_change_window",
                    "temporal_lengthscale_relative_change_window",
                    "temporal_trend_absolute_change_window",
                    "early_stop_elbo_plateau_pass",
                    "early_stop_parameter_stability_pass",
                    "early_stop_site_stability_pass",
                    "early_stop_convergence_gate_pass",
                    "early_stop_eligible",
                )
                record.update(
                    {field: window[field] for field in audit_fields}
                )
                (
                    consecutive_passes,
                    stop_reason,
                    _,
                ) = _update_early_stopping_state(
                    completed_iterations=completed_iterations,
                    eligible=bool(window["early_stop_eligible"]),
                    convergence_gate_pass=bool(
                        window["early_stop_convergence_gate_pass"]
                    ),
                    consecutive_passes=consecutive_passes,
                    settings=early_stopping,
                )
                record["early_stop_consecutive_passes"] = (
                    consecutive_passes
                )
                previous_site_snapshot = current_site_snapshot
            history.append(record)

            print(
                json.dumps(
                    record
                )
            )

            if stop_reason is not None:
                break

    final_posterior = model.posterior(
        times=times,
        sites=sites,
    )

    return (
        sites,
        final_posterior,
        pd.DataFrame(history),
    )


def predict_frame(
    *,
    model: STSVGPModel,
    posterior: STPosterior,
    validation_data: TimeData,
    last_training_step: float,
    config: dict[str, Any],
    dtype: tf.dtypes.DType,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast one future origin without conditioning on its targets."""
    delta = (
        validation_data.time_step
        - float(last_training_step)
    )
    if delta <= 0.0:
        raise ValueError(
            "Validation time must be after training time."
        )

    inducing_mean, inducing_covariance = (
        model.forward_inducing_state(
            posterior=posterior,
            delta=delta,
        )
    )

    batch_size = int(
        config["training"][
            "prediction_batch_size"
        ]
    )

    probabilities = []
    latent_variances = []

    for start in range(
        0,
        len(validation_data.targets),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(validation_data.targets),
        )

        latent_mean, latent_variance = (
            model.latent_marginals(
                features=tf.constant(
                    validation_data.features[start:end],
                    dtype=dtype,
                ),
                temporal_trend_time=(
                    validation_data.temporal_trend_time
                ),
                coordinates_km=tf.constant(
                    validation_data.coordinates_km[start:end],
                    dtype=dtype,
                ),
                inducing_mean=inducing_mean,
                inducing_covariance=(
                    inducing_covariance
                ),
                spatial_covariance=(
                    posterior.spatial_covariance
                ),
            )
        )
        probability = probit_predictive_probability(
            marginal_mean=latent_mean,
            marginal_variance=latent_variance,
            dtype=dtype,
        )

        probabilities.append(
            probability.numpy()
        )
        latent_variances.append(
            latent_variance.numpy()
        )

    result = np.concatenate(
        probabilities
    )

    if (
        not np.isfinite(result).all()
        or (result < 0.0).any()
        or (result > 1.0).any()
    ):
        raise RuntimeError(
            "Invalid ST-SVGP probabilities."
        )

    variance_result = np.concatenate(
        latent_variances
    )
    if (
        not np.isfinite(variance_result).all()
        or (variance_result < 0.0).any()
    ):
        raise RuntimeError(
            "Invalid ST-SVGP latent variances."
        )

    return result, variance_result


def probabilistic_metrics(
    y: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(
        probability,
        1.0e-12,
        1.0 - 1.0e-12,
    )
    return {
        "log_loss": float(
            log_loss(y, clipped)
        ),
        "brier_score": float(
            brier_score_loss(
                y,
                probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probability,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probability,
            )
        ),
        "probability_bias": float(
            probability.mean()
            - y.mean()
        ),
    }


def run_fold(
    *,
    frame: pd.DataFrame,
    fold_number: int,
    fold: dict[str, Any],
    config: dict[str, Any],
    dtype: tf.dtypes.DType,
    fitted_state_callback: FoldStateCallback | None = None,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    dataset = config["dataset"]
    origin_column = str(
        dataset["forecast_origin"]
    )

    train_origins = [
        int(value)
        for value in fold[
            "train_origins"
        ]
    ]
    validation_origin = int(
        fold["validation_origin"]
    )

    train_frame = frame.loc[
        frame[origin_column]
        .astype(int)
        .isin(train_origins)
    ].copy()

    validation_frame = frame.loc[
        frame[origin_column]
        .astype(int)
        .eq(validation_origin)
    ].copy()

    preprocessing = fit_preprocessing(
        train_frame,
        config,
    )

    train_data = prepare_time_data(
        frame,
        origins=train_origins,
        preprocessing=preprocessing,
        config=config,
    )
    validation_data = prepare_time_data(
        frame,
        origins=[validation_origin],
        preprocessing=preprocessing,
        config=config,
    )[0]

    model = model_from_preprocessing(
        preprocessing,
        config,
        dtype=dtype,
    )

    sites, posterior, history = train_model(
        model=model,
        time_data=train_data,
        iterations=int(
            config["training"][
                "iterations"
            ]
        ),
        config=config,
        dtype=dtype,
        seed_offset=fold_number,
    )

    early_stopping = config.get("early_stopping", {})
    early_stopping_enabled = bool(
        early_stopping.get("enabled", False)
    )

    if fitted_state_callback is not None:
        fitted_state_callback(
            FittedFoldState(
                model=model,
                sites=sites,
                preprocessing=preprocessing,
                train_origins=tuple(train_origins),
                validation_origin=validation_origin,
            )
        )

    probability, latent_variance = predict_frame(
        model=model,
        posterior=posterior,
        validation_data=validation_data,
        last_training_step=(
            train_data[-1].time_step
        ),
        config=config,
        dtype=dtype,
    )

    y = (
        validation_data.targets
        .astype(int)
    )
    metrics = probabilistic_metrics(
        y,
        probability,
    )

    predictions = validation_frame[
        [
            str(dataset["cell_id"]),
            origin_column,
            str(dataset["target_year"]),
            str(dataset["target"]),
        ]
    ].copy()

    predictions[
        "probability_raw"
    ] = probability
    predictions["fold"] = fold_number

    reporting = config.get("reporting", {})
    annual_diagnostics = bool(
        reporting.get(
            "annual_temporal_diagnostics",
            False,
        )
    )
    if annual_diagnostics:
        predictions["latent_variance"] = (
            latent_variance
        )

    record: dict[str, object] = {
        "fold": fold_number,
        "train_origins": ",".join(
            map(str, train_origins)
        ),
        "validation_origin": (
            validation_origin
        ),
        "train_rows": int(
            len(train_frame)
        ),
        "validation_rows": int(
            len(validation_frame)
        ),
        "training_prevalence": float(
            train_frame[
                str(dataset["target"])
            ].mean()
        ),
        "observed_prevalence": float(
            y.mean()
        ),
        "mean_predicted_probability": float(
            probability.mean()
        ),
        **metrics,
        "spatial_lengthscale_x_km": float(
            model.spatial_lengthscales[
                0
            ].numpy()
        ),
        "spatial_lengthscale_y_km": float(
            model.spatial_lengthscales[
                1
            ].numpy()
        ),
        "temporal_lengthscale_steps": float(
            model.temporal_lengthscale
            .numpy()
        ),
        "kernel_variance": float(
            model.variance.numpy()
        ),
        "linear_intercept": float(
            model.beta0.numpy()
        ),
        "minimum_site_precision_eigenvalue": float(
            tf.reduce_min(
                tf.linalg.eigvalsh(
                    site_precision(sites)
                )
            ).numpy()
        ),
    }

    if early_stopping_enabled:
        final_history = history.iloc[-1]
        convergence_demonstrated = bool(
            final_history["early_stop_consecutive_passes"]
            >= int(early_stopping["patience_checkpoints"])
        )
        record.update(
            {
                "completed_iterations": int(
                    final_history["completed_iterations"]
                ),
                "stop_reason": (
                    "CONVERGENCE_RULE"
                    if convergence_demonstrated
                    else "MAX_ITERATIONS"
                ),
                "convergence_demonstrated": convergence_demonstrated,
            }
        )

    if annual_diagnostics:
        validation_target_years = (
            validation_frame[
                str(dataset["target_year"])
            ]
            .astype(int)
            .unique()
        )
        if len(validation_target_years) != 1:
            raise ValueError(
                "A validation origin must map to one target year."
            )

        calibration = calibration_metrics(
            y,
            probability,
            n_bins=int(
                reporting.get(
                    "calibration_bins",
                    10,
                )
            ),
            strategy=str(
                reporting.get(
                    "calibration_strategy",
                    "quantile",
                )
            ),
        )
        step_years = float(
            config["time"]["step_years"]
        )
        record.update(
            {
                "validation_target_year": int(
                    validation_target_years[0]
                ),
                "observed_positive_rate": float(
                    y.mean()
                ),
                "ece": calibration["ece"],
                "calibration_intercept": calibration[
                    "calibration_intercept"
                ],
                "calibration_slope": calibration[
                    "calibration_slope"
                ],
                "mean_latent_variance": float(
                    latent_variance.mean()
                ),
                "median_latent_variance": float(
                    np.median(latent_variance)
                ),
                "temporal_lengthscale_years": float(
                    model.temporal_lengthscale.numpy()
                    * step_years
                ),
                "natural_gradient_gamma": float(
                    history[
                        "natural_gradient_gamma"
                    ].iloc[-1]
                ),
                "minimum_natural_gradient_gamma": float(
                    history[
                        "natural_gradient_gamma"
                    ].min()
                ),
            }
        )

    history.insert(
        0,
        "fold",
        fold_number,
    )

    return (
        record,
        predictions,
        history,
    )


def save_final_state(
    *,
    model: STSVGPModel,
    sites: DenseCviSites,
    preprocessing: FoldPreprocessing,
    config: dict[str, Any],
) -> Path:
    """Persist enough state to reproduce the pre-test ST-SVGP fit."""
    model_directory = Path(
        config["outputs"][
            "model_directory"
        ]
    )
    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        model_directory
        / "st_svgp_state.npz"
    )

    np.savez_compressed(
        path,
        beta0=float(
            model.beta0.numpy()
        ),
        beta=model.beta.numpy(),
        spatial_lengthscales_km=(
            model.spatial_lengthscales
            .numpy()
        ),
        temporal_lengthscale_steps=float(
            model.temporal_lengthscale
            .numpy()
        ),
        kernel_variance=float(
            model.variance.numpy()
        ),
        inducing_locations_km=(
            model.inducing_locations_km
            .numpy()
        ),
        feature_mean=(
            preprocessing.feature_mean
        ),
        feature_scale=(
            preprocessing.feature_scale
        ),
        coordinate_center_km=(
            preprocessing
            .coordinate_center_km
        ),
        site_lambda1=(
            sites.lambda1.numpy()
        ),
        site_lambda2=(
            sites.lambda2.numpy()
        ),
    )

    return path


def run_final_fit(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    dtype: tf.dtypes.DType,
) -> tuple[Path, pd.DataFrame]:
    """Fit every pre-test origin without evaluating 2020."""
    dataset = config["dataset"]
    origin_column = str(
        dataset["forecast_origin"]
    )
    origins = [
        int(value)
        for value in config[
            "final_fit"
        ]["origins"]
    ]

    train_frame = frame.loc[
        frame[origin_column]
        .astype(int)
        .isin(origins)
    ].copy()

    preprocessing = fit_preprocessing(
        train_frame,
        config,
    )

    train_data = prepare_time_data(
        frame,
        origins=origins,
        preprocessing=preprocessing,
        config=config,
    )

    model = model_from_preprocessing(
        preprocessing,
        config,
        dtype=dtype,
    )

    sites, _, history = train_model(
        model=model,
        time_data=train_data,
        iterations=int(
            config["training"][
                "final_iterations"
            ]
        ),
        config=config,
        dtype=dtype,
        seed_offset=100,
    )

    path = save_final_state(
        model=model,
        sites=sites,
        preprocessing=preprocessing,
        config=config,
    )

    history.insert(
        0,
        "fold",
        "final_fit",
    )

    return path, history


def write_preflight(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, object]:
    output = config["outputs"]
    metadata_directory = Path(
        output["metadata_directory"]
    )
    metadata_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    locked_block = config.get(
        "locked_block",
        config.get("final_test", {}),
    )
    step_years = float(
        config["time"]["step_years"]
    )
    initial_lengthscale_steps = float(
        config["kernel"][
            "temporal_initial_lengthscale_steps"
        ]
    )

    payload = {
        "status": "PASS",
        "model": "st_svgp",
        "feature_set": config[
            "feature_set"
        ],
        "rows": int(len(frame)),
        "linear_predictors": list(
            config[
                "linear_predictors"
            ]
        ),
        "spatial_inducing_points": int(
            config["inducing"][
                "spatial_points"
            ]
        ),
        "inference": {
            "method": (
                "dense_cvi_natural_gradient"
            ),
            "filtering": "sequential",
            "smoothing": "rts",
            "natural_gradient_gamma": float(
                config["inference"][
                    "natural_gradient_gamma"
                ]
            ),
        },
        "kernel": {
            "temporal_initial_lengthscale_steps": (
                initial_lengthscale_steps
            ),
            "temporal_lengthscale_trainable": bool(
                config["kernel"].get(
                    "temporal_lengthscale_trainable",
                    True,
                )
            ),
        },
        "rolling_validation": (
            config["rolling_validation"][
                "folds"
            ]
        ),
        "final_fit_origins": (
            config["final_fit"][
                "origins"
            ]
        ),
        "final_test_locked_origins": (
            locked_block.get("origins", [])
        ),
        "final_test_evaluated": False,
        "runtime": {
            "device": config["compute"][
                "device"
            ],
            "float_type": config[
                "compute"
            ]["float_type"],
            "jitter": float(
                config["compute"][
                    "jitter"
                ]
            ),
        },
    }

    if "development" in config:
        payload["kernel"][
            "temporal_initial_lengthscale_years"
        ] = initial_lengthscale_steps * step_years
        payload["locked_target_years"] = (
            locked_block.get("target_years", [])
        )
        payload["maximum_development_target_year"] = (
            config["development"][
                "maximum_target_year"
            ]
        )

    (
        metadata_directory
        / "preflight.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return payload


def _sample_standard_deviation(
    values: pd.Series,
) -> float:
    return float(values.astype(float).std(ddof=1))


def _coefficient_of_variation(
    values: pd.Series,
) -> float:
    numeric = values.astype(float)
    mean = float(numeric.mean())
    if math.isclose(mean, 0.0):
        return float("nan")
    return _sample_standard_deviation(numeric) / abs(mean)


def write_annual_temporal_diagnostic(
    fold_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> Path:
    """Write the annual diagnostic from observed rolling-fold results."""
    reporting = config["reporting"]
    retained_metrics = pd.read_csv(
        Path(
            reporting[
                "retained_five_year_metrics_path"
            ]
        )
    )
    retained_predictions = pd.read_parquet(
        Path(
            reporting[
                "retained_five_year_predictions_path"
            ]
        )
    )

    retained_calibration_rows = []
    for fold, part in retained_predictions.groupby(
        "fold",
        sort=True,
    ):
        values = calibration_metrics(
            part["target_transition_5y"].to_numpy(
                dtype=int
            ),
            part["probability_raw"].to_numpy(
                dtype=float
            ),
            n_bins=int(reporting["calibration_bins"]),
            strategy=str(
                reporting["calibration_strategy"]
            ),
        )
        retained_calibration_rows.append(
            {"fold": int(fold), **values}
        )
    retained_calibration = pd.DataFrame(
        retained_calibration_rows
    )

    retained_step_years = float(
        reporting["retained_five_year_step_years"]
    )
    retained_lengthscale_years = (
        retained_metrics[
            "temporal_lengthscale_steps"
        ].astype(float)
        * retained_step_years
    )
    annual_lengthscale_years = fold_metrics[
        "temporal_lengthscale_years"
    ].astype(float)

    annual_median = float(
        annual_lengthscale_years.median()
    )
    retained_median = float(
        retained_lengthscale_years.median()
    )
    if math.isclose(
        annual_median,
        retained_median,
        rel_tol=0.05,
    ):
        qualitative_scale = "approximately the same"
    elif annual_median < retained_median:
        qualitative_scale = "shorter"
    else:
        qualitative_scale = "longer"

    annual_lengthscale_cv = _coefficient_of_variation(
        annual_lengthscale_years
    )
    retained_lengthscale_cv = _coefficient_of_variation(
        retained_lengthscale_years
    )
    annual_bias_sd = _sample_standard_deviation(
        fold_metrics["probability_bias"]
    )
    retained_bias_sd = _sample_standard_deviation(
        retained_metrics["probability_bias"]
    )
    annual_ece_sd = _sample_standard_deviation(
        fold_metrics["ece"]
    )
    retained_ece_sd = _sample_standard_deviation(
        retained_calibration["ece"]
    )
    annual_slope_sd = _sample_standard_deviation(
        fold_metrics["calibration_slope"]
    )
    retained_slope_sd = _sample_standard_deviation(
        retained_calibration["calibration_slope"]
    )

    if annual_lengthscale_cv < retained_lengthscale_cv:
        identifiability = (
            "The annual folds have lower relative temporal-lengthscale "
            "dispersion than the retained five-year folds, which is "
            "descriptive evidence of improved temporal identifiability."
        )
    else:
        identifiability = (
            "The annual folds do not have lower relative temporal-lengthscale "
            "dispersion than the retained five-year folds, so these results "
            "do not provide descriptive evidence of improved temporal "
            "identifiability."
        )

    uncertainty_rows = "; ".join(
        f"{int(row.validation_target_year)}: mean {row.mean_latent_variance:.6g}, "
        f"median {row.median_latent_variance:.6g}"
        for row in fold_metrics.itertuples(index=False)
    )
    lines = [
        "# Annual ST-SVGP temporal diagnostic",
        "",
        "Status: EXPERIMENTAL / PROVISIONAL. Annual manual mapping validation remains deferred.",
        "The locked 2020-2025 target block was not evaluated.",
        "",
        "1. **Temporal lengthscale stability.** "
        f"Annual physical lengthscales range from {annual_lengthscale_years.min():.6g} "
        f"to {annual_lengthscale_years.max():.6g} years (CV {annual_lengthscale_cv:.6g}).",
        "2. **Physical comparison.** "
        f"The annual median is {annual_median:.6g} years and the retained five-year "
        f"median is {retained_median:.6g} years; the annual scale is {qualitative_scale}.",
        "3. **Probability-bias stability.** "
        f"Fold SD is {annual_bias_sd:.6g} annually versus {retained_bias_sd:.6g} "
        "for the retained five-year folds.",
        "4. **Calibration stability.** "
        f"ECE fold SD is {annual_ece_sd:.6g} annually versus {retained_ece_sd:.6g} "
        f"retained; calibration-slope fold SD is {annual_slope_sd:.6g} annually "
        f"versus {retained_slope_sd:.6g} retained.",
        "5. **Latent uncertainty.** " + uncertainty_rows + ".",
        "6. **Temporal identifiability.** " + identifiability,
        "",
        "One-year and five-year Log Loss values describe different forecasting events "
        "and are not ranked directly here.",
    ]

    path = (
        Path(config["outputs"]["metrics_directory"])
        / str(
            reporting[
                "annual_temporal_diagnostic_filename"
            ]
        )
    )
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return path


def run(
    config_path: Path,
    *,
    preflight_only: bool,
    rolling_only: bool,
) -> dict[str, object]:
    config = load_config(
        config_path
    )
    dtype = configure_runtime(
        config
    )
    frame = load_dataset(
        config
    )

    preflight = write_preflight(
        frame,
        config,
    )

    if preflight_only:
        return preflight

    metrics_directory = Path(
        config["outputs"][
            "metrics_directory"
        ]
    )
    predictions_directory = Path(
        config["outputs"][
            "predictions_directory"
        ]
    )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    predictions_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_records = []
    prediction_tables = []
    history_tables = []

    for fold_number, fold in enumerate(
        config["rolling_validation"][
            "folds"
        ],
        start=1,
    ):
        record, predictions, history = (
            run_fold(
                frame=frame,
                fold_number=fold_number,
                fold=fold,
                config=config,
                dtype=dtype,
            )
        )

        fold_records.append(record)
        prediction_tables.append(
            predictions
        )
        history_tables.append(history)

    fold_metrics = pd.DataFrame(
        fold_records
    )
    oof_predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )
    history = pd.concat(
        history_tables,
        ignore_index=True,
    )

    reporting = config.get("reporting", {})
    fold_metrics_path = (
        metrics_directory
        / str(
            reporting.get(
                "rolling_metrics_filename",
                "st_svgp_rolling_validation_metrics.csv",
            )
        )
    )
    oof_path = (
        predictions_directory
        / "st_svgp_oof_predictions.parquet"
    )
    history_path = (
        metrics_directory
        / "st_svgp_training_history.csv"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )
    oof_predictions.to_parquet(
        oof_path,
        index=False,
    )
    history.to_csv(
        history_path,
        index=False,
    )

    temporal_summary_path: Path | None = None
    diagnostic_path: Path | None = None
    if bool(
        reporting.get(
            "annual_temporal_diagnostics",
            False,
        )
    ):
        temporal_columns = [
            "fold",
            "validation_origin",
            "validation_target_year",
            "temporal_lengthscale_steps",
            "temporal_lengthscale_years",
            "spatial_lengthscale_x_km",
            "spatial_lengthscale_y_km",
            "kernel_variance",
            "mean_latent_variance",
            "median_latent_variance",
            "natural_gradient_gamma",
            "minimum_natural_gradient_gamma",
            "minimum_site_precision_eigenvalue",
        ]
        temporal_summary_path = (
            metrics_directory
            / str(
                reporting[
                    "temporal_parameter_summary_filename"
                ]
            )
        )
        fold_metrics[temporal_columns].to_csv(
            temporal_summary_path,
            index=False,
        )
        diagnostic_path = write_annual_temporal_diagnostic(
            fold_metrics,
            config,
        )

    summary = {
        "folds": int(
            len(fold_metrics)
        ),
        "mean_log_loss": float(
            fold_metrics[
                "log_loss"
            ].mean()
        ),
        "std_log_loss": float(
            fold_metrics[
                "log_loss"
            ].std(ddof=1)
        ),
        "mean_brier_score": float(
            fold_metrics[
                "brier_score"
            ].mean()
        ),
        "mean_pr_auc": float(
            fold_metrics[
                "pr_auc"
            ].mean()
        ),
        "mean_roc_auc": float(
            fold_metrics[
                "roc_auc"
            ].mean()
        ),
        "mean_probability_bias": float(
            fold_metrics[
                "probability_bias"
            ].mean()
        ),
        "final_test_evaluated": False,
    }

    summary_path = (
        metrics_directory
        / "st_svgp_rolling_validation_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    final_state: str | None = None

    if not rolling_only:
        state_path, final_history = (
            run_final_fit(
                frame=frame,
                config=config,
                dtype=dtype,
            )
        )
        final_state = str(
            state_path
        )

        final_history.to_csv(
            metrics_directory
            / "st_svgp_final_training_history.csv",
            index=False,
        )

    return {
        "status": "PASS",
        "model": "st_svgp",
        "rolling_metrics": str(
            fold_metrics_path
        ),
        "rolling_summary": str(
            summary_path
        ),
        "oof_predictions": str(
            oof_path
        ),
        "training_history": str(
            history_path
        ),
        "temporal_parameter_summary": (
            str(temporal_summary_path)
            if temporal_summary_path is not None
            else None
        ),
        "annual_temporal_diagnostic": (
            str(diagnostic_path)
            if diagnostic_path is not None
            else None
        ),
        "final_state": final_state,
        "final_test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/modeling/"
            "st_svgp_experiment.yaml"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )
    parser.add_argument(
        "--rolling-only",
        action="store_true",
    )

    args = parser.parse_args()

    result = run(
        args.config,
        preflight_only=(
            args.preflight_only
        ),
        rolling_only=(
            args.rolling_only
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
