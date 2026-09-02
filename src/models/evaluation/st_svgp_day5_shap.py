"""Reload preflight and bounded SHAP sampling for retained ST-SVGP folds.

Context-conditioned covariate SHAP perturbs only the nine observed predictors.
Projected coordinates and validation time stay fixed at one real representative
context. These values measure model sensitivity under that context; they are not
causal effects and need not decompose a row's OOF prediction at its own location.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.models.evaluation.st_svgp_day5_diagnostics import (
    DEFAULT_CONFIG_PATH,
    HorizonSpec,
    load_yaml,
    prepare_horizon_diagnostics,
    resolve_horizons,
)

STATE_FILENAME = "st_svgp_state.npz"
PREPROCESSING_KEYS = {"feature_mean", "feature_scale", "coordinate_center_km"}
INDUCING_KEYS = {"inducing_locations_km"}
KERNEL_KEYS = {
    "beta0",
    "beta",
    "spatial_lengthscales_km",
    "temporal_lengthscale_steps",
    "kernel_variance",
}
VARIATIONAL_KEYS = {"site_lambda1", "site_lambda2"}
ALL_STATE_KEYS = PREPROCESSING_KEYS | INDUCING_KEYS | KERNEL_KEYS | VARIATIONAL_KEYS
PRIMARY_CONTEXTS = ("near_built", "peripheral")
PRIMARY_ERROR_LEVELS = ("low_error", "high_error")


@dataclass
class ContextConditionedCovariatePredictor:
    """Call the retained prediction path with fixed projected space and time."""

    model: Any
    posterior: Any
    preprocessing: Any
    config: dict[str, Any]
    dtype: Any
    last_training_step: float
    context_x_m: float
    context_y_m: float
    context_origin: int

    def __call__(self, observed_features: np.ndarray) -> np.ndarray:
        """Predict probabilities while varying observed covariates only."""
        from src.models.train_st_svgp import TimeData, predict_frame, time_step

        features = np.asarray(observed_features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        expected_features = len(self.config["linear_predictors"])
        if features.ndim != 2 or features.shape[1] != expected_features:
            raise ValueError(f"Expected a matrix with {expected_features} observed features.")
        scaled = (features - self.preprocessing.feature_mean) / self.preprocessing.feature_scale
        fixed_coordinate = (
            np.array([self.context_x_m, self.context_y_m], dtype=np.float64) / 1000.0
            - self.preprocessing.coordinate_center_km
        )
        validation_data = TimeData(
            origin=int(self.context_origin),
            time_step=time_step(int(self.context_origin), self.config),
            features=scaled,
            coordinates_km=np.repeat(fixed_coordinate[None, :], len(features), axis=0),
            targets=np.zeros(len(features), dtype=np.float64),
            row_index=np.arange(len(features)),
        )
        probabilities, _ = predict_frame(
            model=self.model,
            posterior=self.posterior,
            validation_data=validation_data,
            last_training_step=self.last_training_step,
            config=self.config,
            dtype=self.dtype,
        )
        return probabilities


def expected_fold_state_path(model_directory: Path, fold: int) -> Path:
    """Return the unambiguous rolling-fold state location required by Day 5."""
    return model_directory / f"fold_{int(fold)}" / STATE_FILENAME


def _state_component_status(path: Path) -> tuple[dict[str, Any], set[str]]:
    status: dict[str, Any] = {
        "state_path": str(path),
        "model_state_found": path.is_file(),
        "preprocessing_state_found": False,
        "inducing_locations_found": False,
        "kernel_parameters_found": False,
        "variational_cvi_state_found": False,
    }
    if not path.is_file():
        return status, set()
    with np.load(path) as state:
        keys = set(state.files)
    status.update(
        {
            "preprocessing_state_found": PREPROCESSING_KEYS.issubset(keys),
            "inducing_locations_found": INDUCING_KEYS.issubset(keys),
            "kernel_parameters_found": KERNEL_KEYS.issubset(keys),
            "variational_cvi_state_found": VARIATIONAL_KEYS.issubset(keys),
        }
    )
    return status, keys


def _reload_and_reproduce(
    *,
    state_path: Path,
    fold: int,
    frame: pd.DataFrame,
    model_config: dict[str, Any],
    random_state: int,
    rtol: float,
    atol: float,
) -> tuple[bool, bool, str | None]:
    """Reload one real fold state and compare a deterministic OOF sample."""
    import tensorflow as tf
    from src.models.st_svgp.block_inference import DenseCviSites
    from src.models.train_st_svgp import (
        FoldPreprocessing,
        configure_runtime,
        model_from_preprocessing,
        predict_frame,
        prepare_time_data,
        time_step,
    )

    try:
        with np.load(state_path) as state:
            preprocessing = FoldPreprocessing(
                feature_mean=state["feature_mean"],
                feature_scale=state["feature_scale"],
                coordinate_center_km=state["coordinate_center_km"],
                inducing_locations_km=state["inducing_locations_km"],
            )
            dtype = configure_runtime(model_config)
            model = model_from_preprocessing(preprocessing, model_config, dtype=dtype)
            model.beta0.assign(float(state["beta0"]))
            model.beta.assign(state["beta"])
            model.log_spatial_lengthscales.assign(
                np.log(state["spatial_lengthscales_km"])
            )
            model.log_temporal_lengthscale.assign(
                np.log(float(state["temporal_lengthscale_steps"]))
            )
            model.log_variance.assign(np.log(float(state["kernel_variance"])))
            sites = DenseCviSites(
                lambda1=tf.constant(state["site_lambda1"], dtype=dtype),
                lambda2=tf.constant(state["site_lambda2"], dtype=dtype),
            )

        fold_config = model_config["rolling_validation"]["folds"][fold - 1]
        train_origins = [int(value) for value in fold_config["train_origins"]]
        times = tf.constant(
            [time_step(origin, model_config) for origin in train_origins], dtype=dtype
        )
        posterior = model.posterior(times=times, sites=sites)
        sample = (
            frame.loc[frame["fold"].eq(fold)]
            .sort_values(["cell_id", "forecast_origin"])
            .sample(n=min(10, int(frame["fold"].eq(fold).sum())), random_state=random_state)
            .copy()
        )
        target = str(model_config["dataset"]["target"])
        sample[target] = sample["y_true"]
        validation_data = prepare_time_data(
            sample,
            origins=[int(fold_config["validation_origin"])],
            preprocessing=preprocessing,
            config=model_config,
        )[0]
        probability, _ = predict_frame(
            model=model,
            posterior=posterior,
            validation_data=validation_data,
            last_training_step=time_step(train_origins[-1], model_config),
            config=model_config,
            dtype=dtype,
        )
        assert_oof_reproduction(
            probability,
            sample["probability_raw"].to_numpy(dtype=float),
            rtol=rtol,
            atol=atol,
        )
        return True, True, None
    except Exception as error:  # Preflight records exact local reconstruction failures.
        return False, False, f"{type(error).__name__}: {error}"


def assert_oof_reproduction(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Fail loudly when a reloaded fold materially differs from retained OOF."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def inspect_horizon_states(
    spec: HorizonSpec, day5_config: dict[str, Any]
) -> dict[str, Any]:
    """Inspect every fold and require exact OOF reproduction before passing."""
    model_config = load_yaml(spec.model_config_path)
    frame, _, _, _ = prepare_horizon_diagnostics(spec, day5_config)
    model_directory = Path(model_config["outputs"]["model_directory"])
    shap_config = day5_config["shap"]
    fold_results: list[dict[str, Any]] = []
    for fold in range(1, len(model_config["rolling_validation"]["folds"]) + 1):
        state_path = expected_fold_state_path(model_directory, fold)
        component_status, keys = _state_component_status(state_path)
        result = {
            "fold": fold,
            **component_status,
            "prediction_callable_reconstructed": False,
            "oof_reproduction_check_passed": False,
            "error": None,
        }
        if ALL_STATE_KEYS.issubset(keys):
            callable_ready, reproduced, error = _reload_and_reproduce(
                state_path=state_path,
                fold=fold,
                frame=frame,
                model_config=model_config,
                random_state=int(day5_config["random_state"]) + fold,
                rtol=float(shap_config["reproduction_rtol"]),
                atol=float(shap_config["reproduction_atol"]),
            )
            result.update(
                {
                    "prediction_callable_reconstructed": callable_ready,
                    "oof_reproduction_check_passed": reproduced,
                    "error": error,
                }
            )
        else:
            missing = sorted(ALL_STATE_KEYS.difference(keys))
            result["error"] = (
                "Rolling fold state not found."
                if not state_path.is_file()
                else f"Rolling fold state is missing keys: {', '.join(missing)}"
            )
        fold_results.append(result)

    ready = all(row["oof_reproduction_check_passed"] for row in fold_results)
    return {
        "status": "PASS" if ready else "FAIL",
        "model_directory": str(model_directory),
        "folds": fold_results,
    }


def select_representative_contexts(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Choose one real nearest-to-centroid observation per fold/context."""
    rows: list[dict[str, Any]] = []
    eligible = frame.loc[frame["distance_context"].isin(PRIMARY_CONTEXTS)]
    for (fold, context), part in eligible.groupby(
        ["fold", "distance_context"], observed=True, sort=True
    ):
        coordinates = part[["x_center_m", "y_center_m"]].to_numpy(dtype=float)
        center = coordinates.mean(axis=0)
        distance = np.square(coordinates - center).sum(axis=1)
        selected = part.iloc[int(np.argmin(distance))]
        rows.append(
            {
                "horizon": horizon,
                "fold": int(fold),
                "target_period": f"{int(selected['forecast_origin'])}->{int(selected['target_year'])}",
                "context_id": f"{horizon}_fold_{int(fold)}_{context}",
                "context_type": str(context),
                "cell_id": int(selected["cell_id"]),
                "x": float(selected["x_center_m"]),
                "y": float(selected["y_center_m"]),
                "time": int(selected["forecast_origin"]),
                "selection_rule": "real observation nearest projected-coordinate centroid",
            }
        )
    return pd.DataFrame(rows)


def exclude_locked_rows(frame: pd.DataFrame, spec: HorizonSpec) -> pd.DataFrame:
    """Return only rows eligible for pre-test SHAP design."""
    eligible = frame.copy()
    if spec.maximum_target_year is not None:
        eligible = eligible.loc[eligible["target_year"].le(spec.maximum_target_year)]
    if spec.locked_forecast_origins:
        eligible = eligible.loc[
            ~eligible["forecast_origin"].isin(spec.locked_forecast_origins)
        ]
    return eligible.copy()


def select_background_rows(
    frame: pd.DataFrame,
    features: list[str],
    *,
    spec: HorizonSpec,
    size: int,
    random_state: int,
) -> pd.DataFrame:
    """Select deterministic nearest real rows to standardized KMeans centers."""
    if size < 1 or size > 50:
        raise ValueError("SHAP background size must be between 1 and 50.")
    ordered = exclude_locked_rows(frame, spec).sort_values(
        ["cell_id", "forecast_origin"]
    ).reset_index(drop=True)
    if ordered.empty:
        raise ValueError("No pre-test rows are eligible for the SHAP background.")
    values = ordered[features].to_numpy(dtype=float)
    scale = values.std(axis=0, ddof=0)
    scale[scale <= 0.0] = 1.0
    standardized = (values - values.mean(axis=0)) / scale
    clusters = min(size, len(ordered))
    centers = KMeans(
        n_clusters=clusters, n_init=10, random_state=random_state
    ).fit(standardized).cluster_centers_
    selected: list[int] = []
    used: set[int] = set()
    for center in centers:
        distances = np.square(standardized - center).sum(axis=1)
        for index in np.argsort(distances, kind="mergesort"):
            candidate = int(index)
            if candidate not in used:
                used.add(candidate)
                selected.append(candidate)
                break
    return ordered.iloc[selected].copy().reset_index(drop=True)


def select_explained_rows(
    frame: pd.DataFrame,
    *,
    spec: HorizonSpec,
    maximum_rows: int,
    rows_per_stratum: int,
    random_state: int,
) -> pd.DataFrame:
    """Select bounded real rows across fold/class/context/error strata."""
    if maximum_rows > 120:
        raise ValueError("Explained SHAP rows cannot exceed 120 before authorization.")
    pretest = exclude_locked_rows(frame, spec)
    eligible = pretest.loc[
        pretest["distance_context"].isin(PRIMARY_CONTEXTS)
        & pretest["error_level"].isin(PRIMARY_ERROR_LEVELS)
    ].copy()
    if eligible.empty:
        raise ValueError("No pre-test rows are eligible for the explained SHAP sample.")
    eligible = eligible.sort_values(["fold", "y_true", "cell_id", "forecast_origin"])
    strata = ["fold", "y_true", "distance_context", "error_level"]
    selections: list[pd.DataFrame] = []
    for group_number, (_, part) in enumerate(
        eligible.groupby(strata, observed=True, sort=True), start=1
    ):
        selections.append(
            part.sample(
                n=min(rows_per_stratum, len(part)),
                random_state=random_state + group_number,
            )
        )
    selected = pd.concat(selections).drop_duplicates(["cell_id", "forecast_origin"])
    selected = selected.iloc[:maximum_rows]
    remaining_slots = maximum_rows - len(selected)
    if remaining_slots > 0:
        selected_keys = pd.MultiIndex.from_frame(selected[["cell_id", "forecast_origin"]])
        eligible_keys = pd.MultiIndex.from_frame(eligible[["cell_id", "forecast_origin"]])
        remaining = eligible.loc[~eligible_keys.isin(selected_keys)]
        if not remaining.empty:
            extra = remaining.sample(
                n=min(remaining_slots, len(remaining)), random_state=random_state
            )
            selected = pd.concat([selected, extra], ignore_index=True)
    return selected.reset_index(drop=True)


def run_preflight(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Write explicit per-horizon/per-fold state and reproduction status."""
    day5_config = load_yaml(config_path)
    output_root = Path(day5_config["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    horizons = {
        spec.name: inspect_horizon_states(spec, day5_config)
        for spec in resolve_horizons(day5_config)
    }
    ready = all(values["status"] == "PASS" for values in horizons.values())
    result = {
        "status": "PASS" if ready else "FAIL",
        "shap_dependency_declared": True,
        "shap_dependency_installed": importlib.util.find_spec("shap") is not None,
        "horizons": horizons,
        "reconstruction_required": not ready,
        "reconstruction_path": (
            "Rerun each unchanged rolling fold with the retained config and seed while "
            "persisting model parameters, fold preprocessing, inducing locations, and "
            "dense CVI sites immediately after training. No checkpoint/resume or surrogate."
            if not ready
            else None
        ),
    }
    path = output_root / "model_reload_preflight.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path = output_root / "day5_model_diagnostics.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        pending = (
            "- Context-conditioned covariate SHAP is pending the rolling-state reload "
            "preflight."
        )
        readiness = (
            "- Context-conditioned covariate SHAP is technically ready: all fold states "
            "reload and reproduce retained OOF probabilities."
            if ready
            else "- Context-conditioned covariate SHAP is not technically ready: real "
            "rolling-fold states are unavailable or failed OOF reproduction."
        )
        report_path.write_text(report.replace(pending, readiness), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def run_pilot(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Run no SHAP work unless every retained rolling fold reproduces its OOF rows."""
    result = run_preflight(config_path)
    if result["status"] != "PASS":
        raise RuntimeError(
            "SHAP pilot blocked: real rolling fold states are unavailable or do not "
            "reproduce retained OOF probabilities."
        )
    if not result["shap_dependency_installed"]:
        raise RuntimeError("SHAP pilot blocked: install the declared project dependency.")
    raise RuntimeError(
        "SHAP pilot implementation remains authorization-blocked until real fold states exist."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.pilot:
        run_pilot(arguments.config)
    else:
        run_preflight(arguments.config)