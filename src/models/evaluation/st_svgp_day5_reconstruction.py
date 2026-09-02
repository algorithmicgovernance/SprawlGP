"""Strict deterministic reconstruction gate for retained ST-SVGP OOF folds.

The preflight path reads frozen inputs only. Gate A retrains fold 1 for each
horizon from scratch, persists the fitted state outside canonical model trees,
reloads it, and compares keyed validation probabilities with retained OOF data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from src.models.evaluation.st_svgp_day5_diagnostics import (
    DEFAULT_CONFIG_PATH,
    JOIN_KEYS,
    HorizonSpec,
    resolve_horizons,
)
from src.models.st_svgp.block_inference import DenseCviSites
from src.models.train_st_svgp import (
    FittedFoldState,
    FoldPreprocessing,
    STSVGPModel,
    configure_runtime,
    load_config,
    load_dataset,
    model_from_preprocessing,
    predict_frame,
    prepare_time_data,
    run_fold,
    time_step,
)

STATE_ROOT = Path("artifacts/models/day5_reconstructed")
REPORT_PATH = Path("reports/modeling/day5_diagnostics/reconstruction_gate_a.json")
ALL_FOLDS_REPORT_PATH = Path(
    "reports/modeling/day5_diagnostics/reconstruction_all_folds.json"
)
STATE_FILENAME = "st_svgp_state.npz"
OOF_FILENAME = "st_svgp_oof_predictions.parquet"
METADATA_FILENAME = "metadata.json"
SCHEMA_VERSION = 1
GATE_FOLD = 1
ROLLING_FOLDS = (1, 2, 3)
EXPECTED_HORIZONS = ("annual_1y", "five_year_5y")
EXPECTED_CONFIG_PATHS = {
    "annual_1y": Path("configs/modeling/st_svgp_annual/convergence_3000.yaml"),
    "five_year_5y": Path("configs/modeling/st_svgp.yaml"),
}
EXPECTED_OOF_PATHS = {
    "annual_1y": Path(
        "reports/modeling/st_svgp_annual/experiments/convergence_3000/"
        "predictions/st_svgp_oof_predictions.parquet"
    ),
    "five_year_5y": Path(
        "reports/modeling/st_svgp/predictions/st_svgp_oof_predictions.parquet"
    ),
}
EXPECTED_TARGETS = {
    "annual_1y": "target_transition_1y",
    "five_year_5y": "target_transition_5y",
}
REQUIRED_ARRAY_KEYS = {
    "beta0",
    "beta",
    "log_spatial_lengthscales",
    "log_temporal_lengthscale",
    "log_variance",
    "spatial_lengthscales_km",
    "temporal_lengthscale_steps",
    "kernel_variance",
    "inducing_locations_km",
    "feature_mean",
    "feature_scale",
    "coordinate_center_km",
    "site_lambda1",
    "site_lambda2",
}


@dataclass(frozen=True)
class LoadedFoldState:
    """A prediction-ready fold rebuilt solely from persisted files."""

    model: STSVGPModel
    sites: DenseCviSites
    preprocessing: FoldPreprocessing
    posterior: Any
    config: dict[str, Any]
    dtype: tf.dtypes.DType
    last_training_step: float
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest for one immutable input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata() -> dict[str, Any]:
    """Capture the environment relevant to deterministic reconstruction."""
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "tensorflow", "tensorflow-macos"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def _oof_path(config: dict[str, Any]) -> Path:
    return Path(config["outputs"]["predictions_directory"]) / OOF_FILENAME


def _assert_expected_path(actual: Path, expected: Path, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected}; got {actual}.")


def state_directory(
    horizon: str,
    fold: int,
    *,
    root: Path = STATE_ROOT,
) -> Path:
    """Resolve an isolated state directory for one historical rolling fold."""
    if horizon not in EXPECTED_HORIZONS:
        raise ValueError(f"Unexpected reconstruction horizon: {horizon}.")
    if int(fold) not in ROLLING_FOLDS:
        raise ValueError(f"Reconstruction fold must be one of {ROLLING_FOLDS}.")
    return root / horizon / f"fold_{int(fold)}"


def gate_a_fold(config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Select exactly Fold 1; later folds require a separate authorization path."""
    folds = config["rolling_validation"]["folds"]
    if len(folds) != len(ROLLING_FOLDS):
        raise ValueError(f"Rolling validation must contain folds {ROLLING_FOLDS}.")
    return GATE_FOLD, folds[GATE_FOLD - 1]


def rolling_fold(config: dict[str, Any], fold_number: int) -> dict[str, Any]:
    """Resolve one unchanged rolling split by its one-based fold number."""
    folds = config["rolling_validation"]["folds"]
    if len(folds) != len(ROLLING_FOLDS):
        raise ValueError(f"Rolling validation must contain folds {ROLLING_FOLDS}.")
    if int(fold_number) not in ROLLING_FOLDS:
        raise ValueError(f"Reconstruction fold must be one of {ROLLING_FOLDS}.")
    return folds[int(fold_number) - 1]


def _validate_locked_rows(
    frame: pd.DataFrame,
    spec: HorizonSpec,
    *,
    label: str,
) -> None:
    """Reject any locked row before training or comparison."""
    target_year = pd.to_numeric(frame["target_year"], errors="raise").astype(int)
    origin = pd.to_numeric(frame["forecast_origin"], errors="raise").astype(int)
    if spec.maximum_target_year is not None and target_year.gt(
        spec.maximum_target_year
    ).any():
        raise ValueError(
            f"{label} contains target_year after {spec.maximum_target_year}."
        )
    if spec.locked_forecast_origins and origin.isin(spec.locked_forecast_origins).any():
        raise ValueError(f"{label} contains locked forecast origin 2020.")


def _validate_oof_contract(
    frame: pd.DataFrame,
    spec: HorizonSpec,
    config: dict[str, Any],
) -> None:
    dataset = config["dataset"]
    required = {
        *JOIN_KEYS,
        "target_year",
        "fold",
        "probability_raw",
        str(dataset["target"]),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Retained OOF predictions are missing: {', '.join(missing)}")
    if frame.duplicated(list(JOIN_KEYS), keep=False).any():
        raise ValueError("Retained OOF predictions contain duplicate stable keys.")
    _validate_locked_rows(frame, spec, label=f"{spec.name} retained OOF")

    expected = {
        number: int(fold["validation_origin"])
        for number, fold in enumerate(config["rolling_validation"]["folds"], start=1)
    }
    actual_folds = set(pd.to_numeric(frame["fold"], errors="raise").astype(int))
    if actual_folds != set(expected):
        raise ValueError(f"Retained OOF folds must be exactly {sorted(expected)}.")
    for fold_number, validation_origin in expected.items():
        part = frame.loc[pd.to_numeric(frame["fold"], errors="raise").eq(fold_number)]
        origins = set(pd.to_numeric(part["forecast_origin"], errors="raise").astype(int))
        if origins != {validation_origin}:
            raise ValueError(
                f"Fold {fold_number} must contain validation origin {validation_origin}."
            )


def _resolved_specs(day5_config: dict[str, Any]) -> list[HorizonSpec]:
    specs = resolve_horizons(day5_config)
    if tuple(spec.name for spec in specs) != EXPECTED_HORIZONS:
        raise ValueError(f"Reconstruction horizons must be exactly {EXPECTED_HORIZONS}.")
    return specs


def run_preflight(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Validate frozen Gate-A inputs without training or creating artifacts."""
    day5_config = _load_yaml(config_path)
    horizon_rows = []
    for spec in _resolved_specs(day5_config):
        expected_config_path = EXPECTED_CONFIG_PATHS[spec.name]
        _assert_expected_path(spec.model_config_path, expected_config_path, "Source config")
        config = load_config(spec.model_config_path)
        target = str(config["dataset"]["target"])
        if target != EXPECTED_TARGETS[spec.name]:
            raise ValueError(f"{spec.name} target must be {EXPECTED_TARGETS[spec.name]}.")

        iterations = int(config["training"]["iterations"])
        if spec.name == "annual_1y" and iterations != 3000:
            raise ValueError("Annual Gate A requires exactly 3000 iterations.")
        if spec.name == "five_year_5y" and iterations != int(
            _load_yaml(EXPECTED_CONFIG_PATHS[spec.name])["training"]["iterations"]
        ):
            raise ValueError("Five-year iterations must come unchanged from its source config.")

        oof_path = _oof_path(config)
        _assert_expected_path(oof_path, EXPECTED_OOF_PATHS[spec.name], "Retained OOF")
        oof = pd.read_parquet(oof_path)
        _validate_oof_contract(oof, spec, config)

        fold_number, fold = gate_a_fold(config)
        validation_origin = int(fold["validation_origin"])
        if spec.name == "annual_1y" and validation_origin + int(
            config["time"]["step_years"]
        ) >= 2020:
            raise ValueError("Annual Fold 1 must remain strictly before target year 2020.")
        if spec.name == "five_year_5y" and validation_origin == 2020:
            raise ValueError("Five-year Fold 1 cannot use forecast origin 2020.")

        horizon_rows.append(
            {
                "horizon": spec.name,
                "fold": fold_number,
                "source_config_path": str(spec.model_config_path),
                "source_config_sha256": sha256_file(spec.model_config_path),
                "retained_oof_path": str(oof_path),
                "retained_oof_sha256": sha256_file(oof_path),
                "dataset_path": str(config["dataset"]["path"]),
                "target": target,
                "iterations": iterations,
                "seed": int(config["training"]["random_state"]),
                "validation_origin": validation_origin,
                "persisted_state_path": str(
                    state_directory(spec.name, fold_number) / STATE_FILENAME
                ),
                "locked_rows_used": 0,
            }
        )

    return {
        "status": "PASS",
        "training_performed": False,
        "gate_folds": horizon_rows,
        "runtime": runtime_metadata(),
        "locked_rows_used": 0,
    }


def _state_metadata(
    *,
    fitted: FittedFoldState,
    horizon: str,
    fold: int,
    config_path: Path,
    config: dict[str, Any],
    retained_oof_path: Path,
) -> dict[str, Any]:
    dataset = config["dataset"]
    return {
        "schema_version": SCHEMA_VERSION,
        "horizon": horizon,
        "fold": int(fold),
        "source_config_path": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "retained_oof_path": str(retained_oof_path),
        "retained_oof_sha256": sha256_file(retained_oof_path),
        "dataset": {
            "path": str(dataset["path"]),
            "identifier": Path(dataset["path"]).parent.name,
            "target": str(dataset["target"]),
            "cell_id": str(dataset["cell_id"]),
            "forecast_origin": str(dataset["forecast_origin"]),
            "target_year": str(dataset["target_year"]),
            "x_coordinate": str(dataset["x_coordinate"]),
            "y_coordinate": str(dataset["y_coordinate"]),
        },
        "random_seed": {
            "base": int(config["training"]["random_state"]),
            "fold_offset": int(fold),
            "effective": int(config["training"]["random_state"]) + int(fold),
        },
        "feature_names": [str(value) for value in config["linear_predictors"]],
        "train_origins": list(fitted.train_origins),
        "validation_origin": int(fitted.validation_origin),
        "iterations": int(config["training"]["iterations"]),
        "temporal_normalization": {
            "origin_year": int(config["time"]["origin_year"]),
            "step_years": int(config["time"]["step_years"]),
            "training_time_steps": [
                time_step(origin, config) for origin in fitted.train_origins
            ],
        },
        "kernel": {
            "spatial_family": str(config["kernel"]["spatial_family"]),
            "temporal_family": str(config["kernel"]["temporal_family"]),
            "temporal_lengthscale_trainable": bool(
                config["kernel"].get("temporal_lengthscale_trainable", True)
            ),
        },
        "inference": {
            "method": str(config["inference"]["method"]),
            "filtering": str(config["inference"]["filtering"]),
            "smoothing": str(config["inference"]["smoothing"]),
            "quadrature_degree": int(config["inference"]["quadrature_degree"]),
            "likelihood": "bernoulli_probit",
            "variational_state": "dense_cvi_natural_parameters",
        },
        "numerics": {
            "dtype": str(config["compute"]["float_type"]),
            "device": str(config["compute"]["device"]),
            "jitter": float(config["compute"]["jitter"]),
        },
        "array_keys": sorted(REQUIRED_ARRAY_KEYS),
        "runtime": runtime_metadata(),
        "locked_rows_used": 0,
    }


def persist_fold_state(
    fitted: FittedFoldState,
    *,
    horizon: str,
    fold: int,
    config_path: Path,
    config: dict[str, Any],
    retained_oof_path: Path,
    root: Path = STATE_ROOT,
) -> Path:
    """Persist all numeric state needed by the existing prediction callable."""
    directory = state_directory(horizon, fold, root=root)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / STATE_FILENAME
    model = fitted.model
    np.savez_compressed(
        state_path,
        beta0=np.asarray(model.beta0.numpy()),
        beta=np.asarray(model.beta.numpy()),
        log_spatial_lengthscales=np.asarray(model.log_spatial_lengthscales.numpy()),
        log_temporal_lengthscale=np.asarray(model.log_temporal_lengthscale.numpy()),
        log_variance=np.asarray(model.log_variance.numpy()),
        spatial_lengthscales_km=np.asarray(model.spatial_lengthscales.numpy()),
        temporal_lengthscale_steps=np.asarray(model.temporal_lengthscale.numpy()),
        kernel_variance=np.asarray(model.variance.numpy()),
        inducing_locations_km=np.asarray(fitted.preprocessing.inducing_locations_km),
        feature_mean=np.asarray(fitted.preprocessing.feature_mean),
        feature_scale=np.asarray(fitted.preprocessing.feature_scale),
        coordinate_center_km=np.asarray(fitted.preprocessing.coordinate_center_km),
        site_lambda1=np.asarray(fitted.sites.lambda1.numpy()),
        site_lambda2=np.asarray(fitted.sites.lambda2.numpy()),
    )
    metadata = _state_metadata(
        fitted=fitted,
        horizon=horizon,
        fold=fold,
        config_path=config_path,
        config=config,
        retained_oof_path=retained_oof_path,
    )
    metadata["state_path"] = str(state_path)
    metadata["state_sha256"] = sha256_file(state_path)
    (directory / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return state_path


def load_fold_state(
    directory: Path,
    *,
    expected_config_path: Path,
) -> LoadedFoldState:
    """Rebuild a prediction-ready fold without any original in-memory object."""
    metadata_path = directory / METADATA_FILENAME
    state_path = directory / STATE_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if Path(metadata["source_config_path"]) != expected_config_path:
        raise ValueError("Persisted source config path does not match the requested config.")
    if sha256_file(expected_config_path) != str(metadata["source_config_sha256"]):
        raise ValueError("Persisted source config hash no longer matches the source config.")
    if sha256_file(state_path) != str(metadata["state_sha256"]):
        raise ValueError("Persisted state hash does not match metadata.")

    config = load_config(expected_config_path)
    dtype = configure_runtime(config)
    with np.load(state_path, allow_pickle=False) as state:
        missing = sorted(REQUIRED_ARRAY_KEYS.difference(state.files))
        if missing:
            raise ValueError(f"Persisted state is missing arrays: {', '.join(missing)}")
        preprocessing = FoldPreprocessing(
            feature_mean=np.asarray(state["feature_mean"]),
            feature_scale=np.asarray(state["feature_scale"]),
            coordinate_center_km=np.asarray(state["coordinate_center_km"]),
            inducing_locations_km=np.asarray(state["inducing_locations_km"]),
        )
        model = model_from_preprocessing(preprocessing, config, dtype=dtype)
        model.beta0.assign(state["beta0"])
        model.beta.assign(state["beta"])
        model.log_spatial_lengthscales.assign(state["log_spatial_lengthscales"])
        model.log_temporal_lengthscale.assign(state["log_temporal_lengthscale"])
        model.log_variance.assign(state["log_variance"])
        sites = DenseCviSites(
            lambda1=tf.constant(state["site_lambda1"], dtype=dtype),
            lambda2=tf.constant(state["site_lambda2"], dtype=dtype),
        )

    train_origins = tuple(int(value) for value in metadata["train_origins"])
    times = tf.constant([time_step(origin, config) for origin in train_origins], dtype=dtype)
    posterior = model.posterior(times=times, sites=sites)
    return LoadedFoldState(
        model=model,
        sites=sites,
        preprocessing=preprocessing,
        posterior=posterior,
        config=config,
        dtype=dtype,
        last_training_step=time_step(train_origins[-1], config),
        metadata=metadata,
    )


def compare_oof_probabilities(
    retained: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    target_column: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare probabilities after a one-to-one stable-key join."""
    required_retained = {*JOIN_KEYS, "target_year", target_column, "probability_raw"}
    required_reconstructed = {
        *JOIN_KEYS,
        "target_year",
        target_column,
        "probability_reconstructed",
    }
    for frame, required, label in (
        (retained, required_retained, "retained"),
        (reconstructed, required_reconstructed, "reconstructed"),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} predictions are missing: {', '.join(missing)}")
        if frame.duplicated(list(JOIN_KEYS), keep=False).any():
            raise ValueError(f"{label} predictions contain duplicate stable keys.")

    joined = retained.loc[:, sorted(required_retained)].merge(
        reconstructed.loc[:, sorted(required_reconstructed)],
        on=list(JOIN_KEYS),
        how="outer",
        validate="one_to_one",
        suffixes=("_original", "_reconstructed"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Retained and reconstructed OOF row sets differ.")
    for column in ("target_year", target_column):
        if not joined[f"{column}_original"].eq(
            joined[f"{column}_reconstructed"]
        ).all():
            raise ValueError(f"Retained and reconstructed {column} values differ.")

    y_true = joined[f"{target_column}_original"].to_numpy(dtype=int)
    original = joined["probability_raw"].to_numpy(dtype=float)
    rebuilt = joined["probability_reconstructed"].to_numpy(dtype=float)
    difference = rebuilt - original
    absolute = np.abs(difference)
    clipped_original = np.clip(original, 1.0e-12, 1.0 - 1.0e-12)
    clipped_rebuilt = np.clip(rebuilt, 1.0e-12, 1.0 - 1.0e-12)
    passed = bool(np.all(np.isclose(rebuilt, original, rtol=rtol, atol=atol)))
    return {
        "rows": int(len(joined)),
        "max_absolute_probability_difference": float(absolute.max(initial=0.0)),
        "mean_absolute_probability_difference": float(absolute.mean()),
        "RMSE_probability_difference": float(np.sqrt(np.mean(np.square(difference)))),
        "PR_AUC_original": float(average_precision_score(y_true, original)),
        "PR_AUC_reconstructed": float(average_precision_score(y_true, rebuilt)),
        "ROC_AUC_original": float(roc_auc_score(y_true, original)),
        "ROC_AUC_reconstructed": float(roc_auc_score(y_true, rebuilt)),
        "LogLoss_original": float(log_loss(y_true, clipped_original)),
        "LogLoss_reconstructed": float(log_loss(y_true, clipped_rebuilt)),
        "rtol": float(rtol),
        "atol": float(atol),
        "reproduction_status": "PASS" if passed else "FAIL",
    }


def _likely_failure_cause(error: Exception | None, comparison_failed: bool) -> str:
    if comparison_failed:
        return "stochastic non-reproducibility"
    if isinstance(error, (FileNotFoundError, KeyError)):
        return "missing historical state"
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "environment/library differences"
    return "implementation mismatch"


def reconstruct_fold(
    spec: HorizonSpec,
    diagnostics_config: dict[str, Any],
    *,
    fold_number: int,
) -> dict[str, Any]:
    """Train, persist, release, reload, and compare one authorized fold."""
    config = load_config(spec.model_config_path)
    fold = rolling_fold(config, fold_number)
    oof_path = _oof_path(config)
    iterations = int(config["training"]["iterations"])
    seed = int(config["training"]["random_state"])
    train_origins = [int(value) for value in fold["train_origins"]]
    validation_origin = int(fold["validation_origin"])
    result: dict[str, Any] = {
        "horizon": spec.name,
        "fold": int(fold_number),
        "source_config_path": str(spec.model_config_path),
        "source_config_sha256": sha256_file(spec.model_config_path),
        "retained_oof_path": str(oof_path),
        "retained_oof_sha256": sha256_file(oof_path),
        "iterations": iterations,
        "seed": seed,
        "seed_offset": int(fold_number),
        "effective_seed": seed + int(fold_number),
        "train_origins": train_origins,
        "validation_origin": validation_origin,
        "persisted_state_path": str(
            state_directory(spec.name, fold_number) / STATE_FILENAME
        ),
        "locked_rows_used": 0,
        "reproduction_status": "FAIL",
    }
    try:
        frame = load_dataset(config)
        validation_frame = frame.loc[
            frame[str(config["dataset"]["forecast_origin"])]
            .astype(int)
            .eq(validation_origin)
        ].copy()
        _validate_locked_rows(
            validation_frame,
            spec,
            label=f"{spec.name} Fold {fold_number}",
        )

        dtype = configure_runtime(config)

        def save_callback(fitted: FittedFoldState) -> None:
            persist_fold_state(
                fitted,
                horizon=spec.name,
                fold=fold_number,
                config_path=spec.model_config_path,
                config=config,
                retained_oof_path=oof_path,
            )

        original_run_result = run_fold(
            frame=frame,
            fold_number=fold_number,
            fold=fold,
            config=config,
            dtype=dtype,
            fitted_state_callback=save_callback,
        )
        del original_run_result, frame
        tf.keras.backend.clear_session()
        gc.collect()

        loaded = load_fold_state(
            state_directory(spec.name, fold_number),
            expected_config_path=spec.model_config_path,
        )
        reloaded_frame = load_dataset(loaded.config)
        validation_data = prepare_time_data(
            reloaded_frame,
            origins=[validation_origin],
            preprocessing=loaded.preprocessing,
            config=loaded.config,
        )[0]
        probability, _ = predict_frame(
            model=loaded.model,
            posterior=loaded.posterior,
            validation_data=validation_data,
            last_training_step=loaded.last_training_step,
            config=loaded.config,
            dtype=loaded.dtype,
        )
        dataset = loaded.config["dataset"]
        reconstructed = reloaded_frame.loc[
            reloaded_frame[str(dataset["forecast_origin"])]
            .astype(int)
            .eq(validation_origin),
            [
                str(dataset["cell_id"]),
                str(dataset["forecast_origin"]),
                str(dataset["target_year"]),
                str(dataset["target"]),
            ],
        ].copy()
        reconstructed["probability_reconstructed"] = probability

        retained = pd.read_parquet(oof_path)
        retained = retained.loc[
            pd.to_numeric(retained["fold"], errors="raise").eq(fold_number)
        ].copy()
        _validate_locked_rows(
            retained,
            spec,
            label=f"{spec.name} retained Fold {fold_number}",
        )
        shap_config = diagnostics_config["shap"]
        comparison = compare_oof_probabilities(
            retained,
            reconstructed,
            target_column=str(dataset["target"]),
            rtol=float(shap_config["reproduction_rtol"]),
            atol=float(shap_config["reproduction_atol"]),
        )
        result.update(comparison)
        if comparison["reproduction_status"] == "FAIL":
            result["likely_failure_cause"] = _likely_failure_cause(None, True)
        return result
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["likely_failure_cause"] = _likely_failure_cause(error, False)
        return result


def reconstruct_gate_fold(
    spec: HorizonSpec,
    diagnostics_config: dict[str, Any],
) -> dict[str, Any]:
    """Retain the validated Gate-A entry point for Fold 1."""
    return reconstruct_fold(
        spec,
        diagnostics_config,
        fold_number=GATE_FOLD,
    )


def _fold_one_file_hashes() -> dict[str, str]:
    """Snapshot every persisted Fold-1 file that later runs must not alter."""
    paths = [
        state_directory(horizon, GATE_FOLD) / filename
        for horizon in EXPECTED_HORIZONS
        for filename in (STATE_FILENAME, METADATA_FILENAME)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Fold-1 state files: " + ", ".join(missing))
    return {str(path): sha256_file(path) for path in paths}


def _require_unchanged_file_hashes(expected: dict[str, str]) -> None:
    """Fail immediately if an immutable Fold-1 state file changed."""
    actual = {path: sha256_file(Path(path)) for path in expected}
    if actual != expected:
        raise RuntimeError("An existing Fold-1 reconstruction state was modified.")


def _load_gate_a_results(preflight: dict[str, Any]) -> list[dict[str, Any]]:
    """Load and validate the two immutable PASS results used in consolidation."""
    manifest = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("gate") != "A":
        raise ValueError("The Gate-A reconstruction manifest must have PASS status.")
    if int(manifest.get("locked_rows_used", -1)) != 0:
        raise ValueError("The Gate-A manifest must report zero locked rows.")

    expected_hashes = {
        row["horizon"]: (
            row["source_config_sha256"],
            row["retained_oof_sha256"],
        )
        for row in preflight["gate_folds"]
    }
    results: list[dict[str, Any]] = []
    for row in manifest.get("folds", []):
        horizon = str(row.get("horizon"))
        if (
            horizon not in EXPECTED_HORIZONS
            or int(row.get("fold", -1)) != GATE_FOLD
            or row.get("reproduction_status") != "PASS"
            or int(row.get("locked_rows_used", -1)) != 0
        ):
            raise ValueError("Gate A must contain only passing Fold-1 results.")
        config_hash, oof_hash = expected_hashes[horizon]
        if row.get("source_config_sha256") != config_hash:
            raise ValueError(f"{horizon} Gate-A source config hash changed.")
        if row.get("retained_oof_sha256") != oof_hash:
            raise ValueError(f"{horizon} Gate-A retained OOF hash changed.")

        metadata_path = state_directory(horizon, GATE_FOLD) / METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        enriched = dict(row)
        enriched.update(
            {
                "seed_offset": int(metadata["random_seed"]["fold_offset"]),
                "effective_seed": int(metadata["random_seed"]["effective"]),
                "train_origins": [int(value) for value in metadata["train_origins"]],
                "validation_origin": int(metadata["validation_origin"]),
            }
        )
        results.append(enriched)

    if {(row["horizon"], row["fold"]) for row in results} != {
        (horizon, GATE_FOLD) for horizon in EXPECTED_HORIZONS
    }:
        raise ValueError("Gate A must contain one Fold-1 result per horizon.")
    return results


def remaining_fold_plan(
    specs: list[HorizonSpec],
) -> list[tuple[HorizonSpec, int]]:
    """Return the only four folds authorized after Gate A, in fail-stop order."""
    by_name = {spec.name: spec for spec in specs}
    if tuple(by_name) != EXPECTED_HORIZONS:
        raise ValueError(f"Reconstruction horizons must be exactly {EXPECTED_HORIZONS}.")
    return [
        (by_name["annual_1y"], 2),
        (by_name["annual_1y"], 3),
        (by_name["five_year_5y"], 2),
        (by_name["five_year_5y"], 3),
    ]


def reconstruct_remaining_folds(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Reconstruct only Folds 2 and 3 and consolidate all six fold results."""
    preflight = run_preflight(config_path)
    diagnostics_config = _load_yaml(config_path)
    specs = _resolved_specs(diagnostics_config)
    fold_one_hashes = _fold_one_file_hashes()
    fold_results = _load_gate_a_results(preflight)

    for spec, fold_number in remaining_fold_plan(specs):
        _require_unchanged_file_hashes(fold_one_hashes)
        fold_result = reconstruct_fold(
            spec,
            diagnostics_config,
            fold_number=fold_number,
        )
        _require_unchanged_file_hashes(fold_one_hashes)
        fold_results.append(fold_result)
        if fold_result["reproduction_status"] != "PASS":
            break

    order = {horizon: index for index, horizon in enumerate(EXPECTED_HORIZONS)}
    fold_results.sort(key=lambda row: (order[row["horizon"]], int(row["fold"])))
    passed = len(fold_results) == len(EXPECTED_HORIZONS) * len(ROLLING_FOLDS) and all(
        row["reproduction_status"] == "PASS" for row in fold_results
    )
    manifest = {
        "status": "PASS" if passed else "FAIL",
        "source_config_hashes": {
            row["horizon"]: row["source_config_sha256"]
            for row in preflight["gate_folds"]
        },
        "retained_oof_hashes": {
            row["horizon"]: row["retained_oof_sha256"]
            for row in preflight["gate_folds"]
        },
        "runtime": runtime_metadata(),
        "folds": fold_results,
        "fold_one_file_hashes": fold_one_hashes,
        "locked_rows_used": 0,
        "scientific_confirmation": (
            "No hyperparameter was changed, no reconstructed prediction was used for "
            "model selection, no final fit was run, and no locked test row was used."
        ),
    }
    ALL_FOLDS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALL_FOLDS_REPORT_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_gate_a(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Run only both Fold-1 reconstructions, stopping at the first failure."""
    preflight = run_preflight(config_path)
    day5_config = _load_yaml(config_path)
    fold_results = []
    for spec in _resolved_specs(day5_config):
        fold_result = reconstruct_gate_fold(spec, day5_config)
        fold_results.append(fold_result)
        if fold_result["reproduction_status"] != "PASS":
            break

    passed = len(fold_results) == len(EXPECTED_HORIZONS) and all(
        row["reproduction_status"] == "PASS" for row in fold_results
    )
    manifest = {
        "status": "PASS" if passed else "FAIL",
        "gate": "A",
        "source_config_hashes": {
            row["horizon"]: row["source_config_sha256"]
            for row in preflight["gate_folds"]
        },
        "retained_oof_hashes": {
            row["horizon"]: row["retained_oof_sha256"]
            for row in preflight["gate_folds"]
        },
        "runtime": runtime_metadata(),
        "folds": fold_results,
        "locked_rows_used": 0,
        "scientific_confirmation": (
            "No hyperparameter was changed, no model was selected using reconstructed "
            "predictions, no final fit was run, and no locked test row was used."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--gate-a", action="store_true")
    mode.add_argument("--remaining-folds", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.preflight_only:
        result = run_preflight(args.config)
    elif args.gate_a:
        result = run_gate_a(args.config)
    else:
        result = reconstruct_remaining_folds(args.config)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
