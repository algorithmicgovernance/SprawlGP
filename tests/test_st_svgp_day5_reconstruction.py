"""Focused contracts for strict Day-5 ST-SVGP reconstruction."""

from __future__ import annotations

import gc
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import yaml
from src.models.evaluation import st_svgp_day5_reconstruction as reconstruction
from src.models.evaluation.st_svgp_day5_diagnostics import HorizonSpec
from src.models.st_svgp.block_inference import DenseCviSites
from src.models.train_st_svgp import (
    FittedFoldState,
    FoldPreprocessing,
    TimeData,
    model_from_preprocessing,
    predict_frame,
    run_fold,
)


def synthetic_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "model": "st_svgp",
        "dataset": {
            "path": str(tmp_path / "dataset.parquet"),
            "target": "target_transition_5y",
            "forecast_origin": "forecast_origin",
            "target_year": "target_year",
            "cell_id": "cell_id",
            "x_coordinate": "x_center_m",
            "y_coordinate": "y_center_m",
        },
        "linear_predictors": ["feature_a", "feature_b"],
        "time": {"origin_year": 2000, "step_years": 5},
        "rolling_validation": {
            "folds": [
                {"train_origins": [2000], "validation_origin": 2005},
                {"train_origins": [2000, 2005], "validation_origin": 2010},
                {"train_origins": [2000, 2005, 2010], "validation_origin": 2015},
            ]
        },
        "final_fit": {"origins": [2000, 2005, 2010, 2015]},
        "final_test": {"origins": [2020], "evaluate": False},
        "kernel": {
            "spatial_family": "matern32",
            "temporal_family": "matern32_markov",
            "spatial_initial_lengthscale_km": 2.0,
            "temporal_initial_lengthscale_steps": 1.5,
            "temporal_lengthscale_trainable": True,
            "variance": 1.0,
        },
        "inducing": {"spatial_points": 2, "train_locations": False, "kmeans_n_init": 1},
        "inference": {
            "method": "cvi_natural_gradient",
            "filtering": "sequential",
            "smoothing": "rts",
            "initial_site_precision": 0.001,
            "natural_gradient_gamma": 0.01,
            "minimum_gamma": 1.0e-6,
            "maximum_retries": 2,
            "minimum_site_eigenvalue": 1.0e-9,
            "quadrature_degree": 10,
        },
        "training": {
            "batch_size": 4,
            "iterations": 3,
            "final_iterations": 3,
            "adam_learning_rate": 0.001,
            "gradient_clip_norm": 10.0,
            "log_every": 1,
            "prediction_batch_size": 8,
            "random_state": 123,
        },
        "compute": {"device": "cpu", "float_type": "float64", "jitter": 1.0e-6},
        "outputs": {
            "metadata_directory": str(tmp_path / "metadata"),
            "model_directory": str(tmp_path / "canonical_model"),
            "metrics_directory": str(tmp_path / "metrics"),
            "predictions_directory": str(tmp_path / "predictions"),
        },
    }


def fitted_state(config: dict[str, Any]) -> FittedFoldState:
    preprocessing = FoldPreprocessing(
        feature_mean=np.array([1.0, -2.0]),
        feature_scale=np.array([2.0, 4.0]),
        coordinate_center_km=np.array([0.5, -0.25]),
        inducing_locations_km=np.array([[0.0, 0.0], [1.0, 0.5]]),
    )
    model = model_from_preprocessing(preprocessing, config, dtype=tf.float64)
    model.beta0.assign(0.15)
    model.beta.assign([0.2, -0.3])
    model.log_spatial_lengthscales.assign(np.log([1.7, 2.3]))
    model.log_temporal_lengthscale.assign(np.log(1.2))
    model.log_variance.assign(np.log(0.8))
    sites = DenseCviSites(
        lambda1=tf.constant([[0.1, -0.2]], dtype=tf.float64),
        lambda2=tf.constant([[[-0.5, 0.0], [0.0, -0.5]]], dtype=tf.float64),
    )
    return FittedFoldState(
        model=model,
        sites=sites,
        preprocessing=preprocessing,
        train_origins=(2000,),
        validation_origin=2005,
    )


def annual_spec() -> HorizonSpec:
    return HorizonSpec(
        name="annual_1y",
        label="Annual",
        model_config_path=Path("annual.yaml"),
        output_subdirectory="annual_1y",
        maximum_target_year=2019,
        locked_forecast_origins=(),
    )


def five_year_spec() -> HorizonSpec:
    return HorizonSpec(
        name="five_year_5y",
        label="Five year",
        model_config_path=Path("five.yaml"),
        output_subdirectory="five_year_5y",
        maximum_target_year=None,
        locked_forecast_origins=(2020,),
    )


def test_run_fold_persistence_is_opt_in_and_preserves_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)
    frame = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "forecast_origin": [2000, 2000, 2005, 2005],
            "target_year": [2005, 2005, 2010, 2010],
            "target_transition_5y": [0, 1, 0, 1],
            "x_center_m": [0.0, 1000.0, 0.0, 1000.0],
            "y_center_m": [0.0, 0.0, 1000.0, 1000.0],
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [1.0, 2.0, 3.0, 4.0],
        }
    )
    state = fitted_state(config)

    def fake_train_model(**_: object) -> tuple[DenseCviSites, object, pd.DataFrame]:
        return state.sites, object(), pd.DataFrame({"iteration": [0]})

    monkeypatch.setattr(
        "src.models.train_st_svgp.fit_preprocessing",
        lambda *_: state.preprocessing,
    )
    monkeypatch.setattr(
        "src.models.train_st_svgp.model_from_preprocessing",
        lambda *_args, **_kwargs: state.model,
    )
    monkeypatch.setattr("src.models.train_st_svgp.train_model", fake_train_model)
    monkeypatch.setattr(
        "src.models.train_st_svgp.predict_frame",
        lambda **_: (np.array([0.2, 0.8]), np.array([0.4, 0.5])),
    )

    parameters = inspect.signature(run_fold).parameters
    assert parameters["fitted_state_callback"].default is None
    baseline = run_fold(
        frame=frame,
        fold_number=1,
        fold=config["rolling_validation"]["folds"][0],
        config=config,
        dtype=tf.float64,
    )
    observed: list[FittedFoldState] = []
    persisted = run_fold(
        frame=frame,
        fold_number=1,
        fold=config["rolling_validation"]["folds"][0],
        config=config,
        dtype=tf.float64,
        fitted_state_callback=observed.append,
    )

    assert len(observed) == 1
    assert baseline[0] == persisted[0]
    pd.testing.assert_frame_equal(baseline[1], persisted[1])
    pd.testing.assert_frame_equal(baseline[2], persisted[2])


def test_frozen_configs_and_retained_oof_files_are_unchanged() -> None:
    expected = {
        Path("configs/modeling/st_svgp.yaml"): (
            "aee54552739c73d48578bc3e6812ac48e4c260b076e14f96d8ab5b88e07670f0"
        ),
        Path("configs/modeling/st_svgp_annual/convergence_3000.yaml"): (
            "e2e4df0ccbe93a295964ff5223e722e7b2f15dd57ad26bb415404e9ab84d1efd"
        ),
        Path("reports/modeling/st_svgp/predictions/st_svgp_oof_predictions.parquet"): (
            "e21a1c4254114834086abac42b19c53c138fd9613c61905b0b035f760da150d0"
        ),
        Path(
            "reports/modeling/st_svgp_annual/experiments/convergence_3000/"
            "predictions/st_svgp_oof_predictions.parquet"
        ): "1e96f1495ba756a6cc83c716cdeac220ee4252b3309a3bf105942e1fe2a28aa3",
    }
    assert {path: reconstruction.sha256_file(path) for path in expected} == expected


def test_gate_a_selects_only_fold_one_and_paths_support_authorized_later_folds(
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)
    fold_number, fold = reconstruction.gate_a_fold(config)

    assert fold_number == 1
    assert fold["validation_origin"] == 2005
    assert reconstruction.state_directory("annual_1y", 2, root=tmp_path) == (
        tmp_path / "annual_1y/fold_2"
    )
    with pytest.raises(ValueError, match="one of"):
        reconstruction.state_directory("annual_1y", 4, root=tmp_path)


def test_remaining_plan_contains_only_folds_two_and_three() -> None:
    plan = reconstruction.remaining_fold_plan([annual_spec(), five_year_spec()])

    assert [(spec.name, fold) for spec, fold in plan] == [
        ("annual_1y", 2),
        ("annual_1y", 3),
        ("five_year_5y", 2),
        ("five_year_5y", 3),
    ]


@pytest.mark.parametrize(
    ("fold_number", "train_origins", "validation_origin", "effective_seed"),
    [
        (2, [2000, 2005], 2010, 125),
        (3, [2000, 2005, 2010], 2015, 126),
    ],
)
def test_reconstruction_preserves_fold_specific_split_and_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fold_number: int,
    train_origins: list[int],
    validation_origin: int,
    effective_seed: int,
) -> None:
    config = synthetic_config(tmp_path)
    monkeypatch.setattr(reconstruction, "load_config", lambda _: config)
    monkeypatch.setattr(reconstruction, "sha256_file", lambda _: "frozen-hash")
    monkeypatch.setattr(
        reconstruction,
        "load_dataset",
        lambda _: (_ for _ in ()).throw(RuntimeError("stop before training")),
    )

    result = reconstruction.reconstruct_fold(
        five_year_spec(),
        {"shap": {"reproduction_rtol": 1.0e-6, "reproduction_atol": 1.0e-8}},
        fold_number=fold_number,
    )

    assert result["fold"] == fold_number
    assert result["train_origins"] == train_origins
    assert result["validation_origin"] == validation_origin
    assert result["seed"] == 123
    assert result["seed_offset"] == fold_number
    assert result["effective_seed"] == effective_seed
    assert result["iterations"] == 3


def test_state_round_trip_contains_required_components_and_reproduces_prediction(
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    oof_path = tmp_path / "retained.parquet"
    oof_path.write_bytes(b"retained-oof-test")
    fitted = fitted_state(config)
    validation_data = TimeData(
        origin=2005,
        time_step=1.0,
        features=np.array([[0.0, 0.0], [1.0, -1.0]]),
        coordinates_km=np.array([[0.25, 0.1], [0.75, 0.4]]),
        targets=np.zeros(2),
        row_index=np.array([0, 1]),
    )
    original_posterior = fitted.model.posterior(
        times=tf.constant([0.0], dtype=tf.float64), sites=fitted.sites
    )
    expected, _ = predict_frame(
        model=fitted.model,
        posterior=original_posterior,
        validation_data=validation_data,
        last_training_step=0.0,
        config=config,
        dtype=tf.float64,
    )
    root = tmp_path / "day5_reconstructed"
    state_path = reconstruction.persist_fold_state(
        fitted,
        horizon="five_year_5y",
        fold=1,
        config_path=config_path,
        config=config,
        retained_oof_path=oof_path,
        root=root,
    )

    with np.load(state_path, allow_pickle=False) as state:
        assert reconstruction.REQUIRED_ARRAY_KEYS.issubset(state.files)
    metadata = json.loads(
        (state_path.parent / reconstruction.METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["feature_names"] == config["linear_predictors"]
    assert metadata["temporal_normalization"]["training_time_steps"] == [0.0]
    assert metadata["inference"]["variational_state"] == "dense_cvi_natural_parameters"
    assert metadata["numerics"] == {"dtype": "float64", "device": "cpu", "jitter": 1.0e-6}

    del fitted, original_posterior
    gc.collect()
    loaded = reconstruction.load_fold_state(
        state_path.parent, expected_config_path=config_path
    )
    actual, _ = predict_frame(
        model=loaded.model,
        posterior=loaded.posterior,
        validation_data=validation_data,
        last_training_step=loaded.last_training_step,
        config=loaded.config,
        dtype=loaded.dtype,
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_oof_comparison_uses_stable_keys_not_row_order() -> None:
    retained = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "forecast_origin": [2005] * 4,
            "target_year": [2010] * 4,
            "target_transition_5y": [0, 1, 0, 1],
            "probability_raw": [0.1, 0.8, 0.2, 0.9],
        }
    )
    reconstructed = retained.rename(
        columns={"probability_raw": "probability_reconstructed"}
    ).sample(frac=1.0, random_state=7)

    result = reconstruction.compare_oof_probabilities(
        retained,
        reconstructed,
        target_column="target_transition_5y",
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    assert result["reproduction_status"] == "PASS"
    assert result["max_absolute_probability_difference"] == 0.0


def test_locked_annual_and_five_year_rows_are_rejected() -> None:
    annual = pd.DataFrame({"forecast_origin": [2019], "target_year": [2020]})
    five_year = pd.DataFrame({"forecast_origin": [2020], "target_year": [2025]})

    with pytest.raises(ValueError, match="target_year"):
        reconstruction._validate_locked_rows(annual, annual_spec(), label="annual")
    with pytest.raises(ValueError, match="locked forecast origin 2020"):
        reconstruction._validate_locked_rows(five_year, five_year_spec(), label="five")


def test_gate_a_failure_stops_before_the_second_horizon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = [annual_spec(), five_year_spec()]
    preflight = {
        "status": "PASS",
        "gate_folds": [
            {
                "horizon": spec.name,
                "source_config_sha256": f"config-{spec.name}",
                "retained_oof_sha256": f"oof-{spec.name}",
            }
            for spec in specs
        ],
    }
    calls: list[str] = []

    monkeypatch.setattr(reconstruction, "REPORT_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(reconstruction, "run_preflight", lambda _: preflight)
    monkeypatch.setattr(reconstruction, "_load_yaml", lambda _: {})
    monkeypatch.setattr(reconstruction, "_resolved_specs", lambda _: specs)
    monkeypatch.setattr(reconstruction, "runtime_metadata", lambda: {"test": True})

    def fail_first(spec: HorizonSpec, _: dict[str, Any]) -> dict[str, Any]:
        calls.append(spec.name)
        return {"horizon": spec.name, "fold": 1, "reproduction_status": "FAIL"}

    monkeypatch.setattr(reconstruction, "reconstruct_gate_fold", fail_first)
    result = reconstruction.run_gate_a(tmp_path / "unused.yaml")

    assert result["status"] == "FAIL"
    assert calls == ["annual_1y"]
    assert len(result["folds"]) == 1


def test_remaining_fold_failure_stops_later_folds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = [annual_spec(), five_year_spec()]
    preflight = {
        "status": "PASS",
        "gate_folds": [
            {
                "horizon": spec.name,
                "source_config_sha256": f"config-{spec.name}",
                "retained_oof_sha256": f"oof-{spec.name}",
            }
            for spec in specs
        ],
    }
    gate_results = [
        {"horizon": spec.name, "fold": 1, "reproduction_status": "PASS"}
        for spec in specs
    ]
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(reconstruction, "ALL_FOLDS_REPORT_PATH", tmp_path / "all.json")
    monkeypatch.setattr(reconstruction, "run_preflight", lambda _: preflight)
    monkeypatch.setattr(reconstruction, "_load_yaml", lambda _: {})
    monkeypatch.setattr(reconstruction, "_resolved_specs", lambda _: specs)
    monkeypatch.setattr(reconstruction, "_fold_one_file_hashes", lambda: {"fold1": "hash"})
    monkeypatch.setattr(reconstruction, "_require_unchanged_file_hashes", lambda _: None)
    monkeypatch.setattr(reconstruction, "_load_gate_a_results", lambda _: gate_results)
    monkeypatch.setattr(reconstruction, "runtime_metadata", lambda: {"test": True})

    def reconstruct(
        spec: HorizonSpec,
        _: dict[str, Any],
        *,
        fold_number: int,
    ) -> dict[str, Any]:
        calls.append((spec.name, fold_number))
        return {
            "horizon": spec.name,
            "fold": fold_number,
            "reproduction_status": "FAIL" if fold_number == 3 else "PASS",
        }

    monkeypatch.setattr(reconstruction, "reconstruct_fold", reconstruct)
    result = reconstruction.reconstruct_remaining_folds(tmp_path / "unused.yaml")

    assert result["status"] == "FAIL"
    assert calls == [("annual_1y", 2), ("annual_1y", 3)]
    assert not any(horizon == "five_year_5y" for horizon, _ in calls)


def test_fold_one_hash_guard_detects_overwrite(tmp_path: Path) -> None:
    state = tmp_path / "fold_1/st_svgp_state.npz"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"original")
    expected = {str(state): reconstruction.sha256_file(state)}

    reconstruction._require_unchanged_file_hashes(expected)
    state.write_bytes(b"overwritten")

    with pytest.raises(RuntimeError, match="Fold-1"):
        reconstruction._require_unchanged_file_hashes(expected)


def test_persistence_cannot_overwrite_canonical_artifacts(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    oof_path = tmp_path / "retained.parquet"
    oof_path.write_bytes(b"retained")
    canonical = Path(config["outputs"]["model_directory"])
    canonical.mkdir(parents=True)
    sentinel = canonical / "sentinel.bin"
    sentinel.write_bytes(b"canonical")

    path = reconstruction.persist_fold_state(
        fitted_state(config),
        horizon="annual_1y",
        fold=1,
        config_path=config_path,
        config=config,
        retained_oof_path=oof_path,
        root=tmp_path / "artifacts/models/day5_reconstructed",
    )

    assert path.is_relative_to(tmp_path / "artifacts/models/day5_reconstructed")
    assert sentinel.read_bytes() == b"canonical"
    assert list(canonical.iterdir()) == [sentinel]


def test_make_targets_are_limited_to_authorized_reconstruction_modes() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    preflight = makefile.split(
        "\nday5-st-svgp-reconstruction-preflight:", maxsplit=1
    )[1].split("\n\n", maxsplit=1)[0]
    gate = makefile.split("\nday5-st-svgp-reconstruct-gate-a:", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]
    remaining = makefile.split(
        "\nst-svgp-reconstruct-remaining-folds:", maxsplit=1
    )[1].split("\n\n", maxsplit=1)[0]

    assert "--preflight-only" in preflight
    assert "--gate-a" not in preflight
    assert "--gate-a" in gate
    assert "train_st_svgp" not in preflight
    assert "--rolling-only" not in gate
    assert "final-fit" not in gate
    assert "--remaining-folds" in remaining
    assert "--gate-a" not in remaining
    assert "--rolling-only" not in remaining
    assert "final-fit" not in remaining
