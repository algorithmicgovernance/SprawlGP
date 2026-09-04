"""Contracts for the isolated annual ST-SVGP convergence diagnostic."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from src.models.evaluation.st_svgp_annual_convergence import (
    artifact_paths,
    build_convergence_comparison,
    build_window_diagnostics,
    render_convergence_report,
)
from src.models.train_st_svgp import validate_development_splits

BASELINE_CONFIG_PATH = Path("configs/modeling/st_svgp_annual.yaml")
CONVERGENCE_CONFIG_PATH = Path("configs/modeling/st_svgp_annual/convergence_1500.yaml")
BASELINE_CONFIG_SHA256 = "2e4df618765de57ebdea7e9882f9669384860cd3946b35d8222a6bcc1f176405"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_canonical_annual_config_is_unchanged() -> None:
    assert hashlib.sha256(BASELINE_CONFIG_PATH.read_bytes()).hexdigest() == (
        BASELINE_CONFIG_SHA256
    )


def test_convergence_config_changes_only_iterations_and_outputs() -> None:
    baseline = load_yaml(BASELINE_CONFIG_PATH)
    convergence = load_yaml(CONVERGENCE_CONFIG_PATH)

    assert baseline["training"]["iterations"] == 1000
    assert convergence["training"]["iterations"] == 1500
    assert convergence["training"]["final_iterations"] == baseline["training"][
        "final_iterations"
    ]

    normalized = copy.deepcopy(convergence)
    normalized["training"]["iterations"] = baseline["training"]["iterations"]
    normalized["outputs"] = baseline["outputs"]
    assert normalized == baseline


def test_convergence_config_preserves_temporal_lock_seed_and_isolation() -> None:
    baseline = load_yaml(BASELINE_CONFIG_PATH)
    convergence = load_yaml(CONVERGENCE_CONFIG_PATH)

    assert convergence["rolling_validation"]["folds"] == [
        {"train_origins": list(range(2000, 2010)), "validation_origin": 2010},
        {"train_origins": list(range(2000, 2015)), "validation_origin": 2015},
        {"train_origins": list(range(2000, 2018)), "validation_origin": 2018},
    ]
    assert convergence["training"]["random_state"] == baseline["training"][
        "random_state"
    ] == 20260809
    assert convergence["development"]["maximum_target_year"] == 2019
    assert convergence["locked_block"]["evaluate"] is False
    validate_development_splits(convergence)

    for key, convergence_path in convergence["outputs"].items():
        assert "/experiments/convergence_1500" in convergence_path
        assert convergence_path != baseline["outputs"][key]


def test_make_targets_are_rolling_only_and_do_not_overwrite_baseline() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    training_block = makefile.split("st-svgp-annual-convergence-1500:", maxsplit=1)[1]
    training_block = training_block.split("\n\n", maxsplit=1)[0]

    assert "--config $(ST_SVGP_ANNUAL_CONVERGENCE_1500_CONFIG)" in training_block
    assert "--rolling-only" in training_block
    assert "--preflight-only" not in training_block
    assert "st_svgp_annual_convergence" in training_block
    assert "st-svgp-annual-final-fit" not in training_block
    assert "$(ST_SVGP_ANNUAL_CONFIG)" not in training_block


def test_comparison_paths_read_baseline_and_write_only_isolated_outputs() -> None:
    config = load_yaml(CONVERGENCE_CONFIG_PATH)
    paths = artifact_paths(config)

    assert paths.baseline_parameters == Path(
        "reports/modeling/st_svgp_annual/metrics/temporal_parameter_summary.csv"
    )
    assert paths.baseline_probability_metrics == Path(
        "reports/modeling/st_svgp_annual/metrics/probability_metrics_by_fold.csv"
    )
    for path in (
        paths.extended_parameters,
        paths.extended_probability_metrics,
        paths.extended_history,
        paths.extended_predictions,
        paths.comparison,
        paths.window_diagnostics,
        paths.report,
    ):
        assert "experiments/convergence_1500" in str(path)


def _parameter_table(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "temporal_lengthscale_years": np.array([3.0, 3.1, 3.2]) + offset,
            "spatial_lengthscale_x_km": np.array([3.5, 3.6, 3.7]) + offset,
            "spatial_lengthscale_y_km": np.array([3.4, 3.5, 3.6]) + offset,
            "kernel_variance": np.array([0.6, 0.61, 0.62]) + offset,
            "mean_latent_variance": np.array([0.12, 0.13, 0.14]) + offset,
            "median_latent_variance": np.array([0.11, 0.12, 0.13]) + offset,
        }
    )


def _metric_table(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "log_loss": np.array([0.2, 0.3, 0.4]) + offset,
            "brier_score": np.array([0.1, 0.11, 0.12]) + offset,
            "pr_auc": np.array([0.2, 0.21, 0.22]) + offset,
            "roc_auc": np.array([0.7, 0.71, 0.72]) + offset,
            "ece": np.array([0.03, 0.04, 0.05]) + offset,
            "calibration_slope": np.array([0.8, 0.9, 1.0]) + offset,
            "probability_bias": np.array([-0.01, 0.0, 0.01]) + offset,
        }
    )


def _history() -> pd.DataFrame:
    rows = []
    for fold in (1, 2, 3):
        for iteration, multiplier in (
            (1000, 1.000),
            (1100, 1.002),
            (1200, 1.004),
            (1300, 1.005),
            (1400, 1.006),
            (1499, 1.007),
        ):
            rows.append(
                {
                    "fold": fold,
                    "iteration": iteration,
                    "elbo_stochastic": -1000.0 + iteration / 10 + fold,
                    "temporal_lengthscale_steps": 3.0 * multiplier,
                    "spatial_lengthscale_x_km": 3.5 * multiplier,
                    "spatial_lengthscale_y_km": 3.6 * multiplier,
                    "kernel_variance": 0.6 * multiplier,
                }
            )
    return pd.DataFrame(rows)


def test_comparison_and_window_diagnostics_use_fold_aligned_artifacts() -> None:
    comparison = build_convergence_comparison(
        _parameter_table(0.0),
        _parameter_table(0.01),
        _metric_table(0.0),
        _metric_table(0.001),
    )
    diagnostics = build_window_diagnostics(_history(), step_years=1.0)

    assert comparison["fold"].tolist() == [1, 2, 3]
    assert comparison.loc[0, "baseline_temporal_lengthscale_years"] == 3.0
    assert comparison.loc[0, "extended_temporal_lengthscale_years"] == 3.01
    assert comparison.loc[2, "extended_roc_auc"] == 0.721
    assert {
        "baseline_mean_latent_variance",
        "extended_mean_latent_variance",
        "relative_spatial_x_change",
        "relative_spatial_y_change",
    }.issubset(comparison.columns)

    assert diagnostics["earlier_window_first_iteration"].eq(1000).all()
    assert diagnostics["earlier_window_last_iteration"].eq(1200).all()
    assert diagnostics["final_window_first_iteration"].eq(1300).all()
    assert diagnostics["final_window_last_iteration"].eq(1499).all()
    assert diagnostics["elbo_all_finite"].all()

    report = render_convergence_report(comparison, diagnostics)
    assert "minibatch noise" in report
    assert "not an optimizer convergence theorem" in report
    assert report.rstrip().endswith("PARAMETERS_PRACTICALLY_STABILIZED")
