"""Contracts for the final annual ST-SVGP optimization-duration experiment."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from src.models.evaluation import st_svgp_annual_convergence_3000 as convergence

CANONICAL_CONFIG_PATH = Path("configs/modeling/st_svgp_annual.yaml")
CONVERGENCE_1500_CONFIG_PATH = Path(
    "configs/modeling/st_svgp_annual/convergence_1500.yaml"
)
CONVERGENCE_3000_CONFIG_PATH = Path(
    "configs/modeling/st_svgp_annual/convergence_3000.yaml"
)
FIVE_YEAR_CONFIG_PATH = Path("configs/modeling/st_svgp.yaml")
CANONICAL_CONFIG_SHA256 = "2e4df618765de57ebdea7e9882f9669384860cd3946b35d8222a6bcc1f176405"
CONVERGENCE_1500_CONFIG_SHA256 = (
    "ab26f97e988b2b2ec01638ccacb94777166895188ed6e5bc2d74eda81d500f8d"
)
FIVE_YEAR_CONFIG_SHA256 = "aee54552739c73d48578bc3e6812ac48e4c260b076e14f96d8ab5b88e07670f0"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _make_target(name: str) -> str:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    return makefile.split(f"\n{name}:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]


def test_locked_configs_are_unchanged() -> None:
    assert hashlib.sha256(CANONICAL_CONFIG_PATH.read_bytes()).hexdigest() == (
        CANONICAL_CONFIG_SHA256
    )
    assert hashlib.sha256(CONVERGENCE_1500_CONFIG_PATH.read_bytes()).hexdigest() == (
        CONVERGENCE_1500_CONFIG_SHA256
    )
    assert hashlib.sha256(FIVE_YEAR_CONFIG_PATH.read_bytes()).hexdigest() == (
        FIVE_YEAR_CONFIG_SHA256
    )


def test_3000_config_changes_only_iterations_and_output_paths() -> None:
    canonical = load_yaml(CANONICAL_CONFIG_PATH)
    experiment = load_yaml(CONVERGENCE_3000_CONFIG_PATH)

    assert canonical["training"]["iterations"] == 1000
    assert experiment["training"]["iterations"] == 3000
    assert experiment["training"]["final_iterations"] == canonical["training"][
        "final_iterations"
    ]

    normalized = copy.deepcopy(experiment)
    normalized["training"]["iterations"] = canonical["training"]["iterations"]
    normalized["outputs"] = canonical["outputs"]
    assert normalized == canonical


def test_3000_config_preserves_seed_folds_features_and_temporal_lock() -> None:
    canonical = load_yaml(CANONICAL_CONFIG_PATH)
    experiment = load_yaml(CONVERGENCE_3000_CONFIG_PATH)

    assert experiment["training"]["random_state"] == canonical["training"][
        "random_state"
    ] == 20260809
    assert experiment["linear_predictors"] == canonical["linear_predictors"]
    assert len(experiment["linear_predictors"]) == 9
    assert experiment["inference"] == canonical["inference"]
    assert experiment["kernel"]["temporal_initial_lengthscale_steps"] == 7.5
    assert experiment["rolling_validation"]["folds"] == [
        {"train_origins": list(range(2000, 2010)), "validation_origin": 2010},
        {"train_origins": list(range(2000, 2015)), "validation_origin": 2015},
        {"train_origins": list(range(2000, 2018)), "validation_origin": 2018},
    ]
    assert experiment["development"]["maximum_target_year"] == 2019
    assert experiment["locked_block"]["evaluate"] is False
    assert max(
        origin + experiment["time"]["step_years"]
        for fold in experiment["rolling_validation"]["folds"]
        for origin in [*fold["train_origins"], fold["validation_origin"]]
    ) == 2019
    assert all(
        "/experiments/convergence_3000" in path
        for path in experiment["outputs"].values()
    )

    result = convergence.validate_preflight(CONVERGENCE_3000_CONFIG_PATH)
    assert result["architecture_parity"] is True
    assert result["final_test_evaluated"] is False


def test_make_targets_keep_preflight_training_and_evaluation_separate() -> None:
    preflight = _make_target("st-svgp-annual-convergence-3000-preflight")
    training = _make_target("st-svgp-annual-convergence-3000")
    evaluation = _make_target("st-svgp-annual-convergence-3000-evaluate")

    assert "st_svgp_annual_convergence_3000" in preflight
    assert "src.models.train_st_svgp" in preflight
    assert preflight.count("--preflight-only") == 2
    assert "--rolling-only" not in preflight

    assert "src.models.train_st_svgp" in training
    assert "$(ST_SVGP_ANNUAL_CONVERGENCE_3000_CONFIG)" in training
    assert "--rolling-only" in training
    assert "--preflight-only" not in training
    assert "final-fit" not in training
    assert "$(ST_SVGP_ANNUAL_CONFIG)" not in training
    assert "$(ST_SVGP_CONFIG)" not in training

    assert "st_svgp_annual_convergence_3000" in evaluation
    assert "train_st_svgp" not in evaluation


def test_probability_and_coverage_delegate_to_retained_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_yaml(CONVERGENCE_3000_CONFIG_PATH)
    paths = convergence.experiment_paths(config)
    calls: dict[str, dict] = {}

    def fake_probability(**kwargs: object) -> None:
        calls["probability"] = kwargs

    def fake_mondrian(**kwargs: object) -> None:
        calls["mondrian"] = kwargs

    monkeypatch.setattr(convergence, "run_probability_evaluation", fake_probability)
    monkeypatch.setattr(convergence, "run_annual_mondrian", fake_mondrian)
    convergence.evaluate_probability_and_coverage(config, paths)

    probability_model = calls["probability"]["models"][0]
    mondrian_model = calls["mondrian"]["model"]
    assert probability_model is mondrian_model
    assert probability_model.path == paths.predictions
    assert probability_model.maximum_target_year == 2019
    assert calls["mondrian"]["marginal_coverage_path"] == paths.marginal_coverage


def _budget_frames(iterations: int) -> convergence.BudgetFrames:
    fold = np.array([1, 2, 3])
    offset = (iterations - 1000) / 1_000_000
    parameters = pd.DataFrame(
        {
            "fold": fold,
            "temporal_lengthscale_years": [3.0, 3.1, 3.2],
            "spatial_lengthscale_x_km": [3.5, 3.6, 3.7],
            "spatial_lengthscale_y_km": [3.4, 3.5, 3.6],
            "kernel_variance": [0.60, 0.61, 0.62],
            "mean_latent_variance": [0.12, 0.13, 0.14],
            "median_latent_variance": [0.11, 0.12, 0.13],
        }
    )
    metrics = pd.DataFrame(
        {
            "fold": fold,
            "pr_auc": np.array([0.20, 0.21, 0.22]) + offset,
            "roc_auc": np.array([0.70, 0.71, 0.72]) + offset,
            "log_loss": [0.20, 0.30, 0.40],
            "brier_score": [0.10, 0.11, 0.12],
            "ece": [0.03, 0.04, 0.05],
            "calibration_intercept": [-0.1, -0.1, -0.1],
            "calibration_slope": [0.9, 0.9, 0.9],
            "probability_bias": [-0.01, 0.0, 0.01],
            "absolute_probability_bias": [0.01, 0.0, 0.01],
        }
    )
    prediction_rows = []
    for fold_number, target_year in ((1, 2011), (2, 2016), (3, 2019)):
        for target in (0, 0, 0, 1):
            prediction_rows.append(
                {
                    "fold": fold_number,
                    "target_year": target_year,
                    "target_transition_1y": target,
                    "probability_raw": 0.2,
                }
            )
    coverage = pd.DataFrame(
        {
            "fold": [2, 3],
            "empirical_coverage": [0.8, 0.81],
            "positive_class_coverage": [0.7, 0.71],
            "negative_class_coverage": [0.82, 0.83],
            "average_set_size": [1.0, 1.1],
            "singleton_rate": [1.0, 0.9],
            "both_labels_rate": [0.0, 0.1],
            "empty_set_rate": [0.0, 0.0],
        }
    )
    return convergence.BudgetFrames(
        parameters=parameters,
        probability_metrics=metrics,
        predictions=pd.DataFrame(prediction_rows),
        marginal_coverage=coverage,
        mondrian_coverage=coverage,
    )


def test_budget_comparison_aligns_folds_and_calculates_lift() -> None:
    comparison = convergence.build_optimization_budget_comparison(
        {budget: _budget_frames(budget) for budget in (1000, 1500, 3000)}
    )

    assert len(comparison) == 9
    assert comparison.groupby("iterations")["fold"].apply(tuple).to_dict() == {
        1000: (1, 2, 3),
        1500: (1, 2, 3),
        3000: (1, 2, 3),
    }
    first = comparison.iloc[0]
    assert first["observed_prevalence"] == pytest.approx(0.25)
    assert first["pr_auc_prevalence_lift"] == pytest.approx(0.8)
    assert np.isnan(first["marginal_empirical_coverage"])
    assert comparison.loc[comparison["fold"].eq(2), "mondrian_positive_class_coverage"].notna().all()


def _history() -> pd.DataFrame:
    rows = []
    for fold in (1, 2, 3):
        for iteration in (2000, 2200, 2400, 2600, 2800, 2999):
            final = iteration >= 2600
            rows.append(
                {
                    "fold": fold,
                    "iteration": iteration,
                    "elbo_stochastic": -1000.0 + iteration / 10 + fold,
                    "temporal_lengthscale_steps": 3.0 * (1.005 if final else 1.0),
                    "spatial_lengthscale_x_km": 3.5 * (1.03 if final else 1.0),
                    "spatial_lengthscale_y_km": 3.6 * (1.08 if final else 1.0),
                    "kernel_variance": 0.6,
                }
            )
    return pd.DataFrame(rows)


def test_window_diagnostics_use_actual_logs_and_descriptive_thresholds() -> None:
    diagnostics = convergence.build_window_diagnostics(_history(), step_years=1.0)

    assert diagnostics["earlier_window_first_iteration"].eq(2000).all()
    assert diagnostics["earlier_window_last_iteration"].eq(2400).all()
    assert diagnostics["final_window_first_iteration"].eq(2600).all()
    assert diagnostics["final_window_last_iteration"].eq(2999).all()
    assert diagnostics["elbo_all_finite"].all()
    assert diagnostics["temporal_lengthscale_years_drift_interpretation"].eq(
        "practically plateaued"
    ).all()
    assert diagnostics["spatial_lengthscale_x_km_drift_interpretation"].eq(
        "slowly stabilizing"
    ).all()
    assert diagnostics["spatial_lengthscale_y_km_drift_interpretation"].eq(
        "still materially drifting"
    ).all()


def test_assessment_covers_all_domains_and_ends_with_one_label() -> None:
    comparison = convergence.build_optimization_budget_comparison(
        {budget: _budget_frames(budget) for budget in (1000, 1500, 3000)}
    )
    stable_history = _history().assign(
        spatial_lengthscale_x_km=3.5,
        spatial_lengthscale_y_km=3.6,
    )
    diagnostics = convergence.build_window_diagnostics(stable_history, step_years=1.0)
    report = convergence.render_assessment(comparison, diagnostics, diagnostics)

    for heading in (
        "## 1. Parameter stability",
        "## 2. Temporal identifiability",
        "## 3. Discrimination",
        "## 4. Probability quality",
        "## 5. Calibration shape",
        "## 6. Uncertainty",
        "## 7. ELBO",
        "## 8. Fold robustness",
    ):
        assert heading in report
    assert report.rstrip().endswith("STRONG_ANNUAL_ST_SVGP_SUPPORTED")