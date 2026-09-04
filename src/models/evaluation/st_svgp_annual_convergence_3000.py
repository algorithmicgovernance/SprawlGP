"""Evaluate the final annual ST-SVGP optimization-duration experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from src.models.evaluation.evaluate_probabilities import (
    ModelInput,
    run_annual_mondrian,
)
from src.models.evaluation.evaluate_probabilities import (
    run as run_probability_evaluation,
)
from src.models.evaluation.prediction_sets import (
    temporal_mondrian_prediction_set_coverage,
)

DEFAULT_CONFIG_PATH = Path("configs/modeling/st_svgp_annual/convergence_3000.yaml")
CANONICAL_CONFIG_PATH = Path("configs/modeling/st_svgp_annual.yaml")
BASELINE_METRICS_DIRECTORY = Path("reports/modeling/st_svgp_annual/metrics")
BASELINE_PREDICTIONS_DIRECTORY = Path("reports/modeling/st_svgp_annual/predictions")
CONVERGENCE_1500_METRICS_DIRECTORY = Path(
    "reports/modeling/st_svgp_annual/experiments/convergence_1500/metrics"
)
CONVERGENCE_1500_PREDICTIONS_DIRECTORY = Path(
    "reports/modeling/st_svgp_annual/experiments/convergence_1500/predictions"
)
EXPECTED_FOLDS = (1, 2, 3)
EXPECTED_COVERAGE_FOLDS = (2, 3)
EXPECTED_BUDGETS = (1000, 1500, 3000)
EARLIER_LATE_WINDOW = (2000, 2400)
FINAL_LATE_WINDOW = (2600, 3000)
PRACTICAL_PARAMETER_DRIFT = 0.01
SLOW_PARAMETER_DRIFT = 0.05


@dataclass(frozen=True)
class BudgetFrames:
    """Read-only inputs needed for one optimization budget."""

    parameters: pd.DataFrame
    probability_metrics: pd.DataFrame
    predictions: pd.DataFrame
    marginal_coverage: pd.DataFrame
    mondrian_coverage: pd.DataFrame


@dataclass(frozen=True)
class ExperimentPaths:
    """Isolated 3000-run artifacts and report outputs."""

    metrics_directory: Path
    predictions: Path
    parameters: Path
    probability_metrics: Path
    marginal_coverage: Path
    mondrian_coverage: Path
    history: Path
    budget_comparison: Path
    window_diagnostics: Path
    assessment: Path


def load_config(path: Path) -> dict[str, Any]:
    """Load one YAML configuration."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def experiment_paths(config: Mapping[str, Any]) -> ExperimentPaths:
    """Resolve only experiment-local 3000 paths."""
    metrics_directory = Path(config["outputs"]["metrics_directory"])
    predictions_directory = Path(config["outputs"]["predictions_directory"])
    return ExperimentPaths(
        metrics_directory=metrics_directory,
        predictions=predictions_directory / "st_svgp_oof_predictions.parquet",
        parameters=(
            metrics_directory
            / str(config["reporting"]["temporal_parameter_summary_filename"])
        ),
        probability_metrics=metrics_directory / "probability_metrics_by_fold.csv",
        marginal_coverage=metrics_directory / "prediction_set_coverage_80.csv",
        mondrian_coverage=metrics_directory / "prediction_set_coverage_80_mondrian.csv",
        history=metrics_directory / "st_svgp_training_history.csv",
        budget_comparison=metrics_directory / "optimization_budget_comparison.csv",
        window_diagnostics=metrics_directory / "convergence_window_diagnostics.csv",
        assessment=metrics_directory / "strong_candidate_assessment.md",
    )


def validate_preflight(
    config_path: Path = DEFAULT_CONFIG_PATH,
    canonical_path: Path = CANONICAL_CONFIG_PATH,
) -> dict[str, Any]:
    """Verify that the 3000 config changes only budget and isolated outputs."""
    canonical = load_config(canonical_path)
    experiment = load_config(config_path)
    if int(experiment["training"]["iterations"]) != 3000:
        raise ValueError("The convergence experiment requires exactly 3000 iterations.")

    expected_outputs = {
        "metadata_directory": (
            "data/metadata/modeling/st_svgp_annual/experiments/convergence_3000"
        ),
        "model_directory": (
            "artifacts/models/st_svgp_annual/experiments/convergence_3000"
        ),
        "metrics_directory": (
            "reports/modeling/st_svgp_annual/experiments/convergence_3000/metrics"
        ),
        "predictions_directory": (
            "reports/modeling/st_svgp_annual/experiments/convergence_3000/predictions"
        ),
    }
    if experiment["outputs"] != expected_outputs:
        raise ValueError("The 3000 outputs are not exactly isolated as required.")

    normalized = copy.deepcopy(experiment)
    normalized["training"]["iterations"] = canonical["training"]["iterations"]
    normalized["outputs"] = canonical["outputs"]
    if normalized != canonical:
        raise ValueError(
            "The 3000 config differs scientifically from the canonical annual config."
        )

    expected_folds = [
        {"train_origins": list(range(2000, 2010)), "validation_origin": 2010},
        {"train_origins": list(range(2000, 2015)), "validation_origin": 2015},
        {"train_origins": list(range(2000, 2018)), "validation_origin": 2018},
    ]
    if experiment["rolling_validation"]["folds"] != expected_folds:
        raise ValueError("The annual rolling folds changed.")
    if int(experiment["development"]["maximum_target_year"]) != 2019:
        raise ValueError("The maximum development target year must be 2019.")
    if bool(experiment["locked_block"]["evaluate"]):
        raise ValueError("The locked annual block must not be evaluated.")

    horizon = int(experiment["time"]["step_years"])
    development_origins = [
        int(origin)
        for fold in expected_folds
        for origin in [*fold["train_origins"], fold["validation_origin"]]
    ]
    if max(origin + horizon for origin in development_origins) != 2019:
        raise ValueError("The rolling folds do not end at target year 2019.")

    return {
        "status": "PASS",
        "iterations": 3000,
        "architecture_parity": True,
        "random_state": int(experiment["training"]["random_state"]),
        "folds": len(expected_folds),
        "features": len(experiment["linear_predictors"]),
        "inference": str(experiment["inference"]["method"]),
        "temporal_initial_lengthscale_steps": float(
            experiment["kernel"]["temporal_initial_lengthscale_steps"]
        ),
        "maximum_development_target_year": 2019,
        "outputs_isolated": True,
        "final_test_evaluated": False,
    }


def _validated_fold_table(
    frame: pd.DataFrame,
    required_columns: set[str],
    label: str,
    expected_folds: tuple[int, ...] = EXPECTED_FOLDS,
) -> pd.DataFrame:
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")
    result = frame.copy()
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    if result["fold"].duplicated().any():
        raise ValueError(f"{label} contains duplicate folds.")
    folds = tuple(sorted(result["fold"].tolist()))
    if folds != expected_folds:
        raise ValueError(f"{label} must contain folds {expected_folds}; got {folds}.")
    return result.sort_values("fold").reset_index(drop=True)


def _prediction_prevalence(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"fold", "target_transition_1y", "target_year"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"{label} predictions are missing: {', '.join(missing)}")
    target_year = pd.to_numeric(predictions["target_year"], errors="raise")
    if (target_year >= 2020).any():
        raise ValueError(f"{label} predictions contain target year >= 2020.")
    table = predictions.copy()
    table["fold"] = pd.to_numeric(table["fold"], errors="raise").astype(int)
    folds = tuple(sorted(table["fold"].unique().tolist()))
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"{label} predictions must contain folds {EXPECTED_FOLDS}.")
    return (
        table.groupby("fold", as_index=False)["target_transition_1y"]
        .mean()
        .rename(columns={"target_transition_1y": "observed_prevalence"})
    )


def _coverage_columns(frame: pd.DataFrame, prefix: str, label: str) -> pd.DataFrame:
    source_columns = [
        "fold",
        "empirical_coverage",
        "positive_class_coverage",
        "negative_class_coverage",
        "average_set_size",
        "singleton_rate",
        "both_labels_rate",
        "empty_set_rate",
    ]
    validated = _validated_fold_table(
        frame,
        set(source_columns),
        label,
        expected_folds=EXPECTED_COVERAGE_FOLDS,
    )
    return validated.loc[:, source_columns].rename(
        columns={
            column: f"{prefix}_{column}"
            for column in source_columns
            if column != "fold"
        }
    )


def build_optimization_budget_comparison(
    budgets: Mapping[int, BudgetFrames],
) -> pd.DataFrame:
    """Build one aligned row per budget and rolling fold."""
    if tuple(sorted(budgets)) != EXPECTED_BUDGETS:
        raise ValueError(f"Budgets must be exactly {EXPECTED_BUDGETS}.")

    parameter_columns = {
        "fold",
        "temporal_lengthscale_years",
        "spatial_lengthscale_x_km",
        "spatial_lengthscale_y_km",
        "kernel_variance",
        "mean_latent_variance",
        "median_latent_variance",
    }
    metric_columns = {
        "fold",
        "pr_auc",
        "roc_auc",
        "log_loss",
        "brier_score",
        "ece",
        "calibration_intercept",
        "calibration_slope",
        "probability_bias",
        "absolute_probability_bias",
    }
    tables: list[pd.DataFrame] = []
    for iterations in EXPECTED_BUDGETS:
        budget = budgets[iterations]
        parameters = _validated_fold_table(
            budget.parameters,
            parameter_columns,
            f"{iterations} parameter summary",
        ).loc[:, sorted(parameter_columns, key=lambda value: value != "fold")]
        metrics = _validated_fold_table(
            budget.probability_metrics,
            metric_columns,
            f"{iterations} probability metrics",
        ).loc[:, sorted(metric_columns, key=lambda value: value != "fold")]
        prevalence = _prediction_prevalence(
            budget.predictions,
            str(iterations),
        )
        marginal = _coverage_columns(
            budget.marginal_coverage,
            "marginal",
            f"{iterations} marginal coverage",
        )
        mondrian = _coverage_columns(
            budget.mondrian_coverage,
            "mondrian",
            f"{iterations} Mondrian coverage",
        )

        table = parameters.merge(metrics, on="fold", validate="one_to_one")
        table = table.merge(prevalence, on="fold", validate="one_to_one")
        table = table.merge(marginal, on="fold", how="left", validate="one_to_one")
        table = table.merge(mondrian, on="fold", how="left", validate="one_to_one")
        table.insert(0, "iterations", iterations)
        table["pr_auc_prevalence_lift"] = table["pr_auc"] / table[
            "observed_prevalence"
        ]
        tables.append(table)

    return pd.concat(tables, ignore_index=True).sort_values(
        ["iterations", "fold"]
    ).reset_index(drop=True)


def _relative_window_difference(earlier_mean: float, final_mean: float) -> float:
    if math.isclose(earlier_mean, 0.0):
        return float("nan")
    return abs(final_mean - earlier_mean) / abs(earlier_mean)


def build_window_diagnostics(
    history: pd.DataFrame,
    step_years: float,
    earlier_window: tuple[int, int] = EARLIER_LATE_WINDOW,
    final_window: tuple[int, int] = FINAL_LATE_WINDOW,
) -> pd.DataFrame:
    """Summarize 3000-run parameter drift using actual logged iterations."""
    required = {
        "fold",
        "iteration",
        "elbo_stochastic",
        "temporal_lengthscale_steps",
        "spatial_lengthscale_x_km",
        "spatial_lengthscale_y_km",
        "kernel_variance",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"3000 training history is missing: {', '.join(missing)}")

    numeric = history.copy()
    for column in required:
        numeric[column] = pd.to_numeric(numeric[column], errors="raise")
    numeric["fold"] = numeric["fold"].astype(int)
    numeric["iteration"] = numeric["iteration"].astype(int)
    folds = tuple(sorted(numeric["fold"].unique().tolist()))
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"3000 history must contain folds {EXPECTED_FOLDS}.")
    numeric["temporal_lengthscale_years"] = (
        numeric["temporal_lengthscale_steps"] * float(step_years)
    )

    parameters = (
        "temporal_lengthscale_years",
        "spatial_lengthscale_x_km",
        "spatial_lengthscale_y_km",
        "kernel_variance",
    )
    rows: list[dict[str, float | int | bool | str]] = []
    for fold in EXPECTED_FOLDS:
        fold_history = numeric.loc[numeric["fold"] == fold]
        earlier = fold_history.loc[
            fold_history["iteration"].between(*earlier_window, inclusive="both")
        ]
        final = fold_history.loc[
            fold_history["iteration"].between(*final_window, inclusive="both")
        ]
        if earlier.empty or final.empty:
            raise ValueError(f"Fold {fold} has no logged points in a late window.")

        row: dict[str, float | int | bool | str] = {
            "fold": fold,
            "earlier_window_first_iteration": int(earlier["iteration"].min()),
            "earlier_window_last_iteration": int(earlier["iteration"].max()),
            "earlier_window_logged_points": int(len(earlier)),
            "final_window_first_iteration": int(final["iteration"].min()),
            "final_window_last_iteration": int(final["iteration"].max()),
            "final_window_logged_points": int(len(final)),
        }
        for parameter in parameters:
            earlier_mean = float(earlier[parameter].mean())
            final_mean = float(final[parameter].mean())
            drift = _relative_window_difference(earlier_mean, final_mean)
            row[f"{parameter}_earlier_window_mean"] = earlier_mean
            row[f"{parameter}_final_window_mean"] = final_mean
            row[f"{parameter}_relative_window_mean_difference"] = drift
            row[f"{parameter}_drift_interpretation"] = _drift_interpretation(drift)

        earlier_elbo = earlier["elbo_stochastic"]
        final_elbo = final["elbo_stochastic"]
        row.update(
            {
                "earlier_late_window_mean_elbo": float(earlier_elbo.mean()),
                "final_window_mean_elbo": float(final_elbo.mean()),
                "final_window_elbo_standard_deviation": float(
                    final_elbo.std(ddof=1)
                ),
                "elbo_all_finite": bool(
                    np.isfinite(fold_history["elbo_stochastic"]).all()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _drift_interpretation(value: float) -> str:
    if value <= PRACTICAL_PARAMETER_DRIFT:
        return "practically plateaued"
    if value <= SLOW_PARAMETER_DRIFT:
        return "slowly stabilizing"
    return "still materially drifting"


def _load_mondrian_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply the retained Mondrian protocol without writing locked outputs."""
    _prediction_prevalence(predictions, "Mondrian input")
    return temporal_mondrian_prediction_set_coverage(
        frame=predictions,
        target_column="target_transition_1y",
        probability_column="probability_raw",
        fold_column="fold",
        target_coverage=0.80,
    )


def _read_budget_frames(paths: ExperimentPaths) -> dict[int, BudgetFrames]:
    baseline_predictions = pd.read_parquet(
        BASELINE_PREDICTIONS_DIRECTORY / "st_svgp_oof_predictions.parquet"
    )
    convergence_1500_predictions = pd.read_parquet(
        CONVERGENCE_1500_PREDICTIONS_DIRECTORY / "st_svgp_oof_predictions.parquet"
    )
    return {
        1000: BudgetFrames(
            parameters=pd.read_csv(
                BASELINE_METRICS_DIRECTORY / "temporal_parameter_summary.csv"
            ),
            probability_metrics=pd.read_csv(
                BASELINE_METRICS_DIRECTORY / "probability_metrics_by_fold.csv"
            ),
            predictions=baseline_predictions,
            marginal_coverage=pd.read_csv(
                BASELINE_METRICS_DIRECTORY / "prediction_set_coverage_80.csv"
            ),
            mondrian_coverage=pd.read_csv(
                BASELINE_METRICS_DIRECTORY
                / "prediction_set_coverage_80_mondrian.csv"
            ),
        ),
        1500: BudgetFrames(
            parameters=pd.read_csv(
                CONVERGENCE_1500_METRICS_DIRECTORY
                / "temporal_parameter_summary.csv"
            ),
            probability_metrics=pd.read_csv(
                CONVERGENCE_1500_METRICS_DIRECTORY
                / "probability_metrics_by_fold.csv"
            ),
            predictions=convergence_1500_predictions,
            marginal_coverage=pd.read_csv(
                CONVERGENCE_1500_METRICS_DIRECTORY
                / "prediction_set_coverage_80.csv"
            ),
            mondrian_coverage=_load_mondrian_from_predictions(
                convergence_1500_predictions
            ),
        ),
        3000: BudgetFrames(
            parameters=pd.read_csv(paths.parameters),
            probability_metrics=pd.read_csv(paths.probability_metrics),
            predictions=pd.read_parquet(paths.predictions),
            marginal_coverage=pd.read_csv(paths.marginal_coverage),
            mondrian_coverage=pd.read_csv(paths.mondrian_coverage),
        ),
    }


def _mean_metric(table: pd.DataFrame, column: str) -> float:
    return float(table[column].mean())


def _decision_label(
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[str, dict[str, bool]]:
    by_budget = {
        budget: comparison.loc[comparison["iterations"] == budget]
        for budget in EXPECTED_BUDGETS
    }
    drift_columns = [
        f"{parameter}_relative_window_mean_difference"
        for parameter in (
            "temporal_lengthscale_years",
            "spatial_lengthscale_x_km",
            "spatial_lengthscale_y_km",
            "kernel_variance",
        )
    ]
    maximum_drift = float(diagnostics[drift_columns].to_numpy(dtype=float).max())
    temporal = by_budget[3000]["temporal_lengthscale_years"]
    temporal_cv = float(temporal.std(ddof=1) / abs(temporal.mean()))

    discrimination = all(
        _mean_metric(by_budget[3000], metric)
        >= _mean_metric(by_budget[1000], metric) - 0.01
        for metric in ("pr_auc", "roc_auc")
    )
    probability_quality = all(
        _mean_metric(by_budget[3000], metric)
        <= _mean_metric(by_budget[1000], metric) + 0.01
        for metric in (
            "log_loss",
            "brier_score",
            "ece",
            "absolute_probability_bias",
        )
    ) and (
        abs(_mean_metric(by_budget[3000], "calibration_slope") - 1.0)
        <= abs(_mean_metric(by_budget[1000], "calibration_slope") - 1.0) + 0.10
    )
    uncertainty = (
        _mean_metric(by_budget[3000], "mondrian_positive_class_coverage")
        >= _mean_metric(by_budget[1000], "mondrian_positive_class_coverage") - 0.01
    )
    fold_gains = by_budget[3000].set_index("fold")["pr_auc"] - by_budget[1000].set_index(
        "fold"
    )["pr_auc"]
    fold2 = comparison.loc[comparison["fold"].eq(2)].set_index("iterations")
    fold2_robust = not (
        float(fold2.loc[3000, "pr_auc"]) < float(fold2.loc[1000, "pr_auc"]) - 0.01
        and float(fold2.loc[3000, "roc_auc"])
        < float(fold2.loc[1000, "roc_auc"]) - 0.01
    )
    evidence = {
        "parameter_stability": maximum_drift <= SLOW_PARAMETER_DRIFT,
        "temporal_identifiability": temporal_cv <= 0.10,
        "discrimination": discrimination,
        "probability_quality": probability_quality,
        "uncertainty": uncertainty,
        "fold_robustness": int((fold_gains >= 0.0).sum()) >= 2 and fold2_robust,
        "finite_elbo": bool(diagnostics["elbo_all_finite"].all()),
    }
    core = [value for key, value in evidence.items() if key != "finite_elbo"]
    if all(core) and evidence["finite_elbo"]:
        return "STRONG_ANNUAL_ST_SVGP_SUPPORTED", evidence
    if sum(core) >= 4 and evidence["finite_elbo"]:
        return "ANNUAL_ST_SVGP_PROMISING_BUT_NOT_STRONG", evidence
    return "ANNUAL_ST_SVGP_NOT_SUPPORTED_AS_STRONG", evidence


def render_assessment(
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    diagnostics_1500: pd.DataFrame,
) -> str:
    """Render the multi-domain Strong Annual ST-SVGP assessment."""
    label, evidence = _decision_label(comparison, diagnostics)
    by_budget = {
        budget: comparison.loc[comparison["iterations"] == budget].set_index("fold")
        for budget in EXPECTED_BUDGETS
    }
    drift_parameters = (
        "temporal_lengthscale_years",
        "spatial_lengthscale_x_km",
        "spatial_lengthscale_y_km",
        "kernel_variance",
    )
    drift_columns = [
        f"{parameter}_relative_window_mean_difference"
        for parameter in drift_parameters
    ]
    maximum_3000_drift = float(diagnostics[drift_columns].to_numpy(dtype=float).max())
    maximum_1500_drift = float(
        diagnostics_1500[drift_columns].to_numpy(dtype=float).max()
    )
    temporal = by_budget[3000]["temporal_lengthscale_years"]
    temporal_cv = float(temporal.std(ddof=1) / abs(temporal.mean()))

    lines = [
        "# Strong Annual ST-SVGP assessment",
        "",
        (
            "This assessment compares only the pre-2020 annual rolling folds. No final "
            "fit, five-year final test, or locked 2020-2025 annual target was used."
        ),
        "",
        "## 1. Parameter stability",
        "",
        (
            f"Maximum late-window drift changed from {maximum_1500_drift:.2%} at "
            f"1500 to {maximum_3000_drift:.2%} at 3000. The 3000 maximum is "
            f"{_drift_interpretation(maximum_3000_drift)}. Windows use their actual "
            "logged iterations and are descriptive, not an optimizer theorem."
        ),
        "",
        "| Fold | Temporal drift | Spatial x drift | Spatial y drift | Variance drift |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | "
            f"{row.temporal_lengthscale_years_relative_window_mean_difference:.2%} | "
            f"{row.spatial_lengthscale_x_km_relative_window_mean_difference:.2%} | "
            f"{row.spatial_lengthscale_y_km_relative_window_mean_difference:.2%} | "
            f"{row.kernel_variance_relative_window_mean_difference:.2%} |"
        )

    lines.extend(
        [
            "",
            "## 2. Temporal identifiability",
            "",
            (
                f"The 3000 fold temporal scales span {temporal.min():.6g} to "
                f"{temporal.max():.6g} years with CV {temporal_cv:.2%}."
            ),
            "",
            "## 3. Discrimination",
            "",
            "| Fold | Budget | Prevalence | PR-AUC | PR-AUC/prevalence | ROC-AUC |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {int(row.iterations)} | "
            f"{row.observed_prevalence:.6g} | {row.pr_auc:.6g} | "
            f"{row.pr_auc_prevalence_lift:.6g} | {row.roc_auc:.6g} |"
        )

    lines.extend(
        [
            "",
            "## 4. Probability quality",
            "",
            "| Fold | Budget | Log Loss | Brier | ECE | Probability bias |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {int(row.iterations)} | {row.log_loss:.6g} | "
            f"{row.brier_score:.6g} | {row.ece:.6g} | {row.probability_bias:.6g} |"
        )

    lines.extend(
        [
            "",
            "## 5. Calibration shape",
            "",
            (
                "Calibration slopes are assessed with Log Loss, Brier, ECE, and bias; "
                "ECE is not used alone."
            ),
            "",
            "| Fold | Slope 1000 | Slope 1500 | Slope 3000 | Intercept 3000 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in EXPECTED_FOLDS:
        lines.append(
            f"| {fold} | {by_budget[1000].loc[fold, 'calibration_slope']:.6g} | "
            f"{by_budget[1500].loc[fold, 'calibration_slope']:.6g} | "
            f"{by_budget[3000].loc[fold, 'calibration_slope']:.6g} | "
            f"{by_budget[3000].loc[fold, 'calibration_intercept']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## 6. Uncertainty",
            "",
            (
                "Mondrian positive coverage is compared with the 1000 run together "
                "with average set size and both-label cost. Fold 1 has no strictly "
                "earlier OOF calibration fold."
            ),
            "",
            "| Fold | Budget | Marginal coverage | Marginal positive | Mondrian coverage | Mondrian positive | Mondrian negative | Average size | Both labels | Empty |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.dropna(subset=["mondrian_empirical_coverage"]).itertuples(
        index=False
    ):
        lines.append(
            f"| {int(row.fold)} | {int(row.iterations)} | "
            f"{row.marginal_empirical_coverage:.6g} | "
            f"{row.marginal_positive_class_coverage:.6g} | "
            f"{row.mondrian_empirical_coverage:.6g} | "
            f"{row.mondrian_positive_class_coverage:.6g} | "
            f"{row.mondrian_negative_class_coverage:.6g} | "
            f"{row.mondrian_average_set_size:.6g} | "
            f"{row.mondrian_both_labels_rate:.6g} | "
            f"{row.mondrian_empty_set_rate:.6g} |"
        )

    lines.extend(
        [
            "",
            "## 7. ELBO",
            "",
            "| Fold | Earlier-late mean | Final mean | Final SD | All finite |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.earlier_late_window_mean_elbo:.6g} | "
            f"{row.final_window_mean_elbo:.6g} | "
            f"{row.final_window_elbo_standard_deviation:.6g} | "
            f"{bool(row.elbo_all_finite)} |"
        )
    lines.append(
        "All finite values indicate stochastic ELBO noise rather than numerical "
        "divergence." if evidence["finite_elbo"] else "Non-finite ELBO values indicate numerical divergence."
    )

    fold2_pr = [by_budget[budget].loc[2, "pr_auc"] for budget in EXPECTED_BUDGETS]
    fold2_roc = [by_budget[budget].loc[2, "roc_auc"] for budget in EXPECTED_BUDGETS]
    lines.extend(
        [
            "",
            "## 8. Fold robustness",
            "",
            (
                "The decision checks whether PR-AUC gains occur in at least two folds "
                "and whether difficult Fold 2 avoids joint material PR-AUC and ROC-AUC "
                "degradation. Fold 2 PR-AUC is "
                + " -> ".join(f"{value:.6g}" for value in fold2_pr)
                + "; ROC-AUC is "
                + " -> ".join(f"{value:.6g}" for value in fold2_roc)
                + "."
            ),
            "",
            "## Decision evidence",
            "",
        ]
    )
    for domain, supported in evidence.items():
        lines.append(f"- {domain.replace('_', ' ').title()}: {supported}")
    lines.extend(["", label, ""])
    return "\n".join(lines)


def evaluate_probability_and_coverage(
    config: Mapping[str, Any],
    paths: ExperimentPaths,
) -> None:
    """Run the retained marginal and Mondrian evaluators for the 3000 OOF file."""
    model = ModelInput(
        name="st_svgp_annual_convergence_3000",
        path=paths.predictions,
        target_column=str(config["dataset"]["target"]),
        target_year_column=str(config["dataset"]["target_year"]),
        maximum_target_year=int(config["development"]["maximum_target_year"]),
    )
    run_probability_evaluation(
        output_directory=paths.metrics_directory,
        n_bins=int(config["reporting"]["calibration_bins"]),
        strategy=str(config["reporting"]["calibration_strategy"]),
        models=[model],
    )
    run_annual_mondrian(
        output_directory=paths.metrics_directory,
        model=model,
        marginal_coverage_path=paths.marginal_coverage,
    )


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str | bool]:
    """Evaluate 3000 OOF predictions and write only isolated outputs."""
    validate_preflight(config_path)
    config = load_config(config_path)
    paths = experiment_paths(config)
    evaluate_probability_and_coverage(config, paths)

    comparison = build_optimization_budget_comparison(_read_budget_frames(paths))
    diagnostics = build_window_diagnostics(
        pd.read_csv(paths.history),
        step_years=float(config["time"]["step_years"]),
    )
    diagnostics_1500 = pd.read_csv(
        CONVERGENCE_1500_METRICS_DIRECTORY / "convergence_window_diagnostics.csv"
    )
    comparison.to_csv(paths.budget_comparison, index=False)
    diagnostics.to_csv(paths.window_diagnostics, index=False)
    paths.assessment.write_text(
        render_assessment(comparison, diagnostics, diagnostics_1500),
        encoding="utf-8",
    )
    return {
        "probability_metrics": str(paths.probability_metrics),
        "marginal_coverage": str(paths.marginal_coverage),
        "mondrian_coverage": str(paths.mondrian_coverage),
        "optimization_budget_comparison": str(paths.budget_comparison),
        "window_diagnostics": str(paths.window_diagnostics),
        "assessment": str(paths.assessment),
        "final_test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = validate_preflight(args.config) if args.preflight_only else run(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()