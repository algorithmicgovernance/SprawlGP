"""Compare the isolated annual 1500-iteration run with its 1000-run baseline."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from src.models.evaluation.evaluate_probabilities import ModelInput
from src.models.evaluation.evaluate_probabilities import run as run_probability_evaluation

DEFAULT_CONFIG_PATH = Path("configs/modeling/st_svgp_annual/convergence_1500.yaml")
BASELINE_METRICS_DIRECTORY = Path("reports/modeling/st_svgp_annual/metrics")
EXPECTED_FOLDS = (1, 2, 3)
EARLIER_LATE_WINDOW = (1000, 1200)
FINAL_LATE_WINDOW = (1300, 1500)
PRACTICAL_PARAMETER_DRIFT = 0.01
SLOW_PARAMETER_DRIFT = 0.05
DISCRIMINATION_CHANGE = 0.02
PROBABILITY_CHANGE = 0.02
CALIBRATION_SLOPE_CHANGE = 0.10


@dataclass(frozen=True)
class ArtifactPaths:
    """Canonical inputs and isolated convergence inputs/outputs."""

    baseline_parameters: Path
    baseline_probability_metrics: Path
    extended_parameters: Path
    extended_probability_metrics: Path
    extended_history: Path
    extended_predictions: Path
    comparison: Path
    window_diagnostics: Path
    report: Path


def load_config(path: Path) -> dict[str, Any]:
    """Load the convergence configuration."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def artifact_paths(
    config: dict[str, Any],
    baseline_metrics_directory: Path = BASELINE_METRICS_DIRECTORY,
) -> ArtifactPaths:
    """Resolve retained baseline inputs and isolated experiment artifacts."""
    reporting = config["reporting"]
    metrics_directory = Path(config["outputs"]["metrics_directory"])
    predictions_directory = Path(config["outputs"]["predictions_directory"])
    return ArtifactPaths(
        baseline_parameters=(
            baseline_metrics_directory
            / str(reporting["temporal_parameter_summary_filename"])
        ),
        baseline_probability_metrics=(
            baseline_metrics_directory / "probability_metrics_by_fold.csv"
        ),
        extended_parameters=(
            metrics_directory / str(reporting["temporal_parameter_summary_filename"])
        ),
        extended_probability_metrics=(metrics_directory / "probability_metrics_by_fold.csv"),
        extended_history=(metrics_directory / "st_svgp_training_history.csv"),
        extended_predictions=(predictions_directory / "st_svgp_oof_predictions.parquet"),
        comparison=(metrics_directory / "convergence_comparison.csv"),
        window_diagnostics=(metrics_directory / "convergence_window_diagnostics.csv"),
        report=(metrics_directory / "convergence_diagnostic.md"),
    )


def _validated_fold_table(
    frame: pd.DataFrame,
    required_columns: set[str],
    label: str,
) -> pd.DataFrame:
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")
    result = frame.copy()
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    if result["fold"].duplicated().any():
        raise ValueError(f"{label} contains duplicate folds.")
    folds = tuple(sorted(result["fold"].tolist()))
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"{label} must contain exactly folds {EXPECTED_FOLDS}; got {folds}.")
    return result.set_index("fold").sort_index()


def _relative_change(baseline: float, extended: float) -> float:
    if math.isclose(baseline, 0.0):
        return float("nan")
    return (extended - baseline) / abs(baseline)


def build_convergence_comparison(
    baseline_parameters: pd.DataFrame,
    extended_parameters: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    extended_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build one exact baseline-versus-extended row per rolling fold."""
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
        "log_loss",
        "brier_score",
        "pr_auc",
        "roc_auc",
        "ece",
        "calibration_slope",
        "probability_bias",
    }
    baseline_parameters_by_fold = _validated_fold_table(
        baseline_parameters, parameter_columns, "Baseline parameter summary"
    )
    extended_parameters_by_fold = _validated_fold_table(
        extended_parameters, parameter_columns, "Extended parameter summary"
    )
    baseline_metrics_by_fold = _validated_fold_table(
        baseline_metrics, metric_columns, "Baseline probability metrics"
    )
    extended_metrics_by_fold = _validated_fold_table(
        extended_metrics, metric_columns, "Extended probability metrics"
    )

    rows: list[dict[str, float | int]] = []
    for fold in EXPECTED_FOLDS:
        baseline_parameter_row = baseline_parameters_by_fold.loc[fold]
        extended_parameter_row = extended_parameters_by_fold.loc[fold]
        baseline_metric_row = baseline_metrics_by_fold.loc[fold]
        extended_metric_row = extended_metrics_by_fold.loc[fold]
        row: dict[str, float | int] = {"fold": fold}

        for parameter in (
            "temporal_lengthscale_years",
            "spatial_lengthscale_x_km",
            "spatial_lengthscale_y_km",
        ):
            baseline_value = float(baseline_parameter_row[parameter])
            extended_value = float(extended_parameter_row[parameter])
            output_name = parameter.removesuffix("_years").removesuffix("_km")
            row[f"baseline_{parameter}"] = baseline_value
            row[f"extended_{parameter}"] = extended_value
            row[f"relative_{output_name}_change"] = _relative_change(
                baseline_value, extended_value
            )

        for parameter in (
            "kernel_variance",
            "mean_latent_variance",
            "median_latent_variance",
        ):
            row[f"baseline_{parameter}"] = float(baseline_parameter_row[parameter])
            row[f"extended_{parameter}"] = float(extended_parameter_row[parameter])

        for metric in (
            "log_loss",
            "brier_score",
            "pr_auc",
            "roc_auc",
            "ece",
            "calibration_slope",
            "probability_bias",
        ):
            row[f"baseline_{metric}"] = float(baseline_metric_row[metric])
            row[f"extended_{metric}"] = float(extended_metric_row[metric])
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison = comparison.rename(
        columns={
            "relative_temporal_lengthscale_change": (
                "relative_temporal_lengthscale_change"
            ),
            "relative_spatial_lengthscale_x_change": "relative_spatial_x_change",
            "relative_spatial_lengthscale_y_change": "relative_spatial_y_change",
        }
    )
    return comparison


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
    """Summarize late parameter drift and stochastic ELBO by actual log points."""
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
        raise ValueError(f"Extended training history is missing: {', '.join(missing)}")

    numeric_history = history.copy()
    numeric_history["fold"] = pd.to_numeric(numeric_history["fold"], errors="raise").astype(int)
    numeric_history["iteration"] = pd.to_numeric(
        numeric_history["iteration"], errors="raise"
    ).astype(int)
    folds = tuple(sorted(numeric_history["fold"].unique().tolist()))
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"Extended history must contain exactly folds {EXPECTED_FOLDS}.")
    numeric_history["temporal_lengthscale_years"] = (
        pd.to_numeric(numeric_history["temporal_lengthscale_steps"], errors="raise")
        * step_years
    )

    parameter_columns = (
        "temporal_lengthscale_years",
        "spatial_lengthscale_x_km",
        "spatial_lengthscale_y_km",
        "kernel_variance",
    )
    rows: list[dict[str, float | int | bool]] = []
    for fold in EXPECTED_FOLDS:
        fold_history = numeric_history.loc[numeric_history["fold"] == fold]
        earlier = fold_history.loc[
            fold_history["iteration"].between(*earlier_window, inclusive="both")
        ]
        final = fold_history.loc[
            fold_history["iteration"].between(*final_window, inclusive="both")
        ]
        if earlier.empty or final.empty:
            raise ValueError(
                f"Fold {fold} has no logged points in one or both late windows."
            )

        row: dict[str, float | int | bool] = {
            "fold": fold,
            "earlier_window_first_iteration": int(earlier["iteration"].min()),
            "earlier_window_last_iteration": int(earlier["iteration"].max()),
            "earlier_window_logged_points": int(len(earlier)),
            "final_window_first_iteration": int(final["iteration"].min()),
            "final_window_last_iteration": int(final["iteration"].max()),
            "final_window_logged_points": int(len(final)),
        }
        for parameter in parameter_columns:
            earlier_mean = float(earlier[parameter].mean())
            final_mean = float(final[parameter].mean())
            row[f"{parameter}_earlier_window_mean"] = earlier_mean
            row[f"{parameter}_final_window_mean"] = final_mean
            row[f"{parameter}_relative_window_mean_difference"] = (
                _relative_window_difference(earlier_mean, final_mean)
            )

        earlier_elbo = pd.to_numeric(earlier["elbo_stochastic"], errors="raise")
        final_elbo = pd.to_numeric(final["elbo_stochastic"], errors="raise")
        row.update(
            {
                "earlier_late_window_mean_elbo": float(earlier_elbo.mean()),
                "final_window_mean_elbo": float(final_elbo.mean()),
                "final_window_elbo_standard_deviation": float(final_elbo.std(ddof=1)),
                "final_window_elbo_minimum": float(final_elbo.min()),
                "final_window_elbo_maximum": float(final_elbo.max()),
                "elbo_all_finite": bool(
                    np.isfinite(earlier_elbo).all() and np.isfinite(final_elbo).all()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _format_number(value: float) -> str:
    return f"{value:.8g}"


def _parameter_drift_description(value: float) -> str:
    if value <= PRACTICAL_PARAMETER_DRIFT:
        return "practically plateaued"
    if value <= SLOW_PARAMETER_DRIFT:
        return "slowly stabilizing"
    return "still strongly drifting"


def render_convergence_report(
    comparison: pd.DataFrame,
    window_diagnostics: pd.DataFrame,
) -> str:
    """Render the requested descriptive convergence assessment."""
    temporal_drift_column = (
        "temporal_lengthscale_years_relative_window_mean_difference"
    )
    spatial_x_drift_column = (
        "spatial_lengthscale_x_km_relative_window_mean_difference"
    )
    spatial_y_drift_column = (
        "spatial_lengthscale_y_km_relative_window_mean_difference"
    )
    variance_drift_column = "kernel_variance_relative_window_mean_difference"
    maximum_temporal_drift = float(window_diagnostics[temporal_drift_column].max())
    maximum_spatial_drift = float(
        window_diagnostics[[spatial_x_drift_column, spatial_y_drift_column]]
        .to_numpy(dtype=float)
        .max()
    )
    maximum_parameter_drift = float(
        window_diagnostics[
            [
                temporal_drift_column,
                spatial_x_drift_column,
                spatial_y_drift_column,
                variance_drift_column,
            ]
        ]
        .to_numpy(dtype=float)
        .max()
    )
    extended_temporal = comparison["extended_temporal_lengthscale_years"]
    temporal_cv = float(extended_temporal.std(ddof=1) / abs(extended_temporal.mean()))
    discrimination_change = max(
        float(
            (comparison[f"extended_{metric}"] - comparison[f"baseline_{metric}"])
            .abs()
            .max()
        )
        for metric in ("pr_auc", "roc_auc")
    )
    probability_changes = {
        metric: float(
            (comparison[f"extended_{metric}"] - comparison[f"baseline_{metric}"])
            .abs()
            .max()
        )
        for metric in (
            "log_loss",
            "brier_score",
            "ece",
            "calibration_slope",
            "probability_bias",
        )
    }
    probability_stable = all(
        value
        <= (CALIBRATION_SLOPE_CHANGE if metric == "calibration_slope" else PROBABILITY_CHANGE)
        for metric, value in probability_changes.items()
    )
    results_stable = discrimination_change <= DISCRIMINATION_CHANGE and probability_stable
    elbo_finite = bool(window_diagnostics["elbo_all_finite"].all())

    if maximum_parameter_drift <= PRACTICAL_PARAMETER_DRIFT and elbo_finite:
        status = "PARAMETERS_PRACTICALLY_STABILIZED"
    elif maximum_parameter_drift <= SLOW_PARAMETER_DRIFT and results_stable and elbo_finite:
        status = "PARAMETERS_STILL_SLOWLY_EVOLVING_BUT_RESULTS_STABLE"
    else:
        status = "PARAMETERS_NOT_STABILIZED"

    lines = [
        "# Annual ST-SVGP 1500-iteration convergence diagnostic",
        "",
        (
            "This is a convergence diagnostic only; no hyperparameter was selected or "
            "tuned using the locked block."
        ),
        "The locked 2020-2025 target block was not evaluated, and no final fit was run.",
        "",
        "## Descriptive conventions",
        "",
        (
            "Late-window relative parameter drift is the absolute difference between the "
            "1000-1200 and 1300-1500 window means, divided by the earlier-window mean, "
            "using actual logged iterations. Drift <=1% is described as practically "
            "plateaued, 1-5% as slowly stabilizing, and >5% as strongly drifting. These "
            "are descriptive review thresholds, not an optimizer convergence theorem."
        ),
        (
            "Absolute fold-level changes of 0.02 for PR-AUC/ROC-AUC and probability "
            "metrics, and 0.10 for calibration slope, are used only to flag material "
            "validation changes for review."
        ),
        "",
        "## Final parameters and validation metrics",
        "",
        (
            "| Fold | Temporal years (1000 / 1500) | Spatial x km (1000 / 1500) | "
            "Spatial y km (1000 / 1500) | Variance (1000 / 1500) | PR-AUC "
            "(1000 / 1500) | ROC-AUC (1000 / 1500) |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | "
            f"{_format_number(row.baseline_temporal_lengthscale_years)} / "
            f"{_format_number(row.extended_temporal_lengthscale_years)} | "
            f"{_format_number(row.baseline_spatial_lengthscale_x_km)} / "
            f"{_format_number(row.extended_spatial_lengthscale_x_km)} | "
            f"{_format_number(row.baseline_spatial_lengthscale_y_km)} / "
            f"{_format_number(row.extended_spatial_lengthscale_y_km)} | "
            f"{_format_number(row.baseline_kernel_variance)} / "
            f"{_format_number(row.extended_kernel_variance)} | "
            f"{_format_number(row.baseline_pr_auc)} / "
            f"{_format_number(row.extended_pr_auc)} | "
            f"{_format_number(row.baseline_roc_auc)} / "
            f"{_format_number(row.extended_roc_auc)} |"
        )

    lines.extend(
        [
            "",
            (
                "| Fold | Log Loss (1000 / 1500) | Brier (1000 / 1500) | ECE "
                "(1000 / 1500) | Calibration slope (1000 / 1500) | Probability "
                "bias (1000 / 1500) | Mean latent variance (1000 / 1500) |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.baseline_log_loss:.8g} / {row.extended_log_loss:.8g} | "
            f"{row.baseline_brier_score:.8g} / {row.extended_brier_score:.8g} | "
            f"{row.baseline_ece:.8g} / {row.extended_ece:.8g} | "
            f"{row.baseline_calibration_slope:.8g} / "
            f"{row.extended_calibration_slope:.8g} | "
            f"{row.baseline_probability_bias:.8g} / {row.extended_probability_bias:.8g} | "
            f"{row.baseline_mean_latent_variance:.8g} / "
            f"{row.extended_mean_latent_variance:.8g} |"
        )

    lines.extend(
        [
            "",
            "## Late-window parameter drift",
            "",
            (
                "| Fold | Actual earlier window | Actual final window | Temporal | "
                "Spatial x | Spatial y | Variance |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in window_diagnostics.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {int(row.earlier_window_first_iteration)}-"
            f"{int(row.earlier_window_last_iteration)} | "
            f"{int(row.final_window_first_iteration)}-{int(row.final_window_last_iteration)} | "
            f"{getattr(row, temporal_drift_column):.4%} | "
            f"{getattr(row, spatial_x_drift_column):.4%} | "
            f"{getattr(row, spatial_y_drift_column):.4%} | "
            f"{getattr(row, variance_drift_column):.4%} |"
        )

    lines.extend(
        [
            "",
            "## Stochastic ELBO window summary",
            "",
            "| Fold | Earlier late-window mean | Final-window mean | Final-window SD | Final min / max |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in window_diagnostics.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.earlier_late_window_mean_elbo:.8g} | "
            f"{row.final_window_mean_elbo:.8g} | "
            f"{row.final_window_elbo_standard_deviation:.8g} | "
            f"{row.final_window_elbo_minimum:.8g} / {row.final_window_elbo_maximum:.8g} |"
        )

    temporal_description = _parameter_drift_description(maximum_temporal_drift)
    spatial_description = _parameter_drift_description(maximum_spatial_drift)
    lines.extend(
        [
            "",
            "## Answers",
            "",
            (
                "1. **Temporal movement after iteration 1000.** The maximum fold-level "
                f"late-window drift is {maximum_temporal_drift:.4%}, classified as "
                f"{temporal_description}."
            ),
            (
                "2. **Spatial movement after iteration 1000.** The maximum x/y "
                f"late-window drift is {maximum_spatial_drift:.4%}, classified as "
                f"{spatial_description}."
            ),
            (
                "3. **Common temporal scale across folds.** Extended temporal "
                f"lengthscales range from {extended_temporal.min():.8g} to "
                f"{extended_temporal.max():.8g} years (CV {temporal_cv:.4%}); they "
                f"{'remain similar' if temporal_cv <= SLOW_PARAMETER_DRIFT else 'do not remain closely aligned'}."
            ),
            (
                "4. **Validation discrimination.** The largest absolute PR-AUC/ROC-AUC "
                f"change is {discrimination_change:.8g}; discrimination "
                f"{'did not materially change' if discrimination_change <= DISCRIMINATION_CHANGE else 'materially changed'}."
            ),
            (
                "5. **Probability quality.** Maximum absolute changes are "
                + ", ".join(
                    f"{metric}={value:.8g}" for metric, value in probability_changes.items()
                )
                + f"; probability quality {'did not materially change' if probability_stable else 'materially changed'}."
            ),
            (
                "6. **Stochastic ELBO stability.** "
                + (
                    "All late-window ELBO values are finite. The reported variability "
                    "is treated as minibatch noise; there is no non-finite evidence of "
                    "numerical divergence."
                    if elbo_finite
                    else "At least one late-window ELBO value is non-finite, which is evidence of numerical instability."
                )
            ),
            (
                "7. **Practical stabilization at 1500 iterations.** The descriptive "
                f"maximum parameter drift is {maximum_parameter_drift:.4%}; the status "
                "below records the evidence without changing the canonical iteration count."
            ),
            "",
            status,
            "",
        ]
    )
    return "\n".join(lines)


def run(
    config_path: Path = DEFAULT_CONFIG_PATH,
    baseline_metrics_directory: Path = BASELINE_METRICS_DIRECTORY,
) -> dict[str, str]:
    """Evaluate isolated predictions, compare artifacts, and write diagnostics."""
    config = load_config(config_path)
    if int(config["training"]["iterations"]) != 1500:
        raise ValueError("The convergence diagnostic requires exactly 1500 iterations.")
    if int(config["development"]["maximum_target_year"]) >= 2020:
        raise ValueError("The convergence diagnostic must keep target years below 2020.")
    if bool(config["locked_block"]["evaluate"]):
        raise ValueError("The locked block must not be evaluated.")

    paths = artifact_paths(config, baseline_metrics_directory)
    run_probability_evaluation(
        output_directory=paths.extended_probability_metrics.parent,
        n_bins=int(config["reporting"]["calibration_bins"]),
        strategy=str(config["reporting"]["calibration_strategy"]),
        models=[
            ModelInput(
                name="st_svgp_annual_convergence_1500",
                path=paths.extended_predictions,
                target_column=str(config["dataset"]["target"]),
                target_year_column=str(config["dataset"]["target_year"]),
                maximum_target_year=int(config["development"]["maximum_target_year"]),
            )
        ],
    )

    comparison = build_convergence_comparison(
        pd.read_csv(paths.baseline_parameters),
        pd.read_csv(paths.extended_parameters),
        pd.read_csv(paths.baseline_probability_metrics),
        pd.read_csv(paths.extended_probability_metrics),
    )
    window_diagnostics = build_window_diagnostics(
        pd.read_csv(paths.extended_history),
        step_years=float(config["time"]["step_years"]),
    )
    comparison.to_csv(paths.comparison, index=False)
    window_diagnostics.to_csv(paths.window_diagnostics, index=False)
    paths.report.write_text(
        render_convergence_report(comparison, window_diagnostics),
        encoding="utf-8",
    )
    return {
        "comparison": str(paths.comparison),
        "window_diagnostics": str(paths.window_diagnostics),
        "report": str(paths.report),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    outputs = run(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()