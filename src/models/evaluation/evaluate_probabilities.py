"""Evaluate probability quality, calibration and conformal coverage from OOF files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.evaluation.calibration import calibration_metrics, reliability_table
from src.models.evaluation.prediction_sets import temporal_prediction_set_coverage
from src.models.metrics import probabilistic_metrics


TARGET_CANDIDATES = (
    "target_transition_5y",
    "target",
    "y_true",
)

PROBABILITY_CANDIDATES = (
    "probability_raw",
    "predicted_probability",
    "probability",
)


@dataclass(frozen=True)
class ModelInput:
    """One model OOF prediction artifact."""

    name: str
    path: Path


def resolve_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Resolve target/probability names across baseline artifacts."""
    target = next((col for col in TARGET_CANDIDATES if col in frame.columns), None)
    probability = next(
        (col for col in PROBABILITY_CANDIDATES if col in frame.columns),
        None,
    )

    if target is None or probability is None:
        raise ValueError(
            "Could not resolve target/probability columns. "
            f"Columns={list(frame.columns)}"
        )
    return target, probability


def evaluate_model(
    model: ModelInput,
    n_bins: int,
    strategy: str,
    target_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fold metrics, reliability bins and temporal coverage diagnostics."""
    if not model.path.is_file():
        raise FileNotFoundError(f"OOF predictions not found: {model.path}")

    frame = pd.read_parquet(model.path)
    target_col, probability_col = resolve_columns(frame)

    if "fold" in frame.columns:
        groups: list[tuple[object, pd.DataFrame]] = list(frame.groupby("fold"))
    else:
        groups = [("all", frame)]

    metric_rows: list[dict[str, object]] = []
    reliability_rows: list[pd.DataFrame] = []

    for fold, part in groups:
        y = part[target_col].to_numpy(dtype=int)
        p = part[probability_col].to_numpy(dtype=float)

        row = {
            "model": model.name,
            "fold": fold,
            **probabilistic_metrics(y, p),
            **calibration_metrics(y, p, n_bins=n_bins, strategy=strategy),
        }
        metric_rows.append(row)

        bins = reliability_table(y, p, n_bins=n_bins, strategy=strategy)
        bins["model"] = model.name
        bins["fold"] = fold
        reliability_rows.append(bins)

    fold_metrics = pd.DataFrame(metric_rows)
    reliability = pd.concat(reliability_rows, ignore_index=True)

    if "fold" in frame.columns:
        coverage = temporal_prediction_set_coverage(
            frame=frame,
            target_column=target_col,
            probability_column=probability_col,
            fold_column="fold",
            target_coverage=target_coverage,
        )
    else:
        coverage = pd.DataFrame()

    if not coverage.empty:
        coverage.insert(0, "model", model.name)

    return fold_metrics, reliability, coverage


def build_summary(
    fold_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate metrics by model for concise comparison."""
    metrics_summary = (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_ece=("ece", "mean"),
            mean_calibration_intercept=("calibration_intercept", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            mean_probability_bias=("probability_bias", "mean"),
        )
    )

    if coverage.empty:
        metrics_summary["mean_empirical_coverage"] = float("nan")
        metrics_summary["mean_average_set_size"] = float("nan")
        metrics_summary["mean_singleton_rate"] = float("nan")
        metrics_summary["mean_absolute_coverage_gap"] = float("nan")
        return metrics_summary

    coverage_summary = (
        coverage.groupby("model", as_index=False)
        .agg(
            coverage_folds=("fold", "nunique"),
            mean_empirical_coverage=("empirical_coverage", "mean"),
            mean_absolute_coverage_gap=("absolute_coverage_gap", "mean"),
            mean_average_set_size=("average_set_size", "mean"),
            mean_singleton_rate=("singleton_rate", "mean"),
            mean_both_labels_rate=("both_labels_rate", "mean"),
            mean_empty_set_rate=("empty_set_rate", "mean"),
            mean_positive_class_coverage=("positive_class_coverage", "mean"),
            mean_negative_class_coverage=("negative_class_coverage", "mean"),
        )
    )

    return metrics_summary.merge(coverage_summary, on="model", how="left")


def run(
    output_directory: Path,
    target_coverage: float = 0.80,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict[str, str]:
    """Run evaluation for LR and selected SVGP-Adam from saved OOF predictions."""
    output_directory.mkdir(parents=True, exist_ok=True)

    models = [
        ModelInput(
            name="logistic_regression",
            path=Path(
                "reports/modeling/logistic_regression/predictions/"
                "logistic_oof_predictions.parquet"
            ),
        ),
        ModelInput(
            name="svgp_adam",
            path=Path(
                "reports/modeling/svgp/predictions/"
                "svgp_oof_predictions.parquet"
            ),
        ),
    ]

    fold_tables = []
    reliability_tables = []
    coverage_tables = []

    for model in models:
        fold_metrics, reliability, coverage = evaluate_model(
            model=model,
            n_bins=n_bins,
            strategy=strategy,
            target_coverage=target_coverage,
        )
        fold_tables.append(fold_metrics)
        reliability_tables.append(reliability)
        if not coverage.empty:
            coverage_tables.append(coverage)

    fold_metrics = pd.concat(fold_tables, ignore_index=True)
    reliability = pd.concat(reliability_tables, ignore_index=True)
    coverage = (
        pd.concat(coverage_tables, ignore_index=True)
        if coverage_tables
        else pd.DataFrame()
    )
    summary = build_summary(fold_metrics, coverage)

    summary_path = output_directory / "probability_evaluation_summary.csv"
    fold_path = output_directory / "probability_metrics_by_fold.csv"
    reliability_path = output_directory / "reliability_bins.csv"
    coverage_path = output_directory / "prediction_set_coverage_80.csv"
    manifest_path = output_directory / "evaluation_manifest.json"

    summary.to_csv(summary_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    reliability.to_csv(reliability_path, index=False)
    if coverage.empty:
        pd.DataFrame(
            columns=[
                "model",
                "fold",
                "calibration_rows",
                "evaluation_rows",
                "q_hat",
                "target_coverage",
                "empirical_coverage",
                "coverage_gap",
                "absolute_coverage_gap",
                "average_set_size",
                "singleton_rate",
                "both_labels_rate",
                "empty_set_rate",
                "positive_class_coverage",
                "negative_class_coverage",
            ]
        ).to_csv(coverage_path, index=False)
    else:
        coverage.to_csv(coverage_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_coverage": float(target_coverage),
        "binning": {
            "n_bins": int(n_bins),
            "strategy": strategy,
        },
        "models": [
            {
                "name": model.name,
                "oof_path": str(model.path),
            }
            for model in models
        ],
        "outputs": {
            "summary": str(summary_path),
            "metrics_by_fold": str(fold_path),
            "reliability_bins": str(reliability_path),
            "prediction_set_coverage": str(coverage_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "summary": str(summary_path),
        "metrics_by_fold": str(fold_path),
        "reliability_bins": str(reliability_path),
        "prediction_set_coverage": str(coverage_path),
        "manifest": str(manifest_path),
    }


def main() -> None:
    """CLI entry-point used by Makefile target modeling-probability-evaluation."""
    paths = run(output_directory=Path("reports/modeling/calibration"))
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
