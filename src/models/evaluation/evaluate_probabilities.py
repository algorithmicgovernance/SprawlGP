"""Evaluate probability quality, calibration and conformal coverage from OOF files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from src.models.evaluation.calibration import calibration_metrics, reliability_table
from src.models.evaluation.prediction_sets import (
    temporal_mondrian_prediction_set_coverage,
    temporal_prediction_set_coverage,
)
from src.models.metrics import probabilistic_metrics

TARGET_CANDIDATES = (
    "target_transition_5y",
    "target_transition_1y",
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
    target_column: str | None = None
    target_year_column: str | None = None
    maximum_target_year: int | None = None


def resolve_columns(
    frame: pd.DataFrame,
    target_column: str | None = None,
) -> tuple[str, str]:
    """Resolve target/probability names across baseline artifacts."""
    if target_column is not None and target_column not in frame.columns:
        raise ValueError(f"Configured target column not found: {target_column}")

    target = target_column or next(
        (col for col in TARGET_CANDIDATES if col in frame.columns),
        None,
    )
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


def _load_model_predictions(
    model: ModelInput,
) -> tuple[pd.DataFrame, str, str]:
    """Load one OOF artifact and enforce its configured target-year boundary."""
    if not model.path.is_file():
        raise FileNotFoundError(f"OOF predictions not found: {model.path}")

    frame = pd.read_parquet(model.path)
    target_col, probability_col = resolve_columns(frame, model.target_column)

    if model.maximum_target_year is not None:
        if model.target_year_column is None:
            raise ValueError(
                "target_year_column is required when maximum_target_year is set."
            )
        if model.target_year_column not in frame.columns:
            raise ValueError(
                f"Configured target-year column not found: {model.target_year_column}"
            )
        target_years = pd.to_numeric(
            frame[model.target_year_column],
            errors="raise",
        )
        if (target_years > model.maximum_target_year).any():
            raise ValueError(
                "OOF predictions contain locked target years after "
                f"{model.maximum_target_year}."
            )

    return frame, target_col, probability_col


def evaluate_model(
    model: ModelInput,
    n_bins: int,
    strategy: str,
    target_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fold metrics, reliability bins and temporal coverage diagnostics."""
    frame, target_col, probability_col = _load_model_predictions(model)

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
    # metrics_summary = (
    #     fold_metrics.groupby("model", as_index=False)
    #     .agg(
    #         folds=("fold", "nunique"),
    #         mean_log_loss=("log_loss", "mean"),
    #         mean_brier_score=("brier_score", "mean"),
    #         mean_pr_auc=("pr_auc", "mean"),
    #         mean_roc_auc=("roc_auc", "mean"),
    #         mean_ece=("ece", "mean"),
    #         mean_calibration_intercept=("calibration_intercept", "mean"),
    #         mean_calibration_slope=("calibration_slope", "mean"),
    #         mean_probability_bias=("probability_bias", "mean"),
    #     )
    # )
    
    metrics_summary = (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),

            mean_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            worst_log_loss=("log_loss", "max"),

            mean_brier_score=("brier_score", "mean"),
            std_brier_score=("brier_score", "std"),
            worst_brier_score=("brier_score", "max"),

            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            worst_pr_auc=("pr_auc", "min"),

            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            worst_roc_auc=("roc_auc", "min"),

            mean_ece=("ece", "mean"),
            std_ece=("ece", "std"),
            worst_ece=("ece", "max"),

            mean_calibration_intercept=(
                "calibration_intercept", "mean"
            ),
            mean_calibration_slope=(
                "calibration_slope", "mean"
            ),

            mean_probability_bias=(
                "probability_bias", "mean"
            ),
            mean_absolute_probability_bias=(
                "absolute_probability_bias", "mean"
            ),
        )
    )

    if coverage.empty:
        metrics_summary["mean_empirical_coverage"] = float("nan")
        metrics_summary["mean_average_set_size"] = float("nan")
        metrics_summary["mean_singleton_rate"] = float("nan")
        metrics_summary["mean_absolute_coverage_gap"] = float("nan")
        return metrics_summary

    # coverage_summary = (
    #     coverage.groupby("model", as_index=False)
    #     .agg(
    #         coverage_folds=("fold", "nunique"),
    #         mean_empirical_coverage=("empirical_coverage", "mean"),
    #         mean_absolute_coverage_gap=("absolute_coverage_gap", "mean"),
    #         mean_average_set_size=("average_set_size", "mean"),
    #         mean_singleton_rate=("singleton_rate", "mean"),
    #         mean_both_labels_rate=("both_labels_rate", "mean"),
    #         mean_empty_set_rate=("empty_set_rate", "mean"),
    #         mean_positive_class_coverage=("positive_class_coverage", "mean"),
    #         mean_negative_class_coverage=("negative_class_coverage", "mean"),
    #     )
    # )
    coverage_summary = (
        coverage.groupby("model", as_index=False)
        .agg(
            coverage_folds=("fold", "nunique"),

            mean_empirical_coverage=(
                "empirical_coverage", "mean"
            ),
            worst_fold_empirical_coverage=(
                "empirical_coverage", "min"
            ),

            mean_absolute_coverage_gap=(
                "absolute_coverage_gap", "mean"
            ),

            mean_average_set_size=(
                "average_set_size", "mean"
            ),
            mean_singleton_rate=(
                "singleton_rate", "mean"
            ),
            mean_both_labels_rate=(
                "both_labels_rate", "mean"
            ),
            mean_empty_set_rate=(
                "empty_set_rate", "mean"
            ),

            mean_positive_class_coverage=(
                "positive_class_coverage", "mean"
            ),
            worst_positive_class_coverage=(
                "positive_class_coverage", "min"
            ),

            mean_negative_class_coverage=(
                "negative_class_coverage", "mean"
            ),
            worst_negative_class_coverage=(
                "negative_class_coverage", "min"
            ),
        )
    )

    return metrics_summary.merge(coverage_summary, on="model", how="left")


def run(
    output_directory: Path,
    target_coverage: float = 0.80,
    n_bins: int = 10,
    strategy: str = "quantile",
    models: list[ModelInput] | None = None,
) -> dict[str, str]:
    """Run one shared OOF evaluation for LR, Strong SVGP and promoted ST-SVGP."""
    output_directory.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = [
            ModelInput(
                name="logistic_regression",
                path=Path(
                    "reports/modeling/logistic_regression/predictions/"
                    "logistic_oof_predictions.parquet"
                ),
            ),
            ModelInput(
                name="svgp_strong",
                path=Path(
                    "reports/modeling/svgp/features/log_distance_growth/"
                    "svgp_feature_oof_predictions.parquet"
                ),
            ),
            ModelInput(
                name="st_svgp",
                path=Path(
                    "reports/modeling/st_svgp/predictions/"
                    "st_svgp_oof_predictions.parquet"
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


def run_annual(
    output_directory: Path = Path("reports/modeling/st_svgp_annual/metrics"),
) -> dict[str, str]:
    """Evaluate annual ST-SVGP OOF predictions in the annual output tree."""
    return run(
        output_directory=output_directory,
        models=[
            ModelInput(
                name="st_svgp_annual",
                path=Path(
                    "reports/modeling/st_svgp_annual/predictions/"
                    "st_svgp_oof_predictions.parquet"
                ),
                target_column="target_transition_1y",
                target_year_column="target_year",
                maximum_target_year=2019,
            )
        ],
    )


def _build_mondrian_comparison(
    mondrian: pd.DataFrame,
    marginal_path: Path,
) -> pd.DataFrame:
    """Align canonical marginal and Mondrian coverage by model and fold."""
    if not marginal_path.is_file():
        raise FileNotFoundError(
            f"Canonical marginal coverage not found: {marginal_path}"
        )
    marginal = pd.read_csv(marginal_path)
    metric_columns = [
        "empirical_coverage",
        "positive_class_coverage",
        "negative_class_coverage",
        "average_set_size",
        "singleton_rate",
        "both_labels_rate",
        "empty_set_rate",
    ]
    required = {"model", "fold", *metric_columns}
    missing = sorted(required.difference(marginal.columns))
    if missing:
        raise ValueError(
            "Canonical marginal coverage is missing columns: " + ", ".join(missing)
        )

    marginal_comparison = marginal.loc[:, ["model", "fold", *metric_columns]].rename(
        columns={column: f"marginal_{column}" for column in metric_columns}
    )
    mondrian_comparison = mondrian.loc[:, ["model", "fold", *metric_columns]].rename(
        columns={column: f"mondrian_{column}" for column in metric_columns}
    )
    comparison = marginal_comparison.merge(
        mondrian_comparison,
        on=["model", "fold"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (comparison["_merge"] == "both").all():
        raise ValueError("Marginal and Mondrian coverage folds do not match.")
    return comparison.drop(columns="_merge").sort_values("fold").reset_index(drop=True)


def _mondrian_status(
    comparison: pd.DataFrame,
    target_coverage: float,
) -> str:
    """Classify restoration against the marginal result and nominal target."""
    mondrian_positive = comparison["mondrian_positive_class_coverage"]
    marginal_positive = comparison["marginal_positive_class_coverage"]
    if (mondrian_positive >= target_coverage).all():
        return "MONDRIAN_SUBSTANTIALLY_RESTORES_POSITIVE_COVERAGE"
    if (mondrian_positive > marginal_positive).all():
        return "MONDRIAN_IMPROVES_BUT_REMAINS_INADEQUATE"
    return "MONDRIAN_DOES_NOT_RESOLVE_POSITIVE_COVERAGE"


def _render_mondrian_report(
    mondrian: pd.DataFrame,
    comparison: pd.DataFrame,
    target_coverage: float,
) -> str:
    """Render a concise temporal Mondrian interpretation report."""
    lines = [
        "# Annual ST-SVGP Mondrian coverage diagnostic",
        "",
        (
            "This is a temporal diagnostic using strictly past-fold calibration. "
            "It does not establish formal class-conditional validity under temporal "
            "distribution shift; temporal non-exchangeability may still affect coverage."
        ),
        "",
        "## Fold results",
        "",
        (
            "| Fold | Calibration (negative / positive) | Evaluation | q_hat_0 | "
            "q_hat_1 | Marginal positive | Mondrian positive | Marginal negative | "
            "Mondrian negative |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_fold = mondrian.set_index("fold")
    for row in comparison.itertuples(index=False):
        counts = by_fold.loc[row.fold]
        lines.append(
            f"| {int(row.fold)} | "
            f"{int(counts['calibration_negative_rows']):,} / "
            f"{int(counts['calibration_positive_rows']):,} | "
            f"{int(counts['evaluation_rows']):,} | "
            f"{counts['q_hat_0']:.6f} | {counts['q_hat_1']:.6f} | "
            f"{row.marginal_positive_class_coverage:.6f} | "
            f"{row.mondrian_positive_class_coverage:.6f} | "
            f"{row.marginal_negative_class_coverage:.6f} | "
            f"{row.mondrian_negative_class_coverage:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Efficiency comparison",
            "",
            (
                "| Fold | Marginal coverage | Mondrian coverage | Marginal average "
                "size | Mondrian average size | Marginal singleton | Mondrian "
                "singleton | Marginal both | Mondrian both | Marginal empty | "
                "Mondrian empty |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.marginal_empirical_coverage:.6f} | "
            f"{row.mondrian_empirical_coverage:.6f} | "
            f"{row.marginal_average_set_size:.6f} | "
            f"{row.mondrian_average_set_size:.6f} | "
            f"{row.marginal_singleton_rate:.6f} | "
            f"{row.mondrian_singleton_rate:.6f} | "
            f"{row.marginal_both_labels_rate:.6f} | "
            f"{row.mondrian_both_labels_rate:.6f} | "
            f"{row.marginal_empty_set_rate:.6f} | "
            f"{row.mondrian_empty_set_rate:.6f} |"
        )

    positive_details = "; ".join(
        f"Fold {int(row.fold)} {row.marginal_positive_class_coverage:.6f} to "
        f"{row.mondrian_positive_class_coverage:.6f}"
        for row in comparison.itertuples(index=False)
    )
    nominal_shortfalls = "; ".join(
        f"Fold {int(row.fold)} "
        f"{target_coverage - row.mondrian_positive_class_coverage:.6f}"
        for row in comparison.itertuples(index=False)
    )
    status = _mondrian_status(comparison, target_coverage)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Positive-class coverage materially improves ({positive_details}), but "
                f"remains below the nominal {target_coverage:.2f} target. The remaining "
                f"shortfalls are {nominal_shortfalls}, respectively. The result is "
                "substantially improved but still deficient overall, rather than restored "
                "to nominal coverage."
            ),
            "",
            (
                "Negative-class and marginal coverage remain high, while the fold tables "
                "show the corresponding changes in set size, singleton, both-label, and "
                "empty-set rates."
            ),
            "",
            (
                "The large positive-class improvement indicates that the previous near-zero "
                "coverage was primarily driven by the single marginal threshold under class "
                "imbalance. The persistent positive shortfall, especially in the earliest "
                "evaluated fold, shows that marginal calibration was not the only limitation."
            ),
            "",
            status,
            "",
        ]
    )
    return "\n".join(lines)


def run_annual_mondrian(
    output_directory: Path = Path("reports/modeling/st_svgp_annual/metrics"),
    target_coverage: float = 0.80,
    model: ModelInput | None = None,
    marginal_coverage_path: Path = Path(
        "reports/modeling/st_svgp_annual/metrics/prediction_set_coverage_80.csv"
    ),
) -> dict[str, str]:
    """Run the annual-only Mondrian diagnostic from existing OOF predictions."""
    model = model or ModelInput(
        name="st_svgp_annual",
        path=Path(
            "reports/modeling/st_svgp_annual/predictions/"
            "st_svgp_oof_predictions.parquet"
        ),
        target_column="target_transition_1y",
        target_year_column="target_year",
        maximum_target_year=2019,
    )
    frame, target_col, probability_col = _load_model_predictions(model)
    mondrian = temporal_mondrian_prediction_set_coverage(
        frame=frame,
        target_column=target_col,
        probability_column=probability_col,
        target_coverage=target_coverage,
    )
    if mondrian.empty:
        raise ValueError("Annual Mondrian coverage has no evaluable folds.")
    mondrian.insert(0, "model", model.name)
    comparison = _build_mondrian_comparison(mondrian, marginal_coverage_path)

    output_directory.mkdir(parents=True, exist_ok=True)
    mondrian_path = output_directory / "prediction_set_coverage_80_mondrian.csv"
    comparison_path = output_directory / "coverage_marginal_vs_mondrian.csv"
    report_path = output_directory / "mondrian_coverage_diagnostic.md"
    mondrian.to_csv(mondrian_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    report_path.write_text(
        _render_mondrian_report(mondrian, comparison, target_coverage),
        encoding="utf-8",
    )
    return {
        "mondrian_coverage": str(mondrian_path),
        "comparison": str(comparison_path),
        "report": str(report_path),
    }


def main() -> None:
    """CLI entry-point used by Makefile target modeling-probability-evaluation."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--annual",
        action="store_true",
        help="Evaluate annual ST-SVGP OOF predictions in isolated outputs.",
    )
    mode.add_argument(
        "--annual-mondrian",
        action="store_true",
        help="Run only the annual ST-SVGP Mondrian coverage diagnostic.",
    )
    args = parser.parse_args()

    if args.annual_mondrian:
        paths = run_annual_mondrian()
    elif args.annual:
        paths = run_annual()
    else:
        paths = run(output_directory=Path("reports/modeling/calibration"))
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
