"""Feature-only SVGP experiments using the locked selected Adam implementation.

This module imports the current SVGP-Adam pipeline and reuses evaluate_fold().
It does not reimplement GP, Adam, or kernel construction.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.feature_engineering.urban_expansion import add_candidate_features, predictors_for
from src.models.evaluation.calibration import calibration_metrics, reliability_table
from src.models.evaluation.prediction_sets import temporal_prediction_set_coverage
from src.models.train_svgp import (
    add_log_population,
    configure_runtime,
    evaluate_fold,
    load_config,
    load_dataset,
    summarise_folds,
    validate_temporal_contract,
)


def _output_root(feature_set: str) -> Path:
    return Path("reports/modeling/svgp/features") / feature_set


def _build_config_for_feature_set(
    base_config: dict[str, Any],
    feature_set: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["linear_predictors"] = list(predictors_for(feature_set))
    return config


def _calibration_and_reliability(
    predictions: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_rows: list[dict[str, float | int]] = []
    reliability_parts: list[pd.DataFrame] = []

    for fold, part in predictions.groupby("fold"):
        metrics = calibration_metrics(
            part[target_column].to_numpy(dtype=int),
            part["probability_raw"].to_numpy(dtype=float),
            n_bins=10,
            strategy="quantile",
        )
        calibration_rows.append(
            {
                "fold": int(fold),
                **metrics,
            }
        )

        bins = reliability_table(
            part[target_column].to_numpy(dtype=int),
            part["probability_raw"].to_numpy(dtype=float),
            n_bins=10,
            strategy="quantile",
        )
        bins["fold"] = int(fold)
        reliability_parts.append(bins)

    return (
        pd.DataFrame(calibration_rows),
        pd.concat(reliability_parts, ignore_index=True),
    )


def run_feature_experiment(
    feature_set: str,
    config_path: Path,
) -> dict[str, str]:
    """Run SVGP rolling validation with one explicit feature-set change only."""


    
    base_config = load_config(config_path)

    runtime = configure_runtime(base_config)
    validate_temporal_contract(base_config)

    # 1. Load only the frozen/raw columns expected by the selected SVGP.
    frame = load_dataset(base_config)

    # 2. Create the already-approved population transformation.
    frame = add_log_population(
        frame,
        base_config,
    )

    # 3. Create all experimental feature candidates OUTSIDE the model.
    frame = add_candidate_features(frame)

    # 4. Only now tell evaluate_fold which candidate columns to use.
    config = _build_config_for_feature_set(
        base_config,
        feature_set,
    )

    target_column = str(
        base_config["dataset"]["target"]
    )

    output_root = _output_root(feature_set)
    output_root.mkdir(parents=True, exist_ok=True)

    metric_records = []
    prediction_frames = []
    history_frames = []


    for fold_index, fold in enumerate(config["rolling_validation"]["folds"], start=1):
        record, predictions, history = evaluate_fold(
            frame,
            config,
            fold_index,
            fold,
        )
        metric_records.append(record)
        prediction_frames.append(predictions)
        history_frames.append(history)

    fold_metrics = pd.DataFrame(metric_records)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    training_history = pd.concat(history_frames, ignore_index=True)

    # calibration_by_fold, reliability = _calibration_and_reliability(
    #     predictions=oof_predictions,
    #     target_column=target_column,
    # )
    # fold_metrics = fold_metrics.merge(calibration_by_fold, on="fold", how="left")
    calibration_by_fold, reliability = _calibration_and_reliability(
        predictions=oof_predictions,
        target_column=target_column,
    )

    # evaluate_fold() already provides probability_bias and the core
    # probabilistic metrics. Merge only the additional calibration diagnostics
    # required by the feature-engineering experiment.
    calibration_columns = [
        "fold",
        "ece",
        "calibration_intercept",
        "calibration_slope",
        "absolute_probability_bias",
    ]

    missing_calibration_columns = [
        column
        for column in calibration_columns
        if column not in calibration_by_fold.columns
    ]

    if missing_calibration_columns:
        raise ValueError(
            "Missing calibration diagnostics: "
            + ", ".join(missing_calibration_columns)
        )

    fold_metrics = fold_metrics.merge(
        calibration_by_fold[calibration_columns],
        on="fold",
        how="left",
        validate="one_to_one",
    )

    coverage = temporal_prediction_set_coverage(
        frame=oof_predictions,
        target_column=target_column,
        probability_column="probability_raw",
        fold_column="fold",
        target_coverage=0.80,
    )
    
    required_summary_columns = {
        "log_loss",
        "brier_score",
        "pr_auc",
        "roc_auc",
        "probability_bias",
        "ece",
        "calibration_intercept",
        "calibration_slope",
        "absolute_probability_bias",
    }

    missing_summary_columns = sorted(
        required_summary_columns.difference(fold_metrics.columns)
    )

    if missing_summary_columns:
        raise ValueError(
            "Feature experiment is missing required summary metrics: "
            + ", ".join(missing_summary_columns)
        )

    summary = summarise_folds(fold_metrics)
    summary["feature_set"] = feature_set
    summary["mean_ece"] = float(fold_metrics["ece"].mean())
    summary["mean_calibration_intercept"] = float(
        fold_metrics["calibration_intercept"].mean()
    )
    summary["mean_calibration_slope"] = float(
        fold_metrics["calibration_slope"].mean()
    )
    summary["mean_absolute_probability_bias"] = float(
        fold_metrics["absolute_probability_bias"].mean()
    )

    if coverage.empty:
        summary["mean_empirical_coverage"] = float("nan")
        summary["mean_average_set_size"] = float("nan")
        summary["mean_singleton_rate"] = float("nan")
        summary["mean_positive_class_coverage"] = float("nan")
        summary["mean_negative_class_coverage"] = float("nan")
        summary["worst_fold_empirical_coverage"] = float("nan")
    else:
        summary["mean_empirical_coverage"] = float(coverage["empirical_coverage"].mean())
        summary["mean_average_set_size"] = float(coverage["average_set_size"].mean())
        summary["mean_singleton_rate"] = float(coverage["singleton_rate"].mean())
        summary["mean_positive_class_coverage"] = float(
            coverage["positive_class_coverage"].mean()
        )
        summary["mean_negative_class_coverage"] = float(
            coverage["negative_class_coverage"].mean()
        )
        summary["worst_fold_empirical_coverage"] = float(
            coverage["empirical_coverage"].min()
        )

    metrics_path = output_root / "svgp_feature_validation_metrics.csv"
    summary_path = output_root / "svgp_feature_validation_summary.csv"
    reliability_path = output_root / "svgp_feature_reliability_bins.csv"
    coverage_path = output_root / "svgp_feature_prediction_set_coverage.csv"
    history_path = output_root / "svgp_feature_training_history.csv"
    oof_path = output_root / "svgp_feature_oof_predictions.parquet"
    manifest_path = output_root / "svgp_feature_manifest.json"

    fold_metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    reliability.to_csv(reliability_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    training_history.to_csv(history_path, index=False)
    oof_predictions.to_parquet(oof_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": feature_set,
        "config_path": str(config_path),
        "svgp_model_source": "src/models/train_svgp.py",
        "evaluate_fold_reused": True,
        "final_fit_executed": False,
        "runtime": runtime,
        "linear_predictors": list(config["linear_predictors"]),
        "final_test_locked_origins": list(config["final_test"]["origins"]),
        "outputs": {
            "validation_metrics": str(metrics_path),
            "validation_summary": str(summary_path),
            "reliability_bins": str(reliability_path),
            "prediction_set_coverage": str(coverage_path),
            "training_history": str(history_path),
            "oof_predictions": str(oof_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "feature_set": feature_set,
        "validation_metrics": str(metrics_path),
        "validation_summary": str(summary_path),
        "reliability_bins": str(reliability_path),
        "prediction_set_coverage": str(coverage_path),
        "training_history": str(history_path),
        "oof_predictions": str(oof_path),
        "manifest": str(manifest_path),
    }


def build_feature_summary(base_directory: Path) -> Path:
    """Compare all completed feature experiments in one table."""
    roots = sorted(path for path in base_directory.iterdir() if path.is_dir())

    frames = []
    for root in roots:
        summary_path = root / "svgp_feature_validation_summary.csv"
        if summary_path.is_file():
            frame = pd.read_csv(summary_path)
            frame["feature_set"] = frame.get("feature_set", root.name)
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No feature summary files found under "
            f"{base_directory}. Run modeling-svgp-feature first."
        )

    comparison = pd.concat(frames, ignore_index=True)
    comparison_path = base_directory / "feature_experiment_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    return comparison_path


def main() -> None:
    """Run one feature experiment or summarize completed experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modeling/svgp_experiment.yaml"),
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="base",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
    )
    args = parser.parse_args()

    if args.summarize:
        comparison_path = build_feature_summary(Path("reports/modeling/svgp/features"))
        print(json.dumps({"comparison": str(comparison_path)}, indent=2))
        return

    result = run_feature_experiment(
        feature_set=args.feature_set,
        config_path=args.config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
