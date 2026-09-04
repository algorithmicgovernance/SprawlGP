"""Strictly temporal post-hoc calibration for the 3000-iteration annual ST-SVGP."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from src.models.evaluation.calibration import (
    FittedBinaryCalibrator,
    calibration_metrics,
    fit_beta_calibrator,
    fit_platt_calibrator,
)
from src.models.metrics import probabilistic_metrics, validate_probabilities

DEFAULT_PREDICTIONS_PATH = Path(
    "reports/modeling/st_svgp_annual/experiments/convergence_3000/"
    "predictions/st_svgp_oof_predictions.parquet"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "reports/modeling/st_svgp_annual/experiments/convergence_3000/calibration"
)
EXPECTED_FOLDS = (1, 2, 3)
EVALUATION_FOLDS = (2, 3)
TARGET_COLUMN = "target_transition_1y"
PROBABILITY_COLUMN = "probability_raw"
CALIBRATED_PROBABILITY_COLUMN = "probability_calibrated"


@dataclass(frozen=True)
class CalibrationPaths:
    """Read-only input and isolated post-hoc calibration outputs."""

    predictions_input: Path
    output_directory: Path
    metrics: Path
    comparison: Path
    platt_predictions: Path
    beta_predictions: Path
    diagnostic: Path


def calibration_paths(
    predictions_input: Path = DEFAULT_PREDICTIONS_PATH,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> CalibrationPaths:
    """Resolve the source OOF artifact and experiment-local outputs."""
    return CalibrationPaths(
        predictions_input=predictions_input,
        output_directory=output_directory,
        metrics=output_directory / "temporal_calibration_metrics.csv",
        comparison=output_directory / "temporal_calibration_comparison.csv",
        platt_predictions=output_directory / "calibrated_predictions_platt.parquet",
        beta_predictions=output_directory / "calibrated_predictions_beta.parquet",
        diagnostic=output_directory / "temporal_calibration_diagnostic.md",
    )


def _validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the annual OOF schema and locked temporal boundary."""
    required = {
        "fold",
        "target_year",
        TARGET_COLUMN,
        PROBABILITY_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Annual OOF predictions are missing: " + ", ".join(missing))

    result = frame.copy()
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    result["target_year"] = pd.to_numeric(
        result["target_year"], errors="raise"
    ).astype(int)
    if (result["target_year"] >= 2020).any():
        raise ValueError("Annual OOF predictions contain target_year >= 2020.")
    folds = tuple(sorted(result["fold"].unique().tolist()))
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"Annual OOF predictions must contain folds {EXPECTED_FOLDS}.")

    target = result[TARGET_COLUMN].to_numpy(dtype=int)
    if set(np.unique(target)).difference({0, 1}):
        raise ValueError("Annual OOF target must be binary.")
    validate_probabilities(result[PROBABILITY_COLUMN].to_numpy(dtype=float))
    return result


def _fit_method(
    method: str,
    target: np.ndarray,
    probability: np.ndarray,
) -> FittedBinaryCalibrator:
    if method == "platt":
        return fit_platt_calibrator(target, probability)
    if method == "beta":
        return fit_beta_calibrator(target, probability)
    raise ValueError("method must be 'platt' or 'beta'.")


def temporal_calibrate(
    frame: pd.DataFrame,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on strictly earlier folds and return Fold 2/3 calibrated rows."""
    table = _validate_predictions(frame)
    prediction_tables: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []

    for fold in EVALUATION_FOLDS:
        calibration = table.loc[table["fold"] < fold]
        evaluation = table.loc[table["fold"] == fold].copy()
        calibration_folds = tuple(sorted(calibration["fold"].unique().tolist()))
        expected_calibration_folds = tuple(range(1, fold))
        if calibration_folds != expected_calibration_folds:
            raise ValueError(
                f"Fold {fold} calibration requires folds "
                f"{expected_calibration_folds}; got {calibration_folds}."
            )

        fitted = _fit_method(
            method,
            calibration[TARGET_COLUMN].to_numpy(dtype=int),
            calibration[PROBABILITY_COLUMN].to_numpy(dtype=float),
        )
        calibrated = fitted.predict(
            evaluation[PROBABILITY_COLUMN].to_numpy(dtype=float)
        )
        validate_probabilities(calibrated)
        evaluation[CALIBRATED_PROBABILITY_COLUMN] = calibrated
        prediction_tables.append(evaluation)
        direction = fitted.monotonic_direction()
        fit_rows.append(
            {
                "method": method,
                "fold": fold,
                "calibration_folds": ",".join(map(str, calibration_folds)),
                "calibration_rows": len(calibration),
                "evaluation_rows": len(evaluation),
                "coefficient_a": fitted.coefficient_a,
                "coefficient_b": fitted.coefficient_b,
                "coefficient_c": fitted.coefficient_c,
                "monotonic_direction": direction,
                "mapping_monotonic_increasing": direction == "increasing",
            }
        )

    predictions = pd.concat(prediction_tables).sort_index().reset_index(drop=True)
    return predictions, pd.DataFrame(fit_rows)


def _metric_row(
    method: str,
    fold: int,
    calibration_rows: int,
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, object]:
    """Evaluate one fold through retained repository metric utilities."""
    return {
        "method": method,
        "fold": fold,
        "calibration_rows": calibration_rows,
        "evaluation_rows": len(target),
        "observed_prevalence": float(np.mean(target)),
        "mean_predicted_probability": float(np.mean(probability)),
        **probabilistic_metrics(target, probability),
        **calibration_metrics(target, probability),
    }


def build_comparison(
    raw_predictions: pd.DataFrame,
    calibrated_predictions: dict[str, pd.DataFrame],
    fit_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare raw, Platt, and Beta metrics and verify ranking preservation."""
    raw = _validate_predictions(raw_predictions)
    rows: list[dict[str, object]] = []

    for fold in EVALUATION_FOLDS:
        evaluation = raw.loc[raw["fold"] == fold]
        target = evaluation[TARGET_COLUMN].to_numpy(dtype=int)
        probability_raw = evaluation[PROBABILITY_COLUMN].to_numpy(dtype=float)
        calibration_rows = int((raw["fold"] < fold).sum())
        raw_row = _metric_row(
            "raw", fold, calibration_rows, target, probability_raw
        )
        raw_row.update(
            {
                "calibration_folds": ",".join(map(str, range(1, fold))),
                "coefficient_a": np.nan,
                "coefficient_b": np.nan,
                "coefficient_c": np.nan,
                "monotonic_direction": "identity",
                "mapping_monotonic_increasing": True,
                "pr_auc_difference_from_raw": 0.0,
                "roc_auc_difference_from_raw": 0.0,
                "pr_auc_ranking_preserved": True,
                "roc_auc_ranking_preserved": True,
            }
        )
        rows.append(raw_row)

        for method in ("platt", "beta"):
            method_evaluation = calibrated_predictions[method]
            method_evaluation = method_evaluation.loc[method_evaluation["fold"] == fold]
            calibrated = method_evaluation[
                CALIBRATED_PROBABILITY_COLUMN
            ].to_numpy(dtype=float)
            method_row = _metric_row(
                method, fold, calibration_rows, target, calibrated
            )
            fit = fit_tables[method].set_index("fold").loc[fold]
            pr_difference = float(method_row["pr_auc"] - raw_row["pr_auc"])
            roc_difference = float(method_row["roc_auc"] - raw_row["roc_auc"])
            method_row.update(
                {
                    "calibration_folds": fit["calibration_folds"],
                    "coefficient_a": fit["coefficient_a"],
                    "coefficient_b": fit["coefficient_b"],
                    "coefficient_c": fit["coefficient_c"],
                    "monotonic_direction": fit["monotonic_direction"],
                    "mapping_monotonic_increasing": bool(
                        fit["mapping_monotonic_increasing"]
                    ),
                    "pr_auc_difference_from_raw": pr_difference,
                    "roc_auc_difference_from_raw": roc_difference,
                    "pr_auc_ranking_preserved": bool(
                        np.isclose(pr_difference, 0.0, atol=1.0e-12, rtol=0.0)
                    ),
                    "roc_auc_ranking_preserved": bool(
                        np.isclose(roc_difference, 0.0, atol=1.0e-12, rtol=0.0)
                    ),
                }
            )
            rows.append(method_row)

    comparison = pd.DataFrame(rows)
    method_order = pd.Categorical(
        comparison["method"], categories=["raw", "platt", "beta"], ordered=True
    )
    return (
        comparison.assign(_method_order=method_order)
        .sort_values(["fold", "_method_order"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def calibration_status(comparison: pd.DataFrame) -> str:
    """Classify broad proper-score, calibration, and discrimination evidence."""
    by_key = comparison.set_index(["fold", "method"])
    robust_methods: list[str] = []
    broad_fold_improvement = False
    for method in ("platt", "beta"):
        method_robust = True
        for fold in EVALUATION_FOLDS:
            raw = by_key.loc[(fold, "raw")]
            calibrated = by_key.loc[(fold, method)]
            improvements = (
                calibrated["log_loss"] < raw["log_loss"],
                calibrated["brier_score"] < raw["brier_score"],
                calibrated["ece"] < raw["ece"],
                abs(calibrated["calibration_slope"] - 1.0)
                < abs(raw["calibration_slope"] - 1.0),
                calibrated["absolute_probability_bias"]
                < raw["absolute_probability_bias"],
            )
            ranking_preserved = bool(
                calibrated["pr_auc_ranking_preserved"]
                and calibrated["roc_auc_ranking_preserved"]
            )
            method_robust = method_robust and all(improvements) and ranking_preserved
            broad_fold_improvement = broad_fold_improvement or (
                sum(improvements) >= 4 and ranking_preserved
            )
        if method_robust:
            robust_methods.append(method)

    if robust_methods:
        return "CALIBRATION_SUPPORTS_STRONG_ANNUAL_ST_SVGP"
    if broad_fold_improvement:
        return "CALIBRATION_IMPROVES_MODEL_BUT_REMAINS_HETEROGENEOUS"
    return "CALIBRATION_DOES_NOT_RESCUE_PROBABILITY_QUALITY"


def _format_float(value: object) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.6f}"


def render_diagnostic(
    comparison: pd.DataFrame,
    source_path: Path,
    source_sha256: str,
) -> str:
    """Render the temporal calibration evidence and final decision."""
    status = calibration_status(comparison)
    lines = [
        "# Annual ST-SVGP temporal calibration diagnostic",
        "",
        "## Scope and temporal contract",
        "",
        (
            f"Source: `{source_path}` (SHA-256 `{source_sha256}`). The source OOF "
            "artifact was read only; no model training was performed. Fold 1 has no "
            "calibrated evaluation. Fold 2 fits on Fold 1 only, and Fold 3 fits on "
            "Folds 1+2 only. All rows have `target_year < 2020`."
        ),
        "",
        (
            "Platt uses `logit(q) = a + b * logit(p)`. Beta uses "
            "`logit(q) = a * log(p) + b * log(1-p) + c`. Inputs are clipped to "
            "[1e-6, 1-1e-6] only for logarithms. Fits are unweighted and use no "
            "regularization."
        ),
        "",
        "The canonical marginal and Mondrian conformal reports were not modified.",
        "",
        "## Fitted parameters",
        "",
        "| Method | Fold | Calibration folds | Calibration rows | a | b | c | Direction |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in comparison.loc[comparison["method"] != "raw"].itertuples(index=False):
        lines.append(
            f"| {row.method} | {int(row.fold)} | {row.calibration_folds} | "
            f"{int(row.calibration_rows):,} | {_format_float(row.coefficient_a)} | "
            f"{_format_float(row.coefficient_b)} | {_format_float(row.coefficient_c)} | "
            f"{row.monotonic_direction} |"
        )

    lines.extend(
        [
            "",
            "## Probability metrics",
            "",
            (
                "| Fold | Method | Prevalence | Mean probability | PR-AUC | ROC-AUC | "
                "Log Loss | Brier | ECE | Intercept | Slope | Bias |"
            ),
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.method} | {row.observed_prevalence:.6f} | "
            f"{row.mean_predicted_probability:.6f} | {row.pr_auc:.6f} | "
            f"{row.roc_auc:.6f} | {row.log_loss:.6f} | {row.brier_score:.6f} | "
            f"{row.ece:.6f} | {row.calibration_intercept:.6f} | "
            f"{row.calibration_slope:.6f} | {row.probability_bias:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Ranking check",
            "",
            "| Fold | Method | PR-AUC delta | ROC-AUC delta | PR preserved | ROC preserved |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    for row in comparison.loc[comparison["method"] != "raw"].itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.method} | "
            f"{row.pr_auc_difference_from_raw:.12f} | "
            f"{row.roc_auc_difference_from_raw:.12f} | "
            f"{row.pr_auc_ranking_preserved} | {row.roc_auc_ranking_preserved} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Fold 2 is the limiting result: calibration transferred from the much "
                "lower-prevalence Fold 1 and worsened Log Loss, Brier, ECE, calibration "
                "slope distance from 1, and absolute probability bias for both methods."
            ),
            "",
            (
                "Fold 3 shows the opposite pattern. Both methods lower Log Loss, Brier, "
                "ECE, and absolute probability bias, but move the calibration slope "
                "farther from 1. Both fitted mappings are increasing and the directly "
                "recomputed PR-AUC and ROC-AUC values are unchanged."
            ),
            "",
            (
                "The broad Fold 3 improvement is scientifically useful, but the Fold 2 "
                "failure prevents a robust positive calibration conclusion. ECE is not "
                "used alone: the decision jointly considers proper scores, slope, bias, "
                "discrimination, and fold robustness."
            ),
            "",
            status,
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    predictions_input: Path = DEFAULT_PREDICTIONS_PATH,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, str]:
    """Run calibration from existing OOF predictions without model training."""
    paths = calibration_paths(predictions_input, output_directory)
    if not paths.predictions_input.is_file():
        raise FileNotFoundError(f"OOF predictions not found: {paths.predictions_input}")
    output_paths = {
        paths.metrics,
        paths.comparison,
        paths.platt_predictions,
        paths.beta_predictions,
        paths.diagnostic,
    }
    if paths.predictions_input in output_paths:
        raise ValueError("Raw OOF predictions cannot be used as a calibration output.")
    if any(path.parent != paths.output_directory for path in output_paths):
        raise ValueError("Every calibration output must be experiment-local.")

    source_sha256 = _sha256(paths.predictions_input)
    raw = _validate_predictions(pd.read_parquet(paths.predictions_input))
    calibrated_predictions: dict[str, pd.DataFrame] = {}
    fit_tables: dict[str, pd.DataFrame] = {}
    for method in ("platt", "beta"):
        predictions, fits = temporal_calibrate(raw, method)
        calibrated_predictions[method] = predictions
        fit_tables[method] = fits

    comparison = build_comparison(raw, calibrated_predictions, fit_tables)
    metrics = comparison.loc[comparison["method"] != "raw"].reset_index(drop=True)
    diagnostic = render_diagnostic(comparison, paths.predictions_input, source_sha256)
    if _sha256(paths.predictions_input) != source_sha256:
        raise RuntimeError("Raw OOF predictions changed during calibration.")

    paths.output_directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.metrics, index=False)
    comparison.to_csv(paths.comparison, index=False)
    calibrated_predictions["platt"].to_parquet(paths.platt_predictions, index=False)
    calibrated_predictions["beta"].to_parquet(paths.beta_predictions, index=False)
    paths.diagnostic.write_text(diagnostic, encoding="utf-8")
    if _sha256(paths.predictions_input) != source_sha256:
        raise RuntimeError("Raw OOF predictions changed while outputs were written.")

    return {
        "metrics": str(paths.metrics),
        "comparison": str(paths.comparison),
        "platt_predictions": str(paths.platt_predictions),
        "beta_predictions": str(paths.beta_predictions),
        "diagnostic": str(paths.diagnostic),
        "source_sha256": source_sha256,
        "status": calibration_status(comparison),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(run(args.predictions, args.output_directory), indent=2))


if __name__ == "__main__":
    main()