"""Train the selected Logistic Regression baseline.

The selected baseline is fixed from the previous Logistic Regression
experiments. It uses SAVI, log-transformed population density and no linear
forecast-year predictor. C is fixed at 0.1. Rolling temporal folds are kept
only to report historical generalisation; they no longer select features or
hyperparameters. The final model is refitted on 2000, 2005, 2010 and 2015,
while 2020→2025 remains locked for later final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import probabilistic_metrics, validate_probabilities


def load_config(path: Path) -> dict[str, Any]:
    """Load the selected Logistic Regression configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The Logistic Regression configuration must be a mapping.")
    return config


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_temporal_contract(config: dict[str, Any]) -> None:
    """Validate chronological rolling folds and the locked final test."""
    for index, fold in enumerate(config["rolling_validation"]["folds"], start=1):
        train_origins = [int(value) for value in fold["train_origins"]]
        validation_origin = int(fold["validation_origin"])
        if not train_origins or max(train_origins) >= validation_origin:
            raise ValueError(f"Rolling fold {index} is not chronological.")

    final_fit = [int(value) for value in config["final_fit"]["origins"]]
    final_test = [int(value) for value in config["final_test"]["origins"]]
    if not final_fit or not final_test:
        raise ValueError("final_fit and final_test origins must be defined.")
    if max(final_fit) >= min(final_test):
        raise ValueError("Final-fit origins must precede the final-test origin.")
    if bool(config["final_test"].get("evaluate", False)):
        raise ValueError("The final test must remain locked during baseline training.")


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load the frozen cell-time table and validate the baseline inputs."""
    dataset = config["dataset"]
    path = Path(dataset["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Modelling dataset not found: {path}")

    frame = pd.read_parquet(path)
    target = str(dataset["target"])
    origin = str(dataset["forecast_origin"])
    target_year = str(dataset["target_year"])
    cell_id = str(dataset["cell_id"])
    population = str(
        config["derived_features"]["log_population_density_t"]["source"]
    )
    raw_features = [
        feature
        for feature in config["features"]
        if feature != "log_population_density_t"
    ]
    required = {target, origin, target_year, cell_id, population, *raw_features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Missing modelling columns: " + ", ".join(missing))

    if frame.duplicated([cell_id, origin]).any():
        raise ValueError(f"Duplicate ({cell_id}, {origin}) rows were detected.")

    y = pd.to_numeric(frame[target], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"{target} must contain both binary classes.")

    origin_values = pd.to_numeric(frame[origin], errors="raise").astype(int)
    target_year_values = pd.to_numeric(frame[target_year], errors="raise").astype(int)
    if not target_year_values.eq(origin_values + 5).all():
        raise ValueError(f"{target_year} must equal {origin} + 5.")

    expected_origins = {
        *[
            int(value)
            for fold in config["rolling_validation"]["folds"]
            for value in fold["train_origins"]
        ],
        *[
            int(fold["validation_origin"])
            for fold in config["rolling_validation"]["folds"]
        ],
        *[int(value) for value in config["final_fit"]["origins"]],
        *[int(value) for value in config["final_test"]["origins"]],
    }
    missing_origins = sorted(expected_origins.difference(set(origin_values.unique())))
    if missing_origins:
        raise ValueError(f"Configured forecast origins are absent: {missing_origins}")

    return frame


def add_log_population(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create log1p population density without altering the source column."""
    result = frame.copy()
    specification = config["derived_features"]["log_population_density_t"]
    if str(specification["transformation"]) != "log1p":
        raise ValueError("The selected baseline requires log1p population density.")

    source = str(specification["source"])
    population = pd.to_numeric(result[source], errors="raise").astype(float)
    if not np.isfinite(population.to_numpy()).all():
        raise ValueError(f"{source} contains non-finite values.")
    if population.lt(0).any():
        raise ValueError(f"{source} cannot contain negative values.")

    result["log_population_density_t"] = np.log1p(population)
    return result


def validate_features(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    split_name: str,
) -> None:
    """Reject missing or non-finite selected predictors."""
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    missing = values.isna().sum()
    missing = missing[missing.gt(0)]
    if not missing.empty:
        details = ", ".join(f"{name}={count}" for name, count in missing.items())
        raise ValueError(f"Invalid predictors in {split_name}: {details}")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite predictors detected in {split_name}.")


def select_rows(
    frame: pd.DataFrame,
    origin_column: str,
    origins: list[int],
) -> pd.DataFrame:
    """Return a defensive copy for selected forecast origins."""
    return frame.loc[frame[origin_column].astype(int).isin(origins)].copy()


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build the fixed standardised L2 Logistic Regression baseline."""
    model = config["logistic_regression"]
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(model["C"]),
                    solver=str(model["solver"]),
                    max_iter=int(model["max_iter"]),
                    class_weight=None,
                    random_state=int(model["random_state"]),
                ),
            ),
        ]
    )


def constant_probability_metrics(
    targets: pd.Series,
    probability: float,
) -> dict[str, float]:
    """Evaluate a training-prevalence constant probability reference."""
    values = np.full(len(targets), float(probability), dtype=float)
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    return {
        "constant_log_loss": float(log_loss(targets.astype(int), clipped)),
        "constant_brier_score": float(
            brier_score_loss(targets.astype(int), values)
        ),
    }


def evaluate_rolling_folds(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the fixed baseline on three chronological historical folds."""
    dataset = config["dataset"]
    origin = str(dataset["forecast_origin"])
    target = str(dataset["target"])
    features = tuple(str(value) for value in config["features"])

    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for fold_index, fold in enumerate(config["rolling_validation"]["folds"], start=1):
        train_origins = [int(value) for value in fold["train_origins"]]
        validation_origin = int(fold["validation_origin"])
        train = select_rows(frame, origin, train_origins)
        validation = select_rows(frame, origin, [validation_origin])

        validate_features(train, features, f"fold {fold_index} training")
        validate_features(validation, features, f"fold {fold_index} validation")

        x_train = train.loc[:, features].astype(float)
        y_train = train[target].astype(int)
        x_validation = validation.loc[:, features].astype(float)
        y_validation = validation[target].astype(int)

        model = build_pipeline(config)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
        validate_probabilities(probabilities)

        metrics = probabilistic_metrics(y_validation.to_numpy(), probabilities)
        training_prevalence = float(y_train.mean())
        observed_prevalence = float(y_validation.mean())

        records.append(
            {
                "fold": fold_index,
                "train_origins": ",".join(map(str, train_origins)),
                "validation_origin": validation_origin,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "training_prevalence": training_prevalence,
                "observed_prevalence": observed_prevalence,
                "mean_predicted_probability": float(probabilities.mean()),
                "probability_bias": float(probabilities.mean() - observed_prevalence),
                **metrics,
                **constant_probability_metrics(y_validation, training_prevalence),
            }
        )

        fold_predictions = validation.loc[
            :,
            [
                str(dataset["cell_id"]),
                origin,
                str(dataset["target_year"]),
                target,
            ],
        ].copy()
        fold_predictions["probability_raw"] = probabilities
        fold_predictions["fold"] = fold_index
        predictions.append(fold_predictions)

    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True)


def summarise_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average historical metrics with equal weight for each validation period."""
    return pd.DataFrame(
        [
            {
                "folds": int(fold_metrics["fold"].nunique()),
                "mean_log_loss": float(fold_metrics["log_loss"].mean()),
                "std_log_loss": float(fold_metrics["log_loss"].std()),
                "mean_brier_score": float(fold_metrics["brier_score"].mean()),
                "std_brier_score": float(fold_metrics["brier_score"].std()),
                "mean_pr_auc": float(fold_metrics["pr_auc"].mean()),
                "std_pr_auc": float(fold_metrics["pr_auc"].std()),
                "mean_roc_auc": float(fold_metrics["roc_auc"].mean()),
                "std_roc_auc": float(fold_metrics["roc_auc"].std()),
                "mean_probability_bias": float(
                    fold_metrics["probability_bias"].mean()
                ),
                "mean_constant_log_loss": float(
                    fold_metrics["constant_log_loss"].mean()
                ),
                "mean_constant_brier_score": float(
                    fold_metrics["constant_brier_score"].mean()
                ),
            }
        ]
    )


def fit_final_model(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Pipeline, int]:
    """Fit the selected baseline on all pre-test origins only."""
    dataset = config["dataset"]
    origin = str(dataset["forecast_origin"])
    target = str(dataset["target"])
    features = tuple(str(value) for value in config["features"])
    origins = [int(value) for value in config["final_fit"]["origins"]]
    final_frame = select_rows(frame, origin, origins)
    validate_features(final_frame, features, "final fit")

    model = build_pipeline(config)
    model.fit(
        final_frame.loc[:, features].astype(float),
        final_frame[target].astype(int),
    )
    return model, int(len(final_frame))


def write_preflight(
    frame: pd.DataFrame,
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, object]:
    """Write the fixed feature and temporal contract for reproducibility."""
    metadata_directory = Path(config["outputs"]["metadata_directory"])
    metadata_directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for index, fold in enumerate(config["rolling_validation"]["folds"], start=1):
        records.append(
            {
                "role": "historical_validation",
                "fold": index,
                "train_origins": ",".join(map(str, fold["train_origins"])),
                "validation_origin": int(fold["validation_origin"]),
            }
        )
    records.append(
        {
            "role": "final_fit",
            "fold": "",
            "train_origins": ",".join(map(str, config["final_fit"]["origins"])),
            "validation_origin": "",
        }
    )
    records.append(
        {
            "role": "final_test_locked",
            "fold": "",
            "train_origins": "",
            "validation_origin": ",".join(map(str, config["final_test"]["origins"])),
        }
    )

    split_path = metadata_directory / "temporal_split_manifest.csv"
    pd.DataFrame(records).to_csv(split_path, index=False)

    dataset_path = Path(config["dataset"]["path"])
    preflight = {
        "status": "PASS",
        "model": "logistic_regression",
        "role": "selected_baseline",
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "dataset_rows": int(len(frame)),
        "features": list(config["features"]),
        "C": float(config["logistic_regression"]["C"]),
        "rolling_validation_folds": config["rolling_validation"]["folds"],
        "final_fit_origins": config["final_fit"]["origins"],
        "final_test_locked_origins": config["final_test"]["origins"],
        "final_test_evaluated": False,
        "config_sha256": sha256(config_path),
        "split_manifest": str(split_path),
    }
    (metadata_directory / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    return preflight


def train_baseline(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, object]:
    """Evaluate the fixed historical folds, refit, and persist the baseline."""
    outputs = config["outputs"]
    model_directory = Path(outputs["model_directory"])
    metrics_directory = Path(outputs["metrics_directory"])
    predictions_directory = Path(outputs["predictions_directory"])
    for directory in [model_directory, metrics_directory, predictions_directory]:
        directory.mkdir(parents=True, exist_ok=True)

    fold_metrics, oof_predictions = evaluate_rolling_folds(frame, config)
    fold_metrics_path = metrics_directory / "logistic_rolling_validation_metrics.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False)

    summary = summarise_folds(fold_metrics)
    summary_path = metrics_directory / "logistic_rolling_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    oof_path = predictions_directory / "logistic_oof_predictions.parquet"
    oof_predictions.to_parquet(oof_path, index=False)

    model, final_fit_rows = fit_final_model(frame, config)
    model_path = model_directory / "model.joblib"
    joblib.dump(model, model_path)

    features = tuple(str(value) for value in config["features"])
    coefficients = model.named_steps["classifier"].coef_[0]
    coefficient_path = metrics_directory / "logistic_coefficients.csv"
    pd.DataFrame(
        {
            "feature": features,
            "coefficient_standardised": coefficients,
            "odds_ratio_per_standard_deviation": np.exp(coefficients),
        }
    ).to_csv(coefficient_path, index=False)

    row = summary.iloc[0]
    model_card = {
        "status": "PASS",
        "model": "logistic_regression",
        "role": "selected_baseline",
        "C": float(config["logistic_regression"]["C"]),
        "feature_count": len(features),
        "features": list(features),
        "historical_validation": {
            "folds": int(row["folds"]),
            "mean_log_loss": float(row["mean_log_loss"]),
            "mean_brier_score": float(row["mean_brier_score"]),
            "mean_pr_auc": float(row["mean_pr_auc"]),
            "mean_roc_auc": float(row["mean_roc_auc"]),
            "mean_probability_bias": float(row["mean_probability_bias"]),
        },
        "final_fit_rows": final_fit_rows,
        "final_fit_origins": [int(value) for value in config["final_fit"]["origins"]],
        "final_test_locked_origins": [
            int(value) for value in config["final_test"]["origins"]
        ],
        "final_test_evaluated": False,
        "calibration_applied": False,
        "oof_predictions_available_for_calibration": True,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "fold_metrics": str(fold_metrics_path),
        "rolling_summary": str(summary_path),
        "oof_predictions": str(oof_path),
        "coefficients": str(coefficient_path),
    }
    (model_directory / "model_card.json").write_text(
        json.dumps(model_card, indent=2) + "\n",
        encoding="utf-8",
    )
    return model_card


def run(config_path: Path, preflight_only: bool = False) -> dict[str, object]:
    """Run preflight and optionally fit the selected baseline."""
    config = load_config(config_path)
    validate_temporal_contract(config)
    frame = add_log_population(load_dataset(config), config)
    preflight = write_preflight(frame, config, config_path)
    if preflight_only:
        return preflight
    return train_baseline(frame, config)


def main() -> None:
    """Run the selected Logistic Regression baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.preflight_only), indent=2))


if __name__ == "__main__":
    main()
