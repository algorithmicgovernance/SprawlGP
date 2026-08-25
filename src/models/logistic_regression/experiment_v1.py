"""Train Logistic Regression without touching calibration or final test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .experiment_v1_data import (
    ExperimentData,
    build_split_summary,
    load_experiment,
    split_frame,
)
from ..metrics import probabilistic_metrics, validate_probabilities


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pipeline(
    c_value: float,
    config: dict[str, Any],
) -> Pipeline:
    """Construct the only logistic_regression model."""
    model_config = config["logistic_regression"]
    classifier = LogisticRegression(
        C=float(c_value),
        penalty=str(model_config["penalty"]),
        solver=str(model_config["solver"]),
        max_iter=int(model_config["max_iter"]),
        class_weight=None,
        random_state=int(model_config["random_state"]),
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def prepare_xy(
    experiment: ExperimentData,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract frozen predictors and the binary target."""
    return (
        frame.loc[:, experiment.features].astype(float),
        frame.loc[:, experiment.target].astype(int),
    )


def write_preflight(
    experiment: ExperimentData,
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, object]:
    """Write immutable split and feature metadata."""
    metadata_directory = Path(
        config["outputs"]["metadata_directory"]
    )
    metadata_directory.mkdir(parents=True, exist_ok=True)

    split_path = metadata_directory / "split_manifest.csv"
    build_split_summary(experiment).to_csv(split_path, index=False)

    feature_manifest = {
        "version": 1,
        "dataset_path": config["dataset"]["path"],
        "dataset_sha256": sha256(Path(config["dataset"]["path"])),
        "experiment_config": str(config_path),
        "experiment_config_sha256": sha256(config_path),
        "target": experiment.target,
        "features": list(experiment.features),
        "excluded_by_design": [
            "population_density_t",
            "current OSM variables",
            "target-year variables",
            "transition_quality",
            "persistence-correction fields",
        ],
        "calibration_status": "RESERVED_FOR_CALIBRATION",
        "final_test_status": "LOCKED_NOT_EVALUATED",
    }
    feature_path = metadata_directory / "feature_manifest.yaml"
    with feature_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(feature_manifest, stream, sort_keys=False)

    return {
        "status": "PASS",
        "dataset_rows": int(len(experiment.frame)),
        "features": list(experiment.features),
        "train_origins": list(experiment.train_origins),
        "validation_origins": list(experiment.validation_origins),
        "calibration_origins": list(experiment.calibration_origins),
        "test_origins": list(experiment.test_origins),
        "calibration_used": False,
        "final_test_evaluated": False,
        "split_manifest": str(split_path),
        "feature_manifest": str(feature_path),
    }


def train_logistic(
    experiment: ExperimentData,
    config: dict[str, Any],
) -> dict[str, object]:
    """Train and persist the Logistic Regression baseline.

    Candidate regularisation values are fitted on forecast origins 2000
    and 2005 and selected by validation log loss on origin 2010. The
    selected model is then refitted on origins 2000, 2005 and 2010.

    Calibration origin 2015 and final-test origin 2020 are never used by
    this function.

    Args:
        experiment: Validated dataset and temporal split contract.
        config: Modelling configuration containing model and output settings.

    Returns:
        A model-card dictionary describing the fitted pipeline, selected
        regularisation and protected temporal periods.

    Raises:
        RuntimeError: If no candidate model produces valid probabilities.
        ValueError: If probabilities or model inputs are invalid.
    """
    output = config["outputs"]
    model_directory = Path(output["model_directory"])
    metrics_directory = Path(output["metrics_directory"])
    predictions_directory = Path(output["predictions_directory"])

    model_directory.mkdir(parents=True, exist_ok=True)
    metrics_directory.mkdir(parents=True, exist_ok=True)
    predictions_directory.mkdir(parents=True, exist_ok=True)

    train_frame = split_frame(experiment, experiment.train_origins)
    validation_frame = split_frame(
        experiment, experiment.validation_origins
    )

    x_train, y_train = prepare_xy(experiment, train_frame)
    x_validation, y_validation = prepare_xy(
        experiment, validation_frame
    )

    search_records: list[dict[str, float]] = []
    best_c: float | None = None
    best_log_loss = float("inf")
    best_probabilities = None

    for c_value in config["logistic_regression"]["c_values"]:
        pipeline = build_pipeline(float(c_value), config)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_validation)[:, 1]
        metrics = probabilistic_metrics(
            y_validation.to_numpy(), probabilities
        )
        search_records.append({"C": float(c_value), **metrics})
        if metrics["log_loss"] < best_log_loss:
            best_log_loss = metrics["log_loss"]
            best_c = float(c_value)
            best_probabilities = probabilities.copy()

    if best_c is None or best_probabilities is None:
        raise RuntimeError("No valid Logistic Regression model was fitted.")

    validate_probabilities(best_probabilities)

    pd.DataFrame(search_records).to_csv(
        metrics_directory / "logistic_search.csv",
        index=False,
    )

    validation_predictions = validation_frame[
        [
            experiment.cell_id,
            experiment.forecast_origin,
            experiment.target_year,
            experiment.target,
        ]
    ].copy()
    validation_predictions["probability_raw"] = best_probabilities
    validation_predictions["model"] = "logistic_regression"
    validation_predictions["split"] = "validation"
    validation_predictions["C"] = best_c
    prediction_path = (
        predictions_directory
        / "logistic_validation_predictions.parquet"
    )
    validation_predictions.to_parquet(prediction_path, index=False)

    selected_metrics = next(
        record for record in search_records if record["C"] == best_c
    )
    with (
        metrics_directory / "logistic_validation.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "model": "logistic_regression",
                "selected_C": best_c,
                "selection_metric": "log_loss",
                "validation_metrics": {
                    key: value
                    for key, value in selected_metrics.items()
                    if key != "C"
                },
                "selection_training_origins": list(
                    experiment.train_origins
                ),
                "selection_validation_origins": list(
                    experiment.validation_origins
                ),
                "calibration_origins_used": [],
                "test_origins_used": [],
                "final_test_evaluated": False,
            },
            stream,
            indent=2,
        )

    refit_frame = pd.concat(
        [train_frame, validation_frame],
        ignore_index=True,
    )
    x_refit, y_refit = prepare_xy(experiment, refit_frame)
    final_pipeline = build_pipeline(best_c, config)
    final_pipeline.fit(x_refit, y_refit)

    model_path = model_directory / "model.joblib"
    joblib.dump(final_pipeline, model_path)

    classifier = final_pipeline.named_steps["classifier"]
    pd.DataFrame(
        {
            "feature": experiment.features,
            "coefficient_standardised": classifier.coef_[0],
        }
    ).to_csv(
        metrics_directory / "logistic_coefficients.csv",
        index=False,
    )

    model_card = {
        "status": "PASS",
        "model": "logistic_regression",
        "selected_C": best_c,
        "feature_count": len(experiment.features),
        "train_rows_for_selection": int(len(train_frame)),
        "validation_rows_for_selection": int(len(validation_frame)),
        "refit_rows": int(len(refit_frame)),
        "refit_origins": list(
            experiment.train_origins + experiment.validation_origins
        ),
        "calibration_used": False,
        "calibration_reserved_origins": list(
            experiment.calibration_origins
        ),
        "final_test_evaluated": False,
        "final_test_locked_origins": list(experiment.test_origins),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "validation_predictions": str(prediction_path),
        "validation_probabilities_finite": True,
    }
    with (
        model_directory / "model_card.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(model_card, stream, indent=2)

    return model_card


def main() -> None:
    """Run model preflight and optional training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()

    experiment, config = load_experiment(arguments.config)
    print(
        json.dumps(
            write_preflight(
                experiment, config, arguments.config
            ),
            indent=2,
        )
    )
    if not arguments.preflight_only:
        print(
            json.dumps(
                train_logistic(experiment, config),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
