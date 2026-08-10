"""Load and validate the frozen cell-time modelling dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class ExperimentData:
    """Validated dataset and immutable modelling configuration."""

    frame: pd.DataFrame
    features: tuple[str, ...]
    target: str
    forecast_origin: str
    target_year: str
    cell_id: str
    train_origins: tuple[int, ...]
    validation_origins: tuple[int, ...]
    calibration_origins: tuple[int, ...]
    test_origins: tuple[int, ...]


def load_config(path: Path) -> dict[str, Any]:
    """Load one YAML experiment configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The experiment configuration must be a mapping.")
    return config


def _as_int_tuple(values: list[int], name: str) -> tuple[int, ...]:
    """Validate and freeze one list of forecast origins."""
    if not values:
        raise ValueError(f"{name} must contain at least one forecast origin.")
    origins = tuple(int(value) for value in values)
    if len(set(origins)) != len(origins):
        raise ValueError(f"{name} contains duplicate origins: {origins}")
    return origins


def validate_split_contract(experiment: ExperimentData) -> None:
    """Ensure chronological, disjoint and available forecast-origin splits."""
    named_splits = {
        "train": set(experiment.train_origins),
        "validation": set(experiment.validation_origins),
        "calibration": set(experiment.calibration_origins),
        "test": set(experiment.test_origins),
    }
    names = tuple(named_splits)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = named_splits[left_name] & named_splits[right_name]
            if overlap:
                raise ValueError(
                    f"Splits {left_name} and {right_name} overlap: "
                    f"{sorted(overlap)}"
                )

    ordered = (
        list(experiment.train_origins)
        + list(experiment.validation_origins)
        + list(experiment.calibration_origins)
        + list(experiment.test_origins)
    )
    if ordered != sorted(ordered):
        raise ValueError("Forecast-origin splits must be chronological.")

    available = set(
        pd.to_numeric(
            experiment.frame[experiment.forecast_origin],
            errors="raise",
        ).astype(int)
    )
    missing = sorted(set(ordered) - available)
    if missing:
        raise ValueError(f"Configured forecast origins are absent: {missing}")


def load_experiment(config_path: Path) -> tuple[ExperimentData, dict[str, Any]]:
    """Load and validate the frozen modelling experiment.

    The function reads the YAML configuration and the cell-time Parquet
    dataset. It verifies the required columns, binary target, unique
    cell-time keys, finite predictors, chronological splits and the
    five-year relationship between forecast origin and target year.

    Args:
        config_path: Path to the modelling experiment YAML file.

    Returns:
        A validated ExperimentData object and the raw configuration.

    Raises:
        FileNotFoundError: If the configured Parquet dataset is absent.
        ValueError: If columns, targets, predictors or temporal splits
            violate the modelling contract.
    """
    config = load_config(config_path)
    dataset_config = config["dataset"]
    split_config = config["splits"]

    dataset_path = Path(dataset_config["path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Modelling dataset not found: {dataset_path}")

    frame = pd.read_parquet(dataset_path)
    features = tuple(str(value) for value in config["features"])

    target = str(dataset_config["target"])
    forecast_origin = str(dataset_config["forecast_origin"])
    target_year = str(dataset_config["target_year"])
    cell_id = str(dataset_config["cell_id"])

    required = {*features, target, forecast_origin, target_year, cell_id}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "The dataset is missing required modelling columns: "
            + ", ".join(missing)
        )

    if frame.duplicated([cell_id, forecast_origin]).any():
        raise ValueError(
            f"Duplicate ({cell_id}, {forecast_origin}) rows detected."
        )

    target_values = set(
        pd.to_numeric(frame[target], errors="raise")
        .dropna()
        .astype(int)
        .unique()
    )
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{target} must be binary; observed {sorted(target_values)}."
        )

    feature_frame = frame.loc[:, features].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid_counts = feature_frame.isna().sum()
    invalid_counts = invalid_counts[invalid_counts.gt(0)]
    if not invalid_counts.empty:
        details = ", ".join(
            f"{column}={count}"
            for column, count in invalid_counts.items()
        )
        raise ValueError(
            "Missing or non-numeric feature values are not allowed: "
            + details
        )

    expected_target_year = (
        pd.to_numeric(frame[forecast_origin], errors="raise").astype(int) + 5
    )
    observed_target_year = pd.to_numeric(
        frame[target_year],
        errors="raise",
    ).astype(int)
    if not observed_target_year.eq(expected_target_year).all():
        raise ValueError(
            f"{target_year} must equal {forecast_origin} + 5."
        )

    experiment = ExperimentData(
        frame=frame,
        features=features,
        target=target,
        forecast_origin=forecast_origin,
        target_year=target_year,
        cell_id=cell_id,
        train_origins=_as_int_tuple(
            split_config["train_origins"], "train_origins"
        ),
        validation_origins=_as_int_tuple(
            split_config["validation_origins"], "validation_origins"
        ),
        calibration_origins=_as_int_tuple(
            split_config["calibration_origins"], "calibration_origins"
        ),
        test_origins=_as_int_tuple(
            split_config["test_origins"], "test_origins"
        ),
    )
    validate_split_contract(experiment)
    return experiment, config


def split_frame(
    experiment: ExperimentData,
    origins: tuple[int, ...],
) -> pd.DataFrame:
    """Return a defensive copy for the requested forecast origins."""
    mask = (
        experiment.frame[experiment.forecast_origin]
        .astype(int)
        .isin(origins)
    )
    return experiment.frame.loc[mask].copy()


def build_split_summary(experiment: ExperimentData) -> pd.DataFrame:
    """Build a compact split manifest without exposing protected outcomes."""
    records: list[dict[str, object]] = []
    definitions = [
        ("train", experiment.train_origins, True),
        ("validation", experiment.validation_origins, True),
        ("calibration", experiment.calibration_origins, False),
        ("test", experiment.test_origins, False),
    ]

    for split_name, origins, target_inspected in definitions:
        subset = split_frame(experiment, origins)
        for origin, group in subset.groupby(
            experiment.forecast_origin,
            sort=True,
        ):
            record: dict[str, object] = {
                "split": split_name,
                "forecast_origin": int(origin),
                "target_year": int(origin) + 5,
                "row_count": int(len(group)),
                "target_inspected": target_inspected,
                "positive_count": "",
                "positive_rate": "",
            }
            if target_inspected:
                record["positive_count"] = int(
                    group[experiment.target].sum()
                )
                record["positive_rate"] = float(
                    group[experiment.target].mean()
                )
            records.append(record)
    return pd.DataFrame.from_records(records)
