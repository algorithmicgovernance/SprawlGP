"""Contract tests for the isolated provisional annual ST-SVGP."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from src.feature_engineering.urban_expansion import add_candidate_features
from src.models.train_st_svgp import (
    validate_development_splits,
    write_annual_temporal_diagnostic,
)

ANNUAL_CONFIG_PATH = Path("configs/modeling/st_svgp_annual.yaml")
FIVE_YEAR_CONFIG_PATH = Path("configs/modeling/st_svgp.yaml")
FIVE_YEAR_CONFIG_SHA256 = (
    "aee54552739c73d48578bc3e6812ac48e4c260b076e14f96d8ab5b88e07670f0"
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_annual_config_preserves_promoted_architecture() -> None:
    annual = load_yaml(ANNUAL_CONFIG_PATH)
    promoted = load_yaml(FIVE_YEAR_CONFIG_PATH)

    assert annual["dataset"]["path"] == (
        "data/final/yaounde_urban_expansion_30m_annual_v1/"
        "cell_time_dataset.parquet"
    )
    assert annual["dataset"]["target"] == "target_transition_1y"
    assert "recent_local_growth_1y_t" in annual["linear_predictors"]
    assert "recent_local_growth_5y_t" not in annual["linear_predictors"]
    assert annual["time"]["step_years"] == 1
    assert annual["kernel"]["temporal_initial_lengthscale_steps"] == 7.5
    assert annual["kernel"]["temporal_lengthscale_trainable"] is True
    assert annual["inducing"]["spatial_points"] == 64
    assert annual["inducing"]["train_locations"] is False
    assert annual["feature_set"] == promoted["feature_set"]
    assert annual["kernel"]["spatial_family"] == promoted["kernel"]["spatial_family"]
    assert annual["kernel"]["temporal_family"] == promoted["kernel"]["temporal_family"]
    assert annual["inference"] == promoted["inference"]
    assert annual["training"] == promoted["training"]
    assert annual["compute"] == promoted["compute"]


def test_annual_temporal_partitions_are_exact_and_locked() -> None:
    config = load_yaml(ANNUAL_CONFIG_PATH)
    folds = config["rolling_validation"]["folds"]

    assert len(folds) == 3
    assert folds == [
        {
            "train_origins": list(range(2000, 2010)),
            "validation_origin": 2010,
        },
        {
            "train_origins": list(range(2000, 2015)),
            "validation_origin": 2015,
        },
        {
            "train_origins": list(range(2000, 2018)),
            "validation_origin": 2018,
        },
    ]
    assert config["final_fit"]["origins"] == list(range(2000, 2019))
    assert config["locked_block"] == {
        "origins": list(range(2019, 2025)),
        "target_years": list(range(2020, 2026)),
        "evaluate": False,
    }
    validate_development_splits(config)


@pytest.mark.parametrize(
    ("partition", "origin"),
    [
        ("rolling_training", 2019),
        ("rolling_validation", 2019),
        ("final_fit", 2019),
    ],
)
def test_development_rejects_target_year_2020_or_later(
    partition: str,
    origin: int,
) -> None:
    config = copy.deepcopy(load_yaml(ANNUAL_CONFIG_PATH))
    if partition == "rolling_training":
        config["rolling_validation"]["folds"][0]["train_origins"].append(origin)
    elif partition == "rolling_validation":
        config["rolling_validation"]["folds"][0]["validation_origin"] = origin
    else:
        config["final_fit"]["origins"].append(origin)

    with pytest.raises(ValueError, match="at most 2019"):
        validate_development_splits(config)


def test_annual_outputs_are_isolated_from_five_year_outputs() -> None:
    annual = load_yaml(ANNUAL_CONFIG_PATH)
    promoted = load_yaml(FIVE_YEAR_CONFIG_PATH)

    for key, annual_path in annual["outputs"].items():
        assert "st_svgp_annual" in annual_path
        assert annual_path != promoted["outputs"][key]

    assert annual["reporting"]["rolling_metrics_filename"] == "rolling_metrics.csv"
    assert annual["reporting"]["temporal_parameter_summary_filename"] == (
        "temporal_parameter_summary.csv"
    )
    assert annual["reporting"]["annual_temporal_diagnostic_filename"] == (
        "annual_temporal_diagnostic.md"
    )


def test_annual_derived_features_match_promoted_transformations() -> None:
    frame = pd.DataFrame(
        {
            "population_density_t": [0.0, 99.0],
            "distance_to_built_m_t": [0.0, 999.0],
            "slope_degrees": [1.0, 2.0],
            "built_fraction_11x11_t": [0.25, 0.5],
            "recent_local_growth_1y_t": [0.2, 0.4],
        }
    )

    result = add_candidate_features(
        frame,
        recent_growth_column="recent_local_growth_1y_t",
    )

    np.testing.assert_allclose(
        result["log_population_density_t"],
        np.log1p(frame["population_density_t"]),
    )
    np.testing.assert_allclose(
        result["log_distance_to_built_m_t"],
        np.log1p(frame["distance_to_built_m_t"]),
    )
    np.testing.assert_allclose(
        result["built_fraction_x_recent_growth_t"],
        frame["built_fraction_11x11_t"] * frame["recent_local_growth_1y_t"],
    )


def test_five_year_config_is_unchanged() -> None:
    digest = hashlib.sha256(FIVE_YEAR_CONFIG_PATH.read_bytes()).hexdigest()
    assert digest == FIVE_YEAR_CONFIG_SHA256


def test_annual_diagnostic_uses_observed_fold_results(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(load_yaml(ANNUAL_CONFIG_PATH))
    metrics_directory = tmp_path / "metrics"
    metrics_directory.mkdir()
    retained_metrics_path = tmp_path / "retained_metrics.csv"
    retained_predictions_path = tmp_path / "retained_predictions.parquet"

    pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "temporal_lengthscale_steps": [1.0, 1.5, 2.0],
            "probability_bias": [0.02, -0.01, 0.03],
        }
    ).to_csv(retained_metrics_path, index=False)
    pd.DataFrame(
        {
            "fold": np.repeat([1, 2, 3], 6),
            "target_transition_5y": np.tile([0, 0, 0, 1, 1, 1], 3),
            "probability_raw": np.tile(
                [0.05, 0.15, 0.3, 0.6, 0.75, 0.9],
                3,
            ),
        }
    ).to_parquet(retained_predictions_path, index=False)

    config["outputs"]["metrics_directory"] = str(metrics_directory)
    config["reporting"]["retained_five_year_metrics_path"] = str(
        retained_metrics_path
    )
    config["reporting"]["retained_five_year_predictions_path"] = str(
        retained_predictions_path
    )
    annual_metrics = pd.DataFrame(
        {
            "validation_target_year": [2011, 2016, 2019],
            "temporal_lengthscale_years": [6.5, 7.0, 7.5],
            "probability_bias": [0.01, 0.02, 0.0],
            "ece": [0.03, 0.04, 0.05],
            "calibration_slope": [0.9, 1.0, 1.1],
            "mean_latent_variance": [0.4, 0.5, 0.6],
            "median_latent_variance": [0.3, 0.4, 0.5],
        }
    )

    path = write_annual_temporal_diagnostic(annual_metrics, config)
    report = path.read_text(encoding="utf-8")

    assert path.parent == metrics_directory
    assert "EXPERIMENTAL / PROVISIONAL" in report
    assert "locked 2020-2025 target block was not evaluated" in report
    assert "different forecasting events" in report