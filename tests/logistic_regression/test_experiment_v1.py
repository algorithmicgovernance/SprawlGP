"""Reproducibility checks for the retained V1 experiment."""

from pathlib import Path

from src.models.logistic_regression.experiment_v1_data import load_config


CONFIG = Path("configs/modeling/logistic_regression/experiment_v1.yaml")


def test_v1_preserves_original_features() -> None:
    config = load_config(CONFIG)
    assert config["version"] == 1
    assert config["features"] == [
        "ndbi_t",
        "distance_to_built_m_t",
        "built_fraction_11x11_t",
        "recent_local_growth_5y_t",
        "elevation_m",
        "slope_degrees",
        "x_center_m",
        "y_center_m",
        "forecast_origin",
    ]


def test_v1_preserves_original_split() -> None:
    config = load_config(CONFIG)
    assert config["splits"] == {
        "train_origins": [2000, 2005],
        "validation_origins": [2010],
        "calibration_origins": [2015],
        "test_origins": [2020],
    }
