"""Focused contracts for Day-5 ST-SVGP OOF diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.evaluation.st_svgp_day5_diagnostics import (
    HorizonSpec,
    assign_distance_context,
    assign_error_level,
    join_oof_to_modeling_data,
    summarize_diagnostics,
    uncertainty_summaries,
)

PREDICTORS = [
    "ndbi_t",
    "savi_t",
    "log_distance_to_built_m_t",
    "built_fraction_11x11_t",
    "recent_local_growth_1y_t",
    "elevation_m",
    "slope_degrees",
    "log_population_density_t",
    "built_fraction_x_recent_growth_t",
]


def model_config() -> dict:
    return {
        "dataset": {
            "target": "target_transition_1y",
            "forecast_origin": "forecast_origin",
            "target_year": "target_year",
            "cell_id": "cell_id",
            "x_coordinate": "x_center_m",
            "y_coordinate": "y_center_m",
        },
        "feature_engineering": {"recent_growth_column": "recent_local_growth_1y_t"},
        "linear_predictors": PREDICTORS,
        "time": {"step_years": 1},
        "rolling_validation": {
            "folds": [{"train_origins": [2009], "validation_origin": 2010}]
        },
    }


def annual_spec() -> HorizonSpec:
    return HorizonSpec(
        name="annual_1y",
        label="Annual 1-year ST-SVGP",
        model_config_path=None,  # type: ignore[arg-type]
        output_subdirectory="annual_1y",
        maximum_target_year=2019,
        locked_forecast_origins=(),
    )


def modeling_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4, 5, 6],
            "forecast_origin": [2010] * 6,
            "target_transition_1y": [0, 1, 0, 1, 0, 1],
            "x_center_m": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "y_center_m": [60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
            "ndbi_t": np.linspace(0.1, 0.6, 6),
            "savi_t": np.linspace(0.6, 0.1, 6),
            "distance_to_built_m_t": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "built_fraction_11x11_t": np.linspace(0.0, 0.5, 6),
            "recent_local_growth_1y_t": np.linspace(0.1, 0.6, 6),
            "elevation_m": np.linspace(700.0, 705.0, 6),
            "slope_degrees": np.linspace(1.0, 6.0, 6),
            "population_density_t": np.linspace(100.0, 600.0, 6),
        }
    )


def oof_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [3, 1, 6, 2, 5, 4],
            "forecast_origin": [2010] * 6,
            "target_year": [2011] * 6,
            "target_transition_1y": [0, 0, 1, 1, 0, 1],
            "probability_raw": [0.2, 0.1, 0.7, 0.8, 0.4, 0.6],
            "fold": [1] * 6,
            "latent_variance": [0.3, 0.1, 0.6, 0.2, 0.5, 0.4],
        }
    )


def joined_frame() -> pd.DataFrame:
    return join_oof_to_modeling_data(
        oof_predictions(),
        modeling_data(),
        spec=annual_spec(),
        model_config=model_config(),
        probability_epsilon=1.0e-12,
    )


def test_oof_join_uses_stable_keys_and_computes_errors() -> None:
    result = joined_frame()

    assert result["cell_id"].tolist() == [3, 1, 6, 2, 5, 4]
    assert result["x_center_m"].tolist() == [30.0, 10.0, 60.0, 20.0, 50.0, 40.0]
    np.testing.assert_allclose(result["signed_error"], [0.2, 0.1, -0.3, -0.2, 0.4, -0.4])
    np.testing.assert_allclose(result["absolute_error"], [0.2, 0.1, 0.3, 0.2, 0.4, 0.4])
    expected_loss = -np.log([0.8, 0.9, 0.7, 0.8, 0.6, 0.6])
    np.testing.assert_allclose(result["per_cell_log_loss"], expected_loss)
    np.testing.assert_array_equal(
        result["latent_variance"], oof_predictions()["latent_variance"]
    )


@pytest.mark.parametrize("source", ["oof", "data"])
def test_oof_join_rejects_duplicate_keys(source: str) -> None:
    oof = oof_predictions()
    data = modeling_data()
    if source == "oof":
        oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    else:
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        join_oof_to_modeling_data(
            oof,
            data,
            spec=annual_spec(),
            model_config=model_config(),
            probability_epsilon=1.0e-12,
        )


def test_oof_join_rejects_missing_key() -> None:
    with pytest.raises(ValueError, match="did not join"):
        join_oof_to_modeling_data(
            oof_predictions(),
            modeling_data().loc[lambda frame: frame["cell_id"].ne(3)],
            spec=annual_spec(),
            model_config=model_config(),
            probability_epsilon=1.0e-12,
        )


def test_annual_lock_rejects_target_year_2020() -> None:
    oof = oof_predictions()
    oof["target_year"] = 2020
    with pytest.raises(ValueError, match="target_year after 2019"):
        join_oof_to_modeling_data(
            oof,
            modeling_data(),
            spec=annual_spec(),
            model_config=model_config(),
            probability_epsilon=1.0e-12,
        )


def test_five_year_lock_rejects_final_test_origin() -> None:
    config = model_config()
    config["time"]["step_years"] = 5
    config["rolling_validation"]["folds"][0]["validation_origin"] = 2020
    oof = oof_predictions().rename(
        columns={"target_transition_1y": "target_transition_5y"}
    )
    oof["forecast_origin"] = 2020
    oof["target_year"] = 2025
    config["dataset"]["target"] = "target_transition_5y"
    data = modeling_data().rename(
        columns={"target_transition_1y": "target_transition_5y"}
    )
    data["forecast_origin"] = 2020
    spec = HorizonSpec(
        name="five_year_5y",
        label="Five-year ST-SVGP",
        model_config_path=None,  # type: ignore[arg-type]
        output_subdirectory="five_year_5y",
        maximum_target_year=None,
        locked_forecast_origins=(2020,),
    )

    with pytest.raises(ValueError, match="locked forecast origin"):
        join_oof_to_modeling_data(
            oof,
            data,
            spec=spec,
            model_config=config,
            probability_epsilon=1.0e-12,
        )


def test_fold_specific_distance_and_error_strata_are_deterministic() -> None:
    distance_frame, cuts = assign_distance_context(joined_frame(), (1 / 3, 2 / 3))
    error_frame, thresholds = assign_error_level(distance_frame, (0.25, 0.75))

    contexts_by_cell = distance_frame.set_index("cell_id")["distance_context"].astype(
        "string"
    )
    assert contexts_by_cell.loc[[1, 2]].tolist() == ["near_built", "near_built"]
    assert contexts_by_cell.loc[[3, 4]].tolist() == ["intermediate", "intermediate"]
    assert contexts_by_cell.loc[[5, 6]].tolist() == ["peripheral", "peripheral"]
    assert cuts["fold"].tolist() == [1]
    assert len(thresholds) == 2
    assert set(error_frame["error_level"].astype("string")) == {
        "low_error",
        "middle",
        "high_error",
    }


def test_grouped_and_uncertainty_summaries_are_correct() -> None:
    frame, _ = assign_distance_context(joined_frame(), (1 / 3, 2 / 3))
    frame, _ = assign_error_level(frame, (0.25, 0.75))
    grouped = summarize_diagnostics(frame)
    correlations, quintiles = uncertainty_summaries(frame, bins=3)

    assert set(grouped["group_type"]) == {
        "fold",
        "true_class",
        "distance_context",
        "fold_x_distance_context",
        "fold_x_true_class",
    }
    assert correlations["latent_variance_available"].all()
    assert set(quintiles["latent_uncertainty_quintile"]) == {1, 2, 3}
    assert quintiles.loc[quintiles["fold"].eq("overall"), "rows"].sum() == len(frame)


def test_missing_latent_variance_is_reported_not_synthesized() -> None:
    frame = joined_frame().drop(columns="latent_variance")
    correlations, quintiles = uncertainty_summaries(frame, bins=5)

    assert not correlations["latent_variance_available"].any()
    assert correlations["spearman_log_loss"].isna().all()
    assert quintiles.empty