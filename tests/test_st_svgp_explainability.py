"""Focused contracts for final context-conditioned ST-SVGP SHAP analysis."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from src.models.evaluation import st_svgp_explainability as explainability
from src.models.evaluation.st_svgp_day5_diagnostics import HorizonSpec


def annual_spec() -> HorizonSpec:
    return HorizonSpec(
        name="annual_1y",
        label="Annual",
        model_config_path=Path("annual.yaml"),
        output_subdirectory="annual_1y",
        maximum_target_year=2019,
        locked_forecast_origins=(),
    )


def five_year_spec() -> HorizonSpec:
    return HorizonSpec(
        name="five_year_5y",
        label="Five year",
        model_config_path=Path("five.yaml"),
        output_subdirectory="five_year_5y",
        maximum_target_year=None,
        locked_forecast_origins=(2020,),
    )


def sampling_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        for context_number, context in enumerate(explainability.PRIMARY_CONTEXTS):
            for row_number in range(30):
                cell_id = fold * 1000 + context_number * 100 + row_number
                row: dict[str, Any] = {
                    "cell_id": cell_id,
                    "fold": fold,
                    "forecast_origin": 2009 + fold,
                    "target_year": 2010 + fold,
                    "distance_context": context,
                    "x_center_m": float(row_number * 30 + context_number * 1000),
                    "y_center_m": float(row_number * 15 + fold * 1000),
                }
                for feature_number, feature in enumerate(
                    explainability.EXPECTED_FEATURES["annual_1y"]
                ):
                    row[feature] = float(cell_id + feature_number * row_number)
                rows.append(row)
    return pd.DataFrame(rows)


def test_exact_perturbable_feature_contract_and_kernel_budget() -> None:
    assert set(explainability.EXPECTED_FEATURES) == {"annual_1y", "five_year_5y"}
    assert all(len(features) == 9 for features in explainability.EXPECTED_FEATURES.values())
    for excluded in ("x_center_m", "y_center_m", "forecast_origin", "time_step"):
        assert all(excluded not in features for features in explainability.EXPECTED_FEATURES.values())
    assert explainability.NSAMPLES == 128


def test_background_is_exactly_forty_deterministic_real_rows() -> None:
    frame = sampling_frame()
    features = list(explainability.EXPECTED_FEATURES["annual_1y"])
    first = explainability.select_background(frame, features)
    second = explainability.select_background(
        frame.sample(frac=1.0, random_state=11), features
    )

    assert len(first) == 40
    assert first["cell_id"].tolist() == second["cell_id"].tolist()
    assert set(first["cell_id"]).issubset(set(frame["cell_id"]))
    assert not first.duplicated(["cell_id", "forecast_origin"]).any()


def test_six_contexts_use_real_centroid_rows_and_at_most_ten_explanations() -> None:
    frame = sampling_frame()
    contexts, explained = explainability.select_contexts_and_explanations(
        frame,
        list(explainability.EXPECTED_FEATURES["annual_1y"]),
        horizon="annual_1y",
    )

    assert len(contexts) == 6
    assert set(contexts["distance_context"]) == set(explainability.PRIMARY_CONTEXTS)
    assert set(contexts["context_cell_id"]).issubset(set(frame["cell_id"]))
    assert explained.groupby("context_id").size().eq(10).all()
    assert not explained.duplicated(["context_id", "cell_id", "forecast_origin"]).any()


def test_prediction_wrapper_scales_covariates_and_fixes_space_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_predict_frame(**kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        validation_data = kwargs["validation_data"]
        captured["features"] = validation_data.features
        captured["coordinates"] = validation_data.coordinates_km
        captured["origin"] = validation_data.origin
        captured["time_step"] = validation_data.time_step
        return np.full(len(validation_data.features), 0.4), np.zeros(
            len(validation_data.features)
        )

    monkeypatch.setattr(explainability, "predict_frame", fake_predict_frame)
    state = SimpleNamespace(
        preprocessing=SimpleNamespace(
            feature_mean=np.array([1.0, 2.0]),
            feature_scale=np.array([2.0, 4.0]),
            coordinate_center_km=np.array([0.1, 0.2]),
        ),
        config={
            "linear_predictors": ["a", "b"],
            "time": {"origin_year": 2000, "step_years": 5},
        },
        model=object(),
        posterior=object(),
        last_training_step=2.0,
        dtype=object(),
    )
    predictor = explainability.ContextConditionedPredictor(
        state=state,
        context_x_m=1000.0,
        context_y_m=2000.0,
        context_origin=2015,
    )

    probability = predictor(np.array([[1.0, 2.0], [3.0, 6.0]]))

    np.testing.assert_allclose(captured["features"], [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(captured["coordinates"], [[0.9, 1.8], [0.9, 1.8]])
    assert captured["origin"] == 2015
    assert captured["time_step"] == 3.0
    np.testing.assert_allclose(probability, [0.4, 0.4])


def test_locked_periods_are_rejected() -> None:
    annual = pd.DataFrame({"forecast_origin": [2019], "target_year": [2020]})
    five_year = pd.DataFrame({"forecast_origin": [2020], "target_year": [2025]})
    with pytest.raises(ValueError, match="locked target year"):
        explainability._validate_pretest_rows(annual, annual_spec())
    with pytest.raises(ValueError, match="locked forecast origin"):
        explainability._validate_pretest_rows(five_year, five_year_spec())


def test_shap_output_validation_requires_finite_additive_probabilities() -> None:
    values = np.array([[0.1, -0.05], [0.2, 0.1]])
    predicted = np.array([0.55, 0.8])
    errors = explainability.validate_shap_result(values, 0.5, predicted)
    np.testing.assert_allclose(errors, 0.0, atol=1.0e-12)

    with pytest.raises(ValueError, match="finite"):
        explainability.validate_shap_result(
            np.array([[np.nan, 0.0]]), 0.5, np.array([0.5])
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        explainability.validate_shap_result(
            np.array([[0.0, 0.0]]), 0.5, np.array([1.2])
        )


def test_run_checks_all_six_states_before_horizon_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = [annual_spec(), five_year_spec()]
    calls: list[tuple[str, int]] = []
    frame = pd.DataFrame({"forecast_origin": [2015], "target_year": [2019]})
    monkeypatch.setattr(explainability, "load_yaml", lambda _: {})
    monkeypatch.setattr(explainability, "resolve_horizons", lambda _: specs)
    monkeypatch.setattr(
        explainability,
        "prepare_horizon_diagnostics",
        lambda *_: (frame, pd.DataFrame(), pd.DataFrame(), {}),
    )

    def fake_sanity(
        spec: HorizonSpec, _: pd.DataFrame, *, fold: int
    ) -> tuple[Any, dict[str, Any]]:
        calls.append((spec.name, fold))
        return object(), {"horizon": spec.name, "fold": fold, "status": "PASS"}

    monkeypatch.setattr(explainability, "sanity_check_state", fake_sanity)
    monkeypatch.setattr(
        explainability,
        "_run_horizon",
        lambda spec, *_: {
            "runtime_seconds": 1.0,
            "contexts": 6,
            "background_rows": 40,
            "explained_rows": 60,
            "nsamples": 128,
            "mean_absolute_additivity_error": 0.0,
            "maximum_absolute_additivity_error": 0.0,
            "contexts_frame": pd.DataFrame({"horizon": [spec.name]}),
        },
    )
    monkeypatch.setattr(explainability, "_write_report", lambda *_: None)

    result = explainability.run(output_root=tmp_path / "explainability")

    assert calls == [
        ("annual_1y", 1),
        ("annual_1y", 2),
        ("annual_1y", 3),
        ("five_year_5y", 1),
        ("five_year_5y", 2),
        ("five_year_5y", 3),
    ]
    assert result["six_state_sanity"] == "PASS"


def test_final_target_is_isolated_and_contains_no_training_or_reconstruction() -> None:
    assert explainability.OUTPUT_ROOT == Path(
        "reports/modeling/st_svgp_explainability"
    )
    source = inspect.getsource(explainability.run)
    assert "run_fold" not in source
    assert "reconstruct_fold" not in source
    makefile = Path("Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("\nst-svgp-shap:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "st_svgp_explainability" in recipe
    for forbidden in ("train_st_svgp", "reconstruction", "final-fit", "calibrat", "conformal"):
        assert forbidden not in recipe


def test_all_six_verified_state_paths_exist_and_match_feature_order() -> None:
    for horizon, features in explainability.EXPECTED_FEATURES.items():
        for fold in (1, 2, 3):
            directory = explainability.state_directory(
                horizon, fold, root=explainability.STATE_ROOT
            )
            metadata = pd.read_json(directory / "metadata.json", typ="series")
            assert metadata["feature_names"] == list(features)
            assert metadata["locked_rows_used"] == 0
            assert (directory / "st_svgp_state.npz").is_file()