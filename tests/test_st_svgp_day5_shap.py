"""Focused contracts for Day-5 context-conditioned SHAP preflight."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src.models.evaluation import st_svgp_day5_shap as day5_shap
from src.models.evaluation.st_svgp_day5_diagnostics import HorizonSpec


def annual_spec() -> HorizonSpec:
    return HorizonSpec(
        name="annual_1y",
        label="Annual 1-year ST-SVGP",
        model_config_path=Path("unused.yaml"),
        output_subdirectory="annual_1y",
        maximum_target_year=2019,
        locked_forecast_origins=(),
    )


def sampling_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cell_id = 0
    for fold in (1, 2, 3):
        for y_true in (0, 1):
            for context in day5_shap.PRIMARY_CONTEXTS:
                for error_level in day5_shap.PRIMARY_ERROR_LEVELS:
                    for repetition in range(6):
                        cell_id += 1
                        rows.append(
                            {
                                "cell_id": cell_id,
                                "fold": fold,
                                "forecast_origin": 2010 + fold,
                                "target_year": 2011 + fold,
                                "y_true": y_true,
                                "distance_context": context,
                                "error_level": error_level,
                                "x_center_m": float(cell_id * 10),
                                "y_center_m": float(cell_id * 5),
                                "feature_a": float(cell_id),
                                "feature_b": float((cell_id % 11) + repetition),
                            }
                        )
    rows.append(
        {
            "cell_id": 9999,
            "fold": 3,
            "forecast_origin": 2019,
            "target_year": 2020,
            "y_true": 1,
            "distance_context": "near_built",
            "error_level": "high_error",
            "x_center_m": 0.0,
            "y_center_m": 0.0,
            "feature_a": 0.0,
            "feature_b": 0.0,
        }
    )
    return pd.DataFrame(rows)


def test_background_is_deterministic_bounded_and_uses_real_unlocked_rows() -> None:
    frame = sampling_frame()
    first = day5_shap.select_background_rows(
        frame,
        ["feature_a", "feature_b"],
        spec=annual_spec(),
        size=40,
        random_state=20260809,
    )
    second = day5_shap.select_background_rows(
        frame.sample(frac=1.0, random_state=7),
        ["feature_a", "feature_b"],
        spec=annual_spec(),
        size=40,
        random_state=20260809,
    )

    assert len(first) == 40
    assert first["cell_id"].tolist() == second["cell_id"].tolist()
    assert set(first["cell_id"]).issubset(set(frame["cell_id"]))
    assert 9999 not in set(first["cell_id"])


def test_background_rejects_more_than_fifty_rows() -> None:
    with pytest.raises(ValueError, match="between 1 and 50"):
        day5_shap.select_background_rows(
            sampling_frame(),
            ["feature_a", "feature_b"],
            spec=annual_spec(),
            size=51,
            random_state=1,
        )


def test_explained_sample_respects_strata_cap_seed_and_lock() -> None:
    frame = sampling_frame()
    first = day5_shap.select_explained_rows(
        frame,
        spec=annual_spec(),
        maximum_rows=120,
        rows_per_stratum=5,
        random_state=20260809,
    )
    second = day5_shap.select_explained_rows(
        frame,
        spec=annual_spec(),
        maximum_rows=120,
        rows_per_stratum=5,
        random_state=20260809,
    )

    assert len(first) == 120
    assert first["cell_id"].tolist() == second["cell_id"].tolist()
    assert 9999 not in set(first["cell_id"])
    counts = first.groupby(
        ["fold", "y_true", "distance_context", "error_level"], observed=True
    ).size()
    assert counts.eq(5).all()


def test_explained_sample_cannot_exceed_pre_authorization_cap() -> None:
    with pytest.raises(ValueError, match="cannot exceed 120"):
        day5_shap.select_explained_rows(
            sampling_frame(),
            spec=annual_spec(),
            maximum_rows=121,
            rows_per_stratum=5,
            random_state=1,
        )


def test_representative_contexts_are_real_nearest_to_center_rows() -> None:
    contexts = day5_shap.select_representative_contexts(
        sampling_frame().loc[lambda frame: frame["target_year"].le(2019)],
        "annual_1y",
    )

    assert len(contexts) == 6
    assert set(contexts["context_type"]) == set(day5_shap.PRIMARY_CONTEXTS)
    assert set(contexts["cell_id"]).issubset(set(sampling_frame()["cell_id"]))
    assert contexts["selection_rule"].str.contains("real observation").all()


def test_primary_wrapper_scales_features_and_fixes_space_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_predict_frame(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        validation_data = kwargs["validation_data"]
        captured["features"] = validation_data.features
        captured["coordinates"] = validation_data.coordinates_km
        captured["time_step"] = validation_data.time_step
        rows = len(validation_data.features)
        return np.full(rows, 0.25), np.full(rows, 0.5)

    monkeypatch.setattr("src.models.train_st_svgp.predict_frame", fake_predict_frame)
    wrapper = day5_shap.ContextConditionedCovariatePredictor(
        model=object(),
        posterior=object(),
        preprocessing=SimpleNamespace(
            feature_mean=np.array([1.0, 2.0]),
            feature_scale=np.array([2.0, 4.0]),
            coordinate_center_km=np.array([0.1, 0.2]),
        ),
        config={
            "linear_predictors": ["feature_a", "feature_b"],
            "time": {"origin_year": 2000, "step_years": 5},
            "training": {"prediction_batch_size": 32},
        },
        dtype=object(),
        last_training_step=2.0,
        context_x_m=1000.0,
        context_y_m=2000.0,
        context_origin=2015,
    )

    probability = wrapper(np.array([[1.0, 2.0], [3.0, 6.0]]))

    np.testing.assert_allclose(captured["features"], [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(captured["coordinates"], [[0.9, 1.8], [0.9, 1.8]])
    assert captured["time_step"] == 3.0
    np.testing.assert_allclose(probability, [0.25, 0.25])


def test_reload_preflight_cannot_pass_without_real_fold_state(tmp_path: Path) -> None:
    path = day5_shap.expected_fold_state_path(tmp_path, 1)
    status, keys = day5_shap._state_component_status(path)

    assert not status["model_state_found"]
    assert not status["preprocessing_state_found"]
    assert not status["variational_cvi_state_found"]
    assert not keys


def test_oof_reproduction_rejects_material_difference() -> None:
    with pytest.raises(AssertionError):
        day5_shap.assert_oof_reproduction(
            np.array([0.1, 0.2]),
            np.array([0.1, 0.25]),
            rtol=1.0e-6,
            atol=1.0e-8,
        )


def test_day5_make_targets_contain_no_training_or_final_fit() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    target_names = (
        "day5-st-svgp-preflight",
        "day5-st-svgp-oof-diagnostics",
        "day5-st-svgp-shap-preflight",
        "day5-st-svgp-shap-pilot",
    )
    for name in target_names:
        recipe = makefile.split(f"\n{name}:", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
        assert "train_st_svgp" not in recipe
        assert "final-fit" not in recipe
        assert "--rolling-only" not in recipe