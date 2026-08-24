"""Contracts for strictly temporal annual ST-SVGP post-hoc calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.models.evaluation import st_svgp_annual_temporal_calibration as temporal


def _predictions() -> pd.DataFrame:
    rows = []
    probabilities = {
        1: [0.05, 0.20, 0.65, 0.85],
        2: [0.10, 0.30, 0.60, 0.90],
        3: [0.15, 0.35, 0.70, 0.95],
    }
    target_years = {1: 2011, 2: 2016, 3: 2019}
    for fold in (1, 2, 3):
        for row_number, (target, probability) in enumerate(
            zip((0, 0, 1, 1), probabilities[fold], strict=True)
        ):
            rows.append(
                {
                    "cell_id": fold * 10 + row_number,
                    "forecast_origin": target_years[fold] - 1,
                    "target_year": target_years[fold],
                    "target_transition_1y": target,
                    "probability_raw": probability,
                    "fold": fold,
                    "latent_variance": 0.1,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["platt", "beta"])
def test_temporal_calibration_uses_only_strictly_earlier_folds(method: str) -> None:
    predictions, fits = temporal.temporal_calibrate(_predictions(), method)

    assert sorted(predictions["fold"].unique().tolist()) == [2, 3]
    assert fits["fold"].tolist() == [2, 3]
    assert fits["calibration_folds"].tolist() == ["1", "1,2"]
    assert fits["calibration_rows"].tolist() == [4, 8]
    assert fits["evaluation_rows"].tolist() == [4, 4]
    assert predictions[temporal.CALIBRATED_PROBABILITY_COLUMN].between(0.0, 1.0).all()


@pytest.mark.parametrize("method", ["platt", "beta"])
def test_future_fold_perturbation_does_not_change_fold_2(method: str) -> None:
    original_predictions, original_fits = temporal.temporal_calibrate(
        _predictions(), method
    )
    changed = _predictions()
    changed.loc[changed["fold"] == 3, "probability_raw"] = [0.99, 0.98, 0.02, 0.01]
    perturbed_predictions, perturbed_fits = temporal.temporal_calibrate(changed, method)

    original_fold_2 = original_predictions.loc[
        original_predictions["fold"] == 2, temporal.CALIBRATED_PROBABILITY_COLUMN
    ]
    perturbed_fold_2 = perturbed_predictions.loc[
        perturbed_predictions["fold"] == 2, temporal.CALIBRATED_PROBABILITY_COLUMN
    ]
    np.testing.assert_allclose(original_fold_2, perturbed_fold_2)
    pd.testing.assert_series_equal(
        original_fits.loc[original_fits["fold"] == 2].iloc[0],
        perturbed_fits.loc[perturbed_fits["fold"] == 2].iloc[0],
    )


def test_target_year_2020_is_rejected_before_outputs(tmp_path: Path) -> None:
    predictions = _predictions()
    predictions.loc[predictions.index[-1], "target_year"] = 2020
    predictions_path = tmp_path / "raw.parquet"
    output_directory = tmp_path / "calibration"
    predictions.to_parquet(predictions_path, index=False)

    with pytest.raises(ValueError, match="target_year >= 2020"):
        temporal.run(predictions_path, output_directory)

    assert not output_directory.exists()


def test_run_preserves_raw_artifact_and_writes_only_isolated_outputs(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "raw.parquet"
    output_directory = tmp_path / "calibration"
    _predictions().to_parquet(predictions_path, index=False)
    raw_before = predictions_path.read_bytes()

    result = temporal.run(predictions_path, output_directory)

    assert predictions_path.read_bytes() == raw_before
    assert sorted(path.name for path in output_directory.iterdir()) == [
        "calibrated_predictions_beta.parquet",
        "calibrated_predictions_platt.parquet",
        "temporal_calibration_comparison.csv",
        "temporal_calibration_diagnostic.md",
        "temporal_calibration_metrics.csv",
    ]
    assert set(pd.read_csv(result["comparison"])["method"]) == {
        "raw",
        "platt",
        "beta",
    }
    for key in ("platt_predictions", "beta_predictions"):
        calibrated = pd.read_parquet(result[key])
        assert calibrated[temporal.CALIBRATED_PROBABILITY_COLUMN].between(0.0, 1.0).all()


def test_comparison_reuses_repository_metric_utilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _predictions()
    calibrated_predictions = {}
    fit_tables = {}
    for method in ("platt", "beta"):
        calibrated_predictions[method], fit_tables[method] = temporal.temporal_calibrate(
            raw, method
        )
    calls = {"probabilistic": 0, "calibration": 0}
    retained_probabilistic_metrics = temporal.probabilistic_metrics
    retained_calibration_metrics = temporal.calibration_metrics

    def count_probabilistic(*args: object, **kwargs: object) -> dict[str, float]:
        calls["probabilistic"] += 1
        return retained_probabilistic_metrics(*args, **kwargs)

    def count_calibration(*args: object, **kwargs: object) -> dict[str, float]:
        calls["calibration"] += 1
        return retained_calibration_metrics(*args, **kwargs)

    monkeypatch.setattr(temporal, "probabilistic_metrics", count_probabilistic)
    monkeypatch.setattr(temporal, "calibration_metrics", count_calibration)

    comparison = temporal.build_comparison(raw, calibrated_predictions, fit_tables)

    assert calls == {"probabilistic": 6, "calibration": 6}
    assert len(comparison) == 6


def test_make_target_runs_only_temporal_calibration() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    target = makefile.split("\nst-svgp-annual-3000-calibration:", maxsplit=1)[1]
    target = target.split("\n\n", maxsplit=1)[0]

    assert "st_svgp_annual_temporal_calibration" in target
    assert "train_st_svgp" not in target
    assert "rolling-only" not in target
    assert "final-fit" not in target