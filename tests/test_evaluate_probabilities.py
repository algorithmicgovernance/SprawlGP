"""Tests for shared five-year and isolated annual probability evaluation."""

from pathlib import Path

import pandas as pd
import pytest
from src.models.evaluation import evaluate_probabilities
from src.models.evaluation.evaluate_probabilities import (
    ModelInput,
    evaluate_model,
    resolve_columns,
)


def test_resolve_columns_supports_annual_target() -> None:
    frame = pd.DataFrame(
        {
            "target_transition_1y": [0, 1],
            "probability_raw": [0.1, 0.9],
        }
    )

    assert resolve_columns(frame) == (
        "target_transition_1y",
        "probability_raw",
    )


def test_resolve_columns_preserves_five_year_precedence() -> None:
    frame = pd.DataFrame(
        {
            "target_transition_5y": [0, 1],
            "target_transition_1y": [1, 0],
            "probability_raw": [0.1, 0.9],
        }
    )

    assert resolve_columns(frame) == (
        "target_transition_5y",
        "probability_raw",
    )


def test_run_annual_uses_isolated_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"prediction_set_coverage": "annual"}

    monkeypatch.setattr(evaluate_probabilities, "run", fake_run)

    paths = evaluate_probabilities.run_annual()

    assert captured["output_directory"] == Path(
        "reports/modeling/st_svgp_annual/metrics"
    )
    models = captured["models"]
    assert isinstance(models, list)
    assert models == [
        ModelInput(
            name="st_svgp_annual",
            path=Path(
                "reports/modeling/st_svgp_annual/predictions/"
                "st_svgp_oof_predictions.parquet"
            ),
            target_column="target_transition_1y",
            target_year_column="target_year",
            maximum_target_year=2019,
        )
    ]
    assert paths == {"prediction_set_coverage": "annual"}


def test_annual_evaluation_rejects_locked_target_years(tmp_path: Path) -> None:
    predictions_path = tmp_path / "annual_oof.parquet"
    pd.DataFrame(
        {
            "target_transition_1y": [0, 1],
            "target_year": [2019, 2020],
            "probability_raw": [0.1, 0.9],
            "fold": [1, 2],
        }
    ).to_parquet(predictions_path, index=False)

    with pytest.raises(ValueError, match="locked target years after 2019"):
        evaluate_model(
            model=ModelInput(
                name="st_svgp_annual",
                path=predictions_path,
                target_column="target_transition_1y",
                target_year_column="target_year",
                maximum_target_year=2019,
            ),
            n_bins=10,
            strategy="quantile",
            target_coverage=0.80,
        )


def test_annual_mondrian_writes_only_diagnostic_outputs(tmp_path: Path) -> None:
    predictions_path = tmp_path / "annual_oof.parquet"
    marginal_path = tmp_path / "prediction_set_coverage_80.csv"
    output_directory = tmp_path / "metrics"
    pd.DataFrame(
        {
            "target_transition_1y": [0, 1, 0, 1, 0, 1],
            "target_year": [2011, 2011, 2016, 2016, 2019, 2019],
            "probability_raw": [0.1, 0.6, 0.2, 0.7, 0.3, 0.8],
            "fold": [1, 1, 2, 2, 3, 3],
        }
    ).to_parquet(predictions_path, index=False)
    pd.DataFrame(
        {
            "model": ["st_svgp_annual", "st_svgp_annual"],
            "fold": [2, 3],
            "empirical_coverage": [0.5, 0.5],
            "positive_class_coverage": [0.0, 0.0],
            "negative_class_coverage": [1.0, 1.0],
            "average_set_size": [0.5, 0.5],
            "singleton_rate": [0.5, 0.5],
            "both_labels_rate": [0.0, 0.0],
            "empty_set_rate": [0.5, 0.5],
        }
    ).to_csv(marginal_path, index=False)
    marginal_before = marginal_path.read_bytes()

    paths = evaluate_probabilities.run_annual_mondrian(
        output_directory=output_directory,
        model=ModelInput(
            name="st_svgp_annual",
            path=predictions_path,
            target_column="target_transition_1y",
            target_year_column="target_year",
            maximum_target_year=2019,
        ),
        marginal_coverage_path=marginal_path,
    )

    mondrian = pd.read_csv(paths["mondrian_coverage"])
    assert list(mondrian["fold"]) == [2, 3]
    assert list(mondrian["calibration_rows"]) == [2, 4]
    assert list(mondrian["calibration_positive_rows"]) == [1, 2]
    assert list(mondrian["calibration_negative_rows"]) == [1, 2]
    assert marginal_path.read_bytes() == marginal_before
    assert sorted(path.name for path in output_directory.iterdir()) == [
        "coverage_marginal_vs_mondrian.csv",
        "mondrian_coverage_diagnostic.md",
        "prediction_set_coverage_80_mondrian.csv",
    ]


def test_annual_mondrian_rejects_target_year_2020(tmp_path: Path) -> None:
    predictions_path = tmp_path / "annual_oof.parquet"
    pd.DataFrame(
        {
            "target_transition_1y": [0, 1],
            "target_year": [2019, 2020],
            "probability_raw": [0.1, 0.9],
            "fold": [1, 2],
        }
    ).to_parquet(predictions_path, index=False)

    with pytest.raises(ValueError, match="locked target years after 2019"):
        evaluate_probabilities.run_annual_mondrian(
            output_directory=tmp_path / "metrics",
            model=ModelInput(
                name="st_svgp_annual",
                path=predictions_path,
                target_column="target_transition_1y",
                target_year_column="target_year",
                maximum_target_year=2019,
            ),
            marginal_coverage_path=tmp_path / "unused.csv",
        )

    assert not (tmp_path / "metrics").exists()