"""Focused tests for the isolated annual persistence diagnostic."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from src.analysis.annual_dataset import diagnostic_validation as diagnostic

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/annual/annual_diagnostic.yaml"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the fixed annual diagnostic configuration."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_config_pins_two_240_cell_episode_samples(config: dict[str, Any]) -> None:
    """Pin the approved episodes and exact 70/40/5/5 initial quotas."""
    assert len(config["episodes"]) == 2
    assert list(config["episodes"].values()) == [
        {"before_year": 2015, "peak_year": 2016, "after_year": 2017},
        {"before_year": 2019, "peak_year": 2020, "after_year": 2021},
    ]
    assert config["initial_quotas_per_episode"] == {
        "new_built_then_reversal": 70,
        "new_built_persistent_raw": 40,
        "stable_nonbuilt_control": 5,
        "stable_built_control": 5,
    }
    assert sum(config["initial_quotas_per_episode"].values()) == 120
    assert len(config["episodes"]) * sum(config["initial_quotas_per_episode"].values()) == 240
    assert config["seed"] == 20260809
    assert config["minimum_spacing_m"] == 250
    assert config["candidate_oversampling_factor"] == 20
    assert config["reserve_target_per_episode"] == {
        "new_built_then_reversal": 140,
        "new_built_persistent_raw": 80,
        "stable_nonbuilt_control": 10,
        "stable_built_control": 10,
    }


def test_candidate_requests_are_bounded_per_stratum(config: dict[str, Any]) -> None:
    """Request each stratum independently below the safe getInfo limit."""
    expected = {
        "new_built_then_reversal": 4200,
        "new_built_persistent_raw": 2400,
        "stable_nonbuilt_control": 300,
        "stable_built_control": 300,
    }
    observed = {
        sample_type: diagnostic.candidate_request_size(config, sample_type)
        for sample_type in diagnostic.SAMPLE_TYPES
    }
    assert observed == expected
    assert max(observed.values()) <= diagnostic.MAX_GETINFO_FEATURES
    assert len(config["episodes"]) * len(observed) == 8


def test_candidate_request_guard_never_silently_reduces(config: dict[str, Any]) -> None:
    """Require further batching if any one-class request exceeds the safe limit."""
    oversized = yaml.safe_load(yaml.safe_dump(config))
    oversized["reserve_target_per_episode"]["new_built_then_reversal"] = 156
    with pytest.raises(ValueError, match="batch this stratum further"):
        diagnostic.candidate_request_size(oversized, "new_built_then_reversal")


@pytest.mark.parametrize(
    ("raw", "final", "expected"),
    [
        ((0, 1, 0), (0, 1, 1), "new_built_then_reversal"),
        ((0, 1, 1), (0, 1, 1), "new_built_persistent_raw"),
        ((0, 0, 0), (0, 0, 0), "stable_nonbuilt_control"),
        ((1, 1, 1), (1, 1, 1), "stable_built_control"),
        ((0, 1, 0), (1, 1, 1), None),
        ((0, 0, 0), (0, 1, 1), None),
    ],
)
def test_state_definitions_are_exact(
    raw: tuple[int, int, int], final: tuple[int, int, int], expected: str | None
) -> None:
    """Require exact raw patterns and final-state guards for each stratum."""
    row = {
        **dict(zip(("raw_state_before", "raw_state_peak", "raw_state_after"), raw, strict=True)),
        **dict(
            zip(("final_state_before", "final_state_peak", "final_state_after"), final, strict=True)
        ),
    }
    assert diagnostic.sample_type_for_states(row) == expected


def test_hard_thinning_is_deterministic_and_never_relaxes() -> None:
    """Keep only points satisfying the exact supplied spacing."""
    candidates = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "x_center_m": [0.0, 100.0, 250.0, 500.0],
            "y_center_m": [0.0, 0.0, 0.0, 0.0],
        }
    )
    first = diagnostic.hard_greedy_thin(candidates, 4, 250)
    second = diagnostic.hard_greedy_thin(candidates, 4, 250)
    assert first["cell_id"].tolist() == [1, 3, 4]
    pd.testing.assert_frame_equal(first, second)
    coordinates = first[["x_center_m", "y_center_m"]].to_numpy()
    distances = np.sqrt(((coordinates[:, None] - coordinates[None, :]) ** 2).sum(axis=2))
    assert distances[np.triu_indices(len(first), k=1)].min() >= 250
    assert len(first) < 4


def synthetic_candidates(config: dict[str, Any]) -> pd.DataFrame:
    """Create ample globally unique candidates spaced 300 metres apart."""
    records = []
    cell_id = 0
    for episode_index, (episode, years) in enumerate(config["episodes"].items()):
        for type_index, sample_type in enumerate(diagnostic.SAMPLE_TYPES):
            target = (
                config["initial_quotas_per_episode"][sample_type]
                + config["reserve_target_per_episode"][sample_type]
            )
            for number in range(target):
                cell_id += 1
                records.append(
                    {
                        "episode": episode,
                        "sample_type": sample_type,
                        "before_year": years["before_year"],
                        "peak_year": years["peak_year"],
                        "after_year": years["after_year"],
                        "cell_id": cell_id,
                        "x_center_m": number * 300.0,
                        "y_center_m": (episode_index * 10 + type_index) * 1000.0,
                    }
                )
    return pd.DataFrame(records)


def test_selection_is_exact_unique_disjoint_and_deterministic(config: dict[str, Any]) -> None:
    """Produce exact initial and reserve quotas with no duplicate identities."""
    candidates = synthetic_candidates(config)
    first_initial, first_reserve, shortages = diagnostic.select_initial_and_reserve(
        candidates, config
    )
    second_initial, second_reserve, _ = diagnostic.select_initial_and_reserve(candidates, config)
    assert len(first_initial) == 240
    assert len(first_reserve) == 480
    assert not shortages
    assert first_initial["sample_id"].is_unique
    assert first_initial["cell_id"].is_unique
    assert first_reserve["sample_id"].is_unique
    assert set(first_initial["sample_id"]).isdisjoint(first_reserve["sample_id"])
    assert set(first_initial["cell_id"]).isdisjoint(first_reserve["cell_id"])
    assert first_initial.groupby("episode").size().to_dict() == {
        "episode_2016": 120,
        "episode_2020": 120,
    }
    for episode in config["episodes"]:
        counts = (
            first_initial[first_initial["episode"] == episode]
            .groupby("sample_type")
            .size()
            .to_dict()
        )
        assert counts == config["initial_quotas_per_episode"]
    pd.testing.assert_frame_equal(first_initial, second_initial)
    pd.testing.assert_frame_equal(first_reserve, second_reserve)


def test_initial_shortage_raises_instead_of_relaxing(config: dict[str, Any]) -> None:
    """Fail an initial quota when hard spacing leaves too few points."""
    candidates = synthetic_candidates(config)
    mask = (
        (candidates["episode"] == "episode_2016")
        & (candidates["sample_type"] == "new_built_then_reversal")
    )
    candidates.loc[mask, "x_center_m"] = 0.0
    with pytest.raises(RuntimeError, match="Initial quota shortage"):
        diagnostic.select_initial_and_reserve(candidates, config)


def test_reviewer_frame_is_exact_blank_blinded_and_deterministic() -> None:
    """Expose only reviewer-safe fields in a stable randomized order."""
    sample = pd.DataFrame(
        {
            "sample_id": [f"sample_{number:03d}" for number in range(30)],
            "before_year": [2015] * 30,
            "peak_year": [2016] * 30,
            "after_year": [2017] * 30,
            "longitude": np.linspace(11.4, 11.6, 30),
            "latitude": np.linspace(3.7, 4.0, 30),
            "episode": ["hidden"] * 30,
            "sample_type": ["hidden"] * 30,
            "raw_state_peak": [1] * 30,
            "ndbi_peak": [0.1] * 30,
        }
    )
    first = diagnostic.build_review_frame(sample, 20260809)
    second = diagnostic.build_review_frame(sample, 20260809)
    assert first.columns.tolist() == diagnostic.REVIEW_COLUMNS
    assert first["sample_id"].tolist() != sample["sample_id"].tolist()
    assert set(first["sample_id"]) == set(sample["sample_id"])
    assert (first[diagnostic.ANNOTATION_COLUMNS] == "").all().all()
    assert not {"episode", "sample_type", "raw_state_peak", "ndbi_peak"}.intersection(first.columns)
    pd.testing.assert_frame_equal(first, second)


def test_output_paths_are_isolated_from_legacy_validation(config: dict[str, Any]) -> None:
    """Never address the old five-year manual-label artifacts."""
    diagnostic.validate_config(config, ROOT)
    configured = {str(ROOT / value) for value in config["outputs"].values()}
    assert str(ROOT / "data/validation/manual_labels.csv") not in configured
    assert all("data/validation/annual_diagnostic" in path for path in configured)


def test_module_has_no_earth_engine_mutation_or_export_path() -> None:
    """Keep the diagnostic implementation structurally read-only."""
    source = inspect.getsource(diagnostic)
    forbidden = ("Export.image", "Export.table", ".start()", "createAsset", "deleteAsset")
    assert not any(token in source for token in forbidden)


def test_review_label_domains_are_restricted() -> None:
    """Accept only 0, 1, U and the three allowed confidence levels."""
    frame = pd.DataFrame(columns=diagnostic.REVIEW_COLUMNS)
    diagnostic.validate_review_labels(frame)
    invalid = pd.DataFrame([{column: "" for column in diagnostic.REVIEW_COLUMNS}])
    invalid.loc[0, "reference_label_peak"] = "maybe"
    with pytest.raises(ValueError, match="Invalid labels"):
        diagnostic.validate_review_labels(invalid)


def test_wilson_interval_matches_known_example() -> None:
    """Pin the standard 95 percent Wilson interval for 50 of 100."""
    lower, upper = diagnostic.wilson_interval(50, 100)
    assert lower == pytest.approx(0.40383153, abs=1e-8)
    assert upper == pytest.approx(0.59616847, abs=1e-8)