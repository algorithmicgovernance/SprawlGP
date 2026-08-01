"""Pure tests for the Landsat version 3 selection policy."""

from __future__ import annotations

import pandas as pd

from src.analysis.landsat.build_catalog import (
    select_epoch_protocol,
)


CONFIG = {
    "selection": {
        "minimum_coverage_one_observation_pct": 95.0,
        "preferred_coverage_three_observations_pct": 80.0,
        "coverage_comparison_tolerance_pct": 0.5,
    }
}


def candidate(
    *,
    window: str,
    months: int,
    mode: str,
    coverage_1: float,
    coverage_3: float,
    median: float,
    scenes: int,
) -> dict[str, object]:
    """Create one synthetic exact-window result."""
    return {
        "epoch": 2020,
        "window_name": window,
        "number_of_months": months,
        "window_start": "2020-01-01",
        "window_end": "2020-04-01",
        "sensor_mode": mode,
        "sensor_keys": "LC08",
        "scene_count": scenes,
        "coverage_at_least_1_core_pct": coverage_1,
        "coverage_at_least_3_core_pct": coverage_3,
        "mean_valid_observations_core": median,
        "median_valid_observations_core": median,
        "maximum_valid_observations_core": scenes,
    }


def test_shortest_acceptable_window_is_selected() -> None:
    """Choose a passing three-month window over a passing longer window."""
    frame = pd.DataFrame(
        [
            candidate(
                window="Jan-Mar_3m",
                months=3,
                mode="primary_only",
                coverage_1=99.8,
                coverage_3=84.0,
                median=3.0,
                scenes=8,
            ),
            candidate(
                window="Jan-Jun_6m",
                months=6,
                mode="primary_only",
                coverage_1=100.0,
                coverage_3=99.0,
                median=7.0,
                scenes=15,
            ),
        ]
    )

    selected = select_epoch_protocol(frame, CONFIG)

    assert selected["window_name"] == "Jan-Mar_3m"
    assert bool(selected["threshold_passed"])
    assert (
        selected["selection_status"]
        == "PASS_SHORTEST_ACCEPTABLE"
    )


def test_primary_mode_is_kept_when_it_passes() -> None:
    """Do not add a supplemental sensor when primary data already pass."""
    frame = pd.DataFrame(
        [
            candidate(
                window="Jan-Mar_3m",
                months=3,
                mode="primary_only",
                coverage_1=96.0,
                coverage_3=60.0,
                median=2.0,
                scenes=5,
            ),
            candidate(
                window="Jan-Mar_3m",
                months=3,
                mode="primary_plus_supplemental",
                coverage_1=99.0,
                coverage_3=75.0,
                median=3.0,
                scenes=9,
            ),
        ]
    )

    selected = select_epoch_protocol(frame, CONFIG)

    assert selected["sensor_mode"] == "primary_only"


def test_supplemental_mode_is_used_when_needed() -> None:
    """Use supplemental sensors when they materially recover coverage."""
    frame = pd.DataFrame(
        [
            candidate(
                window="Mar-May_3m",
                months=3,
                mode="primary_only",
                coverage_1=92.0,
                coverage_3=0.0,
                median=1.0,
                scenes=1,
            ),
            candidate(
                window="Mar-May_3m",
                months=3,
                mode="primary_plus_supplemental",
                coverage_1=98.0,
                coverage_3=70.0,
                median=2.0,
                scenes=6,
            ),
        ]
    )

    selected = select_epoch_protocol(frame, CONFIG)

    assert (
        selected["sensor_mode"]
        == "primary_plus_supplemental"
    )
    assert bool(selected["threshold_passed"])


def test_fallback_maximizes_exact_coverage() -> None:
    """Use an explicit best-coverage fallback when no window reaches 95%."""
    frame = pd.DataFrame(
        [
            candidate(
                window="Jul-Sep_3m",
                months=3,
                mode="primary_only",
                coverage_1=31.0,
                coverage_3=0.0,
                median=0.0,
                scenes=1,
            ),
            candidate(
                window="Jan-Dec_12m",
                months=12,
                mode="primary_only",
                coverage_1=91.0,
                coverage_3=50.0,
                median=2.0,
                scenes=14,
            ),
        ]
    )

    selected = select_epoch_protocol(frame, CONFIG)

    assert selected["window_name"] == "Jan-Dec_12m"
    assert not bool(selected["threshold_passed"])
    assert (
        selected["selection_status"]
        == "FALLBACK_BELOW_95"
    )
