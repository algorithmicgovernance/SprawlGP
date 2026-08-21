"""Static and generated-output checks for the annual Landsat catalogue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from src.analysis.landsat.build_annual_catalog import (
    QUALITY_FLAGS,
    annual_sensor_policy,
    calendar_year_window,
    quality_flag,
    validate_annual_configuration,
)
from src.analysis.orchestration.preflight import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/annual/landsat_catalog_annual.yaml"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the annual catalogue configuration."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_requested_years_are_exact_and_consecutive(config: dict[str, Any]) -> None:
    """Require exactly one observation request for every year from 2000 to 2025."""
    years = validate_annual_configuration(config)
    assert years == list(range(2000, 2026))
    assert len(years) == 26
    assert len(set(years)) == 26


def test_primary_windows_never_use_future_imagery(config: dict[str, Any]) -> None:
    """Represent each year with its exact half-open calendar interval."""
    catalog = config["catalog"]
    assert catalog["temporal_mode"] == "calendar_year"
    assert catalog["window_end_exclusive"] is True
    assert catalog["allow_future_imagery"] is False
    assert catalog["automatic_temporal_fallback"] is False

    for year in catalog["years"]:
        start, end_exclusive = calendar_year_window(int(year))
        assert start == f"{year}-01-01"
        assert end_exclusive == f"{int(year) + 1}-01-01"
        assert pd.Timestamp(end_exclusive) - pd.Timedelta(days=1) == pd.Timestamp(
            f"{year}-12-31"
        )

    manifest = pd.DataFrame(
        {
            "epoch": [2020],
            "sensor_key": ["LC08"],
            "acquisition_date": ["2021-01-01"],
            "earth_engine_asset_id": ["LANDSAT/example"],
            "candidate_for_composite": [True],
        }
    )
    protocol = {"temporal_mode": "calendar_year", "epochs": {2020: {}}}
    with pytest.raises(ValueError, match="outside the represented year"):
        validate_manifest(manifest, protocol, {2020}, {"LC08"})


def test_orchestration_uses_a_median_composite() -> None:
    """Pin the order-independent annual composite reducer."""
    orchestration = yaml.safe_load(
        (ROOT / "configs/annual/orchestrate_sources_annual.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert orchestration["landsat"]["composite_reducer"] == "median"


def test_sensor_eras_resolve_to_supported_collections(config: dict[str, Any]) -> None:
    """Resolve every year once and reject unconfigured Landsat sensor names."""
    supported = set(config["collections"])
    for year in config["catalog"]["years"]:
        policy = annual_sensor_policy(config, int(year))
        selected = set(policy["primary"] + policy["supplemental"])
        diagnostic = set(policy["diagnostic_only"])
        assert policy["primary"]
        assert selected.issubset(supported)
        assert diagnostic.issubset(supported)


def test_frozen_grid_is_reused_exactly(config: dict[str, Any]) -> None:
    """Reference the authoritative grid and its frozen regression copy."""
    study = config["study_area"]
    assert study["grid_specification"] == "data/metadata/grid_specification.json"
    assert study["grid_reference"] == "tests/reference/grid_v1.json"
    grid = json.loads((ROOT / study["grid_specification"]).read_text(encoding="utf-8"))
    reference = json.loads((ROOT / study["grid_reference"]).read_text(encoding="utf-8"))
    for field in ("crs", "resolution_m", "transform", "width", "height", "extent"):
        assert grid[field] == reference[field]


def test_quality_status_is_always_explicit() -> None:
    """Distinguish passing, limited and scene-free annual observations."""
    assert quality_flag(4, 95.0) == "PASS"
    assert quality_flag(4, 94.999) == "LIMITED"
    assert quality_flag(0, 100.0) == "FAIL"
    assert {quality_flag(4, 95.0), quality_flag(4, 90.0), quality_flag(0, 0.0)} == (
        QUALITY_FLAGS
    )


def test_generated_annual_outputs_when_available(config: dict[str, Any]) -> None:
    """Validate all generated years, selected dates and selected sensors."""
    output_dir = ROOT / config["outputs"]["directory"]
    quality_path = output_dir / "annual_quality_summary.csv"
    selected_path = output_dir / "selected_scene_manifest.csv"
    if not quality_path.is_file() or not selected_path.is_file():
        pytest.skip("Annual Earth Engine catalogue has not been generated locally.")

    quality = pd.read_csv(quality_path)
    selected = pd.read_csv(selected_path)
    assert quality["year"].astype(int).tolist() == list(range(2000, 2026))
    assert set(quality["quality_flag"]).issubset(QUALITY_FLAGS)
    assert (quality.loc[quality["quality_flag"] == "PASS", "selected_scene_count"] >= 1).all()

    acquisition_dates = pd.to_datetime(selected["acquisition_date"], errors="raise")
    assert (acquisition_dates.dt.year == selected["year"].astype(int)).all()
    assert set(selected["sensor_key"].astype(str)).issubset(config["collections"])
