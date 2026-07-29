"""
Critical checks are performed:
    1. All submitted composite assets completed successfully.
    2. The outputs are correctly aligned to the project grid (EPSG:32632, 946x1278).
    3. Observation counts are physically plausible (bounded by selected scene count).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from src.analysis.landsat.build_composites import LANDSAT_BANDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/orchestrate_sources.yaml"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the authoritative configuration."""
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture(scope="session")
def metadata_dir(config: dict[str, Any]) -> Path:
    """Return the generated metadata directory."""
    path = Path(config["metadata"]["directory"])
    return path if path.is_absolute() else PROJECT_ROOT / path


@pytest.fixture(scope="session")
def composite_manifest(config: dict[str, Any], metadata_dir: Path) -> pd.DataFrame:
    """Load the validated Landsat composite manifest."""
    return pd.read_csv(metadata_dir / config["metadata"]["landsat_composite_manifest"])


@pytest.fixture(scope="session")
def epoch_quality(config: dict[str, Any], metadata_dir: Path) -> pd.DataFrame:
    """Load validated observation-count statistics."""
    return pd.read_csv(metadata_dir / config["metadata"]["landsat_epoch_quality"])


# -------- Test 1 : Integrity --------
def test_all_composites_succeeded(composite_manifest: pd.DataFrame) -> None:
    """Ensure every submitted composite asset completed successfully."""
    assert not composite_manifest.empty, "No composite assets were generated."
    assert (composite_manifest["status"] == "PASS").all(), (
        "At least one composite asset has a non-PASS status."
    )


# -------- Test 2 : Grid alignment --------
def test_grid_definition_is_exact(composite_manifest: pd.DataFrame) -> None:
    """Ensure all composites use the exact frozen grid (CRS, width, height)."""
    assert (composite_manifest["crs"] == "EPSG:32632").all(), (
        "All assets must use EPSG:32632."
    )
    assert set(composite_manifest["width"].astype(int)) == {946}, (
        "Asset width must be 946 pixels."
    )
    assert set(composite_manifest["height"].astype(int)) == {1278}, (
        "Asset height must be 1278 pixels."
    )


# -------- Test 3 : Physical plausibility --------
def test_observation_counts_are_plausible(epoch_quality: pd.DataFrame) -> None:
    """Ensure observation counts are non-negative and never exceed the number of selected scenes."""
    assert (epoch_quality["minimum_observation_count"] >= 0).all(), (
        "Minimum observation count must be >= 0."
    )
    assert (
        epoch_quality["maximum_observation_count"]
        <= epoch_quality["selected_scene_count"]
    ).all(), (
        "Maximum observation count cannot exceed the number of selected scenes."
    )

    # S anity check: the core area should have at least some coverage
    assert (epoch_quality["covered_grid_pct"] > 0.0).all(), (
        "Every epoch should cover at least some part of the grid."
    )
    
# Critical Landsat 9 harmonisation checks for orchestration.    
def test_landsat9_uses_the_oli_band_mapping() -> None:
    """Ensure LC09 is harmonised exactly like LC08."""
    assert LANDSAT_BANDS["LC09"] == LANDSAT_BANDS["LC08"]
    assert list(LANDSAT_BANDS["LC09"].values()) == [
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
    ]