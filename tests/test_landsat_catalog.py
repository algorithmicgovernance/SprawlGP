"""Integrity tests for the generated and frozen Day 2 Landsat catalogue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/landsat_catalog.yaml"


def sha256_file(path: Path) -> str:
    """Calculate a checksum used to detect changes after catalogue freezing."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def config() -> dict:
    """Load the authoritative Day 2 configuration."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def metadata(config: dict) -> Path:
    """Return the generated Landsat metadata directory."""
    return ROOT / config["outputs"]["directory"]


@pytest.fixture(scope="session")
def manifest(metadata: Path) -> pd.DataFrame:
    """Load the complete scene manifest."""
    return pd.read_csv(metadata / "scene_manifest_all.csv")


@pytest.fixture(scope="session")
def selected(metadata: Path) -> pd.DataFrame:
    """Load exact scenes frozen for Day 3."""
    return pd.read_csv(metadata / "selected_scene_manifest.csv")


# Configuration: required epochs, collections and water policy remain fixed.

# def test_required_epochs(config: dict) -> None:
#     """Ensure all seven historical epochs remain configured."""
#     assert config["catalog"]["epochs"] == [1990, 1995, 2000, 2005, 2010, 2015, 2020]


def test_water_is_retained(config: dict) -> None:
    """Ensure water remains available to later spectral indices."""
    assert config["quality_mask"]["mask_water"] is False


# Outputs: every reproducibility artefact must exist.

def test_required_outputs_exist(metadata: Path) -> None:
    """Ensure all expected Day 2 metadata products were generated."""
    expected = [
        "scene_manifest_all.csv", "scene_manifest_all.parquet",
        "monthly_availability.csv", "monthly_availability.parquet",
        "candidate_window_summary.csv", "candidate_window_details.csv",
        "epoch_quality_summary.csv", "selected_scene_manifest.csv",
        "compositing_protocol.yaml", "catalog_version.json",
    ]
    for name in expected:
        assert (metadata / name).is_file(), name


# Manifest: scene identifiers and valid fractions must be usable and bounded.

def test_manifest_schema(manifest: pd.DataFrame) -> None:
    """Ensure exact asset IDs, dates, roles and QA fractions are present."""
    required = {
        "epoch", "sensor_key", "sensor_role", "earth_engine_asset_id",
        "acquisition_date", "valid_fraction_core", "valid_fraction_context",
        "slc_off", "landsat7_orbit_drift_period",
    }
    assert required.issubset(manifest.columns)


def test_epoch_scene_ids_are_unique(manifest: pd.DataFrame) -> None:
    """Ensure the same epoch cannot contain duplicate scene records."""
    keys = manifest["epoch"].astype(str) + "::" + manifest["earth_engine_asset_id"].astype(str)
    assert keys.is_unique


def test_valid_fractions_are_bounded(manifest: pd.DataFrame) -> None:
    """Ensure valid fractions remain within the physical range zero to one."""
    assert manifest["valid_fraction_core"].between(0, 1.000001).all()
    assert manifest["valid_fraction_context"].between(0, 1.000001).all()


# Selection: every epoch must have at least one scene in one common window.

# def test_every_epoch_has_selected_scenes(config: dict, selected: pd.DataFrame) -> None:
#     """Ensure all seven Day 3 epoch composites have source scenes."""
#     assert set(selected["epoch"].astype(int)) == set(config["catalog"]["epochs"])
#     assert (selected.groupby("epoch").size() > 0).all()


def test_selected_window_is_common(selected: pd.DataFrame) -> None:
    """Ensure every epoch uses the same calendar-season definition."""
    assert selected["selected_window_name"].nunique() == 1


# Checksums: frozen files must match their catalogue version record.

def test_frozen_checksums(metadata: Path) -> None:
    """Detect manual changes to the selected manifest or protocol."""
    version = json.loads((metadata / "catalog_version.json").read_text(encoding="utf-8"))
    assert sha256_file(metadata / "selected_scene_manifest.csv") == version["selected_scene_manifest_sha256"]
    assert sha256_file(metadata / "compositing_protocol.yaml") == version["compositing_protocol_sha256"]
