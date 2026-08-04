"""Critical integrity tests for the generated Day 2 catalogue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/landsat_catalog.yaml"


def sha256_file(path: Path) -> str:
    """Calculate a checksum used after catalogue freezing."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the authoritative Landsat catalogue configuration."""
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def metadata(config: dict[str, Any]) -> Path:
    """Return the generated Landsat metadata directory."""
    return ROOT / config["outputs"]["directory"]


@pytest.fixture(scope="session")
def manifest(metadata: Path) -> pd.DataFrame:
    """Load the complete diagnostic scene manifest."""
    return pd.read_csv(metadata / "scene_manifest_all.csv")


@pytest.fixture(scope="session")
def selected(metadata: Path) -> pd.DataFrame:
    """Load exact scenes frozen for orchestration."""
    return pd.read_csv(
        metadata / "selected_scene_manifest.csv"
    )


def test_required_epochs_and_landsat9(
    config: dict[str, Any],
) -> None:
    """Ensure 2025 and Landsat 9 remain configured."""
    assert config["catalog"]["epochs"] == [
        # 1990,
        # 1995,
        2000,
        2005,
        2010,
        2015,
        2020,
        2025,
    ]
    assert config["collections"]["LC09"]["id"] == (
        "LANDSAT/LC09/C02/T1_L2"
    )
    policy = config["epoch_sensor_policy"][2025]
    assert set(policy["primary"]) == {"LC08", "LC09"}


def test_water_is_retained(
    config: dict[str, Any],
) -> None:
    """Ensure water remains available to later indices."""
    assert config["quality_mask"]["mask_water"] is False


def test_manifest_schema_and_unique_ids(
    manifest: pd.DataFrame,
) -> None:
    """Ensure exact scene IDs and bounded QA fractions exist."""
    required = {
        "epoch",
        "sensor_key",
        "sensor_role",
        "earth_engine_asset_id",
        "acquisition_date",
        "valid_fraction_core",
        "valid_fraction_context",
    }
    assert required.issubset(manifest.columns)

    keys = (
        manifest["epoch"].astype(str)
        + "::"
        + manifest["earth_engine_asset_id"].astype(str)
    )
    assert keys.is_unique
    assert manifest["valid_fraction_core"].between(
        0,
        1.000001,
    ).all()
    assert manifest["valid_fraction_context"].between(
        0,
        1.000001,
    ).all()


def test_selected_epochs_are_configured_and_2025_is_recovered(
    config: dict[str, Any],
    selected: pd.DataFrame,
) -> None:
    """Accept missing early epochs but require a usable 2025 set."""
    configured = set(config["catalog"]["epochs"])
    observed = set(selected["epoch"].astype(int))
    assert observed.issubset(configured)
    assert 2025 in observed

    selected_2025 = selected[
        selected["epoch"].astype(int) == 2025
    ]
    assert len(selected_2025) >= 2
    assert set(selected_2025["sensor_key"]).issubset(
        {"LC08", "LC09"}
    )


def test_frozen_checksums_when_outputs_exist(
    metadata: Path,
) -> None:
    """Detect manual changes after the revised catalogue is frozen."""
    version_path = metadata / "catalog_version.json"

    if not version_path.is_file():
        pytest.skip("The revised catalogue has not been frozen.")

    version = json.loads(
        version_path.read_text(encoding="utf-8")
    )
    assert sha256_file(
        metadata / "selected_scene_manifest.csv"
    ) == version["selected_scene_manifest_sha256"]
    assert sha256_file(
        metadata / "compositing_protocol.yaml"
    ) == version["compositing_protocol_sha256"]
