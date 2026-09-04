"""Isolation checks for annual Landsat configuration and destinations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ANNUAL_CONFIG_DIR = ROOT / "configs/annual"
CATALOG_PATH = ANNUAL_CONFIG_DIR / "landsat_catalog_annual.yaml"
ORCHESTRATION_PATH = ANNUAL_CONFIG_DIR / "orchestrate_sources_annual.yaml"
ANNUAL_ASSET_ROOT = "projects/urban-sprawl-ssa/assets/sprawlgp/annual_v1"


def load_config(path: Path) -> dict[str, Any]:
    """Load one annual YAML configuration."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_annual_asset_root_and_overwrite_guard() -> None:
    """Pin annual exports to the isolated non-overwriting asset root."""
    config = load_config(ORCHESTRATION_PATH)
    assert config["exports"]["asset_root"] == ANNUAL_ASSET_ROOT
    assert config["exports"]["overwrite_existing_assets"] is False
    assert config["orchestration"]["landsat_only"] is True


def test_annual_local_paths_are_isolated() -> None:
    """Keep annual metadata, checkpoints and reports below annual directories."""
    catalog = load_config(CATALOG_PATH)
    orchestration = load_config(ORCHESTRATION_PATH)
    assert catalog["outputs"]["directory"].startswith("data/metadata/annual/")
    assert catalog["outputs"]["checkpoint_directory"] == (
        "data/metadata/annual/landsat/_checkpoints"
    )
    assert catalog["outputs"]["report_directory"].startswith("reports/annual/")
    assert orchestration["metadata"]["directory"].startswith("data/metadata/annual/")
    assert orchestration["exports"]["task_manifest"].startswith("data/metadata/annual/")
    assert orchestration["reports"]["directory"].startswith("reports/annual/")


def test_no_annual_config_references_legacy_v2_assets() -> None:
    """Reject any legacy five-year asset root reference in annual configuration."""
    annual_files = [path for path in ANNUAL_CONFIG_DIR.rglob("*") if path.is_file()]
    assert annual_files
    for path in annual_files:
        assert "/assets/sprawlgp/v2" not in path.read_text(encoding="utf-8"), path


def test_intended_annual_landsat_asset_names() -> None:
    """Derive the exact composite and valid-count destinations for all years."""
    config = load_config(ORCHESTRATION_PATH)
    folder = f"{config['exports']['asset_root']}/{config['exports']['landsat_folder']}"
    years = config["landsat"]["epochs"]
    assert len(years) == 26
    assert f"{folder}/composite_2000" == f"{ANNUAL_ASSET_ROOT}/landsat/composite_2000"
    assert f"{folder}/valid_count_2025" == f"{ANNUAL_ASSET_ROOT}/landsat/valid_count_2025"