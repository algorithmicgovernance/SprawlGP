"""Validate frozen Day 1 and Day 2 dependencies before Day 3 submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    find_project_root,
    load_grid_specification,
    load_json,
    load_yaml,
    metadata_directory,
    resolve_project_path,
    sha256_file,
    stable_object_hash,
)

EXPECTED_EPOCHS = {1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025}

# What we actually have due to the lack of early data for Yaounde
# EXPECTED_EPOCHS = {2005, 2010, 2015, 2020, 2025}

def asset_id_column(manifest: pd.DataFrame) -> str:
    """Return the supported column containing exact Earth Engine asset IDs."""
    for candidate in ("earth_engine_asset_id", "asset_id"):
        if candidate in manifest.columns:
            return candidate

    raise ValueError(
        "The selected scene manifest needs an `earth_engine_asset_id` or "
        "`asset_id` column."
    )


def compare_grid_reference(
    grid: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    """Compare grid-defining fields against the accepted Day 1 reference."""
    fields = (
        "crs",
        "resolution_m",
        "transform",
        "width",
        "height",
        "extent",
    )

    for field in fields:
        if grid.get(field) != reference.get(field):
            raise ValueError(
                f"Frozen grid mismatch for '{field}': "
                f"{grid.get(field)!r} != {reference.get(field)!r}"
            )


def compare_catalog_reference(
    catalog_version: dict[str, Any],
    catalog_reference: dict[str, Any],
) -> None:
    """Reject a Day 2 catalogue that differs from its committed reference."""
    if catalog_version != catalog_reference:
        raise ValueError(
            "catalog_version.json differs from landsat_catalog_v1.json. "
            "Re-freeze Day 2 only after an intentional scientific change."
        )


def validate_manifest(
    manifest: pd.DataFrame,
    protocol: dict[str, Any],
    expected_epochs: set[int] | None = None,
    supported_sensors: set[str] | None = None,
) -> str:
    """Validate epochs, exact asset IDs and protocol representation."""
    expected = expected_epochs or EXPECTED_EPOCHS
    required = {"epoch", "sensor_key", "acquisition_date"}
    missing = required.difference(manifest.columns)

    if missing:
        raise ValueError(
            f"Selected scene manifest is missing columns: {sorted(missing)}"
        )

    id_column = asset_id_column(manifest)

    if manifest[id_column].isna().any():
        raise ValueError("At least one selected scene has no exact asset ID.")

    observed_epochs = set(manifest["epoch"].astype(int))

    # Accept a subset
    if not observed_epochs.issubset(expected):
        invalid = observed_epochs - expected
        raise ValueError(
            f"Found epochs that are not in the expected list {sorted(expected)}: "
            f"{sorted(invalid)}"
        )

    if supported_sensors is not None:
        observed_sensors = set(manifest["sensor_key"].astype(str))
        unknown_sensors = observed_sensors.difference(supported_sensors)
        if unknown_sensors:
            raise ValueError(
                f"Selected manifest uses unsupported sensors: {sorted(unknown_sensors)}"
            )

    if protocol.get("temporal_mode") == "calendar_year":
        acquisition_years = pd.to_datetime(
            manifest["acquisition_date"], errors="raise"
        ).dt.year
        represented_years = manifest["epoch"].astype(int)
        if (acquisition_years.to_numpy() != represented_years.to_numpy()).any():
            raise ValueError(
                "Calendar-year manifests cannot use imagery outside the represented year."
            )

    if "candidate_for_composite" in manifest.columns:
        values = manifest["candidate_for_composite"].astype(str).str.lower()
        invalid = ~values.isin({"true", "1"})

        if invalid.any():
            raise ValueError(
                "The selected scene manifest contains a scene not marked as "
                "a composite candidate."
            )

    protocol_epochs = set(int(value) for value in protocol["epochs"])
    
    # print("Protocol epochs: ", protocol_epochs)
    # print("Expected epochs: ", EXPECTED_EPOCHS)
    
    if not protocol_epochs.issubset(expected):
        invalid = protocol_epochs - expected
        raise ValueError(
            f"The compositing protocol contains invalid epochs: {sorted(invalid)}"
        )

    return id_column

def dependency_row(name: str, path: Path, status: str = "PASS") -> dict[str, Any]:
    """Create one traceable dependency-manifest record."""
    return {
        "dependency_name": name,
        "dependency_path": str(path),
        "observed_sha256": sha256_file(path),
        "status": status,
    }


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Execute all dependency checks and write the dependency manifest."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    inputs = config["inputs"]

    paths = {
        key: resolve_project_path(value, project_root)
        for key, value in inputs.items()
    }

    required_paths = [
        "grid_specification",
        "grid_reference",
        "selected_scene_manifest",
        "compositing_protocol",
        "catalog_version",
        "landsat_catalog_config",
    ]

    if "catalog_reference" in paths:
        required_paths.append("catalog_reference")

    for key in required_paths:
        if not paths[key].is_file():
            raise FileNotFoundError(f"Missing Day 3 dependency: {paths[key]}")

    grid = load_grid_specification(paths["grid_specification"])
    grid_reference = load_json(paths["grid_reference"])
    compare_grid_reference(grid, grid_reference)

    if grid["crs"] != "EPSG:32632":
        raise ValueError(f"Unexpected project CRS: {grid['crs']}")

    catalog_version = load_json(paths["catalog_version"])
    if "catalog_reference" in paths:
        catalog_reference = load_json(paths["catalog_reference"])
        compare_catalog_reference(catalog_version, catalog_reference)

    manifest = pd.read_csv(paths["selected_scene_manifest"])
    protocol = load_yaml(paths["compositing_protocol"])
    catalog_config = load_yaml(paths["landsat_catalog_config"])
    id_column = validate_manifest(
        manifest,
        protocol,
        expected_epochs=set(int(value) for value in config["landsat"]["epochs"]),
        supported_sensors=set(catalog_config["collections"]),
    )

    rows = [
        dependency_row("grid_specification", paths["grid_specification"]),
        dependency_row("grid_reference", paths["grid_reference"]),
        dependency_row("selected_scene_manifest", paths["selected_scene_manifest"]),
        dependency_row("compositing_protocol", paths["compositing_protocol"]),
        dependency_row("catalog_version", paths["catalog_version"]),
        dependency_row("landsat_catalog_config", paths["landsat_catalog_config"]),
    ]
    if "catalog_reference" in paths:
        rows.append(dependency_row("catalog_reference", paths["catalog_reference"]))

    output_dir = metadata_directory(config, project_root)
    output_path = output_dir / config["metadata"]["dependency_manifest"]
    pd.DataFrame(rows).to_csv(output_path, index=False)

    result = {
        "project_root": str(project_root),
        "grid_sha256": stable_object_hash(grid),
        "catalog_sha256": sha256_file(paths["catalog_version"]),
        "protocol_sha256": sha256_file(paths["compositing_protocol"]),
        "selected_scene_manifest_sha256": sha256_file(
            paths["selected_scene_manifest"]
        ),
        "selected_scene_count": int(len(manifest)),
        "asset_id_column": id_column,
        "epochs": sorted(int(value) for value in manifest["epoch"].unique()),
        "status": "PASS",
    }

    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the Day 3 preflight command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate frozen Day 1 and Day 2 dependencies."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the preflight command."""
    arguments = parse_arguments()
    run_preflight(arguments.config.resolve())


if __name__ == "__main__":
    main()
