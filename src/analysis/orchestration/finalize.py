"""Monitor Day 3 tasks, validate completed assets and create final metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    asset_exists,
    exact_grid_region,
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_json,
    load_yaml,
    metadata_directory,
    report_directory,
    resolve_project_path,
    sha256_file,
    stable_object_hash,
    write_json,
)

TERMINAL_SUCCESS_STATES = {"COMPLETED", "EXISTS"}
TERMINAL_FAILURE_STATES = {"FAILED", "CANCELLED", "CANCEL_REQUESTED"}


def refresh_task_manifest(task_frame: pd.DataFrame) -> pd.DataFrame:
    """Refresh submitted task states with one Earth Engine task-list request."""
    refreshed = task_frame.copy()
    available = {
        task.id: task.status()
        for task in ee.batch.Task.list()
        if task.id is not None
    }

    for index, row in refreshed.iterrows():
        if str(row["state"]) == "EXISTS" or not str(row["task_id"]):
            continue

        task_id = str(row["task_id"])

        if task_id not in available:
            raise LookupError(
                f"Earth Engine task '{task_id}' was not found among recent tasks."
            )

        status = available[task_id]
        refreshed.at[index, "state"] = status.get("state", "UNKNOWN")
        refreshed.at[index, "error_message"] = status.get("error_message", "")

    return refreshed


def projection_summary(image: ee.Image) -> dict[str, Any]:
    """Return first-band CRS and affine transform metadata."""
    projection = image.select(0).projection().getInfo()
    return {
        "crs": projection.get("crs"),
        "transform": projection.get("transform"),
    }


def count_statistics(
    image: ee.Image,
    grid: dict[str, Any],
    selected_scene_count: int,
) -> dict[str, float]:
    """Calculate compact observation-depth QA statistics on the project grid."""
    count_band = image.select("valid_observation_count")
    reducer = (
        ee.Reducer.minMax()
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
    )
    values = count_band.reduceRegion(
        reducer=reducer,
        geometry=exact_grid_region(grid),
        crs=grid["crs"],
        crsTransform=grid["transform"],
        maxPixels=10_000_000,
        tileScale=4,
    ).getInfo()

    coverage = (
        ee.Image.pixelArea()
        .updateMask(count_band.gt(0))
        .rename("covered_area")
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=exact_grid_region(grid),
            crs=grid["crs"],
            crsTransform=grid["transform"],
            maxPixels=10_000_000,
            tileScale=4,
        )
        .get("covered_area")
        .getInfo()
    )
    grid_area = float(grid["width"] * grid["height"] * grid["resolution_m"] ** 2)

    return {
        "minimum_observation_count": float(
            values.get("valid_observation_count_min", 0)
        ),
        "maximum_observation_count": float(
            values.get("valid_observation_count_max", 0)
        ),
        "mean_observation_count": float(
            values.get("valid_observation_count_mean", 0)
        ),
        "median_observation_count": float(
            values.get("valid_observation_count_median", 0)
        ),
        "covered_grid_pct": float(coverage or 0) / grid_area * 100,
        "selected_scene_count": int(selected_scene_count),
    }


def validate_asset_grid(
    asset_id: str,
    expected_bands: list[str],
    grid: dict[str, Any],
) -> tuple[ee.Image, dict[str, Any]]:
    """Validate asset existence, band names, CRS and exact affine transform."""
    if not asset_exists(asset_id):
        raise FileNotFoundError(f"Earth Engine asset not found: {asset_id}")

    image = ee.Image(asset_id)
    bands = image.bandNames().getInfo()

    if bands != expected_bands:
        raise ValueError(
            f"Unexpected bands in {asset_id}: {bands}; expected {expected_bands}."
        )

    projection = projection_summary(image)

    if projection["crs"] != grid["crs"]:
        raise ValueError(
            f"Unexpected CRS in {asset_id}: {projection['crs']}"
        )

    observed_transform = [float(value) for value in projection["transform"]]
    expected_transform = [float(value) for value in grid["transform"]]

    if not np.allclose(observed_transform, expected_transform, atol=1e-9):
        raise ValueError(
            f"Unexpected transform in {asset_id}: {observed_transform}"
        )

    return image, projection


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one compact QA figure and immediately release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_figures(
    composite_manifest: pd.DataFrame,
    epoch_quality: pd.DataFrame,
    auxiliary_manifest: pd.DataFrame,
    report_dir: Path,
    config: dict[str, Any],
) -> None:
    """Create three lightweight figures from validated metadata tables."""
    ordered = composite_manifest.sort_values("epoch")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(ordered["epoch"].astype(str), ordered["selected_scene_count"])
    axis.set_title("Selected Landsat scene count by epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Selected scenes")
    save_figure(
        figure,
        report_dir / config["reports"]["selected_scene_counts_figure"],
    )

    quality = epoch_quality.sort_values("epoch")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        quality["epoch"],
        quality["median_observation_count"],
        marker="o",
        label="Median",
    )
    axis.plot(
        quality["epoch"],
        quality["maximum_observation_count"],
        marker="o",
        label="Maximum",
    )
    axis.set_title("Valid-observation depth by epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Observation count")
    axis.legend()
    save_figure(
        figure,
        report_dir / config["reports"]["observation_depth_figure"],
    )

    alignment_counts = auxiliary_manifest["representation"].value_counts()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(alignment_counts.index.astype(str), alignment_counts.values)
    axis.set_title("Auxiliary-source spatial representations")
    axis.set_xlabel("Representation")
    axis.set_ylabel("Source count")
    axis.tick_params(axis="x", rotation=20)
    save_figure(
        figure,
        report_dir / config["reports"]["source_alignment_figure"],
    )


def write_report(
    composite_manifest: pd.DataFrame,
    epoch_quality: pd.DataFrame,
    auxiliary_manifest: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a concise Day 3 report from validated manifests."""
    lines = [
        "# Landsat composites and auxiliary-source integration",
        "",
        "## Landsat outputs",
        "",
        "| Epoch | Sensors | Scenes | Covered grid | Median observations | Status |",
        "|---:|---|---:|---:|---:|:---:|",
    ]

    quality_index = epoch_quality.set_index("epoch")

    for row in composite_manifest.sort_values("epoch").itertuples(index=False):
        quality = quality_index.loc[int(row.epoch)]
        lines.append(
            "| "
            f"{int(row.epoch)} | {row.sensors} | {int(row.selected_scene_count)} | "
            f"{float(quality['covered_grid_pct']):.2f}% | "
            f"{float(quality['median_observation_count']):.2f} | PASS |"
        )

    lines.extend(
        [
            "",
            "## Auxiliary sources",
            "",
            "| Source | Spatial representation | Temporal representation | Intended role |",
            "|---|---|---|---|",
        ]
    )

    for row in auxiliary_manifest.itertuples(index=False):
        lines.append(
            "| "
            f"{row.source} | {row.representation} | "
            f"{row.temporal_representation} | {row.intended_role} |"
        )

    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- Landsat 7 SLC-off gaps are filled only by valid observations from other dates.",
            "- Mixed-sensor epochs are documented through sensor-specific count bands.",
            "- GHSL remains a modelled auxiliary source on its native 100 m grid.",
            "- OpenStreetMap is a contemporary snapshot and not historical evidence.",
            "- SRTM represents terrain observed approximately around the year 2000.",
            "",
            "No spectral index, Otsu threshold or built-up label was generated during Day 3.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_landsat_only_report(
    composite_manifest: pd.DataFrame,
    epoch_quality: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write the annual Landsat-only orchestration report."""
    lines = [
        "# Annual Landsat composite exports",
        "",
        "| Year | Sensors | Scenes | Covered grid | Median observations | Status |",
        "|---:|---|---:|---:|---:|:---:|",
    ]
    quality_index = epoch_quality.set_index("epoch")

    for row in composite_manifest.sort_values("epoch").itertuples(index=False):
        quality = quality_index.loc[int(row.epoch)]
        lines.append(
            f"| {int(row.epoch)} | {row.sensors} | {int(row.selected_scene_count)} | "
            f"{float(quality['covered_grid_pct']):.2f}% | "
            f"{float(quality['median_observation_count']):.2f} | PASS |"
        )

    lines.extend(
        [
            "",
            "Only median Landsat reflectance composites and valid-observation-count "
            "assets were validated. No terrain, auxiliary, built-up, transition or "
            "modelling products were created.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_finalize(config_path: Path, status_only: bool) -> dict[str, Any]:
    """Refresh task states and finalize metadata after all exports complete."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    landsat_only = bool(config.get("orchestration", {}).get("landsat_only", False))
    initialize_earth_engine(config["project"]["earth_engine_project"])

    task_path = resolve_project_path(config["exports"]["task_manifest"], project_root)

    if not task_path.is_file():
        raise FileNotFoundError("Submit Day 3 tasks before checking their status.")

    tasks = pd.read_csv(task_path, keep_default_na=False)
    tasks = refresh_task_manifest(tasks)
    tasks.to_csv(task_path, index=False)

    state_counts = tasks["state"].value_counts().to_dict()
    print(json.dumps({"task_states": state_counts}, indent=2))

    if status_only:
        return {"task_states": state_counts}

    failures = tasks[tasks["state"].isin(TERMINAL_FAILURE_STATES)]

    if not failures.empty:
        raise RuntimeError(
            "At least one Day 3 export failed. Inspect earth_engine_tasks.csv."
        )

    incomplete = tasks[~tasks["state"].isin(TERMINAL_SUCCESS_STATES)]

    if not incomplete.empty:
        raise RuntimeError(
            "Earth Engine exports are not complete. Run `make day3-status` later."
        )

    grid = load_grid_specification(
        resolve_project_path(config["inputs"]["grid_specification"], project_root)
    )
    composite_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for epoch in config["landsat"]["epochs"]:
        # Search for a task for an existing epoch.
        composite_row_ = tasks[
            (tasks["product_type"] == "landsat_composite")
            & (tasks["epoch"].astype(str) == str(epoch))
        ]
        if composite_row_.empty:
            print(f"Skipping epoch {epoch}: no composite task found.")
            continue
        composite_row = composite_row_.iloc[0]

        count_row_ = tasks[
            (tasks["product_type"] == "landsat_valid_count")
            & (tasks["epoch"].astype(str) == str(epoch))
        ]
        if count_row_.empty:
            print(f"Skipping epoch {epoch}: no count task found.")
            continue
        count_row = count_row_.iloc[0]

        composite_image, projection = validate_asset_grid(
            str(composite_row["asset_id"]),
            list(config["landsat"]["common_bands"]),
            grid,
        )
        count_image = ee.Image(str(count_row["asset_id"]))
        count_bands = count_image.bandNames().getInfo()

        if not count_bands or count_bands[0] != "valid_observation_count":
            raise ValueError(f"Invalid count bands for epoch {epoch}: {count_bands}")

        validate_asset_grid(
            str(count_row["asset_id"]),
            count_bands,
            grid,
        )
        statistics = count_statistics(
            count_image,
            grid,
            int(composite_row["selected_scene_count"]),
        )

        if statistics["maximum_observation_count"] > statistics["selected_scene_count"]:
            raise ValueError(
                f"Count maximum exceeds selected scenes for epoch {epoch}."
            )

        sensors = composite_image.get("scene_counts_by_sensor").getInfo() or "{}"
        composite_rows.append(
            {
                "epoch": int(epoch),
                "asset_id": composite_row["asset_id"],
                "count_asset_id": count_row["asset_id"],
                "selected_scene_count": int(composite_row["selected_scene_count"]),
                "selected_scene_ids_sha256": composite_row[
                    "selected_scene_ids_sha256"
                ],
                "sensors": sensors,
                "band_names": ",".join(config["landsat"]["common_bands"]),
                "crs": projection["crs"],
                "transform": json.dumps(projection["transform"]),
                "width": int(grid["width"]),
                "height": int(grid["height"]),
                "status": "PASS",
            }
        )
        quality_rows.append({"epoch": int(epoch), **statistics})

    if not landsat_only:
        terrain_rows = tasks[tasks["product_type"] == "terrain"]
        if terrain_rows.empty:
            raise RuntimeError("Terrain task missing from task manifest.")
        terrain_row = terrain_rows.iloc[0]
        validate_asset_grid(
            str(terrain_row["asset_id"]),
            list(config["srtm"]["output_bands"]),
            grid,
        )

    metadata_dir = metadata_directory(config, project_root)
    composite_manifest = pd.DataFrame(composite_rows)
    epoch_quality = pd.DataFrame(quality_rows)
    composite_manifest.to_csv(
        metadata_dir / config["metadata"]["landsat_composite_manifest"],
        index=False,
    )
    epoch_quality.to_csv(
        metadata_dir / config["metadata"]["landsat_epoch_quality"],
        index=False,
    )

    dependencies_path = metadata_dir / config["metadata"]["dependency_manifest"]
    version_payload = {
        "day3_version": int(config["version"]),
        "grid_specification_sha256": sha256_file(
            resolve_project_path(config["inputs"]["grid_specification"], project_root)
        ),
        "catalog_version_sha256": sha256_file(
            resolve_project_path(config["inputs"]["catalog_version"], project_root)
        ),
        "compositing_protocol_sha256": sha256_file(
            resolve_project_path(config["inputs"]["compositing_protocol"], project_root)
        ),
        "dependency_manifest_sha256": sha256_file(dependencies_path),
        "task_manifest_sha256": sha256_file(task_path),
        "landsat_composite_manifest_sha256": stable_object_hash(composite_rows),
        "landsat_epoch_quality_sha256": stable_object_hash(quality_rows),
        "earth_engine_assets": tasks["asset_id"].astype(str).tolist(),
    }

    report_dir = report_directory(config, project_root)
    if landsat_only:
        write_landsat_only_report(
            composite_manifest,
            epoch_quality,
            report_dir / config["reports"]["day3_report"],
        )
    else:
        auxiliary_path = metadata_dir / config["metadata"]["auxiliary_source_manifest"]
        auxiliary_manifest = pd.read_csv(auxiliary_path)
        osm_metadata_path = resolve_project_path(
            config["osm"]["source_metadata"], project_root
        )
        if config["osm"]["required_for_day3"] and not osm_metadata_path.is_file():
            raise FileNotFoundError(
                "OSM metadata is missing. Run `make day3-osm` before finalization."
            )
        version_payload.update(
            {
                "ghsl_epoch_manifest_sha256": sha256_file(
                    metadata_dir / config["metadata"]["ghsl_epoch_manifest"]
                ),
                "grid_linkage_sha256": sha256_file(
                    metadata_dir / config["metadata"]["grid_linkage"]
                ),
                "osm_source_sha256": (
                    load_json(osm_metadata_path)["source_sha256"]
                    if osm_metadata_path.is_file()
                    else None
                ),
            }
        )
        create_figures(
            composite_manifest,
            epoch_quality,
            auxiliary_manifest,
            report_dir,
            config,
        )
        write_report(
            composite_manifest,
            epoch_quality,
            auxiliary_manifest,
            report_dir / config["reports"]["day3_report"],
        )

    version_payload["stable_signature"] = stable_object_hash(version_payload)
    write_json(
        metadata_dir / config["metadata"]["day3_version"],
        version_payload,
    )

    print(json.dumps(version_payload, indent=2))
    return version_payload


def parse_arguments() -> argparse.Namespace:
    """Parse status and finalization command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor and finalize Day 3 Earth Engine outputs."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run task monitoring or complete Day 3 finalization."""
    arguments = parse_arguments()
    run_finalize(arguments.config.resolve(), arguments.status_only)


if __name__ == "__main__":
    main()
