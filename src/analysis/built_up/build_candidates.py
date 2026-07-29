"""Validate orchestration outputs, calculate indices and submit candidate assets.

All histograms and Otsu thresholds are calculated and validated before any
export task is started. This prevents a failed epoch-index pair from leaving a
partially submitted candidate dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

from src.analysis.orchestration.common import (
    asset_exists,
    ensure_asset_folder,
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_json,
    load_yaml,
    metadata_directory,
    resolve_project_path,
    sha256_file,
    stable_object_hash,
    write_json,
)
from src.analysis.built_up.indices import (
    CANDIDATE_BANDS,
    CANDIDATE_INDICES,
    CONTINUOUS_INDICES,
    CONTINUOUS_INDEX_BANDS,
    INDEX_BANDS,
    build_candidate_stack,
    build_index_stack,
)
from src.analysis.built_up.otsu import PASS, OtsuResult, otsu_from_histogram


SOURCE_COMPOSITE_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]


@dataclass
class EpochBundle:
    """Hold lazy Earth Engine images and local metadata for one epoch."""

    epoch: int
    source_composite_asset: str
    source_count_asset: str
    index_image: ee.Image
    candidate_image: ee.Image
    thresholds: dict[str, float]
    threshold_rows: list[dict[str, Any]]
    area_rows: list[dict[str, Any]]
    histograms: dict[str, Any]



def validate_index_configuration(
    config: dict[str, Any],
) -> None:
    """Ensure YAML index roles match the implemented pipeline schema."""
    configured_continuous = list(
        config["indices"]["continuous"]
    )
    configured_candidates = list(
        config["indices"]["candidates"]
    )

    if configured_continuous != CONTINUOUS_INDICES:
        raise ValueError(
            "Configured continuous indices do not match the implemented "
            f"schema: {configured_continuous} != {CONTINUOUS_INDICES}"
        )

    if configured_candidates != CANDIDATE_INDICES:
        raise ValueError(
            "Configured candidate indices do not match the implemented "
            f"schema: {configured_candidates} != {CANDIDATE_INDICES}"
        )

    if not set(configured_candidates).issubset(
        configured_continuous
    ):
        raise ValueError(
            "Every candidate index must also be retained as a "
            "continuous index."
        )

    if "ibi" in configured_candidates:
        raise ValueError(
            "IBI must remain excluded from candidate generation."
        )
    
    expected_directions = set(CANDIDATE_INDICES)
    configured_directions = set(
        config["classification"]["built_up_direction"]
    )
    missing_directions = (
        expected_directions - configured_directions
    )
    unexpected_directions = (
        configured_directions - expected_directions
    )
    if missing_directions or unexpected_directions:
        raise ValueError(
            "Classification directions do not match candidate indices. "
            f"Missing: {sorted(missing_directions)}. "
            f"Unexpected: {sorted(unexpected_directions)}. "
            f"Expected: {sorted(expected_directions)}."
        )


def load_ee_geometry(path: Path) -> ee.Geometry:
    """Load one dissolved vector boundary as a WGS84 Earth Engine geometry."""
    frame = gpd.read_file(path)

    if frame.crs is None or len(frame) != 1:
        raise ValueError(
            f"Expected one georeferenced dissolved boundary in {path}."
        )

    geometry = frame.to_crs("EPSG:4326").geometry.iloc[0]

    if geometry is None or geometry.is_empty:
        raise ValueError(f"Boundary geometry is empty: {path}")

    return ee.Geometry(mapping(geometry))


def compare_grid_reference(
    grid: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    """Compare only the fields that define the accepted raster grid."""
    for field in (
        "crs",
        "resolution_m",
        "transform",
        "width",
        "height",
        "extent",
    ):
        if grid.get(field) != reference.get(field):
            raise ValueError(
                f"Frozen grid mismatch for '{field}': "
                f"{grid.get(field)!r} != {reference.get(field)!r}"
            )


def image_metadata(asset_id: str) -> dict[str, Any]:
    """Return band names, types, dimensions, CRS and transform for an asset."""
    information = ee.Image(asset_id).getInfo()
    bands = information.get("bands", [])

    if not bands:
        raise ValueError(f"Earth Engine asset has no bands: {asset_id}")

    first = bands[0]
    return {
        "band_names": [str(band["id"]) for band in bands],
        "band_types": {
            str(band["id"]): band.get("data_type", {})
            for band in bands
        },
        "width": int(first["dimensions"][0]),
        "height": int(first["dimensions"][1]),
        "crs": first.get("crs"),
        "transform": [
            float(value)
            for value in first.get("crs_transform", [])
        ],
    }


def validate_source_asset(
    asset_id: str,
    expected_bands: list[str],
    grid: dict[str, Any],
) -> None:
    """Validate a source asset's schema and exact Day 1 grid."""
    if not asset_exists(asset_id):
        raise FileNotFoundError(f"Earth Engine asset not found: {asset_id}")

    metadata = image_metadata(asset_id)

    if metadata["band_names"] != expected_bands:
        raise ValueError(
            f"Unexpected source bands in {asset_id}: "
            f"{metadata['band_names']}; expected {expected_bands}."
        )

    if metadata["crs"] != grid["crs"]:
        raise ValueError(
            f"Unexpected CRS in {asset_id}: {metadata['crs']}"
        )

    observed_transform = metadata["transform"]
    expected_transform = [float(value) for value in grid["transform"]]

    if not np.allclose(
        observed_transform,
        expected_transform,
        atol=1e-9,
    ):
        raise ValueError(
            f"Unexpected transform in {asset_id}: {observed_transform}"
        )

    if (
        metadata["width"] != int(grid["width"])
        or metadata["height"] != int(grid["height"])
    ):
        raise ValueError(
            f"Unexpected dimensions in {asset_id}: "
            f"{metadata['width']} x {metadata['height']}"
        )


def select_completed_composites(
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select passed orchestration epochs and apply the optional include list."""
    required = {
        "epoch",
        "asset_id",
        "count_asset_id",
        "status",
    }
    missing = required.difference(manifest.columns)

    if missing:
        raise ValueError(
            f"Composite manifest is missing columns: {sorted(missing)}"
        )

    completed = manifest[
        manifest["status"].astype(str).str.upper() == "PASS"
    ].copy()

    include = config.get("epochs", {}).get("include")

    if include:
        allowed = {int(value) for value in include}
        completed = completed[
            completed["epoch"].astype(int).isin(allowed)
        ]

    completed["epoch"] = completed["epoch"].astype(int)
    completed = completed.sort_values("epoch").reset_index(drop=True)

    if completed.empty:
        raise ValueError(
            "No completed orchestration composite remains after filtering."
        )

    if not completed["epoch"].is_unique:
        raise ValueError("Composite manifest contains duplicate epochs.")

    return completed


def dependency_row(name: str, path: Path) -> dict[str, Any]:
    """Create one local-file dependency record with a true SHA-256 checksum."""
    return {
        "dependency_name": name,
        "dependency_path": str(path),
        "observed_sha256": sha256_file(path),
        "status": "PASS",
    }


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Validate local dependencies and completed source assets."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    validate_index_configuration(config)
    inputs = config["inputs"]
    paths = {
        name: resolve_project_path(value, project_root)
        for name, value in inputs.items()
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing built-up candidate dependency '{name}': {path}"
            )

    grid = load_grid_specification(paths["grid_specification"])
    reference = load_json(paths["grid_reference"])
    compare_grid_reference(grid, reference)

    if grid["crs"] != "EPSG:32632":
        raise ValueError(f"Unexpected project CRS: {grid['crs']}")

    orchestration_version = load_json(paths["orchestration_version"])
    composite_manifest = pd.read_csv(paths["composite_manifest"])
    completed = select_completed_composites(composite_manifest, config)

    initialize_earth_engine(config["project"]["earth_engine_project"])

    for row in completed.itertuples(index=False):
        validate_source_asset(
            str(row.asset_id),
            SOURCE_COMPOSITE_BANDS,
            grid,
        )
        count_metadata = image_metadata(str(row.count_asset_id))

        if (
            not count_metadata["band_names"]
            or count_metadata["band_names"][0]
            != "valid_observation_count"
        ):
            raise ValueError(
                f"Invalid count asset bands for epoch {row.epoch}: "
                f"{count_metadata['band_names']}"
            )

        validate_source_asset(
            str(row.count_asset_id),
            count_metadata["band_names"],
            grid,
        )

    metadata_dir = metadata_directory(config, project_root)
    dependency_path = (
        metadata_dir / config["metadata"]["dependency_manifest"]
    )
    dependency_rows = [
        dependency_row("grid_specification", paths["grid_specification"]),
        dependency_row("grid_reference", paths["grid_reference"]),
        dependency_row(
            "orchestration_version",
            paths["orchestration_version"],
        ),
        dependency_row("composite_manifest", paths["composite_manifest"]),
        dependency_row("core_boundary", paths["core_boundary"]),
        dependency_row("context_boundary", paths["context_boundary"]),
    ]
    pd.DataFrame(dependency_rows).to_csv(dependency_path, index=False)

    epoch_count = int(len(completed))
    result = {
        "status": "PASS",
        "completed_epochs": completed["epoch"].tolist(),
        "epoch_count": epoch_count,
        "continuous_index_layers": (
            epoch_count * len(CONTINUOUS_INDEX_BANDS)
        ),
        "binary_candidate_maps": (
            epoch_count * len(CANDIDATE_INDICES)
        ),
        "expected_export_tasks": epoch_count * 2,
        "grid_sha256": stable_object_hash(grid),
        "orchestration_stable_signature": orchestration_version.get(
            "stable_signature"
        ),
        "dependency_manifest": str(dependency_path),
    }

    print(json.dumps(result, indent=2))
    return result


def histogram_reducer(config: dict[str, Any]):
    """Construct the configured Earth Engine histogram reducer."""
    histogram = config["histogram"]
    arguments: dict[str, Any] = {
        "maxBuckets": int(histogram["max_buckets"]),
        "maxRaw": int(histogram["max_raw"]),
    }

    if histogram.get("min_bucket_width") is not None:
        arguments["minBucketWidth"] = float(
            histogram["min_bucket_width"]
        )

    return ee.Reducer.histogram(**arguments)


def calculate_histograms(
    index_images: dict[str, ee.Image],
    count_image: ee.Image,
    core_geometry: ee.Geometry,
    grid: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate diagnostic histograms with one reduction per epoch.

    IBI remains in the histogram archive for transparency, but Otsu threshold
    calculation later iterates only over ``CANDIDATE_INDICES``.
    """
    minimum = int(
        config["classification"][
            "minimum_valid_observations_for_threshold"
        ]
    )
    threshold_mask = count_image.select(
        "valid_observation_count"
    ).gte(minimum)
    diagnostic_stack = ee.Image.cat(
        [
            index_images[name].updateMask(threshold_mask)
            for name in CONTINUOUS_INDICES
        ]
    ).select(CONTINUOUS_INDICES)

    return diagnostic_stack.reduceRegion(
        reducer=histogram_reducer(config),
        geometry=core_geometry,
        crs=grid["crs"],
        crsTransform=grid["transform"],
        maxPixels=int(config["histogram"]["max_pixels"]),
        tileScale=int(config["histogram"]["tile_scale"]),
    ).getInfo()


def candidate_area_statistics(
    candidate_image: ee.Image,
    core_geometry: ee.Geometry,
    grid: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate all candidate pixel counts and areas in one reduction."""
    pixel_area = ee.Image.pixelArea()
    one = ee.Image.constant(1)
    bands: list[ee.Image] = []

    for name in CANDIDATE_INDICES:
        built = candidate_image.select(f"built_{name}")
        valid = candidate_image.select(f"valid_{name}").eq(1)

        bands.extend(
            [
                one.updateMask(valid).rename(f"{name}_valid_pixels"),
                one.updateMask(built.eq(1)).rename(
                    f"{name}_built_pixels"
                ),
                one.updateMask(built.eq(0)).rename(
                    f"{name}_nonbuilt_pixels"
                ),
                pixel_area.updateMask(valid).rename(
                    f"{name}_valid_area_m2"
                ),
                pixel_area.updateMask(built.eq(1)).rename(
                    f"{name}_built_area_m2"
                ),
                pixel_area.updateMask(built.eq(0)).rename(
                    f"{name}_nonbuilt_area_m2"
                ),
            ]
        )

    return ee.Image.cat(bands).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=core_geometry,
        crs=grid["crs"],
        crsTransform=grid["transform"],
        maxPixels=int(config["histogram"]["max_pixels"]),
        tileScale=int(config["histogram"]["tile_scale"]),
    ).getInfo()


def threshold_row(
    *,
    epoch: int,
    index_name: str,
    result: OtsuResult,
    source_composite_asset: str,
    source_count_asset: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build one auditable threshold-table record."""
    row = {
        "epoch": int(epoch),
        "index_name": index_name,
        "threshold_strategy": config["classification"][
            "primary_strategy"
        ],
        "threshold_value": result.threshold,
        "classification_direction": config["classification"][
            "built_up_direction"
        ][index_name],
        "histogram_bucket_count": result.histogram_bucket_count,
        "histogram_valid_pixel_count": (
            result.histogram_valid_pixel_count
        ),
        "histogram_min": result.histogram_min,
        "histogram_max": result.histogram_max,
        "histogram_mean": result.histogram_mean,
        "threshold_region": config["classification"][
            "threshold_region"
        ],
        "minimum_observation_count": config["classification"][
            "minimum_valid_observations_for_threshold"
        ],
        "status": result.status,
        "failure_reason": result.failure_reason,
        "source_composite_asset": source_composite_asset,
        "source_count_asset": source_count_asset,
        "formula_version": int(config["indices"]["formula_version"]),
        "grid_version": 1,
    }
    row["threshold_record_sha256"] = stable_object_hash(row)
    return row


def enrich_threshold_and_area_rows(
    threshold_rows: list[dict[str, Any]],
    area_values: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach candidate class counts and build the compact area-summary rows."""
    area_rows: list[dict[str, Any]] = []

    for row in threshold_rows:
        name = row["index_name"]
        valid_pixels = int(round(float(
            area_values.get(f"{name}_valid_pixels", 0) or 0
        )))
        built_pixels = int(round(float(
            area_values.get(f"{name}_built_pixels", 0) or 0
        )))
        nonbuilt_pixels = int(round(float(
            area_values.get(f"{name}_nonbuilt_pixels", 0) or 0
        )))
        valid_area_ha = float(
            area_values.get(f"{name}_valid_area_m2", 0) or 0
        ) / 10_000
        built_area_ha = float(
            area_values.get(f"{name}_built_area_m2", 0) or 0
        ) / 10_000
        nonbuilt_area_ha = float(
            area_values.get(f"{name}_nonbuilt_area_m2", 0) or 0
        ) / 10_000
        built_fraction = (
            built_area_ha / valid_area_ha
            if valid_area_ha > 0
            else None
        )

        row.update(
            {
                "candidate_valid_pixel_count_core": valid_pixels,
                "candidate_built_pixel_count_core": built_pixels,
                "candidate_nonbuilt_pixel_count_core": nonbuilt_pixels,
                "candidate_built_fraction_core": built_fraction,
                "candidate_built_area_ha_core": built_area_ha,
            }
        )
        row["threshold_record_sha256"] = stable_object_hash(
            {
                key: value
                for key, value in row.items()
                if key != "threshold_record_sha256"
            }
        )

        area_rows.append(
            {
                "epoch": int(row["epoch"]),
                "index_name": name,
                "threshold": row["threshold_value"],
                "valid_area_ha": valid_area_ha,
                "candidate_built_area_ha": built_area_ha,
                "candidate_nonbuilt_area_ha": nonbuilt_area_ha,
                "candidate_built_fraction": built_fraction,
                "change_from_previous_epoch_ha": None,
                "quality_flag": "PASS",
            }
        )

    return threshold_rows, area_rows


def add_area_changes(area_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate diagnostic change from the previous available epoch."""
    result = area_frame.sort_values(
        ["index_name", "epoch"]
    ).copy()
    result["change_from_previous_epoch_ha"] = result.groupby(
        "index_name"
    )["candidate_built_area_ha"].diff()
    return result.sort_values(
        ["epoch", "index_name"]
    ).reset_index(drop=True)



def export_grid_region(grid: dict[str, Any]) -> ee.Geometry:
    """Return an inset export rectangle that selects exactly the frozen cells.

    The configured extent describes the outer raster edges. A small interior
    inset prevents Earth Engine from including a neighbouring pixel whose edge
    merely touches that rectangle, while every intended edge pixel remains
    intersected. The affine transform remains the authoritative grid origin.
    """
    extent = grid["extent"]
    resolution = float(grid["resolution_m"])
    inset = resolution / 4.0

    return ee.Geometry.Rectangle(
        [
            float(extent["xmin"]) + inset,
            float(extent["ymin"]) + inset,
            float(extent["xmax"]) - inset,
            float(extent["ymax"]) - inset,
        ],
        proj=str(grid["crs"]),
        geodesic=False,
    )


def start_export(
    image: ee.Image,
    *,
    description: str,
    asset_id: str,
    pyramiding_policy: dict[str, str],
    grid: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Start an exact-grid asset export or record an existing output."""
    overwrite = bool(config["exports"]["overwrite_existing_assets"])

    if asset_exists(asset_id) and not overwrite:
        return {
            "task_id": "",
            "state": "EXISTS",
            "description": description,
            "asset_id": asset_id,
        }

    transform = [float(value) for value in grid["transform"]]
    export_image = image.setDefaultProjection(
        str(grid["crs"]),
        transform,
    )

    task = ee.batch.Export.image.toAsset(
        image=export_image,
        description=description,
        assetId=asset_id,
        pyramidingPolicy=pyramiding_policy,
        region=export_grid_region(grid),
        crs=str(grid["crs"]),
        crsTransform=transform,
        maxPixels=int(config["exports"]["max_pixels"]),
        shardSize=int(config["exports"]["shard_size"]),
        priority=int(config["exports"]["priority"]),
        overwrite=overwrite,
    )
    task.start()
    status = task.status()

    return {
        "task_id": status.get("id", task.id),
        "state": status.get("state", "READY"),
        "description": description,
        "asset_id": asset_id,
    }


def build_all_epoch_bundles(
    completed: pd.DataFrame,
    core_geometry: ee.Geometry,
    grid: dict[str, Any],
    config: dict[str, Any],
) -> tuple[
    list[EpochBundle],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Calculate every epoch and return failures before any export starts."""
    bundles: list[EpochBundle] = []
    histogram_archive: dict[str, Any] = {}
    all_threshold_rows: list[dict[str, Any]] = []
    all_area_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in completed.itertuples(index=False):
        epoch = int(row.epoch)
        composite_asset = str(row.asset_id)
        count_asset = str(row.count_asset_id)
        composite = ee.Image(composite_asset)
        count = ee.Image(count_asset)

        index_stack, index_images = build_index_stack(
            composite,
            count,
            epsilon=float(config["indices"]["denominator_epsilon"]),
            savi_l=float(config["indices"]["savi_l"]),
            minimum_valid_observations=int(
                config["classification"][
                    "minimum_valid_observations_for_candidate"
                ]
            ),
        )
        histograms = calculate_histograms(
            index_images,
            count,
            core_geometry,
            grid,
            config,
        )
        histogram_archive[str(epoch)] = histograms
        thresholds: dict[str, float] = {}
        epoch_threshold_rows: list[dict[str, Any]] = []
        epoch_failures: list[dict[str, Any]] = []

        for index_name in CANDIDATE_INDICES:
            result = otsu_from_histogram(histograms.get(index_name))
            epoch_threshold_rows.append(
                threshold_row(
                    epoch=epoch,
                    index_name=index_name,
                    result=result,
                    source_composite_asset=composite_asset,
                    source_count_asset=count_asset,
                    config=config,
                )
            )

            if result.status == PASS and result.threshold is not None:
                thresholds[index_name] = float(result.threshold)
            else:
                epoch_failures.append(
                    {
                        "epoch": epoch,
                        "index_name": index_name,
                        "status": result.status,
                        "reason": result.failure_reason,
                    }
                )

        all_threshold_rows.extend(epoch_threshold_rows)

        if epoch_failures:
            failures.extend(epoch_failures)
            continue

        candidate_image = build_candidate_stack(
            index_images,
            thresholds,
        )
        area_values = candidate_area_statistics(
            candidate_image,
            core_geometry,
            grid,
            config,
        )
        epoch_threshold_rows, area_rows = enrich_threshold_and_area_rows(
            epoch_threshold_rows,
            area_values,
        )

        # Replace the preliminary rows with their area-enriched equivalents.
        del all_threshold_rows[-len(epoch_threshold_rows):]
        all_threshold_rows.extend(epoch_threshold_rows)
        all_area_rows.extend(area_rows)
        bundles.append(
            EpochBundle(
                epoch=epoch,
                source_composite_asset=composite_asset,
                source_count_asset=count_asset,
                index_image=index_stack,
                candidate_image=candidate_image,
                thresholds=thresholds,
                threshold_rows=epoch_threshold_rows,
                area_rows=area_rows,
                histograms=histograms,
            )
        )

    return (
        bundles,
        histogram_archive,
        all_threshold_rows,
        all_area_rows,
        failures,
    )

def write_pre_export_metadata(
    threshold_rows: list[dict[str, Any]],
    area_rows: list[dict[str, Any]],
    histogram_archive: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    grid: dict[str, Any],
) -> tuple[Path, Path, Path, str, str]:
    """Write threshold, histogram, area and processing-recipe metadata."""
    metadata_dir = metadata_directory(config, project_root)
    threshold_path = metadata_dir / config["metadata"]["threshold_table"]
    area_path = metadata_dir / config["metadata"]["candidate_area_summary"]
    histogram_path = metadata_dir / config["metadata"]["histograms"]
    recipe_path = metadata_dir / config["metadata"]["processing_recipe"]

    pd.DataFrame(threshold_rows).sort_values(
        ["epoch", "index_name"]
    ).to_csv(threshold_path, index=False)
    area_frame = pd.DataFrame(area_rows)

    if area_frame.empty:
        area_frame.to_csv(area_path, index=False)
    else:
        add_area_changes(area_frame).to_csv(
            area_path,
            index=False,
        )
    write_json(histogram_path, histogram_archive)

    orchestration_path = resolve_project_path(
        config["inputs"]["orchestration_version"],
        project_root,
    )
    recipe = {
        "pipeline_stage": "built_up_candidates",
        "pipeline_version": int(config["version"]),
        "source_orchestration_sha256": sha256_file(orchestration_path),
        "grid_sha256": stable_object_hash(grid),
        "formula_version": int(config["indices"]["formula_version"]),
        "denominator_epsilon": float(
            config["indices"]["denominator_epsilon"]
        ),
        "savi_l": float(config["indices"]["savi_l"]),
        "index_band_schema": INDEX_BANDS,
        "candidate_band_schema": CANDIDATE_BANDS,
        "threshold_strategy": config["classification"][
            "primary_strategy"
        ],
        "threshold_region": config["classification"]["threshold_region"],
        "application_region": config["classification"][
            "application_region"
        ],
        "minimum_valid_observations_for_threshold": config[
            "classification"
        ]["minimum_valid_observations_for_threshold"],
        "minimum_valid_observations_for_candidate": config[
            "classification"
        ]["minimum_valid_observations_for_candidate"],
        "classification_directions": {
            name: config["classification"]["built_up_direction"][name]
            for name in CANDIDATE_INDICES
        },
        "pooled_thresholds_computed": False,
    }
    write_json(recipe_path, recipe)

    return (
        threshold_path,
        area_path,
        recipe_path,
        sha256_file(threshold_path),
        sha256_file(recipe_path),
    )


def submit_exports(
    bundles: list[EpochBundle],
    config: dict[str, Any],
    project_root: Path,
    grid: dict[str, Any],
    threshold_table_sha256: str,
    processing_recipe_sha256: str,
) -> pd.DataFrame:
    """Attach provenance properties and submit two exports per epoch."""
    asset_root = config["exports"]["asset_root"].rstrip("/")
    index_folder = f"{asset_root}/{config['exports']['index_folder']}"
    candidate_folder = (
        f"{asset_root}/{config['exports']['candidate_folder']}"
    )
    ensure_asset_folder(asset_root)
    ensure_asset_folder(index_folder)
    ensure_asset_folder(candidate_folder)

    orchestration_sha256 = sha256_file(
        resolve_project_path(
            config["inputs"]["orchestration_version"],
            project_root,
        )
    )
    tasks: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []

    for bundle in bundles:
        index_asset = f"{index_folder}/indices_{bundle.epoch}"
        candidate_asset = (
            f"{candidate_folder}/candidates_{bundle.epoch}"
        )
        index_recipe = {
            "source_composite_asset": bundle.source_composite_asset,
            "source_count_asset": bundle.source_count_asset,
            "source_orchestration_sha256": orchestration_sha256,
            "grid_version": 1,
            "formula_version": int(
                config["indices"]["formula_version"]
            ),
            "denominator_epsilon": float(
                config["indices"]["denominator_epsilon"]
            ),
            "savi_l": float(config["indices"]["savi_l"]),
            "band_schema": INDEX_BANDS,
            "output_asset_id": index_asset,
        }
        index_recipe_sha256 = stable_object_hash(index_recipe)
        candidate_recipe = {
            **index_recipe,
            "threshold_strategy": config["classification"][
                "primary_strategy"
            ],
            "thresholds": bundle.thresholds,
            "classification_direction": config["classification"][
                "built_up_direction"
            ],
            "band_schema": CANDIDATE_BANDS,
            "output_asset_id": candidate_asset,
        }
        candidate_recipe_sha256 = stable_object_hash(candidate_recipe)

        index_image = bundle.index_image.set(
            {
                "pipeline_stage": "built_up_candidates",
                "pipeline_version": int(config["version"]),
                "epoch": bundle.epoch,
                "source_composite_asset": bundle.source_composite_asset,
                "source_observation_count_asset": (
                    bundle.source_count_asset
                ),
                "grid_version": 1,
                "formula_version": int(
                    config["indices"]["formula_version"]
                ),
                "denominator_epsilon": float(
                    config["indices"]["denominator_epsilon"]
                ),
                "savi_l": float(config["indices"]["savi_l"]),
                "recipe_sha256": index_recipe_sha256,
                "processing_recipe_sha256": processing_recipe_sha256,
            }
        )
        candidate_properties = {
            "pipeline_stage": "built_up_candidates",
            "pipeline_version": int(config["version"]),
            "epoch": bundle.epoch,
            "source_composite_asset": bundle.source_composite_asset,
            "source_observation_count_asset": bundle.source_count_asset,
            "grid_version": 1,
            "formula_version": int(
                config["indices"]["formula_version"]
            ),
            "threshold_strategy": config["classification"][
                "primary_strategy"
            ],
            "threshold_table_sha256": threshold_table_sha256,
            "recipe_sha256": candidate_recipe_sha256,
            "processing_recipe_sha256": processing_recipe_sha256,
        }

        for name, threshold in bundle.thresholds.items():
            candidate_properties[f"threshold_{name}"] = float(threshold)

        candidate_image = bundle.candidate_image.set(
            candidate_properties
        )

        index_task = start_export(
            index_image,
            description=f"sprawlgp_indices_{bundle.epoch}",
            asset_id=index_asset,
            pyramiding_policy={
                ".default": "mean",
                "valid_composite": "mode",
            },
            grid=grid,
            config=config,
        )
        index_task.update(
            {
                "epoch": bundle.epoch,
                "product_type": "indices",
                "source_composite_asset": (
                    bundle.source_composite_asset
                ),
                "source_count_asset": bundle.source_count_asset,
                "recipe_sha256": index_recipe_sha256,
            }
        )
        tasks.append(index_task)
        output_rows.append(
            {
                **index_task,
                "band_names": ",".join(INDEX_BANDS),
                "continuous_index_layers": len(CONTINUOUS_INDEX_BANDS),
                "binary_candidate_layers": 0,
            }
        )

        candidate_task = start_export(
            candidate_image,
            description=f"sprawlgp_candidates_{bundle.epoch}",
            asset_id=candidate_asset,
            pyramiding_policy={".default": "mode"},
            grid=grid,
            config=config,
        )
        candidate_task.update(
            {
                "epoch": bundle.epoch,
                "product_type": "candidates",
                "source_composite_asset": (
                    bundle.source_composite_asset
                ),
                "source_count_asset": bundle.source_count_asset,
                "recipe_sha256": candidate_recipe_sha256,
            }
        )
        tasks.append(candidate_task)
        output_rows.append(
            {
                **candidate_task,
                "band_names": ",".join(CANDIDATE_BANDS),
                "continuous_index_layers": 0,
                "binary_candidate_layers": len(CANDIDATE_INDICES),
            }
        )

    task_frame = pd.DataFrame(tasks)
    task_path = resolve_project_path(
        config["exports"]["task_manifest"],
        project_root,
    )
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_frame.to_csv(task_path, index=False)

    output_frame = pd.DataFrame(output_rows)
    output_frame.to_csv(
        metadata_directory(config, project_root)
        / config["metadata"]["output_manifest"],
        index=False,
    )
    return task_frame


def run_submission(config_path: Path, submit: bool) -> pd.DataFrame:
    """Run preflight, validate all thresholds and optionally submit exports."""
    if not submit:
        raise ValueError("Pass --submit to create Earth Engine export tasks.")

    preflight = run_preflight(config_path)
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    grid = load_grid_specification(
        resolve_project_path(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    completed = select_completed_composites(
        pd.read_csv(
            resolve_project_path(
                config["inputs"]["composite_manifest"],
                project_root,
            )
        ),
        config,
    )
    core_geometry = load_ee_geometry(
        resolve_project_path(
            config["inputs"]["core_boundary"],
            project_root,
        )
    )
    (
        bundles,
        histogram_archive,
        threshold_rows,
        area_rows,
        failures,
    ) = build_all_epoch_bundles(
        completed,
        core_geometry,
        grid,
        config,
    )
    (
        _,
        _,
        _,
        threshold_sha256,
        recipe_sha256,
    ) = write_pre_export_metadata(
        threshold_rows,
        area_rows,
        histogram_archive,
        config,
        project_root,
        grid,
    )

    if failures:
        raise RuntimeError(
            "At least one Otsu threshold failed. Diagnostic histograms and "
            "threshold rows were written, and no export task was submitted.\n"
            + json.dumps(failures, indent=2)
        )

    tasks = submit_exports(
        bundles,
        config,
        project_root,
        grid,
        threshold_sha256,
        recipe_sha256,
    )

    print(
        json.dumps(
            {
                "preflight_status": preflight["status"],
                "processed_epochs": [
                    bundle.epoch for bundle in bundles
                ],
                "submitted_or_existing_tasks": len(tasks),
                "task_manifest": config["exports"]["task_manifest"],
            },
            indent=2,
        )
    )
    return tasks


def parse_arguments() -> argparse.Namespace:
    """Parse preflight and submission command-line options."""
    parser = argparse.ArgumentParser(
        description="Build spectral indices and Otsu candidate maps."
    )
    parser.add_argument("--config", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the requested built-up candidate operation."""
    arguments = parse_arguments()

    if arguments.preflight_only:
        run_preflight(arguments.config.resolve())
        return

    run_submission(arguments.config.resolve(), arguments.submit)


if __name__ == "__main__":
    main()
