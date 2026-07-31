"""Compute tracking metrics and freeze the immutable provisional release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from scipy import ndimage

try:
    import ee
except ImportError:  # Enables local unit tests without Earth Engine.
    ee = None

from src.analysis.orchestration.common import (
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_yaml,
    resolve_project_path,
    sha256_file,
    stable_object_hash,
    write_json,
    write_yaml,
)


FINAL_CELL_TIME_COLUMNS = [
    "cell_id",
    "row",
    "column",
    "x_center_m",
    "y_center_m",
    "longitude",
    "latitude",
    "transition_order",
    "forecast_origin",
    "target_year",
    "transition_id",
    "target_transition_5y",
    "transition_quality",
    "built_state_raw_t",
    "built_state_final_t",
    "built_state_valid_t",
    "valid_observation_count_t",
    "blue_t",
    "green_t",
    "red_t",
    "nir_t",
    "swir1_t",
    "swir2_t",
    "savi_t",
    "mndwi_t",
    "ndbi_t",
    "ibui_t",
    "ndbsui_t",
    "built_fraction_3x3_t",
    "built_fraction_5x5_t",
    "built_fraction_11x11_t",
    "neighbourhood_valid_fraction_3x3_t",
    "neighbourhood_valid_fraction_5x5_t",
    "neighbourhood_valid_fraction_11x11_t",
    "distance_to_built_m_t",
    "recent_local_growth_5y_t",
    "recent_local_growth_available_t",
    "elevation_m",
    "slope_degrees",
    "population_density_t",
    "population_source_year",
]


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError(
            "The Earth Engine Python API is required to download tracking "
            "state bands."
        )


def metadata_directory(
    config: dict[str, Any],
    project_root: Path,
) -> Path:
    """Create and return the final-dataset metadata directory."""
    path = resolve_project_path(
        config["metadata"]["directory"],
        project_root,
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def final_directory(
    config: dict[str, Any],
    project_root: Path,
) -> Path:
    """Create and return the version-1 final directory."""
    path = resolve_project_path(
        config["outputs"]["final_directory"],
        project_root,
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_grid_region(grid: dict[str, Any]):
    """Return an inset rectangle selecting exactly the frozen grid cells."""
    require_earth_engine()
    extent = grid["extent"]
    inset = float(grid["resolution_m"]) / 4.0
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


def load_raster_manifest(path: Path) -> pd.DataFrame:
    """Load the validated Earth Engine raster output manifest."""
    if not path.is_file():
        raise FileNotFoundError(
            "Finalize raster products before final dataset release."
        )

    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "product_type",
        "epoch",
        "asset_id",
        "state",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Raster manifest is missing columns: {sorted(missing)}"
        )

    if not frame["state"].astype(str).str.upper().eq("PASS").all():
        raise ValueError("All raster products must have PASS status.")

    return frame


def tracking_stack_image(
    raster_manifest: pd.DataFrame,
    epochs: list[int],
):
    """Create one compact image for local fixed-support tracking metrics."""
    require_earth_engine()
    state_rows = raster_manifest[
        raster_manifest["product_type"] == "state"
    ].copy()
    state_rows["epoch"] = state_rows["epoch"].astype(int)
    support_rows = raster_manifest[
        raster_manifest["product_type"] == "tracking_support"
    ]

    if len(support_rows) != 1:
        raise ValueError("Exactly one tracking support asset is required.")

    support = (
        ee.Image(str(support_rows.iloc[0]["asset_id"]))
        .select("tracking_common_support")
        .unmask(0)
        .toUint8()
    )
    bands = []

    for epoch in epochs:
        rows = state_rows[state_rows["epoch"] == epoch]

        if len(rows) != 1:
            raise ValueError(f"Expected one final-state asset for {epoch}.")

        band = (
            ee.Image(str(rows.iloc[0]["asset_id"]))
            .select("built_state_final")
            .updateMask(support.eq(1))
            .unmask(255)
            .rename(f"built_{epoch}")
            .toUint8()
        )
        bands.append(band)

    bands.append(support.rename("tracking_common_support"))
    return ee.Image.cat(bands)


def download_tracking_stack(
    raster_manifest: pd.DataFrame,
    grid: dict[str, Any],
    epochs: list[int],
    output_path: Path,
) -> Path:
    """Download the small tracking stack as one exact-grid GeoTIFF."""
    require_earth_engine()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tracking_stack_image(raster_manifest, epochs)
    url = image.getDownloadURL(
        {
            "bands": [
                *[f"built_{epoch}" for epoch in epochs],
                "tracking_common_support",
            ],
            "region": export_grid_region(grid),
            "crs": str(grid["crs"]),
            "crs_transform": [
                float(value) for value in grid["transform"]
            ],
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    output_path.write_bytes(response.content)

    with rasterio.open(output_path) as dataset:
        if (
            dataset.width != int(grid["width"])
            or dataset.height != int(grid["height"])
        ):
            raise ValueError(
                "Downloaded tracking stack has unexpected dimensions: "
                f"{dataset.width} x {dataset.height}"
            )

        if dataset.crs is None or dataset.crs.to_string() != grid["crs"]:
            raise ValueError(
                f"Downloaded tracking stack has unexpected CRS: "
                f"{dataset.crs}"
            )

        observed_transform = list(dataset.transform)[:6]

        if not np.allclose(
            observed_transform,
            [float(value) for value in grid["transform"]],
            atol=1e-9,
        ):
            raise ValueError(
                "Downloaded tracking stack has an unexpected transform: "
                f"{observed_transform}"
            )

        if dataset.count != len(epochs) + 1:
            raise ValueError(
                "Downloaded tracking stack has an unexpected band count."
            )

    return output_path


def compute_patch_metrics(
    built_mask: np.ndarray,
    pixel_area_ha: float,
) -> tuple[int, float, float]:
    """Return NUMP, MPS and built area using eight-neighbour connectivity."""
    built = np.asarray(built_mask, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    _, number_of_patches = ndimage.label(
        built,
        structure=structure,
    )
    built_area_ha = float(built.sum()) * float(pixel_area_ha)
    mean_patch_size_ha = (
        built_area_ha / number_of_patches
        if number_of_patches > 0
        else 0.0
    )
    return int(number_of_patches), mean_patch_size_ha, built_area_ha


def core_area_ha(path: Path) -> float:
    """Calculate the administrative-core area in hectares."""
    frame = gpd.read_file(path)

    if frame.crs is None:
        raise ValueError("The core boundary has no CRS.")

    value = float(
        frame.to_crs("EPSG:32632").geometry.area.sum()
    ) / 10_000

    if value <= 0:
        raise ValueError("The administrative-core area is zero.")

    return value


def compute_tracking_metrics(
    tracking_raster: Path,
    epochs: list[int],
    grid: dict[str, Any],
    core_area: float,
    warning_pct: float,
) -> pd.DataFrame:
    """Calculate built area, PBA, NUMP and MPS on fixed support."""
    with rasterio.open(tracking_raster) as dataset:
        data = dataset.read()

    support = data[-1] == 1
    support_cells = int(support.sum())

    if support_cells == 0:
        raise ValueError("Tracking common support contains no valid cells.")

    pixel_area_ha = (
        float(grid["resolution_m"]) ** 2 / 10_000
    )
    support_area_ha = support_cells * pixel_area_ha
    support_pct = support_area_ha / core_area * 100
    quality_flag = (
        "PASS"
        if support_pct >= float(warning_pct)
        else "LIMITED_COMMON_SPATIAL_SUPPORT"
    )
    rows = []

    for band_index, epoch in enumerate(epochs):
        built = (data[band_index] == 1) & support
        nump, mps_ha, built_area_ha = compute_patch_metrics(
            built,
            pixel_area_ha,
        )
        rows.append(
            {
                "epoch": int(epoch),
                "tracking_support_area_ha": support_area_ha,
                "tracking_support_pct_of_core": support_pct,
                "built_area_ha": built_area_ha,
                "pba": built_area_ha / support_area_ha,
                "nump": nump,
                "mps_ha": mps_ha,
                "tracking_quality_flag": quality_flag,
            }
        )

    return pd.DataFrame(rows)


def feature_schema() -> dict[str, Any]:
    """Return the modelling-neutral primary feature schema."""
    return {
        "identifiers": [
            "cell_id",
            "row",
            "column",
            "x_center_m",
            "y_center_m",
            "longitude",
            "latitude",
            "transition_order",
            "transition_id",
            "forecast_origin",
            "target_year",
        ],
        "spatial_coordinates": ["x_center_m", "y_center_m"],
        "temporal_coordinate": ["forecast_origin"],
        "covariates": [
            "built_state_raw_t",
            "built_state_final_t",
            "built_state_valid_t",
            "valid_observation_count_t",
            "blue_t",
            "green_t",
            "red_t",
            "nir_t",
            "swir1_t",
            "swir2_t",
            "savi_t",
            "mndwi_t",
            "ndbi_t",
            "ibui_t",
            "ndbsui_t",
            "built_fraction_3x3_t",
            "built_fraction_5x5_t",
            "built_fraction_11x11_t",
            "neighbourhood_valid_fraction_3x3_t",
            "neighbourhood_valid_fraction_5x5_t",
            "neighbourhood_valid_fraction_11x11_t",
            "distance_to_built_m_t",
            "recent_local_growth_5y_t",
            "recent_local_growth_available_t",
            "elevation_m",
            "slope_degrees",
            "population_density_t",
            "population_source_year",
        ],
        "label": ["target_transition_5y"],
        "quality": ["transition_quality"],
    }


def data_dictionary() -> dict[str, Any]:
    """Return concise definitions for all released tables."""
    cell_time = {
        "cell_id": "Stable row-major identifier on frozen grid version 1.",
        "row": "Zero-based raster row.",
        "column": "Zero-based raster column.",
        "x_center_m": "Cell-centre easting in EPSG:32632 metres.",
        "y_center_m": "Cell-centre northing in EPSG:32632 metres.",
        "longitude": "Cell-centre longitude in decimal degrees.",
        "latitude": "Cell-centre latitude in decimal degrees.",
        "transition_order": "Chronological index of the five-year transition.",
        "transition_id": "String identifying the origin and target years.",
        "forecast_origin": "Year at which predictors are observed.",
        "target_year": "Five-year target year.",
        "target_transition_5y": "One when an eligible non-built cell becomes built.",
        "transition_quality": "Zero ineligible, one valid, two high support.",
        "built_state_raw_t": "Raw NDBI candidate state at forecast origin.",
        "built_state_final_t": "Persistence-consistent state at origin.",
        "built_state_valid_t": "One when the origin state is observed.",
        "valid_observation_count_t": "Valid Landsat observations at origin.",
        "blue_t": "Blue surface reflectance at origin.",
        "green_t": "Green surface reflectance at origin.",
        "red_t": "Red surface reflectance at origin.",
        "nir_t": "Near-infrared surface reflectance at origin.",
        "swir1_t": "First short-wave infrared reflectance at origin.",
        "swir2_t": "Second short-wave infrared reflectance at origin.",
        "savi_t": "Soil-adjusted vegetation index at origin.",
        "mndwi_t": "Modified normalized difference water index at origin.",
        "ndbi_t": "Normalized difference built-up index at origin.",
        "ibui_t": "Index-based built-up index at origin.",
        "ndbsui_t": "Normalized difference built-up and soil index at origin.",
        "built_fraction_3x3_t": "Built fraction among valid 3x3 neighbours.",
        "built_fraction_5x5_t": "Built fraction among valid 5x5 neighbours.",
        "built_fraction_11x11_t": "Built fraction among valid 11x11 neighbours.",
        "neighbourhood_valid_fraction_3x3_t": "Valid share of 3x3 window.",
        "neighbourhood_valid_fraction_5x5_t": "Valid share of 5x5 window.",
        "neighbourhood_valid_fraction_11x11_t": "Valid share of 11x11 window.",
        "distance_to_built_m_t": "Euclidean distance to origin-year built land.",
        "recent_local_growth_5y_t": "Local growth fraction in prior transition.",
        "recent_local_growth_available_t": "One when prior growth is available.",
        "elevation_m": "SRTM elevation in metres.",
        "slope_degrees": "SRTM-derived slope in degrees.",
        "population_density_t": "GHSL inhabitants per square kilometre at origin.",
        "population_source_year": "GHSL epoch used for the population predictor.",
    }
    demand = {
        "period_start": "Transition origin year.",
        "period_end": "Transition target year.",
        "common_valid_cells": "Cells observed at both transition endpoints.",
        "common_valid_area_ha": "Pairwise common-valid area in hectares.",
        "common_valid_pct_of_core": "Common-valid share of administrative core.",
        "eligible_nonbuilt_cells": "Common-valid cells non-built at origin.",
        "eligible_nonbuilt_area_ha": "Eligible non-built area in hectares.",
        "new_built_cells": "Eligible cells converted during the transition.",
        "observed_new_built_area_ha": "Observed converted area in hectares.",
        "population_start": "Population on pairwise common support at origin.",
        "population_end": "Population on pairwise common support at target.",
        "population_change": "Population end minus population start.",
        "demand_scope": "Spatial support used for demand calculation.",
        "quality_flag": "Pairwise support quality flag.",
    }
    tracking = {
        "epoch": "Built-state epoch.",
        "tracking_support_area_ha": "Fixed all-epoch valid support area.",
        "tracking_support_pct_of_core": "Fixed support share of core area.",
        "built_area_ha": "Built area on the fixed tracking support.",
        "pba": "Built area divided by fixed tracking support area.",
        "nump": "Eight-connected built patch count.",
        "mps_ha": "Mean built patch size in hectares.",
        "tracking_quality_flag": "Fixed-support quality flag.",
    }
    return {
        "cell_time_dataset.parquet": cell_time,
        "historical_demand.csv": demand,
        "urban_sprawl_metrics.csv": tracking,
    }


def validate_final_tables(
    cell_time: pd.DataFrame,
    demand: pd.DataFrame,
    metrics: pd.DataFrame,
    epochs: list[int],
) -> None:
    """Apply final leakage, consistency and tracking checks."""
    missing_columns = set(FINAL_CELL_TIME_COLUMNS).difference(
        cell_time.columns
    )

    if missing_columns:
        raise ValueError(
            f"Cell-time dataset is missing columns: {sorted(missing_columns)}"
        )

    if cell_time.duplicated(["cell_id", "forecast_origin"]).any():
        raise ValueError("Cell-time keys are not unique.")

    if not (
        cell_time["target_year"].astype(int)
        == cell_time["forecast_origin"].astype(int) + 5
    ).all():
        raise ValueError("Cell-time target horizons are not five years.")

    if "temporal_partition" in cell_time.columns:
        raise ValueError("Modelling partitions must not be stored in Day 6.")

    if not set(cell_time["target_transition_5y"].astype(int)).issubset(
        {0, 1}
    ):
        raise ValueError("Cell-time labels are not binary.")

    if not (cell_time["built_state_final_t"].astype(int) == 0).all():
        raise ValueError("A model row is built at forecast origin.")

    expected_origins = epochs[:-1]

    if sorted(cell_time["forecast_origin"].unique().tolist()) != expected_origins:
        raise ValueError("Cell-time origins do not match the four transitions.")

    if len(demand) != len(epochs) - 1:
        raise ValueError("Historical demand must contain four transitions.")

    expected_area = demand["new_built_cells"] * 0.09

    if not np.allclose(
        demand["observed_new_built_area_ha"],
        expected_area,
        atol=1e-6,
    ):
        raise ValueError("Demand area is inconsistent with 30 m cell counts.")

    for row in demand.itertuples(index=False):
        group = cell_time[
            (
                cell_time["forecast_origin"].astype(int)
                == int(row.period_start)
            )
            & (
                cell_time["target_year"].astype(int)
                == int(row.period_end)
            )
        ]
        eligible_cells = int(len(group))
        new_built_cells = int(
            group["target_transition_5y"].astype(int).sum()
        )

        if int(row.eligible_nonbuilt_cells) != eligible_cells:
            raise ValueError(
                "Demand eligible-cell counts do not match the "
                "cell-time dataset."
            )

        if int(row.new_built_cells) != new_built_cells:
            raise ValueError(
                "Demand new-built counts do not match the "
                "cell-time dataset."
            )

    if metrics["epoch"].astype(int).tolist() != epochs:
        raise ValueError("Tracking metrics do not cover all five epochs.")

    if metrics["tracking_support_area_ha"].nunique() != 1:
        raise ValueError("Tracking metrics do not use one fixed support.")

    if not metrics["pba"].between(0, 1).all():
        raise ValueError("PBA values must lie between zero and one.")

    if (metrics["nump"] < 0).any() or (metrics["mps_ha"] < 0).any():
        raise ValueError("Patch metrics cannot be negative.")


def write_dataset_card(
    path: Path,
    config: dict[str, Any],
    row_count: int,
    positive_count: int,
    metrics: pd.DataFrame,
) -> None:
    """Write the concise release documentation."""
    epochs = [int(value) for value in config["mapping"]["epochs"]]
    transitions = [
        f"{start}–{end}"
        for start, end in zip(epochs[:-1], epochs[1:])
    ]
    support_pct = float(
        metrics.iloc[0]["tracking_support_pct_of_core"]
    )
    content = f"""# Yaoundé Urban Expansion 30 m — Version 1

## Purpose

This dataset supports historical urban-expansion tracking and five-year
cell-level built-up conversion modelling for Yaoundé, Cameroon.

## Release status

- Release: `{config['release']['semantic_version']}`
- Status: `{config['release']['release_status']}`
- Mapping method: `{config['mapping']['selected_index']}`
- Mapping selection: `{config['release']['mapping_selection_status']}`
- Manual validation complete: `{str(config['release']['manual_validation_complete']).lower()}`

## Spatial and temporal coverage

- CRS: EPSG:32632
- Resolution: 30 m
- Epochs: {', '.join(str(value) for value in epochs)}
- Transitions: {', '.join(transitions)}
- Tracking support: {support_pct:.2f}% of the administrative core

## Main table

`cell_time_dataset.parquet` contains one row for each pairwise-valid cell that
is non-built at the forecast origin. It contains {row_count:,} rows, including
{positive_count:,} observed conversions.

All predictors are measured at the forecast origin. Recent growth uses only
the preceding transition. No modelling partition is embedded in the release.
GHSL population for 2025 is used only in the historical demand summary and is
a projected epoch.

## Tracking

`urban_sprawl_metrics.csv` reports built area, PBA, NUMP and MPS on one fixed
all-epoch valid support. `historical_demand.csv` reports observed conversion
on pairwise common-valid support.

## Limitations

NDBI is the operational mapping method for this frozen provisional release.
Expert manual validation is pending. A later release will be produced if the
validated mapping method changes. Spatial-support percentages must be retained
when interpreting historical change.
"""
    path.write_text(content, encoding="utf-8")


def checksum_lines(paths: list[Path], base: Path) -> list[str]:
    """Return deterministic SHA-256 manifest lines."""
    return [
        f"{sha256_file(path)}  {path.relative_to(base)}"
        for path in sorted(paths)
    ]


def run_finalize(config_path: Path) -> dict[str, Any]:
    """Compute tracking metrics, documentation, checksums and release version."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    metadata_dir = metadata_directory(config, project_root)
    final_dir = final_directory(config, project_root)
    version_path = (
        metadata_dir / config["metadata"]["final_dataset_version"]
    )

    if (
        bool(config["release"]["protect_existing_release"])
        and version_path.is_file()
    ):
        raise FileExistsError(
            "Version 1 is already frozen and cannot be overwritten."
        )

    grid = load_grid_specification(
        resolve_project_path(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    epochs = [int(value) for value in config["mapping"]["epochs"]]
    raster_manifest_path = (
        metadata_dir / config["metadata"]["raster_output_manifest"]
    )
    raster_manifest = load_raster_manifest(raster_manifest_path)
    initialize_earth_engine(
        config["project"]["earth_engine_project"]
    )
    tracking_raster = resolve_project_path(
        config["outputs"]["staging_tracking_raster"],
        project_root,
    )
    download_tracking_stack(
        raster_manifest,
        grid,
        epochs,
        tracking_raster,
    )
    metrics = compute_tracking_metrics(
        tracking_raster,
        epochs,
        grid,
        core_area_ha(
            resolve_project_path(
                config["inputs"]["core_boundary"],
                project_root,
            )
        ),
        float(config["quality"]["tracking_support_warning_pct"]),
    )
    metrics_path = (
        final_dir / config["outputs"]["urban_sprawl_metrics"]
    )
    metrics.to_csv(metrics_path, index=False)
    cell_time_path = (
        final_dir / config["outputs"]["cell_time_dataset"]
    )
    demand_path = final_dir / config["outputs"]["historical_demand"]

    if not cell_time_path.is_file() or not demand_path.is_file():
        raise FileNotFoundError(
            "Assemble the cell-time dataset and demand table before finalizing."
        )

    cell_time = pd.read_parquet(cell_time_path)
    demand = pd.read_csv(demand_path)
    validate_final_tables(cell_time, demand, metrics, epochs)
    feature_schema_path = (
        final_dir / config["outputs"]["primary_feature_schema"]
    )
    dictionary_path = final_dir / config["outputs"]["data_dictionary"]
    card_path = final_dir / config["outputs"]["dataset_card"]
    write_yaml(feature_schema_path, feature_schema())
    write_yaml(dictionary_path, data_dictionary())
    write_dataset_card(
        card_path,
        config,
        int(len(cell_time)),
        int(cell_time["target_transition_5y"].sum()),
        metrics,
    )
    output_manifest_path = (
        metadata_dir / config["metadata"]["output_manifest"]
    )
    raster_rows = raster_manifest.copy()
    raster_rows["storage"] = "earth_engine_asset"
    raster_rows["path"] = raster_rows["asset_id"]
    raster_rows["sha256"] = raster_rows["recipe_sha256"]
    local_files = [
        cell_time_path,
        demand_path,
        metrics_path,
        feature_schema_path,
        dictionary_path,
        card_path,
    ]
    local_rows = pd.DataFrame(
        [
            {
                "product_type": path.stem,
                "epoch": "",
                "period_start": "",
                "period_end": "",
                "asset_id": "",
                "band_names": "",
                "state": "PASS",
                "recipe_sha256": "",
                "storage": "local_file",
                "path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
            }
            for path in local_files
        ]
    )
    output_manifest = pd.concat(
        [
            raster_rows[local_rows.columns],
            local_rows,
        ],
        ignore_index=True,
    )
    output_manifest.to_csv(output_manifest_path, index=False)
    version = {
        "dataset_id": config["release"]["dataset_id"],
        "version": config["release"]["semantic_version"],
        "release_status": config["release"]["release_status"],
        "mapping_method": config["mapping"]["selected_index"],
        "mapping_selection_status": config["release"][
            "mapping_selection_status"
        ],
        "manual_validation_complete": config["release"][
            "manual_validation_complete"
        ],
        "epochs": epochs,
        "transitions": [
            f"{start}_{end}"
            for start, end in zip(epochs[:-1], epochs[1:])
        ],
        "transition_horizon_years": int(
            config["mapping"]["transition_horizon_years"]
        ),
        "grid_version": 1,
        "crs": grid["crs"],
        "resolution_m": grid["resolution_m"],
        "state_count": len(epochs),
        "transition_count": len(epochs) - 1,
        "cell_time_rows": int(len(cell_time)),
        "positive_transitions": int(
            cell_time["target_transition_5y"].sum()
        ),
        "primary_table": config["outputs"]["cell_time_dataset"],
        "immutable_release": True,
        "output_manifest_sha256": sha256_file(output_manifest_path),
        "outputs": {
            str(path.relative_to(final_dir)): sha256_file(path)
            for path in local_files
        },
    }
    version["stable_signature"] = stable_object_hash(version)
    write_json(version_path, version)
    checksum_path = metadata_dir / config["metadata"]["checksums"]
    checksum_targets = [
        *local_files,
        output_manifest_path,
        version_path,
    ]
    checksum_path.write_text(
        "\n".join(
            checksum_lines(checksum_targets, project_root)
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "PASS",
        "dataset_id": version["dataset_id"],
        "version": version["version"],
        "cell_time_rows": version["cell_time_rows"],
        "positive_transitions": version["positive_transitions"],
        "tracking_support_pct_of_core": float(
            metrics.iloc[0]["tracking_support_pct_of_core"]
        ),
        "final_directory": str(final_dir),
        "version_file": str(version_path),
        "checksums": str(checksum_path),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse finalization arguments."""
    parser = argparse.ArgumentParser(
        description="Freeze the provisional Yaoundé dataset version 1."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Finalize the dataset release."""
    arguments = parse_arguments()
    run_finalize(arguments.config.resolve())


if __name__ == "__main__":
    main()
