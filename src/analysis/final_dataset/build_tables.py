"""Export and assemble the eligible-cell time dataset and demand table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

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
    stable_object_hash,
    write_json,
)


PASS_STATES = {"PASS", "COMPLETED", "EXISTS"}

IDENTIFIER_COLUMNS = [
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
]

LABEL_COLUMNS = [
    "target_transition_5y",
    "transition_quality",
]

STATE_COLUMNS = [
    "built_state_raw_t",
    "built_state_final_t",
    "built_state_valid_t",
    "valid_observation_count_t",
]

SPECTRAL_COLUMNS = [
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
]

URBAN_CONTEXT_COLUMNS = [
    "built_fraction_3x3_t",
    "built_fraction_5x5_t",
    "built_fraction_11x11_t",
    "neighbourhood_valid_fraction_3x3_t",
    "neighbourhood_valid_fraction_5x5_t",
    "neighbourhood_valid_fraction_11x11_t",
    "distance_to_built_m_t",
    "recent_local_growth_5y_t",
    "recent_local_growth_available_t",
]

STATIC_COLUMNS = [
    "elevation_m",
    "slope_degrees",
    "population_density_t",
    "population_source_year",
]

TABLE_COLUMNS = (
    IDENTIFIER_COLUMNS
    + LABEL_COLUMNS
    + STATE_COLUMNS
    + SPECTRAL_COLUMNS
    + URBAN_CONTEXT_COLUMNS
    + STATIC_COLUMNS
)

FORBIDDEN_FEATURE_TOKENS = (
    "_target",
    "target_blue",
    "target_green",
    "target_red",
    "target_nir",
    "target_swir",
    "temporal_partition",
)


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError(
            "The Earth Engine Python API is required for table exports."
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
    """Create and return the version-1 output directory."""
    path = resolve_project_path(
        config["outputs"]["final_directory"],
        project_root,
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_core_geometry(path: Path):
    """Load the core boundary as an Earth Engine geometry."""
    require_earth_engine()
    frame = gpd.read_file(path)

    if frame.crs is None:
        raise ValueError("The core boundary has no CRS.")

    geographic = frame.to_crs("EPSG:4326")
    geometry = geographic.geometry.iloc[0]

    if len(geographic) != 1:
        geometry = geographic.geometry.unary_union

    if geometry.is_empty:
        raise ValueError("The core boundary is empty.")

    return ee.Geometry(geometry.__geo_interface__)


def core_area_ha(path: Path) -> float:
    """Return official administrative-core area in hectares."""
    frame = gpd.read_file(path)

    if frame.crs is None:
        raise ValueError("The core boundary has no CRS.")

    area = float(
        frame.to_crs("EPSG:32632").geometry.area.sum()
    ) / 10_000

    if area <= 0:
        raise ValueError("The core boundary area is zero.")

    return area


def load_raster_manifest(
    path: Path,
) -> pd.DataFrame:
    """Load finalized state, transition and support asset references."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "product_type",
        "epoch",
        "period_start",
        "period_end",
        "asset_id",
        "state",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Raster output manifest is missing columns: {sorted(missing)}"
        )

    frame["state"] = frame["state"].astype(str).str.upper()

    if not frame["state"].isin(PASS_STATES).all():
        raise ValueError("Raster output manifest contains non-PASS products.")

    return frame


def load_day4_sources(
    path: Path,
    epochs: list[int],
) -> pd.DataFrame:
    """Join each epoch to its composite, count and continuous-index assets."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "epoch",
        "product_type",
        "asset_id",
        "source_composite_asset",
        "source_count_asset",
        "state",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Day 4 output manifest is missing columns: {sorted(missing)}"
        )

    frame["epoch"] = frame["epoch"].astype(int)
    frame["state"] = frame["state"].astype(str).str.upper()
    frame = frame[
        frame["state"].isin(PASS_STATES)
        & frame["epoch"].isin(epochs)
    ]
    candidates = frame[
        frame["product_type"] == "candidates"
    ][
        [
            "epoch",
            "source_composite_asset",
            "source_count_asset",
        ]
    ].copy()
    indices = frame[
        frame["product_type"] == "indices"
    ][["epoch", "asset_id"]].rename(
        columns={"asset_id": "index_asset"}
    )
    joined = candidates.merge(
        indices,
        on="epoch",
        how="inner",
        validate="one_to_one",
    ).sort_values("epoch")

    if joined["epoch"].tolist() != epochs:
        raise ValueError(
            "Day 4 sources do not cover the configured epochs: "
            f"{joined['epoch'].tolist()} != {epochs}"
        )

    return joined.reset_index(drop=True)


def asset_map(
    raster_manifest: pd.DataFrame,
) -> tuple[dict[int, str], dict[tuple[int, int], str], str]:
    """Build state, transition and support lookup dictionaries."""
    state_rows = raster_manifest[
        raster_manifest["product_type"] == "state"
    ].copy()
    transition_rows = raster_manifest[
        raster_manifest["product_type"] == "transition"
    ].copy()
    support_rows = raster_manifest[
        raster_manifest["product_type"] == "tracking_support"
    ]

    if len(support_rows) != 1:
        raise ValueError("Exactly one tracking support asset is required.")

    states = {
        int(row.epoch): str(row.asset_id)
        for row in state_rows.itertuples(index=False)
    }
    transitions = {
        (int(row.period_start), int(row.period_end)): str(row.asset_id)
        for row in transition_rows.itertuples(index=False)
    }
    return states, transitions, str(support_rows.iloc[0]["asset_id"])


def assert_feature_schema_has_no_leakage(
    columns: list[str],
) -> None:
    """Reject target-year predictors and modelling partitions."""
    feature_columns = [
        column
        for column in columns
        if column not in LABEL_COLUMNS
        and column not in {"target_year", "transition_order"}
    ]

    for column in feature_columns:
        lowered = column.casefold()

        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(
                f"Potential target leakage in feature column: {column}"
            )


def projection_from_grid(grid: dict[str, Any]):
    """Construct the exact frozen Earth Engine projection."""
    require_earth_engine()
    return ee.Projection(
        str(grid["crs"]),
        [float(value) for value in grid["transform"]],
    )



def coordinate_bands(grid: dict[str, Any]):
    """Create stable cell IDs and projected pixel-centre coordinates.

    ``ee.Image.pixelCoordinates`` returns coordinates in pixel space for the
    supplied projection. Those values must first be converted to integer row
    and column indices. Projected metre coordinates are then reconstructed
    from the frozen affine transform.
    """
    require_earth_engine()
    projection = projection_from_grid(grid)
    pixel_coordinates = ee.Image.pixelCoordinates(
        projection
    ).toDouble()
    column = (
        pixel_coordinates.select("x")
        .floor()
        .rename("column")
        .toInt32()
    )
    row = (
        pixel_coordinates.select("y")
        .floor()
        .rename("row")
        .toInt32()
    )
    resolution = float(grid["resolution_m"])
    xmin = float(grid["extent"]["xmin"])
    ymax = float(grid["extent"]["ymax"])
    x_center = (
        ee.Image.constant(xmin)
        .add(
            column.toDouble()
            .add(0.5)
            .multiply(resolution)
        )
        .rename("x_center_m")
        .toDouble()
    )
    y_center = (
        ee.Image.constant(ymax)
        .subtract(
            row.toDouble()
            .add(0.5)
            .multiply(resolution)
        )
        .rename("y_center_m")
        .toDouble()
    )
    cell_id = (
        row.toInt64()
        .multiply(int(grid["width"]))
        .add(column.toInt64())
        .rename("cell_id")
    )
    lonlat = ee.Image.pixelLonLat().select(
        ["longitude", "latitude"]
    )

    return ee.Image.cat(
        [
            cell_id,
            row,
            column,
            x_center,
            y_center,
            lonlat,
        ]
    )


def neighbourhood_features(
    state_image: Any,
    sizes: list[int],
):
    """Calculate built fractions without treating invalid cells as non-built."""
    require_earth_engine()
    valid = (
        state_image.select("built_state_valid")
        .eq(1)
        .unmask(0)
        .toFloat()
    )
    built = (
        state_image.select("built_state_final")
        .unmask(0)
        .multiply(valid)
        .toFloat()
    )
    bands = []

    for size in sizes:
        if size <= 0 or size % 2 == 0:
            raise ValueError(
                f"Neighbourhood size must be positive and odd: {size}"
            )

        radius = (int(size) - 1) // 2
        kernel = ee.Kernel.square(
            radius=radius,
            units="pixels",
            normalize=False,
        )
        valid_count = valid.reduceNeighborhood(
            reducer=ee.Reducer.sum(),
            kernel=kernel,
            skipMasked=False,
        )
        built_count = built.reduceNeighborhood(
            reducer=ee.Reducer.sum(),
            kernel=kernel,
            skipMasked=False,
        )
        built_fraction = (
            built_count.divide(valid_count.max(1))
            .where(valid_count.eq(0), 0)
            .rename(f"built_fraction_{size}x{size}_t")
            .toFloat()
        )
        valid_fraction = (
            valid_count.divide(float(size * size))
            .rename(
                f"neighbourhood_valid_fraction_{size}x{size}_t"
            )
            .toFloat()
        )
        bands.extend([built_fraction, valid_fraction])

    return ee.Image.cat(bands)


def distance_to_built(
    state_image: Any,
    grid: dict[str, Any],
):
    """Calculate full-grid Euclidean distance to origin-year built pixels."""
    require_earth_engine()
    diagonal_pixels = int(
        math.ceil(
            math.sqrt(
                int(grid["width"]) ** 2
                + int(grid["height"]) ** 2
            )
        )
    )
    built = (
        state_image.select("built_state_final")
        .unmask(0)
        .eq(1)
    )
    return (
        built.fastDistanceTransform(
            neighborhood=diagonal_pixels,
            units="pixels",
            metric="squared_euclidean",
        )
        .sqrt()
        .multiply(float(grid["resolution_m"]))
        .rename("distance_to_built_m_t")
        .toFloat()
    )


def recent_growth_features(
    previous_transition: Any | None,
    window_size: int,
):
    """Calculate neighbourhood growth from the preceding transition only."""
    require_earth_engine()

    if previous_transition is None:
        return ee.Image.cat(
            [
                ee.Image.constant(0)
                .rename("recent_local_growth_5y_t")
                .toFloat(),
                ee.Image.constant(0)
                .rename("recent_local_growth_available_t")
                .toUint8(),
            ]
        )

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(
            "The recent-growth window must be positive and odd."
        )

    common = (
        previous_transition.select("common_valid")
        .eq(1)
        .unmask(0)
        .toFloat()
    )
    growth = (
        previous_transition.select("target_transition_5y")
        .unmask(0)
        .multiply(common)
        .toFloat()
    )
    kernel = ee.Kernel.square(
        radius=(window_size - 1) // 2,
        units="pixels",
        normalize=False,
    )
    valid_count = common.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=kernel,
        skipMasked=False,
    )
    growth_count = growth.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=kernel,
        skipMasked=False,
    )
    fraction = (
        growth_count.divide(valid_count.max(1))
        .where(valid_count.eq(0), 0)
        .rename("recent_local_growth_5y_t")
        .toFloat()
    )
    available = (
        valid_count.gt(0)
        .rename("recent_local_growth_available_t")
        .toUint8()
    )
    return ee.Image.cat([fraction, available])


# def population_density(
#     epoch: int,
#     config: dict[str, Any],
# ):
#     """Return native-cell GHSL population as persons per square kilometre."""
#     require_earth_engine()
#     asset_id = f"{config['population']['collection'].rstrip('/')}/{epoch}"
#     count = ee.Image(asset_id).select(config["population"]["band"])
#     density = (
#         count.divide(ee.Image.pixelArea())
#         .multiply(1_000_000)
#         .rename("population_density_t")
#         .unmask(0)
#         .toFloat()
#     )
#     source_year = (
#         ee.Image.constant(int(epoch))
#         .rename("population_source_year")
#         .toInt16()
#     )
#     return ee.Image.cat([density, source_year])

def population_density(
    epoch: int,
    config: dict[str, Any],
):
    """Return GHSL population density at the forecast origin.

    GHSL ``population_count`` stores inhabitants per native cell. The
    conversion to inhabitants per square kilometre is therefore performed
    with the native GHSL pixel area before evaluation on the 30 m project grid.

    Args:
        epoch: GHSL epoch matching the forecast origin.
        config: Final-dataset configuration.

    Returns:
        An image containing ``population_density_t`` and
        ``population_source_year``.

    Raises:
        ValueError: If unsupported units or resampling are configured.
    """
    require_earth_engine()

    population_config = config["population"]

    source_unit = population_config.get(
        "source_unit",
        "persons_per_native_cell",
    )
    output_unit = population_config.get(
        "output_unit",
        "persons_per_square_kilometre",
    )
    resampling = population_config.get(
        "resampling",
        "nearest",
    )

    if source_unit != "persons_per_native_cell":
        raise ValueError(
            "GHSL population_count must be interpreted as "
            "'persons_per_native_cell'."
        )

    if output_unit != "persons_per_square_kilometre":
        raise ValueError(
            "Only 'persons_per_square_kilometre' is supported."
        )

    if resampling not in {"nearest", "bilinear"}:
        raise ValueError(
            "Population resampling must be 'nearest' or 'bilinear'."
        )

    asset_id = (
        f"{population_config['collection'].rstrip('/')}/{int(epoch)}"
    )

    count = (
        ee.Image(asset_id)
        .select(population_config["band"])
        .toFloat()
    )

    # Calculate the area on the native GHSL grid.
    native_projection = count.projection()

    native_pixel_area_m2 = (
        ee.Image.pixelArea()
        .reproject(native_projection)
        .rename("native_pixel_area_m2")
    )

    density = (
        count.divide(native_pixel_area_m2)
        .multiply(1_000_000.0)
        .rename("population_density_t")
        .setDefaultProjection(native_projection)
    )

    # Nearest preserves the native piecewise-constant density surface.
    if resampling == "bilinear":
        density = density.resample("bilinear")

    density = density.unmask(0).toFloat()

    source_year = (
        ee.Image.constant(int(epoch))
        .rename("population_source_year")
        .toInt16()
    )

    return ee.Image.cat(
        [
            density,
            source_year,
        ]
    )


def build_feature_image(
    *,
    transition_order: int,
    origin: int,
    target: int,
    state_asset: str,
    transition_asset: str,
    previous_transition_asset: str | None,
    composite_asset: str,
    index_asset: str,
    terrain_asset: str,
    grid: dict[str, Any],
    config: dict[str, Any],
):
    """Assemble all origin-year predictors and one five-year label."""
    require_earth_engine()
    state = ee.Image(state_asset)
    transition = ee.Image(transition_asset)
    previous_transition = (
        ee.Image(previous_transition_asset)
        if previous_transition_asset is not None
        else None
    )
    coordinates = coordinate_bands(grid)
    time_bands = ee.Image.cat(
        [
            ee.Image.constant(int(transition_order))
            .rename("transition_order")
            .toInt16(),
            ee.Image.constant(int(origin))
            .rename("forecast_origin")
            .toInt16(),
            ee.Image.constant(int(target))
            .rename("target_year")
            .toInt16(),
        ]
    )
    labels = transition.select(
        ["target_transition_5y", "transition_quality"]
    )
    state_bands = state.select(
        [
            "built_state_raw",
            "built_state_final",
            "built_state_valid",
            "valid_observation_count",
        ],
        STATE_COLUMNS,
    )
    spectral = ee.Image(composite_asset).select(
        config["predictors"]["spectral_bands"],
        [
            f"{name}_t"
            for name in config["predictors"]["spectral_bands"]
        ],
    )
    indices = ee.Image(index_asset).select(
        config["predictors"]["spectral_indices"],
        [
            f"{name}_t"
            for name in config["predictors"]["spectral_indices"]
        ],
    )
    neighbourhoods = neighbourhood_features(
        state,
        [
            int(value)
            for value in config["predictors"][
                "neighbourhood_sizes_pixels"
            ]
        ],
    )
    distance = distance_to_built(state, grid)
    recent_growth = recent_growth_features(
        previous_transition,
        int(config["predictors"]["recent_growth_window_pixels"]),
    )
    terrain = ee.Image(terrain_asset).select(
        ["elevation_m", "slope_degrees"]
    )
    population = population_density(origin, config)
    image = ee.Image.cat(
        [
            coordinates,
            time_bands,
            labels,
            state_bands,
            spectral,
            indices,
            neighbourhoods,
            distance,
            recent_growth,
            terrain,
            population,
        ]
    ).select(TABLE_COLUMNS)
    eligible = transition.select("eligible_nonbuilt").eq(1)
    complete = image.mask().reduce(ee.Reducer.min()).eq(1)
    return image.updateMask(eligible.And(complete))


def export_table_to_drive(
    collection: Any,
    *,
    description: str,
    file_prefix: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Submit one batch CSV export to the configured Google Drive folder."""
    require_earth_engine()
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=str(config["exports"]["drive_folder"]),
        fileNamePrefix=file_prefix,
        fileFormat="CSV",
        selectors=TABLE_COLUMNS,
    )
    task.start()
    status = task.status()
    return {
        "task_id": status.get("id", task.id),
        "state": status.get("state", "READY"),
        "description": description,
        "file_prefix": file_prefix,
    }


def transition_statistics(
    transition_asset: str,
    origin: int,
    target: int,
    core: Any,
    grid: dict[str, Any],
    core_area: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate compact demand statistics on pairwise common support."""
    require_earth_engine()
    transition = ee.Image(transition_asset)
    common = transition.select("common_valid").eq(1)
    eligible = transition.select("eligible_nonbuilt").eq(1)
    target_image = transition.select("target_transition_5y").unmask(0)
    one = ee.Image.constant(1)
    pixel_area = ee.Image.pixelArea()
    start_density = population_density(origin, config).select(
        "population_density_t"
    )
    end_density = population_density(target, config).select(
        "population_density_t"
    )
    bands = ee.Image.cat(
        [
            one.updateMask(common).rename("common_valid_cells"),
            pixel_area.updateMask(common).rename("common_valid_area_m2"),
            one.updateMask(eligible).rename("eligible_nonbuilt_cells"),
            pixel_area.updateMask(eligible).rename(
                "eligible_nonbuilt_area_m2"
            ),
            one.updateMask(target_image.eq(1)).rename("new_built_cells"),
            pixel_area.updateMask(target_image.eq(1)).rename(
                "new_built_area_m2"
            ),
            start_density.multiply(pixel_area)
            .divide(1_000_000)
            .updateMask(common)
            .rename("population_start"),
            end_density.multiply(pixel_area)
            .divide(1_000_000)
            .updateMask(common)
            .rename("population_end"),
        ]
    )
    values = bands.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=core,
        crs=str(grid["crs"]),
        crsTransform=[float(value) for value in grid["transform"]],
        maxPixels=int(config["exports"]["max_pixels"]),
        tileScale=4,
    ).getInfo()
    common_area_ha = float(
        values.get("common_valid_area_m2", 0) or 0
    ) / 10_000
    support_pct = common_area_ha / core_area * 100
    warning_pct = float(
        config["quality"]["pairwise_support_warning_pct"]
    )
    population_start = float(
        values.get("population_start", 0) or 0
    )
    population_end = float(
        values.get("population_end", 0) or 0
    )
    return {
        "period_start": int(origin),
        "period_end": int(target),
        "common_valid_cells": int(
            round(float(values.get("common_valid_cells", 0) or 0))
        ),
        "common_valid_area_ha": common_area_ha,
        "common_valid_pct_of_core": support_pct,
        "eligible_nonbuilt_cells": int(
            round(float(values.get("eligible_nonbuilt_cells", 0) or 0))
        ),
        "eligible_nonbuilt_area_ha": float(
            values.get("eligible_nonbuilt_area_m2", 0) or 0
        ) / 10_000,
        "new_built_cells": int(
            round(float(values.get("new_built_cells", 0) or 0))
        ),
        "observed_new_built_area_ha": float(
            values.get("new_built_area_m2", 0) or 0
        ) / 10_000,
        "population_start": population_start,
        "population_end": population_end,
        "population_change": population_end - population_start,
        "demand_scope": "PAIRWISE_COMMON_VALID_SUPPORT",
        "quality_flag": (
            "PASS"
            if support_pct >= warning_pct
            else "LIMITED_PAIRWISE_SPATIAL_SUPPORT"
        ),
    }


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Validate finalized raster products and Day 4 feature sources."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    assert_feature_schema_has_no_leakage(TABLE_COLUMNS)
    raster_manifest_path = (
        metadata_directory(config, project_root)
        / config["metadata"]["raster_output_manifest"]
    )

    if not raster_manifest_path.is_file():
        raise FileNotFoundError(
            "Finalize raster products before creating cell-time tables."
        )

    raster_manifest = load_raster_manifest(raster_manifest_path)
    epochs = [int(value) for value in config["mapping"]["epochs"]]
    states, transitions, _ = asset_map(raster_manifest)
    expected_transitions = list(zip(epochs[:-1], epochs[1:]))

    if sorted(states) != epochs:
        raise ValueError(
            f"State epochs do not match configuration: {sorted(states)}"
        )

    if sorted(transitions) != expected_transitions:
        raise ValueError(
            "Transition assets do not match the configured epoch sequence."
        )

    day4_sources = load_day4_sources(
        resolve_project_path(
            config["inputs"]["built_up_output_manifest"],
            project_root,
        ),
        epochs,
    )
    initialize_earth_engine(
        config["project"]["earth_engine_project"]
    )
    result = {
        "status": "PASS",
        "epochs": epochs,
        "transition_count": len(expected_transitions),
        "table_export_count": len(expected_transitions),
        "feature_count": len(TABLE_COLUMNS),
        "drive_folder": config["exports"]["drive_folder"],
        "day4_source_count": int(len(day4_sources)),
    }
    print(json.dumps(result, indent=2))
    return result


def submit_tables(config_path: Path) -> pd.DataFrame:
    """Submit four eligible-cell CSV exports and save demand statistics."""
    preflight = run_preflight(config_path)
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    grid = load_grid_specification(
        resolve_project_path(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    core_path = resolve_project_path(
        config["inputs"]["core_boundary"],
        project_root,
    )
    core = load_core_geometry(core_path)
    core_area = core_area_ha(core_path)
    metadata_dir = metadata_directory(config, project_root)
    raster_manifest = load_raster_manifest(
        metadata_dir / config["metadata"]["raster_output_manifest"]
    )
    states, transitions, _ = asset_map(raster_manifest)
    epochs = [int(value) for value in config["mapping"]["epochs"]]
    sources = load_day4_sources(
        resolve_project_path(
            config["inputs"]["built_up_output_manifest"],
            project_root,
        ),
        epochs,
    ).set_index("epoch")
    terrain_asset = str(config["inputs"]["terrain_asset"])
    projection = projection_from_grid(grid)
    tasks = []
    statistics = []

    for order, (origin, target) in enumerate(
        zip(epochs[:-1], epochs[1:])
    ):
        previous_key = (
            (epochs[order - 1], origin)
            if order > 0
            else None
        )
        feature_image = build_feature_image(
            transition_order=order,
            origin=origin,
            target=target,
            state_asset=states[origin],
            transition_asset=transitions[(origin, target)],
            previous_transition_asset=(
                transitions[previous_key]
                if previous_key is not None
                else None
            ),
            composite_asset=str(
                sources.loc[origin, "source_composite_asset"]
            ),
            index_asset=str(sources.loc[origin, "index_asset"]),
            terrain_asset=terrain_asset,
            grid=grid,
            config=config,
        )
        collection = feature_image.sample(
            region=core,
            projection=projection,
            geometries=False,
            dropNulls=True,
            tileScale=4,
        )
        prefix = (
            f"{config['exports']['table_file_prefix']}_"
            f"{origin}_{target}"
        )
        task = export_table_to_drive(
            collection,
            description=f"sprawlgp_{prefix}",
            file_prefix=prefix,
            config=config,
        )
        task.update(
            {
                "transition_order": order,
                "period_start": origin,
                "period_end": target,
            }
        )
        tasks.append(task)
        statistics.append(
            transition_statistics(
                transitions[(origin, target)],
                origin,
                target,
                core,
                grid,
                core_area,
                config,
            )
        )

    task_frame = pd.DataFrame(tasks)
    task_path = metadata_dir / config["metadata"]["table_task_manifest"]
    task_frame.to_csv(task_path, index=False)
    statistics_path = (
        metadata_dir / config["metadata"]["transition_statistics"]
    )
    write_json(
        statistics_path,
        {
            "demand_scope": "PAIRWISE_COMMON_VALID_SUPPORT",
            "transitions": statistics,
        },
    )
    print(
        json.dumps(
            {
                "preflight_status": preflight["status"],
                "submitted_table_tasks": len(task_frame),
                "drive_folder": config["exports"]["drive_folder"],
                "task_manifest": str(task_path),
                "transition_statistics": str(statistics_path),
            },
            indent=2,
        )
    )
    return task_frame


def refresh_tasks(config_path: Path) -> pd.DataFrame:
    """Refresh table task states with one Earth Engine task-list call."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    initialize_earth_engine(
        config["project"]["earth_engine_project"]
    )
    task_path = (
        metadata_directory(config, project_root)
        / config["metadata"]["table_task_manifest"]
    )

    if not task_path.is_file():
        raise FileNotFoundError(
            "Submit table exports before requesting task status."
        )

    frame = pd.read_csv(task_path, keep_default_na=False)
    statuses = {
        task.id: task.status()
        for task in ee.batch.Task.list()
    }

    for index, row in frame.iterrows():
        task_id = str(row["task_id"]).strip()
        status = statuses.get(task_id)

        if status is not None:
            frame.at[index, "state"] = status.get("state", "UNKNOWN")
            frame.at[index, "error_message"] = status.get(
                "error_message",
                "",
            )

    frame.to_csv(task_path, index=False)
    print(
        json.dumps(
            {
                "task_states": {
                    str(key): int(value)
                    for key, value in frame["state"].value_counts().items()
                }
            },
            indent=2,
        )
    )
    return frame


def read_transition_csvs(
    staging_directory: Path,
    prefix: str,
    origin: int,
    target: int,
) -> pd.DataFrame:
    """Read one Drive export, including possible CSV shards."""
    pattern = f"{prefix}_{origin}_{target}*.csv"
    paths = sorted(staging_directory.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No downloaded CSV matches {staging_directory / pattern}"
        )

    frames = [pd.read_csv(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    result = result.drop(
        columns=["system:index", ".geo"],
        errors="ignore",
    )
    missing = set(TABLE_COLUMNS).difference(result.columns)

    if missing:
        raise ValueError(
            f"Downloaded table {origin}_{target} is missing: "
            f"{sorted(missing)}"
        )

    return result[TABLE_COLUMNS].copy()



def repair_coordinate_columns(
    frame: pd.DataFrame,
    grid: dict[str, Any],
) -> pd.DataFrame:
    """Repair legacy Day 6 exports with pixel coordinates labelled as metres.

    Earlier exports stored the pixel-space centre coordinates returned by
    ``ee.Image.pixelCoordinates`` in ``x_center_m`` and ``y_center_m``. The
    row and column calculations then subtracted projected UTM origins from
    those pixel values, producing invalid indices and colliding cell IDs.

    Correct exports are returned unchanged. Legacy exports are repaired
    deterministically from the pixel-space centre coordinates without
    modifying spectral, label or predictor values.
    """
    required = {
        "cell_id",
        "row",
        "column",
        "x_center_m",
        "y_center_m",
        "forecast_origin",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Cannot validate cell coordinates; missing columns: "
            f"{sorted(missing)}"
        )

    result = frame.copy()
    numeric_columns = [
        "cell_id",
        "row",
        "column",
        "x_center_m",
        "y_center_m",
    ]

    for column_name in numeric_columns:
        result[column_name] = pd.to_numeric(
            result[column_name],
            errors="raise",
        )

    width = int(grid["width"])
    height = int(grid["height"])
    resolution = float(grid["resolution_m"])
    xmin = float(grid["extent"]["xmin"])
    ymax = float(grid["extent"]["ymax"])

    observed_row = result["row"].to_numpy(dtype=float)
    observed_column = result["column"].to_numpy(dtype=float)
    observed_x = result["x_center_m"].to_numpy(dtype=float)
    observed_y = result["y_center_m"].to_numpy(dtype=float)

    expected_x = (
        xmin + (observed_column + 0.5) * resolution
    )
    expected_y = (
        ymax - (observed_row + 0.5) * resolution
    )
    current_indices_valid = (
        (observed_column >= 0)
        & (observed_column < width)
        & (observed_row >= 0)
        & (observed_row < height)
    ).all()
    current_coordinates_valid = (
        np.allclose(
            observed_x,
            expected_x,
            atol=1e-6,
        )
        and np.allclose(
            observed_y,
            expected_y,
            atol=1e-6,
        )
    )

    if current_indices_valid and current_coordinates_valid:
        result["row"] = result["row"].astype("int64")
        result["column"] = result["column"].astype("int64")
        result["cell_id"] = (
            result["row"] * width + result["column"]
        ).astype("int64")
        return result

    legacy_pixel_coordinates = (
        (observed_x >= 0)
        & (observed_x < width)
        & (observed_y >= 0)
        & (observed_y < height)
    )

    if not legacy_pixel_coordinates.all():
        raise ValueError(
            "Cell coordinate columns are neither valid projected centres "
            "nor recognised legacy pixel-space coordinates."
        )

    repaired_column = np.floor(observed_x).astype("int64")
    repaired_row = np.floor(observed_y).astype("int64")

    if not (
        (
            (repaired_column >= 0)
            & (repaired_column < width)
            & (repaired_row >= 0)
            & (repaired_row < height)
        ).all()
    ):
        raise ValueError(
            "Repaired row or column indices fall outside the frozen grid."
        )

    result["column"] = repaired_column
    result["row"] = repaired_row
    result["cell_id"] = (
        repaired_row * width + repaired_column
    ).astype("int64")
    result["x_center_m"] = (
        xmin + (repaired_column + 0.5) * resolution
    )
    result["y_center_m"] = (
        ymax - (repaired_row + 0.5) * resolution
    )

    if result.duplicated(
        ["cell_id", "forecast_origin"]
    ).any():
        raise ValueError(
            "Cell IDs remain duplicated after legacy coordinate repair."
        )

    return result



def validate_cell_time_dataset(frame: pd.DataFrame) -> None:
    """Apply core target, time, uniqueness and leakage consistency checks."""
    assert_feature_schema_has_no_leakage(frame.columns.tolist())

    if frame.empty:
        raise ValueError("The assembled cell-time dataset is empty.")

    if frame.duplicated(["cell_id", "forecast_origin"]).any():
        raise ValueError(
            "(cell_id, forecast_origin) must uniquely identify each row."
        )

    if not (
        frame["target_year"].astype(int)
        == frame["forecast_origin"].astype(int) + 5
    ).all():
        raise ValueError("Every row must use a five-year target horizon.")

    if not set(frame["target_transition_5y"].dropna().astype(int)).issubset(
        {0, 1}
    ):
        raise ValueError("Transition targets must be binary.")

    if not (frame["built_state_final_t"].astype(int) == 0).all():
        raise ValueError("Every model row must be non-built at origin.")

    if not (frame["built_state_valid_t"].astype(int) == 1).all():
        raise ValueError("Every model row must be valid at origin.")

    positive_counts = frame.groupby("cell_id")[
        "target_transition_5y"
    ].sum()

    if (positive_counts > 1).any():
        raise ValueError(
            "A cell cannot record more than one first built conversion."
        )



def synchronise_demand_with_cell_time(
    demand: pd.DataFrame,
    dataset: pd.DataFrame,
    grid: dict[str, Any],
) -> pd.DataFrame:
    """Align demand counts and full-cell areas with the released Parquet.

    Earth Engine area reductions over the administrative polygon may weight
    boundary pixels fractionally. The released cell-time dataset, however,
    contains one row per eligible 30 m grid cell selected by pixel centre.
    Counts and full-cell areas in ``historical_demand.csv`` must therefore be
    derived from the same rows used by the modelling dataset.

    Pairwise common-support area and population values remain unchanged.
    """
    required_demand = {
        "period_start",
        "period_end",
        "eligible_nonbuilt_cells",
        "eligible_nonbuilt_area_ha",
        "new_built_cells",
        "observed_new_built_area_ha",
    }
    missing_demand = required_demand.difference(demand.columns)

    if missing_demand:
        raise ValueError(
            "Historical demand statistics are missing columns: "
            f"{sorted(missing_demand)}"
        )

    required_dataset = {
        "forecast_origin",
        "target_year",
        "target_transition_5y",
    }
    missing_dataset = required_dataset.difference(dataset.columns)

    if missing_dataset:
        raise ValueError(
            "Cell-time dataset is missing demand columns: "
            f"{sorted(missing_dataset)}"
        )

    resolution_m = float(grid["resolution_m"])
    pixel_area_ha = resolution_m * resolution_m / 10_000
    result = demand.copy()

    for index, row in result.iterrows():
        origin = int(row["period_start"])
        target = int(row["period_end"])
        group = dataset[
            (dataset["forecast_origin"].astype(int) == origin)
            & (dataset["target_year"].astype(int) == target)
        ]

        if group.empty:
            raise ValueError(
                f"No cell-time rows exist for transition {origin}_{target}."
            )

        labels = pd.to_numeric(
            group["target_transition_5y"],
            errors="raise",
        ).astype(int)

        if not set(labels).issubset({0, 1}):
            raise ValueError(
                f"Transition {origin}_{target} contains non-binary labels."
            )

        eligible_cells = int(len(group))
        new_built_cells = int(labels.sum())

        result.at[index, "eligible_nonbuilt_cells"] = eligible_cells
        result.at[index, "eligible_nonbuilt_area_ha"] = (
            eligible_cells * pixel_area_ha
        )
        result.at[index, "new_built_cells"] = new_built_cells
        result.at[index, "observed_new_built_area_ha"] = (
            new_built_cells * pixel_area_ha
        )

    return result



def assemble_tables(config_path: Path) -> dict[str, Any]:
    """Combine downloaded transition CSVs into Parquet and demand CSV."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    grid = load_grid_specification(
        resolve_project_path(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    staging = resolve_project_path(
        config["outputs"]["staging_table_directory"],
        project_root,
    )

    if not staging.is_dir():
        raise FileNotFoundError(
            f"Downloaded table directory does not exist: {staging}"
        )

    epochs = [int(value) for value in config["mapping"]["epochs"]]
    frames = []

    for origin, target in zip(epochs[:-1], epochs[1:]):
        frame = read_transition_csvs(
            staging,
            str(config["exports"]["table_file_prefix"]),
            origin,
            target,
        )
        frame["transition_id"] = f"{origin}_{target}"
        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = repair_coordinate_columns(
        dataset,
        grid,
    )
    dataset["cell_id"] = pd.to_numeric(
        dataset["cell_id"],
        errors="raise",
    ).astype("int64")

    for column in (
        "row",
        "column",
        "transition_order",
        "forecast_origin",
        "target_year",
        "target_transition_5y",
        "transition_quality",
        "built_state_raw_t",
        "built_state_final_t",
        "built_state_valid_t",
        "valid_observation_count_t",
        "recent_local_growth_available_t",
        "population_source_year",
    ):
        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="raise",
        ).astype("int64")

    validate_cell_time_dataset(dataset)
    dataset = dataset.sort_values(
        ["transition_order", "cell_id"]
    ).reset_index(drop=True)
    final_dir = final_directory(config, project_root)
    parquet_path = (
        final_dir / config["outputs"]["cell_time_dataset"]
    )
    dataset.to_parquet(parquet_path, index=False)
    statistics_path = (
        metadata_directory(config, project_root)
        / config["metadata"]["transition_statistics"]
    )

    if not statistics_path.is_file():
        raise FileNotFoundError(
            "Transition statistics are missing; resubmit table phase."
        )

    statistics = json.loads(
        statistics_path.read_text(encoding="utf-8")
    )["transitions"]
    demand = pd.DataFrame(statistics)
    demand = synchronise_demand_with_cell_time(
        demand,
        dataset,
        grid,
    )
    demand_path = final_dir / config["outputs"]["historical_demand"]
    demand.to_csv(demand_path, index=False)
    result = {
        "status": "PASS",
        "row_count": int(len(dataset)),
        "transition_count": int(dataset["transition_id"].nunique()),
        "positive_count": int(dataset["target_transition_5y"].sum()),
        "cell_time_dataset": str(parquet_path),
        "historical_demand": str(demand_path),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse table-phase command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export and assemble the cell-time dataset."
    )
    parser.add_argument("--config", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight-only", action="store_true")
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--status-only", action="store_true")
    actions.add_argument("--assemble", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected table-phase action."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()

    if arguments.preflight_only:
        run_preflight(config_path)
    elif arguments.submit:
        submit_tables(config_path)
    elif arguments.status_only:
        refresh_tasks(config_path)
    elif arguments.assemble:
        assemble_tables(config_path)


if __name__ == "__main__":
    main()
