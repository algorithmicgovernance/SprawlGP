"""Build the authoritative Yaoundé study geometries and deterministic 30 m grid.

This module reads the study-area configuration, selects the seven Yaoundé ADM3
units, repairs and dissolves their geometries, creates the context buffer and
convex hull, rasterises the core and context masks, and exports a stable
cell-level Parquet table together with machine-readable quality reports.

All paths stored in the YAML configuration are resolved relative to the
repository root, inferred from ``configs/study_area.yaml``. This allows the same
commands to work on another computer without hard-coded absolute paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin, xy
from shapely import make_valid
from shapely.geometry.base import BaseGeometry


def load_configuration(config_path: Path) -> tuple[dict[str, Any], Path]:
    """Load the YAML configuration and infer the repository root directory.

    The expected configuration location is ``<project-root>/configs/study_area.yaml``.
    Returning the project root alongside the parsed configuration ensures that
    every relative path is interpreted consistently, regardless of the caller's
    current working directory.

    Args:
        config_path: Path to the study-area YAML configuration.

    Returns:
        A tuple containing the parsed configuration dictionary and project root.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML file is empty or does not contain a mapping.
    """
    resolved_config_path = config_path.expanduser().resolve()

    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError("The study-area configuration must contain a YAML mapping.")

    project_root = resolved_config_path.parent.parent
    return configuration, project_root


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root.

    Absolute paths are accepted for exceptional local use, while ordinary
    project paths remain portable and are resolved beneath ``project_root``.

    Args:
        project_root: Absolute path to the repository root.
        path_value: Absolute or project-relative path from the configuration.

    Returns:
        A normalised absolute path.
    """
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a dictionary as formatted UTF-8 JSON, creating directories first.

    Args:
        data: JSON-serialisable dictionary to export.
        output_path: Destination JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file without loading it fully in memory.

    The file is read in one-megabyte chunks so the function remains suitable for
    large geospatial inputs and exported raster products.

    Args:
        path: File whose checksum must be calculated.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the requested file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Cannot calculate checksum; file not found: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def snap_down(value: float, resolution: float, anchor: float) -> float:
    """Snap a coordinate downward to the nearest anchored grid line.

    Args:
        value: Coordinate to snap.
        resolution: Grid-cell size in coordinate-system units.
        anchor: Fixed global grid origin for the relevant axis.

    Returns:
        Largest anchored grid coordinate less than or equal to ``value``.
    """
    return math.floor((value - anchor) / resolution) * resolution + anchor


def snap_up(value: float, resolution: float, anchor: float) -> float:
    """Snap a coordinate upward to the nearest anchored grid line.

    Args:
        value: Coordinate to snap.
        resolution: Grid-cell size in coordinate-system units.
        anchor: Fixed global grid origin for the relevant axis.

    Returns:
        Smallest anchored grid coordinate greater than or equal to ``value``.
    """
    return math.ceil((value - anchor) / resolution) * resolution + anchor


def safe_make_valid(geometry: BaseGeometry | None) -> BaseGeometry | None:
    """Repair an invalid geometry while preserving missing or empty geometries.

    Missing and empty geometries are returned unchanged so they can be counted
    and removed explicitly by the caller rather than being silently converted.

    Args:
        geometry: Shapely geometry to validate.

    Returns:
        A valid Shapely geometry when repair is possible, otherwise the original
        missing or empty value.
    """
    if geometry is None or geometry.is_empty:
        return geometry

    return make_valid(geometry)


def validate_source_boundary(boundaries: gpd.GeoDataFrame, source_path: Path) -> None:
    """Validate the minimum structural requirements of a source boundary layer.

    Args:
        boundaries: Source boundary GeoDataFrame loaded by GeoPandas.
        source_path: Path used only to produce informative error messages.

    Raises:
        ValueError: If the source is empty, lacks a CRS, or lacks a geometry column.
    """
    if boundaries.empty:
        raise ValueError(f"Boundary source contains no features: {source_path}")

    if boundaries.crs is None:
        raise ValueError(
            "The boundary file has no CRS. Do not assign one without checking "
            f"the source metadata: {source_path}"
        )

    if boundaries.geometry.name not in boundaries.columns:
        raise ValueError(f"Boundary source has no active geometry column: {source_path}")


def select_study_units(
    boundaries: gpd.GeoDataFrame,
    boundary_config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Select and validate the seven Yaoundé ADM3 units from the source layer.

    P-codes are preferred when configured because they are more stable than
    names. The selected names and parent department are still checked to detect
    unexpected source changes or configuration errors.

    Args:
        boundaries: Boundary features already expressed in the geographic CRS.
        boundary_config: ``boundary`` section of ``study_area.yaml``.

    Returns:
        Copy of the selected ADM3 features.

    Raises:
        KeyError: If a configured field does not exist.
        ValueError: If the count, names, P-codes, or parent relationship differ
            from the configuration.
    """
    name_field = str(boundary_config["name_field"])
    parent_field = boundary_config.get("parent_field")
    pcode_field = boundary_config.get("pcode_field")
    selected_names = [str(value) for value in boundary_config["selected_names"]]
    selected_pcodes = [str(value) for value in boundary_config.get("selected_pcodes", [])]

    required_fields = [name_field]
    if parent_field:
        required_fields.append(str(parent_field))
    if selected_pcodes and pcode_field:
        required_fields.append(str(pcode_field))

    missing_fields = [field for field in required_fields if field not in boundaries.columns]
    if missing_fields:
        raise KeyError(
            f"Configured boundary fields are absent: {missing_fields}. "
            f"Available columns: {list(boundaries.columns)}"
        )

    if selected_pcodes and pcode_field:
        selected = boundaries[
            boundaries[str(pcode_field)].astype(str).isin(selected_pcodes)
        ].copy()
    else:
        selected = boundaries[
            boundaries[name_field].astype(str).isin(selected_names)
        ].copy()

    expected_count = int(boundary_config["expected_selected_units"])
    if len(selected) != expected_count:
        found_names = selected[name_field].astype(str).tolist()
        raise ValueError(
            f"Expected {expected_count} selected units, but found {len(selected)}: "
            f"{found_names}"
        )

    observed_names = set(selected[name_field].astype(str))
    if observed_names != set(selected_names):
        raise ValueError(
            "Selected unit names do not match the configured names. "
            f"Expected {sorted(selected_names)}, found {sorted(observed_names)}"
        )

    if selected_pcodes and pcode_field:
        observed_pcodes = set(selected[str(pcode_field)].astype(str))
        if observed_pcodes != set(selected_pcodes):
            raise ValueError(
                "Selected unit P-codes do not match the configuration. "
                f"Expected {sorted(selected_pcodes)}, found {sorted(observed_pcodes)}"
            )

    parent_name = boundary_config.get("parent_name")
    if parent_field and parent_name:
        observed_parents = sorted(
            selected[str(parent_field)].dropna().astype(str).unique().tolist()
        )
        if observed_parents != [str(parent_name)]:
            raise ValueError(
                f"Expected parent '{parent_name}', found {observed_parents}"
            )

    return selected


def select_comparison_boundary(
    comparison_boundaries: gpd.GeoDataFrame,
    comparison_config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Select the single ADM2 Mfoundi feature used as a reference geometry.

    Args:
        comparison_boundaries: ADM2 source features.
        comparison_config: ``comparison_boundary`` section of the configuration.

    Returns:
        One-row GeoDataFrame containing Mfoundi.

    Raises:
        KeyError: If the configured name or P-code field is absent.
        ValueError: If the selection does not return exactly one feature.
    """
    name_field = str(comparison_config["name_field"])
    selected_name = str(comparison_config["selected_name"])
    pcode_field = comparison_config.get("pcode_field")
    selected_pcode = comparison_config.get("selected_pcode")

    for field in [name_field, pcode_field]:
        if field and str(field) not in comparison_boundaries.columns:
            raise KeyError(
                f"Configured comparison field '{field}' is absent. "
                f"Available columns: {list(comparison_boundaries.columns)}"
            )

    if pcode_field and selected_pcode:
        selected = comparison_boundaries[
            comparison_boundaries[str(pcode_field)].astype(str) == str(selected_pcode)
        ].copy()
    else:
        selected = comparison_boundaries[
            comparison_boundaries[name_field].astype(str) == selected_name
        ].copy()

    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one ADM2 reference feature for {selected_name}, "
            f"but found {len(selected)}."
        )

    observed_name = str(selected.iloc[0][name_field])
    if observed_name != selected_name:
        raise ValueError(
            f"Expected ADM2 name '{selected_name}', found '{observed_name}'."
        )

    return selected


def write_single_geometry(
    geometry: BaseGeometry,
    crs: CRS | str,
    output_path: Path,
    geometry_name: str,
) -> None:
    """Export one named geometry to a GeoPackage layer.

    Existing output files are replaced to keep reruns deterministic and avoid
    duplicate layers left by earlier executions.

    Args:
        geometry: Geometry to store.
        crs: Coordinate reference system assigned to the geometry.
        output_path: Destination GeoPackage path.
        geometry_name: Layer name and descriptive attribute value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    geodataframe = gpd.GeoDataFrame(
        {"geometry_name": [geometry_name]},
        geometry=[geometry],
        crs=crs,
    )
    geodataframe.to_file(output_path, layer=geometry_name, driver="GPKG")


def write_geodataframe(
    geodataframe: gpd.GeoDataFrame,
    output_path: Path,
    layer_name: str,
) -> None:
    """Export a GeoDataFrame to a clean GeoPackage file.

    Args:
        geodataframe: Features to write.
        output_path: Destination GeoPackage path.
        layer_name: Name of the GeoPackage layer.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    geodataframe.to_file(output_path, layer=layer_name, driver="GPKG")


def write_mask(
    output_path: Path,
    mask: np.ndarray,
    transform: Any,
    crs: CRS | str,
    nodata: int = 0,
) -> None:
    """Write a binary grid mask as a compressed single-band GeoTIFF.

    Args:
        output_path: Destination GeoTIFF path.
        mask: Two-dimensional array containing zeros and ones.
        transform: Rasterio affine transform describing pixel placement.
        crs: Raster coordinate reference system.
        nodata: Value representing cells outside the mask.

    Raises:
        ValueError: If the mask is not two-dimensional or is not binary.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a two-dimensional mask, received shape {mask.shape}.")

    unique_values = set(np.unique(mask).tolist())
    if not unique_values <= {0, 1}:
        raise ValueError(f"Mask must be binary; found values {sorted(unique_values)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(mask.astype(np.uint8), 1)


def build_cell_table(
    context_mask: np.ndarray,
    core_mask: np.ndarray,
    transform: Any,
    analysis_crs: CRS,
    grid_config: dict[str, Any],
) -> pd.DataFrame:
    """Create the model-ready table of context-grid cells with stable identifiers.

    Stable identifiers are derived from global anchored grid indices rather than
    local raster row and column numbers. Consequently, an existing cell keeps the
    same identifier if the context extent is later enlarged without changing the
    CRS, resolution, or anchor.

    Args:
        context_mask: Binary mask defining cells retained in the context area.
        core_mask: Binary mask defining cells inside the reporting core.
        transform: Authoritative raster transform.
        analysis_crs: Projected CRS used by the grid.
        grid_config: ``grid`` section of the YAML configuration.

    Returns:
        DataFrame containing cell IDs, local raster indices, global grid indices,
        centre coordinates, membership flags, and cell area.

    Raises:
        ValueError: If generated cell identifiers are not unique.
    """
    resolution = float(grid_config["resolution_m"])
    anchor_x = float(grid_config["anchor_x_m"])
    anchor_y = float(grid_config["anchor_y_m"])
    grid_id = str(
        grid_config.get(
            "grid_id",
            f"yaounde_epsg{analysis_crs.to_epsg()}_{int(resolution)}m_anchor0_v1",
        )
    )
    grid_version = int(grid_config.get("version", 1))

    rows, columns = np.where(context_mask == 1)
    x_coordinates, y_coordinates = xy(transform, rows, columns, offset="center")
    x_coordinates = np.asarray(x_coordinates, dtype=float)
    y_coordinates = np.asarray(y_coordinates, dtype=float)

    global_x_index = np.floor(
        (x_coordinates - anchor_x) / resolution + 1e-9
    ).astype(np.int64)
    global_y_index = np.floor(
        (y_coordinates - anchor_y) / resolution + 1e-9
    ).astype(np.int64)

    epsg = analysis_crs.to_epsg()
    if epsg is None:
        raise ValueError("The analysis CRS must have an EPSG code for stable cell IDs.")

    resolution_label = int(resolution) if resolution.is_integer() else resolution
    cell_ids = [
        f"EPSG{epsg}_{resolution_label}m_{grid_x}_{grid_y}"
        for grid_x, grid_y in zip(global_x_index, global_y_index, strict=True)
    ]

    cell_table = pd.DataFrame(
        {
            "grid_id": grid_id,
            "grid_version": grid_version,
            "cell_id": cell_ids,
            "row": rows.astype(np.int32),
            "column": columns.astype(np.int32),
            "grid_x_index": global_x_index,
            "grid_y_index": global_y_index,
            "x_center_m": x_coordinates,
            "y_center_m": y_coordinates,
            "in_core": core_mask[rows, columns].astype(bool),
            "in_context": True,
            "cell_area_m2": resolution * resolution,
        }
    )

    if not cell_table["cell_id"].is_unique:
        raise ValueError("Generated cell identifiers are not unique.")

    return cell_table


def compare_adm3_union_with_adm2(
    core_geometry: BaseGeometry,
    adm2_geometry: BaseGeometry,
) -> dict[str, float | bool]:
    """Quantify agreement between the dissolved ADM3 core and ADM2 Mfoundi.

    The symmetric difference identifies land contained in only one of the two
    geometries. A ratio near zero indicates that the seven ADM3 units reproduce
    the ADM2 reference closely.

    Args:
        core_geometry: Dissolved union of Yaoundé I–VII.
        adm2_geometry: Mfoundi ADM2 reference geometry.

    Returns:
        Dictionary containing both areas, intersection, symmetric difference,
        relative difference, and mutual coverage indicators.
    """
    symmetric_difference = core_geometry.symmetric_difference(adm2_geometry)
    intersection = core_geometry.intersection(adm2_geometry)
    reference_area = float(adm2_geometry.area)

    difference_ratio = (
        float(symmetric_difference.area) / reference_area if reference_area > 0 else math.nan
    )

    return {
        "adm3_union_area_km2": float(core_geometry.area) / 1_000_000,
        "adm2_mfoundi_area_km2": reference_area / 1_000_000,
        "intersection_area_km2": float(intersection.area) / 1_000_000,
        "symmetric_difference_area_km2": (
            float(symmetric_difference.area) / 1_000_000
        ),
        "symmetric_difference_ratio": difference_ratio,
        "adm2_covers_adm3_union": bool(adm2_geometry.covers(core_geometry)),
        "adm3_union_covers_adm2": bool(core_geometry.covers(adm2_geometry)),
    }


def build_study_grid(config_path: Path) -> dict[str, Any]:
    """Execute the full boundary-processing and deterministic-grid pipeline.

    Args:
        config_path: Path to ``configs/study_area.yaml``.

    Returns:
        Complete boundary report that is also written to disk.

    Raises:
        FileNotFoundError: If required source files are missing.
        KeyError: If required configuration sections or fields are absent.
        ValueError: If source data, selected units, CRS, or grid dimensions fail
            validation checks.
    """
    config, project_root = load_configuration(config_path)

    boundary_config = config["boundary"]
    projection_config = config["projection"]
    geometry_config = config["geometries"]
    grid_config = config["grid"]
    output_config = config["outputs"]

    source_path = resolve_project_path(project_root, boundary_config["source_file"])
    source_hash = sha256_file(source_path)

    boundaries = gpd.read_file(source_path, layer=boundary_config.get("layer"))
    validate_source_boundary(boundaries, source_path)

    input_feature_count = len(boundaries)
    missing_before = int(boundaries.geometry.isna().sum())
    empty_before = int(boundaries.geometry.is_empty.sum())
    invalid_before = int((~boundaries.geometry.is_valid).sum())
    duplicate_before = int(boundaries.geometry.duplicated().sum())
    geometry_types_before = boundaries.geometry.geom_type.value_counts().to_dict()

    boundaries = boundaries.copy()
    boundaries["geometry"] = boundaries.geometry.apply(safe_make_valid)
    boundaries = boundaries[
        boundaries.geometry.notna() & ~boundaries.geometry.is_empty
    ].copy()

    geographic_crs = projection_config["geographic_crs"]
    analysis_crs = CRS.from_user_input(projection_config["analysis_crs"])
    boundaries_geographic = boundaries.to_crs(geographic_crs)
    selected = select_study_units(boundaries_geographic, boundary_config)
    selected_projected = selected.to_crs(analysis_crs)

    core_geometry = make_valid(selected_projected.geometry.union_all())
    if core_geometry.is_empty:
        raise ValueError("The dissolved Yaoundé core geometry is empty.")

    context_buffer_m = float(geometry_config["context_buffer_m"])
    if context_buffer_m <= 0:
        raise ValueError("The context buffer distance must be strictly positive.")

    context_geometry = make_valid(core_geometry.buffer(context_buffer_m))
    convex_hull_geometry = make_valid(core_geometry.convex_hull)

    core_output = resolve_project_path(project_root, output_config["core_boundary"])
    context_output = resolve_project_path(project_root, output_config["context_boundary"])
    hull_output = resolve_project_path(project_root, output_config["convex_hull"])

    write_single_geometry(core_geometry, analysis_crs, core_output, "yaounde_core")
    write_single_geometry(
        context_geometry,
        analysis_crs,
        context_output,
        "yaounde_context",
    )
    write_single_geometry(
        convex_hull_geometry,
        analysis_crs,
        hull_output,
        "yaounde_convex_hull",
    )

    comparison_metrics: dict[str, Any] | None = None
    if boundary_config.get("comparison_file") and config.get("comparison_boundary"):
        comparison_path = resolve_project_path(
            project_root,
            boundary_config["comparison_file"],
        )
        comparison_boundaries = gpd.read_file(comparison_path)
        validate_source_boundary(comparison_boundaries, comparison_path)
        selected_comparison = select_comparison_boundary(
            comparison_boundaries,
            config["comparison_boundary"],
        ).to_crs(analysis_crs)
        comparison_geometry = make_valid(selected_comparison.geometry.union_all())
        comparison_metrics = compare_adm3_union_with_adm2(
            core_geometry,
            comparison_geometry,
        )

        reference_output_value = output_config.get("adm2_reference_boundary")
        if reference_output_value:
            reference_output = resolve_project_path(project_root, reference_output_value)
            write_geodataframe(
                selected_comparison,
                reference_output,
                "mfoundi_adm2_reference",
            )

        comparison_report_value = output_config.get("boundary_comparison_metrics")
        if comparison_report_value:
            comparison_report_path = resolve_project_path(
                project_root,
                comparison_report_value,
            )
            write_json(comparison_metrics, comparison_report_path)

    resolution = float(grid_config["resolution_m"])
    anchor_x = float(grid_config["anchor_x_m"])
    anchor_y = float(grid_config["anchor_y_m"])
    nodata = int(grid_config.get("nodata", 0))

    if resolution <= 0:
        raise ValueError("Grid resolution must be strictly positive.")

    xmin, ymin, xmax, ymax = context_geometry.bounds
    snapped_xmin = snap_down(xmin, resolution, anchor_x)
    snapped_ymin = snap_down(ymin, resolution, anchor_y)
    snapped_xmax = snap_up(xmax, resolution, anchor_x)
    snapped_ymax = snap_up(ymax, resolution, anchor_y)

    width = int(round((snapped_xmax - snapped_xmin) / resolution))
    height = int(round((snapped_ymax - snapped_ymin) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster dimensions: width={width}, height={height}.")

    transform = from_origin(snapped_xmin, snapped_ymax, resolution, resolution)
    output_shape = (height, width)

    context_mask = rasterize(
        [(context_geometry, 1)],
        out_shape=output_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    core_mask = rasterize(
        [(core_geometry, 1)],
        out_shape=output_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )

    if not np.all(core_mask <= context_mask):
        raise ValueError("The rasterised core mask is not fully contained in the context mask.")

    context_mask_output = resolve_project_path(project_root, output_config["context_mask"])
    core_mask_output = resolve_project_path(project_root, output_config["core_mask"])
    grid_template_output = resolve_project_path(project_root, output_config["grid_template"])

    write_mask(context_mask_output, context_mask, transform, analysis_crs, nodata)
    write_mask(core_mask_output, core_mask, transform, analysis_crs, nodata)
    write_mask(grid_template_output, context_mask, transform, analysis_crs, nodata)

    cell_table = build_cell_table(
        context_mask,
        core_mask,
        transform,
        analysis_crs,
        grid_config,
    )
    cell_output = resolve_project_path(project_root, output_config["cell_table"])
    cell_output.parent.mkdir(parents=True, exist_ok=True)
    cell_table.to_parquet(cell_output, index=False)

    vector_core_area_m2 = float(core_geometry.area)
    raster_core_area_m2 = float(core_mask.sum() * resolution**2)
    relative_area_difference = (
        abs(raster_core_area_m2 - vector_core_area_m2) / vector_core_area_m2
    )

    grid_id = str(cell_table["grid_id"].iloc[0])
    grid_version = int(cell_table["grid_version"].iloc[0])
    grid_specification = {
        "grid_id": grid_id,
        "grid_version": grid_version,
        "crs": analysis_crs.to_string(),
        "epsg": analysis_crs.to_epsg(),
        "resolution_m": resolution,
        "anchor_x_m": anchor_x,
        "anchor_y_m": anchor_y,
        "extent": {
            "xmin": snapped_xmin,
            "ymin": snapped_ymin,
            "xmax": snapped_xmax,
            "ymax": snapped_ymax,
        },
        "width": width,
        "height": height,
        "transform": list(transform)[:6],
        "inclusion_rule": str(grid_config["inclusion_rule"]),
        "extent_geometry": str(grid_config["extent_geometry"]),
        "context_buffer_m": context_buffer_m,
        "context_cell_count": int(context_mask.sum()),
        "core_cell_count": int(core_mask.sum()),
        "cell_id_pattern": f"EPSG{analysis_crs.to_epsg()}_{resolution:g}m_<x-index>_<y-index>",
    }

    boundary_report: dict[str, Any] = {
        "source_file": str(source_path.relative_to(project_root)),
        "source_sha256": source_hash,
        "input_crs": str(boundaries.crs),
        "analysis_crs": analysis_crs.to_string(),
        "input_feature_count": input_feature_count,
        "selected_feature_count": len(selected),
        "selected_names": selected[str(boundary_config["name_field"])].astype(str).tolist(),
        "selected_pcodes": (
            selected[str(boundary_config["pcode_field"])].astype(str).tolist()
            if boundary_config.get("pcode_field")
            else []
        ),
        "parent_name": boundary_config.get("parent_name"),
        "missing_geometry_count_before_fix": missing_before,
        "empty_geometry_count_before_fix": empty_before,
        "invalid_geometry_count_before_fix": invalid_before,
        "duplicate_geometry_count_before_fix": duplicate_before,
        "geometry_types_before_fix": geometry_types_before,
        "geometry_fix_operation": str(boundary_config["geometry_fix"]),
        "invalid_core_after_fix": bool(not core_geometry.is_valid),
        "core_area_vector_km2": vector_core_area_m2 / 1_000_000,
        "core_area_rasterised_km2": raster_core_area_m2 / 1_000_000,
        "relative_vector_raster_area_difference": relative_area_difference,
        "context_area_km2": float(context_geometry.area) / 1_000_000,
        "convex_hull_area_km2": float(convex_hull_geometry.area) / 1_000_000,
        "core_within_context": bool(context_geometry.covers(core_geometry)),
        "adm2_adm3_comparison": comparison_metrics,
        "grid_specification": grid_specification,
    }

    grid_spec_path = resolve_project_path(project_root, output_config["grid_specification"])
    boundary_report_path = resolve_project_path(project_root, output_config["boundary_report"])
    write_json(grid_specification, grid_spec_path)
    write_json(boundary_report, boundary_report_path)

    return boundary_report


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the study-grid construction command.

    Returns:
        Namespace containing the required ``--config`` path.
    """
    parser = argparse.ArgumentParser(
        description="Build the Yaoundé study geometries and deterministic 30 m grid."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to configs/study_area.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the pipeline from the command line and print the final JSON report."""
    arguments = parse_arguments()
    report = build_study_grid(arguments.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
