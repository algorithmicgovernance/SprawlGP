"""Quality-assurance tests for the Yaoundé boundaries and deterministic grid.

These tests combine unit-style checks of grid invariants with integration checks
that read the generated GeoPackage, GeoTIFF, Parquet, JSON, and source boundary
files. Run ``make build-grid`` before executing this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from pyproj import CRS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/study_area.yaml"


def project_path(path_value: str | Path) -> Path:
    """Resolve a test input or output path relative to the repository root.

    Args:
        path_value: Absolute or repository-relative path.

    Returns:
        Absolute normalised path.
    """
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 checksum in memory-efficient chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the authoritative study-area configuration once per test session."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    if not isinstance(loaded_config, dict):
        raise ValueError("The study-area configuration must contain a YAML mapping.")

    return loaded_config


@pytest.fixture(scope="session")
def boundary_report(config: dict[str, Any]) -> dict[str, Any]:
    """Load the generated boundary-processing report once per test session."""
    report_path = project_path(config["outputs"]["boundary_report"])
    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def grid_specification(config: dict[str, Any]) -> dict[str, Any]:
    """Load the generated authoritative grid specification."""
    specification_path = project_path(config["outputs"]["grid_specification"])
    with specification_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def core(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Load the dissolved Yaoundé reporting core."""
    return gpd.read_file(project_path(config["outputs"]["core_boundary"]))


@pytest.fixture(scope="session")
def context(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Load the buffered context geometry used for feature construction."""
    return gpd.read_file(project_path(config["outputs"]["context_boundary"]))


@pytest.fixture(scope="session")
def hull(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Load the convex-hull sensitivity geometry."""
    return gpd.read_file(project_path(config["outputs"]["convex_hull"]))


@pytest.fixture(scope="session")
def cell_table(config: dict[str, Any]) -> pd.DataFrame:
    """Load the model-ready table describing every context-grid cell."""
    return pd.read_parquet(project_path(config["outputs"]["cell_table"]))


# Source and provenance: required inputs must exist and remain traceable.


def test_source_and_metadata_files_exist(config: dict[str, Any]) -> None:
    """Verify that source, comparison, metadata, and checksum files exist."""
    boundary = config["boundary"]

    assert project_path(boundary["source_file"]).is_file()
    assert project_path(boundary["comparison_file"]).is_file()
    assert project_path(boundary["metadata_file"]).is_file()
    assert project_path(boundary["checksum_file"]).is_file()


def test_source_checksum_matches_generated_report(
    config: dict[str, Any],
    boundary_report: dict[str, Any],
) -> None:
    """Verify that the reported source checksum matches the current ADM3 file."""
    source_path = project_path(config["boundary"]["source_file"])
    assert sha256_file(source_path) == boundary_report["source_sha256"]


# Administrative structure: Mfoundi and Yaoundé I–VII must be unambiguous.


def test_configured_administrative_fields_exist(config: dict[str, Any]) -> None:
    """Verify that all configured name, parent, and P-code fields exist."""
    boundary = config["boundary"]
    comparison = config["comparison_boundary"]
    adm3 = gpd.read_file(project_path(boundary["source_file"]))
    adm2 = gpd.read_file(project_path(boundary["comparison_file"]))

    for field in [boundary["name_field"], boundary["parent_field"], boundary["pcode_field"]]:
        assert field in adm3.columns

    for field in [comparison["name_field"], comparison["pcode_field"]]:
        assert field in adm2.columns


def test_exactly_seven_yaounde_units_are_selected(config: dict[str, Any]) -> None:
    """Verify the seven configured Yaoundé ADM3 names, P-codes, and parent."""
    boundary = config["boundary"]
    adm3 = gpd.read_file(project_path(boundary["source_file"]))
    selected = adm3[
        adm3[boundary["pcode_field"]].astype(str).isin(boundary["selected_pcodes"])
    ].copy()

    assert len(selected) == int(boundary["expected_selected_units"])
    assert set(selected[boundary["name_field"]].astype(str)) == set(
        boundary["selected_names"]
    )
    assert set(selected[boundary["pcode_field"]].astype(str)) == set(
        boundary["selected_pcodes"]
    )
    assert set(selected[boundary["parent_field"]].astype(str)) == {
        str(boundary["parent_name"])
    }


def test_mfoundi_exists_once_in_adm2(config: dict[str, Any]) -> None:
    """Verify that the configured ADM2 P-code identifies exactly one Mfoundi."""
    boundary = config["boundary"]
    comparison = config["comparison_boundary"]
    adm2 = gpd.read_file(project_path(boundary["comparison_file"]))
    selected = adm2[
        adm2[comparison["pcode_field"]].astype(str)
        == str(comparison["selected_pcode"])
    ]

    assert len(selected) == 1
    assert str(selected.iloc[0][comparison["name_field"]]) == str(
        comparison["selected_name"]
    )


# Source geometry quality: missing, empty, invalid, and duplicate data are forbidden.


def test_source_geometry_quality(config: dict[str, Any]) -> None:
    """Verify source geometry completeness, validity, uniqueness, and polygon type."""
    adm3 = gpd.read_file(project_path(config["boundary"]["source_file"]))

    assert int(adm3.geometry.isna().sum()) == 0
    assert int(adm3.geometry.is_empty.sum()) == 0
    assert int((~adm3.geometry.is_valid).sum()) == 0
    assert int(adm3.geometry.duplicated().sum()) == 0
    assert set(adm3.geometry.geom_type.unique()) <= {"Polygon", "MultiPolygon"}


def test_geometry_repair_is_documented(
    config: dict[str, Any],
    boundary_report: dict[str, Any],
) -> None:
    """Verify that the configured geometry-repair policy appears in the report."""
    expected_operation = str(config["boundary"]["geometry_fix"])
    assert boundary_report["geometry_fix_operation"] == expected_operation
    assert "invalid_geometry_count_before_fix" in boundary_report
    assert "invalid_core_after_fix" in boundary_report


# ADM2/ADM3 consistency: the seven-unit union should closely reproduce Mfoundi.


def test_adm3_union_matches_adm2_mfoundi(config: dict[str, Any]) -> None:
    """Verify that ADM3 union and ADM2 Mfoundi differ by less than the tolerance."""
    boundary = config["boundary"]
    comparison = config["comparison_boundary"]
    analysis_crs = config["projection"]["analysis_crs"]

    adm3 = gpd.read_file(project_path(boundary["source_file"])).to_crs(analysis_crs)
    selected_adm3 = adm3[
        adm3[boundary["pcode_field"]].astype(str).isin(boundary["selected_pcodes"])
    ]

    adm2 = gpd.read_file(project_path(boundary["comparison_file"])).to_crs(analysis_crs)
    selected_adm2 = adm2[
        adm2[comparison["pcode_field"]].astype(str)
        == str(comparison["selected_pcode"])
    ]

    adm3_union = selected_adm3.geometry.union_all()
    adm2_geometry = selected_adm2.geometry.union_all()
    difference_ratio = (
        adm3_union.symmetric_difference(adm2_geometry).area / adm2_geometry.area
    )
    maximum_ratio = float(
        config.get("quality_thresholds", {}).get(
            "adm2_adm3_max_symmetric_difference_ratio",
            0.02,
        )
    )

    assert difference_ratio < maximum_ratio


# Projection: raw inputs stay geographic and processed outputs use metric UTM.


def test_source_is_preserved_in_geographic_crs(config: dict[str, Any]) -> None:
    """Verify that the raw ADM3 source retains its original EPSG:4326 CRS."""
    source = gpd.read_file(project_path(config["boundary"]["source_file"]))
    expected_epsg = CRS.from_user_input(
        config["projection"]["geographic_crs"]
    ).to_epsg()

    assert source.crs is not None
    assert source.crs.to_epsg() == expected_epsg == 4326


def test_processed_geometries_use_analysis_crs(
    config: dict[str, Any],
    core: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    hull: gpd.GeoDataFrame,
) -> None:
    """Verify that every processed geometry uses projected metre-based UTM 32N."""
    expected_crs = CRS.from_user_input(config["projection"]["analysis_crs"])

    assert expected_crs.to_epsg() == 32632
    assert expected_crs.is_projected
    assert expected_crs.axis_info[0].unit_name.lower() == "metre"
    assert core.crs.to_epsg() == expected_crs.to_epsg()
    assert context.crs.to_epsg() == expected_crs.to_epsg()
    assert hull.crs.to_epsg() == expected_crs.to_epsg()


# Constructed geometries: outputs must exist, be valid, and preserve nesting.


def test_constructed_geometry_files_exist(config: dict[str, Any]) -> None:
    """Verify that the core, context, and convex-hull GeoPackages were created."""
    outputs = config["outputs"]
    assert project_path(outputs["core_boundary"]).is_file()
    assert project_path(outputs["context_boundary"]).is_file()
    assert project_path(outputs["convex_hull"]).is_file()


def test_constructed_geometries_are_valid(
    core: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    hull: gpd.GeoDataFrame,
) -> None:
    """Verify that all constructed geometries are non-empty and spatially valid."""
    for geodataframe in [core, context, hull]:
        assert geodataframe.geometry.is_valid.all()
        assert not geodataframe.geometry.is_empty.any()


def test_context_completely_covers_core(
    core: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
) -> None:
    """Verify that the context buffer fully covers the reporting core."""
    assert context.geometry.iloc[0].covers(core.geometry.iloc[0])


# Raster definition: all grid products must share one 30 m snapped transform.


def test_grid_rasters_share_the_same_definition(config: dict[str, Any]) -> None:
    """Verify that template, core mask, and context mask share one exact grid."""
    outputs = config["outputs"]
    raster_paths = [
        project_path(outputs["grid_template"]),
        project_path(outputs["core_mask"]),
        project_path(outputs["context_mask"]),
    ]
    specifications = []

    for raster_path in raster_paths:
        assert raster_path.is_file()
        with rasterio.open(raster_path) as raster:
            specifications.append(
                (
                    raster.crs,
                    raster.transform,
                    raster.width,
                    raster.height,
                    raster.res,
                )
            )

    assert all(specification == specifications[0] for specification in specifications)


def test_grid_resolution_anchor_and_extent(config: dict[str, Any]) -> None:
    """Verify 30 m resolution and snapping to the configured global anchor."""
    grid = config["grid"]
    resolution = float(grid["resolution_m"])
    anchor_x = float(grid["anchor_x_m"])
    anchor_y = float(grid["anchor_y_m"])
    raster_path = project_path(config["outputs"]["grid_template"])

    with rasterio.open(raster_path) as raster:
        assert raster.res == (resolution, resolution)
        assert np.isclose((raster.transform.c - anchor_x) % resolution, 0)
        assert np.isclose((raster.transform.f - anchor_y) % resolution, 0)
        assert np.isclose((raster.bounds.left - anchor_x) % resolution, 0)
        assert np.isclose((raster.bounds.right - anchor_x) % resolution, 0)
        assert np.isclose((raster.bounds.bottom - anchor_y) % resolution, 0)
        assert np.isclose((raster.bounds.top - anchor_y) % resolution, 0)


def test_core_mask_is_contained_in_context_mask(config: dict[str, Any]) -> None:
    """Verify binary mask values and ensure every core cell is a context cell."""
    with rasterio.open(project_path(config["outputs"]["core_mask"])) as raster:
        core_mask = raster.read(1)
    with rasterio.open(project_path(config["outputs"]["context_mask"])) as raster:
        context_mask = raster.read(1)

    assert set(np.unique(core_mask)) <= {0, 1}
    assert set(np.unique(context_mask)) <= {0, 1}
    assert np.all(core_mask <= context_mask)


def test_pixel_center_rule_is_documented(config: dict[str, Any]) -> None:
    """Verify that the raster inclusion rule is explicitly fixed to pixel centre."""
    assert config["grid"]["inclusion_rule"] == "pixel_center"


# Cell identity: global indices must reproduce coordinates and stable identifiers.


def test_cell_identifiers_are_unique(cell_table: pd.DataFrame) -> None:
    """Verify that every retained context cell has one unique stable identifier."""
    assert cell_table["cell_id"].is_unique


def test_cell_ids_are_derived_from_global_grid_indices(
    config: dict[str, Any],
    cell_table: pd.DataFrame,
) -> None:
    """Reconstruct cell centres and IDs from global anchored grid indices."""
    resolution = float(config["grid"]["resolution_m"])
    anchor_x = float(config["grid"]["anchor_x_m"])
    anchor_y = float(config["grid"]["anchor_y_m"])
    epsg = CRS.from_user_input(config["projection"]["analysis_crs"]).to_epsg()
    resolution_label = int(resolution) if resolution.is_integer() else resolution

    expected_x = anchor_x + (cell_table["grid_x_index"] + 0.5) * resolution
    expected_y = anchor_y + (cell_table["grid_y_index"] + 0.5) * resolution
    expected_ids = (
        f"EPSG{epsg}_{resolution_label}m_"
        + cell_table["grid_x_index"].astype(str)
        + "_"
        + cell_table["grid_y_index"].astype(str)
    )

    assert np.allclose(cell_table["x_center_m"], expected_x)
    assert np.allclose(cell_table["y_center_m"], expected_y)
    assert (cell_table["cell_id"].astype(str) == expected_ids).all()


def test_cell_table_matches_raster_masks(
    config: dict[str, Any],
    cell_table: pd.DataFrame,
) -> None:
    """Verify table row counts, core flags, and 900 m² area against masks."""
    with rasterio.open(project_path(config["outputs"]["core_mask"])) as raster:
        core_mask = raster.read(1)
    with rasterio.open(project_path(config["outputs"]["context_mask"])) as raster:
        context_mask = raster.read(1)

    resolution = float(config["grid"]["resolution_m"])
    assert len(cell_table) == int(context_mask.sum())
    assert int(cell_table["in_core"].sum()) == int(core_mask.sum())
    assert (cell_table["cell_area_m2"] == resolution**2).all()


# Frozen reference: once accepted, grid v1 must not change unintentionally.


# def test_grid_matches_frozen_reference(
#     config: dict[str, Any],
#     grid_specification: dict[str, Any],
# ) -> None:
#     """Compare the generated grid with the committed reference specification.

#     The test is skipped until ``grid.reference_specification`` is configured and
#     the accepted JSON reference has been committed. Once present, any unintended
#     change in CRS, resolution, anchor, extent, transform, dimensions, or cell
#     counts causes the test to fail.
#     """
#     reference_value = config["grid"].get("reference_specification")
#     if not reference_value:
#         pytest.skip("No frozen grid reference is configured yet.")

#     reference_path = project_path(reference_value)
#     if not reference_path.is_file():
#         pytest.skip(f"Frozen grid reference does not exist yet: {reference_path}")

#     with reference_path.open("r", encoding="utf-8") as file:
#         reference = json.load(file)

#     frozen_fields = [
#         "grid_id",
#         "grid_version",
#         "crs",
#         "resolution_m",
#         "anchor_x_m",
#         "anchor_y_m",
#         "extent",
#         "width",
#         "height",
#         "transform",
#         "inclusion_rule",
#         "context_buffer_m",
#         "context_cell_count",
#         "core_cell_count",
#     ]

#     for field in frozen_fields:
#         assert grid_specification[field] == reference[field]


# Area consistency: rasterisation must preserve the vector core area closely.


def test_vector_raster_area_difference_is_small(
    config: dict[str, Any],
    boundary_report: dict[str, Any],
) -> None:
    """Verify that vector and rasterised core areas differ below tolerance."""
    maximum_difference = float(
        config.get("quality_thresholds", {}).get(
            "max_vector_raster_area_difference_ratio",
            0.02,
        )
    )
    observed_difference = float(
        boundary_report["relative_vector_raster_area_difference"]
    )

    assert observed_difference < maximum_difference
