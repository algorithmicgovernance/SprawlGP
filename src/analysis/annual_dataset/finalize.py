"""Validate annual state assets and write provisional release metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

try:
    import ee
except ImportError:  # Enables local unit tests without Earth Engine.
    ee = None

from src.analysis.annual_dataset.build_products import (
    PASS_STATES,
    annual_state_recipe,
    load_annual_sources,
    metadata_directory,
    validate_annual_config,
    validate_existing_state_asset,
)
from src.analysis.built_up.build_candidates import load_ee_geometry
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


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError("The Earth Engine Python API is required for finalization.")


def final_directory(config: dict[str, Any], project_root: Path) -> Path:
    """Create and return the provisional annual release directory."""
    path = resolve_project_path(config["outputs"]["final_directory"], project_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def core_area_ha(path: Path) -> float:
    """Calculate the administrative core area in hectares."""
    frame = gpd.read_file(path)
    if frame.crs is None or frame.empty:
        raise ValueError("The administrative core must be non-empty and georeferenced.")
    area = float(frame.to_crs("EPSG:32632").geometry.area.sum()) / 10_000
    if area <= 0:
        raise ValueError("The administrative core area is zero.")
    return area


def refresh_state_tasks(config_path: Path) -> pd.DataFrame:
    """Refresh annual state export statuses with one Earth Engine task-list call."""
    require_earth_engine()
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    initialize_earth_engine(config["project"]["earth_engine_project"])
    path = metadata_directory(config, project_root) / config["metadata"]["state_task_manifest"]
    if not path.is_file():
        raise FileNotFoundError("Submit annual state products before requesting status.")
    frame = pd.read_csv(path, keep_default_na=False)
    statuses = {task.id: task.status() for task in ee.batch.Task.list()}
    for index, row in frame.iterrows():
        task_id = str(row["task_id"]).strip()
        if task_id in statuses:
            frame.at[index, "state"] = statuses[task_id].get("state", "UNKNOWN")
            frame.at[index, "error_message"] = statuses[task_id].get("error_message", "")
    frame.to_csv(path, index=False)
    print(json.dumps({"task_states": frame["state"].value_counts().to_dict()}, indent=2))
    return frame


def state_statistics(
    asset_id: str,
    thresholds: dict[str, float],
    core_geometry: Any,
    core_area: float,
    grid: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate raw, final, valid, and persistence statistics for one year."""
    require_earth_engine()
    state = ee.Image(asset_id)
    valid = state.select("built_state_valid").eq(1)
    raw = state.select("built_state_raw").eq(1)
    final = state.select("built_state_final").eq(1)
    corrected = state.select("persistence_corrected").eq(1)
    one = ee.Image.constant(1)
    area = ee.Image.pixelArea()
    statistic_bands = [
            one.updateMask(valid).rename("valid_cells"),
            area.updateMask(valid).rename("valid_area_m2"),
            one.updateMask(raw).rename("raw_built_cells"),
            area.updateMask(raw).rename("raw_built_area_m2"),
            one.updateMask(final).rename("final_built_cells"),
            area.updateMask(final).rename("final_built_area_m2"),
            one.updateMask(corrected).rename("persistence_corrected_cells"),
            area.updateMask(corrected).rename("persistence_corrected_area_m2"),
    ]
    for index_name in ("ibui", "ndbsui"):
        if index_name not in thresholds:
            continue
        index = state.select(index_name)
        index_valid = index.mask().reduce(ee.Reducer.min()).eq(1)
        diagnostic_built = index.gt(float(thresholds[index_name]))
        statistic_bands.extend(
            [
                area.updateMask(index_valid).rename(f"{index_name}_valid_area_m2"),
                area.updateMask(diagnostic_built).rename(
                    f"{index_name}_raw_built_area_m2"
                ),
            ]
        )
    bands = ee.Image.cat(statistic_bands)
    values = bands.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=core_geometry,
        crs=grid["crs"],
        crsTransform=[float(value) for value in grid["transform"]],
        maxPixels=int(config["exports"]["max_pixels"]),
        tileScale=int(config["histogram"]["tile_scale"]),
    ).getInfo()
    valid_area = float(values.get("valid_area_m2", 0) or 0) / 10_000
    raw_area = float(values.get("raw_built_area_m2", 0) or 0) / 10_000
    final_area = float(values.get("final_built_area_m2", 0) or 0) / 10_000
    result = {
        "valid_area_core_pct": valid_area / core_area * 100,
        "raw_built_cells_core": int(round(float(values.get("raw_built_cells", 0) or 0))),
        "raw_built_area_ha_core": raw_area,
        "raw_built_fraction_core": raw_area / valid_area if valid_area else None,
        "final_built_cells_core": int(
            round(float(values.get("final_built_cells", 0) or 0))
        ),
        "final_built_area_ha_core": final_area,
        "final_built_fraction_core": final_area / valid_area if valid_area else None,
        "persistence_corrected_cells": int(
            round(float(values.get("persistence_corrected_cells", 0) or 0))
        ),
        "persistence_corrected_area_ha": float(
            values.get("persistence_corrected_area_m2", 0) or 0
        )
        / 10_000,
    }
    for index_name in ("ibui", "ndbsui"):
        valid_m2 = float(values.get(f"{index_name}_valid_area_m2", 0) or 0)
        built_m2 = float(values.get(f"{index_name}_raw_built_area_m2", 0) or 0)
        if index_name in thresholds:
            result[f"{index_name}_raw_built_fraction_core"] = (
                built_m2 / valid_m2 if valid_m2 else None
            )
    return result


def data_dictionary() -> dict[str, Any]:
    """Return concise definitions for the annual cell-time table."""
    return {
        "release_status": "PROVISIONAL_PENDING_ANNUAL_MANUAL_VALIDATION",
        "cell_time_dataset": {
            "cell_id": "Stable row-major identifier on the frozen 30 m grid.",
            "row": "Zero-based row on the frozen grid.",
            "column": "Zero-based column on the frozen grid.",
            "x_center_m": "Cell-centre easting in EPSG:32632 metres.",
            "y_center_m": "Cell-centre northing in EPSG:32632 metres.",
            "longitude": "Cell-centre longitude in decimal degrees.",
            "latitude": "Cell-centre latitude in decimal degrees.",
            "transition_order": "Chronological zero-based transition index.",
            "forecast_origin": "Calendar year in which all predictors are observed.",
            "target_year": "The next calendar year; always forecast_origin + 1.",
            "target_transition_1y": "One for a pairwise-valid non-built to built conversion.",
            "common_valid_1y": "One when both transition endpoint states are valid.",
            "built_state_raw_t": "Origin-year NDBI/Otsu state before persistence.",
            "built_state_final_t": "Origin-year state after absorbing persistence.",
            "built_state_valid_t": "One when the origin-year NDBI state is valid.",
            "valid_observation_count_t": "Valid Landsat observations at the origin.",
            "blue_t": "Blue surface reflectance at the origin.",
            "green_t": "Green surface reflectance at the origin.",
            "red_t": "Red surface reflectance at the origin.",
            "nir_t": "Near-infrared surface reflectance at the origin.",
            "swir1_t": "First short-wave infrared reflectance at the origin.",
            "swir2_t": "Second short-wave infrared reflectance at the origin.",
            "ndbi_t": "Normalized difference built-up index at the origin.",
            "ibui_t": "Index-based built-up index diagnostic at the origin.",
            "ndbsui_t": "Normalized difference built-up and soil index at the origin.",
            "savi_t": "Soil-adjusted vegetation index at the origin.",
            "mndwi_t": "Modified normalized difference water index at the origin.",
            "built_fraction_3x3_t": "Final built fraction among valid 3x3 neighbours.",
            "built_fraction_5x5_t": "Final built fraction among valid 5x5 neighbours.",
            "built_fraction_11x11_t": "Final built fraction among valid 11x11 neighbours.",
            "distance_to_built_m_t": "Distance to final built land at the origin.",
            "recent_local_growth_1y_t": "Local transition fraction from t-1 to t.",
            "recent_local_growth_available_t": "Zero only when no prior annual state exists.",
            "elevation_m": "Existing SRTM elevation predictor.",
            "slope_degrees": "Existing SRTM-derived slope predictor.",
            "population_density_t": "GHSL density from the latest available past epoch.",
            "population_source_year": "Configured GHSL epoch, never later than the origin.",
        },
    }


def write_dataset_card(
    path: Path, limited_years: list[int]
) -> None:
    """Write the provisional annual dataset card and validation limitations."""
    lines = [
        "# Annual Yaounde urban-expansion dataset",
        "",
        "- release_status: PROVISIONAL_PENDING_ANNUAL_MANUAL_VALIDATION",
        "- mapping_method: NDBI",
        "- mapping_validation_status: PROVISIONAL_ANNUAL_PROTOCOL",
        "- transition_horizon_years: 1",
        "- years: 2000-2025",
        "- forecast_origins: 2000-2024",
        "- no_future_imagery: true",
        "- population_policy: latest_available_epoch_at_or_before_origin",
        "",
        "## Observation support",
        "",
        f"Day 2 classified {len(limited_years)} years as LIMITED: "
        f"{', '.join(map(str, limited_years))}.",
        "All other annual source years are PASS; none is FAIL.",
        "",
        "## Scientific status",
        "",
        "NDBI controls all raw states, persistent states, and one-year targets. ",
        "IBUI and NDBSUI are retained only as continuous and threshold diagnostics. ",
        "The calendar-year annual mapping protocol still requires dedicated manual validation; ",
        "old five-year labels do not validate these 26 annual maps.",
        "",
        "## Temporal safeguards",
        "",
        "Predictors use only information available by the forecast origin. Recent local growth ",
        "uses the transition ending at the origin, and GHSL population uses the latest configured ",
        "epoch at or before the origin without interpolation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_products(config_path: Path) -> dict[str, Any]:
    """Validate completed annual states and write state/release metadata."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    years = validate_annual_config(config)
    grid = load_grid_specification(
        resolve_project_path(config["inputs"]["grid_specification"], project_root)
    )
    frame = refresh_state_tasks(config_path)
    incomplete = frame[~frame["state"].astype(str).str.upper().isin(PASS_STATES)]
    if not incomplete.empty:
        raise RuntimeError("Annual state products are not complete.")
    if frame["year"].astype(int).tolist() != years or len(frame) != 26:
        raise ValueError("Annual task manifest must contain 26 ordered state products.")
    metadata_dir = metadata_directory(config, project_root)
    thresholds = pd.read_csv(
        metadata_dir / config["metadata"]["threshold_table"],
        float_precision="round_trip",
    )
    threshold_wide = thresholds.pivot(index="year", columns="index_name", values="threshold")
    sources = {
        source.year: source
        for source in load_annual_sources(
            resolve_project_path(
                config["inputs"]["landsat_composite_manifest"], project_root
            ),
            years,
        )
    }
    for row in frame.itertuples(index=False):
        year = int(row.year)
        source = sources[year]
        recipe = annual_state_recipe(
            year=year,
            thresholds={
                name: float(threshold_wide.loc[year, name])
                for name in threshold_wide.columns
            },
            source_composite_asset=source.composite_asset,
            source_count_asset=source.count_asset,
            grid=grid,
        )
        reconstructed_recipe_sha256 = stable_object_hash(recipe)
        task_recipe_sha256 = str(row.recipe_sha256)
        if reconstructed_recipe_sha256 != task_recipe_sha256:
            raise ValueError(
                "Annual state recipe reconstruction mismatch for "
                f"{year}: task manifest {task_recipe_sha256!r}, "
                f"reconstructed {reconstructed_recipe_sha256!r}."
            )
        validate_existing_state_asset(
            str(row.asset_id),
            expected_recipe_sha256=task_recipe_sha256,
            expected_year=year,
            expected_composite_asset=source.composite_asset,
            expected_count_asset=source.count_asset,
            grid=grid,
        )
    output_manifest = frame[
        ["product_type", "year", "asset_id", "band_names", "state", "recipe_sha256"]
    ].copy()
    output_manifest["state"] = "PASS"
    manifest_path = metadata_dir / config["metadata"]["state_output_manifest"]
    output_manifest.to_csv(manifest_path, index=False)

    require_earth_engine()
    core_path = resolve_project_path(config["inputs"]["core_boundary"], project_root)
    core_geometry = load_ee_geometry(core_path)
    area_ha = core_area_ha(core_path)
    quality = pd.read_csv(
        resolve_project_path(config["inputs"]["annual_quality_summary"], project_root)
    )
    rows: list[dict[str, Any]] = []
    for row in output_manifest.itertuples(index=False):
        year = int(row.year)
        quality_row = quality[quality["year"].astype(int) == year].iloc[0]
        year_thresholds = {
            name: float(threshold_wide.loc[year, name])
            for name in threshold_wide.columns
        }
        statistics = state_statistics(
            str(row.asset_id), year_thresholds, core_geometry, area_ha, grid, config
        )
        summary = {
            "year": year,
            "day2_quality_flag": quality_row["quality_flag"],
            "selected_sensors": quality_row["selected_sensors"],
            "selected_scene_count": int(quality_row["selected_scene_count"]),
            "coverage_at_least_1_core_pct": float(
                quality_row["coverage_at_least_1_core_pct"]
            ),
            "coverage_at_least_3_core_pct": float(
                quality_row["coverage_at_least_3_core_pct"]
            ),
            "median_observation_count": float(quality_row["median_observation_count"]),
            "ndbi_otsu_threshold": float(threshold_wide.loc[year, "ndbi"]),
            **statistics,
        }
        for name in ("ibui", "ndbsui"):
            if name in threshold_wide.columns:
                summary[f"{name}_otsu_threshold"] = float(threshold_wide.loc[year, name])
        rows.append(summary)
    state_summary = pd.DataFrame(rows)
    state_summary_path = metadata_dir / config["metadata"]["state_summary"]
    state_summary.to_csv(state_summary_path, index=False)
    release_dir = final_directory(config, project_root)
    state_summary.to_csv(release_dir / config["outputs"]["annual_state_summary"], index=False)
    write_yaml(release_dir / config["outputs"]["data_dictionary"], data_dictionary())
    limited_years = quality.loc[
        quality["quality_flag"].astype(str).eq("LIMITED"), "year"
    ].astype(int).tolist()
    write_dataset_card(release_dir / config["outputs"]["dataset_card"], limited_years)
    version = {
        "pipeline_stage": "annual_dataset_products",
        "pipeline_version": int(config["version"]),
        "release_status": config["release"]["release_status"],
        "mapping_method": "ndbi",
        "mapping_validation_status": "PROVISIONAL_ANNUAL_PROTOCOL",
        "state_count": 26,
        "transition_count": 25,
        "transition_asset_count": 0,
        "state_manifest_sha256": sha256_file(manifest_path),
        "state_summary_sha256": sha256_file(state_summary_path),
        "annual_orchestration_sha256": sha256_file(
            resolve_project_path(
                config["inputs"]["annual_orchestration_version"], project_root
            )
        ),
    }
    version["stable_signature"] = stable_object_hash(version)
    version_path = metadata_dir / config["metadata"]["products_version"]
    write_json(version_path, version)
    result = {"status": "PASS", "state_count": 26, "state_summary": str(state_summary_path)}
    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the annual product-finalization command line."""
    parser = argparse.ArgumentParser(description="Finalize annual state products.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Refresh status or finalize annual state products."""
    arguments = parse_arguments()
    if arguments.status_only:
        refresh_state_tasks(arguments.config.resolve())
    else:
        finalize_products(arguments.config.resolve())


if __name__ == "__main__":
    main()
