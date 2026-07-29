"""Build and freeze the Day 2 Landsat scene catalogue without composites."""

from __future__ import annotations

import argparse
import calendar
import json
import random
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import ee
import pandas as pd
import yaml

from .common import (
    find_project_root,
    initialize_earth_engine,
    load_ee_geometry,
    load_grid_specification,
    load_json,
    load_yaml,
    query_rectangle,
    resolve_project_path,
    sha256_file,
)
from .qa_masks import landsat_valid_mask


@dataclass(frozen=True)
class Window:
    """Describe one consecutive calendar window."""

    name: str
    months: int
    start_month: int


def sensor_policy(config: dict[str, Any], epoch: int) -> dict[str, list[str]]:
    """Return normalized primary, supplemental and diagnostic sensor lists."""
    raw = config["epoch_sensor_policy"].get(epoch)
    if raw is None:
        raw = config["epoch_sensor_policy"][str(epoch)]
    return {key: list(raw.get(key, [])) for key in (
        "primary", "supplemental", "diagnostic_only"
    )}


def diagnostic_range(epoch: int, config: dict[str, Any]) -> tuple[str, str]:
    """Return the three-year half-open diagnostic date range for one epoch."""
    before = int(config["catalog"]["diagnostic_years_before"])
    after = int(config["catalog"]["diagnostic_years_after"])
    return f"{epoch-before}-01-01", f"{epoch+after+1}-01-01"


def safe_number(result: ee.Dictionary, key: str) -> ee.Number:
    """Read a reducer result while replacing an absent key with zero."""
    return ee.Number(ee.Algorithms.If(result.contains(key), result.get(key), 0))


def valid_area(
    mask: ee.Image,
    geometry: ee.Geometry,
    grid: dict[str, Any],
    statistics: dict[str, Any],
) -> ee.Number:
    """Calculate valid area on the exact frozen Day 1 affine grid."""
    result = ee.Image.pixelArea().updateMask(mask).rename("area").reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        crs=grid["crs"],
        crsTransform=grid["transform"],
        maxPixels=int(statistics["max_pixels"]),
        tileScale=int(statistics["tile_scale"]),
    )
    return safe_number(ee.Dictionary(result), "area")


def scene_feature(
    image: ee.Image,
    epoch: int,
    sensor: str,
    role: str,
    config: dict[str, Any],
    grid: dict[str, Any],
    core: ee.Geometry,
    context: ee.Geometry,
    core_area_m2: float,
    context_area_m2: float,
) -> ee.Feature:
    """Convert one scene into a complete manifest row."""
    image = ee.Image(image)
    mask = landsat_valid_mask(image, sensor, config["collections"], config["quality_mask"])
    core_valid = valid_area(mask, core, grid, config["statistics"])
    context_valid = valid_area(mask, context, grid, config["statistics"])
    acquired = ee.Date(image.get("system:time_start"))
    collection_id = config["collections"][sensor]["id"]
    system_index = ee.String(image.get("system:index"))
    asset_id = ee.String(collection_id).cat("/").cat(system_index)
    slc_off = sensor == "LE07" and True

    return ee.Feature(None, {
        "epoch": epoch,
        "relative_year": acquired.get("year").subtract(epoch),
        "sensor_key": sensor,
        "sensor_role": role,
        "collection_id": collection_id,
        "earth_engine_asset_id": asset_id,
        "system_index": system_index,
        "landsat_product_id": image.get("LANDSAT_PRODUCT_ID"),
        "landsat_scene_id": image.get("LANDSAT_SCENE_ID"),
        "acquisition_date": acquired.format("YYYY-MM-dd"),
        "acquisition_year": acquired.get("year"),
        "acquisition_month": acquired.get("month"),
        "day_of_year": acquired.getRelative("day", "year").add(1),
        "wrs_path": image.get("WRS_PATH"),
        "wrs_row": image.get("WRS_ROW"),
        "cloud_cover_scene_pct": image.get("CLOUD_COVER"),
        "cloud_cover_land_pct": image.get("CLOUD_COVER_LAND"),
        "valid_area_core_m2": core_valid,
        "valid_area_context_m2": context_valid,
        "valid_fraction_core": core_valid.divide(core_area_m2),
        "valid_fraction_context": context_valid.divide(context_area_m2),
        "slc_off": ee.Algorithms.If(
            slc_off, acquired.millis().gt(ee.Date("2003-05-31").millis()), False
        ),
        "landsat7_orbit_drift_period": ee.Algorithms.If(
            sensor == "LE07",
            acquired.millis().gt(ee.Date("2017-02-07").millis()),
            False,
        ),
        "has_positive_core_coverage": core_valid.gt(0),
        "candidate_for_composite": core_valid.gt(0).And(
            ee.String(role).compareTo("diagnostic_only").neq(0)
        ),
    })


def is_retryable_earth_engine_error(error: Exception) -> bool:
    """Return whether an Earth Engine error is likely to succeed on retry.

    Retries are reserved for temporary quota, service and transport failures.
    Invalid arguments and deterministic computation errors are raised
    immediately so genuine bugs are not hidden.
    """
    message = str(error).casefold()
    retryable_fragments = (
        "too many concurrent aggregations",
        "quota exceeded",
        "http 429",
        "rate limit",
        "internal error",
        "service unavailable",
        "deadline exceeded",
        "timed out",
    )
    return any(fragment in message for fragment in retryable_fragments)


def get_info_with_retry(
    computed_object: ee.ComputedObject,
    *,
    description: str,
    maximum_attempts: int = 7,
    initial_delay_seconds: float = 3.0,
) -> Any:
    """Evaluate one small Earth Engine request with exponential backoff.

    The function must be used only after the computation has been split into
    small requests. Retrying one very large request would reproduce the same
    aggregation fan-out and would not address the root cause.
    """
    for attempt in range(1, maximum_attempts + 1):
        try:
            return computed_object.getInfo()
        except Exception as error:
            if (
                attempt == maximum_attempts
                or not is_retryable_earth_engine_error(error)
            ):
                raise RuntimeError(
                    f"Earth Engine failed while {description}."
                ) from error

            delay = (
                initial_delay_seconds * (2 ** (attempt - 1))
                + random.uniform(0.0, 1.0)
            )
            print(
                f"Temporary Earth Engine error while {description}. "
                f"Retrying in {delay:.1f} seconds "
                f"({attempt}/{maximum_attempts})..."
            )
            time.sleep(delay)

    raise AssertionError("Unreachable retry state.")


def normalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize manifest data types and apply deterministic row ordering."""
    if frame.empty:
        raise RuntimeError(
            "The Earth Engine query returned no Landsat scenes."
        )

    result = frame.copy()
    result["acquisition_date"] = pd.to_datetime(
        result["acquisition_date"],
        errors="raise",
    )

    integer_columns = (
        "epoch",
        "relative_year",
        "acquisition_year",
        "acquisition_month",
    )
    for column in integer_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("Int64")

    numeric_columns = (
        "valid_area_core_m2",
        "valid_area_context_m2",
        "valid_fraction_core",
        "valid_fraction_context",
        "cloud_cover_scene_pct",
        "cloud_cover_land_pct",
    )
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    boolean_columns = (
        "slc_off",
        "landsat7_orbit_drift_period",
        "has_positive_core_coverage",
        "candidate_for_composite",
    )
    for column in boolean_columns:
        if column in result.columns:
            result[column] = result[column].fillna(False).astype(bool)

    key = (
        result["epoch"].astype(str)
        + "::"
        + result["earth_engine_asset_id"].astype(str)
    )
    if not key.is_unique:
        raise ValueError(
            "Duplicate epoch/asset records were produced in the manifest."
        )

    return result.sort_values(
        ["epoch", "acquisition_date", "sensor_key", "system_index"]
    ).reset_index(drop=True)


def load_manifest_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    """Load a partial manifest so interrupted catalogue runs can resume."""
    if not checkpoint_path.is_file():
        return pd.DataFrame()

    frame = pd.read_csv(checkpoint_path)
    if "acquisition_date" in frame.columns:
        frame["acquisition_date"] = pd.to_datetime(
            frame["acquisition_date"],
            errors="coerce",
        )

    boolean_columns = (
        "slc_off",
        "landsat7_orbit_drift_period",
        "has_positive_core_coverage",
        "candidate_for_composite",
    )
    for column in boolean_columns:
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.casefold()
                .map({"true": True, "false": False})
                .fillna(False)
            )

    return frame


def write_manifest_checkpoint(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
) -> None:
    """Persist completed scene rows after each small Earth Engine request."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        checkpoint_path,
        index=False,
        date_format="%Y-%m-%d",
    )


def build_manifest(
    config: dict[str, Any],
    grid: dict[str, Any],
    core: ee.Geometry,
    context: ee.Geometry,
    query_geom: ee.Geometry,
    core_area_m2: float,
    context_area_m2: float,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Build the manifest sequentially to avoid aggregation fan-out.

    Earth Engine raises ``Too many concurrent aggregations`` when a mapped
    collection contains a regional reduction for every image and the complete
    graph is evaluated at once. This implementation first retrieves scene
    indexes, then evaluates one scene feature per request. Each request contains
    only the core and context reductions for that single scene.
    """
    checkpoint = load_manifest_checkpoint(checkpoint_path)
    rows = checkpoint.to_dict(orient="records")
    completed = {
        (int(row["epoch"]), str(row["earth_engine_asset_id"]))
        for row in rows
        if pd.notna(row.get("epoch"))
        and pd.notna(row.get("earth_engine_asset_id"))
    }

    for epoch_value in config["catalog"]["epochs"]:
        epoch = int(epoch_value)
        policy = sensor_policy(config, epoch)
        start_date, end_date = diagnostic_range(epoch, config)

        for role in ("primary", "supplemental", "diagnostic_only"):
            for sensor in policy[role]:
                collection_id = config["collections"][sensor]["id"]
                collection = (
                    ee.ImageCollection(collection_id)
                    .filterDate(start_date, end_date)
                    .filterBounds(query_geom)
                    .sort("system:time_start")
                )
                system_indexes = get_info_with_retry(
                    collection.aggregate_array("system:index"),
                    description=(
                        f"listing scenes for epoch {epoch}, sensor {sensor}"
                    ),
                ) or []

                total = len(system_indexes)
                print(
                    f"Epoch {epoch}, {sensor} ({role}): "
                    f"{total} diagnostic scenes."
                )

                for position, system_index in enumerate(
                    system_indexes,
                    start=1,
                ):
                    asset_id = f"{collection_id}/{system_index}"
                    manifest_key = (epoch, asset_id)

                    if manifest_key in completed:
                        continue

                    feature = scene_feature(
                        ee.Image(asset_id),
                        epoch,
                        sensor,
                        role,
                        config,
                        grid,
                        core,
                        context,
                        core_area_m2,
                        context_area_m2,
                    )
                    properties = get_info_with_retry(
                        feature.toDictionary(),
                        description=(
                            f"computing scene {position}/{total} "
                            f"for epoch {epoch}, sensor {sensor}"
                        ),
                    )
                    rows.append(dict(properties))
                    completed.add(manifest_key)
                    write_manifest_checkpoint(rows, checkpoint_path)

                    print(
                        f"  Completed {position}/{total}: {system_index}"
                    )

    manifest = normalize_manifest(pd.DataFrame(rows))
    manifest.to_csv(
        checkpoint_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    return manifest

def monthly_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize scene count and scene-level valid coverage by month and sensor."""
    return (
        manifest.groupby(["epoch", "acquisition_month", "sensor_key"], as_index=False)
        .agg(
            scene_count=("earth_engine_asset_id", "count"),
            positive_core_scene_count=("has_positive_core_coverage", "sum"),
            median_valid_fraction_core=("valid_fraction_core", "median"),
            maximum_valid_fraction_core=("valid_fraction_core", "max"),
            median_valid_fraction_context=("valid_fraction_context", "median"),
            median_cloud_cover_scene_pct=("cloud_cover_scene_pct", "median"),
        )
    )


def candidate_windows(config: dict[str, Any]) -> list[Window]:
    """Generate every configured consecutive calendar window."""
    windows: list[Window] = []
    allow_cross = bool(config["candidate_windows"]["allow_cross_year_windows"])
    for length in config["candidate_windows"]["lengths_months"]:
        for start_month in range(1, 13):
            if not allow_cross and start_month + int(length) - 1 > 12:
                continue
            end_month = ((start_month + int(length) - 2) % 12) + 1
            windows.append(Window(
                name=f"{calendar.month_abbr[start_month]}-{calendar.month_abbr[end_month]}_{length}m",
                months=int(length),
                start_month=start_month,
            ))
    return windows


def exact_window_dates(epoch: int, window: Window) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert a calendar window to exact half-open dates for one epoch."""
    crosses = window.start_month + window.months - 1 > 12
    start_year = epoch - 1 if crosses else epoch
    start = pd.Timestamp(date(start_year, window.start_month, 1))
    end = start + pd.DateOffset(months=window.months)
    return start, end


def evaluate_windows(manifest: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Evaluate candidate windows using a fast manifest-derived approximation.

    This first implementation ranks windows from scene-level QA coverage and
    freezes exact scene IDs. Day 3 can recompute full observation-depth rasters
    for the selected window only, avoiding hundreds of expensive reductions.
    """
    records: list[dict[str, Any]] = []
    for window in candidate_windows(config):
        for epoch in config["catalog"]["epochs"]:
            policy = sensor_policy(config, int(epoch))
            modes = [("primary_only", policy["primary"])]
            if policy["supplemental"]:
                modes.append(("primary_plus_supplemental", policy["primary"] + policy["supplemental"]))
            start, end = exact_window_dates(int(epoch), window)
            for mode, sensors in modes:
                rows = manifest[
                    (manifest["epoch"] == int(epoch))
                    & manifest["sensor_key"].isin(sensors)
                    & (manifest["acquisition_date"] >= start)
                    & (manifest["acquisition_date"] < end)
                    & manifest["has_positive_core_coverage"].astype(bool)
                ]
                combined = 0.0 if rows.empty else 1 - float(
                    (1 - rows["valid_fraction_core"].clip(0, 1)).prod()
                )
                depth3 = 0.0 if len(rows) < 3 else float(
                    rows["valid_fraction_core"].nlargest(3).mean()
                )
                records.append({
                    "epoch": int(epoch),
                    "window_name": window.name,
                    "number_of_months": window.months,
                    "window_start": start.date().isoformat(),
                    "window_end": end.date().isoformat(),
                    "sensor_mode": mode,
                    "sensor_keys": ",".join(sensors),
                    "scene_count": len(rows),
                    "coverage_at_least_1_core_pct": combined * 100,
                    "coverage_at_least_3_core_pct": depth3 * 100,
                    "supplemental_sensor_used": mode == "primary_plus_supplemental",
                })
    return pd.DataFrame(records)


def select_protocol(window_results: pd.DataFrame, config: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """Select one common window and one sensor mode per epoch deterministically."""
    primary = window_results[window_results["sensor_mode"] == "primary_only"]
    summary = (
        primary.groupby(["window_name", "number_of_months"], as_index=False)
        .agg(
            minimum_epoch_coverage_1_pct=("coverage_at_least_1_core_pct", "min"),
            minimum_epoch_coverage_3_pct=("coverage_at_least_3_core_pct", "min"),
            median_epoch_coverage_3_pct=("coverage_at_least_3_core_pct", "median"),
            total_scene_count=("scene_count", "sum"),
        )
    )
    minimum = float(config["selection"]["minimum_coverage_one_observation_pct"])
    acceptable = summary[summary["minimum_epoch_coverage_1_pct"] >= minimum]
    pool = acceptable if not acceptable.empty else summary
    pool = pool.sort_values(
        ["number_of_months", "minimum_epoch_coverage_3_pct", "total_scene_count", "window_name"],
        ascending=[True, False, True, True],
    )
    selected_window = str(pool.iloc[0]["window_name"])

    chosen: list[pd.Series] = []
    for epoch, group in window_results[window_results["window_name"] == selected_window].groupby("epoch"):
        primary_row = group[group["sensor_mode"] == "primary_only"].iloc[0]
        supplements = group[group["sensor_mode"] == "primary_plus_supplemental"]
        row = primary_row
        if primary_row["coverage_at_least_1_core_pct"] < minimum and not supplements.empty:
            candidate = supplements.iloc[0]
            tolerance = float(config["selection"]["coverage_comparison_tolerance_pct"])
            if candidate["coverage_at_least_1_core_pct"] - primary_row["coverage_at_least_1_core_pct"] > tolerance:
                row = candidate
        chosen.append(row)
    return selected_window, pd.DataFrame(chosen).sort_values("epoch").reset_index(drop=True)


def exact_selected_scenes(manifest: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Select and freeze exact scene records matching each epoch protocol."""
    outputs: list[pd.DataFrame] = []
    for row in quality.itertuples(index=False):
        sensors = str(row.sensor_keys).split(",")
        start, end = pd.Timestamp(row.window_start), pd.Timestamp(row.window_end)
        subset = manifest[
            (manifest["epoch"] == int(row.epoch))
            & manifest["sensor_key"].isin(sensors)
            & (manifest["acquisition_date"] >= start)
            & (manifest["acquisition_date"] < end)
            & manifest["has_positive_core_coverage"].astype(bool)
        ].copy()
        subset["selected_window_name"] = row.window_name
        subset["selected_sensor_mode"] = row.sensor_mode
        outputs.append(subset)
    return pd.concat(outputs, ignore_index=True)


def write_outputs(
    config_path: Path,
    config: dict[str, Any],
    root: Path,
    manifest: pd.DataFrame,
    monthly: pd.DataFrame,
    windows: pd.DataFrame,
    selected_window: str,
    quality: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    """Write all Day 2 tables, protocol and checksum record."""
    directory = resolve_project_path(config["outputs"]["directory"], root)
    directory.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(directory / "scene_manifest_all.csv", index=False, date_format="%Y-%m-%d")
    manifest.to_parquet(directory / "scene_manifest_all.parquet", index=False)
    monthly.to_csv(directory / "monthly_availability.csv", index=False)
    monthly.to_parquet(directory / "monthly_availability.parquet", index=False)
    windows.to_csv(directory / "candidate_window_details.csv", index=False)
    quality.to_csv(directory / "epoch_quality_summary.csv", index=False)
    selected.to_csv(directory / "selected_scene_manifest.csv", index=False, date_format="%Y-%m-%d")

    primary = windows[windows["sensor_mode"] == "primary_only"]
    summary = primary.groupby(["window_name", "number_of_months"], as_index=False).agg(
        minimum_epoch_coverage_1_pct=("coverage_at_least_1_core_pct", "min"),
        minimum_epoch_coverage_3_pct=("coverage_at_least_3_core_pct", "min"),
        median_epoch_coverage_3_pct=("coverage_at_least_3_core_pct", "median"),
        total_scene_count=("scene_count", "sum"),
    )
    summary.to_csv(directory / "candidate_window_summary.csv", index=False)

    epochs: dict[int, dict[str, Any]] = {}
    for row in quality.itertuples(index=False):
        rows = selected[selected["epoch"] == int(row.epoch)]
        epochs[int(row.epoch)] = {
            "window_name": row.window_name,
            "window_start": row.window_start,
            "window_end": row.window_end,
            "sensor_mode": row.sensor_mode,
            "sensor_keys": str(row.sensor_keys).split(","),
            "scene_count": len(rows),
            "earth_engine_asset_ids": rows["earth_engine_asset_id"].astype(str).tolist(),
        }
    protocol = {
        "catalog_version": int(config["catalog"]["version"]),
        "common_window_name": selected_window,
        "quality_mask": config["quality_mask"],
        "selection_thresholds": config["selection"],
        "epochs": epochs,
    }
    protocol_path = directory / "compositing_protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    version = {
        "catalog_version": protocol["catalog_version"],
        "config_sha256": sha256_file(config_path),
        "selected_scene_manifest_sha256": sha256_file(directory / "selected_scene_manifest.csv"),
        "compositing_protocol_sha256": sha256_file(protocol_path),
        "selected_scene_count": len(selected),
    }
    (directory / "catalog_version.json").write_text(json.dumps(version, indent=2), encoding="utf-8")


def main(config_path: Path) -> None:
    """Execute the complete Day 2 catalogue workflow."""
    root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    initialize_earth_engine(config["catalog"].get("earth_engine_project"))
    study = config["study_area"]
    grid = load_grid_specification(resolve_project_path(study["grid_specification"], root))
    report = load_json(resolve_project_path(study["boundary_report"], root))
    core = load_ee_geometry(resolve_project_path(study["core_boundary"], root))
    context = load_ee_geometry(resolve_project_path(study["context_boundary"], root))
    query_geom = query_rectangle(resolve_project_path(study["context_boundary"], root))
    checkpoint_path = (
        resolve_project_path(config["outputs"]["directory"], root)
        / "_checkpoints"
        / "scene_manifest_partial.csv"
    )
    manifest = build_manifest(
        config,
        grid,
        core,
        context,
        query_geom,
        float(report["core_area_vector_km2"]) * 1_000_000,
        float(report["context_area_km2"]) * 1_000_000,
        checkpoint_path,
    )
    monthly = monthly_summary(manifest)
    windows = evaluate_windows(manifest, config)
    selected_window, quality = select_protocol(windows, config)
    selected = exact_selected_scenes(manifest, quality)
    write_outputs(config_path, config, root, manifest, monthly, windows, selected_window, quality, selected)
    print(json.dumps({
        "selected_window": selected_window,
        "manifest_rows": len(manifest),
        "selected_scene_rows": len(selected),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Day 2 Landsat catalogue.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    main(args.config.resolve())
