"""Build calendar-year Landsat catalogues on the frozen project grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ee
import pandas as pd
import yaml
from src.analysis.orchestration.preflight import compare_grid_reference

from .build_catalog import (
    get_info_with_retry,
    load_manifest_checkpoint,
    normalize_manifest,
    scene_feature,
    validate_checkpoint_configuration,
    write_manifest_checkpoint,
)
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

QUALITY_FLAGS = {"PASS", "LIMITED", "FAIL"}


def calendar_year_window(year: int) -> tuple[str, str]:
    """Return the exact half-open calendar-year observation window."""
    return f"{year:04d}-01-01", f"{year + 1:04d}-01-01"


def annual_sensor_policy(config: dict[str, Any], year: int) -> dict[str, list[str]]:
    """Resolve one year's sensor roles from the compact era rules."""
    matches = [
        era
        for era in config["sensor_eras"]
        if int(era["start_year"]) <= year <= int(era["end_year"])
    ]

    if len(matches) != 1:
        raise ValueError(f"Year {year} must match exactly one sensor era.")

    era = matches[0]
    return {
        role: list(era.get(role, []))
        for role in ("primary", "supplemental", "diagnostic_only")
    }


def validate_annual_configuration(config: dict[str, Any]) -> list[int]:
    """Validate the annual temporal and sensor-policy contract."""
    catalog = config["catalog"]
    years = [int(value) for value in catalog["years"]]

    if years != list(range(2000, 2026)) or len(set(years)) != len(years):
        raise ValueError("Annual catalogue years must be consecutive from 2000 to 2025.")

    expected = {
        "temporal_mode": "calendar_year",
        "window_end_exclusive": True,
        "allow_future_imagery": False,
        "automatic_temporal_fallback": False,
    }
    for key, value in expected.items():
        if catalog.get(key) != value:
            raise ValueError(f"Annual catalogue requires {key}={value!r}.")

    supported = set(config["collections"])
    for year in years:
        policy = annual_sensor_policy(config, year)
        sensors = [sensor for role in policy.values() for sensor in role]
        if not policy["primary"]:
            raise ValueError(f"Year {year} has no primary sensor policy.")
        if len(sensors) != len(set(sensors)):
            raise ValueError(f"Year {year} assigns a sensor to more than one role.")
        unknown = set(sensors).difference(supported)
        if unknown:
            raise ValueError(f"Year {year} uses unsupported sensors: {sorted(unknown)}")

    return years


def build_annual_manifest(
    config_path: Path,
    config: dict[str, Any],
    grid: dict[str, Any],
    core: ee.Geometry,
    context: ee.Geometry,
    query_geom: ee.Geometry,
    core_area_m2: float,
    context_area_m2: float,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Query same-year sensor availability and build a resumable scene manifest."""
    validate_checkpoint_configuration(checkpoint_path, config_path)
    checkpoint = load_manifest_checkpoint(checkpoint_path)
    rows = checkpoint.to_dict(orient="records")
    completed = {
        (int(row["epoch"]), str(row["earth_engine_asset_id"]))
        for row in rows
        if pd.notna(row.get("epoch")) and pd.notna(row.get("earth_engine_asset_id"))
    }

    for year in validate_annual_configuration(config):
        start_date, end_date = calendar_year_window(year)
        policy = annual_sensor_policy(config, year)

        for role in ("primary", "supplemental", "diagnostic_only"):
            for sensor in policy[role]:
                collection_id = config["collections"][sensor]["id"]
                collection = (
                    ee.ImageCollection(collection_id)
                    .filterDate(start_date, end_date)
                    .filterBounds(query_geom)
                    .sort("system:time_start")
                )
                system_indexes = (
                    get_info_with_retry(
                        collection.aggregate_array("system:index"),
                        description=f"listing year {year}, sensor {sensor}",
                    )
                    or []
                )
                print(f"Year {year}, {sensor} ({role}): {len(system_indexes)} scenes.")

                for position, system_index in enumerate(system_indexes, start=1):
                    asset_id = f"{collection_id}/{system_index}"
                    key = (year, asset_id)
                    if key in completed:
                        continue

                    feature = scene_feature(
                        ee.Image(asset_id),
                        year,
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
                            f"computing scene {position}/{len(system_indexes)} "
                            f"for year {year}, sensor {sensor}"
                        ),
                    )
                    rows.append(dict(properties))
                    completed.add(key)
                    write_manifest_checkpoint(rows, checkpoint_path)

    manifest = normalize_manifest(pd.DataFrame(rows))
    year_values = manifest["epoch"].astype("Int64")
    if "year" in manifest.columns:
        manifest["year"] = year_values
    else:
        manifest.insert(0, "year", year_values)
    manifest.to_csv(checkpoint_path, index=False, date_format="%Y-%m-%d")
    return manifest


def annual_observation_metrics(
    rows: pd.DataFrame,
    config: dict[str, Any],
    grid: dict[str, Any],
    core: ee.Geometry,
    core_area_m2: float,
    *,
    description: str,
) -> dict[str, float | int]:
    """Measure valid-observation coverage and depth over the core boundary."""
    if rows.empty:
        return {
            "selected_scene_count": 0,
            "coverage_at_least_1_core_pct": 0.0,
            "coverage_at_least_3_core_pct": 0.0,
            "mean_observation_count": 0.0,
            "median_observation_count": 0.0,
            "minimum_observation_count": 0.0,
            "maximum_observation_count": 0.0,
        }

    masks = [
        landsat_valid_mask(
            ee.Image(str(row.earth_engine_asset_id)),
            str(row.sensor_key),
            config["collections"],
            config["quality_mask"],
        ).toUint16()
        for row in rows.itertuples(index=False)
    ]
    valid_count = (
        ee.ImageCollection.fromImages(masks)
        .sum()
        .rename("valid_count")
        .unmask(0)
        .toFloat()
    )
    pixel_area = ee.Image.pixelArea()
    metrics_image = ee.Image.cat(
        [
            pixel_area.updateMask(valid_count.gte(1)).rename("area_ge_1"),
            pixel_area.updateMask(valid_count.gte(3)).rename("area_ge_3"),
            valid_count,
        ]
    )
    reducer = (
        ee.Reducer.sum()
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
    )
    values = get_info_with_retry(
        metrics_image.reduceRegion(
            reducer=reducer,
            geometry=core,
            crs=grid["crs"],
            crsTransform=grid["transform"],
            maxPixels=int(config["statistics"]["max_pixels"]),
            tileScale=int(config["statistics"]["tile_scale"]),
        ),
        description=description,
    )

    return {
        "selected_scene_count": int(len(rows)),
        "coverage_at_least_1_core_pct": (
            float(values.get("area_ge_1_sum", 0.0)) / core_area_m2 * 100
        ),
        "coverage_at_least_3_core_pct": (
            float(values.get("area_ge_3_sum", 0.0)) / core_area_m2 * 100
        ),
        "mean_observation_count": float(values.get("valid_count_mean", 0.0)),
        "median_observation_count": float(values.get("valid_count_median", 0.0)),
        "minimum_observation_count": float(values.get("valid_count_min", 0.0)),
        "maximum_observation_count": float(values.get("valid_count_max", 0.0)),
    }


def quality_flag(
    scene_count: int,
    coverage_at_least_1_core_pct: float,
    minimum_coverage_pct: float = 95.0,
) -> str:
    """Assign an explicit annual quality status without temporal repair."""
    if scene_count == 0:
        return "FAIL"
    if coverage_at_least_1_core_pct >= minimum_coverage_pct:
        return "PASS"
    return "LIMITED"


def select_annual_observations(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    grid: dict[str, Any],
    core: ee.Geometry,
    core_area_m2: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select primary or needed supplemental scenes independently by year."""
    quality_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    minimum = float(config["selection"]["minimum_coverage_one_observation_pct"])
    preferred = float(config["selection"]["preferred_coverage_three_observations_pct"])
    tolerance = float(config["selection"]["coverage_comparison_tolerance_pct"])

    for year in validate_annual_configuration(config):
        start_date, end_date = calendar_year_window(year)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        policy = annual_sensor_policy(config, year)
        year_rows = manifest[
            (manifest["year"].astype(int) == year)
            & (manifest["acquisition_date"] >= start)
            & (manifest["acquisition_date"] < end)
            & manifest["has_positive_core_coverage"].astype(bool)
        ].copy()
        primary = year_rows[year_rows["sensor_key"].isin(policy["primary"])].copy()
        chosen = primary
        metrics = annual_observation_metrics(
            primary,
            config,
            grid,
            core,
            core_area_m2,
            description=f"evaluating year {year} primary sensors",
        )
        sensor_mode = "primary_only"

        if metrics["coverage_at_least_1_core_pct"] < minimum and policy["supplemental"]:
            combined = year_rows[
                year_rows["sensor_key"].isin(policy["primary"] + policy["supplemental"])
            ].copy()
            combined_metrics = annual_observation_metrics(
                combined,
                config,
                grid,
                core,
                core_area_m2,
                description=f"evaluating year {year} with supplemental sensors",
            )
            gain = (
                float(combined_metrics["coverage_at_least_1_core_pct"])
                - float(metrics["coverage_at_least_1_core_pct"])
            )
            if (
                combined_metrics["coverage_at_least_1_core_pct"] >= minimum
                or gain > tolerance
                or primary.empty
            ):
                chosen = combined
                metrics = combined_metrics
                sensor_mode = "primary_plus_supplemental"

        sensors = sorted(chosen["sensor_key"].astype(str).unique())
        counts = {
            sensor: int(count)
            for sensor, count in chosen["sensor_key"].value_counts().sort_index().items()
        }
        status = quality_flag(
            int(metrics["selected_scene_count"]),
            float(metrics["coverage_at_least_1_core_pct"]),
            minimum,
        )
        quality_rows.append(
            {
                "year": year,
                "epoch": year,
                "window_start": start_date,
                "window_end_exclusive": end_date,
                "sensor_mode": sensor_mode,
                "selected_sensors": ",".join(sensors),
                "scene_counts_by_sensor": json.dumps(counts, sort_keys=True),
                "first_selected_acquisition_date": (
                    chosen["acquisition_date"].min().date().isoformat()
                    if not chosen.empty
                    else ""
                ),
                "last_selected_acquisition_date": (
                    chosen["acquisition_date"].max().date().isoformat()
                    if not chosen.empty
                    else ""
                ),
                **metrics,
                "preferred_depth_reached": (
                    float(metrics["coverage_at_least_3_core_pct"]) >= preferred
                ),
                "quality_flag": status,
            }
        )

        if not chosen.empty:
            chosen["selected_sensor_mode"] = sensor_mode
            chosen["quality_flag"] = status
            selected_frames.append(chosen)

    quality = pd.DataFrame(quality_rows).sort_values("year").reset_index(drop=True)
    selected = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else manifest.iloc[0:0].copy()
    )
    return quality, selected


def annual_catalog_report(quality: pd.DataFrame) -> str:
    """Render the concise annual support and sensor-contribution report."""
    pass_count = int((quality["quality_flag"] == "PASS").sum())
    limited_or_fail = quality.loc[
        quality["quality_flag"] != "PASS", "year"
    ].astype(int).tolist()
    low_depth = quality.loc[
        ~quality["preferred_depth_reached"].astype(bool), "year"
    ].astype(int).tolist()
    lowest = quality.sort_values("coverage_at_least_1_core_pct").iloc[0]
    lines = [
        "# Annual Landsat catalogue",
        "",
        f"- PASS years: {pass_count} of {len(quality)}",
        f"- LIMITED/FAIL years: {limited_or_fail or 'None'}",
        (
            f"- Lowest core coverage: {int(lowest['year'])} "
            f"({float(lowest['coverage_at_least_1_core_pct']):.2f}%)"
        ),
        f"- Years below preferred observation depth: {low_depth or 'None'}",
        "",
        "| Year | Sensors | Scenes | >=1 core | >=3 core | Mean | Median | Quality |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in quality.itertuples(index=False):
        lines.append(
            f"| {int(row.year)} | {row.selected_sensors or '-'} | "
            f"{int(row.selected_scene_count)} | "
            f"{float(row.coverage_at_least_1_core_pct):.2f}% | "
            f"{float(row.coverage_at_least_3_core_pct):.2f}% | "
            f"{float(row.mean_observation_count):.2f} | "
            f"{float(row.median_observation_count):.2f} | {row.quality_flag} |"
        )
    return "\n".join(lines) + "\n"


def write_annual_outputs(
    config_path: Path,
    config: dict[str, Any],
    root: Path,
    manifest: pd.DataFrame,
    quality: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    """Write annual manifests, protocol, version metadata and report."""
    output_dir = resolve_project_path(config["outputs"]["directory"], root)
    report_dir = resolve_project_path(config["outputs"]["report_directory"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "scene_manifest_all.csv", index=False, date_format="%Y-%m-%d")
    selected.to_csv(
        output_dir / "selected_scene_manifest.csv", index=False, date_format="%Y-%m-%d"
    )
    quality.to_csv(output_dir / "annual_quality_summary.csv", index=False)

    epochs: dict[int, dict[str, Any]] = {}
    for row in quality.itertuples(index=False):
        scenes = selected[selected["year"].astype(int) == int(row.year)]
        epochs[int(row.year)] = {
            "window_start": row.window_start,
            "window_end_exclusive": row.window_end_exclusive,
            "sensor_mode": row.sensor_mode,
            "selected_sensors": str(row.selected_sensors).split(",") if row.selected_sensors else [],
            "selected_scene_count": int(row.selected_scene_count),
            "scene_counts_by_sensor": json.loads(row.scene_counts_by_sensor),
            "coverage_at_least_1_core_pct": float(row.coverage_at_least_1_core_pct),
            "coverage_at_least_3_core_pct": float(row.coverage_at_least_3_core_pct),
            "mean_observation_count": float(row.mean_observation_count),
            "median_observation_count": float(row.median_observation_count),
            "minimum_observation_count": float(row.minimum_observation_count),
            "maximum_observation_count": float(row.maximum_observation_count),
            "quality_flag": row.quality_flag,
            "earth_engine_asset_ids": scenes["earth_engine_asset_id"].astype(str).tolist(),
        }

    protocol = {
        "catalog_version": int(config["catalog"]["version"]),
        "selection_scope": "per_year",
        "temporal_mode": "calendar_year",
        "allow_future_imagery": False,
        "automatic_temporal_fallback": False,
        "quality_mask": config["quality_mask"],
        "selection_thresholds": config["selection"],
        "epochs": epochs,
    }
    protocol_path = output_dir / "compositing_protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    selected_path = output_dir / "selected_scene_manifest.csv"
    version = {
        "catalog_version": int(config["catalog"]["version"]),
        "config_sha256": sha256_file(config_path),
        "selected_scene_manifest_sha256": sha256_file(selected_path),
        "compositing_protocol_sha256": sha256_file(protocol_path),
        "selected_scene_count": int(len(selected)),
        "requested_year_count": len(config["catalog"]["years"]),
        "selection_scope": "per_year",
    }
    (output_dir / "catalog_version.json").write_text(
        json.dumps(version, indent=2) + "\n", encoding="utf-8"
    )
    report_path = report_dir / config["outputs"]["report_name"]
    report_path.write_text(annual_catalog_report(quality), encoding="utf-8")


def main(config_path: Path) -> None:
    """Execute the annual calendar-year catalogue workflow."""
    root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    validate_annual_configuration(config)
    initialize_earth_engine(config["catalog"]["earth_engine_project"])
    study = config["study_area"]
    grid = load_grid_specification(resolve_project_path(study["grid_specification"], root))
    grid_reference = load_json(resolve_project_path(study["grid_reference"], root))
    compare_grid_reference(grid, grid_reference)
    report = load_json(resolve_project_path(study["boundary_report"], root))
    core = load_ee_geometry(resolve_project_path(study["core_boundary"], root))
    context = load_ee_geometry(resolve_project_path(study["context_boundary"], root))
    query_geom = query_rectangle(resolve_project_path(study["context_boundary"], root))
    checkpoint_path = (
        resolve_project_path(config["outputs"]["checkpoint_directory"], root)
        / "scene_manifest_partial.csv"
    )

    manifest = build_annual_manifest(
        config_path,
        config,
        grid,
        core,
        context,
        query_geom,
        float(report["core_area_vector_km2"]) * 1_000_000,
        float(report["context_area_km2"]) * 1_000_000,
        checkpoint_path,
    )
    quality, selected = select_annual_observations(
        manifest,
        config,
        grid,
        core,
        float(report["core_area_vector_km2"]) * 1_000_000,
    )
    write_annual_outputs(config_path, config, root, manifest, quality, selected)
    print(
        json.dumps(
            {
                "requested_years": len(quality),
                "pass_years": int((quality["quality_flag"] == "PASS").sum()),
                "limited_years": quality.loc[
                    quality["quality_flag"] == "LIMITED", "year"
                ].astype(int).tolist(),
                "fail_years": quality.loc[
                    quality["quality_flag"] == "FAIL", "year"
                ].astype(int).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build annual calendar-year Landsat catalogue.")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    main(arguments.config.resolve())