"""Build the versioned Landsat catalogue and select one exact window per epoch."""

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


def sensor_policy(
    config: dict[str, Any],
    epoch: int,
) -> dict[str, list[str]]:
    """Return normalized sensor roles for one epoch."""
    raw = config["epoch_sensor_policy"].get(epoch)

    if raw is None:
        raw = config["epoch_sensor_policy"][str(epoch)]

    return {
        key: list(raw.get(key, []))
        for key in (
            "primary",
            "supplemental",
            "diagnostic_only",
        )
    }


def diagnostic_range(
    epoch: int,
    config: dict[str, Any],
) -> tuple[str, str]:
    """Return the half-open diagnostic date range."""
    before = int(config["catalog"]["diagnostic_years_before"])
    after = int(config["catalog"]["diagnostic_years_after"])

    return (
        f"{epoch - before}-01-01",
        f"{epoch + after + 1}-01-01",
    )


def safe_number(
    result: ee.Dictionary,
    key: str,
) -> ee.Number:
    """Read one reducer value and replace a missing value with zero."""
    return ee.Number(
        ee.Algorithms.If(
            result.contains(key),
            result.get(key),
            0,
        )
    )


def valid_area(
    mask: ee.Image,
    geometry: ee.Geometry,
    grid: dict[str, Any],
    statistics: dict[str, Any],
) -> ee.Number:
    """Calculate valid area on the frozen affine grid."""
    result = (
        ee.Image.pixelArea()
        .updateMask(mask)
        .rename("area")
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            crs=grid["crs"],
            crsTransform=grid["transform"],
            maxPixels=int(statistics["max_pixels"]),
            tileScale=int(statistics["tile_scale"]),
        )
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
    """Convert one Landsat scene into one manifest row."""
    image = ee.Image(image)
    valid_mask = landsat_valid_mask(
        image,
        sensor,
        config["collections"],
        config["quality_mask"],
    )
    core_valid = valid_area(
        valid_mask,
        core,
        grid,
        config["statistics"],
    )
    context_valid = valid_area(
        valid_mask,
        context,
        grid,
        config["statistics"],
    )
    acquired = ee.Date(image.get("system:time_start"))
    collection_id = config["collections"][sensor]["id"]
    system_index = ee.String(image.get("system:index"))
    asset_id = (
        ee.String(collection_id)
        .cat("/")
        .cat(system_index)
    )

    return ee.Feature(
        None,
        {
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
            "day_of_year": (
                acquired.getRelative("day", "year").add(1)
            ),
            "wrs_path": image.get("WRS_PATH"),
            "wrs_row": image.get("WRS_ROW"),
            "cloud_cover_scene_pct": image.get("CLOUD_COVER"),
            "cloud_cover_land_pct": image.get(
                "CLOUD_COVER_LAND"
            ),
            "valid_area_core_m2": core_valid,
            "valid_area_context_m2": context_valid,
            "valid_fraction_core": (
                core_valid.divide(core_area_m2)
            ),
            "valid_fraction_context": (
                context_valid.divide(context_area_m2)
            ),
            "slc_off": ee.Algorithms.If(
                sensor == "LE07",
                acquired.millis().gt(
                    ee.Date("2003-05-31").millis()
                ),
                False,
            ),
            "landsat7_orbit_drift_period": (
                ee.Algorithms.If(
                    sensor == "LE07",
                    acquired.millis().gt(
                        ee.Date("2017-02-07").millis()
                    ),
                    False,
                )
            ),
            "has_positive_core_coverage": core_valid.gt(0),
            "candidate_for_composite": (
                core_valid.gt(0).And(
                    ee.String(role)
                    .compareTo("diagnostic_only")
                    .neq(0)
                )
            ),
        },
    )


def is_retryable_earth_engine_error(
    error: Exception,
) -> bool:
    """Return whether an Earth Engine error is temporary."""
    message = str(error).casefold()
    fragments = (
        "too many concurrent aggregations",
        "quota exceeded",
        "http 429",
        "rate limit",
        "internal error",
        "service unavailable",
        "deadline exceeded",
        "timed out",
    )

    return any(fragment in message for fragment in fragments)


def get_info_with_retry(
    computed_object: ee.ComputedObject,
    *,
    description: str,
    maximum_attempts: int = 7,
    initial_delay_seconds: float = 3.0,
) -> Any:
    """Evaluate one small Earth Engine request with backoff."""
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


def normalize_manifest(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize manifest types and ordering."""
    if frame.empty:
        raise RuntimeError(
            "The Earth Engine query returned no Landsat scenes."
        )

    result = frame.copy()
    result["acquisition_date"] = pd.to_datetime(
        result["acquisition_date"],
        errors="raise",
    )

    for column in (
        "epoch",
        "relative_year",
        "acquisition_year",
        "acquisition_month",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("Int64")

    for column in (
        "valid_area_core_m2",
        "valid_area_context_m2",
        "valid_fraction_core",
        "valid_fraction_context",
        "cloud_cover_scene_pct",
        "cloud_cover_land_pct",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    for column in (
        "slc_off",
        "landsat7_orbit_drift_period",
        "has_positive_core_coverage",
        "candidate_for_composite",
    ):
        result[column] = (
            result[column]
            .fillna(False)
            .astype(bool)
        )

    key = (
        result["epoch"].astype(str)
        + "::"
        + result["earth_engine_asset_id"].astype(str)
    )

    if not key.is_unique:
        raise ValueError(
            "Duplicate epoch/asset rows were produced."
        )

    return (
        result.sort_values(
            [
                "epoch",
                "acquisition_date",
                "sensor_key",
                "system_index",
            ]
        )
        .reset_index(drop=True)
    )


def checkpoint_config_path(
    checkpoint_path: Path,
) -> Path:
    """Return the checkpoint configuration-hash path."""
    return checkpoint_path.with_suffix(".config.json")


def validate_checkpoint_configuration(
    checkpoint_path: Path,
    config_path: Path,
) -> None:
    """Refuse to reuse a checkpoint from another configuration."""
    metadata_path = checkpoint_config_path(checkpoint_path)

    if not checkpoint_path.exists():
        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        metadata_path.write_text(
            json.dumps(
                {"config_sha256": sha256_file(config_path)},
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    if not metadata_path.is_file():
        raise RuntimeError(
            "Checkpoint metadata is missing. Archive or remove "
            f"{checkpoint_path.parent} before rerunning."
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    current_hash = sha256_file(config_path)

    if metadata.get("config_sha256") != current_hash:
        raise RuntimeError(
            "The existing Landsat checkpoint belongs to another "
            "configuration. Archive data/metadata/landsat and rerun."
        )


def load_manifest_checkpoint(
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Load a partial manifest for resumable catalogue generation."""
    if not checkpoint_path.is_file():
        return pd.DataFrame()

    frame = pd.read_csv(checkpoint_path)

    if "acquisition_date" in frame.columns:
        frame["acquisition_date"] = pd.to_datetime(
            frame["acquisition_date"],
            errors="coerce",
        )

    for column in (
        "slc_off",
        "landsat7_orbit_drift_period",
        "has_positive_core_coverage",
        "candidate_for_composite",
    ):
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.casefold()
                .map(
                    {
                        "true": True,
                        "false": False,
                    }
                )
                .fillna(False)
            )

    return frame


def write_manifest_checkpoint(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
) -> None:
    """Persist completed scene rows."""
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pd.DataFrame(rows).to_csv(
        checkpoint_path,
        index=False,
        date_format="%Y-%m-%d",
    )


def build_manifest(
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
    """Build one complete resumable scene manifest."""
    validate_checkpoint_configuration(
        checkpoint_path,
        config_path,
    )
    checkpoint = load_manifest_checkpoint(checkpoint_path)
    rows = checkpoint.to_dict(orient="records")
    completed = {
        (
            int(row["epoch"]),
            str(row["earth_engine_asset_id"]),
        )
        for row in rows
        if pd.notna(row.get("epoch"))
        and pd.notna(row.get("earth_engine_asset_id"))
    }

    for epoch_value in config["catalog"]["epochs"]:
        epoch = int(epoch_value)
        policy = sensor_policy(config, epoch)
        start_date, end_date = diagnostic_range(
            epoch,
            config,
        )

        for role in (
            "primary",
            "supplemental",
            "diagnostic_only",
        ):
            for sensor in policy[role]:
                collection_id = (
                    config["collections"][sensor]["id"]
                )
                collection = (
                    ee.ImageCollection(collection_id)
                    .filterDate(start_date, end_date)
                    .filterBounds(query_geom)
                    .sort("system:time_start")
                )
                system_indexes = (
                    get_info_with_retry(
                        collection.aggregate_array(
                            "system:index"
                        ),
                        description=(
                            f"listing epoch {epoch}, sensor {sensor}"
                        ),
                    )
                    or []
                )

                print(
                    f"Epoch {epoch}, {sensor} ({role}): "
                    f"{len(system_indexes)} scenes."
                )

                for position, system_index in enumerate(
                    system_indexes,
                    start=1,
                ):
                    asset_id = (
                        f"{collection_id}/{system_index}"
                    )
                    key = (epoch, asset_id)

                    if key in completed:
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
                            f"computing scene {position}/"
                            f"{len(system_indexes)} for epoch "
                            f"{epoch}, sensor {sensor}"
                        ),
                    )
                    rows.append(dict(properties))
                    completed.add(key)
                    write_manifest_checkpoint(
                        rows,
                        checkpoint_path,
                    )

    manifest = normalize_manifest(pd.DataFrame(rows))
    observed_epochs = set(
        manifest["epoch"].astype(int).unique()
    )
    missing_epochs = sorted(
        set(
            int(epoch)
            for epoch in config["catalog"]["epochs"]
        ).difference(observed_epochs)
    )

    if missing_epochs:
        raise RuntimeError(
            "No catalogue scenes were found for configured epochs: "
            f"{missing_epochs}. Remove unavailable epochs explicitly "
            "or correct their collection policy."
        )

    manifest.to_csv(
        checkpoint_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    return manifest


def monthly_summary(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize catalogue availability by month and sensor."""
    return (
        manifest.groupby(
            [
                "epoch",
                "acquisition_month",
                "sensor_key",
            ],
            as_index=False,
        )
        .agg(
            scene_count=(
                "earth_engine_asset_id",
                "count",
            ),
            positive_core_scene_count=(
                "has_positive_core_coverage",
                "sum",
            ),
            median_valid_fraction_core=(
                "valid_fraction_core",
                "median",
            ),
            maximum_valid_fraction_core=(
                "valid_fraction_core",
                "max",
            ),
            median_valid_fraction_context=(
                "valid_fraction_context",
                "median",
            ),
            median_cloud_cover_scene_pct=(
                "cloud_cover_scene_pct",
                "median",
            ),
        )
    )


def candidate_windows(
    config: dict[str, Any],
) -> list[Window]:
    """Generate all configured consecutive windows."""
    windows: list[Window] = []
    allow_cross = bool(
        config["candidate_windows"][
            "allow_cross_year_windows"
        ]
    )

    for length_value in config[
        "candidate_windows"
    ]["lengths_months"]:
        length = int(length_value)

        for start_month in range(1, 13):
            if (
                not allow_cross
                and start_month + length - 1 > 12
            ):
                continue

            end_month = (
                (start_month + length - 2) % 12
            ) + 1
            windows.append(
                Window(
                    name=(
                        f"{calendar.month_abbr[start_month]}-"
                        f"{calendar.month_abbr[end_month]}_"
                        f"{length}m"
                    ),
                    months=length,
                    start_month=start_month,
                )
            )

    return windows


def exact_window_dates(
    epoch: int,
    window: Window,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert one calendar window to exact half-open dates."""
    crosses_year = (
        window.start_month + window.months - 1 > 12
    )
    start_year = epoch - 1 if crosses_year else epoch
    start = pd.Timestamp(
        date(start_year, window.start_month, 1)
    )
    end = start + pd.DateOffset(months=window.months)

    return start, end


def candidate_modes(
    config: dict[str, Any],
    epoch: int,
) -> list[tuple[str, list[str]]]:
    """Return primary and optional supplemental sensor modes."""
    policy = sensor_policy(config, epoch)
    modes = [("primary_only", policy["primary"])]

    if policy["supplemental"]:
        modes.append(
            (
                "primary_plus_supplemental",
                policy["primary"]
                + policy["supplemental"],
            )
        )

    return modes


def exact_observation_metrics(
    rows: pd.DataFrame,
    config: dict[str, Any],
    grid: dict[str, Any],
    core: ee.Geometry,
    core_area_m2: float,
    *,
    description: str,
) -> dict[str, float]:
    """Calculate exact QA-mask union and observation depth."""
    if rows.empty:
        return {
            "scene_count": 0,
            "coverage_at_least_1_core_pct": 0.0,
            "coverage_at_least_3_core_pct": 0.0,
            "mean_valid_observations_core": 0.0,
            "median_valid_observations_core": 0.0,
            "maximum_valid_observations_core": 0.0,
        }

    masks: list[ee.Image] = []

    for row in rows.itertuples(index=False):
        image = ee.Image(str(row.earth_engine_asset_id))
        masks.append(
            landsat_valid_mask(
                image,
                str(row.sensor_key),
                config["collections"],
                config["quality_mask"],
            ).toUint16()
        )

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
            pixel_area
            .updateMask(valid_count.gte(1))
            .rename("area_ge_1"),
            pixel_area
            .updateMask(valid_count.gte(3))
            .rename("area_ge_3"),
            valid_count,
        ]
    )
    reducer = (
        ee.Reducer.sum()
        .combine(
            ee.Reducer.mean(),
            sharedInputs=True,
        )
        .combine(
            ee.Reducer.percentile([50, 100]),
            sharedInputs=True,
        )
    )
    result = ee.Dictionary(
        metrics_image.reduceRegion(
            reducer=reducer,
            geometry=core,
            crs=grid["crs"],
            crsTransform=grid["transform"],
            maxPixels=int(
                config["statistics"]["max_pixels"]
            ),
            tileScale=int(
                config["statistics"]["tile_scale"]
            ),
        )
    )
    values = get_info_with_retry(
        result,
        description=description,
    )

    area_ge_1 = float(
        values.get("area_ge_1_sum", 0.0)
    )
    area_ge_3 = float(
        values.get("area_ge_3_sum", 0.0)
    )

    return {
        "scene_count": int(len(rows)),
        "coverage_at_least_1_core_pct": (
            area_ge_1 / core_area_m2 * 100
        ),
        "coverage_at_least_3_core_pct": (
            area_ge_3 / core_area_m2 * 100
        ),
        "mean_valid_observations_core": float(
            values.get("valid_count_mean", 0.0)
        ),
        "median_valid_observations_core": float(
            values.get("valid_count_p50", 0.0)
        ),
        "maximum_valid_observations_core": float(
            values.get("valid_count_p100", 0.0)
        ),
    }


def evaluate_windows_exact(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    grid: dict[str, Any],
    core: ee.Geometry,
    core_area_m2: float,
) -> pd.DataFrame:
    """Evaluate every window exactly and independently per epoch."""
    records: list[dict[str, Any]] = []

    for epoch_value in config["catalog"]["epochs"]:
        epoch = int(epoch_value)

        for window in candidate_windows(config):
            start, end = exact_window_dates(
                epoch,
                window,
            )

            for mode, sensors in candidate_modes(
                config,
                epoch,
            ):
                rows = manifest[
                    (manifest["epoch"].astype(int) == epoch)
                    & manifest["sensor_key"].isin(sensors)
                    & (
                        manifest["acquisition_date"]
                        >= start
                    )
                    & (
                        manifest["acquisition_date"]
                        < end
                    )
                    & (
                        manifest[
                            "has_positive_core_coverage"
                        ].astype(bool)
                    )
                ].copy()

                metrics = exact_observation_metrics(
                    rows,
                    config,
                    grid,
                    core,
                    core_area_m2,
                    description=(
                        f"evaluating {epoch} {window.name} "
                        f"{mode}"
                    ),
                )
                records.append(
                    {
                        "epoch": epoch,
                        "window_name": window.name,
                        "number_of_months": window.months,
                        "window_start": (
                            start.date().isoformat()
                        ),
                        "window_end": (
                            end.date().isoformat()
                        ),
                        "sensor_mode": mode,
                        "sensor_keys": ",".join(sensors),
                        "supplemental_sensor_used": (
                            mode
                            == "primary_plus_supplemental"
                        ),
                        **metrics,
                    }
                )

                print(
                    f"{epoch} {window.name} {mode}: "
                    f"{metrics['coverage_at_least_1_core_pct']:.2f}%"
                )

    return pd.DataFrame(records)


def choose_sensor_mode_per_window(
    group: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """Prefer primary sensors when they already pass the threshold."""
    minimum = float(
        config["selection"][
            "minimum_coverage_one_observation_pct"
        ]
    )
    tolerance = float(
        config["selection"][
            "coverage_comparison_tolerance_pct"
        ]
    )
    primary = group[
        group["sensor_mode"] == "primary_only"
    ]

    if primary.empty:
        raise ValueError(
            "Every candidate window requires a primary-only row."
        )

    primary_row = primary.iloc[0]
    supplemental = group[
        group["sensor_mode"]
        == "primary_plus_supplemental"
    ]

    if supplemental.empty:
        return primary_row

    supplemental_row = supplemental.iloc[0]

    if (
        primary_row["coverage_at_least_1_core_pct"]
        >= minimum
    ):
        return primary_row

    gain = (
        supplemental_row[
            "coverage_at_least_1_core_pct"
        ]
        - primary_row[
            "coverage_at_least_1_core_pct"
        ]
    )

    if gain > tolerance:
        return supplemental_row

    return primary_row


def select_epoch_protocol(
    epoch_results: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """Select the shortest acceptable exact window for one epoch."""
    minimum = float(
        config["selection"][
            "minimum_coverage_one_observation_pct"
        ]
    )
    preferred_depth = float(
        config["selection"][
            "preferred_coverage_three_observations_pct"
        ]
    )
    chosen_modes = []

    for _, group in epoch_results.groupby(
        [
            "window_name",
            "number_of_months",
            "window_start",
            "window_end",
        ],
        sort=False,
    ):
        chosen_modes.append(
            choose_sensor_mode_per_window(
                group,
                config,
            )
        )

    candidates = pd.DataFrame(chosen_modes)
    acceptable = candidates[
        candidates[
            "coverage_at_least_1_core_pct"
        ]
        >= minimum
    ].copy()

    if not acceptable.empty:
        shortest = int(
            acceptable["number_of_months"].min()
        )
        pool = acceptable[
            acceptable["number_of_months"]
            == shortest
        ].copy()
        pool["passes_preferred_depth"] = (
            pool[
                "coverage_at_least_3_core_pct"
            ]
            >= preferred_depth
        )
        pool = pool.sort_values(
            [
                "passes_preferred_depth",
                "coverage_at_least_3_core_pct",
                "median_valid_observations_core",
                "coverage_at_least_1_core_pct",
                "scene_count",
                "window_name",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        selected = pool.iloc[0].copy()
        selected["threshold_passed"] = True
        selected["selection_status"] = (
            "PASS_SHORTEST_ACCEPTABLE"
        )
        selected["fallback_reason"] = ""
        return selected

    pool = candidates.sort_values(
        [
            "coverage_at_least_1_core_pct",
            "coverage_at_least_3_core_pct",
            "median_valid_observations_core",
            "number_of_months",
            "scene_count",
            "window_name",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            False,
            True,
        ],
    )
    selected = pool.iloc[0].copy()
    selected["threshold_passed"] = False
    selected["selection_status"] = (
        "FALLBACK_BELOW_95"
    )
    selected["fallback_reason"] = (
        "No evaluated window reached the configured "
        f"{minimum:.1f}% coverage threshold."
    )

    return selected


def select_protocol_per_epoch(
    window_results: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select one exact protocol independently for every epoch."""
    selected = []

    for epoch, group in window_results.groupby(
        "epoch",
        sort=True,
    ):
        row = select_epoch_protocol(
            group,
            config,
        )
        row["epoch"] = int(epoch)
        selected.append(row)

    quality = (
        pd.DataFrame(selected)
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    if set(quality["epoch"].astype(int)) != set(
        int(epoch)
        for epoch in config["catalog"]["epochs"]
    ):
        raise ValueError(
            "The selected protocol does not contain every epoch."
        )

    return quality


def exact_selected_scenes(
    manifest: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze exact scene rows matching each selected epoch protocol."""
    outputs: list[pd.DataFrame] = []

    for row in quality.itertuples(index=False):
        sensors = str(row.sensor_keys).split(",")
        start = pd.Timestamp(row.window_start)
        end = pd.Timestamp(row.window_end)
        subset = manifest[
            (
                manifest["epoch"].astype(int)
                == int(row.epoch)
            )
            & manifest["sensor_key"].isin(sensors)
            & (manifest["acquisition_date"] >= start)
            & (manifest["acquisition_date"] < end)
            & (
                manifest[
                    "has_positive_core_coverage"
                ].astype(bool)
            )
        ].copy()

        if subset.empty:
            raise RuntimeError(
                f"The selected protocol for {row.epoch} "
                "contains no positive-coverage scenes."
            )

        subset["selected_window_name"] = (
            row.window_name
        )
        subset["selected_sensor_mode"] = (
            row.sensor_mode
        )
        subset["selection_status"] = (
            row.selection_status
        )
        outputs.append(subset)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def compare_selected_manifests(
    previous_path: Path | None,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Compare old and new selected-scene membership."""
    key_columns = [
        "epoch",
        "earth_engine_asset_id",
    ]
    new_rows = selected[key_columns].copy()
    new_rows["in_new_manifest"] = True

    if previous_path is None:
        new_rows["in_previous_manifest"] = False
        new_rows["change"] = "ADDED"
        return new_rows.sort_values(key_columns)

    if not previous_path.is_file():
        raise FileNotFoundError(
            f"Previous manifest not found: {previous_path}"
        )

    previous = pd.read_csv(previous_path)
    old_rows = previous[key_columns].copy()
    old_rows["in_previous_manifest"] = True
    comparison = old_rows.merge(
        new_rows,
        on=key_columns,
        how="outer",
    )
    comparison[
        [
            "in_previous_manifest",
            "in_new_manifest",
        ]
    ] = comparison[
        [
            "in_previous_manifest",
            "in_new_manifest",
        ]
    ].fillna(False)
    comparison["change"] = "UNCHANGED"
    comparison.loc[
        comparison["in_new_manifest"]
        & ~comparison["in_previous_manifest"],
        "change",
    ] = "ADDED"
    comparison.loc[
        comparison["in_previous_manifest"]
        & ~comparison["in_new_manifest"],
        "change",
    ] = "REMOVED"

    return comparison.sort_values(key_columns)


def write_outputs(
    config_path: Path,
    config: dict[str, Any],
    root: Path,
    manifest: pd.DataFrame,
    monthly: pd.DataFrame,
    windows: pd.DataFrame,
    quality: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Write the version 3 catalogue and protocol."""
    directory = resolve_project_path(
        config["outputs"]["directory"],
        root,
    )
    directory.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        directory / "scene_manifest_all.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    manifest.to_parquet(
        directory / "scene_manifest_all.parquet",
        index=False,
    )
    monthly.to_csv(
        directory / "monthly_availability.csv",
        index=False,
    )
    monthly.to_parquet(
        directory / "monthly_availability.parquet",
        index=False,
    )
    windows.to_csv(
        directory / "candidate_window_details.csv",
        index=False,
    )

    summary = (
        windows.groupby(
            [
                "epoch",
                "window_name",
                "number_of_months",
            ],
            as_index=False,
        )
        .agg(
            best_coverage_at_least_1_core_pct=(
                "coverage_at_least_1_core_pct",
                "max",
            ),
            best_coverage_at_least_3_core_pct=(
                "coverage_at_least_3_core_pct",
                "max",
            ),
            maximum_scene_count=(
                "scene_count",
                "max",
            ),
        )
    )
    summary.to_csv(
        directory / "candidate_window_summary.csv",
        index=False,
    )
    quality.to_csv(
        directory / "epoch_quality_summary.csv",
        index=False,
    )
    selected.to_csv(
        directory / "selected_scene_manifest.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    comparison.to_csv(
        directory / "old_vs_new_selected_manifest.csv",
        index=False,
    )

    epochs: dict[int, dict[str, Any]] = {}

    for row in quality.itertuples(index=False):
        rows = selected[
            selected["epoch"].astype(int)
            == int(row.epoch)
        ]
        epochs[int(row.epoch)] = {
            "window_name": row.window_name,
            "window_start": row.window_start,
            "window_end": row.window_end,
            "sensor_mode": row.sensor_mode,
            "sensor_keys": str(
                row.sensor_keys
            ).split(","),
            "scene_count": int(len(rows)),
            "coverage_at_least_1_core_pct": float(
                row.coverage_at_least_1_core_pct
            ),
            "coverage_at_least_3_core_pct": float(
                row.coverage_at_least_3_core_pct
            ),
            "mean_valid_observations_core": float(
                row.mean_valid_observations_core
            ),
            "median_valid_observations_core": float(
                row.median_valid_observations_core
            ),
            "threshold_passed": bool(
                row.threshold_passed
            ),
            "selection_status": row.selection_status,
            "fallback_reason": row.fallback_reason,
            "earth_engine_asset_ids": (
                rows[
                    "earth_engine_asset_id"
                ]
                .astype(str)
                .tolist()
            ),
        }

    protocol = {
        "catalog_version": int(
            config["catalog"]["version"]
        ),
        "selection_scope": "per_epoch",
        "quality_mask": config["quality_mask"],
        "selection_thresholds": config["selection"],
        "epochs": epochs,
    }
    protocol_path = (
        directory / "compositing_protocol.yaml"
    )
    protocol_path.write_text(
        __import__("yaml").safe_dump(
            protocol,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    version = {
        "catalog_version": protocol[
            "catalog_version"
        ],
        "config_sha256": sha256_file(config_path),
        "selected_scene_manifest_sha256": (
            sha256_file(
                directory
                / "selected_scene_manifest.csv"
            )
        ),
        "compositing_protocol_sha256": (
            sha256_file(protocol_path)
        ),
        "selected_scene_count": int(len(selected)),
        "selection_scope": "per_epoch",
    }
    (
        directory / "catalog_version.json"
    ).write_text(
        json.dumps(version, indent=2),
        encoding="utf-8",
    )


def main(
    config_path: Path,
    previous_selected_manifest: Path | None,
) -> None:
    """Execute the complete version 3 catalogue workflow."""
    root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    initialize_earth_engine(
        config["catalog"].get(
            "earth_engine_project"
        )
    )
    study = config["study_area"]
    grid = load_grid_specification(
        resolve_project_path(
            study["grid_specification"],
            root,
        )
    )
    report = load_json(
        resolve_project_path(
            study["boundary_report"],
            root,
        )
    )
    core = load_ee_geometry(
        resolve_project_path(
            study["core_boundary"],
            root,
        )
    )
    context = load_ee_geometry(
        resolve_project_path(
            study["context_boundary"],
            root,
        )
    )
    query_geom = query_rectangle(
        resolve_project_path(
            study["context_boundary"],
            root,
        )
    )
    checkpoint_path = (
        resolve_project_path(
            config["outputs"]["directory"],
            root,
        )
        / "_checkpoints"
        / "scene_manifest_partial.csv"
    )
    core_area_m2 = (
        float(report["core_area_vector_km2"])
        * 1_000_000
    )
    context_area_m2 = (
        float(report["context_area_km2"])
        * 1_000_000
    )

    manifest = build_manifest(
        config_path,
        config,
        grid,
        core,
        context,
        query_geom,
        core_area_m2,
        context_area_m2,
        checkpoint_path,
    )
    monthly = monthly_summary(manifest)
    windows = evaluate_windows_exact(
        manifest,
        config,
        grid,
        core,
        core_area_m2,
    )
    quality = select_protocol_per_epoch(
        windows,
        config,
    )
    selected = exact_selected_scenes(
        manifest,
        quality,
    )
    comparison = compare_selected_manifests(
        previous_selected_manifest,
        selected,
    )
    write_outputs(
        config_path,
        config,
        root,
        manifest,
        monthly,
        windows,
        quality,
        selected,
        comparison,
    )

    print(
        json.dumps(
            {
                "catalog_version": int(
                    config["catalog"]["version"]
                ),
                "selection_scope": "per_epoch",
                "manifest_rows": int(len(manifest)),
                "selected_scene_rows": int(
                    len(selected)
                ),
                "selected_protocol": (
                    quality[
                        [
                            "epoch",
                            "window_name",
                            "sensor_mode",
                            "scene_count",
                            "coverage_at_least_1_core_pct",
                            "coverage_at_least_3_core_pct",
                            "selection_status",
                        ]
                    ]
                    .to_dict(orient="records")
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact per-epoch Landsat catalogue."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--previous-selected-manifest",
        type=Path,
    )
    arguments = parser.parse_args()

    main(
        arguments.config.resolve(),
        (
            arguments.previous_selected_manifest.resolve()
            if arguments.previous_selected_manifest
            else None
        ),
    )
