"""Minimal Day 5 validation and mapping-method selection.

The module validates finalized Day 4 candidates, generates a reproducible
three-stratum sample, evaluates manual labels with design-weighted F1 and
reversal rates, then freezes one method for Day 6. It exports no new classified
raster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import ee
except ImportError:
    ee = None

import geopandas as gpd
import numpy as np
import pandas as pd
try:
    import pyogrio
except ImportError:
    pyogrio = None
from shapely.geometry import Point

from src.analysis.orchestration.common import (
    asset_exists,
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_json,
    load_yaml,
    resolve_project_path,
    sha256_file,
    stable_object_hash,
    write_json,
    write_yaml,
)


STRATUM_CODES = {
    "unanimous_nonbuilt": 0,
    "disagreement": 1,
    "unanimous_built": 2,
}
PASS_STATES = {"PASS", "COMPLETED", "EXISTS"}


@dataclass(frozen=True)
class AnchorYear:
    """Represent one eligible validation epoch and its source coverage."""

    epoch: int
    core_coverage_pct: float
    quality_flag: str
    coverage_source: str


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError(
            "The Earth Engine Python API is required for Day 5 spatial "
            "sampling and reversal analysis."
        )


def ensure_directory(path: Path) -> Path:
    """Create and return one directory."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_paths(
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Path]:
    """Resolve configured input paths relative to the repository."""
    return {
        name: resolve_project_path(value, project_root)
        for name, value in config["inputs"].items()
    }


def output_paths(
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Path]:
    """Resolve all Day 5 output files."""
    validation_dir = ensure_directory(
        resolve_project_path(
            config["outputs"]["validation_directory"],
            project_root,
        )
    )
    metadata_dir = ensure_directory(
        resolve_project_path(
            config["outputs"]["metadata_directory"],
            project_root,
        )
    )
    report_dir = ensure_directory(
        resolve_project_path(
            config["outputs"]["report_directory"],
            project_root,
        )
    )

    return {
        "validation_directory": validation_dir,
        "metadata_directory": metadata_dir,
        "report_directory": report_dir,
        "validation_samples": (
            validation_dir / config["outputs"]["validation_samples"]
        ),
        "manual_labels": (
            validation_dir / config["outputs"]["manual_labels"]
        ),
        "validation_dataset": (
            validation_dir / config["outputs"]["validation_dataset"]
        ),
        "method_scores": (
            metadata_dir / config["outputs"]["method_scores"]
        ),
        "selected_protocol": (
            metadata_dir / config["outputs"]["selected_protocol"]
        ),
        "version": metadata_dir / config["outputs"]["version"],
        "report": report_dir / config["outputs"]["report"],
    }


def expected_candidate_bands(methods: Iterable[str]) -> list[str]:
    """Return the dynamic Day 4 candidate-asset schema."""
    method_list = list(methods)
    return (
        [f"built_{name}" for name in method_list]
        + [f"valid_{name}" for name in method_list]
    )


def image_metadata(asset_id: str) -> dict[str, Any]:
    """Read ordered bands and exact grid metadata from an EE asset."""
    information = ee.Image(asset_id).getInfo()
    bands = information.get("bands", [])

    if not bands:
        raise ValueError(f"Earth Engine asset has no bands: {asset_id}")

    first = bands[0]
    return {
        "band_names": [str(band["id"]) for band in bands],
        "width": int(first["dimensions"][0]),
        "height": int(first["dimensions"][1]),
        "crs": str(first["crs"]),
        "transform": [
            float(value)
            for value in first["crs_transform"]
        ],
    }


def compare_grid(
    observed: dict[str, Any],
    expected: dict[str, Any],
    asset_id: str,
) -> None:
    """Reject one-pixel shifts, wrong dimensions and wrong CRS."""
    if observed["crs"] != expected["crs"]:
        raise ValueError(
            f"Unexpected CRS for {asset_id}: {observed['crs']}"
        )

    if not np.allclose(
        observed["transform"],
        [float(value) for value in expected["transform"]],
        atol=1e-9,
    ):
        raise ValueError(
            f"Unexpected transform for {asset_id}: "
            f"{observed['transform']}"
        )

    if (
        observed["width"] != int(expected["width"])
        or observed["height"] != int(expected["height"])
    ):
        raise ValueError(
            f"Unexpected dimensions for {asset_id}: "
            f"{observed['width']} x {observed['height']}"
        )


def candidate_manifest(
    manifest_path: Path,
    methods: list[str],
) -> pd.DataFrame:
    """Load one finalized candidate asset per epoch."""
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "epoch",
        "product_type",
        "asset_id",
        "state",
        "band_names",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Output manifest is missing columns: {sorted(missing)}"
        )

    candidates = frame[
        frame["product_type"].astype(str) == "candidates"
    ].copy()
    candidates["epoch"] = candidates["epoch"].astype(int)
    candidates["state"] = (
        candidates["state"].astype(str).str.upper()
    )
    candidates = candidates[
        candidates["state"].isin(PASS_STATES)
    ].sort_values("epoch")

    if candidates.empty:
        raise ValueError(
            "No finalized Day 4 candidate assets are available."
        )

    if not candidates["epoch"].is_unique:
        raise ValueError(
            "More than one candidate asset exists for an epoch."
        )

    expected = expected_candidate_bands(methods)

    for row in candidates.itertuples(index=False):
        observed = [
            value.strip()
            for value in str(row.band_names).split(",")
            if value.strip()
        ]

        if observed != expected:
            raise ValueError(
                f"Candidate schema mismatch for epoch {row.epoch}: "
                f"{observed} != {expected}"
            )

    return candidates.reset_index(drop=True)


def core_area_ha(core_boundary_path: Path) -> float:
    """Calculate administrative-core area in hectares."""
    frame = gpd.read_file(core_boundary_path)

    if frame.crs is None:
        raise ValueError("The core boundary has no CRS.")

    area = float(
        frame.to_crs("EPSG:32632").geometry.area.sum()
    ) / 10_000

    if area <= 0:
        raise ValueError("The administrative-core area is zero.")

    return area


def detect_coverage_column(frame: pd.DataFrame) -> str | None:
    """Find the strongest available coverage field."""
    candidates = [
        "covered_core_pct",
        "core_coverage_pct",
        "valid_core_coverage_pct",
        "valid_fraction_core",
        "valid_core_fraction",
        "covered_grid_pct",
    ]

    for column in candidates:
        if column in frame.columns:
            return column

    return None


def normalise_percentage(
    value: Any,
    column_name: str,
) -> float:
    """Convert a documented fraction or percentage to percentage units."""
    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"Coverage value is not finite: {value}")

    lower_name = column_name.lower()

    if "fraction" in lower_name:
        number *= 100
    elif "pct" not in lower_name and 0 <= number <= 1.000001:
        number *= 100

    return number


def coverage_table(
    quality_path: Path,
    area_summary_path: Path,
    core_boundary_path: Path,
    candidate_epochs: set[int],
) -> pd.DataFrame:
    """Build one coverage record per candidate epoch.

    Orchestration coverage is preferred. If it is unavailable, valid candidate
    area divided by administrative-core area is used transparently.
    """
    quality = pd.read_csv(quality_path)
    quality["epoch"] = quality["epoch"].astype(int)
    quality = quality[
        quality["epoch"].isin(candidate_epochs)
    ].copy()
    column = detect_coverage_column(quality)
    rows: list[dict[str, Any]] = []

    if column is not None:
        for row in quality.itertuples(index=False):
            values = row._asdict()
            rows.append(
                {
                    "epoch": int(values["epoch"]),
                    "core_coverage_pct": normalise_percentage(
                        values[column],
                        column,
                    ),
                    "coverage_source": column,
                }
            )

    covered_epochs = {int(row["epoch"]) for row in rows}
    missing_epochs = candidate_epochs - covered_epochs

    if missing_epochs:
        areas = pd.read_csv(area_summary_path)
        areas["epoch"] = areas["epoch"].astype(int)
        core_ha = core_area_ha(core_boundary_path)

        for epoch in sorted(missing_epochs):
            valid = areas[
                areas["epoch"] == epoch
            ]["valid_area_ha"]

            if valid.empty:
                raise ValueError(
                    f"No coverage information is available for {epoch}."
                )

            rows.append(
                {
                    "epoch": epoch,
                    "core_coverage_pct": (
                        float(valid.median()) / core_ha * 100
                    ),
                    "coverage_source": (
                        "candidate_valid_area_over_core_area"
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result = (
        result.sort_values(
            ["epoch", "core_coverage_pct"],
            ascending=[True, False],
        )
        .drop_duplicates("epoch", keep="first")
        .sort_values("epoch")
    )

    if set(result["epoch"]) != candidate_epochs:
        raise ValueError(
            "Coverage metadata do not match finalized candidate epochs."
        )

    return result.reset_index(drop=True)


def _evenly_spaced_epochs(
    eligible: pd.DataFrame,
    target_count: int,
) -> list[int]:
    """Select a general number of chronologically distributed epochs."""
    epochs = eligible["epoch"].astype(int).tolist()

    if target_count <= 0:
        raise ValueError("Anchor-year target count must be positive.")

    if target_count >= len(epochs):
        return epochs

    positions = np.linspace(0, len(epochs) - 1, target_count)
    selected = {
        epochs[int(round(position))]
        for position in positions
    }

    if len(selected) < target_count:
        for epoch in epochs:
            selected.add(epoch)

            if len(selected) == target_count:
                break

    return sorted(selected)


def select_anchor_years(
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    config: dict[str, Any],
) -> list[AnchorYear]:
    """Select earliest, temporal-middle and latest eligible epochs."""
    settings = config["anchor_years"]
    fail_floor = float(
        settings["fail_below_core_coverage_pct"]
    )
    warn_floor = float(
        settings["warn_below_core_coverage_pct"]
    )
    target_count = int(settings["target_count"])
    joined = candidates[["epoch"]].merge(
        coverage,
        on="epoch",
        how="left",
        validate="one_to_one",
    )
    eligible = joined[
        joined["core_coverage_pct"] >= fail_floor
    ].sort_values("epoch")

    if eligible.empty:
        raise ValueError(
            "No epoch passes the source-coverage floor."
        )

    epochs = eligible["epoch"].astype(int).tolist()

    if len(epochs) <= target_count:
        selected_epochs = epochs
    elif target_count == 3:
        earliest = epochs[0]
        latest = epochs[-1]
        midpoint = (earliest + latest) / 2.0
        middle = eligible[
            ~eligible["epoch"].isin([earliest, latest])
        ].copy()
        middle["midpoint_distance"] = (
            middle["epoch"] - midpoint
        ).abs()
        middle = middle.sort_values(
            ["midpoint_distance", "core_coverage_pct", "epoch"],
            ascending=[True, False, True],
        )
        selected_epochs = [
            earliest,
            int(middle.iloc[0]["epoch"]),
            latest,
        ]
    else:
        selected_epochs = _evenly_spaced_epochs(
            eligible,
            target_count,
        )

    selected: list[AnchorYear] = []

    for epoch in selected_epochs:
        row = eligible[eligible["epoch"] == epoch].iloc[0]
        coverage_pct = float(row["core_coverage_pct"])
        quality_flag = (
            "PASS"
            if coverage_pct >= warn_floor
            else "LIMITED_SPATIAL_SUPPORT"
        )
        selected.append(
            AnchorYear(
                epoch=int(epoch),
                core_coverage_pct=coverage_pct,
                quality_flag=quality_flag,
                coverage_source=str(row["coverage_source"]),
            )
        )

    return sorted(selected, key=lambda item: item.epoch)


def load_core_geometry(
    path: Path,
) -> tuple[gpd.GeoDataFrame, ee.Geometry]:
    """Load the core boundary locally and in Earth Engine."""
    frame = gpd.read_file(path)

    if frame.crs is None:
        raise ValueError("Core boundary has no CRS.")

    geometry_series = frame.to_crs("EPSG:4326").geometry
    geometry = (
        geometry_series.union_all()
        if hasattr(geometry_series, "union_all")
        else geometry_series.unary_union
    )

    if geometry.is_empty:
        raise ValueError("Core boundary is empty.")

    return frame, ee.Geometry(geometry.__geo_interface__)


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Validate only dependencies that can invalidate method selection."""
    require_earth_engine()
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    paths = load_paths(config, project_root)

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Day 5 dependency '{name}': {path}"
            )

    methods = list(config["candidate_methods"])

    if not methods or len(methods) != len(set(methods)):
        raise ValueError(
            "Candidate methods must be a non-empty unique list."
        )

    if "ibi" in methods:
        raise ValueError(
            "IBI must remain excluded from Day 5 candidates."
        )

    grid = load_grid_specification(paths["grid_specification"])
    reference = load_json(paths["grid_reference"])

    for field in (
        "crs",
        "transform",
        "width",
        "height",
        "resolution_m",
    ):
        if grid.get(field) != reference.get(field):
            raise ValueError(
                f"Frozen grid mismatch for field '{field}'."
            )

    candidates = candidate_manifest(
        paths["output_manifest"],
        methods,
    )
    candidate_epochs = set(candidates["epoch"].astype(int))
    coverage = coverage_table(
        paths["orchestration_quality"],
        paths["candidate_area_summary"],
        paths["core_boundary"],
        candidate_epochs,
    )
    anchors = select_anchor_years(
        candidates,
        coverage,
        config,
    )

    initialize_earth_engine(
        config["project"]["earth_engine_project"]
    )
    expected_bands = expected_candidate_bands(methods)

    for row in candidates.itertuples(index=False):
        asset_id = str(row.asset_id)

        if not asset_exists(asset_id):
            raise FileNotFoundError(
                f"Candidate asset not found: {asset_id}"
            )

        metadata = image_metadata(asset_id)

        if metadata["band_names"] != expected_bands:
            raise ValueError(
                f"Unexpected bands in {asset_id}: "
                f"{metadata['band_names']}"
            )

        compare_grid(metadata, grid, asset_id)

    outputs = output_paths(config, project_root)
    summary = {
        "status": "PASS",
        "candidate_methods": methods,
        "candidate_epochs": sorted(candidate_epochs),
        "anchor_years": [
            {
                "epoch": item.epoch,
                "core_coverage_pct": item.core_coverage_pct,
                "quality_flag": item.quality_flag,
                "coverage_source": item.coverage_source,
            }
            for item in anchors
        ],
        "target_samples": (
            len(anchors)
            * sum(
                int(value)
                for value in config["sampling"][
                    "per_anchor_year"
                ].values()
            )
        ),
        "validation_directory": str(
            outputs["validation_directory"]
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


def candidate_vote_image(
    asset_id: str,
    methods: list[str],
) -> ee.Image:
    """Create common-valid support, vote count and three-stratum code."""
    image = ee.Image(asset_id)
    built = image.select(
        [f"built_{method}" for method in methods]
    )
    valid = image.select(
        [f"valid_{method}" for method in methods]
    )
    common_valid = (
        valid.reduce(ee.Reducer.min())
        .eq(1)
        .rename("common_valid")
    )
    vote = (
        built.reduce(ee.Reducer.sum())
        .rename("candidate_vote_count")
    )
    stratum = (
        ee.Image.constant(STRATUM_CODES["disagreement"])
        .where(vote.eq(0), STRATUM_CODES["unanimous_nonbuilt"])
        .where(
            vote.eq(len(methods)),
            STRATUM_CODES["unanimous_built"],
        )
        .rename("stratum")
        .toUint8()
    )

    return (
        image.select(expected_candidate_bands(methods))
        .addBands(vote)
        .addBands(stratum)
        .updateMask(common_valid)
    )


def stratum_populations(
    sample_image: ee.Image,
    core: ee.Geometry,
    grid: dict[str, Any],
) -> dict[int, int]:
    """Count common-valid cells in each sampling stratum."""
    values = sample_image.select("stratum").reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=core,
        crs=str(grid["crs"]),
        crsTransform=[
            float(value)
            for value in grid["transform"]
        ],
        maxPixels=10_000_000,
        tileScale=4,
    ).getInfo()
    histogram = values.get("stratum", {})

    return {
        code: int(round(float(histogram.get(str(code), 0))))
        for code in STRATUM_CODES.values()
    }


def cell_indices(
    x_m: float,
    y_m: float,
    grid: dict[str, Any],
) -> tuple[int, int, int]:
    """Return row, column and stable cell ID on the frozen grid."""
    xmin = float(grid["extent"]["xmin"])
    ymax = float(grid["extent"]["ymax"])
    resolution = float(grid["resolution_m"])
    column = int(math.floor((x_m - xmin) / resolution))
    row = int(math.floor((ymax - y_m) / resolution))

    if not (
        0 <= column < int(grid["width"])
        and 0 <= row < int(grid["height"])
    ):
        raise ValueError(
            f"Sample coordinate lies outside the frozen grid: "
            f"{x_m}, {y_m}"
        )

    cell_id = row * int(grid["width"]) + column
    return row, column, cell_id


def deterministic_key(
    epoch: int,
    stratum: int,
    cell_id: int,
    seed: int,
) -> str:
    """Create a deterministic order key for separation filtering."""
    payload = f"{seed}:{epoch}:{stratum}:{cell_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def greedy_separation(
    pool: gpd.GeoDataFrame,
    target: int,
    preferred_distance_m: float,
) -> tuple[gpd.GeoDataFrame, float]:
    """Select points using progressively relaxed spatial separation."""
    if pool.empty or target <= 0:
        return pool.iloc[0:0].copy(), preferred_distance_m

    selected_indices: list[int] = []
    selected_geometries: list[Any] = []
    distances = [
        float(preferred_distance_m),
        float(preferred_distance_m) * 2 / 3,
        float(preferred_distance_m) / 3,
        0.0,
    ]
    used_distance = distances[-1]

    for distance in distances:
        used_distance = distance

        for index, row in pool.iterrows():
            if index in selected_indices:
                continue

            geometry = row.geometry

            if all(
                geometry.distance(existing) >= distance
                for existing in selected_geometries
            ):
                selected_indices.append(index)
                selected_geometries.append(geometry)

            if len(selected_indices) >= target:
                break

        if len(selected_indices) >= target:
            break

    selected = pool.loc[selected_indices].copy()
    return selected.iloc[:target], used_distance


def sample_epoch(
    *,
    epoch: int,
    asset_id: str,
    anchor: AnchorYear,
    methods: list[str],
    targets: dict[str, int],
    config: dict[str, Any],
    grid: dict[str, Any],
    core: ee.Geometry,
) -> gpd.GeoDataFrame:
    """Generate one reproducible, separated three-stratum epoch sample."""
    sample_image = candidate_vote_image(asset_id, methods)
    populations = stratum_populations(
        sample_image,
        core,
        grid,
    )
    oversampling_factor = int(
        config["sampling"]["oversampling_factor"]
    )
    class_values = []
    class_points = []

    for name, code in STRATUM_CODES.items():
        target = int(targets[name])
        available = populations[code]
        request = min(
            available,
            max(target, target * oversampling_factor),
        )
        class_values.append(code)
        class_points.append(request)

    projection = ee.Projection(
        str(grid["crs"]),
        [float(value) for value in grid["transform"]],
    )
    collection = sample_image.stratifiedSample(
        numPoints=0,
        classBand="stratum",
        region=core,
        projection=projection,
        seed=int(config["sampling"]["random_seed"]) + epoch,
        classValues=class_values,
        classPoints=class_points,
        dropNulls=True,
        tileScale=4,
        geometries=True,
    )
    features = collection.getInfo().get("features", [])
    records: list[dict[str, Any]] = []

    for feature in features:
        properties = dict(feature.get("properties", {}))
        coordinates = feature["geometry"]["coordinates"]
        records.append(
            {
                **properties,
                "longitude": float(coordinates[0]),
                "latitude": float(coordinates[1]),
                "geometry": Point(
                    float(coordinates[0]),
                    float(coordinates[1]),
                ),
            }
        )

    if not records:
        raise RuntimeError(
            f"No validation sample could be generated for {epoch}."
        )

    frame = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    ).to_crs(str(grid["crs"]))
    frame["x_m"] = frame.geometry.x
    frame["y_m"] = frame.geometry.y
    rows = []
    columns = []
    cell_ids = []

    for row in frame.itertuples(index=False):
        grid_row, grid_column, cell_id = cell_indices(
            float(row.x_m),
            float(row.y_m),
            grid,
        )
        rows.append(grid_row)
        columns.append(grid_column)
        cell_ids.append(cell_id)

    frame["grid_row"] = rows
    frame["grid_column"] = columns
    frame["cell_id"] = cell_ids
    frame["epoch"] = int(epoch)
    frame["source_core_coverage_pct"] = anchor.core_coverage_pct
    frame["source_quality_flag"] = anchor.quality_flag
    frame["coverage_source"] = anchor.coverage_source
    frame["deterministic_order"] = [
        deterministic_key(
            epoch,
            int(stratum),
            int(cell_id),
            int(config["sampling"]["random_seed"]),
        )
        for stratum, cell_id in zip(
            frame["stratum"],
            frame["cell_id"],
        )
    ]
    frame = frame.sort_values(
        "deterministic_order"
    ).reset_index(drop=True)
    name_by_code = {
        code: name
        for name, code in STRATUM_CODES.items()
    }
    selected_frames: list[gpd.GeoDataFrame] = []

    for code, group in frame.groupby("stratum"):
        stratum_name = name_by_code[int(code)]
        target = int(targets[stratum_name])
        selected, actual_distance = greedy_separation(
            group,
            target,
            float(config["sampling"]["minimum_separation_m"]),
        )
        sample_size = len(selected)
        population = populations[int(code)]

        if sample_size == 0:
            raise RuntimeError(
                f"Stratum '{stratum_name}' has no sample for {epoch}."
            )

        selected["stratum_name"] = stratum_name
        selected["stratum_population"] = population
        selected["stratum_sample_size"] = sample_size
        selected["sample_weight"] = population / sample_size
        selected["actual_minimum_separation_m"] = actual_distance
        selected_frames.append(selected)

    result = pd.concat(selected_frames, ignore_index=True)
    result = gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=str(grid["crs"]),
    ).sort_values(
        ["epoch", "stratum", "deterministic_order"]
    )
    result["sample_id"] = [
        f"{epoch}_{int(cell_id):09d}"
        for cell_id in result["cell_id"]
    ]
    return result.reset_index(drop=True)


def attach_ghsl_values(
    samples: gpd.GeoDataFrame,
    manifest_path: Path,
) -> gpd.GeoDataFrame:
    """Attach GHSL built fractions as sample-level diagnostics."""
    result = samples.copy()
    result["ghsl_built_fraction"] = np.nan

    try:
        manifest = pd.read_csv(manifest_path)
    except Exception:
        return result

    required = {"epoch", "asset_id", "band"}

    if not required.issubset(manifest.columns):
        return result

    if "source" in manifest.columns:
        manifest = manifest[
            manifest["source"]
            .astype(str)
            .str.contains("BUILT", case=False, na=False)
        ]

    geographic = result.to_crs("EPSG:4326")

    for epoch, group in geographic.groupby("epoch"):
        rows = manifest[
            manifest["epoch"].astype(int) == int(epoch)
        ]

        if rows.empty:
            continue

        source = rows.iloc[0]
        image = (
            ee.Image(str(source["asset_id"]))
            .select(str(source["band"]))
            .divide(10_000)
            .rename("ghsl_built_fraction")
            .unmask(-9999)
        )
        features = [
            ee.Feature(
                ee.Geometry.Point(
                    [
                        float(row.geometry.x),
                        float(row.geometry.y),
                    ]
                ),
                {"sample_id": str(row.sample_id)},
            )
            for row in group.itertuples(index=False)
        ]

        try:
            sampled = image.sampleRegions(
                collection=ee.FeatureCollection(features),
                scale=100,
                geometries=False,
            ).getInfo()
        except Exception:
            continue

        values = {}

        for feature in sampled.get("features", []):
            properties = feature["properties"]
            value = properties.get("ghsl_built_fraction")

            if value is not None and float(value) > -9998:
                values[str(properties["sample_id"])] = float(value)

        mask = result["sample_id"].isin(values)
        result.loc[mask, "ghsl_built_fraction"] = (
            result.loc[mask, "sample_id"].map(values)
        )

    return result


def attach_osm_building_presence(
    samples: gpd.GeoDataFrame,
    osm_path: Path,
    latest_anchor_year: int,
) -> gpd.GeoDataFrame:
    """Attach current OSM-building support to latest-year samples only."""
    result = samples.copy()
    result["osm_building_present"] = pd.NA
    latest = result[
        result["epoch"].astype(int) == int(latest_anchor_year)
    ].copy()

    if latest.empty:
        return result

    if pyogrio is None:
        return result

    try:
        layers = pyogrio.list_layers(osm_path)
        layer_names = [str(value) for value in layers[:, 0]]
        building_layers = [
            name
            for name in layer_names
            if "building" in name.lower()
        ]

        if not building_layers:
            return result

        points = latest.to_crs("EPSG:32632")
        bounds = points.geometry.buffer(30).total_bounds
        buildings = gpd.read_file(
            osm_path,
            layer=building_layers[0],
            bbox=tuple(bounds),
            engine="pyogrio",
        )

        if buildings.empty:
            result.loc[
                result["epoch"].astype(int) == latest_anchor_year,
                "osm_building_present",
            ] = 0
            return result

        buildings = buildings.to_crs(points.crs)
        cells = points[["sample_id", "geometry"]].copy()
        cells["geometry"] = cells.geometry.buffer(
            15,
            cap_style="square",
        )
        joined = gpd.sjoin(
            cells,
            buildings[["geometry"]],
            how="left",
            predicate="intersects",
        )
        present_ids = set(
            joined.loc[
                joined["index_right"].notna(),
                "sample_id",
            ].astype(str)
        )
        mask = result["epoch"].astype(int) == latest_anchor_year
        result.loc[mask, "osm_building_present"] = (
            result.loc[mask, "sample_id"]
            .astype(str)
            .isin(present_ids)
            .astype(int)
        )
    except Exception:
        return result

    return result


def write_manual_template(
    samples: gpd.GeoDataFrame,
    path: Path,
) -> None:
    """Create a blind review template without method predictions."""
    geographic = samples.to_crs("EPSG:4326")
    template = pd.DataFrame(
        {
            "sample_id": geographic["sample_id"],
            "epoch": geographic["epoch"].astype(int),
            "longitude": geographic.geometry.x,
            "latitude": geographic.geometry.y,
            "manual_label": "",
            "manual_confidence": "",
            "reference_source": "",
            "reference_image_date": "",
            "review_notes": "",
            "reviewer": "",
        }
    )

    if path.is_file():
        existing = pd.read_csv(path, keep_default_na=False)

        if (
            "manual_label" in existing.columns
            and existing["manual_label"]
            .astype(str)
            .str.strip()
            .ne("")
            .any()
        ):
            raise FileExistsError(
                "manual_labels.csv already contains labels and was not "
                "overwritten."
            )

    template.to_csv(path, index=False)


def generate_samples(config_path: Path) -> dict[str, Any]:
    """Run preflight and produce the reproducible blind-review sample."""
    preflight = run_preflight(config_path)
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    paths = load_paths(config, project_root)
    outputs = output_paths(config, project_root)
    methods = list(config["candidate_methods"])
    grid = load_grid_specification(paths["grid_specification"])
    candidates = candidate_manifest(
        paths["output_manifest"],
        methods,
    )
    coverage = coverage_table(
        paths["orchestration_quality"],
        paths["candidate_area_summary"],
        paths["core_boundary"],
        set(candidates["epoch"].astype(int)),
    )
    anchors = select_anchor_years(
        candidates,
        coverage,
        config,
    )
    _, core = load_core_geometry(paths["core_boundary"])
    targets = {
        name: int(value)
        for name, value in config["sampling"][
            "per_anchor_year"
        ].items()
    }
    frames = []

    for anchor in anchors:
        asset_id = str(
            candidates.loc[
                candidates["epoch"] == anchor.epoch,
                "asset_id",
            ].iloc[0]
        )
        frames.append(
            sample_epoch(
                epoch=anchor.epoch,
                asset_id=asset_id,
                anchor=anchor,
                methods=methods,
                targets=targets,
                config=config,
                grid=grid,
                core=core,
            )
        )

    samples = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )
    samples = attach_ghsl_values(
        samples,
        paths["ghsl_manifest"],
    )
    samples = attach_osm_building_presence(
        samples,
        paths["osm_extract"],
        max(anchor.epoch for anchor in anchors),
    )
    samples.to_file(
        outputs["validation_samples"],
        layer="validation_samples",
        driver="GPKG",
    )
    write_manual_template(
        samples,
        outputs["manual_labels"],
    )

    summary = {
        "status": "PASS",
        "anchor_years": [anchor.epoch for anchor in anchors],
        "sample_count": int(len(samples)),
        "sample_counts_by_year": {
            str(key): int(value)
            for key, value in samples[
                "epoch"
            ].value_counts().sort_index().items()
        },
        "sample_counts_by_stratum": {
            str(key): int(value)
            for key, value in samples[
                "stratum_name"
            ].value_counts().items()
        },
        "validation_samples": str(
            outputs["validation_samples"]
        ),
        "manual_labels": str(outputs["manual_labels"]),
        "preflight": preflight,
    }
    print(json.dumps(summary, indent=2))
    return summary


def validate_manual_labels(
    labels: pd.DataFrame,
    samples: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate labels, confidence, reference dates and sample membership."""
    required = {
        "sample_id",
        "epoch",
        "manual_label",
        "manual_confidence",
        "reference_source",
        "reference_image_date",
    }
    missing = required.difference(labels.columns)

    if missing:
        raise ValueError(
            f"Manual label table is missing columns: {sorted(missing)}"
        )

    if not labels["sample_id"].is_unique:
        raise ValueError("Manual label sample IDs are not unique.")

    expected_ids = set(samples["sample_id"].astype(str))
    observed_ids = set(labels["sample_id"].astype(str))

    if expected_ids != observed_ids:
        raise ValueError(
            "Manual label sample IDs do not exactly match the generated "
            "sample."
        )

    result = labels.copy()
    result["manual_label"] = pd.to_numeric(
        result["manual_label"],
        errors="raise",
    ).astype(int)
    allowed_labels = {
        int(value)
        for value in config["manual_review"]["labels"].values()
    }

    if not set(result["manual_label"]).issubset(allowed_labels):
        raise ValueError(
            f"Manual labels must belong to {sorted(allowed_labels)}."
        )

    result["manual_confidence"] = (
        result["manual_confidence"].astype(str).str.upper()
    )
    allowed_confidence = set(
        config["manual_review"]["allowed_confidence"]
    )

    if not set(result["manual_confidence"]).issubset(
        allowed_confidence
    ):
        raise ValueError(
            f"Confidence must belong to {sorted(allowed_confidence)}."
        )

    certain = result["manual_label"].isin([0, 1])

    if result.loc[
        certain,
        "reference_source",
    ].astype(str).str.strip().eq("").any():
        raise ValueError(
            "Labels 0 and 1 require a reference source."
        )

    dates = pd.to_datetime(
        result.loc[certain, "reference_image_date"],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(
            "Labels 0 and 1 require a valid reference image date."
        )

    offsets = (
        dates.dt.year.to_numpy()
        - result.loc[certain, "epoch"].astype(int).to_numpy()
    )
    maximum_offset = int(
        config["manual_review"]["maximum_reference_offset_years"]
    )

    if len(offsets) and np.abs(offsets).max() > maximum_offset:
        raise ValueError(
            "A certain manual label uses reference imagery too far from "
            "its target epoch."
        )

    return result


def weighted_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Calculate binary design-weighted F1."""
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    weights = np.asarray(weights, dtype=float)

    tp = weights[(truth == 1) & (prediction == 1)].sum()
    fp = weights[(truth == 0) & (prediction == 1)].sum()
    fn = weights[(truth == 1) & (prediction == 0)].sum()
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = (
        tp / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall = (
        tp / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    if precision + recall == 0:
        return 0.0

    return float(
        2 * precision * recall / (precision + recall)
    )


def validate_evaluation_population(
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Require enough certain examples and both manual classes."""
    certain = dataset[dataset["manual_label"].isin([0, 1])]
    minimum_per_year = int(
        config["manual_review"]["minimum_reviewed_per_anchor_year"]
    )

    for epoch, group in certain.groupby("epoch"):
        if len(group) < minimum_per_year:
            raise ValueError(
                f"Anchor year {epoch} has only {len(group)} certain labels."
            )

        if set(group["manual_label"].astype(int)) != {0, 1}:
            raise ValueError(
                f"Anchor year {epoch} does not contain both manual classes."
            )

    minimum_pooled = int(
        config["manual_review"]["minimum_pooled_examples_per_class"]
    )
    counts = certain["manual_label"].value_counts()

    for label in [0, 1]:
        if int(counts.get(label, 0)) < minimum_pooled:
            raise ValueError(
                f"Manual class {label} has insufficient pooled examples."
            )


def reversal_rates(
    candidates: pd.DataFrame,
    methods: list[str],
    core: ee.Geometry,
    grid: dict[str, Any],
) -> dict[str, float]:
    """Calculate one common-valid aggregated reversal rate per method."""
    candidates = candidates.sort_values("epoch")
    epochs = candidates["epoch"].astype(int).tolist()
    totals = {
        method: {"built_start": 0.0, "reversal": 0.0}
        for method in methods
    }

    for start_epoch, end_epoch in zip(epochs[:-1], epochs[1:]):
        start_asset = str(
            candidates.loc[
                candidates["epoch"] == start_epoch,
                "asset_id",
            ].iloc[0]
        )
        end_asset = str(
            candidates.loc[
                candidates["epoch"] == end_epoch,
                "asset_id",
            ].iloc[0]
        )
        start = ee.Image(start_asset)
        end = ee.Image(end_asset)
        bands = []

        for method in methods:
            start_built = start.select(f"built_{method}")
            end_built = end.select(f"built_{method}")
            common_valid = (
                start.select(f"valid_{method}").eq(1)
                .And(end.select(f"valid_{method}").eq(1))
            )
            built_start = (
                ee.Image.constant(1)
                .updateMask(common_valid.And(start_built.eq(1)))
                .rename(f"{method}_built_start")
            )
            reversal = (
                ee.Image.constant(1)
                .updateMask(
                    common_valid
                    .And(start_built.eq(1))
                    .And(end_built.eq(0))
                )
                .rename(f"{method}_reversal")
            )
            bands.extend([built_start, reversal])

        values = ee.Image.cat(bands).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=core,
            crs=str(grid["crs"]),
            crsTransform=[
                float(value)
                for value in grid["transform"]
            ],
            maxPixels=10_000_000,
            tileScale=4,
        ).getInfo()

        for method in methods:
            totals[method]["built_start"] += float(
                values.get(f"{method}_built_start", 0) or 0
            )
            totals[method]["reversal"] += float(
                values.get(f"{method}_reversal", 0) or 0
            )

    rates = {}

    for method, values in totals.items():
        denominator = values["built_start"]
        rates[method] = (
            values["reversal"] / denominator
            if denominator > 0
            else math.nan
        )

    return rates


def select_method(
    scores: pd.DataFrame,
    tie_tolerance: float,
) -> tuple[str, str | None]:
    """Apply mean F1, minimum F1 and reversal-rate tie-breaks."""
    required = {
        "method",
        "mean_yearly_weighted_f1",
        "minimum_yearly_weighted_f1",
        "reversal_rate",
    }
    missing = required.difference(scores.columns)

    if missing:
        raise ValueError(
            f"Method scores are missing columns: {sorted(missing)}"
        )

    ordered = scores.sort_values(
        [
            "mean_yearly_weighted_f1",
            "minimum_yearly_weighted_f1",
            "reversal_rate",
            "method",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    best_mean = float(
        ordered.iloc[0]["mean_yearly_weighted_f1"]
    )
    tied = ordered[
        ordered["mean_yearly_weighted_f1"]
        >= best_mean - float(tie_tolerance)
    ].copy()
    best_minimum = float(
        tied["minimum_yearly_weighted_f1"].max()
    )
    tied = tied[
        np.isclose(
            tied["minimum_yearly_weighted_f1"],
            best_minimum,
            atol=1e-12,
        )
    ]
    best_reversal = float(tied["reversal_rate"].min())
    tied = tied[
        np.isclose(
            tied["reversal_rate"],
            best_reversal,
            atol=1e-12,
        )
    ]

    if len(tied) != 1:
        return "MANUAL_DECISION_REQUIRED", None

    return "PASS", str(tied.iloc[0]["method"])


def diagnostic_lines(
    dataset: pd.DataFrame,
    methods: list[str],
    latest_anchor: int,
) -> list[str]:
    """Create concise GHSL and OSM sample-level diagnostics."""
    lines = ["## External diagnostics", ""]

    if dataset["ghsl_built_fraction"].notna().any():
        lines.extend(
            [
                "GHSL is used only as a sample-level diagnostic.",
                "",
                "| Method | GHSL mean where predicted built | "
                "GHSL mean where predicted non-built |",
                "|---|---:|---:|",
            ]
        )

        for method in methods:
            prediction = dataset[f"pred_{method}"]
            built_mean = dataset.loc[
                prediction == 1,
                "ghsl_built_fraction",
            ].mean()
            nonbuilt_mean = dataset.loc[
                prediction == 0,
                "ghsl_built_fraction",
            ].mean()
            lines.append(
                f"| {method} | {built_mean:.4f} | "
                f"{nonbuilt_mean:.4f} |"
            )
    else:
        lines.append(
            "GHSL values were unavailable and did not affect selection."
        )

    lines.extend(["", "### Current OSM", ""])
    latest = dataset[
        dataset["epoch"].astype(int) == int(latest_anchor)
    ]

    if latest["osm_building_present"].notna().any():
        osm = latest[latest["osm_building_present"] == 1]
        lines.append(
            "Sample-level recall among current OSM-building cells:"
        )
        lines.append("")

        for method in methods:
            recall = (
                osm[f"pred_{method}"].mean()
                if not osm.empty
                else math.nan
            )
            lines.append(f"- {method}: {recall:.4f}")
    else:
        lines.append(
            "Current OSM building support was unavailable."
        )

    lines.extend(
        [
            "",
            "Current OSM completeness is uneven, and the snapshot is not "
            "historical evidence for earlier epochs.",
        ]
    )
    return lines


def evaluate(config_path: Path) -> dict[str, Any]:
    """Merge manual labels, calculate minimal scores and choose a method."""
    require_earth_engine()
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    paths = load_paths(config, project_root)
    outputs = output_paths(config, project_root)

    if not outputs["validation_samples"].is_file():
        raise FileNotFoundError(
            "Generate validation samples before evaluation."
        )

    if not outputs["manual_labels"].is_file():
        raise FileNotFoundError(
            "Complete manual_labels.csv before evaluation."
        )

    initialize_earth_engine(
        config["project"]["earth_engine_project"]
    )
    methods = list(config["candidate_methods"])
    samples = gpd.read_file(
        outputs["validation_samples"],
        layer="validation_samples",
    )
    labels = pd.read_csv(
        outputs["manual_labels"],
        keep_default_na=False,
    )
    labels = validate_manual_labels(
        labels,
        samples,
        config,
    )
    dataset = samples.drop(columns="geometry").merge(
        labels.drop(columns=["epoch"]),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    validate_evaluation_population(dataset, config)
    dataset.to_parquet(
        outputs["validation_dataset"],
        index=False,
    )
    certain = dataset[
        dataset["manual_label"].isin([0, 1])
    ].copy()
    yearly_rows = []

    for method in methods:
        prediction_column = f"pred_{method}"

        if prediction_column not in certain.columns:
            raise ValueError(
                f"Missing prediction column: {prediction_column}"
            )

        for epoch, group in certain.groupby("epoch"):
            yearly_rows.append(
                {
                    "method": method,
                    "epoch": int(epoch),
                    "weighted_f1": weighted_f1(
                        group["manual_label"].to_numpy(),
                        group[prediction_column].to_numpy(),
                        group["sample_weight"].to_numpy(),
                    ),
                }
            )

    yearly = pd.DataFrame(yearly_rows)
    grid = load_grid_specification(paths["grid_specification"])
    candidates = candidate_manifest(
        paths["output_manifest"],
        methods,
    )
    _, core = load_core_geometry(paths["core_boundary"])
    reversals = reversal_rates(
        candidates,
        methods,
        core,
        grid,
    )
    uncertain_count = int(
        (dataset["manual_label"] == -1).sum()
    )
    reviewed_count = int(len(certain))
    score_rows = []

    for method in methods:
        values = yearly[
            yearly["method"] == method
        ]["weighted_f1"]
        score_rows.append(
            {
                "method": method,
                "mean_yearly_weighted_f1": float(values.mean()),
                "minimum_yearly_weighted_f1": float(values.min()),
                "reversal_rate": float(reversals[method]),
                "reviewed_samples": reviewed_count,
                "uncertain_samples": uncertain_count,
            }
        )

    scores = pd.DataFrame(score_rows)

    if not np.isfinite(scores["reversal_rate"]).all():
        raise ValueError(
            "At least one method has an undefined reversal rate."
        )

    status, selected = select_method(
        scores,
        float(config["selection"]["tie_tolerance"]),
    )
    scores = scores.sort_values(
        [
            "mean_yearly_weighted_f1",
            "minimum_yearly_weighted_f1",
            "reversal_rate",
            "method",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    scores["rank"] = np.arange(1, len(scores) + 1)
    scores["selected"] = (
        scores["method"] == selected
        if selected is not None
        else False
    )
    scores.to_csv(outputs["method_scores"], index=False)

    report_lines = [
        "# Day 5 — Minimal method selection",
        "",
        f"Selection status: **{status}**",
        "",
        f"Selected method: **{selected or 'none'}**",
        "",
        "## Anchor-year weighted F1",
        "",
        "| Epoch | Method | Weighted F1 |",
        "|---:|---|---:|",
    ]

    for row in yearly.sort_values(
        ["epoch", "method"]
    ).itertuples(index=False):
        report_lines.append(
            f"| {row.epoch} | {row.method} | "
            f"{row.weighted_f1:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Final ranking",
            "",
            "| Rank | Method | Mean yearly weighted F1 | "
            "Minimum yearly weighted F1 | Reversal rate | Selected |",
            "|---:|---|---:|---:|---:|:---:|",
        ]
    )

    for row in scores.itertuples(index=False):
        report_lines.append(
            f"| {row.rank} | {row.method} | "
            f"{row.mean_yearly_weighted_f1:.4f} | "
            f"{row.minimum_yearly_weighted_f1:.4f} | "
            f"{row.reversal_rate:.4f} | "
            f"{str(bool(row.selected)).lower()} |"
        )

    report_lines.extend(
        [
            "",
            *diagnostic_lines(
                dataset,
                methods,
                int(dataset["epoch"].max()),
            ),
            "",
            "## Limitations",
            "",
            "- Metrics apply only to valid Landsat spatial support.",
            "- Historical reference-image quality varies by epoch.",
            "- GHSL is an auxiliary benchmark rather than ground truth.",
            "- Current OSM is not historical evidence.",
            "- Temporal correction is deferred to Day 6.",
        ]
    )
    outputs["report"].write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    result = {
        "status": status,
        "selected_method": selected,
        "method_scores": str(outputs["method_scores"]),
        "validation_dataset": str(
            outputs["validation_dataset"]
        ),
        "report": str(outputs["report"]),
    }
    print(json.dumps(result, indent=2))
    return result


def freeze(config_path: Path) -> dict[str, Any]:
    """Write the compact protocol consumed by Day 6 and freeze its version."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    paths = load_paths(config, project_root)
    outputs = output_paths(config, project_root)

    for path in (
        outputs["method_scores"],
        outputs["validation_dataset"],
        outputs["report"],
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Evaluate methods before freezing: {path}"
            )

    methods = list(config["candidate_methods"])
    scores = pd.read_csv(outputs["method_scores"])
    selected_rows = scores[
        scores["selected"].astype(str).str.lower() == "true"
    ]

    if len(selected_rows) == 1:
        status = "PASS"
        selected = str(selected_rows.iloc[0]["method"])
        evidence = selected_rows.iloc[0]
    else:
        status = config["selection"]["unresolved_tie"]
        selected = None
        evidence = None

    samples = pd.read_parquet(outputs["validation_dataset"])
    candidates = candidate_manifest(
        paths["output_manifest"],
        methods,
    )
    anchor_years = sorted(
        samples["epoch"].astype(int).unique().tolist()
    )
    completed_epochs = sorted(
        candidates["epoch"].astype(int).tolist()
    )
    protocol: dict[str, Any] = {
        "version": int(config["version"]),
        "selection_status": status,
        "selected_method": (
            {
                "index": selected,
                "candidate_band": f"built_{selected}",
                "validity_band": f"valid_{selected}",
            }
            if selected is not None
            else None
        ),
        "candidate_source": {
            "output_manifest": str(
                config["inputs"]["output_manifest"]
            ),
            "threshold_table": str(
                config["inputs"]["threshold_table"]
            ),
            "epochs": completed_epochs,
        },
        "classification": {
            "threshold_method": "epoch_specific_otsu",
            "morphological_filtering": "none",
            "temporal_correction": "deferred_to_day6",
        },
        "selection_evidence": (
            {
                "mean_yearly_weighted_f1": float(
                    evidence["mean_yearly_weighted_f1"]
                ),
                "minimum_yearly_weighted_f1": float(
                    evidence["minimum_yearly_weighted_f1"]
                ),
                "reversal_rate": float(
                    evidence["reversal_rate"]
                ),
            }
            if evidence is not None
            else None
        ),
        "validation": {
            "anchor_years": anchor_years,
            "reviewed_samples": int(
                samples["manual_label"].isin([0, 1]).sum()
            ),
            "uncertain_samples": int(
                (samples["manual_label"] == -1).sum()
            ),
            "random_seed": int(
                config["sampling"]["random_seed"]
            ),
        },
        "excluded_methods": config["excluded_methods"],
        "limitations": [
            "Metrics apply only to valid Landsat spatial support.",
            "Historical reference-image quality varies by epoch.",
            "GHSL is an auxiliary benchmark rather than ground truth.",
            "Current OSM is not historical evidence.",
            "Temporal correction is deferred to Day 6.",
        ],
        "source_versions": {
            "built_up_candidates_sha256": sha256_file(
                paths["built_up_version"]
            ),
            "grid_specification_sha256": sha256_file(
                paths["grid_specification"]
            ),
        },
    }
    protocol["recipe_sha256"] = stable_object_hash(protocol)
    write_yaml(outputs["selected_protocol"], protocol)

    version_payload = {
        "pipeline_stage": "method_selection",
        "pipeline_version": int(config["version"]),
        "selection_status": status,
        "selected_method": selected,
        "method_scores_sha256": sha256_file(
            outputs["method_scores"]
        ),
        "validation_dataset_sha256": sha256_file(
            outputs["validation_dataset"]
        ),
        "selected_protocol_sha256": sha256_file(
            outputs["selected_protocol"]
        ),
        "report_sha256": sha256_file(outputs["report"]),
        "source_built_up_version_sha256": sha256_file(
            paths["built_up_version"]
        ),
    }
    version_payload["stable_signature"] = stable_object_hash(
        version_payload
    )
    write_json(outputs["version"], version_payload)

    result = {
        "selection_status": status,
        "selected_method": selected,
        "protocol": str(outputs["selected_protocol"]),
        "version": str(outputs["version"]),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the single Day 5 command interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate Day 4 candidates and freeze one mapping method."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--generate-samples", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--freeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected Day 5 operation."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()

    if arguments.preflight:
        run_preflight(config_path)
    elif arguments.generate_samples:
        generate_samples(config_path)
    elif arguments.evaluate:
        evaluate(config_path)
    elif arguments.freeze:
        freeze(config_path)


if __name__ == "__main__":
    main()
