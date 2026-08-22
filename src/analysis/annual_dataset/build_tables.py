"""Build and assemble annual eligible-cell modelling tables."""

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
    annual_transitions,
    annual_years,
    build_transition_image,
    population_source_year,
    validate_annual_config,
)
from src.analysis.built_up.build_candidates import validate_source_asset
from src.analysis.final_dataset.build_tables import (
    coordinate_bands,
    distance_to_built,
    neighbourhood_features,
    population_density,
    projection_from_grid,
    repair_coordinate_columns,
)
from src.analysis.orchestration.common import (
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_yaml,
    resolve_project_path,
)

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
LABEL_COLUMNS = ["target_transition_1y"]
PROVENANCE_COLUMNS = ["common_valid_1y"]
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
    "ndbi_t",
    "ibui_t",
    "ndbsui_t",
    "savi_t",
    "mndwi_t",
]
URBAN_CONTEXT_COLUMNS = [
    "built_fraction_3x3_t",
    "built_fraction_5x5_t",
    "built_fraction_11x11_t",
    "distance_to_built_m_t",
    "recent_local_growth_1y_t",
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
    + PROVENANCE_COLUMNS
    + STATE_COLUMNS
    + SPECTRAL_COLUMNS
    + URBAN_CONTEXT_COLUMNS
    + STATIC_COLUMNS
)
PREDICTOR_COLUMNS = STATE_COLUMNS + SPECTRAL_COLUMNS + URBAN_CONTEXT_COLUMNS + STATIC_COLUMNS


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError("The Earth Engine Python API is required for annual tables.")


def assert_annual_schema_has_no_leakage(columns: list[str]) -> None:
    """Reject five-year fields and target-year predictor naming."""
    forbidden_exact = {"target_transition_5y", "recent_local_growth_5y_t"}
    overlap = forbidden_exact.intersection(columns)
    if overlap:
        raise ValueError(f"Five-year fields are forbidden: {sorted(overlap)}")
    for column in PREDICTOR_COLUMNS:
        lowered = column.casefold()
        if "target" in lowered or lowered.endswith("_t1"):
            raise ValueError(f"Potential target-year predictor: {column}")


def load_state_assets(path: Path, years: list[int]) -> dict[int, str]:
    """Load one passed annual state asset for every configured year."""
    if not path.is_file():
        raise FileNotFoundError("Finalize annual state products before building tables.")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"product_type", "year", "asset_id", "state"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Annual state manifest is missing {sorted(missing)}.")
    frame = frame[frame["product_type"] == "annual_state"].copy()
    frame["year"] = frame["year"].astype(int)
    frame = frame[frame["state"].astype(str).str.upper().isin(PASS_STATES)]
    result = {int(row.year): str(row.asset_id) for row in frame.itertuples(index=False)}
    if sorted(result) != years or len(frame) != len(years):
        raise ValueError("Annual state manifest must contain exactly 26 passed states.")
    return result


def load_composite_assets(path: Path, years: list[int]) -> dict[int, str]:
    """Load origin-year reflectance assets from the finalized Day 2 manifest."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"epoch", "asset_id", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Day 2 composite manifest is missing {sorted(missing)}.")
    frame["epoch"] = frame["epoch"].astype(int)
    frame = frame[
        frame["epoch"].isin(years)
        & frame["status"].astype(str).str.upper().eq("PASS")
    ]
    if frame["epoch"].tolist() != years or not frame["epoch"].is_unique:
        raise ValueError("Day 2 manifest must contain one passed composite per annual year.")
    expected_root = "projects/urban-sprawl-ssa/assets/sprawlgp/annual_v1/landsat"
    result = {int(row.epoch): str(row.asset_id) for row in frame.itertuples(index=False)}
    for year, asset_id in result.items():
        if asset_id != f"{expected_root}/composite_{year}":
            raise ValueError(f"Unexpected annual composite source: {asset_id}")
    return result


def prior_growth_transition(
    origin: int, years: list[int]
) -> tuple[int, int] | None:
    """Return the transition ending at an origin, or none for the first year."""
    if origin not in years:
        raise ValueError(f"Forecast origin {origin} is not an annual state year.")
    index = years.index(origin)
    if index == 0:
        return None
    previous = years[index - 1]
    if previous != origin - 1:
        raise ValueError("Recent annual growth requires a consecutive prior state.")
    return previous, origin


def load_core_geometry(path: Path):
    """Load the administrative core as an Earth Engine geometry."""
    require_earth_engine()
    frame = gpd.read_file(path)
    if frame.crs is None or frame.empty:
        raise ValueError("The administrative core must be non-empty and georeferenced.")
    geometry = frame.to_crs("EPSG:4326").geometry.union_all()
    return ee.Geometry(geometry.__geo_interface__)


def recent_growth_features(previous_transition: Any | None, window_size: int):
    """Calculate local growth using only the transition ending at the origin."""
    require_earth_engine()
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("The recent-growth window must be positive and odd.")
    if previous_transition is None:
        return ee.Image.cat(
            [
                ee.Image.constant(0).rename("recent_local_growth_1y_t").toFloat(),
                ee.Image.constant(0)
                .rename("recent_local_growth_available_t")
                .toUint8(),
            ]
        )
    common = previous_transition.select("common_valid_1y").eq(1).unmask(0).toFloat()
    growth = (
        previous_transition.select("target_transition_1y")
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
        reducer=ee.Reducer.sum(), kernel=kernel, skipMasked=False
    )
    growth_count = growth.reduceNeighborhood(
        reducer=ee.Reducer.sum(), kernel=kernel, skipMasked=False
    )
    fraction = (
        growth_count.divide(valid_count.max(1))
        .where(valid_count.eq(0), 0)
        .rename("recent_local_growth_1y_t")
        .toFloat()
    )
    available = valid_count.gt(0).rename("recent_local_growth_available_t").toUint8()
    return ee.Image.cat([fraction, available])


def build_feature_image(
    *,
    transition_order: int,
    origin: int,
    target: int,
    origin_state_asset: str,
    target_state_asset: str,
    previous_state_asset: str | None,
    composite_asset: str,
    terrain_asset: str,
    grid: dict[str, Any],
    config: dict[str, Any],
):
    """Assemble origin-only predictors and one adjacent-year transition label."""
    require_earth_engine()
    if target != origin + 1:
        raise ValueError("Annual feature targets must equal origin + 1.")
    origin_state = ee.Image(origin_state_asset)
    target_state = ee.Image(target_state_asset)
    transition = build_transition_image(origin_state, target_state)
    previous_transition = None
    if previous_state_asset is not None:
        previous_transition = build_transition_image(
            ee.Image(previous_state_asset), origin_state
        )
    time = ee.Image.cat(
        [
            ee.Image.constant(transition_order).rename("transition_order").toInt16(),
            ee.Image.constant(origin).rename("forecast_origin").toInt16(),
            ee.Image.constant(target).rename("target_year").toInt16(),
        ]
    )
    state = origin_state.select(
        [
            "built_state_raw",
            "built_state_final",
            "built_state_valid",
            "valid_observation_count",
        ],
        STATE_COLUMNS,
    )
    composite = ee.Image(composite_asset).select(
        config["predictors"]["spectral_bands"],
        [f"{name}_t" for name in config["predictors"]["spectral_bands"]],
    )
    indices = origin_state.select(
        config["predictors"]["spectral_indices"],
        [f"{name}_t" for name in config["predictors"]["spectral_indices"]],
    )
    neighbourhoods = neighbourhood_features(
        origin_state,
        [int(value) for value in config["predictors"]["neighbourhood_sizes_pixels"]],
    ).select(
        ["built_fraction_3x3_t", "built_fraction_5x5_t", "built_fraction_11x11_t"]
    )
    recent_growth = recent_growth_features(
        previous_transition,
        int(config["predictors"]["recent_growth_window_pixels"]),
    )
    source_year = population_source_year(origin, config["population"]["epochs"])
    if source_year > origin:
        raise AssertionError("Population source year cannot exceed forecast origin.")
    image = ee.Image.cat(
        [
            coordinate_bands(grid),
            time,
            transition.select(["target_transition_1y", "common_valid_1y"]),
            state,
            composite,
            indices,
            neighbourhoods,
            distance_to_built(origin_state, grid),
            recent_growth,
            ee.Image(terrain_asset).select(["elevation_m", "slope_degrees"]),
            population_density(source_year, config),
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
    """Submit one annual CSV table export to the configured Drive folder."""
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


def metadata_directory(config: dict[str, Any], project_root: Path) -> Path:
    """Create and return the isolated annual metadata directory."""
    path = resolve_project_path(config["metadata"]["directory"], project_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def final_directory(config: dict[str, Any], project_root: Path) -> Path:
    """Create and return the provisional annual release directory."""
    path = resolve_project_path(config["outputs"]["final_directory"], project_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Validate finalized annual states and the leakage-safe table schema."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    years = validate_annual_config(config)
    assert_annual_schema_has_no_leakage(TABLE_COLUMNS)
    state_manifest = (
        metadata_directory(config, project_root)
        / config["metadata"]["state_output_manifest"]
    )
    states = load_state_assets(state_manifest, years)
    composites = load_composite_assets(
        resolve_project_path(config["inputs"]["landsat_composite_manifest"], project_root),
        years,
    )
    initialize_earth_engine(config["project"]["earth_engine_project"])
    grid = load_grid_specification(
        resolve_project_path(config["inputs"]["grid_specification"], project_root)
    )
    validate_source_asset(
        str(config["inputs"]["terrain_asset"]),
        ["elevation_m", "slope_degrees"],
        grid,
    )
    result = {
        "status": "PASS",
        "state_count": len(states),
        "transition_count": len(annual_transitions(years)),
        "table_export_count": len(years) - 1,
        "feature_count": len(TABLE_COLUMNS),
        "composite_source_count": len(composites),
    }
    print(json.dumps(result, indent=2))
    return result


def submit_tables(config_path: Path) -> pd.DataFrame:
    """Submit one eligible-cell table export for each one-year transition."""
    preflight = run_preflight(config_path)
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    years = annual_years(config)
    grid = load_grid_specification(
        resolve_project_path(config["inputs"]["grid_specification"], project_root)
    )
    metadata_dir = metadata_directory(config, project_root)
    states = load_state_assets(
        metadata_dir / config["metadata"]["state_output_manifest"], years
    )
    composites = load_composite_assets(
        resolve_project_path(config["inputs"]["landsat_composite_manifest"], project_root),
        years,
    )
    initialize_earth_engine(config["project"]["earth_engine_project"])
    core = load_core_geometry(
        resolve_project_path(config["inputs"]["core_boundary"], project_root)
    )
    projection = projection_from_grid(grid)
    tasks: list[dict[str, Any]] = []
    for order, (origin, target) in enumerate(annual_transitions(years)):
        previous = prior_growth_transition(origin, years)
        feature_image = build_feature_image(
            transition_order=order,
            origin=origin,
            target=target,
            origin_state_asset=states[origin],
            target_state_asset=states[target],
            previous_state_asset=(states[previous[0]] if previous is not None else None),
            composite_asset=composites[origin],
            terrain_asset=str(config["inputs"]["terrain_asset"]),
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
        prefix = f"{config['exports']['table_file_prefix']}_{origin}_{target}"
        task = export_table_to_drive(
            collection,
            description=f"sprawlgp_{prefix}",
            file_prefix=prefix,
            config=config,
        )
        task.update(
            {"transition_order": order, "forecast_origin": origin, "target_year": target}
        )
        tasks.append(task)
    frame = pd.DataFrame(tasks)
    frame.to_csv(metadata_dir / config["metadata"]["table_task_manifest"], index=False)
    print(json.dumps({"preflight": preflight["status"], "task_count": len(frame)}, indent=2))
    return frame


def refresh_tasks(config_path: Path) -> pd.DataFrame:
    """Refresh annual table task states with one Earth Engine task-list call."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    initialize_earth_engine(config["project"]["earth_engine_project"])
    path = metadata_directory(config, project_root) / config["metadata"]["table_task_manifest"]
    if not path.is_file():
        raise FileNotFoundError("Submit annual tables before requesting status.")
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


def read_transition_csvs(
    staging_directory: Path, prefix: str, origin: int, target: int
) -> pd.DataFrame:
    """Read one annual Drive export, including possible CSV shards."""
    paths = sorted(staging_directory.glob(f"{prefix}_{origin}_{target}*.csv"))
    if not paths:
        raise FileNotFoundError(f"No downloaded annual table exists for {origin}-{target}.")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.drop(columns=["system:index", ".geo"], errors="ignore")
    missing = set(TABLE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Annual table {origin}-{target} is missing {sorted(missing)}.")
    return frame[TABLE_COLUMNS].copy()


def validate_cell_time_dataset(frame: pd.DataFrame) -> None:
    """Validate annual target timing, eligibility, uniqueness, and leakage."""
    assert_annual_schema_has_no_leakage(frame.columns.tolist())
    if frame.empty:
        raise ValueError("The annual cell-time dataset is empty.")
    if frame.duplicated(["cell_id", "forecast_origin"]).any():
        raise ValueError("(cell_id, forecast_origin) must uniquely identify rows.")
    if not (frame["target_year"].astype(int) == frame["forecast_origin"].astype(int) + 1).all():
        raise ValueError("Every annual target year must equal forecast origin + 1.")
    if not set(frame["target_transition_1y"].dropna().astype(int)).issubset({0, 1}):
        raise ValueError("Annual transition targets must be binary.")
    if not (frame["built_state_final_t"].astype(int) == 0).all():
        raise ValueError("Every annual model row must be non-built at origin.")
    if not (frame["built_state_valid_t"].astype(int) == 1).all():
        raise ValueError("Every annual model row must be valid at origin.")
    if not (frame["common_valid_1y"].astype(int) == 1).all():
        raise ValueError("Every labelled annual row must be pairwise valid.")
    if not (frame["population_source_year"].astype(int) <= frame["forecast_origin"].astype(int)).all():
        raise ValueError("Population source years cannot be in the future.")
    first = frame[frame["forecast_origin"].astype(int) == 2000]
    if first["recent_local_growth_available_t"].astype(bool).any():
        raise ValueError("Recent annual growth must be unavailable in 2000.")


def assemble_tables(config_path: Path) -> dict[str, Any]:
    """Combine downloaded annual CSV exports into one provisional Parquet table."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    years = validate_annual_config(config)
    grid = load_grid_specification(
        resolve_project_path(config["inputs"]["grid_specification"], project_root)
    )
    staging = resolve_project_path(config["outputs"]["staging_table_directory"], project_root)
    frames: list[pd.DataFrame] = []
    for origin, target in annual_transitions(years):
        frame = read_transition_csvs(
            staging, str(config["exports"]["table_file_prefix"]), origin, target
        )
        frames.append(frame)
    dataset = repair_coordinate_columns(pd.concat(frames, ignore_index=True), grid)
    integer_columns = [
        "cell_id",
        "row",
        "column",
        "transition_order",
        "forecast_origin",
        "target_year",
        "target_transition_1y",
        "common_valid_1y",
        "built_state_raw_t",
        "built_state_final_t",
        "built_state_valid_t",
        "valid_observation_count_t",
        "recent_local_growth_available_t",
        "population_source_year",
    ]
    for column in integer_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="raise").astype("int64")
    validate_cell_time_dataset(dataset)
    dataset = dataset.sort_values(["transition_order", "cell_id"]).reset_index(drop=True)
    output_path = final_directory(config, project_root) / config["outputs"]["cell_time_dataset"]
    dataset.to_parquet(output_path, index=False)
    result = {
        "status": "PASS",
        "row_count": len(dataset),
        "forecast_origins": sorted(dataset["forecast_origin"].unique().tolist()),
        "output": str(output_path),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the annual table command line."""
    parser = argparse.ArgumentParser(description="Build annual eligible-cell tables.")
    parser.add_argument("--config", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight-only", action="store_true")
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--status-only", action="store_true")
    actions.add_argument("--assemble", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected annual table action."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    if arguments.preflight_only:
        run_preflight(config_path)
    elif arguments.submit:
        submit_tables(config_path)
    elif arguments.status_only:
        refresh_tasks(config_path)
    else:
        assemble_tables(config_path)


if __name__ == "__main__":
    main()