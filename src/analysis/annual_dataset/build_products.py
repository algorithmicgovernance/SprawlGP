"""Build isolated annual index and persistent built-state products."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import ee
except ImportError:  # Enables local unit tests without Earth Engine.
    ee = None

from src.analysis.built_up.build_candidates import (
    calculate_histograms,
    load_ee_geometry,
    start_export,
    validate_source_asset,
)
from src.analysis.built_up.indices import build_index_stack
from src.analysis.built_up.otsu import PASS, otsu_from_histogram
from src.analysis.final_dataset.build_products import (
    apply_absorbing_persistence_numpy,
    transition_numpy,
)
from src.analysis.orchestration.common import (
    ensure_asset_folder,
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_json,
    load_yaml,
    resolve_project_path,
    sha256_file,
    stable_object_hash,
    write_json,
)

RETAINED_INDEX_BANDS = ["ndbi", "ibui", "ndbsui", "savi", "mndwi"]
STATE_BANDS = RETAINED_INDEX_BANDS + [
    "built_state_raw",
    "built_state_final",
    "built_state_valid",
    "persistence_corrected",
    "valid_observation_count",
]
TRANSITION_BANDS = [
    "common_valid_1y",
    "eligible_nonbuilt",
    "target_transition_1y",
    "raw_built_to_nonbuilt_reversal",
    "persistence_correction_at_target",
]
PASS_STATES = {"PASS", "COMPLETED", "EXISTS"}


@dataclass(frozen=True)
class AnnualSource:
    """Identify one finalized annual composite and count asset."""

    year: int
    composite_asset: str
    count_asset: str


@dataclass
class AnnualStateBundle:
    """Hold one annual state image and its diagnostic metadata."""

    year: int
    composite_asset: str
    count_asset: str
    image: Any
    thresholds: dict[str, float]
    histograms: dict[str, Any]


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError(
            "The Earth Engine Python API is required for annual products."
        )


def annual_years(config: dict[str, Any]) -> list[int]:
    """Return and validate the configured 2000-2025 annual state years."""
    years = [int(value) for value in config["mapping"]["years"]]
    expected = list(range(2000, 2026))
    if years != expected:
        raise ValueError(f"Annual state years must be {expected}.")
    return years


def annual_transitions(years: list[int]) -> list[tuple[int, int]]:
    """Return every consecutive one-year transition in the state sequence."""
    transitions = list(zip(years[:-1], years[1:], strict=True))
    if any(target != origin + 1 for origin, target in transitions):
        raise ValueError("Every annual target must equal forecast origin + 1.")
    return transitions


def validate_annual_config(config: dict[str, Any]) -> list[int]:
    """Validate annual mapping, isolation, persistence, and leakage guards."""
    years = annual_years(config)
    release = config["release"]
    mapping = config["mapping"]
    exports = config["exports"]
    population = config["population"]
    if mapping["selected_index"] != "ndbi" or mapping["mapping_method"] != "ndbi":
        raise ValueError("Annual version 1 must use NDBI for built states.")
    if mapping["transition_horizon_years"] != 1:
        raise ValueError("The annual transition horizon must be one year.")
    if mapping["mapping_validation_status"] != "PROVISIONAL_ANNUAL_PROTOCOL":
        raise ValueError("Annual mapping must retain provisional validation status.")
    if release["release_status"] != "PROVISIONAL_PENDING_ANNUAL_MANUAL_VALIDATION":
        raise ValueError("Annual release status must remain provisional.")
    if release["manual_validation_complete"] is not False:
        raise ValueError("Annual manual validation must remain incomplete.")
    if mapping["built_up_direction"] != "greater_than":
        raise ValueError("Annual NDBI classification must use greater-than thresholds.")
    if mapping["persistence_rule"] != "absorbing_after_first_valid_built_observation":
        raise ValueError("Annual states must use the existing absorbing persistence rule.")
    if exports["overwrite_existing_assets"] is not False:
        raise ValueError("Annual products must not overwrite existing assets.")
    asset_root = str(exports["asset_root"])
    if not asset_root.endswith("/assets/sprawlgp/annual_v1"):
        raise ValueError("Annual products must remain under the annual_v1 root.")
    if "/assets/sprawlgp/v2" in asset_root:
        raise ValueError("Annual products cannot write to the five-year v2 root.")
    epochs = sorted(int(value) for value in population["epochs"])
    if epochs != [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]:
        raise ValueError("Use the repository's configured GHSL epochs exactly.")
    if population["lookup_policy"] != "latest_available_epoch_at_or_before_origin":
        raise ValueError("Annual population lookup must prohibit future epochs.")
    annual_transitions(years)
    return years


def population_source_year(origin: int, epochs: list[int]) -> int:
    """Select the latest configured population epoch at or before an origin."""
    available = [int(epoch) for epoch in epochs if int(epoch) <= int(origin)]
    if not available:
        raise ValueError(
            f"No population epoch is available at or before origin {origin}."
        )
    source_year = max(available)
    if source_year > int(origin):
        raise AssertionError("Population source year cannot exceed the origin.")
    return source_year


def load_annual_sources(path: Path, years: list[int]) -> list[AnnualSource]:
    """Load exactly one finalized Day 2 composite/count pair per annual year."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"epoch", "asset_id", "count_asset_id", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Annual composite manifest is missing {sorted(missing)}.")
    frame["epoch"] = frame["epoch"].astype(int)
    frame = frame[frame["status"].astype(str).str.upper().isin(PASS_STATES)]
    frame = frame[frame["epoch"].isin(years)].sort_values("epoch")
    if frame["epoch"].tolist() != years or not frame["epoch"].is_unique:
        raise ValueError("Day 2 manifest must contain one passed source for every year.")
    expected_root = "projects/urban-sprawl-ssa/assets/sprawlgp/annual_v1/landsat"
    sources = [
        AnnualSource(
            year=int(row.epoch),
            composite_asset=str(row.asset_id),
            count_asset=str(row.count_asset_id),
        )
        for row in frame.itertuples(index=False)
    ]
    for source in sources:
        if not source.composite_asset.startswith(expected_root + "/composite_"):
            raise ValueError(f"Unexpected annual composite source: {source.composite_asset}")
        if not source.count_asset.startswith(expected_root + "/valid_count_"):
            raise ValueError(f"Unexpected annual count source: {source.count_asset}")
    return sources


def compare_grid_reference(grid: dict[str, Any], reference: dict[str, Any]) -> None:
    """Reject changes to fields defining the frozen project grid."""
    for field in ("crs", "resolution_m", "transform", "width", "height", "extent"):
        if grid.get(field) != reference.get(field):
            raise ValueError(f"Frozen grid mismatch for '{field}'.")


def annual_state_recipe(
    *,
    year: int,
    thresholds: dict[str, float],
    source_composite_asset: str,
    source_count_asset: str,
    grid: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical provenance recipe for one annual state asset."""
    return {
        "year": int(year),
        "mapping_method": "ndbi",
        "thresholds": thresholds,
        "band_schema": STATE_BANDS,
        "source_composite_asset": source_composite_asset,
        "source_count_asset": source_count_asset,
        "grid_sha256": stable_object_hash(grid),
    }


def validate_state_asset_properties(
    asset_id: str,
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Reject annual asset properties that differ from trusted provenance."""
    mismatches = [
        f"{field}: expected {expected_value!r}, observed {observed.get(field)!r}"
        for field, expected_value in expected.items()
        if observed.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Annual state asset provenance mismatch for {asset_id}: "
            + "; ".join(mismatches)
        )


def validate_existing_state_asset(
    asset_id: str,
    *,
    expected_recipe_sha256: str,
    expected_year: int,
    expected_composite_asset: str,
    expected_count_asset: str,
    grid: dict[str, Any],
) -> None:
    """Validate an annual state against trusted task and Day 2 provenance."""
    require_earth_engine()
    validate_source_asset(asset_id, STATE_BANDS, grid)
    properties = ee.Image(asset_id).toDictionary(
        [
            "recipe_sha256",
            "year",
            "mapping_method",
            "mapping_validation_status",
            "source_composite_asset",
            "source_count_asset",
        ]
    ).getInfo()
    expected = {
        "recipe_sha256": str(expected_recipe_sha256),
        "year": int(expected_year),
        "mapping_method": "ndbi",
        "mapping_validation_status": "PROVISIONAL_ANNUAL_PROTOCOL",
        "source_composite_asset": expected_composite_asset,
        "source_count_asset": expected_count_asset,
    }
    validate_state_asset_properties(asset_id, properties, expected)


def build_transition_image(start_state: Any, end_state: Any):
    """Construct one-year labels and diagnostics from adjacent annual states."""
    require_earth_engine()
    start_valid = start_state.select("built_state_valid").eq(1)
    end_valid = end_state.select("built_state_valid").eq(1)
    common = start_valid.And(end_valid).rename("common_valid_1y").unmask(0).toUint8()
    eligible = (
        common.eq(1)
        .And(start_state.select("built_state_final").eq(0))
        .rename("eligible_nonbuilt")
        .unmask(0)
        .toUint8()
    )
    target = (
        eligible.eq(1)
        .And(end_state.select("built_state_final").eq(1))
        .rename("target_transition_1y")
        .updateMask(eligible.eq(1))
        .toUint8()
    )
    reversal = (
        common.eq(1)
        .And(start_state.select("built_state_raw").eq(1))
        .And(end_state.select("built_state_raw").eq(0))
        .rename("raw_built_to_nonbuilt_reversal")
        .unmask(0)
        .toUint8()
    )
    correction = (
        common.eq(1)
        .And(end_state.select("persistence_corrected").eq(1))
        .rename("persistence_correction_at_target")
        .unmask(0)
        .toUint8()
    )
    return ee.Image.cat([common, eligible, target, reversal, correction]).select(
        TRANSITION_BANDS
    )


def build_state_bundles(
    sources: list[AnnualSource],
    core_geometry: Any,
    grid: dict[str, Any],
    config: dict[str, Any],
) -> list[AnnualStateBundle]:
    """Calculate yearly Otsu states and apply absorbing persistence in order."""
    require_earth_engine()
    ever_built = ee.Image.constant(0).toUint8()
    bundles: list[AnnualStateBundle] = []
    diagnostic_indices = list(config["indices"]["diagnostic_otsu"])

    for source in sources:
        composite = ee.Image(source.composite_asset)
        count = ee.Image(source.count_asset)
        index_stack, index_images = build_index_stack(
            composite,
            count,
            epsilon=float(config["indices"]["denominator_epsilon"]),
            savi_l=float(config["indices"]["savi_l"]),
            minimum_valid_observations=int(
                config["classification"]["minimum_valid_observations_for_state"]
            ),
        )
        histograms = calculate_histograms(index_images, count, core_geometry, grid, config)
        thresholds: dict[str, float] = {}
        for index_name in diagnostic_indices:
            result = otsu_from_histogram(histograms.get(index_name))
            if result.status != PASS or result.threshold is None:
                raise RuntimeError(
                    f"Otsu failed for {source.year} {index_name}: {result.failure_reason}"
                )
            thresholds[index_name] = float(result.threshold)

        ndbi = index_images["ndbi"]
        valid = (
            ndbi.mask().reduce(ee.Reducer.min()).rename("built_state_valid").unmask(0).toUint8()
        )
        raw_unmasked = ndbi.gt(thresholds["ndbi"]).unmask(0).toUint8()
        raw = raw_unmasked.updateMask(valid.eq(1)).rename("built_state_raw")
        correction = (
            valid.eq(1)
            .And(raw_unmasked.eq(0))
            .And(ever_built.eq(1))
            .rename("persistence_corrected")
            .unmask(0)
            .toUint8()
        )
        final = (
            raw_unmasked.where(valid.eq(1).And(ever_built.eq(1)), 1)
            .updateMask(valid.eq(1))
            .rename("built_state_final")
            .toUint8()
        )
        retained_indices = index_stack.select(RETAINED_INDEX_BANDS)
        observation_count = (
            count.select("valid_observation_count").unmask(0).toUint16()
        )
        image = ee.Image.cat(
            [retained_indices, raw, final, valid, correction, observation_count]
        ).select(STATE_BANDS)
        image = image.set(
            {
                "pipeline_stage": "annual_dataset_state",
                "pipeline_version": int(config["version"]),
                "year": source.year,
                "mapping_method": "ndbi",
                "mapping_validation_status": "PROVISIONAL_ANNUAL_PROTOCOL",
                "ndbi_otsu_threshold": thresholds["ndbi"],
                "source_composite_asset": source.composite_asset,
                "source_count_asset": source.count_asset,
            }
        )
        bundles.append(
            AnnualStateBundle(
                year=source.year,
                composite_asset=source.composite_asset,
                count_asset=source.count_asset,
                image=image,
                thresholds=thresholds,
                histograms=histograms,
            )
        )
        ever_built = ever_built.Or(valid.eq(1).And(raw_unmasked.eq(1))).unmask(0).toUint8()

    return bundles


def metadata_directory(config: dict[str, Any], project_root: Path) -> Path:
    """Create and return the isolated annual metadata directory."""
    path = resolve_project_path(config["metadata"]["directory"], project_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Validate annual local inputs and the non-overwriting product contract."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    years = validate_annual_config(config)
    local_inputs = {
        name: resolve_project_path(value, project_root)
        for name, value in config["inputs"].items()
        if name != "terrain_asset"
    }
    for name, path in local_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing annual dependency '{name}': {path}")
    grid = load_grid_specification(local_inputs["grid_specification"])
    compare_grid_reference(grid, load_json(local_inputs["grid_reference"]))
    sources = load_annual_sources(local_inputs["landsat_composite_manifest"], years)
    quality = pd.read_csv(local_inputs["annual_quality_summary"])
    if quality["year"].astype(int).tolist() != years:
        raise ValueError("Annual quality summary must cover every year in order.")
    metadata_dir = metadata_directory(config, project_root)
    dependencies = [
        {
            "dependency_name": name,
            "dependency_path": str(path),
            "observed_sha256": sha256_file(path),
            "status": "PASS",
        }
        for name, path in local_inputs.items()
    ]
    pd.DataFrame(dependencies).to_csv(
        metadata_dir / config["metadata"]["dependency_manifest"], index=False
    )
    result = {
        "status": "PASS",
        "state_years": years,
        "state_count": len(sources),
        "transition_count": len(annual_transitions(years)),
        "expected_state_export_tasks": len(sources),
        "transition_asset_count": 0,
        "mapping_method": "ndbi",
    }
    print(json.dumps(result, indent=2))
    return result


def submit_products(config_path: Path) -> pd.DataFrame:
    """Calculate annual thresholds and submit 26 compact state exports."""
    preflight = run_preflight(config_path)
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    grid = load_grid_specification(
        resolve_project_path(config["inputs"]["grid_specification"], project_root)
    )
    sources = load_annual_sources(
        resolve_project_path(config["inputs"]["landsat_composite_manifest"], project_root),
        annual_years(config),
    )
    initialize_earth_engine(config["project"]["earth_engine_project"])
    for source in sources:
        validate_source_asset(source.composite_asset, ["blue", "green", "red", "nir", "swir1", "swir2"], grid)
        count_bands = ee.Image(source.count_asset).bandNames().getInfo()
        if "valid_observation_count" not in count_bands:
            raise ValueError(f"Missing valid observation count: {source.count_asset}")
        validate_source_asset(source.count_asset, count_bands, grid)
    core = load_ee_geometry(
        resolve_project_path(config["inputs"]["core_boundary"], project_root)
    )
    bundles = build_state_bundles(sources, core, grid, config)
    metadata_dir = metadata_directory(config, project_root)
    threshold_rows = [
        {
            "year": bundle.year,
            "index_name": index_name,
            "threshold": threshold,
            "mapping_method": "ndbi",
            "mapping_validation_status": "PROVISIONAL_ANNUAL_PROTOCOL",
            "source_composite_asset": bundle.composite_asset,
            "source_count_asset": bundle.count_asset,
        }
        for bundle in bundles
        for index_name, threshold in bundle.thresholds.items()
    ]
    pd.DataFrame(threshold_rows).to_csv(
        metadata_dir / config["metadata"]["threshold_table"], index=False
    )
    write_json(
        metadata_dir / config["metadata"]["histogram_archive"],
        {str(bundle.year): bundle.histograms for bundle in bundles},
    )
    asset_root = str(config["exports"]["asset_root"]).rstrip("/")
    state_folder = f"{asset_root}/{config['exports']['state_folder']}"
    ensure_asset_folder(asset_root)
    ensure_asset_folder(f"{asset_root}/annual_dataset")
    ensure_asset_folder(state_folder)
    tasks: list[dict[str, Any]] = []
    for bundle in bundles:
        asset_id = f"{state_folder}/state_{bundle.year}"
        recipe = annual_state_recipe(
            year=bundle.year,
            thresholds=bundle.thresholds,
            source_composite_asset=bundle.composite_asset,
            source_count_asset=bundle.count_asset,
            grid=grid,
        )
        recipe_sha256 = stable_object_hash(recipe)
        task = start_export(
            bundle.image.set("recipe_sha256", recipe_sha256),
            description=f"sprawlgp_annual_state_{bundle.year}",
            asset_id=asset_id,
            pyramiding_policy={
                ".default": "mean",
                "built_state_raw": "mode",
                "built_state_final": "mode",
                "built_state_valid": "mode",
                "persistence_corrected": "mode",
                "valid_observation_count": "sample",
            },
            grid=grid,
            config=config,
        )
        if task["state"] == "EXISTS":
            validate_existing_state_asset(
                asset_id,
                expected_recipe_sha256=recipe_sha256,
                expected_year=bundle.year,
                expected_composite_asset=bundle.composite_asset,
                expected_count_asset=bundle.count_asset,
                grid=grid,
            )
        task.update(
            {
                "product_type": "annual_state",
                "year": bundle.year,
                "band_names": ",".join(STATE_BANDS),
                "recipe_sha256": recipe_sha256,
            }
        )
        tasks.append(task)
    task_frame = pd.DataFrame(tasks)
    task_frame.to_csv(
        metadata_dir / config["metadata"]["state_task_manifest"], index=False
    )
    print(json.dumps({"preflight": preflight["status"], "task_count": len(tasks)}, indent=2))
    return task_frame


def parse_arguments() -> argparse.Namespace:
    """Parse the annual product command line."""
    parser = argparse.ArgumentParser(description="Build isolated annual state products.")
    parser.add_argument("--config", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight-only", action="store_true")
    actions.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run annual product preflight or submission."""
    arguments = parse_arguments()
    if arguments.preflight_only:
        run_preflight(arguments.config.resolve())
    else:
        submit_products(arguments.config.resolve())


if __name__ == "__main__":
    main()


__all__ = [
    "RETAINED_INDEX_BANDS",
    "STATE_BANDS",
    "TRANSITION_BANDS",
    "annual_transitions",
    "annual_years",
    "apply_absorbing_persistence_numpy",
    "build_transition_image",
    "load_annual_sources",
    "population_source_year",
    "transition_numpy",
    "validate_annual_config",
]
