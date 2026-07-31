"""Build final built-up states, fixed tracking support and transitions.

The module submits only raster assets. Cell-time tables are created in the
second phase after the raster products have completed and passed validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:  # Enables local unit tests without Earth Engine.
    ee = None

from src.analysis.orchestration.common import (
    asset_exists,
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


STATE_BANDS = [
    "built_state_raw",
    "built_state_final",
    "built_state_valid",
    "persistence_corrected",
    "valid_observation_count",
]

TRANSITION_BANDS = [
    "common_valid",
    "eligible_nonbuilt",
    "target_transition_5y",
    "raw_built_to_nonbuilt_reversal",
    "persistence_correction_at_target",
    "transition_quality",
]

PASS_STATES = {"PASS", "COMPLETED", "EXISTS"}


@dataclass(frozen=True)
class CandidateEpoch:
    """Hold one finalized Day 4 candidate source."""

    epoch: int
    candidate_asset: str
    count_asset: str


@dataclass
class StateBundle:
    """Hold one lazy state image and its provenance."""

    epoch: int
    candidate_asset: str
    count_asset: str
    image: Any


@dataclass
class TransitionBundle:
    """Hold one lazy transition image."""

    period_start: int
    period_end: int
    image: Any


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError(
            "The Earth Engine Python API is required for raster products."
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


def export_grid_region(grid: dict[str, Any]):
    """Return an inset rectangle selecting exactly the frozen grid cells."""
    require_earth_engine()
    extent = grid["extent"]
    inset = float(grid["resolution_m"]) / 4.0

    return ee.Geometry.Rectangle(
        [
            float(extent["xmin"]) + inset,
            float(extent["ymin"]) + inset,
            float(extent["xmax"]) - inset,
            float(extent["ymax"]) - inset,
        ],
        proj=str(grid["crs"]),
        geodesic=False,
    )


def compare_grid_reference(
    grid: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    """Reject changes to fields defining the frozen Day 1 raster grid."""
    for field in (
        "crs",
        "resolution_m",
        "transform",
        "width",
        "height",
        "extent",
    ):
        if grid.get(field) != reference.get(field):
            raise ValueError(
                f"Frozen grid mismatch for '{field}': "
                f"{grid.get(field)!r} != {reference.get(field)!r}"
            )


def image_metadata(asset_id: str) -> dict[str, Any]:
    """Read ordered bands and exact grid information from one image asset."""
    require_earth_engine()
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
            for value in first.get("crs_transform", [])
        ],
    }


def validate_asset_grid(
    asset_id: str,
    expected_bands: list[str],
    grid: dict[str, Any],
    *,
    allow_additional_bands: bool = False,
) -> None:
    """Validate asset existence, band schema and exact frozen grid.

    ``allow_additional_bands`` is intended for observation-count assets.
    Those assets may retain sensor-specific diagnostic count bands while Day 6
    requires only the canonical ``valid_observation_count`` band.
    """
    if not asset_exists(asset_id):
        raise FileNotFoundError(f"Earth Engine asset not found: {asset_id}")

    observed = image_metadata(asset_id)

    observed_bands = observed["band_names"]

    if allow_additional_bands:
        missing_bands = [
            band for band in expected_bands if band not in observed_bands
        ]

        if missing_bands:
            raise ValueError(
                f"Missing required bands in {asset_id}: {missing_bands}; "
                f"observed {observed_bands}."
            )
    elif observed_bands != expected_bands:
        raise ValueError(
            f"Unexpected bands in {asset_id}: {observed_bands}; "
            f"expected exactly {expected_bands}."
        )

    if observed["crs"] != grid["crs"]:
        raise ValueError(
            f"Unexpected CRS in {asset_id}: {observed['crs']}"
        )

    if not np.allclose(
        observed["transform"],
        [float(value) for value in grid["transform"]],
        atol=1e-9,
    ):
        raise ValueError(
            f"Unexpected transform in {asset_id}: "
            f"{observed['transform']}"
        )

    if (
        observed["width"] != int(grid["width"])
        or observed["height"] != int(grid["height"])
    ):
        raise ValueError(
            f"Unexpected dimensions in {asset_id}: "
            f"{observed['width']} x {observed['height']}"
        )


def candidate_epochs(
    manifest_path: Path,
    config: dict[str, Any],
) -> list[CandidateEpoch]:
    """Load exactly one finalized candidate asset for each configured epoch."""
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "epoch",
        "product_type",
        "asset_id",
        "source_count_asset",
        "state",
        "band_names",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Day 4 output manifest is missing columns: {sorted(missing)}"
        )

    candidates = frame[
        frame["product_type"].astype(str) == "candidates"
    ].copy()
    candidates["state"] = (
        candidates["state"].astype(str).str.upper()
    )
    candidates = candidates[
        candidates["state"].isin(PASS_STATES)
    ]
    candidates["epoch"] = candidates["epoch"].astype(int)
    expected_epochs = [
        int(value) for value in config["mapping"]["epochs"]
    ]
    candidates = candidates[
        candidates["epoch"].isin(expected_epochs)
    ].sort_values("epoch")

    if candidates["epoch"].tolist() != expected_epochs:
        raise ValueError(
            "Finalized Day 4 candidate epochs do not match the configured "
            f"sequence: {candidates['epoch'].tolist()} != {expected_epochs}"
        )

    if not candidates["epoch"].is_unique:
        raise ValueError("Candidate manifest contains duplicate epochs.")

    candidate_band = str(config["mapping"]["candidate_band"])
    validity_band = str(config["mapping"]["validity_band"])
    result: list[CandidateEpoch] = []

    for row in candidates.itertuples(index=False):
        bands = [
            value.strip()
            for value in str(row.band_names).split(",")
            if value.strip()
        ]

        for required_band in (candidate_band, validity_band):
            if required_band not in bands:
                raise ValueError(
                    f"Candidate asset for {row.epoch} is missing "
                    f"'{required_band}'."
                )

        result.append(
            CandidateEpoch(
                epoch=int(row.epoch),
                candidate_asset=str(row.asset_id),
                count_asset=str(row.source_count_asset),
            )
        )

    return result


def validate_epoch_spacing(
    epochs: list[int],
    horizon_years: int,
) -> None:
    """Require a strictly increasing sequence at the configured horizon."""
    if len(epochs) < 2:
        raise ValueError("At least two epochs are required.")

    differences = np.diff(np.asarray(epochs, dtype=int))

    if not np.all(differences == int(horizon_years)):
        raise ValueError(
            f"Epoch differences must all equal {horizon_years}: "
            f"{differences.tolist()}"
        )


def apply_absorbing_persistence_numpy(
    raw_states: np.ndarray,
    valid_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation used by consistency tests.

    Invalid cells remain encoded as -1. A cell observed as built remains built
    during all subsequent valid observations.
    """
    raw = np.asarray(raw_states, dtype=int)
    valid = np.asarray(valid_states, dtype=bool)

    if raw.shape != valid.shape or raw.ndim < 1:
        raise ValueError("Raw and valid arrays must have the same shape.")

    final = np.full(raw.shape, -1, dtype=int)
    corrected = np.zeros(raw.shape, dtype=np.uint8)
    ever_built = np.zeros(raw.shape[1:], dtype=bool)

    for index in range(raw.shape[0]):
        current_valid = valid[index]
        current_raw = raw[index] == 1
        correction = current_valid & (~current_raw) & ever_built
        final[index][current_valid] = (
            current_raw[current_valid] | ever_built[current_valid]
        ).astype(int)
        corrected[index] = correction.astype(np.uint8)
        ever_built = ever_built | (current_valid & current_raw)

    return final, corrected


def transition_numpy(
    start_state: np.ndarray,
    end_state: np.ndarray,
    start_valid: np.ndarray,
    end_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference transition eligibility and label implementation."""
    start = np.asarray(start_state, dtype=int)
    end = np.asarray(end_state, dtype=int)
    valid_start = np.asarray(start_valid, dtype=bool)
    valid_end = np.asarray(end_valid, dtype=bool)

    if not (
        start.shape == end.shape == valid_start.shape == valid_end.shape
    ):
        raise ValueError("Transition arrays must share one shape.")

    common = valid_start & valid_end
    eligible = common & (start == 0)
    target = eligible & (end == 1)
    return (
        common.astype(np.uint8),
        eligible.astype(np.uint8),
        target.astype(np.uint8),
    )


def build_state_bundles(
    sources: list[CandidateEpoch],
    config: dict[str, Any],
) -> list[StateBundle]:
    """Create raw and persistence-consistent built-up state images."""
    require_earth_engine()
    candidate_band = str(config["mapping"]["candidate_band"])
    validity_band = str(config["mapping"]["validity_band"])
    ever_built = ee.Image.constant(0).toUint8()
    bundles: list[StateBundle] = []

    for source in sources:
        candidate = ee.Image(source.candidate_asset)
        valid = (
            candidate.select(validity_band)
            .eq(1)
            .rename("built_state_valid")
            .unmask(0)
            .toUint8()
        )
        raw_unmasked = (
            candidate.select(candidate_band)
            .unmask(0)
            .toUint8()
        )
        raw = (
            raw_unmasked.updateMask(valid.eq(1))
            .rename("built_state_raw")
        )
        correction = (
            valid.eq(1)
            .And(raw_unmasked.eq(0))
            .And(ever_built.eq(1))
            .rename("persistence_corrected")
            .unmask(0)
            .toUint8()
        )
        final = (
            raw_unmasked.where(
                valid.eq(1).And(ever_built.eq(1)),
                1,
            )
            .updateMask(valid.eq(1))
            .rename("built_state_final")
            .toUint8()
        )
        count = (
            ee.Image(source.count_asset)
            .select("valid_observation_count")
            .unmask(0)
            .toUint16()
        )
        image = ee.Image.cat(
            [raw, final, valid, correction, count]
        ).select(STATE_BANDS)
        image = image.set(
            {
                "pipeline_stage": "final_dataset_states",
                "pipeline_version": int(config["version"]),
                "epoch": int(source.epoch),
                "mapping_method": config["mapping"]["selected_index"],
                "mapping_selection_status": config["release"][
                    "mapping_selection_status"
                ],
                "source_candidate_asset": source.candidate_asset,
                "source_count_asset": source.count_asset,
            }
        )
        bundles.append(
            StateBundle(
                epoch=source.epoch,
                candidate_asset=source.candidate_asset,
                count_asset=source.count_asset,
                image=image,
            )
        )
        ever_built = (
            ever_built.Or(
                valid.eq(1).And(raw_unmasked.eq(1))
            )
            .unmask(0)
            .toUint8()
        )

    return bundles


def build_tracking_support(states: list[StateBundle]):
    """Intersect valid support across every final state epoch."""
    require_earth_engine()
    support = ee.Image.constant(1).toUint8()

    for bundle in states:
        support = support.And(
            bundle.image.select("built_state_valid").eq(1)
        )

    return (
        support.rename("tracking_common_support")
        .unmask(0)
        .toUint8()
    )


def build_transition_bundles(
    states: list[StateBundle],
    config: dict[str, Any],
) -> list[TransitionBundle]:
    """Construct one eligible-cell transition for each consecutive pair."""
    require_earth_engine()
    minimum_count = int(
        config["quality"]["high_confidence_min_observations"]
    )
    bundles: list[TransitionBundle] = []

    for start, end in zip(states[:-1], states[1:]):
        start_valid = start.image.select("built_state_valid").eq(1)
        end_valid = end.image.select("built_state_valid").eq(1)
        common = (
            start_valid.And(end_valid)
            .rename("common_valid")
            .unmask(0)
            .toUint8()
        )
        start_final = start.image.select("built_state_final")
        end_final = end.image.select("built_state_final")
        eligible = (
            common.eq(1)
            .And(start_final.eq(0))
            .rename("eligible_nonbuilt")
            .unmask(0)
            .toUint8()
        )
        target = (
            eligible.eq(1)
            .And(end_final.eq(1))
            .rename("target_transition_5y")
            .updateMask(eligible.eq(1))
            .toUint8()
        )
        reversal = (
            common.eq(1)
            .And(start.image.select("built_state_raw").eq(1))
            .And(end.image.select("built_state_raw").eq(0))
            .rename("raw_built_to_nonbuilt_reversal")
            .unmask(0)
            .toUint8()
        )
        correction_target = (
            common.eq(1)
            .And(
                end.image.select("persistence_corrected").eq(1)
            )
            .rename("persistence_correction_at_target")
            .unmask(0)
            .toUint8()
        )
        high_quality = (
            eligible.eq(1)
            .And(
                start.image.select("valid_observation_count").gte(
                    minimum_count
                )
            )
            .And(
                end.image.select("valid_observation_count").gte(
                    minimum_count
                )
            )
            .And(correction_target.eq(0))
        )
        quality = (
            ee.Image.constant(0)
            .where(eligible.eq(1), 1)
            .where(high_quality, 2)
            .rename("transition_quality")
            .toUint8()
        )
        image = ee.Image.cat(
            [
                common,
                eligible,
                target,
                reversal,
                correction_target,
                quality,
            ]
        ).select(TRANSITION_BANDS)
        image = image.set(
            {
                "pipeline_stage": "final_dataset_transitions",
                "pipeline_version": int(config["version"]),
                "period_start": int(start.epoch),
                "period_end": int(end.epoch),
                "mapping_method": config["mapping"]["selected_index"],
                "mapping_selection_status": config["release"][
                    "mapping_selection_status"
                ],
            }
        )
        bundles.append(
            TransitionBundle(
                period_start=start.epoch,
                period_end=end.epoch,
                image=image,
            )
        )

    return bundles


def start_export(
    image: Any,
    *,
    description: str,
    asset_id: str,
    pyramiding_policy: dict[str, str],
    grid: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Submit one exact-grid image asset or record an existing asset."""
    require_earth_engine()
    overwrite = bool(
        config["exports"]["overwrite_existing_assets"]
    )

    if asset_exists(asset_id) and not overwrite:
        return {
            "task_id": "",
            "state": "EXISTS",
            "description": description,
            "asset_id": asset_id,
        }

    transform = [float(value) for value in grid["transform"]]
    export_image = image.setDefaultProjection(
        str(grid["crs"]),
        transform,
    )
    task = ee.batch.Export.image.toAsset(
        image=export_image,
        description=description,
        assetId=asset_id,
        pyramidingPolicy=pyramiding_policy,
        region=export_grid_region(grid),
        crs=str(grid["crs"]),
        crsTransform=transform,
        maxPixels=int(config["exports"]["max_pixels"]),
        shardSize=int(config["exports"]["shard_size"]),
        priority=int(config["exports"]["priority"]),
        overwrite=overwrite,
    )
    task.start()
    status = task.status()
    return {
        "task_id": status.get("id", task.id),
        "state": status.get("state", "READY"),
        "description": description,
        "asset_id": asset_id,
    }


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Validate all dependencies required before raster submission."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    inputs = {
        name: resolve_project_path(value, project_root)
        for name, value in config["inputs"].items()
        if name != "terrain_asset"
    }

    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing final-dataset dependency '{name}': {path}"
            )

    if config["mapping"]["selected_index"] != "ndbi":
        raise ValueError("Version 1 must use NDBI as the mapping method.")

    if config["release"]["manual_validation_complete"] is not False:
        raise ValueError(
            "Version 1 must preserve pending manual-validation status."
        )

    grid = load_grid_specification(inputs["grid_specification"])
    reference = load_json(inputs["grid_reference"])
    compare_grid_reference(grid, reference)
    sources = candidate_epochs(
        inputs["built_up_output_manifest"],
        config,
    )
    epochs = [source.epoch for source in sources]
    validate_epoch_spacing(
        epochs,
        int(config["mapping"]["transition_horizon_years"]),
    )
    initialize_earth_engine(
        config["project"]["earth_engine_project"]
    )

    for source in sources:
        candidate_metadata = image_metadata(source.candidate_asset)

        if config["mapping"]["candidate_band"] not in candidate_metadata[
            "band_names"
        ]:
            raise ValueError(
                f"Missing candidate band in {source.candidate_asset}."
            )

        if config["mapping"]["validity_band"] not in candidate_metadata[
            "band_names"
        ]:
            raise ValueError(
                f"Missing validity band in {source.candidate_asset}."
            )

        validate_asset_grid(
            source.count_asset,
            ["valid_observation_count"],
            grid,
            allow_additional_bands=True,
        )

        if candidate_metadata["crs"] != grid["crs"]:
            raise ValueError(
                f"Unexpected candidate CRS: {source.candidate_asset}"
            )

        if not np.allclose(
            candidate_metadata["transform"],
            [float(value) for value in grid["transform"]],
            atol=1e-9,
        ):
            raise ValueError(
                f"Unexpected candidate transform: {source.candidate_asset}"
            )

        if (
            candidate_metadata["width"] != int(grid["width"])
            or candidate_metadata["height"] != int(grid["height"])
        ):
            raise ValueError(
                f"Unexpected candidate dimensions: {source.candidate_asset}"
            )

    terrain_asset = str(config["inputs"]["terrain_asset"])
    validate_asset_grid(
        terrain_asset,
        ["elevation_m", "slope_degrees"],
        grid,
    )
    metadata_dir = metadata_directory(config, project_root)
    dependency_path = (
        metadata_dir / config["metadata"]["dependency_manifest"]
    )
    dependency_rows = [
        {
            "dependency_name": name,
            "dependency_path": str(path),
            "observed_sha256": sha256_file(path),
            "status": "PASS",
        }
        for name, path in inputs.items()
    ]
    dependency_rows.append(
        {
            "dependency_name": "terrain_asset",
            "dependency_path": terrain_asset,
            "observed_sha256": stable_object_hash(terrain_asset),
            "status": "PASS",
        }
    )
    pd.DataFrame(dependency_rows).to_csv(
        dependency_path,
        index=False,
    )
    result = {
        "status": "PASS",
        "mapping_method": config["mapping"]["selected_index"],
        "mapping_selection_status": config["release"][
            "mapping_selection_status"
        ],
        "epochs": epochs,
        "state_count": len(epochs),
        "transition_count": len(epochs) - 1,
        "tracking_support_count": 1,
        "expected_raster_tasks": len(epochs) * 2,
        "dependency_manifest": str(dependency_path),
    }
    print(json.dumps(result, indent=2))
    return result


def submit_rasters(config_path: Path) -> pd.DataFrame:
    """Build all raster products and submit ten batch exports."""
    preflight = run_preflight(config_path)
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    grid = load_grid_specification(
        resolve_project_path(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    sources = candidate_epochs(
        resolve_project_path(
            config["inputs"]["built_up_output_manifest"],
            project_root,
        ),
        config,
    )
    states = build_state_bundles(sources, config)
    support = build_tracking_support(states)
    transitions = build_transition_bundles(states, config)
    asset_root = str(config["exports"]["asset_root"]).rstrip("/")
    state_folder = f"{asset_root}/{config['exports']['state_folder']}"
    transition_folder = (
        f"{asset_root}/{config['exports']['transition_folder']}"
    )
    support_folder = f"{asset_root}/{config['exports']['support_folder']}"

    for folder in (
        asset_root,
        state_folder,
        transition_folder,
        support_folder,
    ):
        ensure_asset_folder(folder)

    task_rows: list[dict[str, Any]] = []

    for state in states:
        asset_id = f"{state_folder}/final_state_{state.epoch}"
        recipe = {
            "product_type": "state",
            "epoch": state.epoch,
            "source_candidate_asset": state.candidate_asset,
            "source_count_asset": state.count_asset,
            "mapping_method": config["mapping"]["selected_index"],
            "band_schema": STATE_BANDS,
            "grid_sha256": stable_object_hash(grid),
        }
        task = start_export(
            state.image.set("recipe_sha256", stable_object_hash(recipe)),
            description=f"sprawlgp_final_state_{state.epoch}",
            asset_id=asset_id,
            pyramiding_policy={
                ".default": "mode",
                "valid_observation_count": "sample",
            },
            grid=grid,
            config=config,
        )
        task.update(
            {
                "product_type": "state",
                "epoch": state.epoch,
                "period_start": "",
                "period_end": "",
                "band_names": ",".join(STATE_BANDS),
                "recipe_sha256": stable_object_hash(recipe),
            }
        )
        task_rows.append(task)

    support_asset = f"{support_folder}/tracking_common_support"
    support_recipe = {
        "product_type": "tracking_support",
        "epochs": [state.epoch for state in states],
        "band_schema": ["tracking_common_support"],
        "grid_sha256": stable_object_hash(grid),
    }
    support_task = start_export(
        support.set("recipe_sha256", stable_object_hash(support_recipe)),
        description="sprawlgp_tracking_common_support",
        asset_id=support_asset,
        pyramiding_policy={".default": "mode"},
        grid=grid,
        config=config,
    )
    support_task.update(
        {
            "product_type": "tracking_support",
            "epoch": "",
            "period_start": "",
            "period_end": "",
            "band_names": "tracking_common_support",
            "recipe_sha256": stable_object_hash(support_recipe),
        }
    )
    task_rows.append(support_task)

    for transition in transitions:
        asset_id = (
            f"{transition_folder}/transition_"
            f"{transition.period_start}_{transition.period_end}"
        )
        recipe = {
            "product_type": "transition",
            "period_start": transition.period_start,
            "period_end": transition.period_end,
            "mapping_method": config["mapping"]["selected_index"],
            "band_schema": TRANSITION_BANDS,
            "grid_sha256": stable_object_hash(grid),
        }
        task = start_export(
            transition.image.set(
                "recipe_sha256",
                stable_object_hash(recipe),
            ),
            description=(
                "sprawlgp_transition_"
                f"{transition.period_start}_{transition.period_end}"
            ),
            asset_id=asset_id,
            pyramiding_policy={".default": "mode"},
            grid=grid,
            config=config,
        )
        task.update(
            {
                "product_type": "transition",
                "epoch": "",
                "period_start": transition.period_start,
                "period_end": transition.period_end,
                "band_names": ",".join(TRANSITION_BANDS),
                "recipe_sha256": stable_object_hash(recipe),
            }
        )
        task_rows.append(task)

    task_frame = pd.DataFrame(task_rows)
    metadata_dir = metadata_directory(config, project_root)
    task_path = metadata_dir / config["metadata"]["raster_task_manifest"]
    task_frame.to_csv(task_path, index=False)
    print(
        json.dumps(
            {
                "preflight_status": preflight["status"],
                "submitted_or_existing_tasks": len(task_frame),
                "task_manifest": str(task_path),
            },
            indent=2,
        )
    )
    return task_frame


def refresh_tasks(config_path: Path) -> pd.DataFrame:
    """Refresh raster task states with one Earth Engine task-list call."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    initialize_earth_engine(
        config["project"]["earth_engine_project"]
    )
    task_path = (
        metadata_directory(config, project_root)
        / config["metadata"]["raster_task_manifest"]
    )

    if not task_path.is_file():
        raise FileNotFoundError(
            "Submit raster products before requesting task status."
        )

    frame = pd.read_csv(task_path, keep_default_na=False)
    statuses = {
        task.id: task.status()
        for task in ee.batch.Task.list()
    }

    for index, row in frame.iterrows():
        task_id = str(row["task_id"]).strip()

        if not task_id:
            continue

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


def finalize_rasters(config_path: Path) -> dict[str, Any]:
    """Validate completed assets and freeze the raster-product manifest."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    grid = load_grid_specification(
        resolve_project_path(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    frame = refresh_tasks(config_path)
    incomplete = frame[
        ~frame["state"].astype(str).str.upper().isin(PASS_STATES)
    ]

    if not incomplete.empty:
        raise RuntimeError(
            "Raster products are not complete:\n"
            + incomplete[
                ["description", "state", "error_message"]
            ].to_string(index=False)
        )

    for row in frame.itertuples(index=False):
        expected_bands = [
            value.strip()
            for value in str(row.band_names).split(",")
            if value.strip()
        ]
        validate_asset_grid(str(row.asset_id), expected_bands, grid)

    metadata_dir = metadata_directory(config, project_root)
    output_path = (
        metadata_dir / config["metadata"]["raster_output_manifest"]
    )
    output = frame[
        [
            "product_type",
            "epoch",
            "period_start",
            "period_end",
            "asset_id",
            "band_names",
            "state",
            "recipe_sha256",
        ]
    ].copy()
    output["state"] = "PASS"
    output.to_csv(output_path, index=False)
    version_path = (
        metadata_dir / config["metadata"]["raster_products_version"]
    )
    version = {
        "pipeline_stage": "final_dataset_rasters",
        "pipeline_version": int(config["version"]),
        "dataset_id": config["release"]["dataset_id"],
        "mapping_method": config["mapping"]["selected_index"],
        "mapping_selection_status": config["release"][
            "mapping_selection_status"
        ],
        "epochs": [int(value) for value in config["mapping"]["epochs"]],
        "state_count": int((output["product_type"] == "state").sum()),
        "transition_count": int(
            (output["product_type"] == "transition").sum()
        ),
        "tracking_support_count": int(
            (output["product_type"] == "tracking_support").sum()
        ),
        "grid_sha256": stable_object_hash(grid),
        "output_manifest_sha256": sha256_file(output_path),
    }
    version["stable_signature"] = stable_object_hash(version)
    write_json(version_path, version)
    result = {
        "status": "PASS",
        "raster_output_manifest": str(output_path),
        "raster_products_version": str(version_path),
        "stable_signature": version["stable_signature"],
    }
    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the raster-product command line."""
    parser = argparse.ArgumentParser(
        description="Build final states, support and transitions."
    )
    parser.add_argument("--config", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight-only", action="store_true")
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--status-only", action="store_true")
    actions.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected raster-product action."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()

    if arguments.preflight_only:
        run_preflight(config_path)
    elif arguments.submit:
        submit_rasters(config_path)
    elif arguments.status_only:
        refresh_tasks(config_path)
    elif arguments.finalize:
        finalize_rasters(config_path)


if __name__ == "__main__":
    main()
