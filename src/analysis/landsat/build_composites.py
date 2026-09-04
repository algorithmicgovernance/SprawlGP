"""Submit exact-scene Landsat composites, count layers and terrain assets.

The script never searches Landsat collections. It reloads only the exact asset
IDs frozen during Day 2, reuses the Day 2 QA-mask implementation, harmonises
sensor bands, applies Collection 2 scaling and submits asynchronous Earth
Engine exports on the exact Day 1 grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ee
import pandas as pd
from src.analysis.landsat.qa_masks import landsat_valid_mask
from src.analysis.orchestration.common import (
    asset_exists,
    ensure_asset_folder,
    exact_grid_region,
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_yaml,
    metadata_directory,
    resolve_project_path,
    stable_object_hash,
    write_json,
    write_yaml,
)
from src.analysis.orchestration.preflight import asset_id_column, run_preflight

LANDSAT_BANDS = {
    "LT05": {
        "SR_B1": "blue",
        "SR_B2": "green",
        "SR_B3": "red",
        "SR_B4": "nir",
        "SR_B5": "swir1",
        "SR_B7": "swir2",
    },
    "LE07": {
        "SR_B1": "blue",
        "SR_B2": "green",
        "SR_B3": "red",
        "SR_B4": "nir",
        "SR_B5": "swir1",
        "SR_B7": "swir2",
    },
    "LC08": {
        "SR_B2": "blue",
        "SR_B3": "green",
        "SR_B4": "red",
        "SR_B5": "nir",
        "SR_B6": "swir1",
        "SR_B7": "swir2",
    },
    "LC09": {
        "SR_B2": "blue",
        "SR_B3": "green",
        "SR_B4": "red",
        "SR_B5": "nir",
        "SR_B6": "swir1",
        "SR_B7": "swir2",
    },
}


def prepare_surface_reflectance(
    image: ee.Image,
    sensor_key: str,
    day3_config: dict[str, Any],
    day2_config: dict[str, Any],
) -> tuple[ee.Image, ee.Image]:
    """Scale, mask and harmonise one exact selected Landsat scene."""
    if sensor_key not in LANDSAT_BANDS:
        raise ValueError(f"Unsupported sensor key: {sensor_key}")

    mapping = LANDSAT_BANDS[sensor_key]
    valid_mask = landsat_valid_mask(
        image,
        sensor_key,
        day2_config["collections"],
        day2_config["quality_mask"],
    )

    prepared = (
        image.select(list(mapping), list(mapping.values()))
        .multiply(float(day3_config["landsat"]["optical_scale"]))
        .add(float(day3_config["landsat"]["optical_offset"]))
        .updateMask(valid_mask)
        .resample("bilinear")
        .toFloat()
        .copyProperties(image, image.propertyNames())
    )

    return prepared, valid_mask.rename("valid")


def build_epoch_products(
    epoch_manifest: pd.DataFrame,
    id_column: str,
    day3_config: dict[str, Any],
    day2_config: dict[str, Any],
) -> tuple[ee.Image, ee.Image, dict[str, int]]:
    """Build one six-band median composite and multi-band count image."""
    prepared_images: list[ee.Image] = []
    all_masks: list[ee.Image] = []
    masks_by_sensor: dict[str, list[ee.Image]] = {}

    for row in epoch_manifest.itertuples(index=False):
        values = row._asdict()
        sensor_key = str(values["sensor_key"])
        asset_id = str(values[id_column])
        image = ee.Image(asset_id)
        prepared, valid_mask = prepare_surface_reflectance(
            image,
            sensor_key,
            day3_config,
            day2_config,
        )
        prepared_images.append(prepared)
        all_masks.append(valid_mask)
        masks_by_sensor.setdefault(sensor_key, []).append(valid_mask)

    if not prepared_images:
        raise ValueError("An epoch cannot be composited without selected scenes.")

    valid_count = (
        ee.ImageCollection.fromImages(all_masks)
        .sum()
        .rename("valid_observation_count")
        .unmask(0)
        .toUint16()
    )

    count_image = valid_count

    if len(masks_by_sensor) > 1:
        for sensor_key in sorted(masks_by_sensor):
            sensor_count = (
                ee.ImageCollection.fromImages(masks_by_sensor[sensor_key])
                .sum()
                .rename(f"valid_count_{sensor_key.lower()}")
                .unmask(0)
                .toUint16()
            )
            count_image = count_image.addBands(sensor_count)

    common_bands = list(day3_config["landsat"]["common_bands"])
    composite = (
        ee.ImageCollection.fromImages(prepared_images)
        .median()
        .select(common_bands)
        .updateMask(valid_count.gt(0))
        .toFloat()
    )

    scene_counts = {
        sensor: len(images)
        for sensor, images in sorted(masks_by_sensor.items())
    }
    return composite, count_image, scene_counts


def start_image_export(
    image: ee.Image,
    description: str,
    asset_id: str,
    grid: dict[str, Any],
    config: dict[str, Any],
    pyramiding_policy: str,
) -> dict[str, Any]:
    """Start one exact-grid Earth Engine asset export or record an existing asset."""
    overwrite = bool(config["exports"]["overwrite_existing_assets"])

    if asset_exists(asset_id) and not overwrite:
        return {
            "task_id": "",
            "state": "EXISTS",
            "description": description,
            "asset_id": asset_id,
        }

    task = ee.batch.Export.image.toAsset(
        image=image,
        description=description,
        assetId=asset_id,
        pyramidingPolicy={".default": pyramiding_policy},
        region=exact_grid_region(grid),
        crs=str(grid["crs"]),
        crsTransform=[float(value) for value in grid["transform"]],
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


def build_terrain_image(config: dict[str, Any]) -> ee.Image:
    """Create the static elevation-and-slope predictor image."""
    dem = ee.Image(config["srtm"]["dataset"]).select(
        config["srtm"]["elevation_band"]
    )
    elevation = dem.rename("elevation_m")
    slope = ee.Terrain.slope(dem).rename("slope_degrees")
    return ee.Image.cat([elevation, slope]).resample("bilinear").toFloat()


def build_ghsl_manifest(
    config: dict[str, Any],
    metadata_dir: Path,
) -> None:
    """Record exact GHSL public assets without resampling or exporting them."""
    rows: list[dict[str, Any]] = []

    for epoch in config["ghsl"]["epochs"]:
        rows.extend(
            [
                {
                    "source": "GHSL_BUILT_SURFACE",
                    "epoch": int(epoch),
                    "asset_id": f"{config['ghsl']['built_collection']}/{epoch}",
                    "band": config["ghsl"]["built_band"],
                    "representation": "native_100m_grid",
                    "resampled_to_30m": False,
                    "ground_truth": False,
                },
                {
                    "source": "GHSL_POPULATION",
                    "epoch": int(epoch),
                    "asset_id": (
                        f"{config['ghsl']['population_collection']}/{epoch}"
                    ),
                    "band": config["ghsl"]["population_band"],
                    "representation": "native_100m_grid",
                    "resampled_to_30m": False,
                    "ground_truth": False,
                },
            ]
        )

    pd.DataFrame(rows).to_csv(
        metadata_dir / config["metadata"]["ghsl_epoch_manifest"],
        index=False,
    )

    sample_assets = {
        "built_surface": rows[0]["asset_id"],
        "population": rows[1]["asset_id"],
    }
    native_grid: dict[str, Any] = {
        "preserve_native_grid": True,
        "resampled_to_project_grid": False,
        "linkage_rule": (
            "Aggregate future Landsat-derived built-up area to the GHSL "
            "native grid before comparison."
        ),
        "assets": {},
    }

    for source_name, asset_id in sample_assets.items():
        image = ee.Image(asset_id)
        projection = image.select(0).projection().getInfo()
        native_grid["assets"][source_name] = projection

    write_json(
        metadata_dir / config["metadata"]["ghsl_native_grid"],
        native_grid,
    )


def write_grid_linkage(
    config: dict[str, Any],
    grid: dict[str, Any],
    metadata_dir: Path,
) -> None:
    """Document whether every source is aligned or linked to the project grid."""
    linkage = {
        "grid_version": 1,
        "project_grid": {
            "crs": grid["crs"],
            "resolution_m": grid["resolution_m"],
            "width": grid["width"],
            "height": grid["height"],
            "transform": grid["transform"],
        },
        "sources": {
            "landsat": {
                "representation": "Earth Engine raster assets",
                "alignment": "exact_project_grid",
                "reflectance_resampling": "bilinear",
                "count_resampling": "nearest_default",
            },
            "srtm": {
                "representation": "Earth Engine raster asset",
                "alignment": "exact_project_grid",
                "resampling": "bilinear",
            },
            "ghsl": {
                "representation": "public Earth Engine raster assets",
                "alignment": "native_100m_grid",
                "resampled_to_30m": False,
            },
            "osm": {
                "representation": "GeoPackage vector layers",
                "crs": config["osm"]["output_crs"],
                "clipped_to": "yaounde_context_buffer",
                "historical_validity": "contemporary_only",
            },
        },
    }
    write_yaml(
        metadata_dir / config["metadata"]["grid_linkage"],
        linkage,
    )


def run_submission(config_path: Path, submit: bool) -> pd.DataFrame:
    """Validate dependencies and submit all minimal Day 3 Earth Engine tasks."""
    if not submit:
        raise ValueError("Pass --submit to create Earth Engine export tasks.")

    preflight = run_preflight(config_path)
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    landsat_only = bool(config.get("orchestration", {}).get("landsat_only", False))
    day2_config = load_yaml(
        resolve_project_path(config["inputs"]["landsat_catalog_config"], project_root)
    )
    grid = load_grid_specification(
        resolve_project_path(config["inputs"]["grid_specification"], project_root)
    )
    manifest = pd.read_csv(
        resolve_project_path(
            config["inputs"]["selected_scene_manifest"], project_root
        )
    )
    id_column = asset_id_column(manifest)

    initialize_earth_engine(config["project"]["earth_engine_project"])

    asset_root = config["exports"]["asset_root"].rstrip("/")
    landsat_folder = f"{asset_root}/{config['exports']['landsat_folder']}"
    ensure_asset_folder(asset_root)
    ensure_asset_folder(landsat_folder)

    if not landsat_only:
        terrain_folder = f"{asset_root}/{config['exports']['terrain_folder']}"
        ensure_asset_folder(terrain_folder)

    tasks: list[dict[str, Any]] = []
    task_prefix = str(config["exports"].get("task_prefix", "sprawlgp"))

    for epoch in config["landsat"]["epochs"]:
        epoch_manifest = manifest[manifest["epoch"].astype(int) == int(epoch)]
        
        # Some configured epochs may have no scenes in the frozen Day 2
        # manifest. This can reflect archive availability, the selected temporal
        # window, quality requirements or catalogue-selection decisions.
        if epoch_manifest.empty:
            print(
                f"Skipping epoch {epoch}: no scenes are available in the "
                "frozen Day 2 selected-scene manifest."
            )
            continue
        
        composite, count_image, scene_counts = build_epoch_products(
            epoch_manifest,
            id_column,
            config,
            day2_config,
        )
        scene_ids = sorted(epoch_manifest[id_column].astype(str).tolist())
        scene_hash = stable_object_hash(scene_ids)

        composite_asset = f"{landsat_folder}/composite_{epoch}"
        count_asset = f"{landsat_folder}/valid_count_{epoch}"

        composite = composite.set(
            {
                "epoch": int(epoch),
                "selected_scene_count": int(len(epoch_manifest)),
                "selected_scene_ids_sha256": scene_hash,
                "scene_counts_by_sensor": json.dumps(scene_counts, sort_keys=True),
                "day3_version": int(config["version"]),
            }
        )
        count_image = count_image.set(
            {
                "epoch": int(epoch),
                "selected_scene_count": int(len(epoch_manifest)),
                "selected_scene_ids_sha256": scene_hash,
                "day3_version": int(config["version"]),
            }
        )

        composite_task = start_image_export(
            composite,
            f"{task_prefix}_composite_{epoch}",
            composite_asset,
            grid,
            config,
            config["exports"]["reflectance_pyramiding_policy"],
        )
        composite_task.update(
            {
                "product_type": "landsat_composite",
                "epoch": int(epoch),
                "selected_scene_count": int(len(epoch_manifest)),
                "selected_scene_ids_sha256": scene_hash,
            }
        )
        tasks.append(composite_task)

        count_task = start_image_export(
            count_image,
            f"{task_prefix}_valid_count_{epoch}",
            count_asset,
            grid,
            config,
            config["exports"]["count_pyramiding_policy"],
        )
        count_task.update(
            {
                "product_type": "landsat_valid_count",
                "epoch": int(epoch),
                "selected_scene_count": int(len(epoch_manifest)),
                "selected_scene_ids_sha256": scene_hash,
            }
        )
        tasks.append(count_task)

    if not landsat_only:
        terrain_asset = f"{terrain_folder}/elevation_slope"
        terrain_task = start_image_export(
            build_terrain_image(config),
            "sprawlgp_elevation_slope",
            terrain_asset,
            grid,
            config,
            config["exports"]["terrain_pyramiding_policy"],
        )
        terrain_task.update(
            {
                "product_type": "terrain",
                "epoch": "static",
                "selected_scene_count": 0,
                "selected_scene_ids_sha256": "",
            }
        )
        tasks.append(terrain_task)

    metadata_dir = metadata_directory(config, project_root)
    task_frame = pd.DataFrame(tasks)
    task_path = resolve_project_path(config["exports"]["task_manifest"], project_root)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_frame.to_csv(task_path, index=False)

    if not landsat_only:
        build_ghsl_manifest(config, metadata_dir)
        write_grid_linkage(config, grid, metadata_dir)
        auxiliary = pd.DataFrame(
            [
                {
                    "source": "SRTM",
                    "asset_id": terrain_asset,
                    "representation": "exact_project_grid",
                    "temporal_representation": "static_approximately_2000",
                    "intended_role": "elevation_and_slope_predictors",
                },
                {
                    "source": "GHSL",
                    "asset_id": "see_ghsl_epoch_manifest.csv",
                    "representation": "native_100m_grid",
                    "temporal_representation": (
                        f"{len(config['ghsl']['epochs'])}_configured_epochs"
                    ),
                    "intended_role": "auxiliary_benchmark_and_population",
                },
                {
                    "source": "OpenStreetMap",
                    "asset_id": config["osm"]["processed_output"],
                    "representation": "vector_EPSG32632",
                    "temporal_representation": "current_snapshot",
                    "intended_role": "recent_validation_and_accessibility",
                },
            ]
        )
        auxiliary.to_csv(
            metadata_dir / config["metadata"]["auxiliary_source_manifest"],
            index=False,
        )

    print(
        json.dumps(
            {
                "preflight_status": preflight["status"],
                "submitted_or_existing_tasks": len(task_frame),
                "task_manifest": config["exports"]["task_manifest"],
            },
            indent=2,
        )
    )
    return task_frame


def parse_arguments() -> argparse.Namespace:
    """Parse the Day 3 export-submission command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit exact-scene Landsat and terrain assets."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the export-submission command."""
    arguments = parse_arguments()
    run_submission(arguments.config.resolve(), arguments.submit)


if __name__ == "__main__":
    main()
