"""Download, date, checksum and clip the current Geofabrik Cameroon snapshot.

The implementation uses Geofabrik's free GeoPackage archive to avoid a hard
runtime dependency on an OSM parser. Spatial filtering is applied while reading
large source layers, followed by exact clipping to the Day 1 context geometry.
"""

from __future__ import annotations

import argparse
import email.utils
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.analysis.orchestration.common import (
    find_project_root,
    load_yaml,
    resolve_project_path,
    sha256_file,
    write_json,
)


def source_snapshot_date(last_modified: str | None) -> str:
    """Convert an HTTP Last-Modified header to a stable ISO snapshot date."""
    if last_modified:
        parsed = email.utils.parsedate_to_datetime(last_modified)
        return parsed.astimezone(timezone.utc).date().isoformat()

    return datetime.now(timezone.utc).date().isoformat()


def download_geofabrik_archive(
    url: str,
    raw_directory: Path,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    """Download the current Geofabrik archive and preserve a dated filename."""
    raw_directory.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, method="GET")

    with urllib.request.urlopen(request) as response:
        last_modified = response.headers.get("Last-Modified")
        snapshot_date = source_snapshot_date(last_modified)
        dated_name = f"cameroon_{snapshot_date.replace('-', '')}_free.gpkg.zip"
        destination = raw_directory / dated_name

        if destination.exists() and not force:
            metadata = {
                "snapshot_date": snapshot_date,
                "source_last_modified": last_modified,
                "download_skipped_existing_file": True,
            }
            return destination, metadata

        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)

    metadata = {
        "snapshot_date": snapshot_date,
        "source_last_modified": last_modified,
        "download_skipped_existing_file": False,
    }
    return destination, metadata


def extract_source_geopackage(archive: Path, raw_directory: Path) -> Path:
    """Extract and return the single GeoPackage contained in the archive."""
    extraction_directory = raw_directory / archive.stem
    extraction_directory.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extraction_directory)

    candidates = list(extraction_directory.rglob("*.gpkg"))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one GeoPackage in {archive}, found {len(candidates)}."
        )

    return candidates[0]


def read_layer_with_bbox(
    source: Path,
    layer: str,
    bbox: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Read only features intersecting a WGS84 bounding box when supported."""
    try:
        return gpd.read_file(source, layer=layer, bbox=bbox, engine="pyogrio")
    except Exception as error:
        raise RuntimeError(
            f"Could not read layer '{layer}' from {source}. Confirm that the "
            "Geofabrik layer names in day3_sources.yaml match the archive."
        ) from error


def clip_and_annotate(
    frame: gpd.GeoDataFrame,
    context: gpd.GeoDataFrame,
    output_crs: str,
    snapshot_date: str,
) -> gpd.GeoDataFrame:
    """Clip one OSM layer and attach explicit contemporary-use metadata."""
    if frame.empty:
        return gpd.GeoDataFrame(frame, geometry="geometry", crs=output_crs)

    projected = frame.to_crs(output_crs)
    context_projected = context.to_crs(output_crs)
    clipped = gpd.clip(projected, context_projected)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    clipped["osm_snapshot_date"] = snapshot_date
    clipped["historical_validity"] = "contemporary_only"
    clipped["primary_historical_predictor"] = False
    return clipped


def filter_classes(
    frame: gpd.GeoDataFrame,
    class_field: str,
    selected_classes: list[str],
) -> gpd.GeoDataFrame:
    """Retain only explicitly configured OSM feature classes."""
    if class_field not in frame.columns:
        raise KeyError(
            f"Configured OSM class field '{class_field}' is absent. "
            f"Available columns: {list(frame.columns)}"
        )

    return frame[frame[class_field].astype(str).isin(selected_classes)].copy()


def run_osm(config_path: Path, force_download: bool) -> dict[str, Any]:
    """Execute the complete current-snapshot OSM extraction workflow."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    osm_config = config["osm"]

    raw_directory = resolve_project_path(osm_config["raw_directory"], project_root)
    archive, download_metadata = download_geofabrik_archive(
        osm_config["source_url"],
        raw_directory,
        force_download,
    )
    source_gpkg = extract_source_geopackage(archive, raw_directory)

    context_path = resolve_project_path(config["regions"]["context"], project_root)
    context = gpd.read_file(context_path)

    if context.crs is None or len(context) != 1:
        raise ValueError("The context boundary must be one valid georeferenced feature.")

    context_wgs84 = context.to_crs("EPSG:4326")
    bbox = tuple(float(value) for value in context_wgs84.total_bounds)
    layers = osm_config["layers"]
    class_field = osm_config["class_field"]

    roads = read_layer_with_bbox(source_gpkg, layers["roads"], bbox)
    roads = filter_classes(
        roads,
        class_field,
        list(osm_config["selected_road_classes"]),
    )
    railways = read_layer_with_bbox(source_gpkg, layers["railways"], bbox)
    railways = filter_classes(
        railways,
        class_field,
        list(osm_config["selected_railway_classes"]),
    )
    buildings = read_layer_with_bbox(source_gpkg, layers["buildings"], bbox)

    snapshot_date = download_metadata["snapshot_date"]
    output_crs = osm_config["output_crs"]
    outputs = {
        "roads_major": clip_and_annotate(
            roads, context, output_crs, snapshot_date
        ),
        "railways": clip_and_annotate(
            railways, context, output_crs, snapshot_date
        ),
        "buildings_reference": clip_and_annotate(
            buildings, context, output_crs, snapshot_date
        ),
    }

    output_path = resolve_project_path(osm_config["processed_output"], project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    for layer_name, frame in outputs.items():
        frame.to_file(output_path, layer=layer_name, driver="GPKG")

    checksum = sha256_file(archive)
    checksum_path = resolve_project_path(osm_config["source_checksum"], project_root)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")

    metadata = {
        "provider": osm_config["provider"],
        "source_dataset": "OpenStreetMap Cameroon extract",
        "source_format": "Geofabrik free GeoPackage archive",
        "source_url": osm_config["source_url"],
        "snapshot_date": snapshot_date,
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "source_filename": archive.name,
        "source_sha256": checksum,
        "licence": "Open Database Licence (ODbL)",
        "temporal_interpretation": "current snapshot only",
        "processed_crs": output_crs,
        "processed_output": str(output_path),
        "feature_counts": {
            name: int(len(frame)) for name, frame in outputs.items()
        },
        **download_metadata,
    }
    write_json(
        resolve_project_path(osm_config["source_metadata"], project_root),
        metadata,
    )

    print(json.dumps(metadata, indent=2))
    return metadata


def parse_arguments() -> argparse.Namespace:
    """Parse current-snapshot OSM extraction arguments."""
    parser = argparse.ArgumentParser(
        description="Download and clip the current Geofabrik Cameroon extract."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the OSM extraction command."""
    arguments = parse_arguments()
    run_osm(arguments.config.resolve(), arguments.force_download)


if __name__ == "__main__":
    main()
