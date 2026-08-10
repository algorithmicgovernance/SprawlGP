"""Audit corrected GHSL population-density conversion in Earth Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ee
import geopandas as gpd
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping in {path}.")
    return payload


def resolve(path_value: str, project_root: Path) -> Path:
    """Resolve one project-relative path."""
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def core_geometry(path: Path) -> ee.Geometry:
    """Load the Yaoundé core boundary as an Earth Engine geometry."""
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError("The core boundary has no CRS.")
    frame = frame.to_crs("EPSG:4326")
    geometry = (
        frame.geometry.union_all()
        if hasattr(frame.geometry, "union_all")
        else frame.geometry.unary_union
    )
    if geometry.is_empty:
        raise ValueError("The core boundary is empty.")
    return ee.Geometry(geometry.__geo_interface__)


def corrected_density(
    epoch: int,
    config: dict[str, Any],
) -> tuple[ee.Image, ee.Image, ee.Projection]:
    """Return native population count, corrected density and projection."""
    population = config["population"]
    asset_id = f"{population['collection'].rstrip('/')}/{int(epoch)}"

    count = (
        ee.Image(asset_id)
        .select(population["band"])
        .toFloat()
    )
    native_projection = count.projection()
    native_area = ee.Image.pixelArea().reproject(native_projection)

    density = (
        count.divide(native_area)
        .multiply(1_000_000.0)
        .rename("population_density_t")
        .setDefaultProjection(native_projection)
        .unmask(0)
        .toFloat()
    )
    return count, density, native_projection


def reduce_value(
    image: ee.Image,
    band: str,
    reducer: ee.Reducer,
    geometry: ee.Geometry,
    **kwargs: Any,
) -> float:
    """Reduce one band and return a numeric scalar."""
    value = (
        image.select(band)
        .reduceRegion(
            reducer=reducer,
            geometry=geometry,
            maxPixels=20_000_000,
            tileScale=4,
            **kwargs,
        )
        .get(band)
        .getInfo()
    )
    return float(value or 0.0)


def audit(config_path: Path) -> pd.DataFrame:
    """Compare native GHSL totals with 30 m density integration."""
    project_root = Path.cwd()
    config = load_yaml(config_path)

    ee.Initialize(project=config["project"]["earth_engine_project"])

    grid = load_json(
        resolve(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    core = core_geometry(
        resolve(
            config["inputs"]["core_boundary"],
            project_root,
        )
    )

    epochs = [
        int(epoch)
        for epoch in config["mapping"]["epochs"]
        if int(epoch) <= 2025
    ]

    rows = []

    for epoch in epochs:
        count, density, native_projection = corrected_density(
            epoch,
            config,
        )
        nominal_scale = float(
            native_projection.nominalScale().getInfo()
        )
        projection_info = native_projection.getInfo()

        native_total = reduce_value(
            count.rename("population_count"),
            "population_count",
            ee.Reducer.sum(),
            core,
            crs=native_projection,
            scale=nominal_scale,
        )
        max_count = reduce_value(
            count.rename("population_count"),
            "population_count",
            ee.Reducer.max(),
            core,
            crs=native_projection,
            scale=nominal_scale,
        )
        max_density = reduce_value(
            density,
            "population_density_t",
            ee.Reducer.max(),
            core,
            crs=native_projection,
            scale=nominal_scale,
        )

        reconstructed = (
            density
            .multiply(ee.Image.pixelArea())
            .divide(1_000_000.0)
            .rename("population_from_density")
        )
        reconstructed_total = reduce_value(
            reconstructed,
            "population_from_density",
            ee.Reducer.sum(),
            core,
            crs=str(grid["crs"]),
            crsTransform=[
                float(value)
                for value in grid["transform"]
            ],
        )

        difference_pct = (
            (reconstructed_total - native_total)
            / native_total
            * 100.0
            if native_total > 0
            else 0.0
        )

        rows.append(
            {
                "epoch": epoch,
                "native_crs": projection_info.get("crs", ""),
                "native_nominal_scale_m": nominal_scale,
                "native_population_total_core": native_total,
                "reconstructed_population_total_30m_core": reconstructed_total,
                "relative_difference_pct": difference_pct,
                "maximum_native_population_count": max_count,
                "maximum_corrected_density_per_km2": max_density,
                "maximum_implied_people_per_30m_cell": (
                    max_density * 0.0009
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run the audit and write CSV and JSON outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_dataset.yaml"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/population_audit"),
    )
    args = parser.parse_args()

    frame = audit(args.config.resolve())
    args.output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_directory / "population_audit.csv"
    json_path = args.output_directory / "population_audit.json"

    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )

    print(frame.round(3).to_string(index=False))
    print()
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
