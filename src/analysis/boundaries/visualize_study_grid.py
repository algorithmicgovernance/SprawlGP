"""Generate reproducible visual quality-control figures for the Day 1 grid.

The figures compare the selected Yaoundé ADM3 units with ADM2 Mfoundi, display
all constructed study geometries, show the rasterised core and context masks,
and provide a detailed grid zoom with example stable cell identifiers.

The script is intended for reproducible quality assurance. QGIS may still be
used for interactive inspection, but these exported figures document the exact
outputs produced by the Python pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml


def load_configuration(config_path: Path) -> tuple[dict[str, Any], Path]:
    """Load the YAML configuration and infer the repository root.

    Args:
        config_path: Path to ``configs/study_area.yaml``.

    Returns:
        Parsed configuration and absolute repository-root path.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML document is empty or invalid.
    """
    resolved_config_path = config_path.expanduser().resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError("The study-area configuration must contain a YAML mapping.")

    return configuration, resolved_config_path.parent.parent


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    """Resolve an absolute or project-relative configured path.

    Args:
        project_root: Absolute path to the repository root.
        path_value: Path value read from the YAML configuration.

    Returns:
        Absolute normalised path.
    """
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save a Matplotlib figure as a PNG and release its memory.

    Args:
        figure: Figure to save.
        output_path: Destination PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def select_configured_features(
    geodataframe: gpd.GeoDataFrame,
    name_field: str,
    selected_names: list[str],
    pcode_field: str | None = None,
    selected_pcodes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Select configured administrative features using P-codes or names.

    P-codes are used when both the field and values are available. Names remain
    the fallback so the visualisation script also works with simpler boundary
    configurations.

    Args:
        geodataframe: Source administrative features.
        name_field: Field containing administrative-unit names.
        selected_names: Exact names expected in the source.
        pcode_field: Optional field containing stable administrative P-codes.
        selected_pcodes: Optional P-codes to select.

    Returns:
        Copy of the selected features.

    Raises:
        KeyError: If the required selection field is absent.
        ValueError: If no matching feature is found.
    """
    if pcode_field and selected_pcodes:
        if pcode_field not in geodataframe.columns:
            raise KeyError(f"Configured P-code field is absent: {pcode_field}")
        selected = geodataframe[
            geodataframe[pcode_field].astype(str).isin(selected_pcodes)
        ].copy()
    else:
        if name_field not in geodataframe.columns:
            raise KeyError(f"Configured name field is absent: {name_field}")
        selected = geodataframe[
            geodataframe[name_field].astype(str).isin(selected_names)
        ].copy()

    if selected.empty:
        raise ValueError("The configured administrative selection returned no features.")

    return selected


def load_visualisation_inputs(
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Load all vector, raster, and cell-table inputs needed by the figures.

    Args:
        config: Parsed study-area configuration.
        project_root: Absolute repository-root path.

    Returns:
        Dictionary containing selected ADM3 units, ADM2 Mfoundi, constructed
        geometries, raster masks, raster metadata, and the cell table.
    """
    boundary = config["boundary"]
    comparison = config["comparison_boundary"]
    outputs = config["outputs"]
    analysis_crs = config["projection"]["analysis_crs"]

    adm3 = gpd.read_file(resolve_project_path(project_root, boundary["source_file"]))
    selected_adm3 = select_configured_features(
        adm3,
        name_field=str(boundary["name_field"]),
        selected_names=[str(value) for value in boundary["selected_names"]],
        pcode_field=boundary.get("pcode_field"),
        selected_pcodes=[str(value) for value in boundary.get("selected_pcodes", [])],
    ).to_crs(analysis_crs)

    adm2 = gpd.read_file(resolve_project_path(project_root, boundary["comparison_file"]))
    selected_adm2 = select_configured_features(
        adm2,
        name_field=str(comparison["name_field"]),
        selected_names=[str(comparison["selected_name"])],
        pcode_field=comparison.get("pcode_field"),
        selected_pcodes=(
            [str(comparison["selected_pcode"])]
            if comparison.get("selected_pcode")
            else None
        ),
    ).to_crs(analysis_crs)

    if len(selected_adm2) != 1:
        raise ValueError(f"Expected one Mfoundi ADM2 feature, found {len(selected_adm2)}.")

    core = gpd.read_file(resolve_project_path(project_root, outputs["core_boundary"]))
    context = gpd.read_file(resolve_project_path(project_root, outputs["context_boundary"]))
    hull = gpd.read_file(resolve_project_path(project_root, outputs["convex_hull"]))
    cells = pd.read_parquet(resolve_project_path(project_root, outputs["cell_table"]))

    core_mask_path = resolve_project_path(project_root, outputs["core_mask"])
    context_mask_path = resolve_project_path(project_root, outputs["context_mask"])

    with rasterio.open(context_mask_path) as raster:
        context_mask = raster.read(1)
        raster_bounds = raster.bounds
        raster_transform = raster.transform
        raster_crs = raster.crs

    with rasterio.open(core_mask_path) as raster:
        core_mask = raster.read(1)

    return {
        "selected_adm3": selected_adm3,
        "selected_adm2": selected_adm2,
        "core": core,
        "context": context,
        "hull": hull,
        "cells": cells,
        "core_mask": core_mask,
        "context_mask": context_mask,
        "raster_bounds": raster_bounds,
        "raster_transform": raster_transform,
        "raster_crs": raster_crs,
    }


def plot_boundary_overview(inputs: dict[str, Any], output_path: Path) -> None:
    """Plot selected ADM3 units and all constructed study geometries together.

    The overview makes it possible to inspect whether the seven subdivisions form
    a coherent core, whether Mfoundi agrees with that core, and how the context
    buffer and convex hull differ from the reporting boundary.

    Args:
        inputs: Dictionary returned by :func:`load_visualisation_inputs`.
        output_path: Destination PNG path.
    """
    figure, axis = plt.subplots(figsize=(10, 10))

    inputs["context"].boundary.plot(
        ax=axis,
        linewidth=1.2,
        linestyle=":",
        label="5 km context buffer",
    )
    inputs["hull"].boundary.plot(
        ax=axis,
        linewidth=1.2,
        linestyle="--",
        label="Convex hull",
    )
    inputs["selected_adm2"].boundary.plot(
        ax=axis,
        linewidth=2.5,
        label="ADM2 Mfoundi",
    )
    inputs["selected_adm3"].boundary.plot(
        ax=axis,
        linewidth=1.0,
        label="Selected ADM3 units",
    )
    inputs["core"].boundary.plot(
        ax=axis,
        linewidth=2.0,
        label="Dissolved Yaoundé core",
    )

    axis.set_title("Yaoundé study boundaries")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_aspect("equal")
    axis.legend()
    save_figure(figure, output_path)


def plot_adm2_adm3_comparison(
    inputs: dict[str, Any],
    output_path: Path,
) -> dict[str, float]:
    """Plot and quantify the difference between ADM3 union and ADM2 Mfoundi.

    Args:
        inputs: Dictionary returned by :func:`load_visualisation_inputs`.
        output_path: Destination PNG path.

    Returns:
        Area metrics describing the overlap and symmetric difference.
    """
    adm3_union = inputs["selected_adm3"].geometry.union_all()
    adm2_geometry = inputs["selected_adm2"].geometry.union_all()
    symmetric_difference = adm3_union.symmetric_difference(adm2_geometry)
    reference_area = float(adm2_geometry.area)
    difference_ratio = (
        float(symmetric_difference.area) / reference_area if reference_area > 0 else math.nan
    )

    figure, axis = plt.subplots(figsize=(10, 10))
    inputs["selected_adm2"].boundary.plot(
        ax=axis,
        linewidth=2.5,
        label="ADM2 Mfoundi",
    )
    inputs["selected_adm3"].boundary.plot(
        ax=axis,
        linewidth=1.0,
        label="Seven ADM3 units",
    )

    if not symmetric_difference.is_empty:
        difference_frame = gpd.GeoDataFrame(
            {"comparison": ["symmetric_difference"]},
            geometry=[symmetric_difference],
            crs=inputs["selected_adm2"].crs,
        )
        difference_frame.plot(ax=axis, alpha=0.5, label="Difference")

    axis.set_title(
        "ADM3 union compared with ADM2 Mfoundi\n"
        f"Symmetric difference ratio: {difference_ratio:.6%}"
    )
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_aspect("equal")
    axis.legend()
    save_figure(figure, output_path)

    return {
        "adm3_union_area_km2": float(adm3_union.area) / 1_000_000,
        "mfoundi_area_km2": reference_area / 1_000_000,
        "symmetric_difference_area_km2": (
            float(symmetric_difference.area) / 1_000_000
        ),
        "symmetric_difference_ratio": difference_ratio,
    }


def plot_core_context_masks(inputs: dict[str, Any], output_path: Path) -> None:
    """Display the complete 30 m context and core masks over the vector core.

    Args:
        inputs: Dictionary returned by :func:`load_visualisation_inputs`.
        output_path: Destination PNG path.
    """
    bounds = inputs["raster_bounds"]
    raster_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    figure, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(
        np.ma.masked_where(inputs["context_mask"] == 0, inputs["context_mask"]),
        extent=raster_extent,
        origin="upper",
        alpha=0.4,
    )
    axis.imshow(
        np.ma.masked_where(inputs["core_mask"] == 0, inputs["core_mask"]),
        extent=raster_extent,
        origin="upper",
        alpha=0.8,
    )
    inputs["core"].boundary.plot(ax=axis, linewidth=1.5)

    axis.set_title("30 m context and core masks")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_aspect("equal")
    save_figure(figure, output_path)


def choose_boundary_zoom_point(core_geometry: Any) -> tuple[float, float]:
    """Choose a reproducible point on the core boundary for the grid-detail view.

    A boundary point is preferable to the polygon centroid because it allows the
    zoomed figure to show both included and excluded grid cells around the exact
    administrative edge.

    Args:
        core_geometry: Dissolved Yaoundé core geometry.

    Returns:
        Easting and northing of a deterministic point on the exterior boundary.
    """
    boundary = core_geometry.boundary
    point = boundary.interpolate(0.25, normalized=True)
    return float(point.x), float(point.y)


def plot_grid_alignment_zoom(
    inputs: dict[str, Any],
    grid_config: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot a detailed 30 m grid zoom and annotate example stable cell IDs.

    The figure demonstrates that grid lines are anchored globally and that cell
    identity comes from global x/y grid indices, not from mutable local raster row
    and column positions.

    Args:
        inputs: Dictionary returned by :func:`load_visualisation_inputs`.
        grid_config: ``grid`` section of ``study_area.yaml``.
        output_path: Destination PNG path.
    """
    resolution = float(grid_config["resolution_m"])
    anchor_x = float(grid_config["anchor_x_m"])
    anchor_y = float(grid_config["anchor_y_m"])
    half_window = float(grid_config.get("visual_zoom_half_window_m", 300.0))

    core_geometry = inputs["core"].geometry.union_all()
    centre_x, centre_y = choose_boundary_zoom_point(core_geometry)
    xmin, xmax = centre_x - half_window, centre_x + half_window
    ymin, ymax = centre_y - half_window, centre_y + half_window

    first_x = math.floor((xmin - anchor_x) / resolution) * resolution + anchor_x
    first_y = math.floor((ymin - anchor_y) / resolution) * resolution + anchor_y

    figure, axis = plt.subplots(figsize=(11, 10))
    inputs["core"].boundary.plot(ax=axis, linewidth=2.0)

    for x_value in np.arange(first_x, xmax + resolution, resolution):
        axis.axvline(x_value, linewidth=0.35)
    for y_value in np.arange(first_y, ymax + resolution, resolution):
        axis.axhline(y_value, linewidth=0.35)

    visible_cells = inputs["cells"][
        inputs["cells"]["x_center_m"].between(xmin, xmax)
        & inputs["cells"]["y_center_m"].between(ymin, ymax)
    ].copy()

    if not visible_cells.empty:
        label_count = min(9, len(visible_cells))
        label_indices = np.linspace(0, len(visible_cells) - 1, label_count).astype(int)
        labelled_cells = visible_cells.iloc[label_indices]

        for row in labelled_cells.itertuples(index=False):
            short_label = f"{row.grid_x_index}_{row.grid_y_index}"
            axis.text(
                row.x_center_m,
                row.y_center_m,
                short_label,
                ha="center",
                va="center",
                fontsize=6,
            )

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_title(
        "Authoritative 30 m grid and example global cell indices\n"
        f"Anchor: ({anchor_x:.0f}, {anchor_y:.0f})"
    )
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_aspect("equal")
    save_figure(figure, output_path)


def generate_visual_report(config_path: Path) -> dict[str, Any]:
    """Generate all Day 1 visual QA figures and comparison metrics.

    Args:
        config_path: Path to the study-area YAML configuration.

    Returns:
        Dictionary listing the output directory and ADM2/ADM3 comparison metrics.
    """
    config, project_root = load_configuration(config_path)
    outputs = config["outputs"]
    inputs = load_visualisation_inputs(config, project_root)

    output_directory = resolve_project_path(
        project_root,
        outputs.get("visual_report_dir", "reports/day1"),
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    plot_boundary_overview(inputs, output_directory / "01_boundary_overview.png")
    comparison_metrics = plot_adm2_adm3_comparison(
        inputs,
        output_directory / "02_adm2_adm3_comparison.png",
    )
    plot_core_context_masks(inputs, output_directory / "03_core_context_masks.png")
    plot_grid_alignment_zoom(
        inputs,
        config["grid"],
        output_directory / "04_grid_alignment_zoom.png",
    )

    comparison_output = outputs.get("boundary_comparison_metrics")
    if comparison_output:
        comparison_path = resolve_project_path(project_root, comparison_output)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(
            json.dumps(comparison_metrics, indent=2),
            encoding="utf-8",
        )

    return {
        "visual_report_directory": str(output_directory.relative_to(project_root)),
        "generated_figures": [
            "01_boundary_overview.png",
            "02_adm2_adm3_comparison.png",
            "03_core_context_masks.png",
            "04_grid_alignment_zoom.png",
        ],
        "comparison": comparison_metrics,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for visual report generation.

    Returns:
        Namespace containing the required ``--config`` path.
    """
    parser = argparse.ArgumentParser(
        description="Generate Day 1 boundary and grid quality-control figures."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to configs/study_area.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the visual report and print a concise JSON output summary."""
    arguments = parse_arguments()
    report = generate_visual_report(arguments.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
