"""Monitor exports, validate assets and finalize built-up candidate metadata."""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import ee
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.orchestration.common import (
    asset_exists,
    exact_grid_region,
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_json,
    load_yaml,
    metadata_directory,
    report_directory,
    resolve_project_path,
    sha256_file,
    stable_object_hash,
    write_json,
)
from src.analysis.built_up.build_candidates import (
    image_metadata,
    load_ee_geometry,
    select_completed_composites,
)
from src.analysis.built_up.indices import (
    CANDIDATE_BANDS,
    CANDIDATE_INDICES,
    CONTINUOUS_INDICES,
    CONTINUOUS_INDEX_BANDS,
    INDEX_BANDS,
)
from src.analysis.built_up.otsu import histogram_quantile


TERMINAL_SUCCESS_STATES = {"COMPLETED", "EXISTS"}
TERMINAL_FAILURE_STATES = {"FAILED", "CANCELLED", "CANCEL_REQUESTED"}


def refresh_task_manifest(task_frame: pd.DataFrame) -> pd.DataFrame:
    """Refresh all task states through one Earth Engine task-list request."""
    refreshed = task_frame.copy()
    available = {
        task.id: task.status()
        for task in ee.batch.Task.list()
        if task.id is not None
    }

    for index, row in refreshed.iterrows():
        if str(row["state"]) == "EXISTS" or not str(row["task_id"]):
            continue

        task_id = str(row["task_id"])

        if task_id not in available:
            raise LookupError(
                f"Earth Engine task '{task_id}' was not found."
            )

        status = available[task_id]
        refreshed.at[index, "state"] = status.get("state", "UNKNOWN")
        refreshed.at[index, "error_message"] = status.get(
            "error_message",
            "",
        )

    return refreshed


def validate_output_grid(
    asset_id: str,
    expected_bands: list[str],
    grid: dict[str, Any],
) -> dict[str, Any]:
    """Validate existence, ordered schema, dimensions, CRS and transform."""
    if not asset_exists(asset_id):
        raise FileNotFoundError(f"Earth Engine asset not found: {asset_id}")

    metadata = image_metadata(asset_id)

    if metadata["band_names"] != expected_bands:
        raise ValueError(
            f"Unexpected bands in {asset_id}: {metadata['band_names']}"
        )

    if metadata["crs"] != grid["crs"]:
        raise ValueError(
            f"Unexpected CRS in {asset_id}: {metadata['crs']}"
        )

    if not np.allclose(
        metadata["transform"],
        [float(value) for value in grid["transform"]],
        atol=1e-9,
    ):
        raise ValueError(
            f"Unexpected transform in {asset_id}: "
            f"{metadata['transform']}"
        )

    if (
        metadata["width"] != int(grid["width"])
        or metadata["height"] != int(grid["height"])
    ):
        raise ValueError(
            f"Unexpected dimensions in {asset_id}: "
            f"{metadata['width']} x {metadata['height']}"
        )

    return metadata


def validate_band_types(
    asset_id: str,
    product_type: str,
) -> None:
    """Validate Float32 index bands and UInt8 categorical bands."""
    band_types = ee.Image(asset_id).bandTypes().getInfo()

    if product_type == "indices":
        for name in INDEX_BANDS[:-1]:
            precision = band_types[name].get("precision")
            if precision != "float":
                raise ValueError(
                    f"Index band {name} in {asset_id} is not float."
                )

        valid_type = band_types["valid_composite"]
        if (
            valid_type.get("precision") != "int"
            or float(valid_type.get("min", -1)) < 0
            or float(valid_type.get("max", 256)) > 255
        ):
            raise ValueError(
                f"valid_composite in {asset_id} is not UInt8-compatible."
            )
    else:
        for name in CANDIDATE_BANDS:
            data_type = band_types[name]
            if (
                data_type.get("precision") != "int"
                or float(data_type.get("min", -1)) < 0
                or float(data_type.get("max", 256)) > 255
            ):
                raise ValueError(
                    f"Candidate band {name} in {asset_id} "
                    "is not UInt8-compatible."
                )


def validate_candidate_domain_and_masks(
    asset_id: str,
    core_geometry: ee.Geometry,
    grid: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Ensure candidate values are binary and masks equal validity bands."""
    image = ee.Image(asset_id)
    checks: list[ee.Image] = []

    for name in CANDIDATE_INDICES:
        built = image.select(f"built_{name}")
        valid = image.select(f"valid_{name}")
        checks.extend(
            [
                built.rename(f"{name}_built"),
                built.mask().unmask(0).neq(valid.eq(1)).rename(
                    f"{name}_mask_mismatch"
                ),
            ]
        )

    values = ee.Image.cat(checks).reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=core_geometry,
        crs=grid["crs"],
        crsTransform=grid["transform"],
        maxPixels=int(config["histogram"]["max_pixels"]),
        tileScale=int(config["histogram"]["tile_scale"]),
    ).getInfo()

    for name in CANDIDATE_INDICES:
        minimum = values.get(f"{name}_built_min")
        maximum = values.get(f"{name}_built_max")
        mismatch = values.get(f"{name}_mask_mismatch_max", 0)

        if minimum != 0 or maximum != 1:
            raise ValueError(
                f"Candidate {name} in {asset_id} does not contain "
                "both binary classes in the administrative core."
            )

        if float(mismatch or 0) != 0:
            raise ValueError(
                f"Candidate mask differs from valid_{name} in {asset_id}."
            )


def validate_asset_properties(
    row: pd.Series,
) -> None:
    """Compare source and recipe properties with the local output manifest."""
    image = ee.Image(str(row["asset_id"]))
    properties = image.toDictionary(
        [
            "source_composite_asset",
            "source_observation_count_asset",
            "recipe_sha256",
        ]
    ).getInfo()

    if properties.get("source_composite_asset") != str(
        row["source_composite_asset"]
    ):
        raise ValueError(
            f"Source-composite property mismatch in {row['asset_id']}."
        )

    if properties.get("source_observation_count_asset") != str(
        row["source_count_asset"]
    ):
        raise ValueError(
            f"Source-count property mismatch in {row['asset_id']}."
        )

    if properties.get("recipe_sha256") != str(row["recipe_sha256"]):
        raise ValueError(
            f"Recipe SHA-256 mismatch in {row['asset_id']}."
        )


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one QA figure and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_histogram_gallery(
    histograms: dict[str, Any],
    threshold_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the required epoch-by-index histogram and threshold gallery."""
    epochs = sorted(int(value) for value in histograms)
    figure, axes = plt.subplots(
        len(epochs),
        len(CANDIDATE_INDICES),
        figsize=(18, 3.2 * len(epochs)),
        squeeze=False,
    )

    for row_index, epoch in enumerate(epochs):
        for column_index, name in enumerate(CANDIDATE_INDICES):
            axis = axes[row_index, column_index]
            histogram = histograms[str(epoch)][name]
            means = np.asarray(histogram["bucketMeans"], dtype=float)
            counts = np.asarray(histogram["histogram"], dtype=float)
            threshold = float(
                threshold_table.loc[
                    (threshold_table["epoch"] == epoch)
                    & (threshold_table["index_name"] == name),
                    "threshold_value",
                ].iloc[0]
            )
            axis.plot(means, counts)
            axis.axvline(threshold, linestyle="--")
            axis.set_title(f"{epoch} — {name}")
            axis.tick_params(labelsize=7)

    figure.suptitle("Index histograms and epoch-specific Otsu thresholds")
    save_figure(figure, output_path)


def read_thumbnail(url: str) -> np.ndarray:
    """Download one Earth Engine PNG thumbnail into a NumPy array."""
    with urllib.request.urlopen(url) as response:
        return mpimg.imread(io.BytesIO(response.read()), format="png")


def index_display_ranges(
    histograms: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    """Calculate robust per-index display ranges shared across all epochs."""
    ranges: dict[str, tuple[float, float]] = {}

    for name in CONTINUOUS_INDICES:
        lower = []
        upper = []

        for epoch_histograms in histograms.values():
            histogram = epoch_histograms[name]
            lower.append(histogram_quantile(histogram, 0.02))
            upper.append(histogram_quantile(histogram, 0.98))

        ranges[name] = (float(min(lower)), float(max(upper)))

    return ranges


def create_map_gallery(
    output_manifest: pd.DataFrame,
    histograms: dict[str, Any],
    grid: dict[str, Any],
    output_path: Path,
    *,
    product_type: str,
    dimensions: int,
) -> None:
    """Create dynamic continuous-index or candidate thumbnail galleries."""
    rows = output_manifest[
        output_manifest["product_type"] == product_type
    ].sort_values("epoch")
    epochs = rows["epoch"].astype(int).tolist()
    display_indices = (
        CONTINUOUS_INDICES
        if product_type == "indices"
        else CANDIDATE_INDICES
    )
    ranges = index_display_ranges(histograms)
    figure, axes = plt.subplots(
        len(epochs),
        len(display_indices),
        figsize=(
            3.2 * len(display_indices),
            3.0 * len(epochs),
        ),
        squeeze=False,
    )

    for row_index, (_, row) in enumerate(rows.iterrows()):
        image = ee.Image(str(row["asset_id"]))

        for column_index, name in enumerate(display_indices):
            axis = axes[row_index, column_index]
            axis.set_xticks([])
            axis.set_yticks([])

            try:
                if product_type == "indices":
                    minimum, maximum = ranges[name]
                    display_image = image.select(name)
                    parameters = {
                        "bands": name,
                        "min": minimum,
                        "max": maximum,
                        "region": exact_grid_region(grid),
                        "dimensions": int(dimensions),
                        "format": "png",
                    }
                else:
                    built = image.select(f"built_{name}")
                    valid = image.select(f"valid_{name}").eq(1)
                    display_image = (
                        ee.Image.constant(2)
                        .where(valid.And(built.eq(0)), 0)
                        .where(valid.And(built.eq(1)), 1)
                        .rename("display")
                    )
                    parameters = {
                        "bands": "display",
                        "min": 0,
                        "max": 2,
                        "palette": [
                            "f7f7f7",
                            "d73027",
                            "bdbdbd",
                        ],
                        "region": exact_grid_region(grid),
                        "dimensions": int(dimensions),
                        "format": "png",
                    }

                axis.imshow(
                    read_thumbnail(
                        display_image.getThumbURL(parameters)
                    )
                )
            except Exception as error:
                axis.text(
                    0.5,
                    0.5,
                    f"Thumbnail unavailable\n{type(error).__name__}",
                    ha="center",
                    va="center",
                )

            axis.set_title(f"{int(row['epoch'])} — {name}")

    title = (
        "Continuous built-up index gallery"
        if product_type == "indices"
        else "Unvalidated Otsu candidate gallery"
    )
    figure.suptitle(title)
    save_figure(figure, output_path)

def write_report(
    output_manifest: pd.DataFrame,
    threshold_table: pd.DataFrame,
    area_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a concise scientific and operational completion report."""
    epochs = sorted(output_manifest["epoch"].astype(int).unique())
    lines = [
        "# Spectral indices and Otsu built-up candidates",
        "",
        "## Output summary",
        "",
        f"- Completed epochs: {', '.join(map(str, epochs))}",
        f"- Epoch count: {len(epochs)}",
        f"- Continuous index layers: "
        f"{len(epochs) * len(CONTINUOUS_INDEX_BANDS)}",
        f"- Binary candidate maps: "
        f"{len(epochs) * len(CANDIDATE_INDICES)}",
        f"- Epoch-specific Otsu thresholds: {len(threshold_table)}",
        "",
        "The binary maps are unvalidated candidate pseudo-labels. No index "
        "has been selected as the final historical built-up mapping method.",
        "",
        "## Thresholds",
        "",
        "| Epoch | Index | Threshold | Valid pixels | Built fraction | Status |",
        "|---:|---|---:|---:|---:|:---:|",
    ]

    area_index = area_summary.set_index(["epoch", "index_name"])

    for row in threshold_table.sort_values(
        ["epoch", "index_name"]
    ).itertuples(index=False):
        area = area_index.loc[(int(row.epoch), row.index_name)]
        lines.append(
            "| "
            f"{int(row.epoch)} | {row.index_name} | "
            f"{float(row.threshold_value):.6f} | "
            f"{int(row.histogram_valid_pixel_count)} | "
            f"{float(area['candidate_built_fraction']):.4f} | "
            f"{row.status} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Thresholds are estimated independently for each epoch.",
            "- Candidate-area changes are quality diagnostics, not validated growth.",
            "- Invalid index pixels remain masked and are never recoded as non-built.",
            "- No temporal persistence correction or transition map is applied.",
            "- IBI remains in continuous assets for transparency but is excluded "
            "from candidate generation and Day 5 method comparison.",
            "- Comparative validation and final method selection belong to the next stage.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_finalize(config_path: Path, status_only: bool) -> dict[str, Any]:
    """Refresh task states and finalize all validated outputs."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    initialize_earth_engine(config["project"]["earth_engine_project"])

    task_path = resolve_project_path(
        config["exports"]["task_manifest"],
        project_root,
    )

    if not task_path.is_file():
        raise FileNotFoundError(
            "Submit built-up candidate tasks before finalization."
        )

    tasks = pd.read_csv(task_path, keep_default_na=False)
    tasks = refresh_task_manifest(tasks)
    tasks.to_csv(task_path, index=False)
    state_counts = tasks["state"].value_counts().to_dict()
    print(json.dumps({"task_states": state_counts}, indent=2))

    if status_only:
        return {"task_states": state_counts}

    failures = tasks[tasks["state"].isin(TERMINAL_FAILURE_STATES)]

    if not failures.empty:
        raise RuntimeError(
            "At least one built-up candidate export failed."
        )

    incomplete = tasks[
        ~tasks["state"].isin(TERMINAL_SUCCESS_STATES)
    ]

    if not incomplete.empty:
        raise RuntimeError(
            "Exports are incomplete. Run the status command later."
        )

    grid = load_grid_specification(
        resolve_project_path(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    core_geometry = load_ee_geometry(
        resolve_project_path(
            config["inputs"]["core_boundary"],
            project_root,
        )
    )
    metadata_dir = metadata_directory(config, project_root)
    output_path = metadata_dir / config["metadata"]["output_manifest"]
    outputs = pd.read_csv(output_path, keep_default_na=False)

    for index, row in outputs.iterrows():
        expected_bands = (
            INDEX_BANDS
            if row["product_type"] == "indices"
            else CANDIDATE_BANDS
        )
        validate_output_grid(
            str(row["asset_id"]),
            expected_bands,
            grid,
        )
        validate_band_types(
            str(row["asset_id"]),
            str(row["product_type"]),
        )
        validate_asset_properties(row)

        if row["product_type"] == "candidates":
            validate_candidate_domain_and_masks(
                str(row["asset_id"]),
                core_geometry,
                grid,
                config,
            )

        outputs.at[index, "state"] = "PASS"

    outputs.to_csv(output_path, index=False)
    threshold_path = metadata_dir / config["metadata"]["threshold_table"]
    area_path = metadata_dir / config["metadata"]["candidate_area_summary"]
    histogram_path = metadata_dir / config["metadata"]["histograms"]
    recipe_path = metadata_dir / config["metadata"]["processing_recipe"]
    dependency_path = (
        metadata_dir / config["metadata"]["dependency_manifest"]
    )
    threshold_table = pd.read_csv(threshold_path)
    area_summary = pd.read_csv(area_path)
    histograms = load_json(histogram_path)

    version_payload = {
        "pipeline_stage": "built_up_candidates",
        "pipeline_version": int(config["version"]),
        "processed_epochs": sorted(
            outputs["epoch"].astype(int).unique().tolist()
        ),
        "grid_specification_sha256": sha256_file(
            resolve_project_path(
                config["inputs"]["grid_specification"],
                project_root,
            )
        ),
        "orchestration_version_sha256": sha256_file(
            resolve_project_path(
                config["inputs"]["orchestration_version"],
                project_root,
            )
        ),
        "dependency_manifest_sha256": sha256_file(dependency_path),
        "histograms_sha256": sha256_file(histogram_path),
        "threshold_table_sha256": sha256_file(threshold_path),
        "candidate_area_summary_sha256": sha256_file(area_path),
        "processing_recipe_sha256": sha256_file(recipe_path),
        "task_manifest_sha256": sha256_file(task_path),
        "output_manifest_sha256": sha256_file(output_path),
        "earth_engine_assets": outputs["asset_id"].astype(str).tolist(),
        "continuous_index_layer_count": int(
            outputs["continuous_index_layers"].sum()
        ),
        "binary_candidate_layer_count": int(
            outputs["binary_candidate_layers"].sum()
        ),
    }
    version_payload["stable_signature"] = stable_object_hash(
        version_payload
    )
    write_json(
        metadata_dir / config["metadata"]["version"],
        version_payload,
    )

    reports = report_directory(config, project_root)
    create_histogram_gallery(
        histograms,
        threshold_table,
        reports / config["reports"]["histogram_gallery"],
    )

    if bool(config["reports"].get("generate_map_galleries", True)):
        create_map_gallery(
            outputs,
            histograms,
            grid,
            reports / config["reports"]["index_gallery"],
            product_type="indices",
            dimensions=int(
                config["reports"]["thumbnail_dimensions"]
            ),
        )
        create_map_gallery(
            outputs,
            histograms,
            grid,
            reports / config["reports"]["candidate_gallery"],
            product_type="candidates",
            dimensions=int(
                config["reports"]["thumbnail_dimensions"]
            ),
        )

    write_report(
        outputs,
        threshold_table,
        area_summary,
        reports / config["reports"]["report"],
    )

    print(json.dumps(version_payload, indent=2))
    return version_payload


def parse_arguments() -> argparse.Namespace:
    """Parse status and finalization command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor and finalize built-up candidate assets."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run task monitoring or complete finalization."""
    arguments = parse_arguments()
    run_finalize(arguments.config.resolve(), arguments.status_only)


if __name__ == "__main__":
    main()
