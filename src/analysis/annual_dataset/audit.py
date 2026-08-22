"""Create annual trajectory and 2015-2020 anomaly evidence outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    import ee
except ImportError:  # Enables local unit tests without Earth Engine.
    ee = None

from src.analysis.annual_dataset.build_products import (
    annual_transitions,
    build_transition_image,
    validate_annual_config,
)
from src.analysis.annual_dataset.build_tables import load_core_geometry, load_state_assets
from src.analysis.annual_dataset.finalize import core_area_ha
from src.analysis.orchestration.common import (
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_yaml,
    resolve_project_path,
)


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError("The Earth Engine Python API is required for the annual audit.")


def report_directory(config: dict[str, Any], project_root: Path) -> Path:
    """Create and return the isolated annual audit directory."""
    path = resolve_project_path(config["audit"]["report_directory"], project_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def transition_statistics(
    origin: int,
    target: int,
    start_asset: str,
    end_asset: str,
    core: Any,
    core_area: float,
    grid: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate one-year transition, reversal, and correction diagnostics."""
    require_earth_engine()
    transition = build_transition_image(ee.Image(start_asset), ee.Image(end_asset))
    common = transition.select("common_valid_1y").eq(1)
    eligible = transition.select("eligible_nonbuilt").eq(1)
    new_built = transition.select("target_transition_1y").unmask(0).eq(1)
    reversal = transition.select("raw_built_to_nonbuilt_reversal").eq(1)
    correction = transition.select("persistence_correction_at_target").eq(1)
    one = ee.Image.constant(1)
    area = ee.Image.pixelArea()
    bands = ee.Image.cat(
        [
            area.updateMask(common).rename("common_valid_area_m2"),
            one.updateMask(eligible).rename("eligible_nonbuilt_cells"),
            one.updateMask(new_built).rename("new_built_cells"),
            area.updateMask(new_built).rename("new_built_area_m2"),
            one.updateMask(reversal).rename("reversal_cells"),
            area.updateMask(reversal).rename("reversal_area_m2"),
            one.updateMask(correction).rename("correction_cells"),
            area.updateMask(correction).rename("correction_area_m2"),
        ]
    )
    values = bands.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=core,
        crs=grid["crs"],
        crsTransform=[float(value) for value in grid["transform"]],
        maxPixels=int(config["exports"]["max_pixels"]),
        tileScale=int(config["histogram"]["tile_scale"]),
    ).getInfo()
    eligible_cells = int(round(float(values.get("eligible_nonbuilt_cells", 0) or 0)))
    new_cells = int(round(float(values.get("new_built_cells", 0) or 0)))
    common_area = float(values.get("common_valid_area_m2", 0) or 0) / 10_000
    return {
        "forecast_origin": origin,
        "target_year": target,
        "common_valid_pct_of_core": common_area / core_area * 100,
        "eligible_nonbuilt_cells": eligible_cells,
        "new_built_cells": new_cells,
        "new_built_area_ha": float(values.get("new_built_area_m2", 0) or 0) / 10_000,
        "conversion_rate": new_cells / eligible_cells if eligible_cells else None,
        "raw_built_to_nonbuilt_reversal_cells": int(
            round(float(values.get("reversal_cells", 0) or 0))
        ),
        "raw_built_to_nonbuilt_reversal_area_ha": float(
            values.get("reversal_area_m2", 0) or 0
        )
        / 10_000,
        "persistence_correction_at_target_cells": int(
            round(float(values.get("correction_cells", 0) or 0))
        ),
        "persistence_correction_at_target_area_ha": float(
            values.get("correction_area_m2", 0) or 0
        )
        / 10_000,
    }


def anchor_protocol_comparison(
    annual_states: pd.DataFrame,
    old_thresholds: pd.DataFrame,
    old_protocol: dict[str, Any],
) -> pd.DataFrame:
    """Compare old and annual NDBI/Otsu results at six common anchor years."""
    anchors = [2000, 2005, 2010, 2015, 2020, 2025]
    old = old_thresholds[
        old_thresholds["index_name"].astype(str).str.casefold().eq("ndbi")
    ].copy()
    old["epoch"] = old["epoch"].astype(int)
    old = old.set_index("epoch")
    annual = annual_states.set_index("year")
    rows = []
    for year in anchors:
        old_fraction = float(old.loc[year, "candidate_built_fraction_core"])
        annual_fraction = float(annual.loc[year, "raw_built_fraction_core"])
        epoch_protocol = old_protocol.get("epochs", {}).get(year) or old_protocol.get(
            "epochs", {}
        ).get(str(year), {})
        rows.append(
            {
                "year": year,
                "old_v2_ndbi_threshold": float(old.loc[year, "threshold_value"]),
                "annual_ndbi_threshold": float(annual.loc[year, "ndbi_otsu_threshold"]),
                "old_v2_raw_built_fraction": old_fraction,
                "annual_raw_built_fraction": annual_fraction,
                "built_fraction_delta_percentage_points": (
                    annual_fraction - old_fraction
                )
                * 100,
                "old_v2_window": epoch_protocol.get("window_name", ""),
                "annual_window": "Jan-Dec",
            }
        )
    return pd.DataFrame(rows)


def five_year_block_comparison(
    annual_transitions_frame: pd.DataFrame, old_demand: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate annual transitions into three old-protocol comparison blocks."""
    rows = []
    for start, end in ((2010, 2015), (2015, 2020), (2020, 2025)):
        annual = annual_transitions_frame[
            (annual_transitions_frame["forecast_origin"] >= start)
            & (annual_transitions_frame["target_year"] <= end)
        ]
        old = old_demand[
            (old_demand["period_start"].astype(int) == start)
            & (old_demand["period_end"].astype(int) == end)
        ]
        if len(annual) != 5 or len(old) != 1:
            raise ValueError(f"Incomplete five-year comparison block {start}-{end}.")
        largest = annual.loc[annual["new_built_area_ha"].idxmax()]
        annual_sum = float(annual["new_built_area_ha"].sum())
        old_area = float(old.iloc[0]["observed_new_built_area_ha"])
        rows.append(
            {
                "period_start": start,
                "period_end": end,
                "sum_annual_new_built_area_ha": annual_sum,
                "old_v2_new_built_area_ha": old_area,
                "difference_ha": annual_sum - old_area,
                "largest_single_year_transition": (
                    f"{int(largest['forecast_origin'])}-{int(largest['target_year'])}"
                ),
                "largest_single_year_new_built_area_ha": float(
                    largest["new_built_area_ha"]
                ),
                "largest_single_year_share_of_annual_block": (
                    float(largest["new_built_area_ha"]) / annual_sum
                    if annual_sum
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    states: pd.DataFrame, transitions: pd.DataFrame, directory: Path, config: dict[str, Any]
) -> None:
    """Create compact annual built-fraction and transition-area figures."""
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(states["year"], states["raw_built_fraction_core"] * 100, marker="o", label="Raw")
    axis.plot(
        states["year"], states["final_built_fraction_core"] * 100, marker="o", label="Persistent"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Built fraction of valid core (%)")
    axis.set_title("Annual NDBI/Otsu built fraction")
    axis.legend()
    figure.savefig(directory / config["audit"]["built_fraction_figure"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(transitions["target_year"], transitions["new_built_area_ha"])
    axis.set_xlabel("Target year")
    axis.set_ylabel("New built area (ha)")
    axis.set_title("Annual persistent non-built to built transitions")
    figure.savefig(directory / config["audit"]["new_built_area_figure"], dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_anomaly_report(
    path: Path, states: pd.DataFrame, transitions: pd.DataFrame
) -> None:
    """Write measured evidence around 2015-2020 without assigning a cause."""
    focus = states[states["year"].between(2014, 2022)].merge(
        transitions,
        left_on="year",
        right_on="target_year",
        how="left",
        validate="one_to_one",
    )
    block = transitions[
        (transitions["forecast_origin"] >= 2015)
        & (transitions["target_year"] <= 2020)
    ]
    largest = block.loc[block["new_built_area_ha"].idxmax()]
    positive_years = int((block["new_built_area_ha"] > 0).sum())
    total_growth = float(block["new_built_area_ha"].sum())
    largest_share = float(largest["new_built_area_ha"]) / total_growth if total_growth else 0
    raw_changes = focus.set_index("year")["raw_built_fraction_core"].diff().dropna()
    oscillates = bool((raw_changes > 0).any() and (raw_changes < 0).any())
    threshold_changes = focus.set_index("year")["ndbi_otsu_threshold"].diff().dropna()
    largest_threshold_year = int(threshold_changes.abs().idxmax())
    support_changes = focus.set_index("year")["coverage_at_least_1_core_pct"].diff().dropna()
    largest_support_year = int(support_changes.abs().idxmax())
    persistence_total = float(focus["persistence_corrected_area_ha"].sum())
    final_raw_gap = (
        focus["final_built_fraction_core"] - focus["raw_built_fraction_core"]
    ) * 100
    lines = [
        "# Annual evidence for the 2015-2020 transition",
        "",
        "This report presents measured annual diagnostics and does not assign a causal explanation.",
        "",
        "| Year | NDBI threshold | Raw built % | Final built % | New built ha | Raw reversal ha | Persistence corrected ha | Sensors | Coverage >=1 % | Coverage >=3 % | Median observations |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {int(row.year)} | {row.ndbi_otsu_threshold:.6f} | "
            f"{row.raw_built_fraction_core * 100:.2f} | "
            f"{row.final_built_fraction_core * 100:.2f} | "
            f"{row.new_built_area_ha:.2f} | "
            f"{row.raw_built_to_nonbuilt_reversal_area_ha:.2f} | "
            f"{row.persistence_corrected_area_ha:.2f} | {row.selected_sensors} | "
            f"{row.coverage_at_least_1_core_pct:.2f} | "
            f"{row.coverage_at_least_3_core_pct:.2f} | "
            f"{row.median_observation_count:.2f} |"
        )
    distribution_statement = (
        f"Growth is distributed across {positive_years} positive annual transitions; "
        f"the largest contributes {largest_share:.1%} of the five-year sum."
        if positive_years > 1
        else "Measured growth is concentrated in one positive annual transition."
    )
    movement_statement = (
        "The raw built fraction oscillates because the annual differences include both increases and decreases."
        if oscillates
        else "The raw built fraction does not oscillate within this window; all non-zero changes have one sign."
    )
    lines.extend(
        [
            "",
            "## Evidence statements",
            "",
            f"- The largest 2015-2020 one-year expansion is {int(largest['forecast_origin'])}-{int(largest['target_year'])}: {largest['new_built_area_ha']:.2f} ha.",
            f"- {distribution_statement}",
            f"- {movement_statement}",
            f"- The largest absolute annual NDBI-threshold step in 2014-2022 ends in {largest_threshold_year} ({threshold_changes.loc[largest_threshold_year]:+.6f}); the median absolute step is {threshold_changes.abs().median():.6f}. No categorical sharpness cutoff was imposed.",
            f"- The largest coverage >=1 change ends in {largest_support_year} ({support_changes.loc[largest_support_year]:+.2f} percentage points).",
            f"- Persistence corrects {persistence_total:.2f} ha summed across the displayed state years; the final-minus-raw built-fraction gap ranges from {final_raw_gap.min():.2f} to {final_raw_gap.max():.2f} percentage points.",
            "",
            "## Unresolved hypotheses",
            "",
            "The outputs should be reviewed jointly for progressive growth, old-window compositing effects, annual NDBI-distribution changes, Otsu-threshold instability, and raw temporal classification instability. These diagnostics do not by themselves establish that any one mechanism caused the old five-year jump.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(config_path: Path) -> dict[str, Any]:
    """Generate all required annual audit tables, figures, and evidence report."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    years = validate_annual_config(config)
    metadata_dir = resolve_project_path(config["metadata"]["directory"], project_root)
    states = pd.read_csv(metadata_dir / config["metadata"]["state_summary"])
    if states["year"].astype(int).tolist() != years:
        raise ValueError("Annual state summary must contain all 26 ordered years.")
    state_assets = load_state_assets(
        metadata_dir / config["metadata"]["state_output_manifest"], years
    )
    initialize_earth_engine(config["project"]["earth_engine_project"])
    grid = load_grid_specification(
        resolve_project_path(config["inputs"]["grid_specification"], project_root)
    )
    core_path = resolve_project_path(config["inputs"]["core_boundary"], project_root)
    core = load_core_geometry(core_path)
    area_ha = core_area_ha(core_path)
    transition_rows = [
        transition_statistics(
            origin,
            target,
            state_assets[origin],
            state_assets[target],
            core,
            area_ha,
            grid,
            config,
        )
        for origin, target in annual_transitions(years)
    ]
    transitions = pd.DataFrame(transition_rows)
    if len(transitions) != 25:
        raise AssertionError("The annual transition audit must contain exactly 25 rows.")
    directory = report_directory(config, project_root)
    states.to_csv(directory / config["audit"]["annual_state_summary"], index=False)
    transitions.to_csv(directory / config["audit"]["annual_transition_summary"], index=False)
    transitions.to_csv(metadata_dir / config["metadata"]["transition_summary"], index=False)
    old_thresholds = pd.read_csv(
        resolve_project_path(config["audit"]["old_threshold_table"], project_root)
    )
    old_protocol = load_yaml(
        resolve_project_path(config["audit"]["old_compositing_protocol"], project_root)
    )
    anchors = anchor_protocol_comparison(states, old_thresholds, old_protocol)
    anchors.to_csv(directory / config["audit"]["anchor_protocol_comparison"], index=False)
    old_demand = pd.read_csv(
        resolve_project_path(config["audit"]["old_historical_demand"], project_root)
    )
    blocks = five_year_block_comparison(transitions, old_demand)
    blocks.to_csv(directory / config["audit"]["five_year_block_comparison"], index=False)
    save_figures(states, transitions, directory, config)
    write_anomaly_report(directory / config["audit"]["anomaly_report"], states, transitions)
    final_dir = resolve_project_path(config["outputs"]["final_directory"], project_root)
    final_dir.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(final_dir / config["outputs"]["annual_transition_summary"], index=False)
    states.to_csv(final_dir / config["outputs"]["annual_state_summary"], index=False)
    result = {
        "status": "PASS",
        "state_rows": len(states),
        "transition_rows": len(transitions),
        "audit_directory": str(directory),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the annual audit command line."""
    parser = argparse.ArgumentParser(description="Create annual anomaly evidence outputs.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the annual trajectory and anomaly audit."""
    arguments = parse_arguments()
    run_audit(arguments.config.resolve())


if __name__ == "__main__":
    main()