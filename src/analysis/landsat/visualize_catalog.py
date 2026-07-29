"""Generate Day 2 Landsat catalogue figures and a concise Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from .common import find_project_root, load_yaml, resolve_project_path


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one figure with stable resolution and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main(config_path: Path) -> None:
    """Create all figures and the human-readable Day 2 report."""
    root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    metadata = resolve_project_path(config["outputs"]["directory"], root)
    reports = resolve_project_path(config["outputs"]["report_directory"], root)
    reports.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(metadata / "scene_manifest_all.csv")
    monthly = pd.read_csv(metadata / "monthly_availability.csv")
    candidates = pd.read_csv(metadata / "candidate_window_summary.csv")
    quality = pd.read_csv(metadata / "epoch_quality_summary.csv")
    selected = pd.read_csv(metadata / "selected_scene_manifest.csv")

    counts = manifest.groupby(["epoch", "sensor_key"]).size().unstack(fill_value=0)
    axis = counts.plot(kind="bar", figsize=(10, 6))
    axis.set(title="Diagnostic Landsat scenes by epoch and sensor", xlabel="Epoch", ylabel="Scene count")
    save_figure(axis.figure, reports / "01_scene_count_by_epoch_sensor.png")

    month_counts = monthly.groupby(["acquisition_month", "sensor_key"])["scene_count"].sum().unstack(fill_value=0)
    axis = month_counts.plot(marker="o", figsize=(10, 6))
    axis.set(title="Monthly Landsat scene availability", xlabel="Month", ylabel="Scene count")
    axis.set_xticks(range(1, 13))
    save_figure(axis.figure, reports / "02_monthly_scene_availability.png")

    coverage = monthly.groupby(["acquisition_month", "sensor_key"])["median_valid_fraction_core"].median().mul(100).unstack()
    axis = coverage.plot(marker="o", figsize=(10, 6))
    axis.set(title="Median valid coverage over Yaoundé core", xlabel="Month", ylabel="Coverage (%)")
    axis.set_xticks(range(1, 13))
    save_figure(axis.figure, reports / "03_monthly_valid_coverage.png")

    axis = candidates.plot(kind="bar", x="window_name", y="minimum_epoch_coverage_3_pct", figsize=(14, 6), legend=False)
    axis.set(title="Candidate windows by worst-epoch observation depth", xlabel="Window", ylabel="Minimum coverage (%)")
    axis.tick_params(axis="x", rotation=75)
    save_figure(axis.figure, reports / "04_candidate_window_comparison.png")

    selected_quality = quality.set_index("epoch")[["coverage_at_least_1_core_pct", "coverage_at_least_3_core_pct"]]
    axis = selected_quality.plot(kind="bar", figsize=(10, 6))
    axis.set(title="Selected-window quality by epoch", xlabel="Epoch", ylabel="Coverage (%)")
    save_figure(axis.figure, reports / "05_selected_window_summary.png")

    lines = [
        "# Day 2 — Landsat catalogue report",
        "",
        f"- Selected common window: `{quality.iloc[0]['window_name']}`",
        f"- Diagnostic scene records: {len(manifest)}",
        f"- Frozen scene records: {len(selected)}",
        "",
        "Day 2 freezes exact Earth Engine asset IDs and does not generate composites.",
    ]
    (reports / "day2_landsat_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the Day 2 catalogue.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    main(args.config.resolve())
