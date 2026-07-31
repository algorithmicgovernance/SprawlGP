"""Repair already-downloaded cell-time CSV coordinate identifiers in place."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.final_dataset.build_tables import (
    repair_coordinate_columns,
)
from src.analysis.orchestration.common import (
    find_project_root,
    load_grid_specification,
    load_yaml,
    resolve_project_path,
)


def main() -> None:
    """Repair every configured downloaded cell-time CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite CSVs after creating .bak copies.",
    )
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    grid = load_grid_specification(
        resolve_project_path(
            config["inputs"]["grid_specification"],
            project_root,
        )
    )
    staging = resolve_project_path(
        config["outputs"]["staging_table_directory"],
        project_root,
    )

    for path in sorted(staging.glob("cell_time_*.csv")):
        frame = pd.read_csv(path)
        before = int(
            frame.duplicated(
                ["cell_id", "forecast_origin"]
            ).sum()
        )
        repaired = repair_coordinate_columns(frame, grid)
        after = int(
            repaired.duplicated(
                ["cell_id", "forecast_origin"]
            ).sum()
        )
        print(
            f"{path.name}: duplicates {before} -> {after}; "
            f"rows={len(repaired)}"
        )

        if arguments.write:
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                path.replace(backup)
            repaired.to_csv(path, index=False)


if __name__ == "__main__":
    main()
