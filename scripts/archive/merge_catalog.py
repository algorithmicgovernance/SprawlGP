from __future__ import annotations

from pathlib import Path

import pandas as pd


CANONICAL = Path("data/metadata/landsat")
RECOVERY = Path("data/metadata/landsat_2025_recovery")
EPOCH = 2025


def replace_epoch(
    canonical_path: Path,
    recovery_path: Path,
) -> pd.DataFrame:
    """Replace one epoch while preserving all other canonical rows."""
    if not canonical_path.exists():
        print(f"⚠️  Skipping {canonical_path.name}: canonical file missing.")
        return pd.DataFrame()
    
    if not recovery_path.exists():
        print(f"⚠️  Skipping {canonical_path.name}: recovery file missing.")
        return pd.DataFrame()

    old = pd.read_csv(canonical_path)
    new = pd.read_csv(recovery_path)

    # Remove old 2025 rows, append new 2025 rows
    merged = pd.concat(
        [
            old[old["epoch"].astype(int) != EPOCH],
            new[new["epoch"].astype(int) == EPOCH],
        ],
        ignore_index=True,
    )

    if merged.empty:
        return merged

    # Determine which sorting columns exist in this particular DataFrame
    possible_sort_cols = ["epoch", "acquisition_date", "sensor_key"]
    sort_cols = [col for col in possible_sort_cols if col in merged.columns]

    if sort_cols:
        merged = merged.sort_values(
            sort_cols,
            na_position="last",
        ).reset_index(drop=True)

    return merged


# List of CSV files to merge
filenames = [
    "scene_manifest_all.csv",
    "monthly_availability.csv",
    "candidate_window_details.csv",
    "epoch_quality_summary.csv",
    "selected_scene_manifest.csv",
]

for filename in filenames:
    print(f"Processing {filename}...")
    output = replace_epoch(
        CANONICAL / filename,
        RECOVERY / filename,
    )
    if not output.empty:
        output.to_csv(CANONICAL / filename, index=False)
        print(f"✅ Updated {filename}")
    else:
        print(f"⚠️  {filename} produced empty output, skipped.")

print("✅ Merged accepted 2025 catalogue records into canonical directory.")