from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd


def normalise_text(value: object) -> str:
    """Return lowercase text without accents for robust name searching."""
    if value is None:
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.casefold().strip()


def inspect_boundary(path: Path) -> dict:
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {path}\n"
            f"Current working directory: {Path.cwd()}\n"
            "Run the command from the repository root and verify the input path."
        )

    if not path.is_file():
        raise ValueError(f"Boundary input is not a file: {path}")
    
    gdf = gpd.read_file(path)

    report = {
        "file": str(path),
        "feature_count": len(gdf),
        "crs": str(gdf.crs),
        "columns": list(gdf.columns),
        "geometry_types": gdf.geometry.geom_type.value_counts().to_dict(),
        "missing_geometry_count": int(gdf.geometry.isna().sum()),
        "empty_geometry_count": int(gdf.geometry.is_empty.sum()),
        "invalid_geometry_count": int((~gdf.geometry.is_valid).sum()),
        "duplicate_geometry_count": int(gdf.geometry.duplicated().sum()),
        "bounds": list(map(float, gdf.total_bounds)),
    }

    print("\n=== BASIC INFORMATION ===")
    print(json.dumps(report, indent=2))

    print("\n=== ATTRIBUTE COLUMNS ===")
    for column in gdf.columns:
        if column != gdf.geometry.name:
            print(f"\n{column}:")
            print(gdf[column].dropna().astype(str).head(15).tolist())

    print("\n=== POSSIBLE YAOUNDE / MFOUNDI RECORDS ===")

    searchable_columns = [
        column
        for column in gdf.columns
        if column != gdf.geometry.name
        and (
            pd.api.types.is_string_dtype(gdf[column])
            or gdf[column].dtype == object
        )
    ]

    matched_rows: set[int] = set()

    for index, row in gdf.iterrows():
        combined = " | ".join(
            normalise_text(row[column])
            for column in searchable_columns
        )

        if "yaounde" in combined or "mfoundi" in combined:
            matched_rows.add(index)

    if not matched_rows:
        print("No matching record found.")
    else:
        display_columns = searchable_columns + [gdf.geometry.name]
        print(gdf.loc[sorted(matched_rows), display_columns].drop(
            columns=[gdf.geometry.name]
        ).to_string())

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--report",
        default=Path("data/metadata/source_boundary_inspection.json"),
        type=Path,
    )
    args = parser.parse_args()

    report = inspect_boundary(args.input)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()