"""Print grid metadata for one source composite and one Day 4 asset."""

from __future__ import annotations

import json
from pathlib import Path

import ee


PROJECT = "urban-sprawl-ssa"
SOURCE_ASSET = (
    "projects/urban-sprawl-ssa/assets/sprawlgp/v1/"
    "landsat/composite_2025"
)
DAY4_ASSET = (
    "projects/urban-sprawl-ssa/assets/sprawlgp/v1/"
    "indices/indices_2025"
)


def image_grid(asset_id: str) -> dict:
    """Return first-band dimensions, CRS and affine transform."""
    information = ee.Image(asset_id).getInfo()
    band = information["bands"][0]
    return {
        "asset_id": asset_id,
        "dimensions": band.get("dimensions"),
        "crs": band.get("crs"),
        "transform": band.get("crs_transform"),
    }


ee.Initialize(project=PROJECT)

reference = json.loads(
    Path("data/metadata/grid_specification.json").read_text(
        encoding="utf-8"
    )
)

print(
    json.dumps(
        {
            "reference": {
                "dimensions": [
                    reference["width"],
                    reference["height"],
                ],
                "crs": reference["crs"],
                "transform": reference["transform"],
            },
            "source_composite": image_grid(SOURCE_ASSET),
            "day4_asset": image_grid(DAY4_ASSET),
        },
        indent=2,
    )
)