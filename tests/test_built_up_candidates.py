"""Critical tests for formulas, Otsu thresholds, outputs and traceability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis.built_up.indices import (
    CANDIDATE_BANDS,
    INDEX_BANDS,
    reference_index_values,
)
from src.analysis.built_up.otsu import (
    FAIL_DEGENERATE_HISTOGRAM,
    FAIL_NO_VALID_PIXELS,
    PASS,
    otsu_from_histogram,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/built_up_candidates.yaml"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the built-up candidate configuration."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(value: str | Path) -> Path:
    """Resolve one repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def generated_tables(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load generated tables or return ``None`` before pipeline finalization."""
    directory = project_path(config["metadata"]["directory"])
    paths = [
        directory / config["metadata"]["output_manifest"],
        directory / config["metadata"]["threshold_table"],
        directory / config["metadata"]["candidate_area_summary"],
    ]

    if not all(path.is_file() for path in paths):
        return None

    return tuple(pd.read_csv(path) for path in paths)


# 1. Formula specification and mask-sensitive numerical safeguards.


def test_index_formulas_on_reference_pixels() -> None:
    """Check all formulas and the two crucial invalid-value safeguards."""
    values = reference_index_values(
        blue=0.1,
        green=0.2,
        red=0.3,
        nir=0.5,
        swir1=0.4,
        swir2=0.45,
    )
    expected = {
        "savi": 0.23076923076923078,
        "mndwi": -0.3333333333333333,
        "ndbi": -0.11111111111111108,
        "ibi": 0.3684210526315789,
        "ibui": -0.008547008547008572,
        "vbswir1_bi": 0.6000000000000001,
        "ndbsui": 0.06442763086842888,
    }

    for name, expected_value in expected.items():
        assert values[name] == pytest.approx(expected_value)

    zero_denominator = reference_index_values(
        blue=0.1,
        green=0.2,
        red=0.3,
        nir=0.5,
        swir1=-0.1,
        swir2=0.45,
    )
    assert zero_denominator["vbswir1_bi"] is None

    negative_square_root_input = reference_index_values(
        blue=0.1,
        green=0.2,
        red=-0.01,
        nir=0.5,
        swir1=0.4,
        swir2=0.45,
    )
    assert negative_square_root_input["ndbsui"] is None


# 2. Otsu must find a defensible split for a clearly bimodal histogram.


def test_otsu_returns_expected_split_for_bimodal_histogram() -> None:
    """Ensure the selected threshold lies between two separated modes."""
    result = otsu_from_histogram(
        {
            "bucketMeans": [0, 1, 2, 8, 9, 10],
            "histogram": [10, 20, 10, 10, 20, 10],
        }
    )

    assert result.status == PASS
    assert result.threshold == pytest.approx(5.0)
    assert 2 < result.threshold < 8


# 3. Empty and degenerate distributions must never receive a fallback threshold.


@pytest.mark.parametrize(
    ("histogram", "expected_status"),
    [
        (None, FAIL_NO_VALID_PIXELS),
        (
            {"bucketMeans": [], "histogram": []},
            FAIL_NO_VALID_PIXELS,
        ),
        (
            {"bucketMeans": [1.0], "histogram": [100]},
            FAIL_DEGENERATE_HISTOGRAM,
        ),
        (
            {
                "bucketMeans": [1.0, 1.0],
                "histogram": [50, 50],
            },
            FAIL_DEGENERATE_HISTOGRAM,
        ),
    ],
)
def test_otsu_rejects_degenerate_histograms(
    histogram: dict[str, Any] | None,
    expected_status: str,
) -> None:
    """Ensure invalid histograms produce explicit failure states."""
    result = otsu_from_histogram(histogram)
    assert result.status == expected_status
    assert result.threshold is None


# 4. Dynamic metadata counts must match passed orchestration epochs.


def test_processed_epochs_and_dynamic_counts(
    config: dict[str, Any],
) -> None:
    """Check two assets, seven indices and five candidates per epoch."""
    tables = generated_tables(config)

    if tables is None:
        pytest.skip("Generated built-up candidate metadata is not available.")

    outputs, thresholds, areas = tables
    source = pd.read_csv(
        project_path(config["inputs"]["composite_manifest"])
    )
    expected = source[
        source["status"].astype(str).str.upper() == "PASS"
    ]["epoch"].astype(int)

    include = config.get("epochs", {}).get("include")

    if include:
        expected = expected[expected.isin([int(value) for value in include])]

    expected_epochs = set(expected)
    observed_epochs = set(outputs["epoch"].astype(int))

    assert observed_epochs == expected_epochs
    assert len(outputs) == len(expected_epochs) * 2
    assert outputs["continuous_index_layers"].sum() == (
        len(expected_epochs) * 7
    )
    assert outputs["binary_candidate_layers"].sum() == (
        len(expected_epochs) * 5
    )
    assert len(thresholds) == len(expected_epochs) * 5
    assert len(areas) == len(expected_epochs) * 5


# 5. Every successful threshold must be finite, bounded and non-degenerate.


def test_thresholds_are_finite_bounded_and_create_two_classes(
    config: dict[str, Any],
) -> None:
    """Validate all scientific acceptance conditions in the threshold table."""
    tables = generated_tables(config)

    if tables is None:
        pytest.skip("Generated built-up candidate metadata is not available.")

    _, thresholds, _ = tables
    successful = thresholds[thresholds["status"] == PASS]

    assert len(successful) == len(thresholds)
    assert np.isfinite(successful["threshold_value"]).all()
    assert (
        successful["threshold_value"]
        >= successful["histogram_min"]
    ).all()
    assert (
        successful["threshold_value"]
        <= successful["histogram_max"]
    ).all()
    assert (successful["histogram_valid_pixel_count"] > 0).all()
    assert (successful["candidate_built_pixel_count_core"] > 0).all()
    assert (successful["candidate_nonbuilt_pixel_count_core"] > 0).all()


# 6–7. Post-export schema, grid, types, binary domain and recipe traceability.


def test_exported_assets_schema_grid_domain_and_traceability(
    config: dict[str, Any],
) -> None:
    """Run the critical integration checks against finalized EE assets."""
    tables = generated_tables(config)

    if tables is None:
        pytest.skip("Generated built-up candidate metadata is not available.")

    outputs, _, _ = tables

    try:
        import ee
        from src.analysis.orchestration.common import initialize_earth_engine
        from src.analysis.built_up.build_candidates import image_metadata
    except ImportError:
        pytest.skip("Earth Engine integration dependencies are unavailable.")

    initialize_earth_engine(config["project"]["earth_engine_project"])
    grid = json.loads(
        project_path(
            config["inputs"]["grid_specification"]
        ).read_text(encoding="utf-8")
    )

    for row in outputs.itertuples(index=False):
        image = ee.Image(str(row.asset_id))
        metadata = image_metadata(str(row.asset_id))
        expected_bands = (
            INDEX_BANDS
            if row.product_type == "indices"
            else CANDIDATE_BANDS
        )

        assert metadata["band_names"] == expected_bands
        assert metadata["crs"] == grid["crs"]
        assert metadata["transform"] == [
            float(value) for value in grid["transform"]
        ]
        assert metadata["width"] == int(grid["width"])
        assert metadata["height"] == int(grid["height"])
        assert image.get("recipe_sha256").getInfo() == row.recipe_sha256

        if row.product_type == "candidates":
            values = image.select(
                [f"built_{name}" for name in config["indices"]["selected"]]
            ).reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=image.geometry(),
                crs=grid["crs"],
                crsTransform=grid["transform"],
                maxPixels=int(config["histogram"]["max_pixels"]),
                tileScale=int(config["histogram"]["tile_scale"]),
            ).getInfo()

            for name in config["indices"]["selected"]:
                assert values[f"built_{name}_min"] == 0
                assert values[f"built_{name}_max"] == 1
