"""Essential leakage, consistency and release tests for dataset version 1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis.final_dataset.build_products import (
    apply_absorbing_persistence_numpy,
    transition_numpy,
)
from src.analysis.final_dataset.build_tables import (
    TABLE_COLUMNS,
    assert_feature_schema_has_no_leakage,
    repair_coordinate_columns,
    synchronise_demand_with_cell_time,
)
from src.analysis.final_dataset.finalize import compute_patch_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/final_dataset.yaml"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the authoritative final-dataset configuration."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    """Resolve one repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    """Calculate a local SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def test_absorbing_persistence_preserves_invalid_cells() -> None:
    """Once built, a cell stays built only at later valid observations."""
    raw = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    valid = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    final, corrected = apply_absorbing_persistence_numpy(raw, valid)

    assert final.tolist() == [
        [0, 1, 0],
        [1, -1, 0],
        [1, 1, 1],
    ]
    assert corrected.tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
    ]


def test_transition_requires_pairwise_valid_nonbuilt_origin() -> None:
    """Targets exist only on pairwise-valid non-built origin cells."""
    start = np.array([0, 0, 1, 0])
    end = np.array([1, 0, 1, 1])
    start_valid = np.array([1, 1, 1, 0], dtype=bool)
    end_valid = np.array([1, 1, 1, 1], dtype=bool)
    common, eligible, target = transition_numpy(
        start,
        end,
        start_valid,
        end_valid,
    )

    assert common.tolist() == [1, 1, 1, 0]
    assert eligible.tolist() == [1, 1, 0, 0]
    assert target.tolist() == [1, 0, 0, 0]


def test_primary_table_schema_has_no_target_leakage() -> None:
    """The released feature table contains no target-year predictors."""
    assert_feature_schema_has_no_leakage(TABLE_COLUMNS)
    assert "temporal_partition" not in TABLE_COLUMNS
    assert "ibi_t" not in TABLE_COLUMNS
    assert "vbswir1_bi_t" not in TABLE_COLUMNS
    assert not any(column.endswith("_target") for column in TABLE_COLUMNS)



def test_legacy_pixel_coordinates_are_repaired() -> None:
    """Repair pixel-space centres and create unique frozen-grid cell IDs."""
    grid = {
        "width": 946,
        "height": 1278,
        "resolution_m": 30,
        "extent": {
            "xmin": 762570,
            "ymax": 444150,
        },
    }
    frame = pd.DataFrame(
        {
            "cell_id": [13951741, 13951741],
            "row": [14775, 14775],
            "column": [-25409, -25409],
            "x_center_m": [328.5, 329.5],
            "y_center_m": [887.5, 887.5],
            "forecast_origin": [2005, 2005],
        }
    )
    repaired = repair_coordinate_columns(frame, grid)

    assert repaired["column"].tolist() == [328, 329]
    assert repaired["row"].tolist() == [887, 887]
    assert repaired["cell_id"].tolist() == [
        887 * 946 + 328,
        887 * 946 + 329,
    ]
    assert repaired["x_center_m"].tolist() == [
        772425.0,
        772455.0,
    ]
    assert repaired["y_center_m"].tolist() == [
        417525.0,
        417525.0,
    ]
    assert not repaired.duplicated(
        ["cell_id", "forecast_origin"]
    ).any()



def test_demand_is_synchronised_with_cell_time_rows() -> None:
    """Use the released eligible rows as the demand count authority."""
    demand = pd.DataFrame(
        {
            "period_start": [2005],
            "period_end": [2010],
            "eligible_nonbuilt_cells": [4],
            "eligible_nonbuilt_area_ha": [0.359],
            "new_built_cells": [1],
            "observed_new_built_area_ha": [0.0897],
            "common_valid_cells": [7],
            "common_valid_area_ha": [0.628],
        }
    )
    dataset = pd.DataFrame(
        {
            "forecast_origin": [2005, 2005, 2005],
            "target_year": [2010, 2010, 2010],
            "target_transition_5y": [1, 0, 1],
        }
    )
    grid = {"resolution_m": 30}
    result = synchronise_demand_with_cell_time(
        demand,
        dataset,
        grid,
    )

    assert int(result.loc[0, "eligible_nonbuilt_cells"]) == 3
    assert result.loc[
        0,
        "eligible_nonbuilt_area_ha",
    ] == pytest.approx(0.27)
    assert int(result.loc[0, "new_built_cells"]) == 2
    assert result.loc[
        0,
        "observed_new_built_area_ha",
    ] == pytest.approx(0.18)
    assert int(result.loc[0, "common_valid_cells"]) == 7
    assert result.loc[
        0,
        "common_valid_area_ha",
    ] == pytest.approx(0.628)


def test_patch_metrics_use_eight_neighbour_connectivity() -> None:
    """Diagonal built pixels form one patch under the frozen rule."""
    built = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    )
    nump, mps_ha, built_area_ha = compute_patch_metrics(
        built,
        pixel_area_ha=0.09,
    )

    assert nump == 1
    assert built_area_ha == pytest.approx(0.27)
    assert mps_ha == pytest.approx(0.27)


def test_generated_release_when_available(
    config: dict[str, Any],
) -> None:
    """Validate the assembled tables and immutable release metadata."""
    final_dir = project_path(config["outputs"]["final_directory"])
    metadata_dir = project_path(config["metadata"]["directory"])
    cell_path = final_dir / config["outputs"]["cell_time_dataset"]
    demand_path = final_dir / config["outputs"]["historical_demand"]
    metrics_path = (
        final_dir / config["outputs"]["urban_sprawl_metrics"]
    )
    version_path = (
        metadata_dir / config["metadata"]["final_dataset_version"]
    )
    checksum_path = metadata_dir / config["metadata"]["checksums"]
    required = [
        cell_path,
        demand_path,
        metrics_path,
        version_path,
        checksum_path,
    ]

    if not all(path.is_file() for path in required):
        pytest.skip("The final dataset release has not been generated.")

    cell_time = pd.read_parquet(cell_path)
    demand = pd.read_csv(demand_path)
    metrics = pd.read_csv(metrics_path)
    
    # Récupérer les époques depuis la configuration
    epochs = [int(e) for e in config["mapping"]["epochs"]]
    expected_transition_count = len(epochs) - 1

    assert not cell_time.duplicated(
        ["cell_id", "forecast_origin"]
    ).any()
    assert set(cell_time["target_transition_5y"].astype(int)) <= {0, 1}
    assert (cell_time["built_state_final_t"].astype(int) == 0).all()
    assert (
        cell_time["target_year"].astype(int)
        == cell_time["forecast_origin"].astype(int) + 5
    ).all()
    assert sorted(cell_time["forecast_origin"].unique().tolist()) == [
        2000,
        2005,
        2010,
        2015,
        2020,
    ]
    assert len(demand) ==  len(epochs) - 1
    assert np.allclose(
        demand["observed_new_built_area_ha"],
        demand["new_built_cells"] * 0.09,
        atol=1e-6,
    )

    for row in demand.itertuples(index=False):
        group = cell_time[
            (
                cell_time["forecast_origin"].astype(int)
                == int(row.period_start)
            )
            & (
                cell_time["target_year"].astype(int)
                == int(row.period_end)
            )
        ]
        assert int(row.eligible_nonbuilt_cells) == len(group)
        assert int(row.new_built_cells) == int(
            group["target_transition_5y"].astype(int).sum()
        )

    assert metrics["epoch"].astype(int).tolist() == [
        2000,
        2005,
        2010,
        2015,
        2020,
        2025,
    ]
    assert metrics["tracking_support_area_ha"].nunique() == 1
    assert metrics["pba"].between(0, 1).all()

    version = yaml.safe_load(version_path.read_text(encoding="utf-8"))
    assert version["release_status"] == "FROZEN_PROVISIONAL_RELEASE"
    assert version["mapping_method"] == "ndbi"
    assert version["manual_validation_complete"] is False
    assert version["state_count"] == len(epochs)
    assert version["transition_count"] == expected_transition_count
    assert version["immutable_release"] is True

    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split("  ", maxsplit=1)
        observed_path = ROOT / relative_path
        assert observed_path.is_file()
        assert sha256_file(observed_path) == expected
