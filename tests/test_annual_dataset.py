"""Static and generated-output checks for the isolated annual dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from src.analysis.annual_dataset.build_products import (
    STATE_BANDS,
    TRANSITION_BANDS,
    annual_state_recipe,
    annual_transitions,
    annual_years,
    apply_absorbing_persistence_numpy,
    load_annual_sources,
    population_source_year,
    transition_numpy,
    validate_annual_config,
    validate_state_asset_properties,
)
from src.analysis.annual_dataset.build_tables import (
    PREDICTOR_COLUMNS,
    TABLE_COLUMNS,
    assert_annual_schema_has_no_leakage,
    load_composite_assets,
    prior_growth_transition,
)
from src.analysis.annual_dataset.finalize import data_dictionary
from src.analysis.orchestration.common import stable_object_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/annual/annual_dataset.yaml"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the isolated annual dataset configuration."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_annual_year_and_transition_contract(config: dict[str, Any]) -> None:
    """Pin 26 annual states and exactly 25 one-year transitions."""
    years = validate_annual_config(config)
    transitions = annual_transitions(years)
    assert years == list(range(2000, 2026))
    assert len(transitions) == 25
    assert all(target == origin + 1 for origin, target in transitions)
    assert transitions[-1] == (2024, 2025)


def test_annual_schema_is_horizon_specific() -> None:
    """Require annual labels and diagnostics without five-year names."""
    schema = STATE_BANDS + TRANSITION_BANDS + TABLE_COLUMNS
    assert "built_state_raw" in schema
    assert "built_state_final" in schema
    assert "target_transition_1y" in schema
    assert "raw_built_to_nonbuilt_reversal" in schema
    assert "persistence_correction_at_target" in schema
    assert "target_transition_5y" not in schema
    assert "recent_local_growth_5y_t" not in schema
    assert "recent_local_growth_1y_t" in schema


def test_predictor_schema_has_no_target_year_inputs() -> None:
    """Keep every predictor at the forecast origin or earlier."""
    assert_annual_schema_has_no_leakage(TABLE_COLUMNS)
    assert not any("target" in column.casefold() for column in PREDICTOR_COLUMNS)


def test_recent_growth_uses_only_the_transition_ending_at_origin() -> None:
    """Make 2000 unavailable and all later growth strictly historical."""
    years = list(range(2000, 2026))
    assert prior_growth_transition(2000, years) is None
    for origin in range(2001, 2025):
        start, end = prior_growth_transition(origin, years) or (None, None)
        assert start == origin - 1
        assert end == origin


def test_annual_config_is_ndbi_only_and_isolated(config: dict[str, Any]) -> None:
    """Keep annual writes non-overwriting and outside legacy v2 assets."""
    assert config["mapping"]["selected_index"] == "ndbi"
    assert config["mapping"]["mapping_validation_status"] == (
        "PROVISIONAL_ANNUAL_PROTOCOL"
    )
    assert config["exports"]["overwrite_existing_assets"] is False
    assert config["exports"]["asset_root"].endswith("/annual_v1")
    assert "/assets/sprawlgp/v2" not in CONFIG_PATH.read_text(encoding="utf-8")
    assert config["outputs"]["final_directory"].endswith(
        "yaounde_urban_expansion_30m_annual_v1"
    )


def test_required_audit_outputs_are_configured(config: dict[str, Any]) -> None:
    """Pin the five evidence tables/report and two required figures."""
    audit = config["audit"]
    assert audit["annual_state_summary"] == "annual_state_summary.csv"
    assert audit["annual_transition_summary"] == "annual_transition_summary.csv"
    assert audit["anchor_protocol_comparison"] == "anchor_protocol_comparison.csv"
    assert audit["five_year_block_comparison"] == "five_year_block_comparison.csv"
    assert audit["anomaly_report"] == "anomaly_2015_2020.md"
    assert audit["built_fraction_figure"] == "annual_built_fraction.png"
    assert audit["new_built_area_figure"] == "annual_new_built_area.png"


def test_day2_manifest_is_the_only_landsat_source(config: dict[str, Any]) -> None:
    """Resolve all annual states from finalized Day 2 asset manifests."""
    years = annual_years(config)
    manifest = ROOT / config["inputs"]["landsat_composite_manifest"]
    sources = load_annual_sources(manifest, years)
    assert len(sources) == 26
    assert sources[0].composite_asset.endswith("/annual_v1/landsat/composite_2000")
    assert sources[-1].count_asset.endswith("/annual_v1/landsat/valid_count_2025")


def test_table_composites_reject_duplicate_or_nonannual_sources(tmp_path: Path) -> None:
    """Keep table reflectance pinned to one annual_v1 composite per year."""
    duplicate = pd.DataFrame(
        {
            "epoch": [2000, 2000],
            "asset_id": [
                "projects/urban-sprawl-ssa/assets/sprawlgp/annual_v1/landsat/composite_2000",
                "projects/urban-sprawl-ssa/assets/sprawlgp/annual_v1/landsat/composite_2000",
            ],
            "status": ["PASS", "PASS"],
        }
    )
    path = tmp_path / "manifest.csv"
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="one passed composite"):
        load_composite_assets(path, [2000])

    duplicate.iloc[:1].assign(
        asset_id="projects/urban-sprawl-ssa/assets/sprawlgp/v2/landsat/composite_2000"
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected annual composite"):
        load_composite_assets(path, [2000])


def test_data_dictionary_covers_the_annual_table_schema() -> None:
    """Document every released annual table column."""
    documented = set(data_dictionary()["cell_time_dataset"])
    assert set(TABLE_COLUMNS).issubset(documented)


def test_round_trip_threshold_csv_preserves_recipe_hash(tmp_path: Path) -> None:
    """Reconstruct the submitted recipe hash from serialized threshold floats."""
    thresholds = {
        "ndbi": -0.1406589466877592,
        "ibui": -0.03128487276649813,
        "ndbsui": -0.06835142346372955,
    }
    grid = {
        "crs": "EPSG:32632",
        "transform": [30, 0, 762570, 0, -30, 444150],
    }
    source_composite = "annual_v1/landsat/composite_2000"
    source_count = "annual_v1/landsat/valid_count_2000"
    original = annual_state_recipe(
        year=2000,
        thresholds=thresholds,
        source_composite_asset=source_composite,
        source_count_asset=source_count,
        grid=grid,
    )
    path = tmp_path / "annual_thresholds.csv"
    pd.DataFrame(
        [
            {"year": 2000, "index_name": name, "threshold": value}
            for name, value in thresholds.items()
        ]
    ).to_csv(path, index=False)
    restored = pd.read_csv(path, float_precision="round_trip")
    restored_thresholds = {
        str(row.index_name): float(row.threshold)
        for row in restored.itertuples(index=False)
    }
    reconstructed = annual_state_recipe(
        year=2000,
        thresholds=restored_thresholds,
        source_composite_asset=source_composite,
        source_count_asset=source_count,
        grid=grid,
    )
    assert stable_object_hash(reconstructed) == stable_object_hash(original)


def test_asset_recipe_hash_must_match_task_manifest_hash() -> None:
    """Reject an asset recipe hash that differs from the submitted task hash."""
    expected = {
        "recipe_sha256": "task-manifest-hash",
        "year": 2000,
        "mapping_method": "ndbi",
    }
    observed = {**expected, "recipe_sha256": "asset-hash"}
    with pytest.raises(
        ValueError,
        match=(
            "recipe_sha256: expected 'task-manifest-hash', "
            "observed 'asset-hash'"
        ),
    ):
        validate_state_asset_properties("state_2000", observed, expected)


def test_population_lookup_never_uses_future_epochs(config: dict[str, Any]) -> None:
    """Use only the latest configured GHSL epoch available by each origin."""
    epochs = config["population"]["epochs"]
    for origin in range(2000, 2025):
        source = population_source_year(origin, epochs)
        assert source <= origin
    assert population_source_year(2006, epochs) == 2005
    with pytest.raises(ValueError, match="No population epoch"):
        population_source_year(1989, epochs)


def test_annual_persistence_reuses_existing_absorbing_rule() -> None:
    """Preserve raw reversals while final states remain absorbing."""
    raw = np.array([[0, 1], [1, 0], [0, 0]])
    valid = np.ones_like(raw, dtype=bool)
    final, corrected = apply_absorbing_persistence_numpy(raw, valid)
    assert raw[:, 1].tolist() == [1, 0, 0]
    assert final[:, 1].tolist() == [1, 1, 1]
    assert corrected[:, 1].tolist() == [0, 1, 1]


def test_transition_labels_only_pairwise_valid_nonbuilt_origins() -> None:
    """Reject positive labels outside pairwise-valid eligible cells."""
    common, eligible, target = transition_numpy(
        np.array([0, 0, 1, 0]),
        np.array([1, 0, 1, 1]),
        np.array([1, 1, 1, 0], dtype=bool),
        np.array([1, 1, 1, 1], dtype=bool),
    )
    assert np.all(target <= eligible)
    assert np.all(target <= common)


def test_generated_annual_dataset_when_available(config: dict[str, Any]) -> None:
    """Apply integration assertions when the annual table has been assembled."""
    path = ROOT / config["outputs"]["final_directory"] / config["outputs"][
        "cell_time_dataset"
    ]
    if not path.is_file():
        pytest.skip("Annual cell-time dataset has not been assembled locally.")
    frame = pd.read_parquet(path)
    assert frame["forecast_origin"].nunique() == 25
    assert sorted(frame["forecast_origin"].unique())[-1] == 2024
    assert (frame["target_year"] == frame["forecast_origin"] + 1).all()
    assert (frame["population_source_year"] <= frame["forecast_origin"]).all()
    assert "target_transition_5y" not in frame.columns
    assert "recent_local_growth_5y_t" not in frame.columns
    assert "target_transition_1y" in frame.columns
    assert "recent_local_growth_1y_t" in frame.columns
    first = frame[frame["forecast_origin"] == 2000]
    assert not first["recent_local_growth_available_t"].astype(bool).any()


def test_generated_annual_summaries_when_available(config: dict[str, Any]) -> None:
    """Validate state persistence accounting and all adjacent transitions."""
    metadata = ROOT / config["metadata"]["directory"]
    state_path = metadata / config["metadata"]["state_summary"]
    transition_path = metadata / config["metadata"]["transition_summary"]
    if not state_path.is_file() or not transition_path.is_file():
        pytest.skip("Annual state and transition summaries have not been generated.")
    states = pd.read_csv(state_path)
    transitions = pd.read_csv(transition_path)
    assert states["year"].astype(int).tolist() == list(range(2000, 2026))
    assert states["year"].nunique() == 26
    np.testing.assert_allclose(
        states["final_built_area_ha_core"],
        states["raw_built_area_ha_core"]
        + states["persistence_corrected_area_ha"],
        atol=1e-6,
        rtol=0,
    )
    # Vector-core reduceRegion weights boundary pixels fractionally before rounding.
    assert (
        states["final_built_cells_core"]
        - states["raw_built_cells_core"]
        - states["persistence_corrected_cells"]
    ).abs().le(1).all()
    assert len(transitions) == 25
    assert not transitions.duplicated(["forecast_origin", "target_year"]).any()
    assert (
        transitions["target_year"].astype(int)
        == transitions["forecast_origin"].astype(int) + 1
    ).all()