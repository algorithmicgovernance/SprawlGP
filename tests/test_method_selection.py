"""Critical tests for Day 5 method selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis.validation.select_mapping_method import (
    expected_candidate_bands,
    select_method,
    weighted_f1,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/mapping_validation.yaml"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the authoritative Day 5 configuration."""
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def project_path(value: str | Path) -> Path:
    """Resolve one repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def test_candidate_schema_excludes_ibi(
    config: dict[str, Any],
) -> None:
    """Ensure IBI remains outside the candidate-method interface."""
    methods = config["candidate_methods"]
    bands = expected_candidate_bands(methods)

    assert "ibi" not in methods
    assert "built_ibi" not in bands
    assert "valid_ibi" not in bands
    assert len(bands) == 2 * len(methods)


def test_weighted_f1_matches_reference_calculation() -> None:
    """Check the only manual-validation metric used for ranking."""
    truth = np.array([1, 1, 0, 0])
    prediction = np.array([1, 0, 1, 0])
    weights = np.array([2.0, 1.0, 1.0, 2.0])

    assert weighted_f1(
        truth,
        prediction,
        weights,
    ) == pytest.approx(2 / 3)


def test_selection_rule_uses_minimum_f1_then_reversal() -> None:
    """Apply the documented tie-breaking order reproducibly."""
    scores = pd.DataFrame(
        [
            {
                "method": "ndbi",
                "mean_yearly_weighted_f1": 0.80,
                "minimum_yearly_weighted_f1": 0.70,
                "reversal_rate": 0.03,
            },
            {
                "method": "ibui",
                "mean_yearly_weighted_f1": 0.79,
                "minimum_yearly_weighted_f1": 0.75,
                "reversal_rate": 0.05,
            },
            {
                "method": "ndbsui",
                "mean_yearly_weighted_f1": 0.60,
                "minimum_yearly_weighted_f1": 0.50,
                "reversal_rate": 0.01,
            },
        ]
    )
    status, selected = select_method(
        scores,
        tie_tolerance=0.02,
    )

    assert status == "PASS"
    assert selected == "ibui"


def test_selection_returns_manual_decision_for_exact_tie() -> None:
    """Never choose alphabetically when every statistic ties."""
    scores = pd.DataFrame(
        [
            {
                "method": "ndbi",
                "mean_yearly_weighted_f1": 0.80,
                "minimum_yearly_weighted_f1": 0.70,
                "reversal_rate": 0.03,
            },
            {
                "method": "ibui",
                "mean_yearly_weighted_f1": 0.80,
                "minimum_yearly_weighted_f1": 0.70,
                "reversal_rate": 0.03,
            },
        ]
    )
    status, selected = select_method(
        scores,
        tie_tolerance=0.02,
    )

    assert status == "MANUAL_DECISION_REQUIRED"
    assert selected is None


def test_generated_outputs_when_available(
    config: dict[str, Any],
) -> None:
    """Check the minimal score and protocol outputs after execution."""
    metadata = project_path(
        config["outputs"]["metadata_directory"]
    )
    score_path = metadata / config["outputs"]["method_scores"]
    protocol_path = (
        metadata / config["outputs"]["selected_protocol"]
    )

    if not score_path.is_file() or not protocol_path.is_file():
        pytest.skip("Day 5 outputs have not been generated.")

    scores = pd.read_csv(score_path)
    assert set(scores["method"]) == set(
        config["candidate_methods"]
    )
    assert "ibi" not in set(scores["method"])
    assert scores["mean_yearly_weighted_f1"].between(
        0,
        1,
    ).all()
    assert scores["minimum_yearly_weighted_f1"].between(
        0,
        1,
    ).all()
    assert scores["reversal_rate"].between(0, 1).all()

    protocol = yaml.safe_load(
        protocol_path.read_text(encoding="utf-8")
    )
    assert protocol["selection_status"] in {
        "PASS",
        "MANUAL_DECISION_REQUIRED",
    }

    if protocol["selection_status"] == "PASS":
        selected = protocol["selected_method"]["index"]
        assert selected in config["candidate_methods"]
        assert protocol["selected_method"][
            "candidate_band"
        ] == f"built_{selected}"
        assert protocol["selected_method"][
            "validity_band"
        ] == f"valid_{selected}"
