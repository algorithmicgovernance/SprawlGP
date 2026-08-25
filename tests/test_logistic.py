"""Tests for the selected Logistic Regression baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.train_logistic import (
    add_log_population,
    build_pipeline,
    summarise_folds,
    validate_temporal_contract,
)


def test_log_population_uses_log1p() -> None:
    frame = pd.DataFrame({"population_density_t": [0.0, 9.0, 99.0]})
    config = {
        "derived_features": {
            "log_population_density_t": {
                "source": "population_density_t",
                "transformation": "log1p",
            }
        }
    }
    result = add_log_population(frame, config)
    assert np.allclose(
        result["log_population_density_t"],
        np.log1p(np.array([0.0, 9.0, 99.0])),
    )


def test_pipeline_uses_fixed_c() -> None:
    config = {
        "logistic_regression": {
            "C": 0.1,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 20260807,
        }
    }
    pipeline = build_pipeline(config)
    assert pipeline.named_steps["classifier"].C == 0.1


def test_temporal_contract_accepts_selected_design() -> None:
    config = {
        "rolling_validation": {
            "folds": [
                {"train_origins": [2000], "validation_origin": 2005},
                {"train_origins": [2000, 2005], "validation_origin": 2010},
                {
                    "train_origins": [2000, 2005, 2010],
                    "validation_origin": 2015,
                },
            ]
        },
        "final_fit": {"origins": [2000, 2005, 2010, 2015]},
        "final_test": {"origins": [2020], "evaluate": False},
    }
    validate_temporal_contract(config)


def test_final_test_cannot_be_enabled() -> None:
    config = {
        "rolling_validation": {
            "folds": [{"train_origins": [2000], "validation_origin": 2005}]
        },
        "final_fit": {"origins": [2000]},
        "final_test": {"origins": [2005], "evaluate": True},
    }
    with pytest.raises(ValueError):
        validate_temporal_contract(config)


def test_fold_summary_uses_equal_period_weight() -> None:
    metrics = pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "log_loss": [0.3, 0.4, 0.8],
            "brier_score": [0.1, 0.12, 0.2],
            "pr_auc": [0.6, 0.4, 0.7],
            "roc_auc": [0.85, 0.84, 0.80],
            "probability_bias": [-0.02, 0.15, -0.21],
            "constant_log_loss": [0.5, 0.35, 0.75],
            "constant_brier_score": [0.16, 0.10, 0.26],
        }
    )
    summary = summarise_folds(metrics).iloc[0]
    assert summary["mean_log_loss"] == pytest.approx(0.5)
