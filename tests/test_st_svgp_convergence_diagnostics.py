import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from src.models.evaluation.st_svgp_convergence_diagnostics import (
    run_diagnostics,
    summarize_fold,
)


def _synthetic_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [1] * 5,
            "iteration": [0, 50, 100, 150, 200],
            "elbo_stochastic": [-100.0, -90.0, -95.0, -80.0, -75.0],
            "minimum_site_precision_eigenvalue": [0.1, 0.5, 0.3, 0.8, 0.4],
            "gradient_global_norm": [5.0, 10.0, 15.0, 20.0, 25.0],
            "spatial_lengthscale_x_km": [2.0, 2.5, 3.0, 3.5, 4.0],
            "spatial_lengthscale_y_km": [4.0, 4.5, 5.0, 5.5, 6.0],
            "temporal_lengthscale_steps": [1.5, 1.6, 1.7, 1.8, 2.0],
            "kernel_variance": [1.0, 0.9, 0.8, 0.7, 0.5],
            "linear_intercept": [0.0, -0.1, -0.2, -0.3, -0.4],
        }
    )


def test_summarize_fold_math() -> None:
    summary = summarize_fold(
        _synthetic_history(),
        {
            "experiment_id": "synthetic",
            "horizon": "five_year_5y",
            "step_years": 5,
            "gradient_clip_norm": 10.0,
        },
        "synthetic.csv",
        "digest",
        smoothing_window_rows=3,
        tail_window_rows=3,
    )

    assert summary["fraction_gradient_above_clip"] == 0.6
    assert summary["median_gradient_to_clip_ratio"] == 1.5
    assert summary["spatial_lengthscale_x_initial_km"] == 2.0
    assert summary["spatial_lengthscale_x_final_km"] == 4.0
    assert summary["spatial_lengthscale_x_change_abs_km"] == 2.0
    assert summary["spatial_lengthscale_x_change_pct"] == 100.0
    assert summary["temporal_lengthscale_initial_years"] == 7.5
    assert summary["temporal_lengthscale_final_years"] == 10.0
    assert np.isfinite(summary["elbo_tail_mean"])
    assert np.isfinite(summary["elbo_tail_slope_per_100_iterations"])


def test_tiny_read_only_diagnostic(tmp_path: Path) -> None:
    source_path = tmp_path / "history.csv"
    _synthetic_history().to_csv(source_path, index=False)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "smoothing_window_rows": 3,
                "tail_window_rows": 3,
                "histories": [
                    {
                        "experiment_id": "synthetic",
                        "horizon": "five_year_5y",
                        "path": str(source_path),
                        "step_years": 5,
                        "gradient_clip_norm": 10.0,
                    }
                ],
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    run_diagnostics(config_path)

    summary_path = output_dir / "convergence_summary.csv"
    assert summary_path.exists()
    assert (output_dir / "figures" / "synthetic_convergence.png").exists()
    assert (output_dir / "baseline_convergence_report.md").exists()
    summary = pd.read_csv(summary_path)
    assert {
        "experiment_id",
        "elbo_tail_range",
        "fraction_gradient_above_clip",
        "temporal_lengthscale_final_years",
    }.issubset(summary.columns)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash