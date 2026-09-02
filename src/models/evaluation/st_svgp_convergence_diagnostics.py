"""Descriptive convergence diagnostics for existing ST-SVGP training histories."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REQUIRED_COLUMNS = {
    "fold",
    "iteration",
    "elbo_stochastic",
    "minimum_site_precision_eigenvalue",
    "gradient_global_norm",
    "spatial_lengthscale_x_km",
    "spatial_lengthscale_y_km",
    "temporal_lengthscale_steps",
    "kernel_variance",
    "linear_intercept",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_history(path: Path) -> pd.DataFrame:
    """Load one history and validate fields needed by the diagnostics."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"History {path} is missing required fields: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"History {path} contains no logged rows")

    frame = frame.copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = sorted(column for column in REQUIRED_COLUMNS if frame[column].isna().any())
    if invalid:
        raise ValueError(f"History {path} has non-numeric or missing values in: {', '.join(invalid)}")
    return frame.sort_values(["fold", "iteration"]).reset_index(drop=True)


def _percent_change(initial: float, final: float) -> float:
    return float((final - initial) / abs(initial) * 100.0) if initial != 0 else float("nan")


def summarize_fold(
    fold_frame: pd.DataFrame,
    history: dict[str, Any],
    source_path: str,
    source_sha256: str,
    smoothing_window_rows: int,
    tail_window_rows: int,
) -> dict[str, Any]:
    """Summarize descriptive diagnostics for one fold."""
    fold_frame = fold_frame.sort_values("iteration").reset_index(drop=True)
    tail = fold_frame.tail(tail_window_rows)
    tail_elbo = tail["elbo_stochastic"]
    tail_iterations = tail["iteration"].to_numpy(dtype=float)
    if len(tail) >= 2 and np.ptp(tail_iterations) > 0:
        tail_slope = float(np.polyfit(tail_iterations, tail_elbo.to_numpy(), 1)[0] * 100.0)
    else:
        tail_slope = float("nan")

    gradient = fold_frame["gradient_global_norm"]
    clip = float(history["gradient_clip_norm"])
    gradient_ratio = gradient / clip
    site_precision = fold_frame["minimum_site_precision_eigenvalue"]
    site_tail = tail["minimum_site_precision_eigenvalue"]

    x_initial = float(fold_frame["spatial_lengthscale_x_km"].iloc[0])
    x_final = float(fold_frame["spatial_lengthscale_x_km"].iloc[-1])
    y_initial = float(fold_frame["spatial_lengthscale_y_km"].iloc[0])
    y_final = float(fold_frame["spatial_lengthscale_y_km"].iloc[-1])
    variance_initial = float(fold_frame["kernel_variance"].iloc[0])
    variance_final = float(fold_frame["kernel_variance"].iloc[-1])
    temporal_initial = float(fold_frame["temporal_lengthscale_steps"].iloc[0])
    temporal_final = float(fold_frame["temporal_lengthscale_steps"].iloc[-1])
    step_years = float(history["step_years"])

    return {
        "experiment_id": history["experiment_id"],
        "horizon": history["horizon"],
        "fold": fold_frame["fold"].iloc[0],
        "source_path": source_path,
        "source_sha256": source_sha256,
        "step_years": step_years,
        "n_logged_points": len(fold_frame),
        "iteration_first": float(fold_frame["iteration"].iloc[0]),
        "iteration_last": float(fold_frame["iteration"].iloc[-1]),
        "elbo_first": float(fold_frame["elbo_stochastic"].iloc[0]),
        "elbo_last": float(fold_frame["elbo_stochastic"].iloc[-1]),
        "elbo_smoothed_last": float(
            fold_frame["elbo_stochastic"]
            .rolling(smoothing_window_rows, min_periods=1)
            .mean()
            .iloc[-1]
        ),
        "elbo_tail_mean": float(tail_elbo.mean()),
        "elbo_tail_std": float(tail_elbo.std(ddof=0)),
        "elbo_tail_range": float(tail_elbo.max() - tail_elbo.min()),
        "elbo_tail_slope_per_100_iterations": tail_slope,
        "gradient_clip_norm": clip,
        "gradient_p50": float(gradient.quantile(0.50)),
        "gradient_p90": float(gradient.quantile(0.90)),
        "gradient_p95": float(gradient.quantile(0.95)),
        "fraction_gradient_above_clip": float((gradient > clip).mean()),
        "median_gradient_to_clip_ratio": float(gradient_ratio.median()),
        "site_precision_initial": float(site_precision.iloc[0]),
        "site_precision_final": float(site_precision.iloc[-1]),
        "site_precision_min": float(site_precision.min()),
        "site_precision_max": float(site_precision.max()),
        "site_precision_tail_range": float(site_tail.max() - site_tail.min()),
        "spatial_lengthscale_x_initial_km": x_initial,
        "spatial_lengthscale_x_final_km": x_final,
        "spatial_lengthscale_x_change_abs_km": abs(x_final - x_initial),
        "spatial_lengthscale_x_change_pct": _percent_change(x_initial, x_final),
        "spatial_lengthscale_y_initial_km": y_initial,
        "spatial_lengthscale_y_final_km": y_final,
        "spatial_lengthscale_y_change_abs_km": abs(y_final - y_initial),
        "spatial_lengthscale_y_change_pct": _percent_change(y_initial, y_final),
        "kernel_variance_initial": variance_initial,
        "kernel_variance_final": variance_final,
        "kernel_variance_change_pct": _percent_change(variance_initial, variance_final),
        "temporal_lengthscale_initial_steps": temporal_initial,
        "temporal_lengthscale_final_steps": temporal_final,
        "temporal_lengthscale_change_pct": _percent_change(temporal_initial, temporal_final),
        "temporal_lengthscale_initial_years": temporal_initial * step_years,
        "temporal_lengthscale_final_years": temporal_final * step_years,
        "linear_intercept_initial": float(fold_frame["linear_intercept"].iloc[0]),
        "linear_intercept_final": float(fold_frame["linear_intercept"].iloc[-1]),
    }


def make_diagnostic_figure(
    frame: pd.DataFrame,
    history: dict[str, Any],
    smoothing_window_rows: int,
    output_path: Path,
) -> None:
    """Write one six-panel diagnostic figure for a history."""
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")
    step_years = float(history["step_years"])

    for color_index, (fold, fold_frame) in enumerate(frame.groupby("fold", sort=True)):
        fold_frame = fold_frame.sort_values("iteration")
        iteration = fold_frame["iteration"]
        color = colors(color_index % 10)
        label = f"Fold {fold:g}" if isinstance(fold, float) else f"Fold {fold}"
        axes[0].plot(iteration, fold_frame["elbo_stochastic"], color=color, alpha=0.25)
        axes[0].plot(
            iteration,
            fold_frame["elbo_stochastic"].rolling(smoothing_window_rows, min_periods=1).mean(),
            color=color,
            label=label,
        )
        axes[1].plot(iteration, fold_frame["gradient_global_norm"], color=color, label=label)
        axes[2].plot(
            iteration,
            fold_frame["minimum_site_precision_eigenvalue"],
            color=color,
            label=label,
        )
        axes[3].plot(
            iteration,
            fold_frame["spatial_lengthscale_x_km"],
            color=color,
            linestyle="-",
            label=f"{label} X",
        )
        axes[3].plot(
            iteration,
            fold_frame["spatial_lengthscale_y_km"],
            color=color,
            linestyle="--",
            label=f"{label} Y",
        )
        axes[4].plot(
            iteration,
            fold_frame["temporal_lengthscale_steps"] * step_years,
            color=color,
            label=label,
        )
        axes[5].plot(iteration, fold_frame["kernel_variance"], color=color, label=label)

    axes[0].set_title("Stochastic ELBO (raw and rolling mean)")
    axes[1].set_title("Logged gradient global norm")
    axes[1].set_yscale("log")
    axes[1].axhline(
        float(history["gradient_clip_norm"]), color="black", linestyle=":", label="Clip threshold"
    )
    axes[2].set_title("Minimum site precision eigenvalue")
    axes[3].set_title("Spatial lengthscales")
    axes[3].set_ylabel("km")
    axes[4].set_title("Temporal lengthscale")
    axes[4].set_ylabel("years")
    axes[5].set_title("Kernel variance")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle(f"{history['experiment_id']} convergence diagnostics")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _format_number(value: float) -> str:
    return f"{value:,.4g}"


def write_report(summary: pd.DataFrame, histories: list[dict[str, Any]], output_path: Path) -> None:
    """Write a short factual Markdown report from the fold summaries."""
    lines = [
        "# ST-SVGP baseline convergence diagnostics",
        "",
        "## Scope",
        "",
        "This is a read-only analysis of existing histories. No model was trained, no "
        "hyperparameter was changed, and no final or locked test was evaluated.",
        "",
        "## Inputs",
        "",
    ]
    for history in histories:
        rows = summary[summary["experiment_id"] == history["experiment_id"]]
        lines.append(
            f"- `{history['experiment_id']}`: `{history['path']}`; SHA-256 "
            f"`{rows['source_sha256'].iloc[0]}`"
        )

    def add_experiment(experiment_id: str) -> None:
        rows = summary[summary["experiment_id"] == experiment_id]
        lines.extend(["", f"### {experiment_id}", ""])
        for row in rows.to_dict("records"):
            lines.append(
                f"- Fold {row['fold']:g}: ELBO tail range {_format_number(row['elbo_tail_range'])} "
                f"and slope {_format_number(row['elbo_tail_slope_per_100_iterations'])} per 100 "
                f"iterations; median gradient/clip ratio "
                f"{_format_number(row['median_gradient_to_clip_ratio'])}, with "
                f"{row['fraction_gradient_above_clip']:.1%} above the configured threshold; site "
                f"precision {_format_number(row['site_precision_min'])} to "
                f"{_format_number(row['site_precision_max'])}; spatial X "
                f"{_format_number(row['spatial_lengthscale_x_initial_km'])} to "
                f"{_format_number(row['spatial_lengthscale_x_final_km'])} km and Y "
                f"{_format_number(row['spatial_lengthscale_y_initial_km'])} to "
                f"{_format_number(row['spatial_lengthscale_y_final_km'])} km; kernel variance "
                f"{_format_number(row['kernel_variance_initial'])} to "
                f"{_format_number(row['kernel_variance_final'])}; temporal lengthscale "
                f"{_format_number(row['temporal_lengthscale_initial_steps'])} to "
                f"{_format_number(row['temporal_lengthscale_final_steps'])} steps = "
                f"{_format_number(row['temporal_lengthscale_initial_years'])} to "
                f"{_format_number(row['temporal_lengthscale_final_years'])} years."
            )

    lines.extend(["", "## Annual baselines"])
    for experiment_id in ("annual_1000", "annual_1500", "annual_3000"):
        if experiment_id in set(summary["experiment_id"]):
            add_experiment(experiment_id)

    lines.extend(["", "## Five-year baseline"])
    if "five_year_canonical" in set(summary["experiment_id"]):
        add_experiment("five_year_canonical")

    gradient_min = summary["fraction_gradient_above_clip"].min()
    gradient_max = summary["fraction_gradient_above_clip"].max()
    lines.extend(
        [
            "",
            "## Diagnostic observations",
            "",
            f"Across folds, {gradient_min:.1%} to {gradient_max:.1%} of logged gradient norms "
            "exceed the configured clipping threshold. These are logged gradient norms compared "
            "with the threshold, not claimed post-clipping norms. Raw ELBO is stochastic, so "
            "non-monotonic values alone do not demonstrate optimization failure, and raw ELBO "
            "magnitudes are not compared across folds or horizons. Minimum site precision is one "
            "CVI stability diagnostic rather than a complete convergence metric. Temporal "
            "lengthscales are reported in both model steps and physical years.",
            "",
            "## Use in future experiments",
            "",
            "This utility will be reused for later optimizer and kernel experiments so their "
            "diagnostics can be compared with these frozen baselines.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostics(config_path: Path) -> pd.DataFrame:
    """Run configured diagnostics and return the combined summary."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    histories = config.get("histories", [])
    if not histories:
        raise ValueError(f"Config {config_path} defines no histories")
    smoothing_window_rows = int(config["smoothing_window_rows"])
    tail_window_rows = int(config["tail_window_rows"])
    if smoothing_window_rows < 1 or tail_window_rows < 1:
        raise ValueError("smoothing_window_rows and tail_window_rows must be positive")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    summaries: list[dict[str, Any]] = []
    source_hashes: dict[Path, str] = {}

    for history in histories:
        source_path = Path(history["path"])
        source_hash = sha256_file(source_path)
        source_hashes[source_path] = source_hash
        frame = load_history(source_path)
        for _, fold_frame in frame.groupby("fold", sort=True):
            summaries.append(
                summarize_fold(
                    fold_frame,
                    history,
                    str(history["path"]),
                    source_hash,
                    smoothing_window_rows,
                    tail_window_rows,
                )
            )
        make_diagnostic_figure(
            frame,
            history,
            smoothing_window_rows,
            figures_dir / f"{history['experiment_id']}_convergence.png",
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "convergence_summary.csv", index=False)
    write_report(summary, histories, output_dir / "baseline_convergence_report.md")

    changed = [str(path) for path, digest in source_hashes.items() if sha256_file(path) != digest]
    if changed:
        raise RuntimeError(f"Source training histories changed during diagnostics: {', '.join(changed)}")
    return summary


def main() -> None:
    """Parse CLI arguments and run diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run_diagnostics(arguments.config)


if __name__ == "__main__":
    main()