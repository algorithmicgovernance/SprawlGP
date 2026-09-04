"""OOF failure diagnostics for the retained annual and five-year ST-SVGPs.

This module reads existing pre-test out-of-fold predictions only. It never
fits a model and never evaluates either locked temporal block.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import spearmanr
from src.feature_engineering.urban_expansion import add_candidate_features

DEFAULT_CONFIG_PATH = Path("configs/modeling/day5_st_svgp_diagnostics.yaml")
JOIN_KEYS = ("cell_id", "forecast_origin")
DISTANCE_COLUMN = "log_distance_to_built_m_t"
DISTANCE_LABELS = ("near_built", "intermediate", "peripheral")


@dataclass(frozen=True)
class HorizonSpec:
    """Resolved immutable inputs for one forecast horizon."""

    name: str
    label: str
    model_config_path: Path
    output_subdirectory: str
    maximum_target_year: int | None
    locked_forecast_origins: tuple[int, ...]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def resolve_horizons(config: dict[str, Any]) -> list[HorizonSpec]:
    """Resolve the two configured horizons without duplicating model science."""
    result: list[HorizonSpec] = []
    for name, values in config["horizons"].items():
        result.append(
            HorizonSpec(
                name=str(name),
                label=str(values["label"]),
                model_config_path=Path(values["model_config"]),
                output_subdirectory=str(values["output_subdirectory"]),
                maximum_target_year=(
                    int(values["maximum_target_year"])
                    if "maximum_target_year" in values
                    else None
                ),
                locked_forecast_origins=tuple(
                    int(value) for value in values.get("locked_forecast_origins", [])
                ),
            )
        )
    return result


def _require_unique_keys(frame: pd.DataFrame, label: str) -> None:
    missing = sorted(set(JOIN_KEYS).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing join keys: {', '.join(missing)}")
    duplicates = frame.duplicated(list(JOIN_KEYS), keep=False)
    if duplicates.any():
        raise ValueError(
            f"{label} contains {int(duplicates.sum())} rows with duplicate "
            f"{'+'.join(JOIN_KEYS)} keys."
        )


def _validate_temporal_lock(
    oof: pd.DataFrame,
    spec: HorizonSpec,
    model_config: dict[str, Any],
) -> None:
    target_year = pd.to_numeric(oof["target_year"], errors="raise").astype(int)
    forecast_origin = pd.to_numeric(oof["forecast_origin"], errors="raise").astype(int)

    if spec.maximum_target_year is not None and target_year.gt(spec.maximum_target_year).any():
        raise ValueError(
            f"{spec.name} OOF predictions contain target_year after "
            f"{spec.maximum_target_year}."
        )
    if spec.locked_forecast_origins and forecast_origin.isin(
        spec.locked_forecast_origins
    ).any():
        raise ValueError(f"{spec.name} OOF predictions contain a locked forecast origin.")

    horizon_years = int(model_config["time"]["step_years"])
    if not target_year.eq(forecast_origin + horizon_years).all():
        raise ValueError("OOF target_year does not equal forecast_origin plus the horizon.")


def _validate_folds(oof: pd.DataFrame, model_config: dict[str, Any]) -> None:
    expected = {
        fold_number: int(fold["validation_origin"])
        for fold_number, fold in enumerate(
            model_config["rolling_validation"]["folds"], start=1
        )
    }
    actual_folds = set(pd.to_numeric(oof["fold"], errors="raise").astype(int).unique())
    if actual_folds != set(expected):
        raise ValueError(f"OOF folds must be exactly {sorted(expected)}.")
    for fold_number, validation_origin in expected.items():
        origins = set(
            pd.to_numeric(
                oof.loc[oof["fold"].eq(fold_number), "forecast_origin"], errors="raise"
            ).astype(int)
        )
        if origins != {validation_origin}:
            raise ValueError(
                f"Fold {fold_number} must contain validation origin {validation_origin}."
            )


def join_oof_to_modeling_data(
    oof: pd.DataFrame,
    modeling_data: pd.DataFrame,
    *,
    spec: HorizonSpec,
    model_config: dict[str, Any],
    probability_epsilon: float,
) -> pd.DataFrame:
    """Join OOF rows one-to-one and derive deterministic row diagnostics."""
    dataset = model_config["dataset"]
    target = str(dataset["target"])
    predictors = [str(value) for value in model_config["linear_predictors"]]
    coordinate_columns = [str(dataset["x_coordinate"]), str(dataset["y_coordinate"])]
    required_oof = {*JOIN_KEYS, "target_year", target, "probability_raw", "fold"}
    missing_oof = sorted(required_oof.difference(oof.columns))
    if missing_oof:
        raise ValueError(f"OOF predictions are missing: {', '.join(missing_oof)}")

    _require_unique_keys(oof, "OOF predictions")
    _validate_temporal_lock(oof, spec, model_config)
    _validate_folds(oof, model_config)

    recent_growth = str(
        model_config.get("feature_engineering", {}).get(
            "recent_growth_column", "recent_local_growth_5y_t"
        )
    )
    prepared_data = add_candidate_features(
        modeling_data, recent_growth_column=recent_growth
    )
    required_data = {*JOIN_KEYS, target, *coordinate_columns, *predictors}
    missing_data = sorted(required_data.difference(prepared_data.columns))
    if missing_data:
        raise ValueError(f"Modeling data are missing: {', '.join(missing_data)}")
    _require_unique_keys(prepared_data, "Modeling data")

    data_columns = [*JOIN_KEYS, target, *coordinate_columns, *predictors]
    data_part = prepared_data.loc[:, data_columns].rename(
        columns={target: "_dataset_target"}
    )
    joined = oof.merge(
        data_part,
        on=list(JOIN_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing_rows = int(joined["_merge"].ne("both").sum())
        raise ValueError(f"{missing_rows} OOF rows did not join to modeling data.")
    if not joined[target].eq(joined["_dataset_target"]).all():
        raise ValueError("OOF and modeling-data targets differ after the keyed join.")

    probability = pd.to_numeric(joined["probability_raw"], errors="raise").astype(float)
    y_true = pd.to_numeric(joined[target], errors="raise").astype(int)
    if not np.isfinite(probability).all() or probability.lt(0.0).any() or probability.gt(1.0).any():
        raise ValueError("OOF probabilities must be finite and in [0, 1].")
    if not set(y_true.unique()).issubset({0, 1}):
        raise ValueError("OOF targets must be binary.")

    clipped = probability.clip(probability_epsilon, 1.0 - probability_epsilon)
    joined["y_true"] = y_true
    joined["signed_error"] = probability - y_true
    joined["absolute_error"] = joined["signed_error"].abs()
    joined["per_cell_log_loss"] = -(
        y_true * np.log(clipped) + (1 - y_true) * np.log1p(-clipped)
    )
    return joined.drop(columns=[target, "_dataset_target", "_merge"])


def assign_distance_context(
    frame: pd.DataFrame, quantiles: tuple[float, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign relative near/intermediate/peripheral contexts within each fold."""
    result = frame.copy()
    result["distance_context"] = pd.NA
    cut_rows: list[dict[str, float | int]] = []
    for fold, index in result.groupby("fold", sort=True).groups.items():
        distance = pd.to_numeric(result.loc[index, DISTANCE_COLUMN], errors="raise")
        lower, upper = (float(value) for value in distance.quantile(quantiles).tolist())
        if not lower < upper:
            raise ValueError(f"Fold {fold} distance tertile cut points are not distinct.")
        result.loc[index, "distance_context"] = pd.cut(
            distance,
            bins=[-np.inf, lower, upper, np.inf],
            labels=DISTANCE_LABELS,
            include_lowest=True,
        ).astype("string")
        cut_rows.append(
            {"fold": int(fold), "near_built_upper": lower, "intermediate_upper": upper}
        )
    result["distance_context"] = pd.Categorical(
        result["distance_context"], categories=DISTANCE_LABELS, ordered=True
    )
    return result, pd.DataFrame(cut_rows)


def assign_error_level(
    frame: pd.DataFrame, quantiles: tuple[float, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign low/middle/high log-loss strata within each fold and true class."""
    result = frame.copy()
    result["error_level"] = "middle"
    threshold_rows: list[dict[str, float | int]] = []
    for (fold, y_true), index in result.groupby(["fold", "y_true"], sort=True).groups.items():
        loss = result.loc[index, "per_cell_log_loss"]
        lower, upper = (float(value) for value in loss.quantile(quantiles).tolist())
        result.loc[index[loss.le(lower)], "error_level"] = "low_error"
        result.loc[index[loss.ge(upper)], "error_level"] = "high_error"
        threshold_rows.append(
            {
                "fold": int(fold),
                "y_true": int(y_true),
                "low_error_upper": lower,
                "high_error_lower": upper,
            }
        )
    result["error_level"] = pd.Categorical(
        result["error_level"],
        categories=("low_error", "middle", "high_error"),
        ordered=True,
    )
    return result, pd.DataFrame(threshold_rows)


def summarize_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize failure diagnostics for the five requested grouping designs."""
    designs = {
        "fold": ["fold"],
        "true_class": ["y_true"],
        "distance_context": ["distance_context"],
        "fold_x_distance_context": ["fold", "distance_context"],
        "fold_x_true_class": ["fold", "y_true"],
    }
    tables: list[pd.DataFrame] = []
    for group_type, columns in designs.items():
        table = (
            frame.groupby(columns, observed=True, dropna=False, sort=True)
            .agg(
                rows=("cell_id", "size"),
                positive_prevalence=("y_true", "mean"),
                mean_predicted_probability=("probability_raw", "mean"),
                mean_signed_error=("signed_error", "mean"),
                mean_absolute_error=("absolute_error", "mean"),
                mean_per_cell_log_loss=("per_cell_log_loss", "mean"),
                median_per_cell_log_loss=("per_cell_log_loss", "median"),
                mean_latent_variance=("latent_variance", "mean")
                if "latent_variance" in frame
                else ("absolute_error", lambda values: np.nan),
                median_latent_variance=("latent_variance", "median")
                if "latent_variance" in frame
                else ("absolute_error", lambda values: np.nan),
            )
            .reset_index()
        )
        table.insert(0, "group_type", group_type)
        tables.append(table)
    return pd.concat(tables, ignore_index=True, sort=False)


def uncertainty_summaries(
    frame: pd.DataFrame, bins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fold/overall uncertainty-error correlations and quintile summaries."""
    group_values: list[int | str] = [*sorted(frame["fold"].unique()), "overall"]
    correlation_rows: list[dict[str, Any]] = []
    quintile_tables: list[pd.DataFrame] = []
    for fold in group_values:
        part = frame if fold == "overall" else frame.loc[frame["fold"].eq(fold)]
        if "latent_variance" not in part:
            correlation_rows.append(
                {
                    "fold": fold,
                    "latent_variance_available": False,
                    "spearman_log_loss": np.nan,
                    "spearman_absolute_error": np.nan,
                }
            )
            continue

        latent = part["latent_variance"].to_numpy(dtype=float)
        correlation_rows.append(
            {
                "fold": fold,
                "latent_variance_available": True,
                "spearman_log_loss": float(
                    spearmanr(latent, part["per_cell_log_loss"]).statistic
                ),
                "spearman_absolute_error": float(
                    spearmanr(latent, part["absolute_error"]).statistic
                ),
            }
        )
        binned = part.assign(
            latent_uncertainty_quintile=pd.qcut(
                part["latent_variance"].rank(method="first"), labels=False, q=bins
            )
            + 1
        )
        table = (
            binned.groupby("latent_uncertainty_quintile", sort=True)
            .agg(
                rows=("cell_id", "size"),
                mean_per_cell_log_loss=("per_cell_log_loss", "mean"),
                mean_absolute_error=("absolute_error", "mean"),
                positive_prevalence=("y_true", "mean"),
            )
            .reset_index()
        )
        table.insert(0, "fold", fold)
        quintile_tables.append(table)
    quintiles = (
        pd.concat(quintile_tables, ignore_index=True)
        if quintile_tables
        else pd.DataFrame(
            columns=[
                "fold",
                "latent_uncertainty_quintile",
                "rows",
                "mean_per_cell_log_loss",
                "mean_absolute_error",
                "positive_prevalence",
            ]
        )
    )
    return pd.DataFrame(correlation_rows), quintiles


def _read_horizon_inputs(
    spec: HorizonSpec, model_config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Read one OOF artifact and only its validation-origin modeling rows."""
    predictions_path = (
        Path(model_config["outputs"]["predictions_directory"])
        / "st_svgp_oof_predictions.parquet"
    )
    dataset_path = Path(model_config["dataset"]["path"])
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    oof = pd.read_parquet(predictions_path)
    dataset = model_config["dataset"]
    predictors = [str(value) for value in model_config["linear_predictors"]]
    recent_growth = str(
        model_config.get("feature_engineering", {}).get(
            "recent_growth_column", "recent_local_growth_5y_t"
        )
    )
    engineered = {
        "log_distance_to_built_m_t",
        "log_population_density_t",
        "built_fraction_x_recent_growth_t",
    }
    columns = {
        *JOIN_KEYS,
        str(dataset["target"]),
        str(dataset["x_coordinate"]),
        str(dataset["y_coordinate"]),
        "population_density_t",
        "distance_to_built_m_t",
        "slope_degrees",
        "built_fraction_11x11_t",
        recent_growth,
        *[predictor for predictor in predictors if predictor not in engineered],
    }
    origins = sorted(pd.to_numeric(oof["forecast_origin"], errors="raise").unique())
    modeling_data = pd.read_parquet(
        dataset_path,
        columns=sorted(columns),
        filters=[("forecast_origin", "in", [int(value) for value in origins])],
    )
    return oof, modeling_data, predictions_path


def prepare_horizon_diagnostics(
    spec: HorizonSpec, day5_config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load, validate, join, and stratify one horizon's retained OOF rows."""
    model_config = load_yaml(spec.model_config_path)
    if model_config.get("model") != "st_svgp":
        raise ValueError(f"Expected an ST-SVGP model config: {spec.model_config_path}")
    oof, modeling_data, predictions_path = _read_horizon_inputs(spec, model_config)
    joined = join_oof_to_modeling_data(
        oof,
        modeling_data,
        spec=spec,
        model_config=model_config,
        probability_epsilon=float(day5_config["probability_epsilon"]),
    )
    distance_quantiles = tuple(
        float(value) for value in day5_config["diagnostics"]["distance_quantiles"]
    )
    error_quantiles = tuple(
        float(value) for value in day5_config["diagnostics"]["error_quantiles"]
    )
    if len(distance_quantiles) != 2 or len(error_quantiles) != 2:
        raise ValueError("Distance and error diagnostics each require two quantiles.")
    joined, distance_cuts = assign_distance_context(joined, distance_quantiles)
    joined, error_thresholds = assign_error_level(joined, error_quantiles)

    metadata = {
        "horizon": spec.name,
        "label": spec.label,
        "status": "PASS",
        "rows": int(len(joined)),
        "folds": sorted(int(value) for value in joined["fold"].unique()),
        "forecast_origins": sorted(
            int(value) for value in joined["forecast_origin"].unique()
        ),
        "target_years": sorted(int(value) for value in joined["target_year"].unique()),
        "oof_columns": list(oof.columns),
        "join_keys": list(JOIN_KEYS),
        "model_config": str(spec.model_config_path),
        "predictions_path": str(predictions_path),
        "dataset_path": str(model_config["dataset"]["path"]),
        "latent_variance_available": "latent_variance" in joined,
        "locked_rows_used": 0,
        "model_state_directory_exists": Path(
            model_config["outputs"]["model_directory"]
        ).is_dir(),
    }
    return joined, distance_cuts, error_thresholds, metadata


def _map_fold(
    frame: pd.DataFrame,
    *,
    label: str,
    output_path: Path,
    extent: tuple[float, float, float, float],
    dpi: int,
) -> None:
    """Write one five-panel projected-coordinate validation map."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes_flat = axes.ravel()
    x = frame["x_center_m"]
    y = frame["y_center_m"]
    observed_cmap = ListedColormap(["#f4f1de", "#c1121f"])
    observed_norm = BoundaryNorm([-0.5, 0.5, 1.5], observed_cmap.N)
    panels: list[tuple[str, str, str, dict[str, Any]]] = [
        ("Observed transition", "y_true", observed_cmap, {"norm": observed_norm}),
        ("Predicted probability", "probability_raw", "viridis", {"vmin": 0, "vmax": 1}),
        ("Signed error", "signed_error", "coolwarm", {"vmin": -1, "vmax": 1}),
        (
            "Per-cell log loss",
            "per_cell_log_loss",
            "magma",
            {"vmin": 0, "vmax": float(frame["per_cell_log_loss"].quantile(0.99))},
        ),
    ]
    for axis, (title, column, cmap, color_options) in zip(axes_flat, panels, strict=False):
        points = axis.scatter(
            x,
            y,
            c=frame[column],
            s=1.0,
            linewidths=0,
            cmap=cmap,
            rasterized=True,
            **color_options,
        )
        axis.set_title(title)
        figure.colorbar(points, ax=axis, fraction=0.046, pad=0.02)

    variance_axis = axes_flat[4]
    if "latent_variance" in frame:
        variance_points = variance_axis.scatter(
            x,
            y,
            c=frame["latent_variance"],
            s=1.0,
            linewidths=0,
            cmap="cividis",
            rasterized=True,
        )
        figure.colorbar(variance_points, ax=variance_axis, fraction=0.046, pad=0.02)
    else:
        variance_axis.text(
            0.5,
            0.5,
            "Unavailable in retained\nOOF artifact",
            ha="center",
            va="center",
            fontsize=11,
            transform=variance_axis.transAxes,
        )
    variance_axis.set_title("Latent posterior variance")
    axes_flat[5].axis("off")

    for axis in axes_flat[:5]:
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Projected x (m)")
        axis.set_ylabel("Projected y (m)")

    fold = int(frame["fold"].iloc[0])
    origin = int(frame["forecast_origin"].iloc[0])
    target_year = int(frame["target_year"].iloc[0])
    figure.suptitle(f"{label}: fold {fold}, {origin} to {target_year}", fontsize=15)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_uncertainty_relationship(
    quintiles: pd.DataFrame, *, label: str, output_path: Path, dpi: int
) -> None:
    """Plot binned latent-variance relationships without raw-point overplotting."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if quintiles.empty:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "Latent posterior variance unavailable\nin retained OOF artifact",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
    else:
        for fold, part in quintiles.groupby("fold", sort=False):
            line_label = "Overall" if fold == "overall" else f"Fold {fold}"
            axes[0].plot(
                part["latent_uncertainty_quintile"],
                part["mean_per_cell_log_loss"],
                marker="o",
                label=line_label,
            )
            axes[1].plot(
                part["latent_uncertainty_quintile"],
                part["mean_absolute_error"],
                marker="o",
                label=line_label,
            )
        axes[0].set_ylabel("Mean per-cell log loss")
        axes[1].set_ylabel("Mean absolute error")
        for axis in axes:
            axis.set_xlabel("Latent posterior variance quintile")
            axis.set_xticks(sorted(quintiles["latent_uncertainty_quintile"].unique()))
            axis.legend(frameon=False)
    figure.suptitle(f"{label}: latent posterior variance versus prediction error")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _write_report(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Write concise evidence-based findings without ranking forecast horizons."""
    lines = [
        "# Day-5 ST-SVGP Model Diagnostics",
        "",
        "These diagnostics use raw pre-test OOF probabilities. Latent variance is described ",
        "only as latent posterior variance; it is not calibrated probability uncertainty.",
        "",
    ]
    for name, heading in (
        ("annual_1y", "Annual 1-year ST-SVGP"),
        ("five_year_5y", "Five-year ST-SVGP"),
    ):
        result = results[name]
        summaries = result["summaries"]
        fold_rows = summaries.loc[summaries["group_type"].eq("fold")]
        distance_rows = summaries.loc[summaries["group_type"].eq("distance_context")]
        hardest_fold = fold_rows.loc[fold_rows["mean_per_cell_log_loss"].idxmax()]
        hardest_context = distance_rows.loc[
            distance_rows["mean_per_cell_log_loss"].idxmax()
        ]
        lines.extend(
            [
                f"## {heading}",
                "",
                f"- Largest mean per-cell loss by relative distance context: "
                f"**{hardest_context['distance_context']}** "
                f"({hardest_context['mean_per_cell_log_loss']:.4f}).",
                f"- Most difficult OOF fold by mean per-cell loss: fold "
                f"**{int(hardest_fold['fold'])}** "
                f"({hardest_fold['mean_per_cell_log_loss']:.4f}).",
            ]
        )
        correlations = result["correlations"]
        overall = correlations.loc[correlations["fold"].eq("overall")].iloc[0]
        if bool(overall["latent_variance_available"]):
            overall_quintiles = result["quintiles"].loc[
                result["quintiles"]["fold"].eq("overall")
            ]
            lowest_variance = overall_quintiles.loc[
                overall_quintiles["latent_uncertainty_quintile"].idxmin()
            ]
            highest_variance = overall_quintiles.loc[
                overall_quintiles["latent_uncertainty_quintile"].idxmax()
            ]
            lines.extend(
                [
                    f"- Overall Spearman correlation with per-cell log loss: "
                    f"{overall['spearman_log_loss']:.3f}; with absolute error: "
                    f"{overall['spearman_absolute_error']:.3f}.",
                ]
            )
            if float(overall["spearman_log_loss"]) > 0.0:
                lines.append(
                    "- Errors tend to increase with latent posterior variance, supporting the "
                    "description that the model is uncertain where it is wrong."
                )
            else:
                lines.append(
                    f"- Confident errors are important: mean loss is "
                    f"{lowest_variance['mean_per_cell_log_loss']:.4f} in the lowest latent-"
                    f"variance quintile versus "
                    f"{highest_variance['mean_per_cell_log_loss']:.4f} in the highest. "
                    "Latent posterior variance does not behave as calibrated error uncertainty."
                )
        else:
            lines.append(
                "- Latent posterior variance is unavailable in the retained OOF artifact, so "
                "uncertainty-error claims cannot be made."
            )
        lines.extend(
            [
                "- Context-conditioned covariate SHAP is pending the rolling-state reload "
                "preflight.",
                "",
            ]
        )

    lines.extend(
        [
            "## Cross-horizon descriptive comparison",
            "",
            "The horizons predict different events and are not ranked by raw probability "
            "metrics. Comparisons are limited to failure geography, temporal heterogeneity, "
            "and the availability of uncertainty and explainability evidence.",
            "",
            "No locked annual target or locked five-year final-test row was used.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, *, preflight_only: bool) -> dict[str, Any]:
    """Validate both horizons and optionally write all OOF diagnostic outputs."""
    day5_config = load_yaml(config_path)
    output_root = Path(day5_config["output_directory"])
    prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    preflight: dict[str, Any] = {"status": "PASS", "horizons": {}}
    for spec in resolve_horizons(day5_config):
        values = prepare_horizon_diagnostics(spec, day5_config)
        prepared[spec.name] = values
        preflight["horizons"][spec.name] = values[3]

    if preflight_only:
        print(json.dumps(preflight, indent=2))
        return preflight

    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    dpi = int(day5_config["diagnostics"]["map_dpi"])
    uncertainty_bins = int(day5_config["diagnostics"]["uncertainty_bins"])
    for spec in resolve_horizons(day5_config):
        frame, distance_cuts, error_thresholds, metadata = prepared[spec.name]
        horizon_root = output_root / spec.output_subdirectory
        metrics_directory = horizon_root / "metrics"
        figures_directory = horizon_root / "figures"
        (horizon_root / "shap").mkdir(parents=True, exist_ok=True)
        metrics_directory.mkdir(parents=True, exist_ok=True)
        figures_directory.mkdir(parents=True, exist_ok=True)

        summaries = summarize_diagnostics(frame)
        correlations, quintiles = uncertainty_summaries(frame, uncertainty_bins)
        frame.to_parquet(metrics_directory / "oof_row_diagnostics.parquet", index=False)
        distance_cuts.to_csv(metrics_directory / "distance_context_cut_points.csv", index=False)
        error_thresholds.to_csv(metrics_directory / "error_level_thresholds.csv", index=False)
        summaries.to_csv(metrics_directory / "diagnostic_summaries.csv", index=False)
        correlations.to_csv(
            metrics_directory / "uncertainty_error_correlations.csv", index=False
        )
        quintiles.to_csv(metrics_directory / "latent_uncertainty_quintiles.csv", index=False)
        (metrics_directory / "diagnostic_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        extent = (
            float(frame["x_center_m"].min()),
            float(frame["x_center_m"].max()),
            float(frame["y_center_m"].min()),
            float(frame["y_center_m"].max()),
        )
        for fold, part in frame.groupby("fold", sort=True):
            _map_fold(
                part,
                label=spec.label,
                output_path=figures_directory / f"fold_{int(fold)}_validation_maps.png",
                extent=extent,
                dpi=dpi,
            )
        _plot_uncertainty_relationship(
            quintiles,
            label=spec.label,
            output_path=figures_directory / "latent_variance_vs_error.png",
            dpi=dpi,
        )
        results[spec.name] = {
            "metadata": metadata,
            "summaries": summaries,
            "correlations": correlations,
            "quintiles": quintiles,
        }

    _write_report(results, output_root / "day5_model_diagnostics.md")
    output = {
        "status": "PASS",
        "output_directory": str(output_root),
        "horizons": {name: values["metadata"] for name, values in results.items()},
    }
    print(json.dumps(output, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config, preflight_only=arguments.preflight_only)