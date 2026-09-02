"""Final context-conditioned covariate SHAP analysis for retained ST-SVGP states.

Only the nine observed model predictors are perturbed. Projected coordinates
and validation time remain fixed at one real representative context, so the
results describe model attribution under that context and are not causal effects.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from src.models.evaluation.st_svgp_day5_diagnostics import (
    DEFAULT_CONFIG_PATH,
    HorizonSpec,
    load_yaml,
    prepare_horizon_diagnostics,
    resolve_horizons,
)
from src.models.evaluation.st_svgp_day5_reconstruction import (
    METADATA_FILENAME,
    STATE_ROOT,
    LoadedFoldState,
    load_fold_state,
    state_directory,
)
from src.models.train_st_svgp import TimeData, predict_frame, prepare_time_data, time_step

OUTPUT_ROOT = Path("reports/modeling/st_svgp_explainability")
PRIMARY_CONTEXTS = ("near_built", "peripheral")
BACKGROUND_SIZE = 40
ROWS_PER_CONTEXT = 10
NSAMPLES = 128
RANDOM_STATE = 20260809
MAX_PATHOLOGICAL_ADDITIVITY_ERROR = 0.05
EXPECTED_FEATURES = {
    "annual_1y": (
        "ndbi_t",
        "savi_t",
        "log_distance_to_built_m_t",
        "built_fraction_11x11_t",
        "recent_local_growth_1y_t",
        "elevation_m",
        "slope_degrees",
        "log_population_density_t",
        "built_fraction_x_recent_growth_t",
    ),
    "five_year_5y": (
        "ndbi_t",
        "savi_t",
        "log_distance_to_built_m_t",
        "built_fraction_11x11_t",
        "recent_local_growth_5y_t",
        "elevation_m",
        "slope_degrees",
        "log_population_density_t",
        "built_fraction_x_recent_growth_t",
    ),
}


@dataclass(frozen=True)
class ContextConditionedPredictor:
    """Predict with one reloaded fold while projected space and time stay fixed."""

    state: LoadedFoldState
    context_x_m: float
    context_y_m: float
    context_origin: int

    def __call__(self, observed_features: np.ndarray) -> np.ndarray:
        features = np.asarray(observed_features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        expected = len(self.state.config["linear_predictors"])
        if features.ndim != 2 or features.shape[1] != expected:
            raise ValueError(f"Expected a matrix with {expected} observed features.")

        preprocessing = self.state.preprocessing
        scaled = (features - preprocessing.feature_mean) / preprocessing.feature_scale
        coordinate = (
            np.array([self.context_x_m, self.context_y_m], dtype=np.float64) / 1000.0
            - preprocessing.coordinate_center_km
        )
        validation_data = TimeData(
            origin=int(self.context_origin),
            time_step=time_step(int(self.context_origin), self.state.config),
            features=scaled,
            coordinates_km=np.repeat(coordinate[None, :], len(features), axis=0),
            targets=np.zeros(len(features), dtype=np.float64),
            row_index=np.arange(len(features)),
        )
        probabilities, _ = predict_frame(
            model=self.state.model,
            posterior=self.state.posterior,
            validation_data=validation_data,
            last_training_step=self.state.last_training_step,
            config=self.state.config,
            dtype=self.state.dtype,
        )
        return np.asarray(probabilities, dtype=np.float64)


def _validate_probabilities(probabilities: np.ndarray, label: str) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{label} probabilities are not finite.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{label} probabilities are outside [0, 1].")


def _validate_pretest_rows(frame: pd.DataFrame, spec: HorizonSpec) -> None:
    target_year = pd.to_numeric(frame["target_year"], errors="raise").astype(int)
    origin = pd.to_numeric(frame["forecast_origin"], errors="raise").astype(int)
    if spec.maximum_target_year is not None and target_year.gt(
        spec.maximum_target_year
    ).any():
        raise ValueError(f"{spec.name} includes a locked target year.")
    if spec.locked_forecast_origins and origin.isin(spec.locked_forecast_origins).any():
        raise ValueError(f"{spec.name} includes a locked forecast origin.")


def _nearest_real_rows(
    frame: pd.DataFrame,
    features: list[str],
    *,
    size: int,
    random_state: int,
) -> pd.DataFrame:
    """Return unique real rows nearest deterministic standardized KMeans centers."""
    ordered = frame.sort_values(["cell_id", "forecast_origin"]).reset_index(drop=True)
    if ordered.empty:
        raise ValueError("No eligible real observations were available.")
    clusters = min(int(size), len(ordered))
    values = ordered[features].to_numpy(dtype=np.float64)
    scale = values.std(axis=0, ddof=0)
    scale[scale <= 0.0] = 1.0
    standardized = (values - values.mean(axis=0)) / scale
    centers = KMeans(
        n_clusters=clusters,
        random_state=random_state,
        n_init=10,
    ).fit(standardized).cluster_centers_
    selected: list[int] = []
    used: set[int] = set()
    for center in centers:
        distances = np.square(standardized - center).sum(axis=1)
        for candidate in np.argsort(distances, kind="mergesort"):
            index = int(candidate)
            if index not in used:
                selected.append(index)
                used.add(index)
                break
    return ordered.iloc[selected].reset_index(drop=True)


def select_background(
    frame: pd.DataFrame,
    features: list[str],
    *,
    size: int = BACKGROUND_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Select the shared horizon background from real pre-test observations."""
    return _nearest_real_rows(
        frame,
        features,
        size=size,
        random_state=random_state,
    )


def select_contexts_and_explanations(
    frame: pd.DataFrame,
    features: list[str],
    *,
    horizon: str,
    rows_per_context: int = ROWS_PER_CONTEXT,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose six real contexts and up to ten real explanations per context."""
    contexts: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []
    eligible = frame.loc[frame["distance_context"].isin(PRIMARY_CONTEXTS)].copy()
    for group_number, ((fold, distance_context), part) in enumerate(
        eligible.groupby(["fold", "distance_context"], observed=True, sort=True),
        start=1,
    ):
        coordinates = part[["x_center_m", "y_center_m"]].to_numpy(dtype=np.float64)
        centroid = coordinates.mean(axis=0)
        context_row = part.iloc[int(np.argmin(np.square(coordinates - centroid).sum(axis=1)))]
        context_id = f"{horizon}_fold_{int(fold)}_{distance_context}"
        contexts.append(
            {
                "horizon": horizon,
                "fold": int(fold),
                "forecast_origin": int(context_row["forecast_origin"]),
                "target_year": int(context_row["target_year"]),
                "context_id": context_id,
                "distance_context": str(distance_context),
                "context_cell_id": context_row["cell_id"],
                "context_x_m": float(context_row["x_center_m"]),
                "context_y_m": float(context_row["y_center_m"]),
                "selection_rule": "real observation nearest coordinate centroid",
            }
        )
        selected = _nearest_real_rows(
            part,
            features,
            size=rows_per_context,
            random_state=random_state + group_number,
        ).copy()
        selected["context_id"] = context_id
        selected["distance_context"] = str(distance_context)
        samples.append(selected)

    context_frame = pd.DataFrame(contexts).sort_values(
        ["horizon", "fold", "distance_context"]
    )
    sample_frame = pd.concat(samples, ignore_index=True)
    if len(context_frame) != 6:
        raise ValueError(f"{horizon} requires exactly six fold-distance contexts.")
    counts = sample_frame.groupby("context_id").size()
    if counts.gt(rows_per_context).any():
        raise ValueError("An explained context exceeds its row cap.")
    if sample_frame.duplicated(["context_id", "cell_id", "forecast_origin"]).any():
        raise ValueError("An explained context contains duplicate observations.")
    return context_frame.reset_index(drop=True), sample_frame


def sanity_check_state(
    spec: HorizonSpec,
    frame: pd.DataFrame,
    *,
    fold: int,
    random_state: int = RANDOM_STATE,
) -> tuple[LoadedFoldState, dict[str, Any]]:
    """Load one state and reproduce a small deterministic retained OOF sample."""
    directory = state_directory(spec.name, fold, root=STATE_ROOT)
    state = load_fold_state(directory, expected_config_path=spec.model_config_path)
    features = [str(value) for value in state.config["linear_predictors"]]
    if tuple(features) != EXPECTED_FEATURES[spec.name]:
        raise ValueError(f"{spec.name} Fold {fold} feature order is not the retained order.")
    if state.metadata.get("feature_names") != features:
        raise ValueError(f"{spec.name} Fold {fold} metadata feature order differs.")
    if int(state.metadata["fold"]) != fold or state.metadata["horizon"] != spec.name:
        raise ValueError(f"{spec.name} Fold {fold} persisted identity differs.")

    fold_rows = frame.loc[frame["fold"].eq(fold)].copy()
    _validate_pretest_rows(fold_rows, spec)
    sample = fold_rows.sort_values(["cell_id", "forecast_origin"]).sample(
        n=min(5, len(fold_rows)),
        random_state=random_state + fold,
    )
    target = str(state.config["dataset"]["target"])
    sample[target] = sample["y_true"]
    validation_origin = int(state.metadata["validation_origin"])
    validation_data = prepare_time_data(
        sample,
        origins=[validation_origin],
        preprocessing=state.preprocessing,
        config=state.config,
    )[0]
    actual, _ = predict_frame(
        model=state.model,
        posterior=state.posterior,
        validation_data=validation_data,
        last_training_step=state.last_training_step,
        config=state.config,
        dtype=state.dtype,
    )
    actual = np.asarray(actual, dtype=np.float64)
    expected = sample["probability_raw"].to_numpy(dtype=np.float64)
    _validate_probabilities(actual, f"{spec.name} Fold {fold}")
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-8)
    difference = np.abs(actual - expected)
    return state, {
        "horizon": spec.name,
        "fold": fold,
        "status": "PASS",
        "state_directory": str(directory),
        "metadata_path": str(directory / METADATA_FILENAME),
        "feature_order_matches": True,
        "prediction_callable_works": True,
        "sample_rows": len(sample),
        "maximum_absolute_oof_difference": float(difference.max(initial=0.0)),
        "probabilities_valid": True,
        "locked_rows_used": 0,
    }


def summarize_importance(shap_values: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean absolute SHAP overall, by fold, and by distance context."""
    tables: list[pd.DataFrame] = []
    designs: tuple[tuple[str, list[str]], ...] = (
        ("overall", []),
        ("fold", ["fold"]),
        ("distance_context", ["distance_context"]),
        ("fold_x_distance_context", ["fold", "distance_context"]),
    )
    for scope, groups in designs:
        if groups:
            summary = (
                shap_values.groupby([*groups, "feature"], sort=True, observed=True)[
                    "shap_value"
                ]
                .apply(lambda values: float(np.abs(values).mean()))
                .rename("mean_absolute_shap")
                .reset_index()
            )
            summary["rank"] = summary.groupby(groups)["mean_absolute_shap"].rank(
                method="first", ascending=False
            )
        else:
            summary = (
                shap_values.groupby("feature", sort=True)["shap_value"]
                .apply(lambda values: float(np.abs(values).mean()))
                .rename("mean_absolute_shap")
                .reset_index()
            )
            summary["rank"] = summary["mean_absolute_shap"].rank(
                method="first", ascending=False
            )
        summary.insert(0, "scope", scope)
        if "fold" not in summary:
            summary["fold"] = pd.NA
        if "distance_context" not in summary:
            summary["distance_context"] = pd.NA
        tables.append(summary)
    result = pd.concat(tables, ignore_index=True)
    result["rank"] = result["rank"].astype(int)
    return result.loc[
        :, ["scope", "fold", "distance_context", "feature", "mean_absolute_shap", "rank"]
    ]


def summarize_direction(shap_values: pd.DataFrame) -> pd.DataFrame:
    """Describe the association between observed feature values and SHAP values."""
    rows: list[dict[str, Any]] = []
    for feature, part in shap_values.groupby("feature", sort=True):
        median = float(part["feature_value"].median())
        low = part.loc[part["feature_value"].le(median), "shap_value"]
        high = part.loc[part["feature_value"].gt(median), "shap_value"]
        correlation = float(
            spearmanr(part["feature_value"], part["shap_value"]).statistic
        )
        low_mean = float(low.mean())
        high_mean = float(high.mean()) if not high.empty else float("nan")
        if np.isfinite(correlation) and correlation >= 0.2 and high_mean > low_mean:
            direction = "higher values generally push probability upward"
        elif np.isfinite(correlation) and correlation <= -0.2 and high_mean < low_mean:
            direction = "higher values generally push probability downward"
        else:
            direction = "mixed / context dependent"
        rows.append(
            {
                "feature": feature,
                "mean_shap_low_feature_values": low_mean,
                "mean_shap_high_feature_values": high_mean,
                "spearman_feature_value_vs_shap": correlation,
                "direction_summary": direction,
            }
        )
    return pd.DataFrame(rows)


def validate_shap_result(
    values: np.ndarray,
    base_value: float,
    predicted: np.ndarray,
) -> np.ndarray:
    """Validate finite SHAP output and return absolute local additivity errors."""
    shap_array = np.asarray(values, dtype=np.float64)
    probability = np.asarray(predicted, dtype=np.float64)
    if not np.isfinite(shap_array).all() or not np.isfinite(base_value):
        raise ValueError("SHAP values and base values must be finite.")
    _validate_probabilities(probability, "SHAP")
    errors = np.abs(float(base_value) + shap_array.sum(axis=1) - probability)
    if not np.isfinite(errors).all():
        raise ValueError("SHAP additivity errors must be finite.")
    return errors


def _kernel_shap_values(
    predictor: ContextConditionedPredictor,
    background: np.ndarray,
    explained: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    import shap

    explainer = shap.KernelExplainer(predictor, background, link="identity")
    values = np.asarray(explainer.shap_values(explained, nsamples=NSAMPLES, silent=True))
    if values.ndim != 2 or values.shape != explained.shape:
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}.")
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    predicted = predictor(explained)
    return values.astype(np.float64), base_value, predicted


def _plot_summary(
    values: np.ndarray,
    feature_values: pd.DataFrame,
    output_path: Path,
) -> None:
    import shap

    plt.figure(figsize=(10, 6.5))
    shap.summary_plot(
        values,
        feature_values,
        feature_names=list(feature_values.columns),
        max_display=len(feature_values.columns),
        show=False,
        plot_size=None,
    )
    plt.title("Context-conditioned ST-SVGP covariate attribution")
    plt.xlabel("SHAP value (change in predicted probability)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_context_comparison(
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    overall = importance.loc[importance["scope"].eq("overall")].sort_values("rank")
    order = overall["feature"].tolist()[::-1]
    context = importance.loc[importance["scope"].eq("distance_context")]
    pivot = context.pivot(
        index="feature", columns="distance_context", values="mean_absolute_shap"
    ).reindex(order)
    positions = np.arange(len(pivot))
    height = 0.36
    figure, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    axis.barh(
        positions - height / 2,
        pivot["near_built"],
        height,
        label="Near built",
        color="#287271",
    )
    axis.barh(
        positions + height / 2,
        pivot["peripheral"],
        height,
        label="Peripheral",
        color="#d96c4a",
    )
    axis.set_yticks(positions, pivot.index)
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title("Attribution strength by distance context")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _run_horizon(
    spec: HorizonSpec,
    frame: pd.DataFrame,
    states: dict[int, LoadedFoldState],
    output_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    features = list(EXPECTED_FEATURES[spec.name])
    _validate_pretest_rows(frame, spec)
    background = select_background(frame, features)
    if len(background) != BACKGROUND_SIZE:
        raise ValueError(f"{spec.name} requires exactly {BACKGROUND_SIZE} background rows.")
    contexts, explained = select_contexts_and_explanations(
        frame,
        features,
        horizon=spec.name,
    )

    long_tables: list[pd.DataFrame] = []
    plot_values: list[np.ndarray] = []
    plot_features: list[pd.DataFrame] = []
    additivity_errors: list[np.ndarray] = []
    for context in contexts.itertuples(index=False):
        rows = explained.loc[explained["context_id"].eq(context.context_id)].copy()
        predictor = ContextConditionedPredictor(
            state=states[int(context.fold)],
            context_x_m=float(context.context_x_m),
            context_y_m=float(context.context_y_m),
            context_origin=int(context.forecast_origin),
        )
        feature_frame = rows.loc[:, features].astype(float)
        values, base_value, predicted = _kernel_shap_values(
            predictor,
            background.loc[:, features].to_numpy(dtype=np.float64),
            feature_frame.to_numpy(dtype=np.float64),
        )
        errors = validate_shap_result(values, base_value, predicted)
        additivity_errors.append(errors)
        plot_values.append(values)
        plot_features.append(feature_frame)
        row_count = len(rows)
        feature_count = len(features)
        long_tables.append(
            pd.DataFrame(
                {
                    "horizon": spec.name,
                    "fold": int(context.fold),
                    "forecast_origin": np.repeat(
                        rows["forecast_origin"].to_numpy(), feature_count
                    ),
                    "target_year": np.repeat(rows["target_year"].to_numpy(), feature_count),
                    "context_id": context.context_id,
                    "distance_context": context.distance_context,
                    "cell_id": np.repeat(rows["cell_id"].to_numpy(), feature_count),
                    "feature": np.tile(features, row_count),
                    "feature_value": feature_frame.to_numpy().reshape(-1),
                    "shap_value": values.reshape(-1),
                    "base_value": base_value,
                    "predicted_probability": np.repeat(predicted, feature_count),
                }
            )
        )

    all_errors = np.concatenate(additivity_errors)
    maximum_error = float(all_errors.max(initial=0.0))
    if maximum_error > MAX_PATHOLOGICAL_ADDITIVITY_ERROR:
        raise RuntimeError(
            f"{spec.name} SHAP additivity error is pathological: {maximum_error:.6f}."
        )

    horizon_root = output_root / spec.name
    horizon_root.mkdir(parents=True, exist_ok=True)
    shap_values = pd.concat(long_tables, ignore_index=True)
    importance = summarize_importance(shap_values)
    direction = summarize_direction(shap_values)
    shap_values.to_parquet(horizon_root / "shap_values.parquet", index=False)
    importance.to_csv(horizon_root / "shap_feature_importance.csv", index=False)
    direction.to_csv(horizon_root / "shap_feature_direction.csv", index=False)
    _plot_summary(
        np.vstack(plot_values),
        pd.concat(plot_features, ignore_index=True),
        horizon_root / "shap_summary.png",
    )
    _plot_context_comparison(
        importance,
        horizon_root / "shap_context_comparison.png",
    )
    return {
        "horizon": spec.name,
        "runtime_seconds": float(time.perf_counter() - started),
        "contexts": len(contexts),
        "background_rows": len(background),
        "explained_rows": int(shap_values[["context_id", "cell_id"]].drop_duplicates().shape[0]),
        "nsamples": NSAMPLES,
        "mean_absolute_additivity_error": float(all_errors.mean()),
        "maximum_absolute_additivity_error": maximum_error,
        "importance": importance,
        "direction": direction,
        "contexts_frame": contexts,
    }


def _context_difference_text(importance: pd.DataFrame, count: int = 3) -> str:
    context = importance.loc[importance["scope"].eq("distance_context")]
    pivot = context.pivot(
        index="feature", columns="distance_context", values="mean_absolute_shap"
    )
    difference = (pivot["near_built"] - pivot["peripheral"]).sort_values(
        key=np.abs, ascending=False
    )
    descriptions = []
    for feature, value in difference.head(count).items():
        stronger = "near built" if value > 0 else "peripheral"
        descriptions.append(f"{feature} ({stronger})")
    return ", ".join(descriptions)


def _write_report(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# ST-SVGP Explainability Summary",
        "",
        "These SHAP results describe model attribution under fixed representative "
        "space-time contexts and are not causal effects.",
        "",
    ]
    for horizon, heading in (
        ("annual_1y", "Annual 1y"),
        ("five_year_5y", "Five-year 5y"),
    ):
        result = results[horizon]
        importance = result["importance"]
        overall = importance.loc[importance["scope"].eq("overall")].sort_values("rank")
        top_three = overall.head(3)["feature"].tolist()
        fold_rows = importance.loc[importance["scope"].eq("fold")]
        fold_descriptions = []
        for fold, part in fold_rows.groupby("fold", sort=True):
            names = part.sort_values("rank").head(3)["feature"].tolist()
            fold_descriptions.append(f"fold {int(fold)}: {', '.join(names)}")
        direction = result["direction"].set_index("feature")["direction_summary"]
        direction_text = "; ".join(
            f"{feature}: {direction.loc[feature]}" for feature in top_three
        )
        fold_rankings_change = len(
            {
                tuple(part.sort_values("rank").head(3)["feature"])
                for _, part in fold_rows.groupby("fold", sort=True)
            }
        ) > 1
        lines.extend(
            [
                f"## {heading}",
                "",
                f"- Top three overall: {', '.join(top_three)}.",
                f"- Rankings {'change' if fold_rankings_change else 'are stable'} across folds; "
                + "; ".join(fold_descriptions)
                + ".",
                "- Strongest near-built versus peripheral differences: "
                + _context_difference_text(importance)
                + ".",
                f"- Direction: {direction_text}.",
                f"- Additivity error: mean {result['mean_absolute_additivity_error']:.6g}, "
                f"maximum {result['maximum_absolute_additivity_error']:.6g}.",
                "",
            ]
        )

    annual_top = set(
        results["annual_1y"]["importance"]
        .loc[lambda table: table["scope"].eq("overall")]
        .nsmallest(3, "rank")["feature"]
    )
    five_top = set(
        results["five_year_5y"]["importance"]
        .loc[lambda table: table["scope"].eq("overall")]
        .nsmallest(3, "rank")["feature"]
    )
    shared = sorted(annual_top & five_top)
    lines.extend(
        [
            "## Cross-horizon",
            "",
            "The horizons share "
            + (", ".join(shared) if shared else "no features")
            + " among their three strongest attributions. Raw SHAP magnitudes are not "
            "compared because the horizons predict different targets.",
            "",
            "No model was trained or tuned, no reconstruction was rerun, no final fit was "
            "run, and no locked test row was used.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    """Run the single final SHAP analysis after all six states pass sanity checks."""
    config = load_yaml(config_path)
    specs = resolve_horizons(config)
    if tuple(spec.name for spec in specs) != ("annual_1y", "five_year_5y"):
        raise ValueError("Explainability requires exactly annual_1y and five_year_5y.")

    prepared: dict[str, pd.DataFrame] = {}
    loaded: dict[str, dict[int, LoadedFoldState]] = {}
    sanity_rows: list[dict[str, Any]] = []
    for spec in specs:
        frame, _, _, _ = prepare_horizon_diagnostics(spec, config)
        _validate_pretest_rows(frame, spec)
        prepared[spec.name] = frame
        loaded[spec.name] = {}
        for fold in (1, 2, 3):
            state, status = sanity_check_state(spec, frame, fold=fold)
            loaded[spec.name][fold] = state
            sanity_rows.append(status)

    output_root.mkdir(parents=True, exist_ok=True)
    sanity = {
        "status": "PASS",
        "states_checked": 6,
        "training_performed": False,
        "reconstruction_performed": False,
        "final_fit_performed": False,
        "locked_rows_used": 0,
        "states": sanity_rows,
    }
    (output_root / "six_state_sanity.json").write_text(
        json.dumps(sanity, indent=2) + "\n", encoding="utf-8"
    )

    results: dict[str, dict[str, Any]] = {}
    context_tables: list[pd.DataFrame] = []
    for spec in specs:
        result = _run_horizon(
            spec,
            prepared[spec.name],
            loaded[spec.name],
            output_root,
        )
        results[spec.name] = result
        context_tables.append(result["contexts_frame"])
    pd.concat(context_tables, ignore_index=True).to_csv(
        output_root / "context_manifest.csv", index=False
    )
    _write_report(results, output_root / "st_svgp_explainability_summary.md")
    run_summary = {
        "status": "PASS",
        "six_state_sanity": "PASS",
        "locked_rows_used": 0,
        "horizons": {
            name: {
                key: value
                for key, value in result.items()
                if key
                in {
                    "runtime_seconds",
                    "contexts",
                    "background_rows",
                    "explained_rows",
                    "nsamples",
                    "mean_absolute_additivity_error",
                    "maximum_absolute_additivity_error",
                }
            }
            for name, result in results.items()
        },
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))
    return run_summary


if __name__ == "__main__":
    run()