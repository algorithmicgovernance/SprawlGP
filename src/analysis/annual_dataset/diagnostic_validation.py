"""Sample and evaluate the isolated annual built-up persistence diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import ee
except ImportError:  # Enables local unit tests without Earth Engine.
    ee = None

from src.analysis.annual_dataset.build_products import STATE_BANDS
from src.analysis.annual_dataset.build_tables import load_core_geometry, load_state_assets
from src.analysis.built_up.build_candidates import validate_source_asset
from src.analysis.final_dataset.build_tables import coordinate_bands, projection_from_grid
from src.analysis.orchestration.common import (
    find_project_root,
    initialize_earth_engine,
    load_grid_specification,
    load_yaml,
    resolve_project_path,
)

SAMPLE_TYPES = (
    "new_built_then_reversal",
    "new_built_persistent_raw",
    "stable_nonbuilt_control",
    "stable_built_control",
)
SAMPLE_TYPE_CODES = {name: index + 1 for index, name in enumerate(SAMPLE_TYPES)}
MAX_GETINFO_FEATURES = 4500
REVIEW_COLUMNS = [
    "sample_id",
    "before_year",
    "peak_year",
    "after_year",
    "longitude",
    "latitude",
    "reference_source",
    "reference_date_before",
    "reference_date_peak",
    "reference_date_after",
    "reference_label_before",
    "reference_label_peak",
    "reference_label_after",
    "reference_confidence",
    "review_note",
]
ANNOTATION_COLUMNS = REVIEW_COLUMNS[6:]
TECHNICAL_COLUMNS = [
    "sample_id",
    "episode",
    "sample_type",
    "before_year",
    "peak_year",
    "after_year",
    "cell_id",
    "row",
    "column",
    "x_center_m",
    "y_center_m",
    "longitude",
    "latitude",
    "raw_state_before",
    "raw_state_peak",
    "raw_state_after",
    "final_state_before",
    "final_state_peak",
    "final_state_after",
    "persistence_corrected_peak",
    "persistence_corrected_after",
    "valid_observation_count_before",
    "valid_observation_count_peak",
    "valid_observation_count_after",
    "ndbi_before",
    "ndbi_peak",
    "ndbi_after",
    "ibui_before",
    "ibui_peak",
    "ibui_after",
    "ndbsui_before",
    "ndbsui_peak",
    "ndbsui_after",
    "ndbi_threshold_before",
    "ndbi_threshold_peak",
    "ndbi_threshold_after",
]


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError("The Earth Engine Python API is required for diagnostic sampling.")


def deterministic_key(seed: int, *values: object) -> str:
    """Return a stable SHA-256 ordering key."""
    payload = ":".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_request_size(config: dict[str, Any], sample_type: str) -> int:
    """Return one stratum's configured oversampled candidate count."""
    requested = (
        int(config["initial_quotas_per_episode"][sample_type])
        + int(config["reserve_target_per_episode"][sample_type])
    ) * int(config["candidate_oversampling_factor"])
    if requested > MAX_GETINFO_FEATURES:
        raise ValueError(
            f"Candidate request for {sample_type} is {requested}, exceeding the "
            f"safe getInfo limit of {MAX_GETINFO_FEATURES}; batch this stratum further."
        )
    return requested


def sample_type_for_states(row: pd.Series | dict[str, Any]) -> str | None:
    """Classify one valid three-year state sequence into a diagnostic stratum."""
    raw = tuple(int(row[f"raw_state_{period}"]) for period in ("before", "peak", "after"))
    final_before = int(row["final_state_before"])
    final_peak = int(row["final_state_peak"])
    final_after = int(row["final_state_after"])
    if raw == (0, 1, 0) and final_before == 0 and final_peak == 1:
        return "new_built_then_reversal"
    if raw == (0, 1, 1) and final_before == 0 and final_peak == 1:
        return "new_built_persistent_raw"
    if raw == (0, 0, 0) and (final_before, final_peak, final_after) == (0, 0, 0):
        return "stable_nonbuilt_control"
    if raw == (1, 1, 1):
        return "stable_built_control"
    return None


def hard_greedy_thin(
    candidates: pd.DataFrame,
    target: int,
    minimum_spacing_m: float,
) -> pd.DataFrame:
    """Select candidates in their supplied order without relaxing spacing."""
    if target <= 0 or candidates.empty:
        return candidates.iloc[0:0].copy()
    selected_indices: list[Any] = []
    selected_xy: list[tuple[float, float]] = []
    minimum_squared = float(minimum_spacing_m) ** 2
    for index, row in candidates.iterrows():
        x_value = float(row["x_center_m"])
        y_value = float(row["y_center_m"])
        if all(
            (x_value - selected_x) ** 2 + (y_value - selected_y) ** 2 >= minimum_squared
            for selected_x, selected_y in selected_xy
        ):
            selected_indices.append(index)
            selected_xy.append((x_value, y_value))
        if len(selected_indices) == target:
            break
    return candidates.loc[selected_indices].copy()


def select_initial_and_reserve(
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Apply deterministic global deduplication and hard within-stratum thinning."""
    seed = int(config["seed"])
    spacing = float(config["minimum_spacing_m"])
    initial_parts: list[pd.DataFrame] = []
    reserve_parts: list[pd.DataFrame] = []
    shortages: list[dict[str, Any]] = []
    used_cell_ids: set[int] = set()
    for episode in config["episodes"]:
        for sample_type in SAMPLE_TYPES:
            initial_target = int(config["initial_quotas_per_episode"][sample_type])
            reserve_target = int(config["reserve_target_per_episode"][sample_type])
            group = candidates[
                (candidates["episode"] == episode)
                & (candidates["sample_type"] == sample_type)
                & ~candidates["cell_id"].astype(int).isin(used_cell_ids)
            ].copy()
            group["deterministic_order"] = [
                deterministic_key(seed, episode, sample_type, int(cell_id))
                for cell_id in group["cell_id"]
            ]
            group = group.sort_values("deterministic_order")
            selected = hard_greedy_thin(group, initial_target + reserve_target, spacing)
            if len(selected) < initial_target:
                raise RuntimeError(
                    f"Initial quota shortage for {episode}/{sample_type}: "
                    f"required {initial_target}, selected {len(selected)} at {spacing:g} m."
                )
            initial = selected.iloc[:initial_target].copy()
            reserve = selected.iloc[initial_target : initial_target + reserve_target].copy()
            if len(reserve) < reserve_target:
                shortages.append(
                    {
                        "episode": episode,
                        "sample_type": sample_type,
                        "requested": reserve_target,
                        "selected": len(reserve),
                    }
                )
            initial_parts.append(initial)
            reserve_parts.append(reserve)
            used_cell_ids.update(selected["cell_id"].astype(int))
    initial_frame = pd.concat(initial_parts, ignore_index=True)
    reserve_frame = pd.concat(reserve_parts, ignore_index=True)
    for frame in (initial_frame, reserve_frame):
        frame["sample_id"] = [
            f"annual_{int(peak_year)}_{int(cell_id):07d}"
            for peak_year, cell_id in zip(frame["peak_year"], frame["cell_id"], strict=True)
        ]
    return initial_frame, reserve_frame, shortages


def validate_config(config: dict[str, Any], project_root: Path) -> None:
    """Validate fixed episodes, quotas, spacing, and isolated output paths."""
    if int(config["seed"]) != 20260809:
        raise ValueError("The annual diagnostic seed must remain 20260809.")
    if float(config["minimum_spacing_m"]) <= 0:
        raise ValueError("The annual diagnostic spacing must be positive.")
    if len(config["episodes"]) != 2:
        raise ValueError("Exactly two annual diagnostic episodes are required.")
    observed_episodes = [
        (
            int(values["before_year"]),
            int(values["peak_year"]),
            int(values["after_year"]),
        )
        for values in config["episodes"].values()
    ]
    if observed_episodes != [(2015, 2016, 2017), (2019, 2020, 2021)]:
        raise ValueError("The annual diagnostic episodes have changed.")
    quotas = config["initial_quotas_per_episode"]
    expected = {
        "new_built_then_reversal": 70,
        "new_built_persistent_raw": 40,
        "stable_nonbuilt_control": 5,
        "stable_built_control": 5,
    }
    if {name: int(value) for name, value in quotas.items()} != expected:
        raise ValueError("Initial diagnostic quotas must remain 70/40/5/5.")
    if sum(int(value) for value in quotas.values()) != 120:
        raise ValueError("Initial quotas must sum to 120 per episode.")
    outputs = [resolve_project_path(value, project_root) for value in config["outputs"].values()]
    legacy = (project_root / "data/validation/manual_labels.csv").resolve()
    for path in outputs:
        resolved = path.resolve()
        if resolved == legacy or "five-year" in str(resolved).casefold() or "/v2/" in str(resolved):
            raise ValueError(f"Diagnostic output is not isolated: {resolved}")
        if "data/validation/annual_diagnostic" not in str(resolved):
            raise ValueError(f"Diagnostic data output is outside its isolated directory: {resolved}")


def load_thresholds(path: Path, years: list[int]) -> dict[tuple[int, str], float]:
    """Load frozen annual thresholds for all diagnostic years and indices."""
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {"year", "index_name", "threshold"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Annual threshold table is missing {sorted(missing)}.")
    frame = frame[frame["year"].astype(int).isin(years)]
    result = {
        (int(row.year), str(row.index_name)): float(row.threshold)
        for row in frame.itertuples(index=False)
    }
    expected = {(year, index_name) for year in years for index_name in ("ndbi", "ibui", "ndbsui")}
    missing_thresholds = expected.difference(result)
    if missing_thresholds:
        raise ValueError(f"Missing annual thresholds: {sorted(missing_thresholds)}")
    return result


def diagnostic_paths(
    config: dict[str, Any], project_root: Path
) -> dict[str, Path]:
    """Resolve configured input and output paths relative to the repository."""
    return {
        name: resolve_project_path(value, project_root)
        for section in ("inputs", "outputs")
        for name, value in config[section].items()
    }


def run_preflight(config_path: Path) -> dict[str, Any]:
    """Perform read-only validation of all annual diagnostic dependencies."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    validate_config(config, project_root)
    paths = diagnostic_paths(config, project_root)
    for name in ("grid_specification", "core_boundary", "state_manifest", "threshold_table"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing diagnostic dependency '{name}': {paths[name]}")
    grid = load_grid_specification(paths["grid_specification"])
    years = sorted(
        int(year)
        for episode in config["episodes"].values()
        for year in episode.values()
    )
    all_states = load_state_assets(paths["state_manifest"], list(range(2000, 2026)))
    states = {year: all_states[year] for year in years}
    thresholds = load_thresholds(paths["threshold_table"], years)
    require_earth_engine()
    initialize_earth_engine(str(config["earth_engine_project"]))
    for year, asset_id in states.items():
        validate_source_asset(asset_id, STATE_BANDS, grid)
        expected_suffix = f"/annual_v1/annual_dataset/states/state_{year}"
        if not asset_id.endswith(expected_suffix):
            raise ValueError(f"Unexpected annual diagnostic state asset: {asset_id}")
    load_core_geometry(paths["core_boundary"])
    result = {
        "status": "PASS",
        "state_count": len(states),
        "threshold_count": len(thresholds),
        "episode_count": len(config["episodes"]),
        "initial_sample_target": 2 * sum(config["initial_quotas_per_episode"].values()),
        "minimum_spacing_m": float(config["minimum_spacing_m"]),
        "read_only": True,
    }
    print(json.dumps(result, indent=2))
    return result


def episode_image(
    state_assets: dict[int, str],
    years: dict[str, int],
    grid: dict[str, Any],
) -> Any:
    """Build a four-stratum read-only image from finalized annual states."""
    require_earth_engine()
    images = {period: ee.Image(state_assets[int(year)]) for period, year in years.items()}
    bands = [coordinate_bands(grid)]
    for period, image in images.items():
        bands.append(
            image.select(
                [
                    "built_state_raw",
                    "built_state_final",
                    "persistence_corrected",
                    "valid_observation_count",
                    "ndbi",
                    "ibui",
                    "ndbsui",
                ],
                [
                    f"raw_state_{period}",
                    f"final_state_{period}",
                    f"persistence_corrected_{period}",
                    f"valid_observation_count_{period}",
                    f"ndbi_{period}",
                    f"ibui_{period}",
                    f"ndbsui_{period}",
                ],
            )
        )
    stack = ee.Image.cat(bands)
    valid = ee.Image.constant(1)
    for image in images.values():
        valid = valid.And(image.select("built_state_valid").eq(1))
    raw_before = stack.select("raw_state_before")
    raw_peak = stack.select("raw_state_peak")
    raw_after = stack.select("raw_state_after")
    final_before = stack.select("final_state_before")
    final_peak = stack.select("final_state_peak")
    final_after = stack.select("final_state_after")
    masks = {
        "new_built_then_reversal": raw_before.eq(0)
        .And(raw_peak.eq(1))
        .And(raw_after.eq(0))
        .And(final_before.eq(0))
        .And(final_peak.eq(1)),
        "new_built_persistent_raw": raw_before.eq(0)
        .And(raw_peak.eq(1))
        .And(raw_after.eq(1))
        .And(final_before.eq(0))
        .And(final_peak.eq(1)),
        "stable_nonbuilt_control": raw_before.eq(0)
        .And(raw_peak.eq(0))
        .And(raw_after.eq(0))
        .And(final_before.eq(0))
        .And(final_peak.eq(0))
        .And(final_after.eq(0)),
        "stable_built_control": raw_before.eq(1).And(raw_peak.eq(1)).And(raw_after.eq(1)),
    }
    class_band = ee.Image.constant(0)
    for sample_type, mask in masks.items():
        class_band = class_band.where(mask, SAMPLE_TYPE_CODES[sample_type])
    return stack.addBands(class_band.rename("sample_type_code").toUint8()).updateMask(
        valid.And(class_band.gt(0))
    )


def retrieve_candidates(
    config: dict[str, Any],
    paths: dict[str, Path],
    grid: dict[str, Any],
    state_assets: dict[int, str],
) -> pd.DataFrame:
    """Retrieve deterministic oversampled candidates without exporting assets."""
    require_earth_engine()
    core = load_core_geometry(paths["core_boundary"])
    projection = projection_from_grid(grid)
    records: list[dict[str, Any]] = []
    request_statistics: list[dict[str, Any]] = []
    for episode, episode_years in config["episodes"].items():
        years = {name.removesuffix("_year"): int(year) for name, year in episode_years.items()}
        image = episode_image(state_assets, years, grid)
        for sample_type in SAMPLE_TYPES:
            code = SAMPLE_TYPE_CODES[sample_type]
            requested = candidate_request_size(config, sample_type)
            collection = image.stratifiedSample(
                numPoints=0,
                classBand="sample_type_code",
                region=core,
                projection=projection,
                seed=int(config["seed"]),
                classValues=[code],
                classPoints=[requested],
                dropNulls=True,
                tileScale=int(config.get("tile_scale", 4)),
                geometries=False,
            )
            features = collection.getInfo().get("features", [])
            request_statistics.append(
                {
                    "episode": episode,
                    "sample_type": sample_type,
                    "requested": requested,
                    "returned": len(features),
                }
            )
            for feature in features:
                properties = dict(feature["properties"])
                properties.update(
                    {
                        "episode": episode,
                        "sample_type": sample_type,
                        **{f"{key}_year": value for key, value in years.items()},
                    }
                )
                properties.pop("sample_type_code")
                records.append(properties)
    if not records:
        raise RuntimeError("Earth Engine returned no annual diagnostic candidates.")
    frame = pd.DataFrame(records)
    integer_columns = [
        "cell_id",
        "row",
        "column",
        *(f"raw_state_{period}" for period in ("before", "peak", "after")),
        *(f"final_state_{period}" for period in ("before", "peak", "after")),
        *(f"persistence_corrected_{period}" for period in ("before", "peak", "after")),
        *(f"valid_observation_count_{period}" for period in ("before", "peak", "after")),
    ]
    for column in integer_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    expected_cell_ids = frame["row"] * int(grid["width"]) + frame["column"]
    if not frame["cell_id"].equals(expected_cell_ids.astype("int64")):
        raise ValueError("Candidate cell IDs do not match the frozen row-major identity.")
    result = frame.drop_duplicates(["episode", "cell_id"]).reset_index(drop=True)
    result.attrs["candidate_requests"] = request_statistics
    return result


def attach_thresholds(
    frame: pd.DataFrame,
    thresholds: dict[tuple[int, str], float],
) -> pd.DataFrame:
    """Attach frozen NDBI thresholds as diagnostic metadata."""
    result = frame.copy()
    for period in ("before", "peak", "after"):
        result[f"ndbi_threshold_{period}"] = [
            thresholds[(int(year), "ndbi")] for year in result[f"{period}_year"]
        ]
    return result


def build_review_frame(sample: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create a deterministically randomized reviewer file with hidden metadata."""
    review = sample[REVIEW_COLUMNS[:6]].copy()
    for column in ANNOTATION_COLUMNS:
        review[column] = ""
    review["review_order"] = [
        deterministic_key(seed, "review", sample_id) for sample_id in review["sample_id"]
    ]
    return review.sort_values("review_order").drop(columns="review_order").reset_index(drop=True)


def generate_sample(config_path: Path) -> dict[str, Any]:
    """Generate the technical sample, blinded review file, and reserve pool."""
    run_preflight(config_path)
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    paths = diagnostic_paths(config, project_root)
    grid = load_grid_specification(paths["grid_specification"])
    all_states = load_state_assets(paths["state_manifest"], list(range(2000, 2026)))
    years = sorted(int(year) for episode in config["episodes"].values() for year in episode.values())
    state_assets = {year: all_states[year] for year in years}
    thresholds = load_thresholds(paths["threshold_table"], years)
    candidates = attach_thresholds(
        retrieve_candidates(config, paths, grid, state_assets), thresholds
    )
    initial, reserve, shortages = select_initial_and_reserve(candidates, config)
    initial = initial[TECHNICAL_COLUMNS].sort_values(
        ["episode", "sample_type", "sample_id"]
    ).reset_index(drop=True)
    reserve = reserve[TECHNICAL_COLUMNS].sort_values(
        ["episode", "sample_type", "sample_id"]
    ).reset_index(drop=True)
    review = build_review_frame(initial, int(config["seed"]))
    output_directory = paths["sample"].parent
    output_directory.mkdir(parents=True, exist_ok=True)
    initial.to_csv(paths["sample"], index=False)
    reserve.to_csv(paths["reserve"], index=False)
    review.to_csv(paths["labels"], index=False)
    result = {
        "status": "PASS",
        "candidate_requests": candidates.attrs["candidate_requests"],
        "initial_count": len(initial),
        "reserve_count": len(reserve),
        "reserve_shortages": shortages,
        "initial_by_stratum": initial.groupby(["episode", "sample_type"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
        "reserve_by_stratum": reserve.groupby(["episode", "sample_type"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
    }
    print(json.dumps(result, indent=2, default=str))
    return result


def validate_review_labels(frame: pd.DataFrame) -> None:
    """Reject unexpected review columns, labels, or confidence values."""
    if frame.columns.tolist() != REVIEW_COLUMNS:
        raise ValueError("Reviewer file does not have the exact blinded schema.")
    allowed_labels = {"", "0", "1", "U"}
    for column in ("reference_label_before", "reference_label_peak", "reference_label_after"):
        observed = set(frame[column].fillna("").astype(str).str.strip())
        if not observed.issubset(allowed_labels):
            raise ValueError(f"Invalid labels in {column}: {sorted(observed - allowed_labels)}")
    confidence = set(frame["reference_confidence"].fillna("").astype(str).str.strip())
    if not confidence.issubset({"", "high", "medium", "low"}):
        raise ValueError("Reference confidence must be high, medium, or low.")


def evidence_deficiencies(joined: pd.DataFrame) -> list[dict[str, Any]]:
    """Return episode/stratum evidence counts that remain below fixed minima."""
    minima = {
        "new_built_then_reversal": 50,
        "new_built_persistent_raw": 30,
        "stable_nonbuilt_control": 3,
        "stable_built_control": 3,
    }
    labels = ["reference_label_before", "reference_label_peak", "reference_label_after"]
    decisive = joined[labels].fillna("").astype(str).apply(lambda column: column.isin({"0", "1"}))
    joined = joined.assign(_decisive=decisive.all(axis=1))
    deficiencies = []
    for episode in joined["episode"].drop_duplicates():
        for sample_type, minimum in minima.items():
            count = int(
                joined.loc[
                    (joined["episode"] == episode)
                    & (joined["sample_type"] == sample_type),
                    "_decisive",
                ].sum()
            )
            if count < minimum:
                deficiencies.append(
                    {"episode": episode, "sample_type": sample_type, "decisive": count, "minimum": minimum}
                )
    return deficiencies


def expand_sample(config_path: Path) -> dict[str, Any]:
    """Write one expansion-only blinded file for currently deficient strata."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    paths = diagnostic_paths(config, project_root)
    sample = pd.read_csv(paths["sample"], keep_default_na=False)
    labels = pd.read_csv(paths["labels"], keep_default_na=False, dtype=str)
    validate_review_labels(labels)
    deficiencies = evidence_deficiencies(sample.merge(labels, on="sample_id", validate="one_to_one"))
    if not deficiencies:
        result = {"status": "EVIDENCE_SUFFICIENT", "added": 0}
        print(json.dumps(result, indent=2))
        return result
    reserve = pd.read_csv(paths["reserve"], keep_default_na=False)
    prior_expansions = sorted(paths["sample"].parent.glob("annual_diagnostic_labels_expansion_*.csv"))
    seen_ids = set(sample["sample_id"].astype(str))
    for path in prior_expansions:
        seen_ids.update(pd.read_csv(path, usecols=["sample_id"])["sample_id"].astype(str))
    selected_parts = []
    remaining = int(config.get("expansion_batch_size", 60))
    for deficiency in deficiencies:
        if remaining <= 0:
            break
        pool = reserve[
            (reserve["episode"] == deficiency["episode"])
            & (reserve["sample_type"] == deficiency["sample_type"])
            & ~reserve["sample_id"].astype(str).isin(seen_ids)
        ]
        needed = max(0, int(deficiency["minimum"]) - int(deficiency["decisive"]))
        selected = pool.head(min(remaining, needed)).copy()
        selected_parts.append(selected)
        seen_ids.update(selected["sample_id"].astype(str))
        remaining -= len(selected)
    added = pd.concat(selected_parts, ignore_index=True) if selected_parts else reserve.iloc[0:0]
    round_number = len(prior_expansions) + 1
    output = paths["sample"].parent / f"annual_diagnostic_labels_expansion_{round_number:02d}.csv"
    build_review_frame(added, int(config["seed"]) + round_number).to_csv(output, index=False)
    result = {"status": "EVIDENCE_EXPANSION_REQUIRED", "added": len(added), "output": str(output)}
    print(json.dumps(result, indent=2))
    return result


def wilson_interval(successes: int, total: int, z_value: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0.")
    proportion = successes / total
    denominator = 1 + z_value**2 / total
    centre = (proportion + z_value**2 / (2 * total)) / denominator
    margin = z_value / denominator * math.sqrt(
        proportion * (1 - proportion) / total + z_value**2 / (4 * total**2)
    )
    return centre - margin, centre + margin


def evaluate(config_path: Path) -> dict[str, Any]:
    """Apply the evidence gate; full reporting begins only after independent review."""
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    paths = diagnostic_paths(config, project_root)
    sample = pd.read_csv(paths["sample"], keep_default_na=False)
    label_paths = [paths["labels"], *sorted(paths["sample"].parent.glob("annual_diagnostic_labels_expansion_*.csv"))]
    labels = pd.concat(
        [pd.read_csv(path, keep_default_na=False, dtype=str) for path in label_paths], ignore_index=True
    )
    validate_review_labels(labels)
    joined = sample.merge(labels, on="sample_id", validate="one_to_one")
    deficiencies = evidence_deficiencies(joined)
    result = {
        "status": "EVIDENCE_EXPANSION_REQUIRED" if deficiencies else "EVIDENCE_SUFFICIENT",
        "deficiencies": deficiencies,
    }
    print(json.dumps(result, indent=2))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the isolated annual diagnostic command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--sample", action="store_true")
    actions.add_argument("--expand", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected read-only diagnostic action."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    if arguments.preflight:
        run_preflight(config_path)
    elif arguments.sample:
        generate_sample(config_path)
    elif arguments.expand:
        expand_sample(config_path)
    else:
        evaluate(config_path)


if __name__ == "__main__":
    main()