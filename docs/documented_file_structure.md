# SprawlGP — Yaoundé Urban Expansion Dataset

This repository builds a reproducible 30 m urban-expansion dataset for Yaoundé,
Cameroon, from Landsat, GHSL, SRTM and OpenStreetMap. The active release is
`yaounde_urban_expansion_30m_v2`, version `2.0.0-provisional`.

## Setup

```bash
python -m venv .venvt
source .venvt/bin/activate
pip install -e .
earthengine authenticate
```

All commands are run from the repository root. Configuration is stored in
`configs/`; notebooks are optional and never modify the pipeline.

## Reproduction overview

Earth Engine jobs are asynchronous and the cell-time CSV exports are downloaded
from Google Drive. A safe clean run is therefore phased rather than hidden in
one uninterrupted command.

```bash
# Step 1
make build-grid

# Step 2
python -m src.analysis.landsat.build_catalog \
--config configs/landsat_catalog.yaml

# Step 3
make orchestration-preflight
make orchestration-submit
make orchestration-status # repeat until completed
make orchestration-finalize
make test-orchestration

# Step 4
make built-up-preflight
make built-up-submit
make built-up-status # repeat until completed
make built-up-finalize
make test-built-up

# Step 5
make validation-preflight
make validation-samples
# Complete data/validation/manual_labels.csv before evaluate/freeze.
make validation-evaluate
make validation-freeze
make test-validation

# Step 6
make dataset-preflight
make dataset-submit-rasters
make dataset-status-rasters # repeat until completed
make dataset-finalize-rasters
make dataset-submit-tables
make dataset-status-tables # repeat until completed
# Download five CSV files into data/staging/cell_time_exports/.
make dataset-assemble
make dataset-finalize
make test-dataset

# Step 7
make handover
```

The exact clean-run procedure and stop conditions are documented in
`docs/reproduction_and_handover.md`.

## Repository inventory

### Root

| Path             | Role                                                               | Tracking |
| ---------------- | ------------------------------------------------------------------ | -------- |
| `Makefile`       | Stable command-line entry points for Days 1–7                      | Git      |
| `README.md`      | Project overview, file inventory and reproduction entry point      | Git      |
| `pyproject.toml` | Python package metadata and dependencies                           | Git      |
| `checksums.txt`  | Top-level source or release checksum record                        | Git      |
| `.gitignore`     | Excludes caches, checkpoints, staging data and large local outputs | Git      |

### Configuration

| File                                                                                                        | Role                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `configs/study_area.yaml`                                                                                   | Boundary selection, CRS, context buffer and frozen 30 m grid                                                              |
| `configs/landsat_catalog.yaml`                                                                              | Epochs, sensor policy, QA rules and exact per-epoch window selection                                                      |
| `configs/orchestrate_sources.yaml`                                                                          | Composite, GHSL, terrain, OSM and Earth Engine export settings                                                            |
| `configs/built_up_candidates.yaml`                                                                          | Spectral indices, candidate methods, Otsu settings and asset roots                                                        |
| `configs/mapping_validation.yaml`                                                                           | Stratified sampling and manual method-comparison protocol                                                                 |
| `configs/final_dataset.yaml`                                                                                | NDBI state, transition, table and release configuration                                                                   |
| `configs/modeling/logistic_regression_experiment.yaml`                                                      | Retained Logistic Regression baseline configuration used for chronological rolling validation                             |
| `configs/modeling/logistic_regression/experiment_v1.yaml`                                                   | Historical Logistic Regression experiment retained for provenance                                                         |
| `configs/modeling/svgp_experiment.yaml`                                                                     | Retained Adam SVGP reference configuration (`M_s=64`) and shared rolling-validation contract                              |
| `configs/modeling/svgp/experiment_v1.yaml`                                                                  | Historical SVGP Natural-Gradient experiment configuration; retained as an experimental branch, not the primary comparator |
| `configs/modeling/st_svgp.yaml`                                                                             | Canonical promoted ST-SVGP candidate: trainable temporal Matérn-3/2 lengthscale initialised at 1.5 five-year steps        |
| `configs/modeling/st_svgp/experiment_free_init_1p0.yaml`                                                    | Retained ST-SVGP diagnostic with trainable temporal lengthscale initialised at 1.0                                        |
| `configs/modeling/st_svgp/temporal_lengthscale_fixed_1p5.yaml`                                              | Retained ST-SVGP diagnostic with temporal lengthscale fixed at 1.5                                                        |
| `configs/modeling/st_svgp/temporal_kernel_validation.yaml`                                                  | Synthetic validation contract for the temporal Matérn-3/2 state-space representation                                      |
| `configs/modeling/st_svgp/filter_smoother_validation.yaml`                                                  | Synthetic Kalman-filter / RTS-smoother validation configuration                                                           |
| `configs/modeling/st_svgp/cvi_natgrad_validation.yaml`                                                      | Synthetic Bernoulli-probit CVI / Natural-Gradient validation configuration                                                |
| `configs/annual/annual_diagnostic.yaml`                                                                     | Targeted independent annual-label diagnostic protocol, including blind sample/reserve generation and evaluation contract  |
| `configs/modeling/day5_st_svgp_diagnostics.yaml`                                                            | Historical diagnostic/reconstruction contract for annual and five-year OOF failure analysis and state reconstruction      |
| `configs/modeling/st_svgp_improvements/baseline_diagnostics.yaml`                                           | Read-only convergence-diagnostic contract for frozen baseline histories                                                   |
| `configs/modeling/st_svgp_improvements/annual/optimizer/`                                                   | Isolated optimizer experiments: clipping, Adam LR, combined optimizer settings and Natural-Gradient gamma                 |
| `configs/modeling/st_svgp_improvements/annual/spatial/`                                                     | Spatial-initialization and weak spatial regularization experiments                                                        |
| `configs/modeling/st_svgp_improvements/annual/temporal/`                                                    | Annual temporal-lengthscale-prior and explicit calendar-time-trend experiments                                            |
| `configs/modeling/st_svgp_improvements/annual/inducing/`                                                    | Trainable-inducing-location experiment built on the time-trend candidate                                                  |
| `configs/modeling/st_svgp_improvements/annual/combined/time_trend_early_stopping.yaml`                      | `EARLY-STOP-A01`: PRIMARY annual development configuration with pre-specified training-stability stopping                 |
| `configs/modeling/st_svgp_improvements/annual/combined/time_trend_early_stopping_clip_100k.yaml`            | Clip-only sensitivity of the PRIMARY (`10 -> 100000`)                                                                     |
| `configs/modeling/st_svgp_improvements/annual/combined/time_trend_stabilized_optimizer_early_stopping.yaml` | Combined optimizer/capacity evidence experiment (`COMB-A02`)                                                              |
| `configs/modeling/st_svgp_improvements/five_year/temporal/`                                                 | Five-year temporal freeze/prior diagnostics retained only as evidence                                                     |

### Data directories

| Path                                                              | Role                                                                | Tracking                               |
| ----------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------- |
| `data/raw/boundaries/`                                            | Downloaded HDX and GeoBoundaries source files and source metadata   | Usually local/Git-LFS                  |
| `data/raw/osm/`                                                   | Timestamped Cameroon OSM extract, checksum and metadata             | Usually local                          |
| `data/processed/boundaries/`                                      | Yaoundé core, context buffer, convex hull and ADM2 reference        | Git or release package                 |
| `data/processed/grid/`                                            | Frozen cell table, masks and 30 m template raster                   | Usually local/release                  |
| `data/processed/osm/yaounde_current_osm.gpkg`                     | Clean context-area roads, railways and buildings                    | Usually local                          |
| `data/staging/cell_time_exports/`                                 | Five downloaded Earth Engine CSV table exports                      | Local only                             |
| `data/staging/final_dataset/tracking_stack.tif`                   | Temporary local tracking raster used for metrics                    | Local only                             |
| `data/validation/validation_samples.gpkg`                         | Stratified Day 5 sample locations and predictions                   | Git/release                            |
| `data/validation/manual_labels.csv`                               | Human labels; currently incomplete                                  | Git when completed                     |
| `data/validation/annual_diagnostic/annual_diagnostic_sample.csv`  | Frozen diagnostic sample selected for independent review            | Git                                    |
| `data/validation/annual_diagnostic/annual_diagnostic_reserve.csv` | Evidence-driven reserve sample                                      | Git                                    |
| `data/validation/annual_diagnostic/annual_diagnostic_labels.csv`  | Blank/current review contract; do not replace with inferred labels  | Git while protocol is active           |
| `data/final/yaounde_urban_expansion_30m_v2/`                      | Final Parquet, demand, metrics, schema, dictionary and dataset card | Release; large Parquet may be external |

### Boundary and grid metadata

| File                                             | Role                                                   |
| ------------------------------------------------ | ------------------------------------------------------ |
| `data/metadata/boundary_report.json`             | Core/context areas, selected units and boundary checks |
| `data/metadata/grid_specification.json`          | Frozen CRS, extent, dimensions, transform and checksum |
| `data/metadata/adm2_adm3_comparison.json`        | ADM2/ADM3 boundary comparison                          |
| `data/metadata/source_boundary_inspection*.json` | Raw boundary field, name and geometry inspection       |

### Landsat metadata

| File/pattern                            | Role                                                            |
| --------------------------------------- | --------------------------------------------------------------- |
| `scene_manifest_all.csv` / `.parquet`   | Complete diagnostic scene catalogue                             |
| `monthly_availability.csv` / `.parquet` | Monthly scene and valid-coverage summaries                      |
| `candidate_window_details.csv`          | Exact per-epoch, per-window observation metrics                 |
| `candidate_window_summary.csv`          | Compact candidate-window comparison                             |
| `epoch_quality_summary.csv`             | Selected window, sensors, depth and coverage for each epoch     |
| `selected_scene_manifest.csv`           | Exact frozen scene IDs consumed by Day 3                        |
| `old_vs_new_selected_manifest.csv`      | Audit of the corrected catalogue against the previous selection |
| `compositing_protocol.yaml`             | Human-readable frozen per-epoch protocol                        |
| `catalog_version.json`                  | Day 2 stable signature and checksums                            |
| `_checkpoints/`                         | Resumable catalogue state; never part of a release              |

### Orchestration metadata

| File                             | Role                                                          |
| -------------------------------- | ------------------------------------------------------------- |
| `auxiliary_source_manifest.csv`  | GHSL, terrain and OSM source inventory                        |
| `dependency_manifest.csv`        | Input paths and hashes consumed by Day 3                      |
| `earth_engine_tasks.csv`         | Submitted composite/count/terrain task IDs and states         |
| `landsat_composite_manifest.csv` | Final composite and count asset IDs                           |
| `landsat_epoch_quality.csv`      | Day 3 support and observation-depth summary                   |
| `ghsl_epoch_manifest.csv`        | GHSL built/population images selected by epoch                |
| `ghsl_native_grid.json`          | Native GHSL grid description                                  |
| `grid_linkage.yaml`              | Relationship between native auxiliary grids and the 30 m grid |
| `orchestrate_version.json`       | Frozen Day 3 signature                                        |

### Built-up-candidate metadata

| File                               | Role                                          |
| ---------------------------------- | --------------------------------------------- |
| `dependency_manifest.csv`          | Day 4 input hashes                            |
| `histograms.json`                  | Valid-pixel histograms used by Otsu           |
| `threshold_table.csv`              | 24 epoch-method Otsu thresholds               |
| `candidate_area_summary.csv`       | Built-up area produced by every candidate     |
| `processing_recipe.json`           | Formula, validity and classification settings |
| `earth_engine_tasks.csv`           | Twelve index/candidate task records           |
| `output_manifest.csv`              | Six index and six candidate asset IDs         |
| `built_up_candidates_version.json` | Frozen Day 4 signature                        |

### Final-dataset metadata

| File                            | Role                                                  |
| ------------------------------- | ----------------------------------------------------- |
| `dependency_manifest.csv`       | Inputs and checksums consumed by Day 6                |
| `earth_engine_raster_tasks.csv` | Six state, five transition and one support task       |
| `raster_output_manifest.csv`    | Final raster asset IDs, bands and recipes             |
| `raster_products_version.json`  | Raster-stage signature                                |
| `earth_engine_table_tasks.csv`  | Five Drive table-export tasks                         |
| `transition_statistics.json`    | Transition counts and support statistics              |
| `output_manifest.csv`           | Final local release-file inventory                    |
| `final_dataset_version.json`    | Release identity, status, counts and stable signature |
| `checksums.sha256`              | SHA-256 checksums for final release files             |

### Modeling metadata

| Path                                                                                       | Role                                                      | Tracking                      |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------- |
| `data/metadata/modeling/logistic_regression/preflight.json`                                | Validated LR dataset, feature and temporal-split contract | Git                           |
| `data/metadata/modeling/logistic_regression/temporal_split_manifest.csv`                   | Chronological LR fold definition                          | Git                           |
| `data/metadata/modeling/svgp/preflight.json`                                               | Retained SVGP preflight contract                          | Git                           |
| `data/metadata/modeling/svgp/diagnostics/adam_cpu32/preflight.json`                        | Historical runtime diagnostic; useful provenance only     | Git if intentionally retained |
| `data/metadata/modeling/svgp/experiments/v1_natgrad/preflight.json`                        | Historical Natural-Gradient SVGP preflight contract       | Git                           |
| `data/metadata/modeling/st_svgp/preflight.json`                                            | Promoted ST-SVGP preflight contract                       | Git                           |
| `data/metadata/modeling/st_svgp/experiments/free_init_1p0/preflight.json`                  | Retained free-init-1.0 ST-SVGP experiment contract        | Git                           |
| `data/metadata/modeling/st_svgp/experiments/temporal_lengthscale_fixed_1p5/preflight.json` | Retained fixed-1.5 ST-SVGP diagnostic contract            | Git                           |

These files are small, auditable records of the exact model/dataset contract.
They are different from fitted checkpoints and row-level OOF predictions, which
are generated artifacts and should normally remain outside Git.

### Code

| Path                           | Role                                                                                                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/analysis/boundaries/`     | Boundary inspection, grid construction and Day 1 figures                                                                           |
| `src/analysis/landsat/`        | QA mask, catalogue, exact window selection, composites and catalogue figures                                                       |
| `src/analysis/orchestration/`  | Day 3 preflight, dependency checks and finalisation                                                                                |
| `src/analysis/auxiliary/`      | OSM extraction                                                                                                                     |
| `src/analysis/built_up/`       | Index formulas, Otsu, exports and Day 4 reports                                                                                    |
| `src/analysis/validation/`     | Stratified samples, manual evaluation and method freezing                                                                          |
| `src/analysis/final_dataset/`  | States, transitions, cell-time tables, metrics and release finalisation                                                            |
| `src/analysis/annual_dataset/` | Generates the targeted annual diagnostic sample/reserve, validates reviewed labels and reports the independent annual-label check  |
| `src/feature_engineering/`     | Model-ready transformations and retained candidate features, including log-distance and built-fraction × recent-growth interaction |
| `src/models/`                  | Logistic Regression, SVGP, ST-SVGP training, probability metrics, calibration and temporal prediction-set evaluation               |
| `src/preprocessing/`           | Reserved for later preprocessing utilities                                                                                         |
| `src/visualization/`           | Reserved for reusable visualisation code                                                                                           |

### Modeling code retained on `feature/modeling-pipeline`

| Path                                                       | Role                                                                                                                                                                                              | Status                      |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `src/models/train_logistic.py`                             | Retained chronological Logistic Regression baseline                                                                                                                                               | Retained                    |
| `src/models/train_svgp.py`                                 | Retained Adam SVGP implementation with Bernoulli-probit likelihood and separable Matérn-3/2 space × time kernel                                                                                   | Retained reference          |
| `src/models/svgp/features/experiment.py`                   | Controlled SVGP feature-set experiments using the selected SVGP implementation                                                                                                                    | Retained                    |
| `src/models/svgp/natgrad/experiment_v1.py`                 | Historical SVGP Natural-Gradient experiment                                                                                                                                                       | Experimental / not selected |
| `src/models/train_st_svgp.py`                              | Full real-data ST-SVGP preflight and rolling-validation entry point; extended for regularization, calendar-time trend, optional trainable inducing locations and pre-specified stability stopping | Retained                    |
| `src/models/st_svgp/state_space.py`                        | Temporal Matérn-3/2 Markov state-space prior                                                                                                                                                      | Retained                    |
| `src/models/st_svgp/filtering.py`                          | Gaussian Kalman filtering and RTS smoothing                                                                                                                                                       | Retained                    |
| `src/models/st_svgp/cvi.py`                                | Bernoulli-probit CVI / Natural-Gradient components                                                                                                                                                | Retained                    |
| `src/models/st_svgp/spatial.py`                            | Anisotropic spatial Matérn-3/2 covariance and sparse conditional                                                                                                                                  | Retained                    |
| `src/models/st_svgp/block_inference.py`                    | Dense spatial CVI-site block filtering/smoothing for inducing states                                                                                                                              | Retained                    |
| `src/models/st_svgp/model.py`                              | Full sparse Markov ST-SVGP model and posterior mapping; supports fixed or explicitly trainable inducing locations while preserving the fixed-location default                                     | Retained                    |
| `src/models/st_svgp/validate_temporal_kernel.py`           | Executable temporal-kernel mathematical validation                                                                                                                                                | Retained                    |
| `src/models/st_svgp/validate_filter_smoother.py`           | Executable filter/smoother validation against dense GP oracle                                                                                                                                     | Retained                    |
| `src/models/st_svgp/validate_cvi_natgrad.py`               | Executable CVI/Natural-Gradient validation                                                                                                                                                        | Retained                    |
| `src/models/evaluation/evaluate_probabilities.py`          | Shared OOF evaluator for LR, Strong SVGP and promoted ST-SVGP                                                                                                                                     | Retained                    |
| `src/models/evaluation/calibration.py`                     | Calibration metrics and reliability tables                                                                                                                                                        | Retained                    |
| `src/models/evaluation/prediction_sets.py`                 | Temporal prediction-set coverage diagnostics                                                                                                                                                      | Retained                    |
| `src/models/evaluation/st_svgp_convergence_diagnostics.py` | Read-only convergence summaries from retained training histories; reports ELBO, gradient/clipping, kernel and CVI-site movement                                                                   | Retained                    |
| `src/models/evaluation/st_svgp_day5_diagnostics.py`        | OOF spatial-error and uncertainty diagnostics for annual/five-year horizons                                                                                                                       | Retained                    |
| `src/models/evaluation/st_svgp_day5_reconstruction.py`     | Reconstructs rolling-fold states from frozen configs/data and requires exact OOF probability reproduction                                                                                         | Retained                    |
| `src/models/evaluation/st_svgp_day5_shap.py`               | Strict reload/preflight and bounded SHAP pilot for reconstructed states                                                                                                                           | Retained                    |
| `src/models/evaluation/st_svgp_explainability.py`          | Final context-conditioned SHAP analysis over the nine observed covariates                                                                                                                         | Retained                    |
| `src/models/metrics.py`                                    | Shared probabilistic scoring utilities                                                                                                                                                            | Retained                    |

### Retained modeling decisions

- **Logistic Regression:** chronological rolling-validation baseline; the
built-fraction × recent-growth interaction is retained in the updated
feature specification.
- **SVGP:** the selected feature set uses
`log_distance_to_built_m_t` and
`built_fraction_x_recent_growth_t`. The gain over the Base SVGP is modest,
but Log Loss, Brier, PR-AUC and ECE move coherently in the right direction.
- **SVGP Natural Gradient:** retained only as an experiment/provenance record;
it is not the primary SVGP comparator.
- **ST-SVGP:** 64 fixed spatial inducing locations, Bernoulli-probit
likelihood, dense time-specific Gaussian CVI pseudo-sites, Natural-Gradient
CVI updates, Adam hyperparameter/mean updates, sequential Kalman filtering
and RTS smoothing.
- **Temporal diagnostics:** free-init-1.0 and fixed-1.5 are retained as
diagnostics. The promoted candidate starts at 1.5 five-year steps and keeps
the temporal lengthscale trainable.
- **Final test:** origin 2020 (`2020 -> 2025`) remains locked and is not used
for model selection.

### Modeling reports worth retaining in Git

| Path/pattern                                                                | Role                                                                     | Recommended tracking                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------- |
| `reports/modeling/calibration/evaluation_manifest.json`                     | Common-comparison inputs and protocol                                    | Git                                            |
| `reports/modeling/calibration/probability_evaluation_summary.csv`           | Aggregate LR / Strong SVGP / ST-SVGP comparison                          | Git                                            |
| `reports/modeling/calibration/probability_metrics_by_fold.csv`              | Fold-level probability/calibration metrics                               | Git                                            |
| `reports/modeling/calibration/reliability_bins.csv`                         | Shared reliability-bin data                                              | Git                                            |
| `reports/modeling/calibration/prediction_set_coverage_80.csv`               | Temporal prediction-set coverage diagnostics                             | Git                                            |
| `reports/modeling/calibration/*.png`                                        | Small reliability figures                                                | Git when intentionally generated               |
| `reports/modeling/logistic_regression/metrics/*.csv`                        | Retained LR metrics and coefficients                                     | Git                                            |
| `reports/modeling/svgp/metrics/*summary*.csv`                               | Retained reference SVGP summary                                          | Git                                            |
| `reports/modeling/svgp/features/*/svgp_feature_manifest.json`               | Exact Base / Log distance / Log distance + growth experiment definitions | Git                                            |
| `reports/modeling/svgp/features/*/svgp_feature_validation_metrics.csv`      | Feature fold metrics                                                     | Git                                            |
| `reports/modeling/svgp/features/*/svgp_feature_validation_summary.csv`      | Feature aggregate summaries                                              | Git                                            |
| `reports/modeling/svgp/features/*/svgp_feature_reliability_bins.csv`        | Feature calibration diagnostics                                          | Git                                            |
| `reports/modeling/svgp/features/*/svgp_feature_prediction_set_coverage.csv` | Feature temporal-coverage diagnostics                                    | Git                                            |
| `reports/modeling/svgp/experiments/v1_natgrad/metrics/*summary*.csv`        | Negative-result summary for historical NatGrad experiment                | Git                                            |
| `reports/modeling/st_svgp/temporal_kernel_validation.json`                  | Temporal Markov-kernel validation result                                 | Git                                            |
| `reports/modeling/st_svgp/cvi_natgrad_validation.json`                      | CVI/Natural-Gradient validation result                                   | Git                                            |
| `reports/modeling/st_svgp/metrics/st_svgp_rolling_validation_metrics.csv`   | Promoted ST-SVGP fold metrics                                            | Git                                            |
| `reports/modeling/st_svgp/metrics/st_svgp_rolling_validation_summary.json`  | Promoted ST-SVGP summary                                                 | Git                                            |
| `reports/modeling/st_svgp/experiments/*/metrics/*validation_metrics.csv`    | Retained diagnostic fold metrics                                         | Git                                            |
| `reports/modeling/st_svgp/experiments/*/metrics/*validation_summary.json`   | Retained diagnostic summaries                                            | Git                                            |
| `**/*_oof_predictions.parquet`                                              | Row-level OOF probabilities; regenerable and potentially large           | Local/release                                  |
| `**/*_training_history.csv`                                                 | Iteration-level optimizer traces                                         | Local/release unless needed for a paper figure |

#### Improvement‑campaign result artifacts

`reports/modeling/st_svgp_improvements/` contains the experiment-local evidence for the optimizer, spatial, temporal, inducing and combined branches. Recommended tracking is deliberately selective:

| Artifact pattern                                         | Role                                                              | Recommended tracking                                      |
| -------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- |
| `**/metadata/preflight.json`                             | Exact experiment/data contract recorded before a run              | Git                                                       |
| `**/diagnostics/*.md`                                    | Human-readable scientific interpretation                          | Git                                                       |
| `**/diagnostics/*.csv`                                   | Compact convergence/prediction comparisons and stopping summaries | Git                                                       |
| `**/diagnostics/figures/*.png`                           | Selected convergence figures used for review/presentation         | Git                                                       |
| `**/metrics/rolling_metrics.csv`                         | Fold-level predictive metrics                                     | Git                                                       |
| `**/metrics/*summary*.json` / `**/metrics/*summary*.csv` | Compact experiment summaries                                      | Git                                                       |
| `**/metrics/temporal_parameter_summary.csv`              | Final temporal-parameter evidence                                 | Git                                                       |
| `**/metrics/st_svgp_training_history.csv`                | Full per-iteration optimization trace                             | Local/release by default; summaries/figures are committed |
| `**/predictions/*.parquet`                               | Row-level OOF predictions                                         | Local/release; ignore in Git                              |

Current campaign interpretation:

- `EARLY-STOP-A01`: **PRIMARY** development configuration; strongest annual discrimination, but the pre-specified training-stability rule was not satisfied and probability reliability remains fold-dependent.
- `TREN-A01`: **KEEP_EXPERIMENTAL**; explicit calendar-time trend gives the clearest aggregate probability-quality improvement among isolated structural interventions.
- optimizer, spatial, temporal-prior, inducing, combined and clip-only sensitivity branches: **KEEP_AS_EVIDENCE**.
- `CAL-EARLY-A01`: **REJECTED** and should not be part of the retained implementation/configuration commit.

#### Failure diagnostics and explainability reports

| Path                                                                        | Role                                           | Recommended tracking         |
| --------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------- |
| `reports/modeling/day5_diagnostics/day5_model_diagnostics.md`               | Consolidated failure-diagnostic interpretation | Git                          |
| `reports/modeling/day5_diagnostics/*/figures/*.png`                         | Fold maps and uncertainty/error figures        | Git                          |
| `reports/modeling/day5_diagnostics/*/metrics/*.csv` / `*.json`              | Compact error/context/uncertainty summaries    | Git                          |
| `reports/modeling/day5_diagnostics/*/metrics/oof_row_diagnostics.parquet`   | Row-level diagnostic table                     | Local/release; ignore in Git |
| `reports/modeling/day5_diagnostics/reconstruction_*.json`                   | Reconstruction and OOF-reproduction gates      | Git                          |
| `reports/modeling/st_svgp_explainability/context_manifest.csv`              | Frozen representative SHAP context definition  | Git                          |
| `reports/modeling/st_svgp_explainability/*/shap_feature_importance.csv`     | Feature-level mean attribution magnitude       | Git                          |
| `reports/modeling/st_svgp_explainability/*/shap_feature_direction.csv`      | Feature attribution direction summaries        | Git                          |
| `reports/modeling/st_svgp_explainability/*/*.png`                           | Final SHAP summary/context figures             | Git                          |
| `reports/modeling/st_svgp_explainability/*/shap_values.parquet`             | Row-level SHAP values                          | Local/release; ignore in Git |
| `reports/modeling/st_svgp_explainability/run_summary.json`                  | Explainability run/provenance summary          | Git                          |
| `reports/modeling/st_svgp_explainability/six_state_sanity.json`             | Reconstructed-state sanity check               | Git                          |
| `reports/modeling/st_svgp_explainability/st_svgp_explainability_summary.md` | Final explainability interpretation            | Git                          |

The explainability pipeline uses verified reconstructed ST-SVGP states and perturbs only observed covariates under fixed representative space-time contexts. The outputs are model attributions, not causal effects.

### Generated/local files not intended for this Git commit

| Path                                                                            | Reason                                                                                             |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `artifacts/models/**/checkpoint*`                                               | Generated TensorFlow/GPflow fitted-state binaries                                                  |
| `artifacts/archive/`                                                            | Historical local artifacts                                                                         |
| `data/archive/`                                                                 | Archived data/release copies; large and unrelated to the modeling source commit                    |
| `reports/modeling/**/predictions/*.parquet`                                     | Regenerable row-level OOF predictions                                                              |
| `reports/modeling/**/oof_row_diagnostics.parquet`                               | Regenerable row-level diagnostics                                                                  |
| `reports/modeling/st_svgp_explainability/**/shap_values.parquet`                | Regenerable SHAP values                                                                            |
| `reports/modeling/st_svgp_improvements/**/metrics/st_svgp_training_history.csv` | Full training traces, large and regenerable                                                        |
| `reports/population_audit/`                                                     | Dataset-audit output unrelated to this modeling commit                                             |
| `apply_st_svgp_candidate_promotion.py`                                          | One-off promotion helper; redundant after the final source/config state is committed               |
| `apply_st_svgp_temporal_freeze_fix_v2.py`                                       | One-off repair helper; regression tests now protect the fix                                        |
| `temporal_lengthscale_fixed_1p5.patch`                                          | Temporary patch representation; redundant once source/config/tests are committed                   |
| `scripts/reorganize_svgp_phase.py`                                              | One-off repository reorganisation helper unless explicitly kept as a documented migration          |
| `tests/test.py`                                                                 | Generic scratch-test name; do not commit unless reviewed and renamed to describe a stable contract |
| `artifacts/models/day5_reconstructed/`                                          | Reconstructed model states, regenerable                                                            |

### Modeling outputs intentionally missing / not yet final

- **2020 -> 2025 final-test metrics:** intentionally absent because the final
test remains locked.
- **2030/2035 forecast products:** not produced yet; model selection and final
evaluation must be frozen first.
- **Strong-SVGP final-fit checkpoint:** the selected feature experiment is
currently a rolling-validation comparator; its manifest records that final
fit was not executed.
- **Promoted ST-SVGP final-test checkpoint/results:** not required before the
pre-test protocol is frozen.
- **ST-SVGP reliability PNG:** not present in the current tree; tabular
reliability information is available in the shared `reliability_bins.csv`.

### Documentation, reports and notebooks

| Path                                          | Role                                                                                    |
| --------------------------------------------- | --------------------------------------------------------------------------------------- |
| `docs/reproduction_and_handover.md`           | Exact clean-run and handover procedure                                                  |
| `docs/known_issues.md`                        | Open scientific and implementation issues                                               |
| `docs/documented_file_structure.md`           | Repository inventory, retained modelling files, generated artifacts and tracking policy |
| `reports/day1/`                               | Boundary and grid QA figures                                                            |
| `reports/orchestration/`                      | Composite/source QA report and figures                                                  |
| `reports/built_up_candidates/`                | Index, candidate and threshold galleries                                                |
| `reports/support_diagnostics/`                | Landsat-support diagnosis that motivated catalogue v3                                   |
| `reports/day7/qa_report.md`                   | Release-level QA summary and blocking issues                                            |
| `notebooks/08_diagnose_landsat_support.ipynb` | Read-only support diagnostic                                                            |
| `notebooks/09_explore_final_dataset_v2.ipynb` | Read-only exploratory analysis; not part of dataset construction                        |
| `reports/modeling/`                           | Rolling-validation summaries, feature experiments, calibration and retained diagnostics |

### Tests and references

| Path                                              | Role                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------ |
| `tests/test_study_grid.py`                        | Boundary/grid invariants                                                       |
| `tests/test_landsat_catalog*.py`                  | Catalogue and selection policy                                                 |
| `tests/test_orchestrate_sources.py`               | Day 3 assets and alignment                                                     |
| `tests/test_built_up_candidates.py`               | Index, Otsu, asset schema and validity                                         |
| `tests/test_method_selection.py`                  | Sampling and validation logic                                                  |
| `tests/test_final_dataset.py`                     | Persistence, transitions, leakage, demand and release                          |
| `tests/test_smoke.py`                             | Package smoke test                                                             |
| `tests/test_feature_engineering.py`               | Candidate-feature transformations and feature contracts                        |
| `tests/test_logistic.py`                          | Retained Logistic Regression model contract                                    |
| `tests/logistic_regression/test_experiment_v1.py` | Historical LR experiment regression tests                                      |
| `tests/test_svgp.py`                              | SVGP architecture, chronology and configuration tests                          |
| `tests/test_calibration.py`                       | Calibration metric tests                                                       |
| `tests/test_prediction_sets.py`                   | Temporal prediction-set coverage tests                                         |
| `tests/test_st_svgp.py`                           | 20 temporal Matérn/state-space tests                                           |
| `tests/test_st_svgp_filtering.py`                 | 5 filtering/smoothing equivalence tests                                        |
| `tests/test_st_svgp_cvi.py`                       | 7 Bernoulli-probit CVI/Natural-Gradient tests                                  |
| `tests/test_st_svgp_model.py`                     | 5 spatial/block/config tests, including frozen-lengthscale regression contract |
| `tests/test_annual_diagnostic.py`                 | Synthetic/contract tests for annual diagnostic sampling and evaluation         |
| `tests/test_st_svgp_convergence_diagnostics.py`   | Tests for convergence diagnostics                                              |
| `tests/test_st_svgp_day5_diagnostics.py`          | Tests for Day5 OOF diagnostics                                                 |
| `tests/test_st_svgp_day5_reconstruction.py`       | Tests for state reconstruction                                                 |
| `tests/test_st_svgp_day5_shap.py`                 | Tests for SHAP pilot                                                           |
| `tests/test_st_svgp_explainability.py`            | Tests for final explainability pipeline                                        |
| `tests/test_st_svgp_regularization.py`            | Tests for regularization features                                              |
| `tests/reference/*.json`                          | Frozen stage signatures; older versions remain provenance records              |

### Makefile entry points added for the retained annual candidates

For an auditable final repository, the Makefile exposes the two retained annual configurations directly:

```text
make st-svgp-tren-a01-preflight
make st-svgp-tren-a01
make st-svgp-primary-preflight
make st-svgp-primary
```

`st-svgp-primary` must remain a rolling-development command (`--rolling-only`); it must not evaluate the locked annual block.

## Release packaging

```bash
make handover
```

This runs the local QA tests and creates:

```text
dist/
├── HANDOVER_MANIFEST.csv
└── yaounde_urban_expansion_30m_v2_handover.zip
```

Large Earth Engine assets and the full Parquet table are referenced, not copied,
unless explicitly added to an external data release.
```
