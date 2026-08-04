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
make orchestration-status       # repeat until completed
make orchestration-finalize
make test-orchestration

# Step 4
make built-up-preflight
make built-up-submit
make built-up-status            # repeat until completed
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
make dataset-status-rasters     # repeat until completed
make dataset-finalize-rasters
make dataset-submit-tables
make dataset-status-tables      # repeat until completed
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

| Path | Role | Tracking |
|---|---|---|
| `Makefile` | Stable command-line entry points for Days 1–7 | Git |
| `README.md` | Project overview, file inventory and reproduction entry point | Git |
| `pyproject.toml` | Python package metadata and dependencies | Git |
| `checksums.txt` | Top-level source or release checksum record | Git |
| `.gitignore` | Excludes caches, checkpoints, staging data and large local outputs | Git |

### Configuration

| File | Role |
|---|---|
| `configs/study_area.yaml` | Boundary selection, CRS, context buffer and frozen 30 m grid |
| `configs/landsat_catalog.yaml` | Epochs, sensor policy, QA rules and exact per-epoch window selection |
| `configs/orchestrate_sources.yaml` | Composite, GHSL, terrain, OSM and Earth Engine export settings |
| `configs/built_up_candidates.yaml` | Spectral indices, candidate methods, Otsu settings and asset roots |
| `configs/mapping_validation.yaml` | Stratified sampling and manual method-comparison protocol |
| `configs/final_dataset.yaml` | NDBI state, transition, table and release configuration |

### Data directories

| Path | Role | Tracking |
|---|---|---|
| `data/raw/boundaries/` | Downloaded HDX and GeoBoundaries source files and source metadata | Usually local/Git-LFS |
| `data/raw/osm/` | Timestamped Cameroon OSM extract, checksum and metadata | Usually local |
| `data/processed/boundaries/` | Yaoundé core, context buffer, convex hull and ADM2 reference | Git or release package |
| `data/processed/grid/` | Frozen cell table, masks and 30 m template raster | Usually local/release |
| `data/processed/osm/yaounde_current_osm.gpkg` | Clean context-area roads, railways and buildings | Usually local |
| `data/staging/cell_time_exports/` | Five downloaded Earth Engine CSV table exports | Local only |
| `data/staging/final_dataset/tracking_stack.tif` | Temporary local tracking raster used for metrics | Local only |
| `data/validation/validation_samples.gpkg` | Stratified Day 5 sample locations and predictions | Git/release |
| `data/validation/manual_labels.csv` | Human labels; currently incomplete | Git when completed |
| `data/final/yaounde_urban_expansion_30m_v2/` | Final Parquet, demand, metrics, schema, dictionary and dataset card | Release; large Parquet may be external |

### Boundary and grid metadata

| File | Role |
|---|---|
| `data/metadata/boundary_report.json` | Core/context areas, selected units and boundary checks |
| `data/metadata/grid_specification.json` | Frozen CRS, extent, dimensions, transform and checksum |
| `data/metadata/adm2_adm3_comparison.json` | ADM2/ADM3 boundary comparison |
| `data/metadata/source_boundary_inspection*.json` | Raw boundary field, name and geometry inspection |

### Landsat metadata

| File/pattern | Role |
|---|---|
| `scene_manifest_all.csv` / `.parquet` | Complete diagnostic scene catalogue |
| `monthly_availability.csv` / `.parquet` | Monthly scene and valid-coverage summaries |
| `candidate_window_details.csv` | Exact per-epoch, per-window observation metrics |
| `candidate_window_summary.csv` | Compact candidate-window comparison |
| `epoch_quality_summary.csv` | Selected window, sensors, depth and coverage for each epoch |
| `selected_scene_manifest.csv` | Exact frozen scene IDs consumed by Day 3 |
| `old_vs_new_selected_manifest.csv` | Audit of the corrected catalogue against the previous selection |
| `compositing_protocol.yaml` | Human-readable frozen per-epoch protocol |
| `catalog_version.json` | Day 2 stable signature and checksums |
| `_checkpoints/` | Resumable catalogue state; never part of a release |

### Orchestration metadata

| File | Role |
|---|---|
| `auxiliary_source_manifest.csv` | GHSL, terrain and OSM source inventory |
| `dependency_manifest.csv` | Input paths and hashes consumed by Day 3 |
| `earth_engine_tasks.csv` | Submitted composite/count/terrain task IDs and states |
| `landsat_composite_manifest.csv` | Final composite and count asset IDs |
| `landsat_epoch_quality.csv` | Day 3 support and observation-depth summary |
| `ghsl_epoch_manifest.csv` | GHSL built/population images selected by epoch |
| `ghsl_native_grid.json` | Native GHSL grid description |
| `grid_linkage.yaml` | Relationship between native auxiliary grids and the 30 m grid |
| `orchestrate_version.json` | Frozen Day 3 signature |

### Built-up-candidate metadata

| File | Role |
|---|---|
| `dependency_manifest.csv` | Day 4 input hashes |
| `histograms.json` | Valid-pixel histograms used by Otsu |
| `threshold_table.csv` | 24 epoch-method Otsu thresholds |
| `candidate_area_summary.csv` | Built-up area produced by every candidate |
| `processing_recipe.json` | Formula, validity and classification settings |
| `earth_engine_tasks.csv` | Twelve index/candidate task records |
| `output_manifest.csv` | Six index and six candidate asset IDs |
| `built_up_candidates_version.json` | Frozen Day 4 signature |

### Final-dataset metadata

| File | Role |
|---|---|
| `dependency_manifest.csv` | Inputs and checksums consumed by Day 6 |
| `earth_engine_raster_tasks.csv` | Six state, five transition and one support task |
| `raster_output_manifest.csv` | Final raster asset IDs, bands and recipes |
| `raster_products_version.json` | Raster-stage signature |
| `earth_engine_table_tasks.csv` | Five Drive table-export tasks |
| `transition_statistics.json` | Transition counts and support statistics |
| `output_manifest.csv` | Final local release-file inventory |
| `final_dataset_version.json` | Release identity, status, counts and stable signature |
| `checksums.sha256` | SHA-256 checksums for final release files |

### Code

| Path | Role |
|---|---|
| `src/analysis/boundaries/` | Boundary inspection, grid construction and Day 1 figures |
| `src/analysis/landsat/` | QA mask, catalogue, exact window selection, composites and catalogue figures |
| `src/analysis/orchestration/` | Day 3 preflight, dependency checks and finalisation |
| `src/analysis/auxiliary/` | OSM extraction |
| `src/analysis/built_up/` | Index formulas, Otsu, exports and Day 4 reports |
| `src/analysis/validation/` | Stratified samples, manual evaluation and method freezing |
| `src/analysis/final_dataset/` | States, transitions, cell-time tables, metrics and release finalisation |
| `src/feature_engineering/` | Reserved for model-ready transformations |
| `src/models/` | Reserved for ST-SVGP and evaluation code |
| `src/preprocessing/` | Reserved for later preprocessing utilities |
| `src/visualization/` | Reserved for reusable visualisation code |

### Documentation, reports and notebooks

| Path | Role |
|---|---|
| `docs/reproduction_and_handover.md` | Exact clean-run and handover procedure |
| `docs/known_issues.md` | Open scientific and implementation issues |
| `docs/documented_file_structure.md` | file inventory |
| `reports/day1/` | Boundary and grid QA figures |
| `reports/orchestration/` | Composite/source QA report and figures |
| `reports/built_up_candidates/` | Index, candidate and threshold galleries |
| `reports/support_diagnostics/` | Landsat-support diagnosis that motivated catalogue v3 |
| `reports/day7/qa_report.md` | Release-level QA summary and blocking issues |
| `notebooks/08_diagnose_landsat_support.ipynb` | Read-only support diagnostic |
| `notebooks/09_explore_final_dataset_v2.ipynb` | Read-only exploratory analysis; not part of dataset construction |
| `outputs/` | Reserved model forecasts, maps and uncertainty products |

### Tests and references

| Path | Role |
|---|---|
| `tests/test_study_grid.py` | Boundary/grid invariants |
| `tests/test_landsat_catalog*.py` | Catalogue and selection policy |
| `tests/test_orchestrate_sources.py` | Day 3 assets and alignment |
| `tests/test_built_up_candidates.py` | Index, Otsu, asset schema and validity |
| `tests/test_method_selection.py` | Sampling and validation logic |
| `tests/test_final_dataset.py` | Persistence, transitions, leakage, demand and release |
| `tests/test_smoke.py` | Package smoke test |
| `tests/reference/*.json` | Frozen stage signatures; older versions remain provenance records |

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
