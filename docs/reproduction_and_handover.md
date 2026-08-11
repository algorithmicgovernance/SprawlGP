# Reproduction and Handover

## Why the pipeline is phased

A fully unattended single command is not safe in the current architecture:
Earth Engine exports are asynchronous, and the five cell-time CSVs are delivered
through Google Drive. The repository therefore provides one documented entry
point and explicit resume points. No notebook editing is required.

## Clean environment

```bash
git clone https://github.com/algorithmicgovernance/SprawlGP.git
cd SprawlGP

python -m venv .venvt
source .venvt/bin/activate
pip install -e .

earthengine authenticate
python - <<'PY'
import ee
ee.Initialize(project="urban-sprawl-ssa")
print(ee.Number(1).getInfo())
PY
```

Do not reuse `_checkpoints/` from another configuration. Do not overwrite an
existing Earth Engine release unless a new asset root has been selected.

## Phase A — local infrastructure and catalogue

```bash
make build-grid

python -m src.analysis.landsat.build_catalog \
  --config configs/landsat_catalog.yaml
```

Acceptance: six epochs in `epoch_quality_summary.csv`, all with selected scenes
and ≥95% exact core coverage.

## Phase B — composites and auxiliary sources

```bash
make orchestration-preflight
make orchestration-submit
make orchestration-status
```

Repeat `make orchestration-status` until all tasks are `COMPLETED` or `EXISTS`.

```bash
make orchestration-finalize
make test-orchestration
```

Acceptance: six composites, six observation-count assets, one terrain asset and
a frozen Day 3 signature.

## Phase C — indices and candidates

```bash
make built-up-preflight
make built-up-submit
make built-up-status
```

Wait for all twelve tasks.

```bash
make built-up-finalize
make test-built-up
```

Acceptance: 42 continuous epoch-index layers, 24 binary candidate layers and
twelve valid assets. IBI remains continuous-only.

## Phase D — validation

```bash
make validation-preflight
make validation-samples
```

Complete `data/validation/manual_labels.csv`. The pipeline must not invent or
auto-fill labels.

```bash
make validation-evaluate
make validation-freeze
make test-validation
```

The current v2 release skipped the completed manual evaluation and therefore
records NDBI as `PROVISIONAL_PENDING_MANUAL_VALIDATION`.

## Phase E — final states and tables

```bash
make dataset-preflight
make dataset-submit-rasters
make dataset-status-rasters
```

Wait for twelve raster tasks, then:

```bash
make dataset-finalize-rasters
make dataset-submit-tables
make dataset-status-tables
```

Wait for five table tasks. Download the following files to
`data/staging/cell_time_exports/`:

```text
cell_time_2000_2005.csv
cell_time_2005_2010.csv
cell_time_2010_2015.csv
cell_time_2015_2020.csv
cell_time_2020_2025.csv
```

Then run:

```bash
make dataset-assemble
make dataset-finalize
make test-dataset
```

Acceptance: six local release files, final metadata, SHA-256 checksums and all
Day 6 tests passing.

## Phase F — handover

```bash
make handover
```

This does not rebuild Earth Engine assets. It verifies the final local release,
runs tests and creates a compact metadata/documentation package.

## Clean-rerun rules

1. Configuration files are the source of truth.
2. Never edit generated rasters manually.
3. Never reuse a checkpoint after changing the catalogue configuration.
4. Use a new Earth Engine asset root for a scientific revision.
5. Preserve old `tests/reference/*.json` files as provenance.
6. Notebooks are read-only exploratory consumers.
7. Record manual downloads and human labels explicitly.
