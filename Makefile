# =============================================================================
#  Administrative boundaries and study grid
# =============================================================================

.PHONY: help install inspect-adm2 inspect-adm3 inspect-boundaries build-grid test-grid day1

help:  ## Display this help with a description of all available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode with dev dependencies
	python -m pip install -e ".[dev]"

inspect-adm2:  ## Inspect the ADM2 layer and generate the inspection report
	python -m src.analysis.boundaries.inspect_boundaries \
		--input data/raw/boundaries/hdx_cod_ab_cmr/extracted/cmr_admin2.geojson \
		--report data/metadata/source_boundary_inspection_adm2.json

inspect-adm3:  ## Inspect the ADM3 layer and generate the inspection report
	python -m src.analysis.boundaries.inspect_boundaries \
		--input data/raw/boundaries/hdx_cod_ab_cmr/extracted/cmr_admin3.geojson \
		--report data/metadata/source_boundary_inspection_adm3.json

inspect-boundaries: inspect-adm2 inspect-adm3  ## Run both ADM2 and ADM3 inspections

build-grid:  ## Build the study grid from the YAML configuration
	python -m src.analysis.boundaries.build_study_grid \
		--config configs/study_area.yaml

visualize-grid: ## Visualize the study grid and generate all diagnostic plots (overview, comparison, masks, and grid alignment) in reports/day1/
	python -m src.analysis.boundaries.visualize_study_grid \
		--config configs/study_area.yaml

test-grid:  ## Run pytest on the study grid module
	pytest tests/test_study_grid.py -v

qa-day1: build-grid visualize-grid test-grid  ## Run the complete QA pipeline (build, visualize, test) without updating the reference baseline

day1: inspect-boundaries build-grid test-grid  ## Run the complete Day 1 pipeline (inspection, grid build, tests)

# =============================================================================
#  — Landsat catalog and compositing windows
# =============================================================================

.PHONY: landsat-catalog landsat-visuals test-landsat day2

landsat-catalog: ## Build the full Landsat scene manifest by querying Earth Engine, compute QA metrics, and select optimal compositing windows
	python -m src.analysis.landsat.build_catalog \
		--config configs/landsat_catalog.yaml

landsat-visuals: ## Generate diagnostic plots for the Landsat catalog availability and selection strategy (reports/day2/)
	python -m src.analysis.landsat.visualize_catalog \
		--config configs/landsat_catalog.yaml

test-landsat: ## Run the pytest suite for the Landsat catalog module with verbose output
	pytest tests/test_landsat_catalog.py -v

day2: landsat-catalog landsat-visuals test-landsat ## Execute the complete Day 2 pipeline: build the Landsat catalog, generate visualizations, and run tests

# =============================================================================
#  — Orchestration (Landsat composites, terrain, OSM)
# =============================================================================

.PHONY: \
	orchestrate-preflight \
	orchestrate-submit \
	orchestrate-status \
	orchestrate-osm \
	orchestrate-finalize \
	test-orchestrate \
	freeze-orchestrate-v1

orchestrate-preflight: ## Check all frozen Day 1 and Day 2 dependencies (grid, catalog, manifests) before submitting Day 3 exports
	python -m src.analysis.orchestration.preflight \
		--config configs/orchestrate_sources.yaml

orchestrate-submit: ## Submit all Earth Engine export tasks for Landsat composites, valid-count layers, and the SRTM terrain product
	python -m src.analysis.landsat.build_composites \
		--config configs/orchestrate_sources.yaml \
		--submit

orchestrate-status:  ## Check the current running/completed status of all submitted Earth Engine export tasks
	python -m src.analysis.orchestration.finalize \
		--config configs/orchestrate_sources.yaml \
		--status-only

orchestrate-osm: ## Download the latest Geofabrik OpenStreetMap extract, clip it to the study area, and save it as a GeoPackage
	python -m src.analysis.auxiliary.build_osm_extract \
		--config configs/orchestrate_sources.yaml

orchestrate-finalize: ## Validate the completed Earth Engine assets, generate QA manifests, summary statistics, and finalize Day 3 metadata
	python -m src.analysis.orchestration.finalize \
		--config configs/orchestrate_sources.yaml

test-orchestrate: ## Run the pytest suite for the orchestration (Day 3) outputs and dependencies
	pytest tests/test_orchestrate_sources.py -v

freeze-orchestrate-v1: ## Copy the validated Day 3 version manifest to tests/reference as the frozen baseline for regression tests
	mkdir -p tests/reference
	cp data/metadata/orchestration/orchestrate_version.json \
		tests/reference/orchestrate_sources_v1.json

# =============================================================================
#  Built-up candidates (spectral indices + Otsu thresholds)
# =============================================================================

.PHONY: \
	built-up-preflight \
	built-up-submit \
	built-up-status \
	built-up-finalize \
	test-built-up \
	freeze-built-up-v1

built-up-preflight: ## Run preflight checks for Day 4: validate inputs, list completed epochs, and confirm expected index/candidate layers without submitting tasks
	python -m src.analysis.built_up.build_candidates \
		--config configs/built_up_candidates.yaml \
		--preflight-only

built-up-submit: ## Compute spectral indices, calculate Otsu thresholds, and submit all Day 4 Earth Engine exports (indices and candidate masks)
	python -m src.analysis.built_up.build_candidates \
		--config configs/built_up_candidates.yaml \
		--submit

built-up-status:  ## Check the current running/completed status of all submitted Day 4 Earth Engine export tasks
	python -m src.analysis.built_up.finalize \
		--config configs/built_up_candidates.yaml \
		--status-only

built-up-finalize: ## Validate the exported Day 4 assets, generate histograms, threshold tables, area summaries, and the final report
	python -m src.analysis.built_up.finalize \
		--config configs/built_up_candidates.yaml

test-built-up: ## Run the pytest suite for the built-up candidates (Day 4) module
	python -m pytest tests/test_built_up_candidates.py -v

freeze-built-up-v1: ## Copy the validated Day 4 version manifest to tests/reference as the frozen baseline for regression tests
	mkdir -p tests/reference
	cp \
		data/metadata/built_up_candidates/built_up_candidates_version.json \
		tests/reference/built_up_candidates_v1.json


# =============================================================================
#  Method selection and validation 
# =============================================================================

.PHONY: \
	validation-preflight \
	validation-samples \
	validation-evaluate \
	validation-freeze \
	test-validation

validation-preflight: ## Verify that all assets and required metadata are available before generating the validation sample
	python -m src.analysis.validation.select_mapping_method \
		--config configs/mapping_validation.yaml \
		--preflight

validation-samples: ## Generate the stratified validation sample (GPKG and blind CSV) for manual built-up labeling
	python -m src.analysis.validation.select_mapping_method \
		--config configs/mapping_validation.yaml \
		--generate-samples

validation-evaluate: ## Merge manual labels, calculate design-weighted validation metrics (F1, reversal rate), and generate method scores
	python -m src.analysis.validation.select_mapping_method \
		--config configs/mapping_validation.yaml \
		--evaluate

validation-freeze: ## Apply the selection rule to freeze the final mapping method and version the selection protocol for next step
	python -m src.analysis.validation.select_mapping_method \
		--config configs/mapping_validation.yaml \
		--freeze

test-validation: ## Run the pytest suite for the method selection (Day 5) module
	python -m pytest tests/test_method_selection.py -v


# =============================================================================
# Final Dataset — Raster products, tables, metrics and release
# =============================================================================

.PHONY: \
	dataset-preflight \
	dataset-submit-rasters \
	dataset-status-rasters \
	dataset-finalize-rasters \
	dataset-submit-tables \
	dataset-status-tables \
	dataset-assemble \
	dataset-finalize \
	test-dataset \
	freeze-dataset-v1

dataset-preflight: ## Validate all (Day 4) candidate assets, the NDBI mapping protocol, and the frozen grid before submitting any Earth Engine exports
	python -m src.analysis.final_dataset.build_products \
		--config configs/final_dataset.yaml \
		--preflight-only

dataset-submit-rasters: ## Submit Earth Engine tasks for the 5 final built-state assets (raw + persistent) and the 4 five-year transition rasters
	python -m src.analysis.final_dataset.build_products \
		--config configs/final_dataset.yaml \
		--submit

dataset-status-rasters: ## Monitor the completion status of the submitted raster export tasks (READY, RUNNING, COMPLETED, or FAILED)
	python -m src.analysis.final_dataset.build_products \
		--config configs/final_dataset.yaml \
		--status-only

dataset-finalize-rasters: ## Validate the exported raster assets (grid alignment, band schema, CRS) and write the raster output manifest
	python -m src.analysis.final_dataset.build_products \
		--config configs/final_dataset.yaml \
		--finalize

dataset-submit-tables: ## Submit four table exports (2005-2010 to 2020-2025) containing eligible-cell feature rows to Google Drive/Cloud Storage as CSV
	python -m src.analysis.final_dataset.build_tables \
		--config configs/final_dataset.yaml \
		--submit

dataset-status-tables: ## Check the batch status of the four submitted table export tasks
	python -m src.analysis.final_dataset.build_tables \
		--config configs/final_dataset.yaml \
		--status-only

dataset-assemble: ## Download the four exported CSVs from staging and merge them into a single consolidated cell-time Parquet dataset
	python -m src.analysis.final_dataset.build_tables \
		--config configs/final_dataset.yaml \
		--assemble

dataset-finalize: ## Compute built area, NUMP, MPS, historical demand, generate the QA report, data dictionary, and freeze the immutable V1 manifest
	python -m src.analysis.final_dataset.finalize \
		--config configs/final_dataset.yaml

test-dataset: ## Run the critical pytest suite for the final dataset (leakage, consistency, uniqueness, and spatial support checks)
	python -m pytest tests/test_final_dataset.py -v

freeze-dataset-v1: ## Copy the authoritative version JSON to the reference directory, pinning the regression baseline for future comparisons
	cp \
		data/metadata/final_dataset/final_dataset_version.json \
		tests/reference/final_dataset_v1.json

# ---------------------------------------------------------------------------
#  local release QA and handover packaging
# ---------------------------------------------------------------------------

.PHONY: \
	handover-check \
	handover-package \
	handover

handover-check:
	python -m pytest \
		tests/test_study_grid.py \
		tests/test_landsat_catalog.py \
		tests/test_landsat_catalog_selection.py \
		tests/test_orchestrate_sources.py \
		tests/test_built_up_candidates.py \
		tests/test_method_selection.py \
		tests/test_final_dataset.py \
		-v
	python scripts/package_handover.py \
		--root . \
		--check-only

handover-package:
	python scripts/package_handover.py \
		--root .

handover: handover-check handover-package
	@echo " Handover package created in dist/."
