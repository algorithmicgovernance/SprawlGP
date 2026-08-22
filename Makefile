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
#  Annual Landsat observations (2000-2025)
# =============================================================================

.PHONY: \
	annual-catalog \
	annual-orchestration-preflight \
	annual-orchestration-submit \
	annual-orchestration-status \
	annual-orchestration-finalize \
	annual-dataset-preflight \
	annual-products-submit \
	annual-products-status \
	annual-products-finalize \
	annual-audit \
	annual-build-tables \
	annual-tables-status \
	annual-tables-assemble \
	test-annual-dataset \
	test-annual

annual-catalog: ## Build the isolated 2000-2025 calendar-year Landsat catalogue
	python -m src.analysis.landsat.build_annual_catalog \
		--config configs/annual/landsat_catalog_annual.yaml

annual-orchestration-preflight: ## Validate annual grid and catalogue dependencies without submitting exports
	python -m src.analysis.orchestration.preflight \
		--config configs/annual/orchestrate_sources_annual.yaml

annual-orchestration-submit: ## Submit annual median composites and valid-count assets only
	python -m src.analysis.landsat.build_composites \
		--config configs/annual/orchestrate_sources_annual.yaml \
		--submit

annual-orchestration-status: ## Check annual Landsat Earth Engine task status
	python -m src.analysis.orchestration.finalize \
		--config configs/annual/orchestrate_sources_annual.yaml \
		--status-only

annual-orchestration-finalize: ## Validate completed annual Landsat assets and write metadata
	python -m src.analysis.orchestration.finalize \
		--config configs/annual/orchestrate_sources_annual.yaml

annual-dataset-preflight: ## Validate annual Day 3 inputs and isolation without submitting tasks
	python -m src.analysis.annual_dataset.build_products \
		--config configs/annual/annual_dataset.yaml \
		--preflight-only

annual-products-submit: ## Submit 26 compact annual index and state products
	python -m src.analysis.annual_dataset.build_products \
		--config configs/annual/annual_dataset.yaml \
		--submit

annual-products-status: ## Check annual state-product Earth Engine task status
	python -m src.analysis.annual_dataset.finalize \
		--config configs/annual/annual_dataset.yaml \
		--status-only

annual-products-finalize: ## Validate annual state assets and write state summaries
	python -m src.analysis.annual_dataset.finalize \
		--config configs/annual/annual_dataset.yaml

annual-audit: ## Build annual transition diagnostics and anomaly evidence outputs
	python -m src.analysis.annual_dataset.audit \
		--config configs/annual/annual_dataset.yaml

annual-build-tables: ## Submit 25 annual eligible-cell table exports
	python -m src.analysis.annual_dataset.build_tables \
		--config configs/annual/annual_dataset.yaml \
		--submit

annual-tables-status: ## Check annual table-export Earth Engine task status
	python -m src.analysis.annual_dataset.build_tables \
		--config configs/annual/annual_dataset.yaml \
		--status-only

annual-tables-assemble: ## Assemble downloaded annual table exports into Parquet
	python -m src.analysis.annual_dataset.build_tables \
		--config configs/annual/annual_dataset.yaml \
		--assemble

test-annual-dataset: ## Run isolated annual Day 3 tests
	python -m pytest \
		tests/test_annual_catalog.py \
		tests/test_annual_asset_isolation.py \
		tests/test_annual_dataset.py \
		-v

test-annual: ## Run all annual catalogue, isolation and Day 3 dataset tests
	pytest \
		tests/test_annual_catalog.py \
		tests/test_annual_asset_isolation.py \
		tests/test_annual_dataset.py \
		-v

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


# =============================================================================
# Baseline modelling — logistic regression experiment
# =============================================================================

# .PHONY: \
# 	modeling-preflight \
# 	modeling-logistic \
# 	modeling-test \
# 	modeling-baseline \
# 	modeling-logistic-v3-preflight \
# 	modeling-logistic-v3-test \
# 	modeling-logistic-v3

# modeling-preflight:  ## Validate the experiment configuration, feature schema and temporal split contract without running the training job
# 	python -m src.models.train_logistic \
# 		--config configs/modeling/experiment_v1.yaml \
# 		--preflight-only

# modeling-logistic:  ## Train the baseline Logistic Regression model on the frozen training set and save the fitted pipeline, coefficients and performance metrics
# 	python -m src.models.train_logistic \
# 		--config configs/modeling/experiment_v1.yaml

# modeling-test:  ## Run the critical unit tests for the baseline modelling pipeline (split integrity, target masking, feature availability)
# 	python -m pytest tests/test_modeling.py -v

# modeling-baseline: modeling-preflight modeling-logistic modeling-test  ## Execute the complete baseline modelling workflow: preflight checks, training and tests	

# modeling-logistic-v3-preflight:  ## Validate Logistic Regression v3 inputs and temporal contract without fitting
# 	python -m src.models.train_logistic_rolling \
# 		--config configs/modeling/experiment_v3.yaml \
# 		--preflight-only

# modeling-logistic-v3-test:  ## Run unit tests for population transformation, temporal splits and candidate selection
# 	python -m pytest \
# 		tests/test_logistic_rolling.py \
# 		-v

# modeling-logistic-v3:  ## Select the LR specification with rolling temporal validation and refit on all pre-test periods
# 	python -m src.models.train_logistic_rolling \
# 		--config configs/modeling/experiment_v3.yaml

# =============================================================================
# Modelling — selected Logistic Regression baseline
# =============================================================================

.PHONY: \
	modeling-logistic-preflight \
	modeling-logistic-test \
	modeling-logistic \
	modeling-logistic-v1

modeling-logistic-preflight:  ## Validate the selected Logistic Regression baseline without fitting
	python -m src.models.train_logistic \
		--config configs/modeling/logistic_regression_experiment.yaml \
		--preflight-only

modeling-logistic-test:  ## Run tests for the selected baseline and retained V1 experiment
	python -m pytest \
		tests/test_logistic.py \
		tests/logistic_regression/test_experiment_v1.py \
		-v

modeling-logistic:  ## Evaluate historical folds and fit the selected Logistic Regression baseline
	python -m src.models.train_logistic \
		--config configs/modeling/logistic_regression_experiment.yaml

modeling-logistic-v1:  ## Reproduce the retained initial Logistic Regression experiment
	python -m src.models.logistic_regression.experiment_v1 \
		--config configs/modeling/logistic_regression/experiment_v1.yaml


# =============================================================================
# Modelling — Sparse Variational Gaussian Process
# =============================================================================

# .PHONY: \
# 	modeling-svgp-preflight \
# 	modeling-svgp-test \
# 	modeling-svgp	\
# 	modeling-svgp-tune \
# 	modeling-svgp-tune-summary

# modeling-svgp-preflight:  ## Validate SVGP inputs, chronology and runtime
# 	python -m src.models.train_svgp \
# 		--config configs/modeling/svgp_experiment.yaml \
# 		--preflight-only

# modeling-svgp-test:  ## Run the architecture-critical SVGP tests
# 	python -m pytest tests/test_svgp.py -v

# modeling-svgp:  ## Run rolling SVGP evaluation and fit the four-origin final model
# 	python -m src.models.train_svgp \
# 		--config configs/modeling/svgp_experiment.yaml


# =============================================================================
# Modelling — Sparse Variational Gaussian Process
# =============================================================================

.PHONY: \
	modeling-svgp-preflight \
	modeling-svgp \
	modeling-probability-evaluation \
	modeling-svgp-feature \
	modeling-svgp-feature-summary \
	modeling-svgp-tune \
	modeling-svgp-tune-summary \
	modeling-svgp-natgrad-v1-preflight \
	modeling-svgp-natgrad-v1 

modeling-svgp-preflight:
	python -m src.models.train_svgp \
		--config configs/modeling/svgp_experiment.yaml \
		--preflight-only

modeling-svgp-test:
	python -m pytest tests/test_svgp.py -v

modeling-svgp:
	python -m src.models.train_svgp \
		--config configs/modeling/svgp_experiment.yaml

modeling-probability-evaluation:
	python -m src.models.evaluation.evaluate_probabilities

# Usage:
# make modeling-svgp-feature FEATURE_SET=log_distance
FEATURE_SET ?= base
modeling-svgp-feature:
	python -m src.models.svgp.features.experiment \
		--config configs/modeling/svgp_experiment.yaml \
		--feature-set $(FEATURE_SET)

modeling-svgp-feature-summary:
	python -m src.models.svgp.features.experiment --summarize

# Usage:
# make modeling-svgp-tune CANDIDATE=natgrad_g1e4
modeling-svgp-tune:
	python -m src.models.tune_svgp \
		--candidate $(CANDIDATE)

modeling-svgp-tune-summary:
	python -m src.models.tune_svgp \
		--summarize

modeling-svgp-natgrad-v1-preflight:
	python -m src.models.svgp.natgrad.experiment_v1 \
		--config configs/modeling/svgp/experiment_v1.yaml \
		--preflight-only

modeling-svgp-natgrad-v1:
	python -m src.models.svgp.natgrad.experiment_v1 \
		--config configs/modeling/svgp/experiment_v1.yaml

# BEGIN ST-SVGP WORKFLOW
# Canonical promoted ST-SVGP candidate: trainable temporal Matérn-3/2 lengthscale,
# initialised at 1.5 five-year steps.
PYTHON ?= python
ST_SVGP_CONFIG := configs/modeling/st_svgp.yaml
ST_SVGP_FIXED_1P5_CONFIG := configs/modeling/st_svgp/temporal_lengthscale_fixed_1p5.yaml
ST_SVGP_VALIDATION_DIR := configs/modeling/st_svgp

.PHONY: st-svgp-tests st-svgp-test-temporal-kernel st-svgp-test-filter-smoother \
	st-svgp-test-cvi st-svgp-test-model st-svgp-validate-temporal-kernel \
	st-svgp-validate-filter-smoother st-svgp-validate-cvi \
	st-svgp-pretraining-validation st-svgp-preflight st-svgp-rolling \
	st-svgp-final-fit st-svgp-fixed-1p5-rolling st-svgp-compare-oof \
	st-svgp-candidate-check

# [01/37] Matérn-3/2 state-space covariance equals direct kernel: ell=0.5, variance=0.3.
# [02/37] Matérn-3/2 state-space covariance equals direct kernel: ell=1.0, variance=1.0.
# [03/37] Matérn-3/2 state-space covariance equals direct kernel: ell=1.5, variance=1.0.
# [04/37] Matérn-3/2 state-space covariance equals direct kernel: ell=3.0, variance=2.0.
# [05/37] Closed-form transition A(delta) equals exp(F delta) at delta=0.0.
# [06/37] Closed-form transition A(delta) equals exp(F delta) at delta=0.1.
# [07/37] Closed-form transition A(delta) equals exp(F delta) at delta=0.25.
# [08/37] Closed-form transition A(delta) equals exp(F delta) at delta=1.0.
# [09/37] Closed-form transition A(delta) equals exp(F delta) at delta=2.5.
# [10/37] Stationarity identity P_inf=A P_inf A^T+Q holds at delta=0.0.
# [11/37] Stationarity identity P_inf=A P_inf A^T+Q holds at delta=0.25.
# [12/37] Stationarity identity P_inf=A P_inf A^T+Q holds at delta=1.0.
# [13/37] Stationarity identity P_inf=A P_inf A^T+Q holds at delta=2.5.
# [14/37] Process-noise matrix Q(delta) is PSD at delta=0.0.
# [15/37] Process-noise matrix Q(delta) is PSD at delta=0.01.
# [16/37] Process-noise matrix Q(delta) is PSD at delta=0.25.
# [17/37] Process-noise matrix Q(delta) is PSD at delta=1.0.
# [18/37] Process-noise matrix Q(delta) is PSD at delta=2.5.
# [19/37] Process-noise matrix Q(delta) is PSD at delta=10.0.
# [20/37] GPflow Matérn-3/2 uses the same covariance convention as the custom kernel.
# [21/37] RTS smoothed marginals equal exact dense-GP posterior marginals on regular times.
# [22/37] RTS smoothed marginals equal exact dense-GP posterior marginals on irregular times.
# [23/37] Kalman-filter Gaussian log marginal likelihood equals exact dense-GP value.
# [24/37] RTS smoothing does not increase marginal function variance relative to filtering.
# [25/37] Every filtered and smoothed covariance matrix remains PSD.
# [26/37] Bernoulli-probit expected log likelihood remains finite.
# [27/37] CVI autodiff moment gradients equal independent finite-difference gradients.
# [28/37] CVI Gaussian-site state-space posterior equals exact dense Gaussian-site posterior.
# [29/37] State-space CVI ELBO equals E_q[log p(y|f)] - KL(q||p).
# [30/37] Natural-Gradient CVI update preserves positive Gaussian-site precision.
# [31/37] Repeated Natural-Gradient updates improve the fixed synthetic ELBO.
# [32/37] Integrated Bernoulli-probit predictive probabilities remain strictly in (0,1).
# [33/37] Anisotropic spatial Matérn-3/2 covariance is symmetric PSD.
# [34/37] Sparse spatial conditional produces positive predictive variances.
# [35/37] Block Kalman/RTS inducing posterior equals exact dense Kronecker space-time GP posterior.
# [36/37] Real config contract keeps 64 inducing points, 9 predictors, CVI and locked 2020 test.
# [37/37] Frozen ell_t is non-trainable and excluded from Adam variables.
st-svgp-tests:
	$(PYTHON) -m pytest \
		tests/test_st_svgp.py \
		tests/test_st_svgp_filtering.py \
		tests/test_st_svgp_cvi.py \
		tests/test_st_svgp_model.py \
		-v

# Run only the 20 temporal Matérn/state-space unit tests.
st-svgp-test-temporal-kernel:
	$(PYTHON) -m pytest tests/test_st_svgp.py -v

# Run only the 5 Gaussian Kalman-filter / RTS-smoother equivalence tests.
st-svgp-test-filter-smoother:
	$(PYTHON) -m pytest tests/test_st_svgp_filtering.py -v

# Run only the 7 Bernoulli-probit CVI / Natural-Gradient validation tests.
st-svgp-test-cvi:
	$(PYTHON) -m pytest tests/test_st_svgp_cvi.py -v

# Run only the 5 real-model spatial/block/config-contract tests.
st-svgp-test-model:
	$(PYTHON) -m pytest tests/test_st_svgp_model.py -v

# Validate direct Matérn covariance, matrix exponential, stationary covariance,
# process-noise PSD and GPflow kernel-convention equality.
st-svgp-validate-temporal-kernel:
	$(PYTHON) -m src.models.st_svgp.validate_temporal_kernel \
		--config $(ST_SVGP_VALIDATION_DIR)/temporal_kernel_validation.yaml

# Validate Kalman filtering and RTS smoothing against a dense exact Gaussian-process oracle.
st-svgp-validate-filter-smoother:
	$(PYTHON) -m src.models.st_svgp.validate_filter_smoother \
		--config $(ST_SVGP_VALIDATION_DIR)/filter_smoother_validation.yaml

# Validate Bernoulli-probit CVI, Natural-Gradient moment updates and ELBO identity.
st-svgp-validate-cvi:
	$(PYTHON) -m src.models.st_svgp.validate_cvi_natgrad \
		--config $(ST_SVGP_VALIDATION_DIR)/cvi_natgrad_validation.yaml

# Run every retained mathematical gate before any real-data candidate training.
st-svgp-pretraining-validation: st-svgp-tests st-svgp-validate-temporal-kernel \
	st-svgp-validate-filter-smoother st-svgp-validate-cvi

# Validate promoted config and dataset contract without fitting the model.
st-svgp-preflight:
	$(PYTHON) -m src.models.train_st_svgp \
		--config $(ST_SVGP_CONFIG) \
		--preflight-only

# Reproduce the promoted free-init-1.5 candidate on the three pre-2020 rolling folds.
st-svgp-rolling:
	$(PYTHON) -m src.models.train_st_svgp \
		--config $(ST_SVGP_CONFIG) \
		--rolling-only

# Fit the promoted candidate on all pre-test origins (2000-2015).
# The canonical config still keeps 2020 locked and does not evaluate it.
st-svgp-final-fit:
	$(PYTHON) -m src.models.train_st_svgp \
		--config $(ST_SVGP_CONFIG)

# Reproduce the retained diagnostic in which ell_t is fixed exactly at 1.5.
st-svgp-fixed-1p5-rolling:
	$(PYTHON) -m src.models.train_st_svgp \
		--config $(ST_SVGP_FIXED_1P5_CONFIG) \
		--rolling-only

# Compare LR, Strong SVGP (log_distance_growth) and promoted ST-SVGP with the
# same probability, calibration and empirical temporal-coverage evaluator.
st-svgp-compare-oof:
	$(PYTHON) -m src.models.evaluation.evaluate_probabilities

# Complete pre-2020 candidate workflow.
st-svgp-candidate-check: st-svgp-pretraining-validation st-svgp-preflight \
	st-svgp-rolling st-svgp-compare-oof
# END ST-SVGP WORKFLOW
