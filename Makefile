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

test-grid:  ## Run pytest on the study grid module
	pytest tests/test_study_grid.py

day1: inspect-boundaries build-grid test-grid  ## Run the complete Day 1 pipeline (inspection, grid build, tests)