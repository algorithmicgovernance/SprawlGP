# Probabilistic Forecasting of Urban Sprawl

A probabilistic geospatial modelling framework for forecasting urban sprawl dynamics from Earth observation data using Bayesian spatio-temporal machine learning.

## Overview

Rapid urbanisation across Sub-Saharan Africa is frequently characterised by fragmented, low-density expansion that outpaces infrastructure provision and extends into peri-urban regions. While satellite-based Earth observation datasets have enabled detailed retrospective analyses of urban growth, relatively few approaches provide spatially explicit forecasts with quantified uncertainty suitable for municipal planning.

This project develops a probabilistic framework for forecasting urban sprawl processes. Using **Yaoundé, Cameroon** as a case study, historical urban expansion is reconstructed from multi-decadal Landsat imagery before forecasting future development patterns through **2035** using a **Spatio-Temporal Sparse Variational Gaussian Process (ST-SVGP)**.

Unlike conventional deterministic forecasting approaches, this framework produces calibrated probabilistic predictions, enabling uncertainty-aware urban planning and supporting more informed land-use decision making.

---

## Objectives

- Reconstruct historical urban expansion (1990–2020) from Landsat imagery.
- Quantify urban sprawl using spatial indicators.
- Forecast future urban sprawl to 2035 using Bayesian spatio-temporal modelling.
- Quantify predictive uncertainty associated with future urban expansion.
- Develop a transferable framework for urban sprawl forecasting using Earth observation data.

---

## Methodology

### 1. Historical Urban Mapping

Annual Landsat imagery is processed to reconstruct built-up land dynamics between **1990 and 2020**.

Outputs include:

- Annual built-up extent
- Urban expansion maps
- Historical growth trajectories

---

### 2. Urban Sprawl Metrics

Spatial analysis is used to derive quantitative indicators describing different forms of urban expansion, including:

- Leapfrog development
- Spatial dispersion
- Fragmentation
- Urban density characteristics

These metrics form the response variables used for forecasting future sprawl behaviour.

---

### 3. Probabilistic Forecasting

Future urban expansion is predicted using a

> **Spatio-Temporal Sparse Variational Gaussian Process (ST-SVGP)**

which provides:

- Bayesian probabilistic predictions
- Predictive uncertainty estimates
- Scalability to long Earth observation time series
- Spatially explicit forecasts through 2035

Unlike deterministic neural networks or regression models, the ST-SVGP captures uncertainty associated with future land-use transitions, making predictions more suitable for evidence-based urban planning.

---

## Workflow

```text
Landsat Time Series (1990–2020)
            │
            ▼
 Historical Built-up Mapping
            │
            ▼
 Urban Sprawl Metrics
            │
            ▼
Feature Engineering
            │
            ▼
      ST-SVGP Model
            │
            ▼
 Probabilistic Urban Forecasts
            │
            ▼
 Uncertainty Analysis
```

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── models/
│   │   ├── st_svgp/
│   │   └── evaluation/
│   ├── analysis/
│   └── visualization/
│
├── outputs/
│   ├── forecasts/
│   ├── uncertainty/
│   ├── figures/
│   └── maps/
│
├── README.md
└── pyproject.toml
```

---


## Data Sources

### Administrative Boundaries

The primary boundary source is the Cameroon Common Operational Dataset for
Administrative Boundaries (`cod-ab-cmr`), provided by OCHA and originally
produced by the Institut National de Cartographie of Cameroon.

The seven ADM3 units corresponding to Yaoundé I–VII were selected using their
administrative P-codes and dissolved to construct the Yaoundé administrative
core. Their union was validated against the ADM2 Mfoundi boundary.

All processed geometries use `EPSG:32632`. A 5 km context buffer and a convex
hull are retained separately. The authoritative Landsat grid has a 30 m
resolution, a fixed `(0, 0)` anchor and stable global cell identifiers.

### Earth Observation

The project uses Landsat Collection 2 Tier 1 Level 2 Surface Reflectance
imagery from:

- Landsat 5 TM: `LANDSAT/LT05/C02/T1_L2`
- Landsat 7 ETM+: `LANDSAT/LE07/C02/T1_L2`
- Landsat 8 OLI/TIRS: `LANDSAT/LC08/C02/T1_L2`

The target observation epochs are:

`1990`, `1995`, `2000`, `2005`, `2010`, `2015` and `2020`.

For each epoch, a three-year diagnostic period was queried to assess scene
availability around the target year. Scene-level QA masking excludes fill,
cloud, dilated cloud, cloud shadow, snow, cirrus where applicable and
radiometric saturation. Water is retained as a valid observation.

Generated output:

- a complete Landsat scene manifest;
- scene-level valid coverage over the administrative core and context area;
- monthly availability summaries;
- candidate compositing-window comparisons;
- an epoch-level quality summary;
- the exact selected Earth Engine scene identifiers;
- a frozen compositing protocol and catalogue checksum.

The selected scene set is recorded in:

- `data/metadata/landsat/selected_scene_manifest.csv`
- `data/metadata/landsat/compositing_protocol.yaml`
- `data/metadata/landsat/catalog_version.json`

Final Landsat composites have not yet been generated. They will be constructed
during ... by loading the exact frozen scene identifiers rather than
re-querying the collections dynamically.

### Auxiliary and Spatial Predictors

The following sources are planned for Day 3 and later stages but have not yet
been integrated into the model-ready dataset:

- SRTM elevation and derived slope;
- OpenStreetMap roads and current infrastructure;
- GHSL built-up surface and population products.

GHSL will remain an auxiliary comparison and validation source, while current
OpenStreetMap data will be documented carefully because its historical
completeness varies.

### Spectral Built-up Candidates

Completed Landsat composites are converted into SAVI, MNDWI, NDBI, IBI, IBUI, VbSWIR1-BI and NDBSUI layers. Epoch-specific Otsu thresholds are estimated inside the Yaoundé administrative core and applied to the context grid.

The resulting binary maps are unvalidated candidate pseudo-labels. Missing or numerically undefined pixels remain masked and are not treated as non-built-up.
Final index selection, comparative validation and temporal correction are performed in later stages.

#### Spatial Predictors

Potential predictors include:

- Distance to roads
- Distance to urban centres
- Existing built-up areas
- Accessibility
- Topography
- Population density
- Additional environmental variables

---

## Model

### Baseline Models
**- Logistic Regression baseline**

The model is fixed to L2 Logistic Regression with `C = 0.1`, `lbfgs` and
`StandardScaler`. 

Historical evaluation uses three chronological folds:

```text
train 2000              → validate 2005
train 2000, 2005        → validate 2010
train 2000, 2005, 2010  → validate 2015
```

These folds measure temporal generalisation only. The fixed model is then
refitted on forecast origins `2000, 2005, 2010, 2015`. Forecast origin `2020`
(`2020→2025`) remains locked for the later common final evaluation.

**- XGBoost baseline**

### Primary Forecasting Model

**Spatio-Temporal Sparse Variational Gaussian Process (ST-SVGP)**

Advantages:

- Bayesian inference
- Scalable Gaussian Processes
- Probabilistic forecasting
- Spatial and temporal modelling
- Predictive uncertainty estimation

---

## Outputs

The framework generates:

- Historical urban expansion maps
- Urban sprawl indicator layers
- Probabilistic urban forecasts (2035)
- Predictive uncertainty maps
- Forecast evaluation metrics

---

## Study Area

**Yaoundé, Cameroon**

Yaoundé provides a representative example of a rapidly expanding Sub-Saharan African city experiencing dispersed urbanisation.

---

## Applications

This framework is designed to support:

- Municipal land-use planning
- Urban growth management
- Infrastructure planning
- Sustainable urban development
- Earth observation research
- Spatio-temporal forecasting research

---

## Key Features

- Multi-decadal Earth observation analysis
- Bayesian spatio-temporal forecasting
- Spatially explicit uncertainty quantification
- Scalable Gaussian Process modelling
- Transferable workflow for rapidly urbanising regions

---

## Citation

If you use this repository in academic work, please cite the associated publication once available.

---

## License

Specify an appropriate open-source license (e.g., MIT, BSD-3-Clause, or GPL-3.0) before distribution.