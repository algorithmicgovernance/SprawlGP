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

### Earth Observation

- Landsat annual imagery (1990–2020)

### Spatial Predictors

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