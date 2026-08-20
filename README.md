# Probabilistic Forecasting of Urban Sprawl

A probabilistic geospatial modelling framework for forecasting urban sprawl dynamics from Earth observation data using Bayesian spatio-temporal machine learning.

## Overview

Rapid urbanisation across Sub-Saharan Africa is frequently characterised by fragmented, low-density expansion that outpaces infrastructure provision and extends into peri-urban regions. While satellite-based Earth observation datasets have enabled detailed retrospective analyses of urban growth, relatively few approaches provide spatially explicit forecasts with quantified uncertainty suitable for municipal planning.

This project develops a probabilistic framework for forecasting urban sprawl processes. Using **Yaoundé, Cameroon** as a case study, historical urban expansion is reconstructed from multi-decadal Landsat imagery before forecasting future development patterns through **2035** using a **Spatio-Temporal Sparse Variational Gaussian Process (ST-SVGP)**.

Unlike conventional deterministic forecasting approaches, this framework produces calibrated probabilistic predictions, enabling uncertainty-aware urban planning and supporting more informed land-use decision making.

---

## Objectives

- Reconstruct historical urban expansion (2000–2025) from Landsat imagery.
- Quantify urban sprawl using spatial indicators.
- Forecast future urban sprawl to 2035 using Bayesian spatio-temporal modelling.
- Quantify predictive uncertainty associated with future urban expansion.
- Develop a transferable framework for urban sprawl forecasting using Earth observation data.

---

## Methodology

### 1. Historical Urban Mapping

Multi-epoch Landsat imagery is processed to reconstruct built-up land dynamics at the target epochs **2000, 2005, 2010, 2015, 2020 and 2025**.

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
Landsat Time Series (2000–2025)
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

`2000`, `2005`, `2010`, `2015`, `2020` and `2025`.

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

### Built-up mapping validation

The operational built-up mapping method was validated using 270 manually
reviewed samples distributed across the six epochs (2000, 2005, 2010, 2015,
2020 and 2025). Of these samples, 205 received a certain built/non-built label
and 65 were retained as uncertain and excluded from the accuracy calculation.

Four candidate methods were evaluated using the predefined design-weighted
validation protocol:

- NDBI
- IBUI
- NDBSUI
- VbSWIR1-BI

NDBI and IBUI produced identical manual-validation performance:

| Method | Mean yearly weighted F1 | Minimum yearly weighted F1 | Reversal rate |
|---|---:|---:|---:|
| NDBI | 0.8859 | 0.5401 | **0.0946** |
| IBUI | 0.8859 | 0.5401 | 0.1055 |
| NDBSUI | 0.8606 | 0.4420 | **0.0620** |
| VbSWIR1-BI | 0.2422 | 0.0417 | 0.4285 |

Because NDBI and IBUI are tied on both the primary metric and the first
tie-breaker, the predefined temporal-consistency tie-break selects NDBI.

A **temporal reversal** is a cell classified as built at one epoch but
classified as non-built at the following epoch, considering only locations
that are valid at both dates. Since established built-up land is expected to
be largely persistent over five-year periods, a lower reversal rate indicates
better temporal consistency.

NDBI is therefore retained and frozen as the built-up mapping method for the
current 30 m dataset. The difference with IBUI is small; the selection should
be interpreted as a marginal preference based on temporal consistency rather
than evidence that NDBI is substantially more accurate.

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

The retained baseline uses chronological rolling validation and the shared
model-ready feature family. The currently retained feature specification
includes the built-fraction × recent-growth interaction
`built_fraction_x_recent_growth_t`.


**- XGBoost baselineis deferred** and is not part of the current retained modelling
comparison.

**Sparse Variational Gaussian Process (SVGP)**

The selected SVGP comparator keeps the Adam-based Bernoulli-probit SVGP
implementation with `M_s = 64` spatial inducing locations. Controlled feature
experiments compared:

- `base`;
- `log_distance`;
- `log_distance_growth`.

`log_distance_growth` is retained because it gives small but coherent
pre-test gains in Log Loss, Brier score, PR-AUC and calibration error. Its
linear mean includes `log_distance_to_built_m_t` and
`built_fraction_x_recent_growth_t`.

A separate Natural-Gradient SVGP implementation is preserved as a historical
experiment because it did not outperform the selected Adam SVGP.

### Primary Forecasting Model

**Spatio-Temporal Sparse Variational Gaussian Process (ST-SVGP)**

The current promoted pre-test candidate uses:

- 64 fixed spatial inducing locations;
- anisotropic spatial Matérn-3/2 covariance;
- temporal Matérn-3/2 Markov state-space representation;
- Bernoulli-probit likelihood;
- dense Gaussian CVI pseudo-sites updated with Natural Gradient;
- Adam updates for the parametric mean and kernel hyperparameters;
- sequential Kalman filtering and RTS smoothing.

The temporal lengthscale is initialised at **1.5 five-year steps** and remains
trainable. The free-init-1.0 and fixed-1.5 variants are retained as diagnostics,
not as the promoted model.

Historical evaluation uses three chronological folds:

```text
train 2000              → validate 2005
train 2000, 2005        → validate 2010
train 2000, 2005, 2010  → validate 2015
```

Forecast origin `2020` (`2020→2025`) remains locked and is not used for model
selection. The current common OOF evaluator compares Logistic Regression,
Strong SVGP and promoted ST-SVGP on proper scoring rules, discrimination,
calibration and temporal prediction-set coverage.

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

<!-- BEGIN ST-SVGP SUMMARY -->
### Sparse variational spatio-temporal Gaussian process (ST-SVGP)

The current ST-SVGP candidate models five-year non-built-to-built conversion as
a Bernoulli-probit process with a nine-feature parametric mean and a separable
Matérn-3/2 residual Gaussian process. Spatial dependence is represented by 64
fixed inducing locations, while the temporal Matérn-3/2 kernel is written in
Markov state-space form and inferred with sequential Kalman filtering and RTS
smoothing. Non-conjugate inference uses dense time-specific Gaussian CVI
pseudo-sites updated by Natural Gradient; Adam updates the linear mean and
kernel hyperparameters.

The promoted candidate is configured in `configs/modeling/st_svgp.yaml`. Its
temporal lengthscale is initialised at 1.5 five-year steps and remains
trainable. Diagnostic and mathematical-validation configurations remain under
`configs/modeling/st_svgp/`. The 2020 -> 2025 period remains locked until the
pre-test protocol and model choices are frozen.

See `docs/modeling_st_svgp.md` for the mathematical validation chain, the 37
retained tests, rolling validation and retained diagnostics.
<!-- END ST-SVGP SUMMARY -->
