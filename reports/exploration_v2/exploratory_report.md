# Yaoundé Urban Expansion Dataset V1 — Exploratory Report

## Release

- Dataset: `yaounde_urban_expansion_30m_v2`
- Version: `2.0.0-provisional`
- Release status: `FROZEN_PROVISIONAL_RELEASE`
- Mapping selection status: `PROVISIONAL_PENDING_MANUAL_VALIDATION`
- Mapping method: `NDBI`

## Dataset scale

- Cell-time rows: 720,605
- Unique cells: 210,842
- Five-year transitions: 5
- Tracking epochs: 6

## Transition summary

| Transition | Eligible cells | Converted cells | New built-up area (ha) | Pairwise support (%) |
|---|---:|---:|---:|---:|
| 2000_2005 | 197,260 | 30,314 | 2,728.26 | 93.84 |
| 2005_2010 | 168,945 | 33,774 | 3,039.66 | 94.35 |
| 2010_2015 | 140,194 | 14,414 | 1,297.26 | 96.84 |
| 2015_2020 | 127,706 | 44,471 | 4,002.39 | 98.08 |
| 2020_2025 | 86,500 | 5,314 | 478.26 | 99.89 |

## Tracking summary

| Epoch | Built area (ha) | PBA | NUMP | MPS (ha) |
|---:|---:|---:|---:|---:|
| 2000 | 12,494.52 | 0.1463 | 4,079 | 3.0631 |
| 2005 | 15,922.08 | 0.1864 | 4,625 | 3.4426 |
| 2010 | 20,901.51 | 0.2448 | 5,348 | 3.9083 |
| 2015 | 24,021.99 | 0.2813 | 6,357 | 3.7788 |
| 2020 | 36,681.39 | 0.4295 | 5,191 | 7.0663 |
| 2025 | 38,197.08 | 0.4473 | 5,392 | 7.0840 |

## Figures

### 01 Tracking Metrics

![01_tracking_metrics](figures/01_tracking_metrics.png)

### 02 Historical Demand

![02_historical_demand](figures/02_historical_demand.png)

### 03 Annualised Conversion Rate

![03_annualised_conversion_rate](figures/03_annualised_conversion_rate.png)

### 04 Population Vs Expansion

![04_population_vs_expansion](figures/04_population_vs_expansion.png)

### 06 Pairwise Support

![06_pairwise_support](figures/06_pairwise_support.png)

### 07 Transition Quality

![07_transition_quality](figures/07_transition_quality.png)

### 08 Observation Count Distribution

![08_observation_count_distribution](figures/08_observation_count_distribution.png)

### 09 Predictor Violins

![09_predictor_violins](figures/09_predictor_violins.png)

### 10 Predictor Correlation

![10_predictor_correlation](figures/10_predictor_correlation.png)

### 11 Exploratory Feature Importance

![11_exploratory_feature_importance](figures/11_exploratory_feature_importance.png)

### 12 Distance To Built Map

![12_distance_to_built_map](figures/12_distance_to_built_map.png)

### 13 Neighbourhood Density Map

![13_neighbourhood_density_map](figures/13_neighbourhood_density_map.png)

### 14 Expansion And Elevation

![14_expansion_and_elevation](figures/14_expansion_and_elevation.png)

### 15 Grid 30M Detail

![15_grid_30m_detail](figures/15_grid_30m_detail.png)

### 16 Final State Gallery

![16_final_state_gallery](figures/16_final_state_gallery.png)

### 17 Transition Gallery

![17_transition_gallery](figures/17_transition_gallery.png)

## Interpretation limits

- NDBI remains provisional pending expert manual validation.
- The cell-time dataset contains eligible non-built cells, not all cells in the city.
- Historical demand is measured on pairwise common-valid spatial support.
- Tracking metrics are measured on the fixed all-epoch common support.
- Exploratory feature importance is not a causal analysis and is not the final forecasting model.
