# Modelling Logistic Regression

## Branch

```bash
git switch feature/data-pipeline
git pull --ff-only origin feature/data-pipeline
git switch -c feature/modeling-pipeline
```

## Frozen split

| Role | Forecast origins | Transition |
|---|---|---|
| Train | 2000, 2005 | 2000→2005, 2005→2010 |
| Validation | 2010 | 2010→2015 |
| Calibration | 2015 | 2015→2020 |
| Final test | 2020 | 2020→2025 |

Day 1 selects the L2 strength on validation and saves raw validation
probabilities. After selection, the saved model is refitted on forecast origins
2000, 2005 and 2010 only. Calibration and final-test outcomes are not used.

## Minimal predictors

- `ndbi_t`
- `distance_to_built_m_t`
- `built_fraction_11x11_t`
- `recent_local_growth_5y_t`
- `elevation_m`
- `slope_degrees`
- `x_center_m`
- `y_center_m`
- `forecast_origin`

Population, current OSM, target-year variables, `transition_quality` and
persistence-correction fields are excluded.

## Run

Append `Makefile.modeling_day1.inc` to the repository Makefile, append
`.gitignore.modeling.inc` to `.gitignore`, then run:

```bash
make modeling-baseline
```

Expected outputs:

```text
data/metadata/modeling/
├── feature_manifest.yaml
└── split_manifest.csv

artifacts/models/logistic_v1/
├── model.joblib
└── model_card.json

reports/modeling_v1/
├── metrics/
│   ├── logistic_coefficients.csv
│   ├── logistic_search.csv
│   └── logistic_validation.json
└── predictions/
    └── logistic_validation_predictions.parquet
```
