# Urban-expansion feature engineering

## Scope

This module starts the feature-engineering phase after the SVGP inference
choice has been frozen.

The goal is to improve predictive performance and probability quality without
introducing additional external datasets.

The selected SVGP remains unchanged during feature construction.

## Baseline

The baseline predictor set is:

```text
ndbi_t
savi_t
distance_to_built_m_t
built_fraction_11x11_t
recent_local_growth_5y_t
elevation_m
slope_degrees
log_population_density_t
```

## Candidate set 1 — nonlinear core

The first controlled experiment changes only two assumptions.

### Log-distance to existing built-up land

```text
log_distance_to_built_m_t = log(1 + distance_to_built_m_t)
```

Rationale: a 100-m increase close to existing urban land need not have the same
effect as a 100-m increase several kilometres away.

### Squared slope

```text
slope_squared_t = slope_degrees²
```

Rationale: the terrain constraint may be nonlinear.

The raw `slope_degrees` term is retained alongside the squared term.

## Candidate set 2 — interactions

This set builds on `nonlinear_core`.

### Built fraction × recent local growth

```text
built_fraction_x_recent_growth_t
```

Tests whether recent local expansion behaves differently in already urbanised
neighbourhoods.

### Built fraction × log population density

```text
built_fraction_x_log_population_t
```

Tests whether population pressure depends on existing local urbanisation.

## Explicit exclusions

The following are deliberately excluded from this first feature-engineering
phase:

- new external datasets;
- x/y coordinates in the parametric linear mean;
- forecast origin in the parametric linear mean;
- `NDBI - SAVI`, because with NDBI and SAVI already present it is an exact
  linear combination in a linear mean;
- multi-scale neighbourhood features that require rebuilding the cell-time
  dataset;
- temporal-difference features that require redesigning the earliest origin.

## Experimental order

Use the selected SVGP-Adam runtime and inference unchanged:

```text
M_s       = 64
inference = Adam
device    = CPU
float     = float32
```

Then compare:

```text
base
  ↓
nonlinear_core
  ↓
interactions
```

Only one feature-set change should be evaluated at a time.

Do not modify the temporal kernel or Natural-Gradient settings during this
feature experiment.
