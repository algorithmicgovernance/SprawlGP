# Yaoundé Urban Expansion 30 m — Version 1

## Purpose

This dataset supports historical urban-expansion tracking and five-year
cell-level built-up conversion modelling for Yaoundé, Cameroon.

## Release status

- Release: `2.0.0-provisional`
- Status: `FROZEN_PROVISIONAL_RELEASE`
- Mapping method: `ndbi`
- Mapping selection: `VALIDATED`
- Manual validation complete: `true`

## Spatial and temporal coverage

- CRS: EPSG:32632
- Resolution: 30 m
- Epochs: 2000, 2005, 2010, 2015, 2020, 2025
- Transitions: 2000–2005, 2005–2010, 2010–2015, 2015–2020, 2020–2025
- Tracking support: 294.75% of the administrative core

## Main table

`cell_time_dataset.parquet` contains one row for each pairwise-valid cell that
is non-built at the forecast origin. It contains 720,605 rows, including
128,287 observed conversions.

All predictors are measured at the forecast origin. Recent growth uses only
the preceding transition. No modelling partition is embedded in the release.
GHSL population for 2025 is used only in the historical demand summary and is
a projected epoch.

## Tracking

`urban_sprawl_metrics.csv` reports built area, PBA, NUMP and MPS on one fixed
all-epoch valid support. `historical_demand.csv` reports observed conversion
on pairwise common-valid support.

## Limitations

NDBI has been confirmed by manual validation as the mapping method for this
release. NDBI and IBUI had identical weighted-F1 performance, with NDBI
retained because of its slightly lower temporal reversal rate. The overall
dataset release remains provisional while the temporal-transition, tracking
support and population-aggregation audits are completed.
