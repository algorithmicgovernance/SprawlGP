# Landsat composites and auxiliary-source integration

## Landsat outputs

| Epoch | Sensors | Scenes | Covered grid | Median observations | Status |
|---:|---|---:|---:|---:|:---:|
| 2005 | {"LE07": 2} | 2 | 9.89% | 0.00 | PASS |
| 2010 | {"LE07": 1} | 1 | 23.92% | 0.00 | PASS |
| 2015 | {"LC08": 2} | 2 | 46.64% | 0.00 | PASS |
| 2020 | {"LC08": 1} | 1 | 29.57% | 0.00 | PASS |
| 2025 | {"LC08": 1} | 1 | 0.52% | 0.00 | PASS |

## Auxiliary sources

| Source | Spatial representation | Temporal representation | Intended role |
|---|---|---|---|
| SRTM | exact_project_grid | static_approximately_2000 | elevation_and_slope_predictors |
| GHSL | native_100m_grid | seven_epochs | auxiliary_benchmark_and_population |
| OpenStreetMap | vector_EPSG32632 | current_snapshot | recent_validation_and_accessibility |

## Known limitations

- Landsat 7 SLC-off gaps are filled only by valid observations from other dates.
- Mixed-sensor epochs are documented through sensor-specific count bands.
- GHSL remains a modelled auxiliary source on its native 100 m grid.
- OpenStreetMap is a contemporary snapshot and not historical evidence.
- SRTM represents terrain observed approximately around the year 2000.

No spectral index, Otsu threshold or built-up label was generated during Day 3.
