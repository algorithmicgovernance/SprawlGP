# Landsat composites and auxiliary-source integration

## Landsat outputs

| Epoch | Sensors | Scenes | Covered grid | Median observations | Status |
|---:|---|---:|---:|---:|:---:|
| 2000 | {"LE07": 6} | 6 | 89.59% | 1.00 | PASS |
| 2005 | {"LE07": 6} | 6 | 90.05% | 2.00 | PASS |
| 2010 | {"LE07": 12} | 12 | 98.65% | 3.00 | PASS |
| 2015 | {"LC08": 7, "LE07": 9} | 16 | 96.73% | 4.00 | PASS |
| 2020 | {"LC08": 12} | 12 | 99.89% | 6.00 | PASS |
| 2025 | {"LC08": 10, "LC09": 9} | 19 | 99.89% | 7.00 | PASS |

## Auxiliary sources

| Source | Spatial representation | Temporal representation | Intended role |
|---|---|---|---|
| SRTM | exact_project_grid | static_approximately_2000 | elevation_and_slope_predictors |
| GHSL | native_100m_grid | 8_configured_epochs | auxiliary_benchmark_and_population |
| OpenStreetMap | vector_EPSG32632 | current_snapshot | recent_validation_and_accessibility |

## Known limitations

- Landsat 7 SLC-off gaps are filled only by valid observations from other dates.
- Mixed-sensor epochs are documented through sensor-specific count bands.
- GHSL remains a modelled auxiliary source on its native 100 m grid.
- OpenStreetMap is a contemporary snapshot and not historical evidence.
- SRTM represents terrain observed approximately around the year 2000.

No spectral index, Otsu threshold or built-up label was generated during Day 3.
