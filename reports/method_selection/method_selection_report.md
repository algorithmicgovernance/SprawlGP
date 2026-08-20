# Day 5 — Minimal method selection

Selection status: **PASS**

Selected method: **ndbi**

## Anchor-year weighted F1

| Epoch | Method | Weighted F1 |
|---:|---|---:|
| 2000 | ibui | 0.5401 |
| 2000 | ndbi | 0.5401 |
| 2000 | ndbsui | 0.4420 |
| 2000 | vbswir1_bi | 0.0962 |
| 2005 | ibui | 1.0000 |
| 2005 | ndbi | 1.0000 |
| 2005 | ndbsui | 1.0000 |
| 2005 | vbswir1_bi | 0.2357 |
| 2010 | ibui | 0.9853 |
| 2010 | ndbi | 0.9853 |
| 2010 | ndbsui | 0.9853 |
| 2010 | vbswir1_bi | 0.6245 |
| 2015 | ibui | 1.0000 |
| 2015 | ndbi | 1.0000 |
| 2015 | ndbsui | 1.0000 |
| 2015 | vbswir1_bi | 0.0417 |
| 2020 | ibui | 0.8728 |
| 2020 | ndbi | 0.8728 |
| 2020 | ndbsui | 0.8728 |
| 2020 | vbswir1_bi | 0.1728 |
| 2025 | ibui | 0.9175 |
| 2025 | ndbi | 0.9175 |
| 2025 | ndbsui | 0.8638 |
| 2025 | vbswir1_bi | 0.2824 |

## Final ranking

| Rank | Method | Mean yearly weighted F1 | Minimum yearly weighted F1 | Reversal rate | Selected |
|---:|---|---:|---:|---:|:---:|
| 1 | ndbi | 0.8859 | 0.5401 | 0.0946 | true |
| 2 | ibui | 0.8859 | 0.5401 | 0.1055 | false |
| 3 | ndbsui | 0.8606 | 0.4420 | 0.0620 | false |
| 4 | vbswir1_bi | 0.2422 | 0.0417 | 0.4285 | false |

## External diagnostics

GHSL is used only as a sample-level diagnostic.

| Method | GHSL mean where predicted built | GHSL mean where predicted non-built |
|---|---:|---:|
| ndbi | 0.3368 | 0.0647 |
| ibui | 0.3365 | 0.0669 |
| vbswir1_bi | 0.2288 | 0.1698 |
| ndbsui | 0.3432 | 0.0604 |

### Current OSM

Sample-level recall among current OSM-building cells:

- ndbi: nan
- ibui: nan
- vbswir1_bi: nan
- ndbsui: nan

Current OSM completeness is uneven, and the snapshot is not historical evidence for earlier epochs.

## Limitations

- Metrics apply only to valid Landsat spatial support.
- Historical reference-image quality varies by epoch.
- GHSL is an auxiliary benchmark rather than ground truth.
- Current OSM is not historical evidence.
- Temporal correction is deferred to Day 6.
