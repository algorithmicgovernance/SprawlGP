# Spectral indices and Otsu built-up candidates

## Output summary

- Completed epochs: 2000, 2005, 2010, 2015, 2020, 2025
- Epoch count: 6
- Continuous index layers: 42
- Binary candidate maps: 24
- Epoch-specific Otsu thresholds: 24

The binary maps are unvalidated candidate pseudo-labels. No index has been selected as the final historical built-up mapping method.

## Thresholds

| Epoch | Index | Threshold | Valid pixels | Built fraction | Status |
|---:|---|---:|---:|---:|:---:|
| 2000 | ibui | -0.039087 | 316189 | 0.3151 | PASS |
| 2000 | ndbi | -0.148485 | 316189 | 0.3408 | PASS |
| 2000 | ndbsui | -0.072257 | 316189 | 0.3920 | PASS |
| 2000 | vbswir1_bi | 0.628928 | 316189 | 0.4933 | PASS |
| 2005 | ibui | 0.046903 | 307858 | 0.3627 | PASS |
| 2005 | ndbi | -0.109358 | 307858 | 0.3965 | PASS |
| 2005 | ndbsui | -0.066416 | 307858 | 0.4249 | PASS |
| 2005 | vbswir1_bi | 0.601527 | 307858 | 0.4934 | PASS |
| 2010 | ibui | 0.062526 | 317920 | 0.5039 | PASS |
| 2010 | ndbi | -0.078144 | 317920 | 0.5136 | PASS |
| 2010 | ndbsui | -0.044926 | 317920 | 0.4748 | PASS |
| 2010 | vbswir1_bi | 0.500051 | 317920 | 0.7413 | PASS |
| 2015 | ibui | -0.039078 | 316089 | 0.4953 | PASS |
| 2015 | ndbi | -0.132792 | 316089 | 0.5124 | PASS |
| 2015 | ndbsui | -0.050799 | 316089 | 0.5223 | PASS |
| 2015 | vbswir1_bi | 0.624314 | 316089 | 0.3708 | PASS |
| 2020 | ibui | 0.148489 | 321917 | 0.6900 | PASS |
| 2020 | ndbi | -0.039066 | 321917 | 0.6994 | PASS |
| 2020 | ndbsui | -0.031242 | 321917 | 0.6375 | PASS |
| 2020 | vbswir1_bi | 0.617100 | 321917 | 0.3574 | PASS |
| 2025 | ibui | 0.078073 | 321917 | 0.6119 | PASS |
| 2025 | ndbi | -0.082012 | 321917 | 0.6236 | PASS |
| 2025 | ndbsui | -0.042983 | 321917 | 0.6531 | PASS |
| 2025 | vbswir1_bi | 0.640531 | 321917 | 0.3969 | PASS |

## Interpretation limits

- Thresholds are estimated independently for each epoch.
- Candidate-area changes are quality diagnostics, not validated growth.
- Invalid index pixels remain masked and are never recoded as non-built.
- No temporal persistence correction or transition map is applied.
- IBI remains in continuous assets for transparency but is excluded from candidate generation and Day 5 method comparison.
- Comparative validation and final method selection belong to the next stage.
