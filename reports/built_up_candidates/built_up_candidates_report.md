# Spectral indices and Otsu built-up candidates

## Output summary

- Completed epochs: 2005, 2010, 2015, 2020, 2025
- Epoch count: 5
- Continuous index layers: 35
- Binary candidate maps: 25
- Epoch-specific Otsu thresholds: 25

The binary maps are unvalidated candidate pseudo-labels. No index has been selected as the final historical built-up mapping method.

## Thresholds

| Epoch | Index | Threshold | Valid pixels | Built fraction | Status |
|---:|---|---:|---:|---:|:---:|
| 2005 | ibi | -10299.646149 | 19571 | 0.9999 | PASS |
| 2005 | ibui | -0.171993 | 19571 | 0.2248 | PASS |
| 2005 | ndbi | -0.226702 | 19571 | 0.2641 | PASS |
| 2005 | ndbsui | -0.119221 | 19571 | 0.3520 | PASS |
| 2005 | vbswir1_bi | 0.437588 | 19571 | 0.7116 | PASS |
| 2010 | ibi | -2175.721497 | 87129 | 0.9999 | PASS |
| 2010 | ibui | -0.046867 | 87129 | 0.4162 | PASS |
| 2010 | ndbi | -0.148432 | 87129 | 0.4430 | PASS |
| 2010 | ndbsui | -0.064452 | 87129 | 0.4841 | PASS |
| 2010 | vbswir1_bi | 0.589782 | 87129 | 0.5198 | PASS |
| 2015 | ibi | -158813.410156 | 191830 | 1.0000 | PASS |
| 2015 | ibui | -0.054694 | 191831 | 0.4849 | PASS |
| 2015 | ndbi | -0.140577 | 191831 | 0.5058 | PASS |
| 2015 | ndbsui | -0.066386 | 191794 | 0.5882 | PASS |
| 2015 | vbswir1_bi | 0.991259 | 191831 | 0.0093 | PASS |
| 2020 | ibi | -62098.378906 | 100467 | 1.0000 | PASS |
| 2020 | ibui | -0.000003 | 100468 | 0.4665 | PASS |
| 2020 | ndbi | -0.113236 | 100468 | 0.5041 | PASS |
| 2020 | ndbsui | -0.054702 | 100451 | 0.5619 | PASS |
| 2020 | vbswir1_bi | -46.593643 | 100468 | 1.0000 | PASS |
| 2025 | ibi | 4715.648560 | 3356 | 0.0006 | PASS |
| 2025 | ibui | 0.202918 | 3356 | 0.6518 | PASS |
| 2025 | ndbi | 0.000176 | 3356 | 0.7176 | PASS |
| 2025 | ndbsui | -0.016548 | 3356 | 0.7590 | PASS |
| 2025 | vbswir1_bi | 0.726627 | 3356 | 0.3855 | PASS |

## Interpretation limits

- Thresholds are estimated independently for each epoch.
- Candidate-area changes are quality diagnostics, not validated growth.
- Invalid index pixels remain masked and are never recoded as non-built.
- No temporal persistence correction or transition map is applied.
- Comparative validation and final method selection belong to the next stage.
