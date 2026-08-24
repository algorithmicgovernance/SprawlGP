# Strong Annual ST-SVGP assessment

This assessment compares only the pre-2020 annual rolling folds. No final fit, five-year final test, or locked 2020-2025 annual target was used.

## 1. Parameter stability

Maximum late-window drift changed from 18.29% at 1500 to 24.34% at 3000. The 3000 maximum is still materially drifting. Windows use their actual logged iterations and are descriptive, not an optimizer theorem.

| Fold | Temporal drift | Spatial x drift | Spatial y drift | Variance drift |
|---:|---:|---:|---:|---:|
| 1 | 24.34% | 3.90% | 5.80% | 2.61% |
| 2 | 23.78% | 6.29% | 3.34% | 5.26% |
| 3 | 24.11% | 4.72% | 6.57% | 5.64% |

## 2. Temporal identifiability

The 3000 fold temporal scales span 1.0254 to 1.05611 years with CV 1.52%.

## 3. Discrimination

| Fold | Budget | Prevalence | PR-AUC | PR-AUC/prevalence | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 0.017033 | 0.113597 | 6.66923 | 0.876366 |
| 2 | 1000 | 0.122756 | 0.231434 | 1.88531 | 0.683283 |
| 3 | 1000 | 0.0449047 | 0.11143 | 2.48148 | 0.75727 |
| 1 | 1500 | 0.017033 | 0.131343 | 7.71106 | 0.889914 |
| 2 | 1500 | 0.122756 | 0.235852 | 1.9213 | 0.713697 |
| 3 | 1500 | 0.0449047 | 0.13919 | 3.09966 | 0.794406 |
| 1 | 3000 | 0.017033 | 0.150395 | 8.8296 | 0.894496 |
| 2 | 3000 | 0.122756 | 0.294137 | 2.3961 | 0.76749 |
| 3 | 3000 | 0.0449047 | 0.175129 | 3.9 | 0.835212 |

## 4. Probability quality

| Fold | Budget | Log Loss | Brier | ECE | Probability bias |
|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 0.131808 | 0.0325864 | 0.0804316 | 0.0804316 |
| 2 | 1000 | 0.426673 | 0.108107 | 0.0629015 | -0.029389 |
| 3 | 1000 | 0.189869 | 0.0449511 | 0.0314663 | -0.00595468 |
| 1 | 1500 | 0.159569 | 0.0397696 | 0.107357 | 0.107357 |
| 2 | 1500 | 0.380779 | 0.106787 | 0.0545574 | -0.0100135 |
| 3 | 1500 | 0.16626 | 0.0438254 | 0.0215119 | 0.0204374 |
| 1 | 3000 | 0.282036 | 0.0789197 | 0.205591 | 0.205591 |
| 2 | 3000 | 0.334473 | 0.101071 | 0.0504385 | 0.0489479 |
| 3 | 3000 | 0.207326 | 0.0535837 | 0.09613 | 0.09613 |

## 5. Calibration shape

Calibration slopes are assessed with Log Loss, Brier, ECE, and bias; ECE is not used alone.

| Fold | Slope 1000 | Slope 1500 | Slope 3000 | Intercept 3000 |
|---:|---:|---:|---:|---:|
| 1 | 1.07138 | 1.32911 | 1.81888 | -2.94297 |
| 2 | 0.294331 | 0.408612 | 0.851733 | -0.632543 |
| 3 | 0.401194 | 0.722335 | 1.23394 | -1.13031 |

## 6. Uncertainty

Mondrian positive coverage is compared with the 1000 run together with average set size and both-label cost. Fold 1 has no strictly earlier OOF calibration fold.

| Fold | Budget | Marginal coverage | Marginal positive | Mondrian coverage | Mondrian positive | Mondrian negative | Average size | Both labels | Empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1000 | 0.731437 | 0.000173868 | 0.768681 | 0.398505 | 0.820481 | 0.995123 | 0 | 0.00487696 |
| 3 | 1000 | 0.914954 | 0.0013885 | 0.924606 | 0.695918 | 0.935358 | 1.26125 | 0.261248 | 0 |
| 2 | 1500 | 0.74102 | 0.000608537 | 0.778083 | 0.380857 | 0.833668 | 0.981623 | 0 | 0.0183766 |
| 3 | 1500 | 0.912335 | 0.00666482 | 0.920117 | 0.818384 | 0.9249 | 1.29463 | 0.29463 | 0 |
| 2 | 3000 | 0.789661 | 0.0278188 | 0.821506 | 0.330262 | 0.890247 | 0.971571 | 0 | 0.0284293 |
| 3 | 3000 | 0.90019 | 0.0211052 | 0.90433 | 0.848653 | 0.906947 | 1.23245 | 0.232455 | 0 |

## 7. ELBO

| Fold | Earlier-late mean | Final mean | Final SD | All finite |
|---:|---:|---:|---:|---:|
| 1 | -196292 | -203170 | 17271.5 | True |
| 2 | -224821 | -226082 | 17430.3 | True |
| 3 | -260554 | -268202 | 21833.2 | True |
All finite values indicate stochastic ELBO noise rather than numerical divergence.

## 8. Fold robustness

The decision checks whether PR-AUC gains occur in at least two folds and whether difficult Fold 2 avoids joint material PR-AUC and ROC-AUC degradation. Fold 2 PR-AUC is 0.231434 -> 0.235852 -> 0.294137; ROC-AUC is 0.683283 -> 0.713697 -> 0.76749.

## Decision evidence

- Parameter Stability: False
- Temporal Identifiability: True
- Discrimination: True
- Probability Quality: False
- Uncertainty: True
- Fold Robustness: True
- Finite Elbo: True

ANNUAL_ST_SVGP_PROMISING_BUT_NOT_STRONG
