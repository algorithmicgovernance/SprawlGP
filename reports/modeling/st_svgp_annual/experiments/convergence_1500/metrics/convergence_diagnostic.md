# Annual ST-SVGP 1500-iteration convergence diagnostic

This is a convergence diagnostic only; no hyperparameter was selected or tuned using the locked block.
The locked 2020-2025 target block was not evaluated, and no final fit was run.

## Descriptive conventions

Late-window relative parameter drift is the absolute difference between the 1000-1200 and 1300-1500 window means, divided by the earlier-window mean, using actual logged iterations. Drift <=1% is described as practically plateaued, 1-5% as slowly stabilizing, and >5% as strongly drifting. These are descriptive review thresholds, not an optimizer convergence theorem.
Absolute fold-level changes of 0.02 for PR-AUC/ROC-AUC and probability metrics, and 0.10 for calibration slope, are used only to flag material validation changes for review.

## Final parameters and validation metrics

| Fold | Temporal years (1000 / 1500) | Spatial x km (1000 / 1500) | Spatial y km (1000 / 1500) | Variance (1000 / 1500) | PR-AUC (1000 / 1500) | ROC-AUC (1000 / 1500) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.0912082 / 2.2317189 | 3.7043316 / 4.1202128 | 3.7566117 / 4.1843217 | 0.5995791 / 0.5848829 | 0.1135972 / 0.1313428 | 0.87636593 / 0.88991359 |
| 2 | 3.0857232 / 2.1978924 | 3.6494439 / 4.1493528 | 3.6259811 / 3.8985283 | 0.62027933 / 0.61874286 | 0.23143372 / 0.23585177 | 0.68328284 / 0.71369658 |
| 3 | 3.0284456 / 2.1672548 | 3.6549439 / 4.0816835 | 3.6141345 / 3.9481405 | 0.65373028 / 0.67048469 | 0.11143036 / 0.13918955 | 0.75726976 / 0.79440625 |

| Fold | Log Loss (1000 / 1500) | Brier (1000 / 1500) | ECE (1000 / 1500) | Calibration slope (1000 / 1500) | Probability bias (1000 / 1500) | Mean latent variance (1000 / 1500) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.13180801 / 0.15956856 | 0.032586384 / 0.039769639 | 0.080431565 / 0.10735735 | 1.0713816 / 1.3291105 | 0.080431565 / 0.10735735 | 0.12336534 / 0.18462802 |
| 2 | 0.42667349 / 0.38077855 | 0.10810697 / 0.1067875 | 0.062901494 / 0.054557397 | 0.294331 / 0.40861169 | -0.029389041 / -0.010013542 | 0.12833156 / 0.20209167 |
| 3 | 0.18986938 / 0.16626018 | 0.04495108 / 0.043825423 | 0.031466316 / 0.021511889 | 0.40119408 / 0.72233454 | -0.0059546805 / 0.020437415 | 0.15283754 / 0.22878845 |

## Late-window parameter drift

| Fold | Actual earlier window | Actual final window | Temporal | Spatial x | Spatial y | Variance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1000-1200 | 1300-1499 | 17.7686% | 5.9686% | 7.2091% | 1.3010% |
| 2 | 1000-1200 | 1300-1499 | 18.2852% | 8.2466% | 3.9729% | 0.0579% |
| 3 | 1000-1200 | 1300-1499 | 18.2179% | 7.3097% | 4.9006% | 1.6229% |

## Stochastic ELBO window summary

| Fold | Earlier late-window mean | Final-window mean | Final-window SD | Final min / max |
|---:|---:|---:|---:|---:|
| 1 | -212210.47 | -198128.02 | 10114.466 | -209619.8 / -188564.74 |
| 2 | -243613.64 | -241955.94 | 27914.591 | -280618.6 / -210562.2 |
| 3 | -261865.34 | -244047.38 | 27326.493 | -271441.17 / -201687.02 |

## Answers

1. **Temporal movement after iteration 1000.** The maximum fold-level late-window drift is 18.2852%, classified as still strongly drifting.
2. **Spatial movement after iteration 1000.** The maximum x/y late-window drift is 8.2466%, classified as still strongly drifting.
3. **Common temporal scale across folds.** Extended temporal lengthscales range from 2.1672548 to 2.2317189 years (CV 1.4664%); they remain similar.
4. **Validation discrimination.** The largest absolute PR-AUC/ROC-AUC change is 0.037136488; discrimination materially changed.
5. **Probability quality.** Maximum absolute changes are log_loss=0.045894942, brier_score=0.0071832553, ece=0.026925788, calibration_slope=0.32114046, probability_bias=0.026925788; probability quality materially changed.
6. **Stochastic ELBO stability.** All late-window ELBO values are finite. The reported variability is treated as minibatch noise; there is no non-finite evidence of numerical divergence.
7. **Practical stabilization at 1500 iterations.** The descriptive maximum parameter drift is 18.2852%; the status below records the evidence without changing the canonical iteration count.

PARAMETERS_NOT_STABILIZED
