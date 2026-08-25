# Strong SVGP baseline — implementation notes

## Current strategy

1. Keep the already working SVGP-Adam as the reproducible reference.
2. Add a CVI-aligned NaturalGradient inference mode to the same model.
3. Keep M_s=64 fixed because the 64/128/512 sensitivity did not materially
   change validation performance.
4. Run a small manual tuning set.
5. Evaluate probability quality, ranking, calibration and temporal robustness.
6. Promote one configuration to `svgp_experiment.yaml` and keep one SVGP only.

## Calibration vs coverage

Coverage is not implemented.

Calibration for binary probabilities means frequency agreement. If many cells
are assigned p≈0.8, approximately 80% of those cells should convert in a
well-calibrated model. This is not the same as 80% classification accuracy.

The supervisor's phrase about an "80% confidence interval" could refer to
posterior interval coverage or prediction-set coverage. That definition must
be clarified before coding it.

## Natural Gradient / CVI alignment

The SVGP retains the Hensman ELBO. `method=natgrad` uses:

- Adam warm-up for q(u), kernel and linear mean;
- GPflow `NaturalGradient` + `XiSqrtMeanVar` for q_mu/q_sqrt;
- Adam for kernel and mean after warm-up;
- conservative gamma backtracking if q(u) becomes numerically invalid.

This is CVI-aligned because Hamelijnck et al. Eqs. 6–8 show the connection
between Gaussian variational natural gradients and Gaussian approximate
likelihood factors. The explicit pseudo-likelihood sites are not constructed
in standard GPflow SVGP; they will be explicit in ST-SVGP filtering/smoothing.

## Feature sets

`base`
- existing eight covariates.

`nonlinear_core`
- replaces raw distance by log1p(distance);
- adds slope squared.

`nonlinear_interactions`
- nonlinear_core;
- built fraction × recent local growth;
- built fraction × log population.

No additional dataset is required.

## Tuning selection

No weighted score.

A candidate must first satisfy:

`mean PR-AUC >= working SVGP-Adam PR-AUC - 0.01`

Among eligible models:

1. lower mean Log Loss;
2. lower Brier;
3. lower ECE.

This preserves the performance–calibration trade-off without hiding it inside
an arbitrary composite metric.
