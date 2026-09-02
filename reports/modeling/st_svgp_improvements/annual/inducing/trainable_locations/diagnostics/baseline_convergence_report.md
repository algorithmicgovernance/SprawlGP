# ST-SVGP baseline convergence diagnostics

## Scope

This is a read-only analysis of existing histories. No model was trained, no hyperparameter was changed, and no final or locked test was evaluated.

## Inputs

- `TREN-A01_time_trend`: `reports/modeling/st_svgp_improvements/annual/temporal/time_trend/metrics/st_svgp_training_history.csv`; SHA-256 `9d80d7cff98dd48786b25ae493b64506ddbef17260a461a2d9f7b862188b7632`
- `IND-A01_trainable_inducing_locations`: `reports/modeling/st_svgp_improvements/annual/inducing/trainable_locations/metrics/st_svgp_training_history.csv`; SHA-256 `154d5e6699aecb9fdb24fee8bca32a86533ae50e57ded0d9f3389914286ffb1b`

## Annual baselines

## Five-year baseline

## Diagnostic observations

Across folds, 100.0% to 100.0% of logged gradient norms exceed the configured clipping threshold. These are logged gradient norms compared with the threshold, not claimed post-clipping norms. Raw ELBO is stochastic, so non-monotonic values alone do not demonstrate optimization failure, and raw ELBO magnitudes are not compared across folds or horizons. Minimum site precision is one CVI stability diagnostic rather than a complete convergence metric. Temporal lengthscales are reported in both model steps and physical years.

## Use in future experiments

This utility will be reused for later optimizer and kernel experiments so their diagnostics can be compared with these frozen baselines.
