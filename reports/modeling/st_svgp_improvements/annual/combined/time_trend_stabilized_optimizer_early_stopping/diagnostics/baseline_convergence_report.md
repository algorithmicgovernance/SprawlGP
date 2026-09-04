# ST-SVGP baseline convergence diagnostics

## Scope

This is a read-only analysis of existing histories. No model was trained, no hyperparameter was changed, and no final or locked test was evaluated.

## Inputs

- `EARLY-STOP-A01`: `reports/modeling/st_svgp_improvements/annual/combined/time_trend_early_stopping/metrics/st_svgp_training_history.csv`; SHA-256 `2e6e6b9b95f442d5a851d9bd3e5deb258ccb0483d45660d378dda5d4b963006c`
- `COMB-A02`: `reports/modeling/st_svgp_improvements/annual/combined/time_trend_stabilized_optimizer_early_stopping/metrics/st_svgp_training_history.csv`; SHA-256 `7809918e171a5d49f0b8ee5f3408b74bbabe8f606456d5bd53abd7f83a24eae2`

## Annual baselines

## Five-year baseline

## Diagnostic observations

Across folds, 5.0% to 100.0% of logged gradient norms exceed the configured clipping threshold. These are logged gradient norms compared with the threshold, not claimed post-clipping norms. Raw ELBO is stochastic, so non-monotonic values alone do not demonstrate optimization failure, and raw ELBO magnitudes are not compared across folds or horizons. Minimum site precision is one CVI stability diagnostic rather than a complete convergence metric. Temporal lengthscales are reported in both model steps and physical years.

## Use in future experiments

This utility will be reused for later optimizer and kernel experiments so their diagnostics can be compared with these frozen baselines.
