# ST-SVGP baseline convergence diagnostics

## Scope

This is a read-only analysis of existing histories. No model was trained, no hyperparameter was changed, and no final or locked test was evaluated.

## Inputs

- `five_year_canonical`: `reports/modeling/st_svgp/metrics/st_svgp_training_history.csv`; SHA-256 `f4103e278f53438261a0dfb1de8bdd74b29b749f4280e32728e6cdaafd388055`
- `TEMP-F01_temporal_lengthscale_frozen`: `reports/modeling/st_svgp_improvements/five_year/temporal/temporal_lengthscale_frozen/metrics/st_svgp_training_history.csv`; SHA-256 `070682305ad46064e3fd3edc4314d66fcf7b1c7cd769801943247b4b9d61158e`

## Annual baselines

## Five-year baseline

### five_year_canonical

- Fold 1: ELBO tail range 6,453 and slope -2,263 per 100 iterations; median gradient/clip ratio 1,062, with 100.0% above the configured threshold; site precision 3.79 to 68.53; spatial X 2.002 to 4.738 km and Y 2.002 to 3.568 km; kernel variance 0.999 to 0.4798; temporal lengthscale 1.5 to 1.5 steps = 7.5 to 7.5 years.
- Fold 2: ELBO tail range 1.928e+04 and slope 6,763 per 100 iterations; median gradient/clip ratio 2,458, with 100.0% above the configured threshold; site precision 1.575 to 58.94; spatial X 2.002 to 4.36 km and Y 2.002 to 3.927 km; kernel variance 0.999 to 0.4892; temporal lengthscale 1.499 to 0.7664 steps = 7.493 to 3.832 years.
- Fold 3: ELBO tail range 1.317e+04 and slope 2,027 per 100 iterations; median gradient/clip ratio 3,536, with 100.0% above the configured threshold; site precision 0.13 to 51.25; spatial X 2.002 to 4.124 km and Y 2.002 to 3.915 km; kernel variance 0.999 to 0.4991; temporal lengthscale 1.499 to 0.7847 steps = 7.493 to 3.923 years.

## Diagnostic observations

Across folds, 100.0% to 100.0% of logged gradient norms exceed the configured clipping threshold. These are logged gradient norms compared with the threshold, not claimed post-clipping norms. Raw ELBO is stochastic, so non-monotonic values alone do not demonstrate optimization failure, and raw ELBO magnitudes are not compared across folds or horizons. Minimum site precision is one CVI stability diagnostic rather than a complete convergence metric. Temporal lengthscales are reported in both model steps and physical years.

## Use in future experiments

This utility will be reused for later optimizer and kernel experiments so their diagnostics can be compared with these frozen baselines.
