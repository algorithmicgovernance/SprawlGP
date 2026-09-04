# ST-SVGP baseline convergence diagnostics

## Scope

This is a read-only analysis of existing histories. No model was trained, no hyperparameter was changed, and no final or locked test was evaluated.

## Inputs

- `annual_1500`: `reports/modeling/st_svgp_annual/experiments/convergence_1500/metrics/st_svgp_training_history.csv`; SHA-256 `813dde139c2b6cdcafafd3640ed3870ead04189eb3f65f5331e09ca43d7895aa`
- `OPT-A01_clip_100k`: `reports/modeling/st_svgp_improvements/annual/optimizer/clip_100k/metrics/st_svgp_training_history.csv`; SHA-256 `ab28ef2beb649b06f16e8332c7b1be3bd1dddaa39b2e309cf623b8b2a0014964`
- `OPT-A02_adam_lr_5e_4`: `reports/modeling/st_svgp_improvements/annual/optimizer/adam_lr_5e_4/metrics/st_svgp_training_history.csv`; SHA-256 `506e5901fd4008418efa406d1ef6801b5e84a035051bcddd6ea62d534471a5f8`
- `OPT-A03_adam_lr_5e_4_clip_100k`: `reports/modeling/st_svgp_improvements/annual/optimizer/adam_lr_5e_4_clip_100k/metrics/st_svgp_training_history.csv`; SHA-256 `86389a6a6a6b547da5f1de27467650f85bcf30985f46bf2854e4f0162ecab89f`

## Annual baselines

### annual_1500

- Fold 1: ELBO tail range 2.106e+04 and slope -1,997 per 100 iterations; median gradient/clip ratio 5,397, with 100.0% above the configured threshold; site precision 0.00099 to 24.39; spatial X 2.002 to 4.12 km and Y 2.002 to 4.184 km; kernel variance 0.999 to 0.5849; temporal lengthscale 7.493 to 2.232 steps = 7.493 to 2.232 years.
- Fold 2: ELBO tail range 7.006e+04 and slope -9,246 per 100 iterations; median gradient/clip ratio 7,318, with 100.0% above the configured threshold; site precision 0.00099 to 10.32; spatial X 2.002 to 4.149 km and Y 2.002 to 3.899 km; kernel variance 0.999 to 0.6187; temporal lengthscale 7.493 to 2.198 steps = 7.493 to 2.198 years.
- Fold 3: ELBO tail range 6.975e+04 and slope -3.041e+04 per 100 iterations; median gradient/clip ratio 7,436, with 100.0% above the configured threshold; site precision 0.00099 to 8.843; spatial X 2.002 to 4.082 km and Y 2.002 to 3.948 km; kernel variance 0.999 to 0.6705; temporal lengthscale 7.493 to 2.167 steps = 7.493 to 2.167 years.

## Five-year baseline

## Diagnostic observations

Across folds, 3.2% to 100.0% of logged gradient norms exceed the configured clipping threshold. These are logged gradient norms compared with the threshold, not claimed post-clipping norms. Raw ELBO is stochastic, so non-monotonic values alone do not demonstrate optimization failure, and raw ELBO magnitudes are not compared across folds or horizons. Minimum site precision is one CVI stability diagnostic rather than a complete convergence metric. Temporal lengthscales are reported in both model steps and physical years.

## Use in future experiments

This utility will be reused for later optimizer and kernel experiments so their diagnostics can be compared with these frozen baselines.
