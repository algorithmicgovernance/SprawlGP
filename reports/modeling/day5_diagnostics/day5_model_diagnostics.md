# Day-5 ST-SVGP Model Diagnostics

These diagnostics use raw pre-test OOF probabilities. Latent variance is described 
only as latent posterior variance; it is not calibrated probability uncertainty.

## Annual 1-year ST-SVGP

- Largest mean per-cell loss by relative distance context: **near_built** (0.4306).
- Most difficult OOF fold by mean per-cell loss: fold **2** (0.3345).
- Overall Spearman correlation with per-cell log loss: -0.298; with absolute error: -0.298.
- Confident errors are important: mean loss is 0.3354 in the lowest latent-variance quintile versus 0.1657 in the highest. Latent posterior variance does not behave as calibrated error uncertainty.
- Context-conditioned covariate SHAP is not technically ready: real rolling-fold states are unavailable or failed OOF reproduction.

## Five-year ST-SVGP

- Largest mean per-cell loss by relative distance context: **near_built** (0.7128).
- Most difficult OOF fold by mean per-cell loss: fold **2** (0.6133).
- Latent posterior variance is unavailable in the retained OOF artifact, so uncertainty-error claims cannot be made.
- Context-conditioned covariate SHAP is not technically ready: real rolling-fold states are unavailable or failed OOF reproduction.

## Cross-horizon descriptive comparison

The horizons predict different events and are not ranked by raw probability metrics. Comparisons are limited to failure geography, temporal heterogeneity, and the availability of uncertainty and explainability evidence.

No locked annual target or locked five-year final-test row was used.
