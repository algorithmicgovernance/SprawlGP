# ST-SVGP Explainability Summary

These SHAP results describe model attribution under fixed representative space-time contexts and are not causal effects.

## Annual 1y

- Top three overall: ndbi_t, log_population_density_t, log_distance_to_built_m_t.
- Rankings change across folds; fold 1: ndbi_t, log_distance_to_built_m_t, log_population_density_t; fold 2: ndbi_t, log_population_density_t, savi_t; fold 3: ndbi_t, log_population_density_t, log_distance_to_built_m_t.
- Strongest near-built versus peripheral differences: ndbi_t (near built), log_distance_to_built_m_t (peripheral), log_population_density_t (near built).
- Direction: ndbi_t: higher values generally push probability upward; log_population_density_t: higher values generally push probability upward; log_distance_to_built_m_t: higher values generally push probability downward.
- Additivity error: mean 9.98044e-17, maximum 3.33067e-16.

## Five-year 5y

- Top three overall: ndbi_t, log_population_density_t, elevation_m.
- Rankings change across folds; fold 1: log_population_density_t, ndbi_t, log_distance_to_built_m_t; fold 2: ndbi_t, log_population_density_t, elevation_m; fold 3: ndbi_t, log_population_density_t, elevation_m.
- Strongest near-built versus peripheral differences: log_distance_to_built_m_t (peripheral), log_population_density_t (near built), ndbi_t (near built).
- Direction: ndbi_t: higher values generally push probability upward; log_population_density_t: higher values generally push probability upward; elevation_m: higher values generally push probability upward.
- Additivity error: mean 1.21199e-16, maximum 3.33067e-16.

## Cross-horizon

The horizons share log_population_density_t, ndbi_t among their three strongest attributions. Raw SHAP magnitudes are not compared because the horizons predict different targets.

No model was trained or tuned, no reconstruction was rerun, no final fit was run, and no locked test row was used.
