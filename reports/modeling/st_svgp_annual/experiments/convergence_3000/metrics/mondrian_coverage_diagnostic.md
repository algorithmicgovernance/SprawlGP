# Annual ST-SVGP Mondrian coverage diagnostic

This is a temporal diagnostic using strictly past-fold calibration. It does not establish formal class-conditional validity under temporal distribution shift; temporal non-exchangeability may still affect coverage.

## Fold results

| Fold | Calibration (negative / positive) | Evaluation | q_hat_0 | q_hat_1 | Marginal positive | Mondrian positive | Marginal negative | Mondrian negative |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 98,914 / 1,714 | 93,706 | 0.360708 | 0.603537 | 0.027819 | 0.330262 | 0.896269 | 0.890247 |
| 3 | 181,117 / 13,217 | 80,192 | 0.318125 | 0.847001 | 0.021105 | 0.848653 | 0.941521 | 0.906947 |

## Efficiency comparison

| Fold | Marginal coverage | Mondrian coverage | Marginal average size | Mondrian average size | Marginal singleton | Mondrian singleton | Marginal both | Mondrian both | Marginal empty | Mondrian empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.789661 | 0.821506 | 0.871897 | 0.971571 | 0.871897 | 0.971571 | 0.000000 | 0.000000 | 0.128103 | 0.028429 |
| 3 | 0.900190 | 0.904330 | 0.933460 | 1.232455 | 0.933460 | 0.767545 | 0.000000 | 0.232455 | 0.066540 | 0.000000 |

## Interpretation

Positive-class coverage materially improves (Fold 2 0.027819 to 0.330262; Fold 3 0.021105 to 0.848653), but remains below the nominal 0.80 target. The remaining shortfalls are Fold 2 0.469738; Fold 3 -0.048653, respectively. The result is substantially improved but still deficient overall, rather than restored to nominal coverage.

Negative-class and marginal coverage remain high, while the fold tables show the corresponding changes in set size, singleton, both-label, and empty-set rates.

The large positive-class improvement indicates that the previous near-zero coverage was primarily driven by the single marginal threshold under class imbalance. The persistent positive shortfall, especially in the earliest evaluated fold, shows that marginal calibration was not the only limitation.

MONDRIAN_IMPROVES_BUT_REMAINS_INADEQUATE
