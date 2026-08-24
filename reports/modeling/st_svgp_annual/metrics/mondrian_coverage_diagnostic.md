# Annual ST-SVGP Mondrian coverage diagnostic

This is a temporal diagnostic using strictly past-fold calibration. It does not establish formal class-conditional validity under temporal distribution shift; temporal non-exchangeability may still affect coverage.

## Fold results

| Fold | Calibration (negative / positive) | Evaluation | q_hat_0 | q_hat_1 | Marginal positive | Mondrian positive | Marginal negative | Mondrian negative |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 98,914 / 1,714 | 93,706 | 0.161546 | 0.834208 | 0.000174 | 0.398505 | 0.833765 | 0.820481 |
| 3 | 181,117 / 13,217 | 80,192 | 0.153584 | 0.982691 | 0.001389 | 0.695918 | 0.957906 | 0.935358 |

## Efficiency comparison

| Fold | Marginal coverage | Mondrian coverage | Marginal average size | Mondrian average size | Marginal singleton | Mondrian singleton | Marginal both | Mondrian both | Marginal empty | Mondrian empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.731437 | 0.768681 | 0.806971 | 0.995123 | 0.806971 | 0.995123 | 0.000000 | 0.000000 | 0.193029 | 0.004877 |
| 3 | 0.914954 | 0.924606 | 0.953137 | 1.261248 | 0.953137 | 0.738752 | 0.000000 | 0.261248 | 0.046863 | 0.000000 |

## Interpretation

Positive-class coverage materially improves (Fold 2 0.000174 to 0.398505; Fold 3 0.001389 to 0.695918), but remains below the nominal 0.80 target. The remaining shortfalls are Fold 2 0.401495; Fold 3 0.104082, respectively. The result is substantially improved but still deficient overall, rather than restored to nominal coverage.

Negative-class and marginal coverage remain high, while the fold tables show the corresponding changes in set size, singleton, both-label, and empty-set rates.

The large positive-class improvement indicates that the previous near-zero coverage was primarily driven by the single marginal threshold under class imbalance. The persistent positive shortfall, especially in the earliest evaluated fold, shows that marginal calibration was not the only limitation.

MONDRIAN_IMPROVES_BUT_REMAINS_INADEQUATE
