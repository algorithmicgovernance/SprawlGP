# Annual ST-SVGP temporal diagnostic

Status: EXPERIMENTAL. Annual manual mapping validation remains deferred.
The locked 2020-2025 target block was not evaluated.

1. **Temporal lengthscale stability.** Annual physical lengthscales range from 3.02845 to 3.09121 years (CV 0.0113285).
2. **Physical comparison.** The annual median is 3.08572 years and the retained five-year median is 3.92339 years; the annual scale is shorter.
3. **Probability-bias stability.** Fold SD is 0.0578393 annually versus 0.223493 for the retained five-year folds.
4. **Calibration stability.** ECE fold SD is 0.0248095 annually versus 0.155584 retained; calibration-slope fold SD is 0.421185 annually versus 0.110069 retained.
5. **Latent uncertainty.** 2011: mean 0.123365, median 0.117765; 2016: mean 0.128332, median 0.122771; 2019: mean 0.152838, median 0.14254.
6. **Temporal identifiability.** The annual folds have lower relative temporal-lengthscale dispersion than the retained five-year folds, which is descriptive evidence of improved temporal identifiability.
7. **Temporal prediction-set coverage.** Fold 2: empirical coverage 0.7314366209, positive-class coverage 0.0001738677, negative-class coverage 0.8337651911, and empty-set rate 0.1930292617. Fold 3: empirical coverage 0.9149541101, positive-class coverage 0.0013885032, negative-class coverage 0.9579062814, and empty-set rate 0.0468625299.

Marginal coverage is dominated by the negative class and therefore does **not** establish useful positive-event uncertainty coverage.

One-year and five-year Log Loss values describe different forecasting events and are not ranked directly here.
