# Annual ST-SVGP temporal calibration diagnostic

## Scope and temporal contract

Source: `reports/modeling/st_svgp_annual/experiments/convergence_3000/predictions/st_svgp_oof_predictions.parquet` (SHA-256 `1e96f1495ba756a6cc83c716cdeac220ee4252b3309a3bf105942e1fe2a28aa3`). The source OOF artifact was read only; no model training was performed. Fold 1 has no calibrated evaluation. Fold 2 fits on Fold 1 only, and Fold 3 fits on Folds 1+2 only. All rows have `target_year < 2020`.

Platt uses `logit(q) = a + b * logit(p)`. Beta uses `logit(q) = a * log(p) + b * log(1-p) + c`. Inputs are clipped to [1e-6, 1-1e-6] only for logarithms. Fits are unweighted and use no regularization.

The canonical marginal and Mondrian conformal reports were not modified.

## Fitted parameters

| Method | Fold | Calibration folds | Calibration rows | a | b | c | Direction |
|---|---:|---|---:|---:|---:|---:|---|
| platt | 2 | 1 | 100,628 | -2.942965 | 1.818876 | NA | increasing |
| beta | 2 | 1 | 100,628 | 1.693911 | -1.986696 | -3.166050 | increasing |
| platt | 3 | 1,2 | 194,334 | -1.619637 | 0.742055 | NA | increasing |
| beta | 3 | 1,2 | 194,334 | 0.740499 | -0.744909 | -1.622195 | increasing |

## Probability metrics

| Fold | Method | Prevalence | Mean probability | PR-AUC | ROC-AUC | Log Loss | Brier | ECE | Intercept | Slope | Bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | raw | 0.122756 | 0.171704 | 0.294137 | 0.767490 | 0.334473 | 0.101071 | 0.050438 | -0.632543 | 0.851733 | 0.048948 |
| 2 | platt | 0.122756 | 0.009945 | 0.294137 | 0.767490 | 0.595898 | 0.117140 | 0.112811 | 0.746175 | 0.468359 | -0.112811 |
| 2 | beta | 0.122756 | 0.009892 | 0.294137 | 0.767490 | 0.593239 | 0.117192 | 0.112864 | 0.808774 | 0.483452 | -0.112864 |
| 3 | raw | 0.044905 | 0.141035 | 0.175129 | 0.835212 | 0.207326 | 0.053584 | 0.096130 | -1.130315 | 1.233936 | 0.096130 |
| 3 | platt | 0.044905 | 0.049244 | 0.175129 | 0.835212 | 0.154783 | 0.039787 | 0.017061 | 1.567936 | 1.664033 | 0.004340 |
| 3 | beta | 0.044905 | 0.049288 | 0.175129 | 0.835212 | 0.154801 | 0.039788 | 0.017087 | 1.568861 | 1.664877 | 0.004383 |

## Ranking check

| Fold | Method | PR-AUC delta | ROC-AUC delta | PR preserved | ROC preserved |
|---:|---|---:|---:|---|---|
| 2 | platt | 0.000000000000 | 0.000000000000 | True | True |
| 2 | beta | 0.000000000000 | 0.000000000000 | True | True |
| 3 | platt | 0.000000000000 | 0.000000000000 | True | True |
| 3 | beta | 0.000000000000 | 0.000000000000 | True | True |

## Interpretation

Fold 2 is the limiting result: calibration transferred from the much lower-prevalence Fold 1 and worsened Log Loss, Brier, ECE, calibration slope distance from 1, and absolute probability bias for both methods.

Fold 3 shows the opposite pattern. Both methods lower Log Loss, Brier, ECE, and absolute probability bias, but move the calibration slope farther from 1. Both fitted mappings are increasing and the directly recomputed PR-AUC and ROC-AUC values are unchanged.

The broad Fold 3 improvement is scientifically useful, but the Fold 2 failure prevents a robust positive calibration conclusion. ECE is not used alone: the decision jointly considers proper scores, slope, bias, discrimination, and fold robustness.

CALIBRATION_IMPROVES_MODEL_BUT_REMAINS_HETEROGENEOUS
