# COMB-A02 diagnostics

## Experiment contract

COMB-A02 is TREN-A01 + OPT-A03 + the unchanged EARLY-STOP-A01 protocol + 128 fixed inducing points. Exact normalized config parity passed against both TREN-A01 and EARLY-STOP-A01. Relative to EARLY-STOP-A01, the only scientific changes were Adam learning rate `0.001 -> 0.0005`, global gradient clip `10.0 -> 100000.0`, and fixed spatial inducing count `64 -> 128`. Output paths were redirected to this experiment.

The nine predictors, temporal-trend mean block, kernels and initializations, temporal lengthscale trainability, CVI natural-gradient gamma, folds, seed, CPU/float64/jitter runtime, development lock, and absence of regularization were unchanged. Inducing locations were initialized by the existing fold-specific training-only k-means path and remained non-trainable.

## Fold-2 optimizer preflight

Training origins were 2000-2014; validation origin 2015 was excluded; the maximum training target year was 2015. The preflight constructed 128 unique fixed training-only k-means centers and fresh CVI sites, then evaluated the trainer's deterministic first minibatch without a Natural Gradient update.

- Raw global gradient norm: `1714268.9956`.
- Clip threshold: `100000.0`; implied scale: `0.0583339022387`; clipping activated; post-clip norm: `100000.0`.
- Raw group norms: beta/beta0/beta_time `1647000.18299`; spatial LS `16113.3035952`; temporal LS `9314.20769565`; variance `475144.392252`.
- All gradients and the disposable Adam update were finite; positive constrained parameters remained positive; inducing locations did not move; every model trainable variable was restored exactly.

## Runtime and stopping

Logged optimization time was `16208.923392` seconds (`4.502479` hours): Fold 1 `3855.873324` s, Fold 2 `5590.071369` s, and Fold 3 `6762.978698` s.

Every fold reached 4000 completed iterations (zero-based iteration 3999) with `MAX_ITERATIONS`. No fold demonstrated convergence. Each fold had 51 eligible checkpoints, zero full convergence-gate passes, final patience zero, and terminal limiting gates `ELBO|PARAMETER|CVI_SITE`.

| Fold | ELBO endpoint/range | Spatial X/Y | Variance | Temporal LS | beta_time | CVI site median/max |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.003025 / 0.012310 | 0.004161 / 0.016785 | 0.014773 | 0.077596 | 0.013637 | 0.116245 / 0.127687 |
| 2 | 0.002450 / 0.018898 | 0.009334 / 0.010484 | 0.021999 | 0.080439 | 0.022979 | 0.151290 / 0.160881 |
| 3 | 0.007895 / 0.007867 | 0.003396 / 0.024401 | 0.009396 | 0.078325 | 0.026136 | 0.167928 / 0.171856 |

## EARLY-STOP-A01 comparison

The higher clip was selective for COMB-A02 rather than active at every logged checkpoint. Gradient p50/p90/p95 changed from `54384/84140/90298 -> 53525/82433/99100` (Fold 1), `74800/118919/134418 -> 67431/111641/132529` (Fold 2), and `86934/123646/133525 -> 85644/126394/136864` (Fold 3). Fractions above clip changed from `1.0` in every EARLY-STOP-A01 fold to `0.05`, `0.20`, and `0.3125`; median gradient/clip ratios changed from `5438`, `7480`, and `8693` to `0.535`, `0.674`, and `0.856`.

Raw ELBO tail range / slope per 100 iterations changed from `37172 / 3396 -> 29423 / 6333`, `71030 / 1246 -> 56314 / 4247`, and `98317 / -27984 -> 83841 / -22493`. Final spatial X/Y changed from `4.891/5.154 -> 3.687/3.506`, `4.921/4.766 -> 3.773/3.277`, and `4.842/4.989 -> 3.795/3.399` km. Final temporal LS changed from `0.769/0.744/0.737 -> 1.619/1.536/1.519`; variance from `0.648/0.758/0.836 -> 1.391/1.648/1.875`; beta_time from `-0.464/-0.547/-0.475 -> -0.343/-0.462/-0.439`; and intercept from `-0.408/-0.520/-0.570 -> -0.309/-0.413/-0.473`.

COMB-A02 did not improve convergence behavior: like EARLY-STOP-A01, every full gate failed through the maximum budget. Terminal gate magnitudes changed unevenly, with CVI-site and temporal-LS stability worse in every fold.

## Prediction comparison

Macro values are unweighted fold means.

| Metric | TREN-A01 | EARLY-STOP-A01 | COMB-A02 | COMB - TREN | COMB - EARLY |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR-AUC | 0.169053 | 0.214129 | 0.169187 | +0.000134 | -0.044942 |
| ROC-AUC | 0.799628 | 0.835135 | 0.804305 | +0.004676 | -0.030830 |
| Log Loss | 0.232468 | 0.254291 | 0.235933 | +0.003465 | -0.018358 |
| Brier | 0.061571 | 0.070275 | 0.064599 | +0.003028 | -0.005676 |
| ECE | 0.054760 | 0.088589 | 0.065195 | +0.010435 | -0.023394 |
| Probability bias | 0.028226 | 0.088059 | 0.049677 | +0.021451 | -0.038382 |
| Calibration slope | 0.806319 | 1.453427 | 0.989364 | +0.183045 | -0.464063 |

Against TREN-A01, COMB-A02 improved ROC-AUC in every fold, but PR-AUC increased only in Fold 2 and was nearly unchanged in the macro mean. Probability quality improved in Fold 2 but worsened in Folds 1 and 3; the macro Log Loss, Brier, ECE, and absolute bias were worse. Calibration slope moved closer to 1 in Folds 2 and 3 but farther above 1 in Fold 1.

Against EARLY-STOP-A01, COMB-A02 reduced PR-AUC and ROC-AUC in every fold. Reliability improved in Folds 1 and 3 but worsened in Fold 2; all macro reliability metrics improved, and the macro calibration slope moved from `1.453427` to `0.989364`.

## Historical OPT-A03 context

Relative to OPT-A03 at 1500, COMB-A02 improved macro PR-AUC by `0.020867`, ROC-AUC by `0.028074`, Log Loss by `0.007159`, and calibration slope from `0.639259` to `0.989364`. Brier worsened by `0.002685`, ECE by `0.008546`, and probability bias by `0.033617`.

This partially supports the under-optimization hypothesis because discrimination recovered relative to OPT-A03, but it does not establish the cause: COMB-A02 also doubled fixed inducing support. The result does not recover EARLY-STOP-A01 discrimination, does not match TREN-A01 reliability overall, and does not demonstrate convergence.

## Assessment

Descriptive assessment: **mixed**. COMB-A02 trades away EARLY-STOP-A01's ranking gain for substantially better average reliability and slope, but lands close to TREN-A01 discrimination with generally worse macro probability reliability. No auto-promotion is supported.

Detailed fold metrics, complete terminal diagnostics, and all eligible checkpoints are recorded in the adjacent CSV artifacts. No post-hoc calibration was performed.