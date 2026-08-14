# SVGP model versions retained at the end of the inference phase

## Purpose

This document freezes the two SVGP implementations retained before starting
feature engineering.

Only one implementation is the current modelling baseline. The second is kept
as a reproducible alternative because it implements Natural-Gradient
variational inference and remains useful for the later ST-SVGP/CVI work.

The final 2020 -> 2025 temporal test remains locked for both versions.

## 1. Selected model — SVGP-Adam

### Status

**Selected current SVGP baseline**

This is the model used for the next feature-engineering experiments.

### Probabilistic model

For eligible cell \(i\) at forecast origin \(t\):

\[
f_{i,t} = \beta_0 + x_{i,t}^{T}\beta + g(s_i,t)
\]

with

\[
g(s,t) \sim GP(0, k_s(s,s')k_t(t,t'))
\]

and

\[
Y_{i,t}\mid f_{i,t}
\sim \operatorname{Bernoulli}[\Phi(f_{i,t})].
\]

### Current predictors

- `ndbi_t`
- `savi_t`
- `distance_to_built_m_t`
- `built_fraction_11x11_t`
- `recent_local_growth_5y_t`
- `elevation_m`
- `slope_degrees`
- `log_population_density_t`

Spatial coordinates and forecast origin are GP inputs only; they are not
included in the parametric linear mean.

### Kernel and sparse approximation

- spatial kernel: Matérn-3/2;
- temporal kernel: Matérn-3/2;
- separable kernel: `k_space * k_time`;
- spatial inducing locations: `M_s = 64`;
- inducing locations fixed;
- full variational covariance;
- Bernoulli-probit likelihood.

`M_s = 64` is retained because the 64 / 128 / 512 sensitivity experiment did
not produce meaningful predictive gains from increasing the number of spatial
inducing points.

### Inference

Adam jointly optimises:

- variational parameters `q_mu`, `q_sqrt`;
- kernel parameters;
- linear-mean coefficients.

Natural Gradient is **not** used by the selected model.

### Reproducible runtime

```yaml
compute:
  device: cpu
  float_type: float32
  jitter: 1.0e-6
```

CPU/float32 is retained because the identical GPU/Metal configuration showed
intermittent `q_mu` numerical failures, whereas CPU/float32 completed all
rolling folds and the final fit with effectively unchanged predictive metrics
and fitted parameters.

This is a runtime choice, not a different statistical model.

### Rolling temporal validation

| Metric | Mean |
|---|---:|
| Log Loss | 0.50198 |
| Brier score | 0.16516 |
| PR-AUC | 0.52976 |
| ROC-AUC | 0.82324 |
| Probability bias | 0.05893 |

Final fitted state:

| Parameter | Value |
|---|---:|
| Spatial lengthscale x | 2.91698 km |
| Spatial lengthscale y | 2.90162 km |
| Temporal lengthscale | 1.49211 five-year steps |
| Kernel variance | 0.73477 |
| Linear intercept | -0.48646 |
| Total inducing count | 256 |

### Canonical files

```text
configs/modeling/svgp_experiment.yaml
src/models/train_svgp.py
tests/test_svgp.py
```

These files define the selected SVGP.

## 2. Retained alternative — SVGP Natural Gradient V1

### Status

**Retained historical/alternative experiment — not the selected SVGP**

The retained configuration is the stable `natgrad_schedule_0025` experiment.

### Statistical model

The probabilistic model, likelihood, kernel, base predictors and `M_s = 64`
remain the same as the selected SVGP.

The difference is the optimisation of the Gaussian variational posterior.

### Natural-Gradient variational inference

The experiment uses:

- Adam for kernel and parametric-mean parameters;
- GPflow NaturalGradient for `q_mu` and `q_sqrt`;
- the natural (`XiNat`) parameterisation;
- the same mini-batch for Adam and NaturalGradient within one iteration;
- a log-linear Natural-Gradient step schedule.

Validated schedule:

```yaml
natural_gradient:
  parameterization: natural
  initial_gamma: 1.0e-6
  target_gamma: 2.5e-3
  ramp_iterations: 800
  minimum_gamma: 1.0e-8
  maximum_retries: 6
```

Runtime retained for this historical experiment:

```yaml
compute:
  device: cpu
  float_type: float64
  jitter: 1.0e-6
```

### CVI status

This experiment implements **Natural-Gradient variational inference**.

It is described as **CVI-aligned**, because the natural-parameter update is
the conceptual bridge to the Gaussian approximate-site interpretation used in
CVI.

It is **not** an explicit full CVI implementation: the experiment does not
construct Gaussian pseudo-likelihood sites for temporal filtering/smoothing.
That explicit representation belongs to the later ST-SVGP implementation.

### Rolling temporal validation

| Metric | Mean |
|---|---:|
| Log Loss | 0.54155 |
| Brier score | 0.17942 |
| PR-AUC | 0.53718 |
| ROC-AUC | 0.82998 |
| ECE | 0.20199 |
| Absolute probability bias | 0.20199 |
| Calibration intercept | -0.70371 |
| Calibration slope | 1.11928 |

The experiment slightly improved discrimination relative to the selected Adam
SVGP but degraded probability quality and calibration. It is therefore kept as
an informative alternative rather than promoted to the main SVGP.

### Canonical files

```text
configs/modeling/svgp/experiment_v1.yaml
src/models/svgp/natgrad/experiment_v1.py
```

## 3. Shared probability evaluation

Shared calibration code belongs outside individual models:

```text
src/models/evaluation/calibration.py
notebooks/10_calibration_analysis.ipynb
```

Current probability evaluation includes:

- Log Loss;
- Brier score;
- PR-AUC;
- ROC-AUC;
- reliability bins / reliability diagrams;
- Expected Calibration Error (ECE);
- calibration intercept;
- calibration slope;
- signed probability bias;
- absolute probability bias.

The same definitions should be used for Logistic Regression, SVGP and future
ST-SVGP models.

### Coverage

Coverage is deliberately **not implemented yet**.

The supervisor's proposed metric — the percentage of observations falling
inside an 80% confidence interval — needs a precise definition for the binary
urban-conversion target before implementation.

This is a deferred requirement, not an omitted one.

## 4. Next modelling phase — feature engineering

Inference choice, sparse-support sensitivity and runtime are now fixed for the
selected SVGP:

```text
M_s       = 64
inference = Adam
runtime   = CPU / float32
```

The next experiments should change predictors only.

Feature engineering is implemented outside the model under:

```text
src/feature_engineering/
configs/feature_engineering/
```

The selected `train_svgp.py` must remain unchanged until a feature candidate is
explicitly evaluated.
