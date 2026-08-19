# ST-SVGP modeling and validation

## Purpose

This document records the current Sparse Variational Spatio-Temporal Gaussian
Process (ST-SVGP) implementation for five-year urban-expansion probabilities in
Yaoundé. It separates the probabilistic model, pre-training mathematical
validation, retained tests, promoted candidate, temporal-lengthscale
diagnostics, chronological validation protocol and final-test lock.

## Current promoted candidate

Canonical configuration:

```text
configs/modeling/st_svgp.yaml
```

Current choices:

```text
feature set                  log_distance_growth
linear predictors            9
spatial inducing points      64 fixed locations
spatial kernel               anisotropic Matérn-3/2
temporal kernel              Matérn-3/2 Markov state-space
temporal ell initial value   1.5 five-year steps
temporal ell trainable       yes
likelihood                   Bernoulli-probit
non-conjugate inference      CVI Gaussian pseudo-sites
CVI site structure           one dense 64 x 64 block per training time
variational update           Natural Gradient
hyperparameter update        Adam
filtering                    sequential Kalman
smoothing                    RTS
runtime                      CPU / float64
jitter                       1e-6
final test                   2020 origin locked
```

The latent function is

\[
f_{i,t}=\beta_0+x_{i,t}^{\top}\beta+g(s_i,t),
\]

with

\[
g(s,t)\sim GP(0,k_s(s,s')k_t(t,t'))
\]

and

\[
Y_{i,t}\mid f_{i,t}\sim\operatorname{Bernoulli}(\Phi(f_{i,t})).
\]

Coordinates and forecast origin are GP/state-space inputs only.

## Pre-training validation philosophy

The real Yaoundé dataset is not used to establish the correctness of the
numerical machinery. Small synthetic Gaussian and binary problems provide
independent dense oracles. The implementation is accepted only after the
state-space, filtering/smoothing, CVI and sparse-space calculations agree with
direct calculations to numerical precision.

```text
direct Matérn-3/2 covariance
        =
state-space covariance
        ↓
closed-form A(delta)
        =
matrix exponential exp(F delta)
        ↓
Kalman + RTS posterior
        =
dense exact Gaussian-process posterior
        ↓
Gaussian CVI-site state-space posterior
        =
dense Gaussian-site posterior
        ↓
CVI state-space ELBO
        =
E_q log p(y|f) - KL(q||p)
        ↓
spatial block Kalman/RTS posterior
        =
dense Kronecker space-time GP posterior
```

## 1. Temporal Matérn-3/2 state-space checks

For temporal lengthscale \(\ell_t\),

\[
\lambda=\sqrt{3}/\ell_t.
\]

The state is

\[
x(t)=[f(t),f'(t)]^\top,
\]

with

\[
F=
\begin{bmatrix}
0 & 1\\
-\lambda^2 & -2\lambda
\end{bmatrix},
\qquad
H=[1,0].
\]

The stationary covariance is

\[
P_\infty=\operatorname{diag}(\sigma^2,\sigma^2\lambda^2).
\]

### State-space covariance equals direct Matérn covariance

The projected Markov-state covariance must equal

\[
k_t(t_i,t_j)=
\sigma^2(1+\lambda|t_i-t_j|)
\exp(-\lambda|t_i-t_j|).
\]

This is checked for several lengthscale/variance pairs.

### Closed-form transition equals matrix exponential

The implementation uses

\[
A(\Delta)=
e^{-\lambda\Delta}
\begin{bmatrix}
1+\lambda\Delta & \Delta\\
-\lambda^2\Delta & 1-\lambda\Delta
\end{bmatrix}.
\]

It is independently checked against

\[
A(\Delta)=\exp(F\Delta).
\]

Multiple \(\Delta\) values are used.

### Process noise preserves stationarity

The discrete process noise is

\[
Q(\Delta)=P_\infty-A(\Delta)P_\infty A(\Delta)^\top.
\]

The code verifies

\[
P_\infty=
A(\Delta)P_\infty A(\Delta)^\top+Q(\Delta),
\]

and verifies that \(Q(\Delta)\) is positive semidefinite.

### GPflow covariance convention

The custom Matérn-3/2 covariance is also compared with GPflow's Matérn-3/2
convention so the ST-SVGP temporal kernel is parameterised consistently with
the earlier SVGP work.

Run:

```bash
make st-svgp-validate-temporal-kernel
```

## 2. Gaussian Kalman filter and RTS smoother checks

A small Gaussian observation problem is solved in two independent ways:

1. sequential Kalman filtering plus backward RTS smoothing;
2. a dense exact Gaussian-process posterior.

The following quantities must agree:

- smoothed latent means;
- smoothed marginal variances;
- Gaussian log marginal likelihood.

Additional checks ensure:

- filtered covariances are positive semidefinite;
- smoothed covariances are positive semidefinite;
- smoothing does not increase function uncertainty relative to filtering.

Both regular and irregular temporal spacings are used so the code cannot
silently assume \(\Delta=1\).

Run:

```bash
make st-svgp-validate-filter-smoother
```

## 3. Bernoulli-probit CVI and Natural Gradient

The true likelihood remains Bernoulli-probit. CVI introduces Gaussian
pseudo-sites only as an inference representation.

For a scalar site,

\[
\tilde p(f)\propto
\exp(\tilde\lambda_1 f+\tilde\lambda_2 f^2),
\]

with precision

\[
\tilde\Lambda=-2\tilde\lambda_2.
\]

The expected Bernoulli-probit log likelihood is evaluated with
Gauss-Hermite quadrature.

### Autodiff equals finite differences

CVI moment derivatives obtained from TensorFlow autodiff are compared with
independent central finite differences.

### Gaussian-site posterior equality

With fixed Gaussian pseudo-sites, filtering/smoothing defines a conjugate
Gaussian model. Its state-space posterior is compared with the exact dense
Gaussian-site posterior.

### CVI ELBO identity

The state-space CVI expression

\[
\mathcal L=
E_q[\log p(y|f)]
-
E_q[\log \tilde p(y|f)]
+
\log \tilde p(y)
\]

must equal the standard variational form

\[
\mathcal L=
E_q[\log p(y|f)]
-
KL(q\Vert p).
\]

The two values are computed independently and compared numerically.

### Natural-Gradient validity

The retained tests verify that:

- Gaussian-site precision remains positive;
- repeated Natural-Gradient updates improve the fixed synthetic ELBO;
- predictive Bernoulli-probit probabilities stay strictly in \((0,1)\).

Run:

```bash
make st-svgp-validate-cvi
```

## 4. Spatial sparse and block state-space checks

### Spatial Matérn covariance

For spatial inducing locations \(Z_s\), the anisotropic Matérn-3/2 covariance

\[
K_{ZZ}^{(s)}
\]

must be symmetric positive semidefinite.

### Sparse conditional variance

For point \(s_i\),

\[
A_i=K_{iZ}K_{ZZ}^{-1}.
\]

The inducing posterior plus conditional residual must produce a positive
marginal variance.

### Block state-space equals dense Kronecker GP

For a small inducing set, the structured model uses

\[
A_n=I_{M_s}\otimes A_t(\Delta_n),
\]

\[
Q_n=K_{ZZ}^{(s)}\otimes Q_t(\Delta_n),
\]

\[
H=I_{M_s}\otimes H_t.
\]

The block Kalman/RTS posterior is compared against a dense space-time GP with

\[
K=K_t\otimes K_s.
\]

Inducing posterior means and covariance blocks must agree. This is the key
matrix-equality test before running the real \(M_s=64\) model.

## 5. Retained 37-test suite

```text
tests/test_st_svgp.py              20
tests/test_st_svgp_filtering.py     5
tests/test_st_svgp_cvi.py           7
tests/test_st_svgp_model.py         5
                                    --
                                    37
```

Run:

```bash
make st-svgp-tests
```

The root `Makefile` contains a numbered `# [xx/37]` comment for every retained
test, describing the mathematical or model contract protected by that test.

The complete pre-training gate is:

```bash
make st-svgp-pretraining-validation
```

It runs the 37 tests plus the three executable validation experiments.

## 6. Temporal-lengthscale experiments

### Free initial value 1.0

The original real-model experiment initialised \(\ell_t=1.0\) and allowed Adam
to learn it. It is retained as:

```text
configs/modeling/st_svgp/experiment_free_init_1p0.yaml
```

### Free initial value 1.5 — promoted candidate

The candidate promoted to the root config initialises \(\ell_t=1.5\) while
keeping it trainable:

```text
configs/modeling/st_svgp.yaml
```

This candidate had the best aggregate pre-test Log Loss/Brier compromise among
the temporal-lengthscale variants evaluated so far. Promotion is provisional,
not a final scientific conclusion.

### Fixed value 1.5 — retained diagnostic

The diagnostic in which \(\ell_t\) is truly fixed remains:

```text
configs/modeling/st_svgp/temporal_lengthscale_fixed_1p5.yaml
```

It is not the promoted model.

## 7. Fold-1 constant growth predictors

At forecast origin 2000:

```text
recent_local_growth_5y_t
built_fraction_x_recent_growth_t
```

are both constant at zero.

The preprocessing rule is:

1. compute mean and scale only on the training fold;
2. if a predictor has zero training variance, assign unit scale;
3. keep the centred value at zero.

This avoids division by zero and future-information leakage. It also avoids
dropping a predictor that becomes informative in later folds.

A separate semantic question remains: if the earliest zero means "previous
growth unavailable" instead of "true zero growth", that is a cold-start feature
design issue. It must be investigated separately, not mixed with kernel or CVI
changes.

## 8. Rolling temporal validation

Pre-test folds:

```text
Fold 1: 2000             -> 2005
Fold 2: 2000, 2005       -> 2010
Fold 3: 2000, 2005, 2010 -> 2015
```

Run the promoted candidate:

```bash
make st-svgp-rolling
```

OOF probabilities:

```text
reports/modeling/st_svgp/predictions/st_svgp_oof_predictions.parquet
```

## 9. Shared LR / Strong SVGP / ST-SVGP comparison

The shared evaluator uses:

```text
Logistic Regression:
reports/modeling/logistic_regression/predictions/logistic_oof_predictions.parquet

Strong SVGP (log_distance_growth):
reports/modeling/svgp/features/log_distance_growth/svgp_feature_oof_predictions.parquet

Promoted ST-SVGP:
reports/modeling/st_svgp/predictions/st_svgp_oof_predictions.parquet
```

Run:

```bash
make st-svgp-compare-oof
```

The common evaluator reports Log Loss, Brier score, PR-AUC, ROC-AUC, ECE,
calibration intercept, calibration slope, probability bias, reliability bins
and empirical temporal prediction-set coverage.

## 10. Final-test lock

The 2020 forecast origin remains locked:

```yaml
final_test:
  origins: [2020]
  evaluate: false
```

Do not use 2020 to choose, tune or repair ST-SVGP. Any remaining model decisions
must use only pre-2020 evidence. Once model choices and evaluation protocol are
frozen, 2020 can be unlocked once for the final comparison.
