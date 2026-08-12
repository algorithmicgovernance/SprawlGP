# Sparse Variational Gaussian Process (SVGP)

## Purpose

This is the first GP implementation after the retained Logistic Regression
baseline. It is intentionally designed so that the later ST-SVGP changes the
inference architecture rather than the scientific model.

For each eligible 30 m cell:

\[
f_{i,t} = \beta_0 + x_{i,t}^{T}\beta + g(s_i,t)
\]

\[
g(s,t) \sim GP\left(0,\;
k_s(s,s')\,k_t(t,t')\right)
\]

\[
Y_{i,t}^{(5)} \mid f_{i,t}
\sim Bernoulli\left[\Phi(f_{i,t})\right].
\]

## Literature mapping

**Hensman, Matthews & Ghahramani (2015), Section 4, Eqs. 17–21**

The sparse variational posterior is defined through inducing variables:

\[
q(u)=N(m,S), \qquad q(f,u)=p(f|u)q(u).
\]

The classification ELBO is:

\[
\mathcal{L}
=
\sum_n E_{q(f_n)}
[\log p(y_n|f_n)]
-
KL[q(u)\|p(u)].
\]

The first term measures data fit. The KL term is the complexity penalty that
keeps the variational posterior close to the GP prior unless the observations
justify departure from it.

**Hamelijnck et al. (2021), Section 2.2 Eq. 5 and Figure 1**

The same SVGP objective is the starting point for ST-SVGP. The present
implementation therefore repeats one set of spatial inducing points at each
training time. ST-SVGP will later replace this generic space-time inducing
representation by spatial inducing states tracked through time.

## Predictors

The parametric mean uses:

```text
ndbi_t
savi_t
distance_to_built_m_t
built_fraction_11x11_t
recent_local_growth_5y_t
elevation_m
slope_degrees
log_population_density_t
```

The GP covariance uses:

```text
space = (x_center_m, y_center_m)
time  = forecast_origin
```

`x_center_m`, `y_center_m` and `forecast_origin` are therefore **not**
additional linear predictors.

Population is transformed as:

```text
log_population_density_t = log1p(population_density_t)
```

## Kernel

Primary kernel:

\[
k((s,t),(s',t'))
=
k^{space}_{Matern\,3/2}(s,s')
\times
k^{time}_{Matern\,3/2}(t,t').
\]

Space is expressed in kilometres. Time is expressed in five-year steps.

The separable Matérn structure is chosen now because the later ST-SVGP will
require an explicit space-time structure and a Markov-compatible temporal
kernel.

## Inducing variables

`spatial_points = 64` means 64 spatial k-means centres.

For a training fold with three forecast origins, standard SVGP receives:

```text
64 × 3 = 192 inducing points
```

The same 64 spatial locations are repeated at every training time.

Inducing locations are fixed after k-means in this first implementation. This
keeps the future SVGP/ST-SVGP comparison tied to the same spatial support.

## Optimisation

Mini-batches estimate the factorised expected log-likelihood term of the ELBO.

Two optimisers are interleaved:

```text
NaturalGradient → q_mu, q_sqrt
Adam            → kernel + linear mean parameters
```

This separation is useful now and is also the conceptual bridge to the
natural-gradient/CVI machinery used by Hamelijnck et al. for ST-SVGP.

No class-balanced mini-batches are used; the original transition prevalence is
preserved because probability quality is a primary outcome.

## Temporal evaluation

The four pre-test origins are intentional in this first study:

```text
train 2000              → validate 2005
train 2000, 2005        → validate 2010
train 2000, 2005, 2010  → validate 2015
```

The first fold is a cold-start diagnostic. With only one training origin, it
cannot provide meaningful empirical identification of temporal dependence; the
temporal lengthscale is largely governed by its initial value.

Final fit:

```text
2000, 2005, 2010, 2015
```

Locked final test:

```text
2020 → 2025
```

The training script never evaluates the final test.
