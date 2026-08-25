"""Real ST-SVGP model components for binary urban expansion."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from src.models.st_svgp.block_inference import (
    BlockFilterResult,
    BlockSmootherResult,
    DenseCviSites,
    block_kalman_filter,
    block_rts_smoother,
    dense_sites_to_gaussian,
    expected_log_dense_sites,
    inducing_function_marginals,
    temporal_block_matrices,
)
from src.models.st_svgp.cvi import (
    expected_log_bernoulli_probit,
    probit_predictive_probability,
)
from src.models.st_svgp.spatial import (
    matern32_spatial_covariance,
    spatial_conditional,
)


@dataclass(frozen=True)
class STPosterior:
    """Posterior over inducing states and inducing function values."""

    filter_result: BlockFilterResult
    smoother_result: BlockSmootherResult
    inducing_means: tf.Tensor
    inducing_covariances: tf.Tensor
    spatial_covariance: tf.Tensor


class STSVGPModel:
    """TensorFlow implementation of the full sparse Markov ST-SVGP."""

    def __init__(
        self,
        *,
        inducing_locations_km: tf.Tensor,
        n_features: int,
        spatial_initial_lengthscale_km: float,
        temporal_initial_lengthscale_steps: float,
        variance_initial: float,
        jitter: float,
        quadrature_degree: int,
        temporal_lengthscale_trainable: bool = True,
        dtype: tf.dtypes.DType = tf.float64,
    ) -> None:
        self.dtype = dtype
        self.jitter = float(jitter)
        self.quadrature_degree = int(
            quadrature_degree
        )

        self.inducing_locations_km = (
            tf.convert_to_tensor(
                inducing_locations_km,
                dtype=dtype,
            )
        )

        if (
            self.inducing_locations_km.shape.rank != 2
            or self.inducing_locations_km.shape[-1] != 2
        ):
            raise ValueError(
                "Inducing locations must have shape [M_s, 2]."
            )

        self.n_spatial = int(
            self.inducing_locations_km.shape[0]
        )
        self.n_features = int(n_features)
        self.temporal_lengthscale_trainable = bool(
            temporal_lengthscale_trainable
        )

        self.beta0 = tf.Variable(
            0.0,
            dtype=dtype,
            name="linear_intercept",
        )
        self.beta = tf.Variable(
            tf.zeros(
                [n_features],
                dtype=dtype,
            ),
            name="linear_coefficients",
        )

        self.log_spatial_lengthscales = tf.Variable(
            tf.math.log(
                tf.fill(
                    [2],
                    tf.cast(
                        spatial_initial_lengthscale_km,
                        dtype,
                    ),
                )
            ),
            name="log_spatial_lengthscales",
        )

        self.log_temporal_lengthscale = tf.Variable(
            tf.math.log(
                tf.cast(
                    temporal_initial_lengthscale_steps,
                    dtype,
                )
            ),
            trainable=self.temporal_lengthscale_trainable,
            name="log_temporal_lengthscale",
        )

        self.log_variance = tf.Variable(
            tf.math.log(
                tf.cast(
                    variance_initial,
                    dtype,
                )
            ),
            name="log_kernel_variance",
        )

    @property
    def spatial_lengthscales(self) -> tf.Tensor:
        return tf.exp(
            self.log_spatial_lengthscales
        )

    @property
    def temporal_lengthscale(self) -> tf.Tensor:
        return tf.exp(
            self.log_temporal_lengthscale
        )

    @property
    def variance(self) -> tf.Tensor:
        return tf.exp(
            self.log_variance
        )

    @property
    def trainable_variables(
        self,
    ) -> list[tf.Variable]:
        variables = [
            self.beta0,
            self.beta,
            self.log_spatial_lengthscales,
            self.log_variance,
        ]

        if self.temporal_lengthscale_trainable:
            variables.insert(
                3,
                self.log_temporal_lengthscale,
            )

        return variables

    def spatial_covariance(self) -> tf.Tensor:
        return matern32_spatial_covariance(
            self.inducing_locations_km,
            self.inducing_locations_km,
            lengthscales_km=(
                self.spatial_lengthscales
            ),
            variance=self.variance,
            dtype=self.dtype,
        )

    def linear_mean(
        self,
        features: tf.Tensor,
    ) -> tf.Tensor:
        features = tf.convert_to_tensor(
            features,
            dtype=self.dtype,
        )
        return (
            self.beta0
            + tf.linalg.matvec(
                features,
                self.beta,
            )
        )

    def posterior(
        self,
        *,
        times: tf.Tensor,
        sites: DenseCviSites,
    ) -> STPosterior:
        """Compute q(u_1:T) through dense-block filtering/smoothing."""
        kzz = self.spatial_covariance()

        pseudo_y, pseudo_v, _ = (
            dense_sites_to_gaussian(
                sites,
                jitter=self.jitter,
            )
        )

        filtered = block_kalman_filter(
            times=times,
            pseudo_observations=pseudo_y,
            pseudo_covariances=pseudo_v,
            spatial_covariance=kzz,
            temporal_lengthscale=(
                self.temporal_lengthscale
            ),
            jitter=self.jitter,
            dtype=self.dtype,
        )

        smoothed = block_rts_smoother(
            times=times,
            filter_result=filtered,
            spatial_covariance=kzz,
            temporal_lengthscale=(
                self.temporal_lengthscale
            ),
            jitter=self.jitter,
            dtype=self.dtype,
        )

        inducing_mean, inducing_covariance = (
            inducing_function_marginals(
                state_means=smoothed.smoothed_means,
                state_covariances=(
                    smoothed.smoothed_covariances
                ),
                n_spatial=self.n_spatial,
                temporal_lengthscale=(
                    self.temporal_lengthscale
                ),
                dtype=self.dtype,
            )
        )

        return STPosterior(
            filter_result=filtered,
            smoother_result=smoothed,
            inducing_means=inducing_mean,
            inducing_covariances=(
                inducing_covariance
            ),
            spatial_covariance=kzz,
        )

    def latent_marginals(
        self,
        *,
        features: tf.Tensor,
        coordinates_km: tf.Tensor,
        inducing_mean: tf.Tensor,
        inducing_covariance: tf.Tensor,
        spatial_covariance: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return q(f_i,t) mean/variance for one time-specific batch."""
        residual_mean, residual_variance, _ = (
            spatial_conditional(
                points_km=coordinates_km,
                inducing_km=(
                    self.inducing_locations_km
                ),
                inducing_covariance=(
                    spatial_covariance
                ),
                inducing_mean=inducing_mean,
                inducing_posterior_covariance=(
                    inducing_covariance
                ),
                lengthscales_km=(
                    self.spatial_lengthscales
                ),
                variance=self.variance,
                jitter=self.jitter,
                dtype=self.dtype,
            )
        )

        return (
            self.linear_mean(features)
            + residual_mean,
            residual_variance,
        )

    def expected_log_likelihood(
        self,
        *,
        targets: tf.Tensor,
        features: tf.Tensor,
        coordinates_km: tf.Tensor,
        inducing_mean: tf.Tensor,
        inducing_covariance: tf.Tensor,
        spatial_covariance: tf.Tensor,
    ) -> tf.Tensor:
        """Return one Bernoulli-probit expected log likelihood per row."""
        latent_mean, latent_variance = (
            self.latent_marginals(
                features=features,
                coordinates_km=coordinates_km,
                inducing_mean=inducing_mean,
                inducing_covariance=(
                    inducing_covariance
                ),
                spatial_covariance=(
                    spatial_covariance
                ),
            )
        )

        return expected_log_bernoulli_probit(
            targets=targets,
            marginal_mean=latent_mean,
            marginal_variance=latent_variance,
            quadrature_degree=(
                self.quadrature_degree
            ),
            dtype=self.dtype,
        )

    def cvi_natural_targets(
        self,
        *,
        targets: tf.Tensor,
        features: tf.Tensor,
        coordinates_km: tf.Tensor,
        inducing_mean: tf.Tensor,
        inducing_covariance: tf.Tensor,
        spatial_covariance: tf.Tensor,
        likelihood_scale: float,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Natural-parameter target for one dense time-specific CVI block."""
        current_mean = tf.identity(
            tf.convert_to_tensor(
                inducing_mean,
                dtype=self.dtype,
            )
        )
        current_covariance = tf.identity(
            tf.convert_to_tensor(
                inducing_covariance,
                dtype=self.dtype,
            )
        )

        with tf.GradientTape(
            watch_accessed_variables=False
        ) as tape:
            tape.watch(
                [
                    current_mean,
                    current_covariance,
                ]
            )

            ell = self.expected_log_likelihood(
                targets=targets,
                features=features,
                coordinates_km=coordinates_km,
                inducing_mean=current_mean,
                inducing_covariance=(
                    current_covariance
                ),
                spatial_covariance=(
                    spatial_covariance
                ),
            )

            objective = (
                tf.cast(
                    likelihood_scale,
                    self.dtype,
                )
                * tf.reduce_sum(ell)
            )

        gradient_mean, gradient_covariance = (
            tape.gradient(
                objective,
                [
                    current_mean,
                    current_covariance,
                ],
            )
        )

        if (
            gradient_mean is None
            or gradient_covariance is None
        ):
            raise RuntimeError(
                "Could not differentiate the ST-SVGP "
                "Bernoulli-probit expected likelihood."
            )

        gradient_covariance = 0.5 * (
            gradient_covariance
            + tf.transpose(
                gradient_covariance
            )
        )

        target_lambda2 = gradient_covariance
        target_lambda1 = (
            gradient_mean
            - 2.0
            * tf.linalg.matvec(
                gradient_covariance,
                current_mean,
            )
        )

        return (
            target_lambda1,
            target_lambda2,
        )

    def stochastic_elbo(
        self,
        *,
        times: tf.Tensor,
        sites: DenseCviSites,
        batches: list[
            tuple[
                tf.Tensor,
                tf.Tensor,
                tf.Tensor,
                float,
            ]
        ],
    ) -> tuple[tf.Tensor, STPosterior]:
        """Compute stochastic ST-SVGP ELBO for one batch per training time."""
        posterior = self.posterior(
            times=times,
            sites=sites,
        )

        expected_true = tf.cast(
            0.0,
            self.dtype,
        )

        for index, (
            features,
            coordinates,
            targets,
            likelihood_scale,
        ) in enumerate(batches):
            ell = self.expected_log_likelihood(
                targets=targets,
                features=features,
                coordinates_km=coordinates,
                inducing_mean=(
                    posterior.inducing_means[
                        index
                    ]
                ),
                inducing_covariance=(
                    posterior.inducing_covariances[
                        index
                    ]
                ),
                spatial_covariance=(
                    posterior.spatial_covariance
                ),
            )

            expected_true = (
                expected_true
                + tf.cast(
                    likelihood_scale,
                    self.dtype,
                )
                * tf.reduce_sum(ell)
            )

        expected_sites = expected_log_dense_sites(
            sites=sites,
            inducing_means=(
                posterior.inducing_means
            ),
            inducing_covariances=(
                posterior.inducing_covariances
            ),
            jitter=self.jitter,
            dtype=self.dtype,
        )

        elbo = (
            expected_true
            - expected_sites
            + posterior.filter_result
            .log_marginal_likelihood
        )

        return elbo, posterior

    def forward_inducing_state(
        self,
        *,
        posterior: STPosterior,
        delta: float,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Predict q(u_*) forward from the last smoothed training state."""
        kzz = posterior.spatial_covariance

        transition, process_noise, observation = (
            temporal_block_matrices(
                spatial_covariance=kzz,
                delta=tf.cast(
                    delta,
                    self.dtype,
                ),
                temporal_lengthscale=(
                    self.temporal_lengthscale
                ),
                dtype=self.dtype,
            )
        )

        last_mean = (
            posterior.smoother_result
            .smoothed_means[-1]
        )
        last_covariance = (
            posterior.smoother_result
            .smoothed_covariances[-1]
        )

        state_mean = tf.linalg.matvec(
            transition,
            last_mean,
        )
        state_covariance = (
            transition
            @ last_covariance
            @ tf.transpose(transition)
            + process_noise
        )

        inducing_mean = tf.linalg.matvec(
            observation,
            state_mean,
        )
        inducing_covariance = (
            observation
            @ state_covariance
            @ tf.transpose(observation)
        )

        return (
            inducing_mean,
            0.5
            * (
                inducing_covariance
                + tf.transpose(
                    inducing_covariance
                )
            ),
        )

    def predict_probability(
        self,
        *,
        features: tf.Tensor,
        coordinates_km: tf.Tensor,
        inducing_mean: tf.Tensor,
        inducing_covariance: tf.Tensor,
        spatial_covariance: tf.Tensor,
    ) -> tf.Tensor:
        """Integrated Bernoulli-probit predictive probability."""
        latent_mean, latent_variance = (
            self.latent_marginals(
                features=features,
                coordinates_km=coordinates_km,
                inducing_mean=inducing_mean,
                inducing_covariance=(
                    inducing_covariance
                ),
                spatial_covariance=(
                    spatial_covariance
                ),
            )
        )

        return probit_predictive_probability(
            marginal_mean=latent_mean,
            marginal_variance=latent_variance,
            dtype=self.dtype,
        )
