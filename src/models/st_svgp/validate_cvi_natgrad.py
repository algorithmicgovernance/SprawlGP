"""Validate Bernoulli-probit CVI / natural-gradient inference.

This is the final mathematical gate before the production spatial ST-SVGP.

Fixed in this experiment:
- Matérn-3/2 temporal GP prior;
- Gaussian state-space filtering + RTS smoothing;
- Bernoulli-probit true likelihood;
- Gaussian CVI pseudo-likelihood factors;
- damped natural-gradient updates.

Not included yet:
- spatial inducing points;
- parametric urban predictors;
- Adam hyperparameter optimisation;
- real dataset / rolling validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

from src.models.st_svgp.cvi import (
    cvi_moment_gradient_targets,
    cvi_state_space_elbo,
    dense_gaussian_site_posterior,
    dense_standard_elbo,
    initialise_cvi_sites,
    natural_gradient_site_update,
    posterior_from_cvi_sites,
    probit_predictive_probability,
)


def load_config(path: Path) -> dict[str, Any]:
    """Load validation configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("Validation config must be a mapping.")

    if config.get("experiment") != "cvi_natgrad_validation":
        raise ValueError(
            "Expected experiment: cvi_natgrad_validation."
        )

    return config


def resolve_dtype(name: str) -> tf.dtypes.DType:
    """Resolve configured TensorFlow dtype."""
    if name == "float64":
        return tf.float64
    if name == "float32":
        return tf.float32
    raise ValueError("dtype must be 'float64' or 'float32'.")


def _max_abs(left: tf.Tensor, right: tf.Tensor) -> float:
    """Return maximum absolute difference."""
    return float(
        tf.reduce_max(
            tf.abs(left - right)
        ).numpy()
    )


def run_validation(config_path: Path) -> dict[str, Any]:
    """Run the CVI/Natural-Gradient Bernoulli-probit validation."""
    config = load_config(config_path)
    dtype = resolve_dtype(str(config["dtype"]))

    kernel = config["temporal_kernel"]
    lengthscale = float(
        kernel["lengthscale_steps"]
    )
    variance = float(
        kernel["variance"]
    )

    synthetic = config["synthetic_binary_data"]
    times = tf.constant(
        synthetic["times"],
        dtype=dtype,
    )
    targets = tf.constant(
        synthetic["targets"],
        dtype=dtype,
    )

    cvi_config = config["cvi"]
    quadrature_degree = int(
        cvi_config["quadrature_degree"]
    )
    gamma = float(
        cvi_config["natural_gradient_gamma"]
    )
    initial_precision = float(
        cvi_config["initial_site_precision"]
    )
    iterations = int(
        cvi_config["iterations"]
    )

    sites = initialise_cvi_sites(
        len(synthetic["times"]),
        initial_precision=initial_precision,
        dtype=dtype,
    )

    history: list[dict[str, float]] = []

    maximum_posterior_mean_error = 0.0
    maximum_posterior_variance_error = 0.0
    maximum_site_log_marginal_error = 0.0
    maximum_elbo_identity_error = 0.0
    minimum_site_precision = float("inf")

    for iteration in range(iterations):
        posterior = posterior_from_cvi_sites(
            times=times,
            sites=sites,
            lengthscale=lengthscale,
            variance=variance,
            dtype=dtype,
        )

        dense_mean, dense_covariance, dense_log_marginal = (
            dense_gaussian_site_posterior(
                times=times,
                sites=sites,
                lengthscale=lengthscale,
                variance=variance,
                dtype=dtype,
            )
        )

        dense_variance = tf.linalg.diag_part(
            dense_covariance
        )

        posterior_mean_error = _max_abs(
            posterior.marginal_mean,
            dense_mean,
        )
        posterior_variance_error = _max_abs(
            posterior.marginal_variance,
            dense_variance,
        )
        site_log_marginal_error = abs(
            float(
                posterior.filter_result
                .log_marginal_likelihood.numpy()
            )
            - float(dense_log_marginal.numpy())
        )

        state_space_elbo = cvi_state_space_elbo(
            targets=targets,
            posterior=posterior,
            quadrature_degree=quadrature_degree,
            dtype=dtype,
        )

        standard_elbo = dense_standard_elbo(
            targets=targets,
            times=times,
            sites=sites,
            lengthscale=lengthscale,
            variance=variance,
            quadrature_degree=quadrature_degree,
            dtype=dtype,
        )

        elbo_identity_error = abs(
            float(state_space_elbo.numpy())
            - float(standard_elbo.numpy())
        )

        precision = -2.0 * sites.lambda2
        current_min_precision = float(
            tf.reduce_min(precision).numpy()
        )

        probabilities = probit_predictive_probability(
            marginal_mean=posterior.marginal_mean,
            marginal_variance=posterior.marginal_variance,
            dtype=dtype,
        )

        history.append(
            {
                "iteration": int(iteration),
                "elbo": float(state_space_elbo.numpy()),
                "posterior_mean_max_abs_error": (
                    posterior_mean_error
                ),
                "posterior_variance_max_abs_error": (
                    posterior_variance_error
                ),
                "site_log_marginal_abs_error": (
                    site_log_marginal_error
                ),
                "elbo_identity_abs_error": (
                    elbo_identity_error
                ),
                "minimum_site_precision": (
                    current_min_precision
                ),
                "minimum_probability": float(
                    tf.reduce_min(probabilities).numpy()
                ),
                "maximum_probability": float(
                    tf.reduce_max(probabilities).numpy()
                ),
            }
        )

        maximum_posterior_mean_error = max(
            maximum_posterior_mean_error,
            posterior_mean_error,
        )
        maximum_posterior_variance_error = max(
            maximum_posterior_variance_error,
            posterior_variance_error,
        )
        maximum_site_log_marginal_error = max(
            maximum_site_log_marginal_error,
            site_log_marginal_error,
        )
        maximum_elbo_identity_error = max(
            maximum_elbo_identity_error,
            elbo_identity_error,
        )
        minimum_site_precision = min(
            minimum_site_precision,
            current_min_precision,
        )

        (
            target_lambda1,
            target_lambda2,
            _,
        ) = cvi_moment_gradient_targets(
            targets=targets,
            marginal_mean=posterior.marginal_mean,
            marginal_variance=posterior.marginal_variance,
            quadrature_degree=quadrature_degree,
            dtype=dtype,
        )

        sites = natural_gradient_site_update(
            sites=sites,
            target_lambda1=target_lambda1,
            target_lambda2=target_lambda2,
            gamma=gamma,
            dtype=dtype,
        )

    final_posterior = posterior_from_cvi_sites(
        times=times,
        sites=sites,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    final_elbo = cvi_state_space_elbo(
        targets=targets,
        posterior=final_posterior,
        quadrature_degree=quadrature_degree,
        dtype=dtype,
    )

    final_probabilities = probit_predictive_probability(
        marginal_mean=final_posterior.marginal_mean,
        marginal_variance=final_posterior.marginal_variance,
        dtype=dtype,
    )

    elbo_start = history[0]["elbo"]
    elbo_last_recorded = history[-1]["elbo"]
    elbo_final = float(final_elbo.numpy())

    elbo_differences = [
        history[index + 1]["elbo"]
        - history[index]["elbo"]
        for index in range(len(history) - 1)
    ]

    minimum_elbo_change = (
        min(elbo_differences)
        if elbo_differences
        else 0.0
    )

    tolerances = config["tolerances"]

    checks = {
        "filter_smoother_matches_dense_site_posterior": (
            maximum_posterior_mean_error
            <= float(tolerances["posterior_mean_atol"])
            and maximum_posterior_variance_error
            <= float(tolerances["posterior_variance_atol"])
        ),
        "filter_site_log_marginal_matches_dense": (
            maximum_site_log_marginal_error
            <= float(
                tolerances[
                    "gaussian_site_log_marginal_atol"
                ]
            )
        ),
        "cvi_elbo_matches_standard_variational_elbo": (
            maximum_elbo_identity_error
            <= float(tolerances["elbo_identity_atol"])
        ),
        "site_precisions_remain_positive": (
            minimum_site_precision > 0.0
        ),
        "predictive_probabilities_are_valid": bool(
            tf.reduce_all(
                (final_probabilities > 0.0)
                & (final_probabilities < 1.0)
            ).numpy()
        ),
        "natural_gradient_improves_elbo": (
            elbo_final > elbo_start
        ),
        "natural_gradient_is_monotone_for_validation_case": (
            minimum_elbo_change >= -1.0e-10
        ),
    }

    result = {
        "status": (
            "PASS"
            if all(checks.values())
            else "FAIL"
        ),
        "experiment": "cvi_natgrad_validation",
        "dtype": dtype.name,
        "temporal_kernel": {
            "family": "matern32_markov",
            "lengthscale_steps": lengthscale,
            "variance": variance,
        },
        "cvi": {
            "quadrature_degree": quadrature_degree,
            "natural_gradient_gamma": gamma,
            "initial_site_precision": initial_precision,
            "iterations": iterations,
        },
        "checks": checks,
        "errors": {
            "maximum_posterior_mean_abs_error": (
                maximum_posterior_mean_error
            ),
            "maximum_posterior_variance_abs_error": (
                maximum_posterior_variance_error
            ),
            "maximum_site_log_marginal_abs_error": (
                maximum_site_log_marginal_error
            ),
            "maximum_elbo_identity_abs_error": (
                maximum_elbo_identity_error
            ),
            "minimum_site_precision_seen": (
                minimum_site_precision
            ),
            "minimum_elbo_change": (
                minimum_elbo_change
            ),
        },
        "elbo": {
            "initial": elbo_start,
            "last_before_final_update": elbo_last_recorded,
            "final": elbo_final,
            "improvement": elbo_final - elbo_start,
        },
        "final_posterior": {
            "marginal_mean": (
                final_posterior
                .marginal_mean.numpy().tolist()
            ),
            "marginal_variance": (
                final_posterior
                .marginal_variance.numpy().tolist()
            ),
            "predictive_probability": (
                final_probabilities.numpy().tolist()
            ),
            "site_precision": (
                (-2.0 * sites.lambda2)
                .numpy().tolist()
            ),
        },
        "history": history,
    }

    output_path = Path(config["output"]["path"])
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/modeling/st_svgp/"
            "cvi_natgrad_validation.yaml"
        ),
    )
    args = parser.parse_args()

    result = run_validation(args.config)

    print(json.dumps(result, indent=2))

    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
