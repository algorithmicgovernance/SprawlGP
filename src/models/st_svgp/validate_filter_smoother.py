"""Validate Kalman filtering and RTS smoothing against exact GP regression.

The direct GP posterior is the oracle. The filter/smoother uses the Markov
state-space representation. Agreement verifies that both inference routes
represent the same Gaussian Matérn-3/2 model.

This experiment deliberately stops before Bernoulli-probit, CVI, natural
gradients, and the ST-SVGP training loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

from src.models.st_svgp.filtering import (
    function_marginals_from_states,
    kalman_filter_matern32,
    rts_smoother_matern32,
)
from src.models.st_svgp.state_space import matern32_covariance


def load_config(path: Path) -> dict[str, Any]:
    """Load the Gaussian filter/smoother validation config."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("Validation config must be a mapping.")

    if config.get("experiment") != "filter_smoother_validation":
        raise ValueError(
            "Expected experiment: filter_smoother_validation."
        )

    return config


def resolve_dtype(name: str) -> tf.dtypes.DType:
    """Resolve configured TensorFlow dtype."""
    if name == "float64":
        return tf.float64
    if name == "float32":
        return tf.float32
    raise ValueError("dtype must be 'float64' or 'float32'.")


def exact_gp_posterior(
    *,
    times: tf.Tensor,
    observations: tf.Tensor,
    observation_noise_variance: float | tf.Tensor,
    lengthscale: float | tf.Tensor,
    variance: float | tf.Tensor,
    dtype: tf.dtypes.DType,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute exact Gaussian GP posterior at the observed times."""
    y = tf.reshape(
        tf.convert_to_tensor(observations, dtype=dtype),
        [-1, 1],
    )

    covariance = matern32_covariance(
        times,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    n = tf.shape(covariance)[0]
    noise = tf.convert_to_tensor(
        observation_noise_variance,
        dtype=dtype,
    )

    noisy_covariance = (
        covariance
        + noise * tf.eye(n, dtype=dtype)
    )

    chol = tf.linalg.cholesky(noisy_covariance)
    alpha = tf.linalg.cholesky_solve(chol, y)

    posterior_mean = tf.reshape(
        covariance @ alpha,
        [-1],
    )

    solved_covariance = tf.linalg.cholesky_solve(
        chol,
        covariance,
    )
    posterior_covariance = (
        covariance
        - covariance @ solved_covariance
    )
    posterior_covariance = 0.5 * (
        posterior_covariance
        + tf.transpose(posterior_covariance)
    )

    log_two_pi = tf.math.log(
        tf.cast(2.0, dtype)
        * tf.constant(3.141592653589793, dtype=dtype)
    )

    log_marginal = (
        -0.5 * tf.reshape(tf.transpose(y) @ alpha, [])
        - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        - 0.5 * tf.cast(n, dtype) * log_two_pi
    )

    return (
        posterior_mean,
        posterior_covariance,
        log_marginal,
    )


def _max_abs(left: tf.Tensor, right: tf.Tensor) -> float:
    """Return max absolute tensor difference."""
    return float(
        tf.reduce_max(
            tf.abs(left - right)
        ).numpy()
    )


def run_validation(config_path: Path) -> dict[str, Any]:
    """Run the Gaussian filtering/smoothing validation experiment."""
    config = load_config(config_path)
    dtype = resolve_dtype(str(config["dtype"]))

    kernel = config["temporal_kernel"]
    lengthscale = float(kernel["lengthscale_steps"])
    variance = float(kernel["variance"])

    synthetic = config["synthetic_data"]
    times = tf.convert_to_tensor(
        synthetic["times"],
        dtype=dtype,
    )
    observations = tf.convert_to_tensor(
        synthetic["observations"],
        dtype=dtype,
    )
    noise_variance = float(
        synthetic["observation_noise_variance"]
    )

    filter_result = kalman_filter_matern32(
        times=times,
        observations=observations,
        observation_noise_variance=noise_variance,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    smoother_result = rts_smoother_matern32(
        times=times,
        filter_result=filter_result,
        lengthscale=lengthscale,
        dtype=dtype,
    )

    smoothed_mean, smoothed_variance = (
        function_marginals_from_states(
            state_means=smoother_result.smoothed_means,
            state_covariances=smoother_result.smoothed_covariances,
        )
    )

    filtered_mean, filtered_variance = (
        function_marginals_from_states(
            state_means=filter_result.filtered_means,
            state_covariances=filter_result.filtered_covariances,
        )
    )

    (
        direct_mean,
        direct_covariance,
        direct_log_marginal,
    ) = exact_gp_posterior(
        times=times,
        observations=observations,
        observation_noise_variance=noise_variance,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    direct_variance = tf.linalg.diag_part(
        direct_covariance
    )

    mean_error = _max_abs(
        smoothed_mean,
        direct_mean,
    )
    variance_error = _max_abs(
        smoothed_variance,
        direct_variance,
    )
    log_marginal_error = abs(
        float(
            filter_result.log_marginal_likelihood.numpy()
        )
        - float(direct_log_marginal.numpy())
    )

    filtered_min_eigenvalue = float(
        tf.reduce_min(
            tf.stack(
                [
                    tf.reduce_min(
                        tf.linalg.eigvalsh(covariance)
                    )
                    for covariance in filter_result.filtered_covariances
                ]
            )
        ).numpy()
    )

    smoothed_min_eigenvalue = float(
        tf.reduce_min(
            tf.stack(
                [
                    tf.reduce_min(
                        tf.linalg.eigvalsh(covariance)
                    )
                    for covariance in smoother_result.smoothed_covariances
                ]
            )
        ).numpy()
    )

    # Conditioning on future observations should never increase a marginal
    # variance relative to the filtered distribution at the same time.
    maximum_smoothing_variance_increase = float(
        tf.reduce_max(
            smoothed_variance - filtered_variance
        ).numpy()
    )

    tolerances = config["tolerances"]

    checks = {
        "smoothed_mean_matches_exact_gp": (
            mean_error
            <= float(tolerances["posterior_mean_atol"])
        ),
        "smoothed_variance_matches_exact_gp": (
            variance_error
            <= float(tolerances["posterior_variance_atol"])
        ),
        "filter_log_marginal_matches_exact_gp": (
            log_marginal_error
            <= float(
                tolerances["log_marginal_likelihood_atol"]
            )
        ),
        "filtered_covariances_are_psd": (
            filtered_min_eigenvalue
            >= -float(tolerances["covariance_psd_tol"])
        ),
        "smoothed_covariances_are_psd": (
            smoothed_min_eigenvalue
            >= -float(tolerances["covariance_psd_tol"])
        ),
        "smoothing_does_not_increase_function_variance": (
            maximum_smoothing_variance_increase
            <= float(tolerances["posterior_variance_atol"])
        ),
    }

    passed = all(checks.values())

    result = {
        "status": "PASS" if passed else "FAIL",
        "experiment": "filter_smoother_validation",
        "dtype": dtype.name,
        "temporal_kernel": {
            "family": "matern32_markov",
            "lengthscale_steps": lengthscale,
            "variance": variance,
            "state_dimension": 2,
        },
        "synthetic_data": {
            "times": [
                float(value)
                for value in synthetic["times"]
            ],
            "observations": [
                float(value)
                for value in synthetic["observations"]
            ],
            "observation_noise_variance": noise_variance,
        },
        "checks": checks,
        "errors": {
            "smoothed_mean_vs_exact_gp_max_abs_error": mean_error,
            "smoothed_variance_vs_exact_gp_max_abs_error": variance_error,
            "filter_vs_exact_log_marginal_abs_error": log_marginal_error,
            "filtered_covariance_min_eigenvalue": (
                filtered_min_eigenvalue
            ),
            "smoothed_covariance_min_eigenvalue": (
                smoothed_min_eigenvalue
            ),
            "maximum_smoothing_variance_increase": (
                maximum_smoothing_variance_increase
            ),
        },
        "log_marginal_likelihood": {
            "kalman_filter": float(
                filter_result.log_marginal_likelihood.numpy()
            ),
            "exact_gp": float(
                direct_log_marginal.numpy()
            ),
        },
        "marginals": {
            "filtered_mean": filtered_mean.numpy().tolist(),
            "filtered_variance": filtered_variance.numpy().tolist(),
            "smoothed_mean": smoothed_mean.numpy().tolist(),
            "smoothed_variance": smoothed_variance.numpy().tolist(),
            "exact_gp_mean": direct_mean.numpy().tolist(),
            "exact_gp_variance": direct_variance.numpy().tolist(),
        },
    }

    output_path = Path(
        config["output"]["path"]
    )
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
            "filter_smoother_validation.yaml"
        ),
    )

    args = parser.parse_args()

    result = run_validation(args.config)

    print(json.dumps(result, indent=2))

    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
