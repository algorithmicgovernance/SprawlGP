"""Validate the Matérn-3/2 temporal state-space representation.

This is the first ST-SVGP gate. It intentionally performs no classification,
no CVI updates, and no training.

The experiment checks that:
1. the state-space covariance equals the direct Matérn-3/2 covariance;
2. the analytic transition equals expm(F * delta);
3. the discrete process noise preserves the stationary covariance;
4. every process-noise covariance is positive semidefinite;
5. when GPflow is available, its Matérn32 kernel agrees with the same
   covariance convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
import tensorflow as tf
import yaml

from src.models.st_svgp.state_space import (
    matern32_continuous_matrices,
    matern32_covariance,
    matern32_process_noise,
    matern32_state_space_covariance,
    matern32_stationary_covariance,
    matern32_transition_matrix,
)


def load_config(path: Path) -> dict[str, Any]:
    """Load temporal-kernel validation configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("Validation config must be a mapping.")

    if config.get("experiment") != "temporal_kernel_validation":
        raise ValueError(
            "Expected experiment: temporal_kernel_validation."
        )

    return config


def resolve_dtype(name: str) -> tf.dtypes.DType:
    """Resolve the configured TensorFlow floating dtype."""
    if name == "float64":
        return tf.float64
    if name == "float32":
        return tf.float32
    raise ValueError(
        "dtype must be either 'float64' or 'float32'."
    )


def _maximum_absolute_error(
    left: tf.Tensor,
    right: tf.Tensor,
) -> float:
    """Return max |left-right| as a Python float."""
    return float(
        tf.reduce_max(
            tf.abs(left - right)
        ).numpy()
    )


def run_validation(
    config_path: Path,
) -> dict[str, Any]:
    """Run all temporal-kernel validation checks."""
    config = load_config(config_path)
    dtype = resolve_dtype(str(config["dtype"]))

    kernel_config = config["temporal_kernel"]
    lengthscale = float(
        kernel_config["lengthscale_steps"]
    )
    variance = float(
        kernel_config["variance"]
    )

    times = tf.convert_to_tensor(
        config["time_grid_steps"],
        dtype=dtype,
    )

    tolerances = config["tolerances"]
    covariance_atol = float(
        tolerances["covariance_atol"]
    )
    transition_atol = float(
        tolerances["transition_atol"]
    )
    stationary_atol = float(
        tolerances["stationary_atol"]
    )
    psd_tol = float(
        tolerances["psd_eigenvalue_tol"]
    )
    gpflow_atol = float(
        tolerances["gpflow_atol"]
    )

    direct_covariance = matern32_covariance(
        times,
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )
    state_space_covariance = (
        matern32_state_space_covariance(
            times,
            lengthscale=lengthscale,
            variance=variance,
            dtype=dtype,
        )
    )

    covariance_error = _maximum_absolute_error(
        direct_covariance,
        state_space_covariance,
    )

    feedback, _, spectral_density, _ = (
        matern32_continuous_matrices(
            lengthscale=lengthscale,
            variance=variance,
            dtype=dtype,
        )
    )
    stationary = matern32_stationary_covariance(
        lengthscale=lengthscale,
        variance=variance,
        dtype=dtype,
    )

    maximum_transition_error = 0.0
    maximum_stationary_error = 0.0
    minimum_process_noise_eigenvalue = float("inf")

    delta_results: list[dict[str, float]] = []

    for raw_delta in config["check_deltas"]:
        delta = tf.convert_to_tensor(
            float(raw_delta),
            dtype=dtype,
        )

        analytic_transition = matern32_transition_matrix(
            delta,
            lengthscale=lengthscale,
            dtype=dtype,
        )
        # expm_transition = tf.linalg.expm(
        #     feedback * delta
        # )

        # transition_error = _maximum_absolute_error(
        #     analytic_transition,
        #     expm_transition,
        # )
        
        expm_transition = scipy.linalg.expm(
            feedback.numpy() * float(delta.numpy())
        )

        transition_error = float(
            np.max(
                np.abs(
                    analytic_transition.numpy()
                    - expm_transition
                )
            )
        )

        process_noise = matern32_process_noise(
            delta,
            lengthscale=lengthscale,
            variance=variance,
            dtype=dtype,
        )

        reconstructed_stationary = (
            analytic_transition
            @ stationary
            @ tf.transpose(analytic_transition)
            + process_noise
        )

        stationary_error = _maximum_absolute_error(
            stationary,
            reconstructed_stationary,
        )

        minimum_eigenvalue = float(
            tf.reduce_min(
                tf.linalg.eigvalsh(process_noise)
            ).numpy()
        )

        maximum_transition_error = max(
            maximum_transition_error,
            transition_error,
        )
        maximum_stationary_error = max(
            maximum_stationary_error,
            stationary_error,
        )
        minimum_process_noise_eigenvalue = min(
            minimum_process_noise_eigenvalue,
            minimum_eigenvalue,
        )

        delta_results.append(
            {
                "delta": float(raw_delta),
                "transition_expm_max_abs_error": transition_error,
                "stationary_identity_max_abs_error": stationary_error,
                "process_noise_min_eigenvalue": minimum_eigenvalue,
            }
        )

    gpflow_available = False
    gpflow_error: float | None = None
    gpflow_message: str | None = None

    try:
        import gpflow

        gpflow_available = True
        gpflow.config.set_default_float(dtype)

        gpflow_kernel = gpflow.kernels.Matern32(
            variance=variance,
            lengthscales=lengthscale,
        )

        gpflow_covariance = gpflow_kernel(
            tf.reshape(times, [-1, 1])
        )

        gpflow_error = _maximum_absolute_error(
            direct_covariance,
            gpflow_covariance,
        )

    except ImportError:
        gpflow_message = "GPflow is not installed."

        if bool(config.get("require_gpflow", False)):
            raise

    checks = {
        "state_space_matches_direct_kernel": (
            covariance_error <= covariance_atol
        ),
        "transition_matches_matrix_exponential": (
            maximum_transition_error <= transition_atol
        ),
        "stationary_covariance_is_preserved": (
            maximum_stationary_error <= stationary_atol
        ),
        "process_noise_is_positive_semidefinite": (
            minimum_process_noise_eigenvalue >= -psd_tol
        ),
    }

    if gpflow_available:
        checks["gpflow_matches_direct_kernel"] = (
            gpflow_error is not None
            and gpflow_error <= gpflow_atol
        )

    passed = all(checks.values())

    result: dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "experiment": "temporal_kernel_validation",
        "dtype": dtype.name,
        "temporal_kernel": {
            "family": "matern32_markov",
            "lengthscale_steps": lengthscale,
            "variance": variance,
            "state_dimension": 2,
            "white_noise_spectral_density": float(
                spectral_density.numpy()
            ),
        },
        "time_grid_steps": [
            float(value)
            for value in config["time_grid_steps"]
        ],
        "checks": checks,
        "errors": {
            "state_space_vs_direct_max_abs_error": covariance_error,
            "transition_vs_expm_max_abs_error": maximum_transition_error,
            "stationary_identity_max_abs_error": maximum_stationary_error,
            "minimum_process_noise_eigenvalue": (
                minimum_process_noise_eigenvalue
            ),
            "gpflow_vs_direct_max_abs_error": gpflow_error,
        },
        "gpflow": {
            "available": gpflow_available,
            "message": gpflow_message,
        },
        "delta_checks": delta_results,
        "direct_covariance": direct_covariance.numpy().tolist(),
        "state_space_covariance": (
            state_space_covariance.numpy().tolist()
        ),
    }

    output_path = Path(
        config["output"]["path"]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
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
            "temporal_kernel_validation.yaml"
        ),
    )

    args = parser.parse_args()

    result = run_validation(
        args.config
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
