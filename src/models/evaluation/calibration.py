"""Calibration diagnostics for binary probabilistic forecasts.

Coverage is intentionally NOT implemented here.

For a binary classifier, calibration asks whether predicted probabilities
match observed frequencies. For example, among cases assigned probability
near 0.8, roughly 80% should be positive in a well-calibrated model.

This is different from interval/prediction-set coverage, which should be
clarified with the supervisor before implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

PROBABILITY_EPSILON = 1.0e-6


@dataclass(frozen=True)
class FittedBinaryCalibrator:
    """Fitted Platt or Beta mapping for binary probabilities."""

    method: str
    coefficient_a: float
    coefficient_b: float
    coefficient_c: float | None = None

    def predict(self, probability: np.ndarray) -> np.ndarray:
        """Apply the fitted mapping to raw probabilities."""
        _, p = _as_binary_arrays(np.zeros(len(probability)), probability)
        clipped = np.clip(p, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        if self.method == "platt":
            linear_predictor = self.coefficient_a + self.coefficient_b * np.log(
                clipped / (1.0 - clipped)
            )
        elif self.method == "beta":
            if self.coefficient_c is None:
                raise ValueError("Beta calibration requires coefficient_c.")
            linear_predictor = (
                self.coefficient_a * np.log(clipped)
                + self.coefficient_b * np.log1p(-clipped)
                + self.coefficient_c
            )
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return expit(linear_predictor)

    def monotonic_direction(self) -> str:
        """Return the mapping direction over the numerically clipped domain."""
        if self.method == "platt":
            if self.coefficient_b > 0.0:
                return "increasing"
            if self.coefficient_b < 0.0:
                return "decreasing"
            return "constant"
        if self.method != "beta":
            raise ValueError(f"Unknown calibration method: {self.method}")

        endpoints = np.array([PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON])
        derivative_numerator = self.coefficient_a - (
            self.coefficient_a + self.coefficient_b
        ) * endpoints
        if np.all(derivative_numerator >= 0.0):
            return "increasing"
        if np.all(derivative_numerator <= 0.0):
            return "decreasing"
        return "non_monotonic"


def _as_binary_arrays(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary labels and finite probabilities."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)

    if y.shape != p.shape:
        raise ValueError("Labels and probabilities must have the same shape.")
    if set(np.unique(y)).difference({0, 1}):
        raise ValueError("Calibration requires binary labels.")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN or Inf.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")

    return y, p


def _fit_logistic_calibrator(
    y_true: np.ndarray,
    probability: np.ndarray,
    method: str,
) -> FittedBinaryCalibrator:
    """Fit one unweighted, unregularized logistic calibration mapping."""
    y, p = _as_binary_arrays(y_true, probability)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibration fitting requires both target classes.")

    clipped = np.clip(p, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    if method == "platt":
        predictors = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    elif method == "beta":
        predictors = np.column_stack((np.log(clipped), np.log1p(-clipped)))
    else:
        raise ValueError("method must be 'platt' or 'beta'.")

    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=500,
    )
    model.fit(predictors, y)

    if method == "platt":
        return FittedBinaryCalibrator(
            method=method,
            coefficient_a=float(model.intercept_[0]),
            coefficient_b=float(model.coef_[0, 0]),
        )
    return FittedBinaryCalibrator(
        method=method,
        coefficient_a=float(model.coef_[0, 0]),
        coefficient_b=float(model.coef_[0, 1]),
        coefficient_c=float(model.intercept_[0]),
    )


def fit_platt_calibrator(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> FittedBinaryCalibrator:
    """Fit logit(q) = a + b * logit(p)."""
    return _fit_logistic_calibrator(y_true, probability, method="platt")


def fit_beta_calibrator(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> FittedBinaryCalibrator:
    """Fit logit(q) = a * log(p) + b * log(1-p) + c."""
    return _fit_logistic_calibrator(y_true, probability, method="beta")


def reliability_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Build reliability bins for observed-vs-predicted probability.

    Quantile bins are preferred for this imbalanced transition problem because
    each bin receives comparable support instead of leaving high-probability
    bins nearly empty.
    """
    y, p = _as_binary_arrays(y_true, probability)

    if strategy == "quantile":
        # duplicates='drop' handles repeated probabilities safely.
        bins = pd.qcut(
            p,
            q=int(n_bins),
            labels=False,
            duplicates="drop",
        )
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        bins = np.clip(
            np.digitize(p, edges[1:-1], right=True),
            0,
            int(n_bins) - 1,
        )
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'.")

    frame = pd.DataFrame(
        {
            "target": y,
            "probability": p,
            "bin": bins,
        }
    )

    result = (
        frame.groupby("bin", observed=True)
        .agg(
            cell_count=("target", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_conversion_rate=("target", "mean"),
        )
        .reset_index()
    )

    result["absolute_calibration_gap"] = (
        result["mean_predicted_probability"]
        - result["observed_conversion_rate"]
    ).abs()
    return result


def expected_calibration_error(
    reliability: pd.DataFrame,
) -> float:
    """Compute support-weighted expected calibration error (ECE)."""
    total = float(reliability["cell_count"].sum())
    if total == 0:
        return float("nan")

    weights = reliability["cell_count"].to_numpy(dtype=float) / total
    gaps = reliability["absolute_calibration_gap"].to_numpy(dtype=float)
    return float(np.sum(weights * gaps))


def calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Estimate logistic calibration intercept and slope.

    The diagnostic model is:

        logit P(Y=1) = intercept + slope * logit(p_hat)

    Ideal values are intercept=0 and slope=1.
    """
    y, p = _as_binary_arrays(y_true, probability)
    clipped = np.clip(p, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=500,
    )
    model.fit(logits, y)

    return (
        float(model.intercept_[0]),
        float(model.coef_[0, 0]),
    )


def calibration_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict[str, float]:
    """Return calibration metrics used across LR, SVGP and future ST-SVGP."""
    y, p = _as_binary_arrays(y_true, probability)
    reliability = reliability_table(
        y,
        p,
        n_bins=n_bins,
        strategy=strategy,
    )
    intercept, slope = calibration_intercept_slope(y, p)
    bias = float(p.mean() - y.mean())

    return {
        "ece": expected_calibration_error(reliability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "probability_bias": bias,
        "absolute_probability_bias": abs(bias),
    }
