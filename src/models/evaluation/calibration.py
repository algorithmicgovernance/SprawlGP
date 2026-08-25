"""Calibration diagnostics for binary probabilistic forecasts.

Coverage is intentionally NOT implemented here.

For a binary classifier, calibration asks whether predicted probabilities
match observed frequencies. For example, among cases assigned probability
near 0.8, roughly 80% should be positive in a well-calibrated model.

This is different from interval/prediction-set coverage, which should be
clarified with the supervisor before implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


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
    eps = 1.0e-6
    clipped = np.clip(p, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    model = LogisticRegression(
        C=np.inf,
        penalty=None,
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
