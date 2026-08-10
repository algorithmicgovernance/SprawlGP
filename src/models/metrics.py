"""Probabilistic evaluation metrics shared by modelling baselines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def validate_probabilities(probabilities: np.ndarray) -> None:
    """Reject non-finite probabilities or values outside [0, 1]."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional.")
    if not np.isfinite(values).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Predicted probabilities must lie in [0, 1].")


def probabilistic_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Return Log Loss, Brier score, PR-AUC and ROC-AUC."""
    y_true = np.asarray(targets, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    validate_probabilities(y_prob)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Both target classes are required for validation metrics.")
    clipped = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
    return {
        "log_loss": float(log_loss(y_true, clipped)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
