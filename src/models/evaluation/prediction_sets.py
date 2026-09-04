"""Conformal prediction-set coverage for binary probabilistic classifiers.

This module is intentionally separate from classical probability calibration.
It implements empirical temporal coverage diagnostics for prediction sets
constructed from OOF probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_binary_arrays(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary targets and finite probabilities."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)

    if y.shape != p.shape:
        raise ValueError("Labels and probabilities must have the same shape.")
    if set(np.unique(y)).difference({0, 1}):
        raise ValueError("Prediction-set coverage requires binary labels.")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN or Inf.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")

    return y, p


def nonconformity_scores(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    """Return binary nonconformity scores s(x,y)=1-p_hat(y|x)."""
    y, p = _as_binary_arrays(y_true, probability)
    return np.where(y == 1, 1.0 - p, p)


def conformal_quantile(
    calibration_scores: np.ndarray,
    alpha: float,
) -> float:
    """Return finite-sample conformal threshold for target coverage 1-alpha."""
    scores = np.asarray(calibration_scores, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("Calibration scores must be non-empty.")
    if not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be finite.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    n = scores.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores)[rank - 1])


def build_prediction_sets(
    probability: np.ndarray,
    q_hat: float,
) -> pd.DataFrame:
    """Build binary prediction sets Gamma(x)={y in {0,1}: s(x,y)<=q_hat}."""
    if not np.isfinite(float(q_hat)):
        raise ValueError("q_hat must be finite.")
    return _build_prediction_sets(
        probability=probability,
        q_hat_0=q_hat,
        q_hat_1=q_hat,
    )


def _build_prediction_sets(
    probability: np.ndarray,
    q_hat_0: float,
    q_hat_1: float,
) -> pd.DataFrame:
    """Build binary prediction sets with one threshold per candidate label."""
    p = np.asarray(probability, dtype=float).reshape(-1)
    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN or Inf.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.isfinite(float(q_hat_0)) or not np.isfinite(float(q_hat_1)):
        raise ValueError("Prediction-set thresholds must be finite.")

    include_0 = p <= q_hat_0
    include_1 = (1.0 - p) <= q_hat_1

    labels = []
    size = include_0.astype(int) + include_1.astype(int)
    for has_0, has_1 in zip(include_0, include_1, strict=True):
        if has_0 and has_1:
            labels.append("{0,1}")
        elif has_0:
            labels.append("{0}")
        elif has_1:
            labels.append("{1}")
        else:
            labels.append("{}")

    return pd.DataFrame(
        {
            "include_0": include_0,
            "include_1": include_1,
            "set_size": size.astype(float),
            "prediction_set": labels,
        }
    )


def build_mondrian_prediction_sets(
    probability: np.ndarray,
    q_hat_0: float,
    q_hat_1: float,
) -> pd.DataFrame:
    """Build class-conditional binary prediction sets."""
    return _build_prediction_sets(
        probability=probability,
        q_hat_0=q_hat_0,
        q_hat_1=q_hat_1,
    )


def _prediction_set_metrics(
    y: np.ndarray,
    sets: pd.DataFrame,
    target_coverage: float,
) -> dict[str, float]:
    """Summarize coverage and efficiency for constructed prediction sets."""
    covered = np.where(
        y == 1,
        sets["include_1"].to_numpy(),
        sets["include_0"].to_numpy(),
    ).astype(float)

    empirical = float(covered.mean()) if len(covered) else float("nan")
    gap = empirical - float(target_coverage)

    positive_mask = y == 1
    negative_mask = y == 0

    positive_cov = (
        float(sets.loc[positive_mask, "include_1"].mean())
        if positive_mask.any()
        else float("nan")
    )
    negative_cov = (
        float(sets.loc[negative_mask, "include_0"].mean())
        if negative_mask.any()
        else float("nan")
    )

    set_size = sets["set_size"].to_numpy(dtype=float)
    return {
        "target_coverage": float(target_coverage),
        "empirical_coverage": empirical,
        "coverage_gap": gap,
        "absolute_coverage_gap": abs(gap),
        "average_set_size": float(set_size.mean()),
        "singleton_rate": float(np.mean(set_size == 1.0)),
        "both_labels_rate": float(np.mean(set_size == 2.0)),
        "empty_set_rate": float(np.mean(set_size == 0.0)),
        "positive_class_coverage": positive_cov,
        "negative_class_coverage": negative_cov,
    }


def prediction_set_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    q_hat: float,
    target_coverage: float,
) -> dict[str, float]:
    """Summarize empirical coverage and set efficiency diagnostics."""
    y, p = _as_binary_arrays(y_true, probability)
    sets = build_prediction_sets(p, q_hat)
    return {
        "q_hat": float(q_hat),
        **_prediction_set_metrics(y, sets, target_coverage),
    }


def mondrian_prediction_set_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    q_hat_0: float,
    q_hat_1: float,
    target_coverage: float,
) -> dict[str, float]:
    """Summarize class-conditional coverage and set efficiency diagnostics."""
    y, p = _as_binary_arrays(y_true, probability)
    sets = build_mondrian_prediction_sets(p, q_hat_0, q_hat_1)
    return {
        "q_hat_0": float(q_hat_0),
        "q_hat_1": float(q_hat_1),
        **_prediction_set_metrics(y, sets, target_coverage),
    }


def temporal_prediction_set_coverage(
    frame: pd.DataFrame,
    target_column: str,
    probability_column: str,
    fold_column: str = "fold",
    target_coverage: float = 0.80,
) -> pd.DataFrame:
    """Evaluate temporal conformal coverage with strictly past-fold calibration.

    Fold 1 is not evaluated because no earlier OOF fold is available.
    """
    if fold_column not in frame.columns:
        raise ValueError(f"Missing fold column: {fold_column}")

    alpha = 1.0 - float(target_coverage)
    if not (0.0 < alpha < 1.0):
        raise ValueError("target_coverage must be in (0, 1).")

    required = {target_column, probability_column, fold_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing columns required for temporal coverage: " + ", ".join(missing)
        )

    table = frame.loc[:, [target_column, probability_column, fold_column]].copy()
    table[fold_column] = pd.to_numeric(table[fold_column], errors="raise").astype(int)

    rows: list[dict[str, float | int]] = []
    for fold in sorted(table[fold_column].unique()):
        evaluation = table.loc[table[fold_column] == fold]
        calibration = table.loc[table[fold_column] < fold]

        if calibration.empty:
            continue

        q_hat = conformal_quantile(
            nonconformity_scores(
                calibration[target_column].to_numpy(dtype=int),
                calibration[probability_column].to_numpy(dtype=float),
            ),
            alpha=alpha,
        )

        metrics = prediction_set_metrics(
            evaluation[target_column].to_numpy(dtype=int),
            evaluation[probability_column].to_numpy(dtype=float),
            q_hat=q_hat,
            target_coverage=target_coverage,
        )
        rows.append(
            {
                "fold": int(fold),
                "calibration_rows": int(len(calibration)),
                "evaluation_rows": int(len(evaluation)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def temporal_mondrian_prediction_set_coverage(
    frame: pd.DataFrame,
    target_column: str,
    probability_column: str,
    fold_column: str = "fold",
    target_coverage: float = 0.80,
) -> pd.DataFrame:
    """Evaluate temporal Mondrian coverage with past-fold calibration only."""
    required = {target_column, probability_column, fold_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing columns required for temporal Mondrian coverage: "
            + ", ".join(missing)
        )

    alpha = 1.0 - float(target_coverage)
    if not (0.0 < alpha < 1.0):
        raise ValueError("target_coverage must be in (0, 1).")

    table = frame.loc[:, [target_column, probability_column, fold_column]].copy()
    table[fold_column] = pd.to_numeric(table[fold_column], errors="raise").astype(int)

    rows: list[dict[str, float | int]] = []
    for fold in sorted(table[fold_column].unique()):
        evaluation = table.loc[table[fold_column] == fold]
        calibration = table.loc[table[fold_column] < fold]
        if calibration.empty:
            continue

        calibration_y, calibration_p = _as_binary_arrays(
            calibration[target_column].to_numpy(dtype=int),
            calibration[probability_column].to_numpy(dtype=float),
        )
        calibration_scores = nonconformity_scores(calibration_y, calibration_p)
        negative_scores = calibration_scores[calibration_y == 0]
        positive_scores = calibration_scores[calibration_y == 1]
        if negative_scores.size == 0 or positive_scores.size == 0:
            raise ValueError(
                f"Fold {fold} requires past calibration rows from both classes."
            )

        q_hat_0 = conformal_quantile(negative_scores, alpha=alpha)
        q_hat_1 = conformal_quantile(positive_scores, alpha=alpha)
        metrics = mondrian_prediction_set_metrics(
            evaluation[target_column].to_numpy(dtype=int),
            evaluation[probability_column].to_numpy(dtype=float),
            q_hat_0=q_hat_0,
            q_hat_1=q_hat_1,
            target_coverage=target_coverage,
        )
        rows.append(
            {
                "fold": int(fold),
                "calibration_rows": int(len(calibration)),
                "calibration_positive_rows": int(positive_scores.size),
                "calibration_negative_rows": int(negative_scores.size),
                "evaluation_rows": int(len(evaluation)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)
