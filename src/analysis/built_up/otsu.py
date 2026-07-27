"""Deterministic local Otsu calculation and histogram validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


PASS = "PASS"
FAIL_NO_VALID_PIXELS = "FAIL_NO_VALID_PIXELS"
FAIL_DEGENERATE_HISTOGRAM = "FAIL_DEGENERATE_HISTOGRAM"
FAIL_EMPTY_CLASS = "FAIL_EMPTY_CLASS"
FAIL_NONFINITE_THRESHOLD = "FAIL_NONFINITE_THRESHOLD"


@dataclass(frozen=True)
class OtsuResult:
    """Represent one successful or failed Otsu threshold calculation."""

    status: str
    threshold: float | None
    failure_reason: str
    histogram_bucket_count: int
    histogram_valid_pixel_count: int
    histogram_min: float | None
    histogram_max: float | None
    histogram_mean: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable result dictionary."""
        return asdict(self)


def _failure(
    status: str,
    reason: str,
    *,
    bucket_count: int = 0,
    valid_pixel_count: int = 0,
    histogram_min: float | None = None,
    histogram_max: float | None = None,
    histogram_mean: float | None = None,
) -> OtsuResult:
    """Build a consistently populated failed Otsu result."""
    return OtsuResult(
        status=status,
        threshold=None,
        failure_reason=reason,
        histogram_bucket_count=bucket_count,
        histogram_valid_pixel_count=valid_pixel_count,
        histogram_min=histogram_min,
        histogram_max=histogram_max,
        histogram_mean=histogram_mean,
    )


def otsu_from_histogram(histogram: dict[str, Any] | None) -> OtsuResult:
    """Calculate Otsu's threshold from an Earth Engine histogram dictionary.

    The threshold is the midpoint between the two adjacent bucket centres at
    the split that maximises between-class variance. No fallback threshold is
    fabricated when the distribution is empty or degenerate.
    """
    if not histogram:
        return _failure(
            FAIL_NO_VALID_PIXELS,
            "No histogram was returned.",
        )

    counts = np.asarray(histogram.get("histogram", []), dtype=float)
    means = np.asarray(histogram.get("bucketMeans", []), dtype=float)

    if counts.size == 0 or means.size == 0 or counts.size != means.size:
        return _failure(
            FAIL_NO_VALID_PIXELS,
            "Histogram counts or bucket centres are missing.",
        )

    finite = np.isfinite(counts) & np.isfinite(means) & (counts > 0)
    counts = counts[finite]
    means = means[finite]

    if counts.size == 0:
        return _failure(
            FAIL_NO_VALID_PIXELS,
            "The histogram contains no positive finite counts.",
        )

    order = np.argsort(means)
    counts = counts[order]
    means = means[order]
    valid_pixel_count = int(round(float(counts.sum())))
    histogram_mean = float(np.average(means, weights=counts))
    histogram_min = float(means.min())
    histogram_max = float(means.max())
    bucket_count = int(counts.size)

    if bucket_count < 2 or np.allclose(means, means[0]):
        return _failure(
            FAIL_DEGENERATE_HISTOGRAM,
            "At least two distinct non-empty histogram bins are required.",
            bucket_count=bucket_count,
            valid_pixel_count=valid_pixel_count,
            histogram_min=histogram_min,
            histogram_max=histogram_max,
            histogram_mean=histogram_mean,
        )

    cumulative_weight = np.cumsum(counts)
    cumulative_sum = np.cumsum(counts * means)
    total_weight = cumulative_weight[-1]
    total_sum = cumulative_sum[-1]

    left_weight = cumulative_weight[:-1]
    right_weight = total_weight - left_weight
    valid_splits = (left_weight > 0) & (right_weight > 0)

    if not valid_splits.any():
        return _failure(
            FAIL_EMPTY_CLASS,
            "No split leaves observations in both classes.",
            bucket_count=bucket_count,
            valid_pixel_count=valid_pixel_count,
            histogram_min=histogram_min,
            histogram_max=histogram_max,
            histogram_mean=histogram_mean,
        )

    left_mean = cumulative_sum[:-1] / left_weight
    right_mean = (total_sum - cumulative_sum[:-1]) / right_weight
    between_variance = (
        left_weight
        * right_weight
        * np.square(left_mean - right_mean)
    )
    between_variance[~valid_splits] = -np.inf
    split_index = int(np.argmax(between_variance))

    if not np.isfinite(between_variance[split_index]):
        return _failure(
            FAIL_EMPTY_CLASS,
            "Between-class variance has no valid finite maximum.",
            bucket_count=bucket_count,
            valid_pixel_count=valid_pixel_count,
            histogram_min=histogram_min,
            histogram_max=histogram_max,
            histogram_mean=histogram_mean,
        )

    threshold = float(
        (means[split_index] + means[split_index + 1]) / 2.0
    )

    if not np.isfinite(threshold):
        return _failure(
            FAIL_NONFINITE_THRESHOLD,
            "The selected threshold is not finite.",
            bucket_count=bucket_count,
            valid_pixel_count=valid_pixel_count,
            histogram_min=histogram_min,
            histogram_max=histogram_max,
            histogram_mean=histogram_mean,
        )

    return OtsuResult(
        status=PASS,
        threshold=threshold,
        failure_reason="",
        histogram_bucket_count=bucket_count,
        histogram_valid_pixel_count=valid_pixel_count,
        histogram_min=histogram_min,
        histogram_max=histogram_max,
        histogram_mean=histogram_mean,
    )


def histogram_quantile(
    histogram: dict[str, Any],
    probability: float,
) -> float:
    """Approximate a quantile from histogram bucket centres and counts."""
    if probability < 0 or probability > 1:
        raise ValueError("Probability must be in the interval [0, 1].")

    counts = np.asarray(histogram["histogram"], dtype=float)
    means = np.asarray(histogram["bucketMeans"], dtype=float)
    finite = np.isfinite(counts) & np.isfinite(means) & (counts > 0)

    if not finite.any():
        raise ValueError("The histogram contains no valid observations.")

    counts = counts[finite]
    means = means[finite]
    order = np.argsort(means)
    counts = counts[order]
    means = means[order]
    cumulative = np.cumsum(counts)
    target = probability * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(means) - 1)
    return float(means[index])
