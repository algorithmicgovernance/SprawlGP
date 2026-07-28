"""Spectral-index formulas, masks and candidate construction."""

from __future__ import annotations

import math
from typing import Any

try:
    import ee
except ImportError:
    ee = None


COMMON_BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]

INTERMEDIATE_INDICES = [
    "savi",
    "mndwi",
]

# Retained in the continuous index assets.
CONTINUOUS_INDICES = [
    "ndbi",
    "ibi",
    "ibui",
    "vbswir1_bi",
    "ndbsui",
]

# Allowed to produce Otsu candidates and enter Day 5 comparison.
CANDIDATE_INDICES = [
    "ndbi",
    "ibui",
    "vbswir1_bi",
    "ndbsui",
]

CONTINUOUS_INDEX_BANDS = (
    INTERMEDIATE_INDICES
    + CONTINUOUS_INDICES
)

INDEX_BANDS = (
    CONTINUOUS_INDEX_BANDS
    + ["valid_composite"]
)

CANDIDATE_BANDS = (
    [f"built_{name}" for name in CANDIDATE_INDICES]
    + [f"valid_{name}" for name in CANDIDATE_INDICES]
)


def require_earth_engine() -> None:
    """Raise an actionable error when Earth Engine is unavailable."""
    if ee is None:
        raise ImportError(
            "The Earth Engine Python API is required for raster "
            "processing."
        )


def safe_ratio(
    numerator: Any,
    denominator: Any,
    epsilon: float,
    name: str,
):
    """Divide images while masking numerically unstable denominators."""
    require_earth_engine()
    valid_denominator = denominator.abs().gt(float(epsilon))

    return (
        numerator.updateMask(valid_denominator)
        .divide(denominator.updateMask(valid_denominator))
        .rename(name)
        .toFloat()
    )


def build_index_stack(
    composite,
    observation_count,
    *,
    epsilon: float,
    savi_l: float,
    minimum_valid_observations: int,
):
    """Calculate seven continuous indices and one validity band.

    IBI remains in the continuous asset for transparency. It is not
    included in ``CANDIDATE_INDICES`` and therefore receives no Otsu
    threshold or binary candidate band.
    """
    require_earth_engine()
    composite = composite.select(COMMON_BANDS)
    count = observation_count.select(
        "valid_observation_count"
    )

    all_bands_valid = (
        composite.mask()
        .reduce(ee.Reducer.min())
        .eq(1)
    )
    observation_valid = count.gte(
        int(minimum_valid_observations)
    )
    base_valid = all_bands_valid.And(observation_valid)

    source = composite.updateMask(base_valid)
    blue = source.select("blue")
    green = source.select("green")
    red = source.select("red")
    nir = source.select("nir")
    swir1 = source.select("swir1")
    swir2 = source.select("swir2")

    savi = safe_ratio(
        nir.subtract(red).multiply(1.0 + float(savi_l)),
        nir.add(red).add(float(savi_l)),
        epsilon,
        "savi",
    )
    mndwi = safe_ratio(
        green.subtract(swir1),
        green.add(swir1),
        epsilon,
        "mndwi",
    )
    ndbi = safe_ratio(
        swir1.subtract(nir),
        swir1.add(nir),
        epsilon,
        "ndbi",
    )

    background = savi.add(mndwi).divide(2.0)
    ibi = safe_ratio(
        ndbi.subtract(background),
        ndbi.add(background),
        epsilon,
        "ibi",
    )
    ibui = (
        ndbi.subtract(savi)
        .subtract(mndwi)
        .rename("ibui")
        .toFloat()
    )
    vbswir1_bi = safe_ratio(
        swir1.subtract(blue),
        swir1.add(blue),
        epsilon,
        "vbswir1_bi",
    )

    nonnegative_inputs = red.gte(0).And(swir1.gte(0))
    geometric_mean = (
        red.max(0)
        .multiply(swir1.max(0))
        .sqrt()
    )
    ndbsui = safe_ratio(
        swir2.add(geometric_mean).subtract(
            red.add(swir1)
        ),
        swir2.add(geometric_mean).add(
            red.add(swir1)
        ),
        epsilon,
        "ndbsui",
    ).updateMask(nonnegative_inputs)

    valid_composite = (
        base_valid.rename("valid_composite")
        .unmask(0)
        .toUint8()
    )

    images = {
        "savi": savi,
        "mndwi": mndwi,
        "ndbi": ndbi,
        "ibi": ibi,
        "ibui": ibui,
        "vbswir1_bi": vbswir1_bi,
        "ndbsui": ndbsui,
    }

    stack = ee.Image.cat(
        [
            images[name]
            for name in CONTINUOUS_INDEX_BANDS
        ]
        + [valid_composite]
    ).select(INDEX_BANDS)

    return stack, images


def build_candidate_stack(
    index_images: dict[str, Any],
    thresholds: dict[str, float],
):
    """Create candidate and validity bands for candidate indices."""
    require_earth_engine()
    built_bands = []
    valid_bands = []

    for index_name in CANDIDATE_INDICES:
        if index_name not in thresholds:
            raise KeyError(
                f"Missing threshold for index '{index_name}'."
            )

        index_image = index_images[index_name]
        valid = (
            index_image.mask()
            .reduce(ee.Reducer.min())
            .rename(f"valid_{index_name}")
            .unmask(0)
            .toUint8()
        )
        built = (
            index_image.gt(float(thresholds[index_name]))
            .rename(f"built_{index_name}")
            .updateMask(valid.eq(1))
            .toUint8()
        )
        built_bands.append(built)
        valid_bands.append(valid)

    return ee.Image.cat(
        built_bands + valid_bands
    ).select(CANDIDATE_BANDS)


def _safe_scalar_ratio(
    numerator: float,
    denominator: float,
    epsilon: float,
) -> float | None:
    """Return a scalar ratio or ``None`` for an unstable denominator."""
    if not math.isfinite(numerator):
        return None

    if not math.isfinite(denominator):
        return None

    if abs(denominator) <= epsilon:
        return None

    return numerator / denominator


def reference_index_values(
    *,
    blue: float,
    green: float,
    red: float,
    nir: float,
    swir1: float,
    swir2: float,
    epsilon: float = 1e-6,
    savi_l: float = 0.5,
) -> dict[str, float | None]:
    """Calculate scalar reference values used by formula tests."""
    savi = _safe_scalar_ratio(
        (nir - red) * (1.0 + savi_l),
        nir + red + savi_l,
        epsilon,
    )
    mndwi = _safe_scalar_ratio(
        green - swir1,
        green + swir1,
        epsilon,
    )
    ndbi = _safe_scalar_ratio(
        swir1 - nir,
        swir1 + nir,
        epsilon,
    )
    vbswir1_bi = _safe_scalar_ratio(
        swir1 - blue,
        swir1 + blue,
        epsilon,
    )

    if savi is None or mndwi is None or ndbi is None:
        ibi = None
        ibui = None
    else:
        background = (savi + mndwi) / 2.0
        ibi = _safe_scalar_ratio(
            ndbi - background,
            ndbi + background,
            epsilon,
        )
        ibui = ndbi - savi - mndwi

    if red < 0 or swir1 < 0:
        ndbsui = None
    else:
        geometric_mean = math.sqrt(red * swir1)
        ndbsui = _safe_scalar_ratio(
            swir2 + geometric_mean - (red + swir1),
            swir2 + geometric_mean + (red + swir1),
            epsilon,
        )

    return {
        "savi": savi,
        "mndwi": mndwi,
        "ndbi": ndbi,
        "ibi": ibi,
        "ibui": ibui,
        "vbswir1_bi": vbswir1_bi,
        "ndbsui": ndbsui,
    }
