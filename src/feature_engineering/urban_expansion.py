"""Candidate urban-expansion features derived from the existing dataset.

These transformations require no additional external datasets.

Important
---------
This module defines candidate predictors for the NEXT SVGP tuning phase.
The currently selected SVGP-Adam remains on its frozen base feature set until
an explicit experiment activates one of these candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_PREDICTORS = (
    "ndbi_t",
    "savi_t",
    "distance_to_built_m_t",
    "built_fraction_11x11_t",
    "recent_local_growth_5y_t",
    "elevation_m",
    "slope_degrees",
    "log_population_density_t",
)

LOG_DISTANCE_PREDICTORS = (
    "ndbi_t",
    "savi_t",
    "log_distance_to_built_m_t",
    "built_fraction_11x11_t",
    "recent_local_growth_5y_t",
    "elevation_m",
    "slope_degrees",
    "log_population_density_t",
)

SLOPE_SQUARED_PREDICTORS = (
    *BASE_PREDICTORS,
    "slope_squared_t",
)

GROWTH_INTERACTION_PREDICTORS = (
    *BASE_PREDICTORS,
    "built_fraction_x_recent_growth_t",
)

LOG_DISTANCE_GROWTH_PREDICTORS = (
    "ndbi_t",
    "savi_t",
    "log_distance_to_built_m_t",
    "built_fraction_11x11_t",
    "recent_local_growth_5y_t",
    "elevation_m",
    "slope_degrees",
    "log_population_density_t",
    "built_fraction_x_recent_growth_t",
)


POPULATION_INTERACTION_PREDICTORS = (
    *BASE_PREDICTORS,
    "built_fraction_x_log_population_t",
)

NONLINEAR_CORE_PREDICTORS = (
    "ndbi_t",
    "savi_t",
    "log_distance_to_built_m_t",
    "built_fraction_11x11_t",
    "recent_local_growth_5y_t",
    "elevation_m",
    "slope_degrees",
    "slope_squared_t",
    "log_population_density_t",
)

INTERACTION_PREDICTORS = (
    *NONLINEAR_CORE_PREDICTORS,
    "built_fraction_x_recent_growth_t",
    "built_fraction_x_log_population_t",
)


def add_candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add non-linear and interaction candidates without new data sources.

    Added variables
    ---------------
    log_population_density_t
        log1p population density. This is already used by the selected model.

    log_distance_to_built_m_t
        log1p distance to existing built-up land. This lets the parametric
        component represent a diminishing proximity effect.

    slope_squared_t
        Simple non-linear terrain term.

    built_fraction_x_recent_growth_t
        Interaction between current neighbourhood urbanisation and recent
        expansion momentum.

    built_fraction_x_log_population_t
        Interaction between neighbourhood urbanisation and population pressure.

    Notes
    -----
    x/y coordinates and forecast year are intentionally NOT created here for
    the parametric mean. In the SVGP design, space/time remain GP inputs.
    """
    result = frame.copy()

    required = {
        "population_density_t",
        "distance_to_built_m_t",
        "slope_degrees",
        "built_fraction_11x11_t",
        "recent_local_growth_5y_t",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(
            "Missing source columns for feature engineering: "
            + ", ".join(missing)
        )

    population = pd.to_numeric(
        result["population_density_t"],
        errors="raise",
    ).astype(float)
    distance = pd.to_numeric(
        result["distance_to_built_m_t"],
        errors="raise",
    ).astype(float)
    slope = pd.to_numeric(
        result["slope_degrees"],
        errors="raise",
    ).astype(float)

    if population.lt(0).any():
        raise ValueError("population_density_t must be non-negative.")
    if distance.lt(0).any():
        raise ValueError("distance_to_built_m_t must be non-negative.")
    if not (
        np.isfinite(population).all()
        and np.isfinite(distance).all()
        and np.isfinite(slope).all()
    ):
        raise ValueError("Feature-engineering source values must be finite.")

    result["log_population_density_t"] = np.log1p(population)
    result["log_distance_to_built_m_t"] = np.log1p(distance)
    result["slope_squared_t"] = slope ** 2

    built_fraction = pd.to_numeric(
        result["built_fraction_11x11_t"],
        errors="raise",
    ).astype(float)
    recent_growth = pd.to_numeric(
        result["recent_local_growth_5y_t"],
        errors="raise",
    ).astype(float)

    result["built_fraction_x_recent_growth_t"] = (
        built_fraction * recent_growth
    )
    result["built_fraction_x_log_population_t"] = (
        built_fraction * result["log_population_density_t"]
    )

    return result


def predictors_for(feature_set: str) -> tuple[str, ...]:
    """Return one explicit candidate feature set."""
    mapping = {
        "base": BASE_PREDICTORS,
        "log_distance": LOG_DISTANCE_PREDICTORS,
        "slope_squared": SLOPE_SQUARED_PREDICTORS,
        "growth_interaction": GROWTH_INTERACTION_PREDICTORS,
        "population_interaction": POPULATION_INTERACTION_PREDICTORS,
        "nonlinear_core": NONLINEAR_CORE_PREDICTORS,
        "interactions": INTERACTION_PREDICTORS,
        "log_distance_growth":LOG_DISTANCE_GROWTH_PREDICTORS,
    }
    try:
        return mapping[feature_set]
    except KeyError as exc:
        raise ValueError(
            "feature_set must be one of: "
            + ", ".join(mapping)
        ) from exc
