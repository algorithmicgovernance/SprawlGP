"""Validate population values after reassembling the final Parquet table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    """Print population QA statistics and reject inflated values."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/final/"
            "yaounde_urban_expansion_30m_v2/"
            "cell_time_dataset.parquet"
        ),
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.dataset)
    required = {
        "forecast_origin",
        "population_density_t",
        "population_source_year",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing population columns: " + ", ".join(missing)
        )

    density = pd.to_numeric(
        frame["population_density_t"],
        errors="raise",
    )

    summary = (
        frame.assign(
            population_density_t=density,
            implied_people_per_30m_cell=density * 0.0009,
        )
        .groupby("forecast_origin")
        .agg(
            rows=("population_density_t", "size"),
            zero_share=(
                "population_density_t",
                lambda values: values.eq(0).mean(),
            ),
            mean_density=("population_density_t", "mean"),
            median_density=("population_density_t", "median"),
            p95_density=(
                "population_density_t",
                lambda values: values.quantile(0.95),
            ),
            p99_density=(
                "population_density_t",
                lambda values: values.quantile(0.99),
            ),
            maximum_density=("population_density_t", "max"),
            maximum_people_in_30m_cell=(
                "implied_people_per_30m_cell",
                "max",
            ),
        )
        .reset_index()
    )

    year_match = (
        frame["population_source_year"].astype(int)
        == frame["forecast_origin"].astype(int)
    )

    print(summary.round(3).to_string(index=False))
    print()
    print(
        "Population source year matches origin:",
        float(year_match.mean()),
    )

    assert density.notna().all()
    assert density.ge(0).all()
    assert year_match.all()
    assert density.max() < 250_000, (
        "Population density remains implausibly high. "
        "The native GHSL area may still be evaluated on the 30 m grid."
    )

    print("\nPASS: corrected population values satisfy local QA checks.")


if __name__ == "__main__":
    main()
