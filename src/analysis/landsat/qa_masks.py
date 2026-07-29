"""Collection 2 quality masks used by the Landsat catalogue."""

from __future__ import annotations

from typing import Any

import ee


OLI_SENSORS = {"LC08", "LC09"}


def bit_is_set(image: ee.Image, band: str, bit: int) -> ee.Image:
    """Return a Boolean image indicating whether one QA bit is set."""
    return image.select(band).bitwiseAnd(1 << bit).neq(0)


def landsat_valid_mask(
    image: ee.Image,
    sensor_key: str,
    collections: dict[str, Any],
    rules: dict[str, Any],
) -> ee.Image:
    """Return one for QA-valid observations and zero for invalid pixels.

    Water is deliberately retained. High-confidence cirrus is removed
    for Landsat 8 and Landsat 9. Fill, dilated cloud, cloud, cloud
    shadow, snow and radiometric saturation are also removed.
    """
    if sensor_key not in collections:
        raise ValueError(f"Unsupported sensor: {sensor_key}")

    valid = ee.Image.constant(1)
    bit_rules = [
        ("mask_fill", 0),
        ("mask_dilated_cloud", 1),
        ("mask_cloud", 3),
        ("mask_cloud_shadow", 4),
        ("mask_snow", 5),
    ]

    for rule_name, bit in bit_rules:
        if rules.get(rule_name, True):
            valid = valid.And(
                bit_is_set(image, "QA_PIXEL", bit).Not()
            )

    if (
        sensor_key in OLI_SENSORS
        and rules.get("mask_cirrus_landsat8", True)
    ):
        valid = valid.And(
            bit_is_set(image, "QA_PIXEL", 2).Not()
        )

    if rules.get("mask_radiometric_saturation", True):
        valid = valid.And(
            image.select("QA_RADSAT").eq(0)
        )

    required = collections[sensor_key]["required_bands"]
    all_bands_present = (
        image.select(required)
        .mask()
        .reduce(ee.Reducer.min())
        .eq(1)
    )

    return (
        valid.And(all_bands_present)
        .rename("valid")
        .toByte()
    )
