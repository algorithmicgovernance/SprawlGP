"""Collection 2 quality masks used by the Day 2 Landsat catalogue."""

from __future__ import annotations

from typing import Any
import ee


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

    Water is deliberately retained. Landsat 8 cirrus is removed using bit 2;
    fill, dilated cloud, cloud, shadow, snow and saturation are also removed.
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
            valid = valid.And(bit_is_set(image, "QA_PIXEL", bit).Not())

    if sensor_key == "LC08" and rules.get("mask_cirrus_landsat8", True):
        valid = valid.And(bit_is_set(image, "QA_PIXEL", 2).Not())

    if rules.get("mask_radiometric_saturation", True):
        valid = valid.And(image.select("QA_RADSAT").eq(0))

    required = collections[sensor_key]["required_bands"]
    all_bands_present = image.select(required).mask().reduce(ee.Reducer.min()).eq(1)
    return valid.And(all_bands_present).rename("valid").toByte()
