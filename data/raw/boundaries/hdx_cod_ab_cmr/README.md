# Boundary source record

## Dataset

- **Dataset:** Cameroon - Subnational Administrative Boundaries
- **Dataset identifier:** `cod-ab-cmr`
- **Provider:** OCHA Field Information Services Section
- **Original source:** Institut National de Cartographie, Cameroon
- **Dataset type:** Common Operational Dataset – Administrative Boundaries
- **Version:** `v01`
- **Administrative coverage:** ADM0–ADM3
- **Units:** 10 ADM1 regions, 58 ADM2 departments and 360 ADM3 arrondissements
- **Boundary valid from:** 2019-01-04
- **Dataset review date:** 2025-10-30
- **Resource modification date:** 2026-01-26
- **Download date:** 2026-07-22

## Downloaded resource

- **Filename:** `cmr_admin_boundaries.geojson.zip`
- **Format:** GeoJSON ZIP archive
- **Resource ID:** `2248ac6d-9675-47de-90d3-1e7ce6e601b1`
- **Approximate size:** 10.5 MB
- **Raw archive modified after download:** No
- **Raw archive SHA-256:** `f24a75c41ee2e107448ff1d2b8769ee970dad376d20cd7093da0c9106814ab5a`

The extracted `cmr_admin2.geojson` and `cmr_admin3.geojson` files are retained
without manual geometry editing.

## Licence and attribution

- **Licence:** Creative Commons Attribution 4.0 International (`CC BY 4.0`)
- **Required attribution:** OCHA Field Information Services Section and
  Institut National de Cartographie, Cameroon
- **Source page:** `https://data.humdata.org/dataset/cod-ab-cmr`

## Project use

The seven Mfoundi ADM3 units identified by P-codes
`CM002007001`–`CM002007007` are dissolved to construct the Yaoundé
administrative core. Their union is compared with the ADM2 Mfoundi boundary
(`CM002007`).

geoBoundaries ADM2 and ADM3 files are used only as independent cross-check
sources and do not define the final study boundary.

## Processing policy

The raw archive is preserved unchanged. Selection, validation, geometry repair,
dissolution, reprojection, buffering, rasterisation and grid construction are
performed reproducibly in code using `configs/study_area.yaml`.