#  Landsat composites and auxiliary-source integration

Compositing converts the frozen catalogue into seven analysis-ready Landsat
surface-reflectance composites and valid-observation-count assets. It also
creates one aligned terrain asset, records exact GHSL public assets on their
native grid, and freezes a current Geofabrik OpenStreetMap snapshot.

No Landsat collection is searched dynamically. No spectral index, Otsu
threshold, built-up label or transition is produced during this stage.

## 1. Preflight

```bash
make orchestrate-preflight
```

This verifies the frozen Day 1 grid, Day 2 catalogue reference, compositing
protocol, seven epochs and exact Earth Engine scene asset IDs.

## 2. Submit Earth Engine tasks

```bash
make orchestrate-submit
```

This submits:

- seven six-band Landsat median composites;
- seven total observation-count assets, including sensor-specific bands only
  for mixed-sensor epochs;
- one two-band SRTM elevation-and-slope asset.

GHSL is not exported. Fourteen exact public asset IDs are recorded and kept on
their native grid.

## 3. Build the OSM snapshot

```bash
make orchestrate-osm
```

The script downloads the current Geofabrik Cameroon GeoPackage archive,
preserves a dated filename and checksum, clips roads, railways and buildings to
the Day 1 context area, and writes EPSG:32632 vector layers.

The archive is large and remains excluded from Git.

## 4. Monitor and finalize

```bash
make orchestrate-status
```

After all Earth Engine tasks report `COMPLETED`:

```bash
make orchestrate-finalize
make test-orchestrate
```

## 5. Freeze Day 3 version 1

After visual and numerical acceptance:

```bash
mkdir -p tests/reference
cp data/metadata/orchestrate/orchestrate_version.json \
  tests/reference/orchestrate_sources_v1.json
```

Never overwrite the frozen reference automatically during normal execution.
