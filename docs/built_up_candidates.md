# Spectral indices and Otsu built-up candidates

This stage consumes only Landsat composites that passed orchestration
finalization. It calculates seven continuous spectral indices and five
epoch-specific Otsu candidate maps per completed epoch.

The binary maps are **candidate pseudo-labels**, not validated built-up maps.

## Outputs per epoch

Index asset:

```text
savi
mndwi
ndbi
ibi
ibui
vbswir1_bi
ndbsui
valid_composite
```

Candidate asset:

```text
built_ndbi
built_ibi
built_ibui
built_vbswir1_bi
built_ndbsui
valid_ndbi
valid_ibi
valid_ibui
valid_vbswir1_bi
valid_ndbsui
```

Undefined index values remain masked. A candidate value of zero means valid
candidate non-built-up, while a masked value means that classification was not
possible.

## Execution

```bash
make built-up-preflight
make built-up-submit
```

All histograms and thresholds are validated before the first export task is
submitted. If any threshold fails, no export is started.

After the tasks finish:

```bash
make built-up-status
make built-up-finalize
make test-built-up
```

After numerical and visual review:

```bash
make freeze-built-up-v1
```

## Scientific limits

- Otsu thresholds are epoch-specific.
- No index is selected as the final historical mapping method.
- Candidate-area changes are diagnostic only.
- No temporal persistence rule or transition map is applied.
- Validation and method selection belong to the following stage.
