# Real-data validation

This directory compares cuDESeq2 with R 4.6.0, DESeq2 1.52.0, and apeglm
1.34.0 on six designs derived from pasilla, airway, and GTEx/recount2.

Both implementations run the standard Wald pipeline: size-factor estimation,
dispersion estimation, Wald fitting and Cook's-distance calculation, eligible
count-outlier replacement and refitting, independent filtering and BH
adjustment, and apeGLM shrinkage. The replacement branch is active for the
300-sample GTEx contrast and inactive for the five 7- or 8-sample designs.

## Reconstruct the inputs

The prepared matrices are derived and intentionally not checked into Git. A
clean checkout can download and reconstruct them exactly:

```bash
make container-build
make container-data
```

[`data_sources.json`](data_sources.json) pins the exact airway 1.32.0 and
pasilla 1.40.0 source archives and the recount2 SRP012682 R object by URL, byte
length, and SHA-256. `scripts/fetch_validation_data.py` refuses a mismatched
download. `prepare_inputs.R` applies the versioned conversion and sample
selection rules, including R seed 1 for the 150+150 GTEx tissue sample.
[`prepared_data_manifest.json`](prepared_data_manifest.json) records the
resulting file hashes and selected GTEx identifiers.

Raw downloads occupy approximately 1.4 GB in ignored `validation/sources/`.
Prepared CSVs are written to ignored `validation/data/`. Either set can be
deleted and regenerated from the two tracked manifests.

## Regenerate the measurements

```bash
make container-r-reference
make container-r-parallel
make container-gpu-benchmark
```

The GPU command must run on an A100 to reproduce the reported hardware
measurement. For a version-only parity refresh without CUDA, run:

```bash
PYTHONPATH=src python bench/reference_parity.py --device cpu
```

This is a correctness run, not a GPU timing run. The benchmark cache under
`bench/cache/` is also derived and ignored; the paper audit reads only committed
artifacts.

To export the R dispersion components and redraw the diagnostic figures:

```bash
Rscript validation/export_r_dispersion_details.R
PYTHONPATH=src python validation/make_paper_figures.py --device cpu
```

## Cases

| case | design | P | samples × genes |
|---|---|---:|---:|
| pasilla | `~ condition` | 2 | 7 × 12,359 |
| pasilla_2fac | `~ type + condition` | 3 | 7 × 12,359 |
| airway_dex | `~ dex` | 2 | 8 × 33,469 |
| airway_cell | `~ cell` | 4 | 8 × 33,469 |
| airway | `~ cell + dex` | 5 | 8 × 33,469 |
| gtex_blood_muscle | `~ tissue` | 2 | 300 × 54,922 |

The authoritative parity record is
[`bench/results/reference_parity.json`](../bench/results/reference_parity.json),
with a human-readable summary in
[`bench/results/TABLES.md`](../bench/results/TABLES.md). At adjusted p < 0.05,
four called sets match R exactly. Airway/dex and GTEx differ by one boundary
gene each, with Jaccard indices 0.99963 and 0.99997. The worst dispersion p95
relative error is 0.0044, and the worst apeGLM-shrunk-LFC Pearson correlation is
0.99722.
