# Real-data validation

This directory compares cuDESeq2 with the current R/Bioconductor reference on
six designs from pasilla, airway, and GTEx. The current committed reference was
generated with R 4.6.0, DESeq2 1.52.0, and apeglm 1.34.0.

Both implementations run the standard Wald pipeline:

1. size-factor estimation;
2. gene-wise, trend, and MAP dispersion estimation;
3. Wald fitting and Cook's-distance calculation;
4. eligible count-outlier replacement and affected-gene refitting;
5. independent filtering, multiple-testing correction, and apeGLM shrinkage.

The replacement branch requires at least seven replicates in a design cell. It
is therefore inactive for the five 7- or 8-sample designs and active for the
300-sample GTEx contrast.

## Reproduce

The benchmark cache is the reference source used by the paper:

```bash
R_BIN=.r46/bin/Rscript PYTHONPATH=src .venv/bin/python bench/bench.py
PYTHONPATH=src .venv/bin/python bench/reference_parity.py --device cuda
```

The second command must run on the A100 to refresh execution-mode comparisons.
For a version-only reference update on a machine without CUDA, use
`--device cpu`; this updates DESeq2 parity but is not a GPU timing run.

Export the R dispersion components used by the diagnostic figure and redraw the
figures:

```bash
PATH="$PWD/.r46/bin:$PATH" .r46/bin/Rscript \
  validation/export_r_dispersion_details.R
PYTHONPATH=src .venv/bin/python validation/make_paper_figures.py --device cpu
```

Refresh the compact per-case reports from the committed captures without
rerunning timing loops:

```bash
PYTHONPATH=src .venv/bin/python validation/validate.py \
  --from-cache --skip-timing
```

## Current datasets

| case | design | P | samples × genes |
|---|---|---:|---:|
| pasilla | `~ condition` | 2 | 7 × 12,359 |
| pasilla_2fac | `~ type + condition` | 3 | 7 × 12,359 |
| airway_dex | `~ dex` | 2 | 8 × 33,469 |
| airway_cell | `~ cell` | 4 | 8 × 33,469 |
| airway | `~ cell + dex` | 5 | 8 × 33,469 |
| gtex_blood_muscle | `~ tissue` | 2 | 300 × 54,922 |

The authoritative current metrics are
[`bench/results/reference_parity.json`](../bench/results/reference_parity.json);
the paper-ready summary is
[`bench/results/TABLES.md`](../bench/results/TABLES.md). At adjusted
`p < 0.05`, four designs reproduce the R called set exactly. Airway/dex and
GTEx differ by one boundary gene each, with Jaccard indices 0.99963 and
0.99997. The worst dispersion p95 relative error is 0.00427, and the worst
apeGLM-shrunk-LFC Pearson correlation is 0.99722.

`validation/results/*.json` contains the same standard-pipeline comparison in a
legacy per-case schema. The timing fields are intentionally omitted when those
files are refreshed with `--skip-timing`; GPU timing belongs in
`bench/results/timings.json`.
