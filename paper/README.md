# cuDESeq2 manuscript

`main.tex` is the review-format manuscript and `main.pdf` is its tracked build.

## Build

From the repository root:

```bash
bash scripts/paper_build.sh
```

Use `bash scripts/paper_build.sh --figures` to regenerate the workflow schematic
and the parity figures before compiling. The parity figure generator reads the
committed/reference benchmark caches; it does not rerun the timing benchmarks.

The build requires `pdflatex` and `bibtex`. Figure regeneration additionally
requires the project Python environment and the validation data described in
`validation/README.md`.

## Manuscript evidence

- Real-data parity across six designs is recorded in
  `bench/results/reference_parity.json` and summarized in Table 2.
- A100 cuDESeq2 and one-worker R timings are recorded in
  `bench/results/timings.json`.
- Same-host one- and 12-worker R timings are recorded in
  `bench/results/r_parallel_a100_12worker.json`.
- GPU execution-mode agreement is recorded in `bench/results/parity.json`.
- Human-readable derived tables are in `bench/results/TABLES.md`.

Every reported timing is the median of five measured repetitions following an
untimed warm-up. The manuscript's measured claims can be checked with:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_paper_numbers.py
```

The related-work discussion and bibliography are populated, and the real-data
validation and same-host multicore R baseline are complete. The reported GPU
measurements are from one NVIDIA A100; no second-GPU generalization is claimed.
