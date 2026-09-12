# cuDESeq2 manuscript

`main.tex` is the review-format manuscript and `main.pdf` is its tracked build.

## Build

From the repository root:

```bash
bash scripts/paper_build.sh
```

Use `bash scripts/paper_build.sh --figures` to regenerate the data-driven parity
figures on CPU before compiling. The two workflow schematics are tracked paper
assets and are not overwritten by this command. Figure generation does not
rerun the timing benchmarks.

The build requires `pdflatex` and `bibtex`. Figure regeneration additionally
requires the project Python environment and the validation data described in
`validation/README.md`.

## Manuscript evidence

- Real-data parity across six designs is recorded in
  `bench/results/reference_parity.json` and summarized in Table 2.
- A100 cuDESeq2 end-to-end and component timings are recorded in
  `bench/results/timings.json`.
- The practical same-host 12-worker R baseline is recorded in
  `bench/results/r_parallel_a100_12worker.json`.
- Exact download URLs/checksums and prepared-input checksums are recorded in
  `validation/data_sources.json` and `validation/prepared_data_manifest.json`.
- GPU execution-mode agreement is recorded in `bench/results/parity.json`.
- Real-GTEx full-workflow scaling, exact nested cohorts, matched 12-worker R
  endpoints, GPU--R parity, peak CUDA
  memory, and the A100 OOM boundary are recorded in
  `bench/results/gtex_scaling_a100.json`.
- Human-readable derived tables are in `bench/results/TABLES.md`.

Main-suite and repeated real-GTEx GPU timings are medians of five measured
repetitions following an untimed warm-up. The practical R comparison uses
direct end-to-end observations. The large-GTEx R endpoints and OOM feasibility
probes are explicitly single observations and are not pooled into the main
headline.
The manuscript's measured claims can be checked from a clean checkout, without
ignored data or caches, with:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_paper_numbers.py
```

To reconstruct the real inputs and rerun the measurements:

```bash
make container-data
make container-r-reference
make container-r-parallel
make container-gpu-benchmark
```

The related-work discussion and bibliography are populated, and the real-data
validation and same-host multicore R baseline are complete. The reported GPU
measurements are from one NVIDIA A100; no second-GPU generalization is claimed.
