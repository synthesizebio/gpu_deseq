# Whitepaper — cuDESeq2 (gpu-deseq)

Draft outline for the gpu-deseq whitepaper.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
# -> main.pdf
```

Requires a TeX distribution (`pdflatex`, `bibtex`). `main.tex` compiles as-is;
`\todo{}` (red) marks content still to write, `\note{}` (blue) marks section
intent, `\fact{}` marks a measured claim to cite from `../benchmarks/RESULTS.md`.

## Status (what's full vs thin)

| Section | State |
|---|---|
| Abstract, Contributions | drafted from committed work |
| Background, Design (4 techniques) | outlined with real mechanisms |
| Correctness / parity table | numbers stubbed from tests; **needs real-dataset validation** |
| Performance (per-step table, roofline, scaling, ablations) | real A100 numbers in; **needs 2nd GPU + bigger-core R** |
| Related work | **stub — needs literature search** |
| Limitations | complete (from README/RESULTS) |

## Data sources

All numbers trace to `../benchmarks/` (RESULTS.md + the `results_*.json` and the
reproducible drivers). Parity from `../tests/test_r_step_parity.py`.

## Before submission (open items)

1. Real published RNA-seq dataset: identical significant-gene set + LFC correlation vs R.
2. Second GPU (e.g. L4) and a 32–64-core R host for the multi-core baseline.
3. Related-work / novelty positioning (verify all citations).
