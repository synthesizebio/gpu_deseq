"""Load R DESeq2 fixtures, run gpu_deseq on the same counts, print drift table.

Run after scripts/generate_r_fixtures.R has populated fixtures/r_deseq2/. Prints
per-fixture drift statistics so we can set test tolerances from measured values
instead of guessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gpu_deseq import DESeqDataset, fit_dispersions, fit_size_factors, results, wald_test


def _load(label: str):
    path = REPO_ROOT / "fixtures" / "r_deseq2" / label
    counts = pd.read_csv(path / "counts.csv", index_col=0)
    coldata = pd.read_csv(path / "coldata.csv", index_col=0)
    ref = pd.read_csv(path / "results.csv").set_index("gene")
    return counts, coldata, ref


def _run(counts, coldata, gene_ids):
    dds = DESeqDataset(counts.to_numpy(dtype=np.float64), coldata, design="~ condition",
                       gene_ids=gene_ids, backend="torch")
    fit_size_factors(dds)
    fit_dispersions(dds)
    res = results(wald_test(dds, contrast="condition[T.treated]"),
                  cooks_filter=False, independent_filter=False)
    return res, dds


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    m = np.isfinite(a) & np.isfinite(b)
    abs_err = np.abs(a[m] - b[m])
    rel_err = abs_err / (np.abs(b[m]) + 1e-12)
    return {
        "n": int(m.sum()),
        "max_abs": float(abs_err.max()) if m.any() else np.nan,
        "p95_abs": float(np.percentile(abs_err, 95)) if m.any() else np.nan,
        "max_rel": float(rel_err.max()) if m.any() else np.nan,
        "p95_rel": float(np.percentile(rel_err, 95)) if m.any() else np.nan,
        "corr":    float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 2 else np.nan,
    }


def main():
    labels = [d.name for d in sorted((REPO_ROOT / "fixtures" / "r_deseq2").iterdir()) if d.is_dir()]
    for label in labels:
        print(f"\n===== {label} =====")
        counts, coldata, ref = _load(label)
        ours, dds = _run(counts, coldata, list(counts.index))
        print(f"  shape: {counts.shape[0]} genes x {counts.shape[1]} samples")
        our_disp = dds.dispersions.cpu().numpy()
        print(f"  dispersion:       {_stats(our_disp, ref['dispersion'].to_numpy())}")
        merged = ours.join(ref, rsuffix="_r")
        for col in ["log2FoldChange", "lfcSE", "stat", "pvalue"]:
            a = merged[col].to_numpy()
            b = merged[f"{col}_r"].to_numpy()
            print(f"  {col:16s}: {_stats(a, b)}")


if __name__ == "__main__":
    main()
