"""Measure gpu_deseq drift vs pydeseq2 on a realistic synthetic dataset.

6-sample toy fixtures put pydeseq2 into a degenerate regime where its own trend
fit falls back to mean. This script runs at n_samples=30, n_genes=1000 so both
engines run on their primary paths, and reports the actual numerical gap.
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gpu_deseq import DESeqDataset, fit_dispersions, fit_size_factors, results, wald_test


def make_data(n_samples: int = 30, n_genes: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    true_lfc = np.concatenate([
        rng.normal(0, 0.2, n_genes - 100),
        rng.choice([-2, -1, 1, 2], 100),
    ])
    rng.shuffle(true_lfc)
    base_mu = np.exp(rng.normal(5, 2, n_genes)).clip(5, 5000)
    size_factors = np.exp(rng.normal(0, 0.2, n_samples))
    disp_true = 0.05 + 5.0 / base_mu  # DESeq2-like trend
    counts = np.zeros((n_genes, n_samples), dtype=np.int64)
    for g in range(n_genes):
        for s in range(n_samples):
            group_lfc = true_lfc[g] if s >= half else 0.0
            mu_gs = base_mu[g] * size_factors[s] * np.exp(group_lfc * np.log(2))
            counts[g, s] = rng.negative_binomial(1.0 / disp_true[g], 1.0 / (1.0 + mu_gs * disp_true[g]))
    coldata = pd.DataFrame({"condition": ["control"] * half + ["treated"] * (n_samples - half)})
    return counts, coldata, true_lfc


def run_gpu(counts, coldata):
    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")
    fit_size_factors(dds)
    fit_dispersions(dds)
    return results(wald_test(dds, contrast="condition[T.treated]"))


def run_pydeseq2(counts, coldata):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts_df = pd.DataFrame(
        counts.T.astype(np.int64),
        columns=[f"gene_{i}" for i in range(counts.shape[0])],
        index=[f"sample_{i}" for i in range(counts.shape[1])],
    )
    metadata = coldata.copy()
    metadata.index = counts_df.index
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~ condition",
                           quiet=True, n_cpus=1, low_memory=True)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=["condition", "treated", "control"],
                           quiet=True, n_cpus=1, cooks_filter=False, independent_filter=False)
        stats.summary()
    return stats.results_df, dds


def compare(ours: pd.DataFrame, ref: pd.DataFrame, label: str) -> None:
    merged = ours.join(ref[["log2FoldChange", "lfcSE", "stat", "pvalue"]], rsuffix="_ref")
    merged = merged[np.isfinite(merged[["log2FoldChange", "log2FoldChange_ref", "lfcSE", "lfcSE_ref"]]).all(axis=1)]
    print(f"\n=== {label} (n={len(merged)} genes) ===")
    for col in ["log2FoldChange", "lfcSE", "stat", "pvalue"]:
        a = merged[col].to_numpy()
        b = merged[f"{col}_ref"].to_numpy()
        abs_err = np.abs(a - b)
        rel_err = np.abs(a - b) / (np.abs(b) + 1e-12)
        print(f"  {col:17s}  max|abs|={abs_err.max():.4g}  p95|abs|={np.percentile(abs_err, 95):.4g}  "
              f"max rel={rel_err.max():.4g}  p95 rel={np.percentile(rel_err, 95):.4g}  "
              f"corr={np.corrcoef(a, b)[0,1]:.6f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--n-genes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    counts, coldata, _ = make_data(args.n_samples, args.n_genes, args.seed)
    print(f"n_samples={args.n_samples} n_genes={args.n_genes} seed={args.seed}")
    ours = run_gpu(counts, coldata)
    ref, dds = run_pydeseq2(counts, coldata)

    # Dispersion drift
    # gpu_deseq keeps its dispersion on the dataset; rerun to grab it
    from gpu_deseq import DESeqDataset as _D
    dds2 = _D(counts, coldata, design="~ condition", backend="torch")
    fit_size_factors(dds2)
    fit_dispersions(dds2)
    our_disp = dds2.dispersions.cpu().numpy()
    ref_disp = np.asarray(dds.var["dispersions"])
    ref_genewise = np.asarray(dds.var["genewise_dispersions"])
    our_genewise = dds2.dispersions_gene_wise.cpu().numpy()
    for label, a, b in [
        ("genewise α (MLE)", our_genewise, ref_genewise),
        ("final α (MAP)",     our_disp,    ref_disp),
    ]:
        mask = np.isfinite(a) & np.isfinite(b)
        d_abs = np.abs(a[mask] - b[mask])
        d_rel = d_abs / (np.abs(b[mask]) + 1e-12)
        print(f"\n=== {label} (n={mask.sum()}) ===")
        print(f"  max|abs|={d_abs.max():.4g}  p95|abs|={np.percentile(d_abs, 95):.4g}  "
              f"max rel={d_rel.max():.4g}  p95 rel={np.percentile(d_rel, 95):.4g}")

    compare(ours, ref, "Wald results")
