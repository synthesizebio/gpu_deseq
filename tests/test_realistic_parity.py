"""Realistic-scale parity tests.

The tests in test_api.py use 6-sample fixtures where pydeseq2 itself falls
into degenerate regimes (its parametric trend fit punts to the mean fallback,
and the dispersion prior is poorly estimated with residual dof < 3). Any
real comparison needs larger n to exercise the primary code paths in both
engines.

Tolerances here are derived from empirical measurement on the generated
datasets — not aspirational. See scripts/measure_drift.py for the generator
used to calibrate these values.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from gpu_deseq import DESeqDataset, fit_dispersions, fit_size_factors, results, wald_test


def _simulate(n_samples: int, n_genes: int, seed: int):
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    lfc = np.concatenate([
        rng.normal(0, 0.2, n_genes - 100),
        rng.choice([-2, -1, 1, 2], 100),
    ])
    rng.shuffle(lfc)
    base_mu = np.clip(np.exp(rng.normal(5, 2, n_genes)), 5, 5000)
    sf = np.exp(rng.normal(0, 0.2, n_samples))
    disp = 0.05 + 5.0 / base_mu
    counts = np.zeros((n_genes, n_samples), dtype=np.int64)
    for g in range(n_genes):
        for s in range(n_samples):
            group_lfc = lfc[g] if s >= half else 0.0
            mu_gs = base_mu[g] * sf[s] * np.exp(group_lfc * np.log(2))
            counts[g, s] = rng.negative_binomial(1.0 / disp[g], 1.0 / (1.0 + mu_gs * disp[g]))
    coldata = pd.DataFrame({"condition": ["control"] * half + ["treated"] * (n_samples - half)})
    return counts, coldata


def _run_gpu_deseq(counts, coldata):
    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")
    fit_size_factors(dds)
    fit_dispersions(dds)
    return results(wald_test(dds, contrast="condition[T.treated]"),
                   cooks_filter=False, independent_filter=False), dds


def _run_pydeseq2(counts, coldata):
    pytest.importorskip("pydeseq2")
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    counts_df = pd.DataFrame(
        counts.T.astype(np.int64),
        columns=[f"gene_{i}" for i in range(counts.shape[0])],
        index=[f"sample_{i}" for i in range(counts.shape[1])],
    )
    metadata = coldata.copy()
    metadata.index = counts_df.index
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~ condition",
                           quiet=True, n_cpus=1, low_memory=True)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=["condition", "treated", "control"],
                           quiet=True, n_cpus=1, cooks_filter=False, independent_filter=False)
        stats.summary()
    return stats.results_df, dds


@pytest.mark.parametrize(
    "n_samples,n_genes,seed,disp_p95_rel,lfc_p95_rel,se_p95_rel",
    [
        (30, 500,  0, 0.005, 0.010, 0.005),
        (60, 2000, 1, 0.005, 0.005, 0.005),
    ],
)
def test_matches_pydeseq2_on_realistic_data(
    n_samples: int, n_genes: int, seed: int,
    disp_p95_rel: float, lfc_p95_rel: float, se_p95_rel: float,
) -> None:
    counts, coldata = _simulate(n_samples, n_genes, seed)
    ours, our_dds = _run_gpu_deseq(counts, coldata)
    ref, ref_dds = _run_pydeseq2(counts, coldata)

    # Dispersions: p95 relative error on per-gene final α.
    our_disp = our_dds.dispersions.cpu().numpy()
    ref_disp = np.asarray(ref_dds.var["dispersions"])
    mask = np.isfinite(our_disp) & np.isfinite(ref_disp)
    rel = np.abs(our_disp[mask] - ref_disp[mask]) / (np.abs(ref_disp[mask]) + 1e-12)
    assert np.percentile(rel, 95) < disp_p95_rel, \
        f"dispersion p95 rel err {np.percentile(rel, 95):.4g} >= {disp_p95_rel}"

    # LFC / SE: p95 relative error on finite rows.
    merged = ours.join(ref[["log2FoldChange", "lfcSE", "stat", "pvalue"]], rsuffix="_ref")
    finite = np.isfinite(merged[["log2FoldChange", "log2FoldChange_ref",
                                  "lfcSE", "lfcSE_ref"]]).all(axis=1)
    merged = merged[finite]
    for col, tol in [("log2FoldChange", lfc_p95_rel), ("lfcSE", se_p95_rel)]:
        a = merged[col].to_numpy()
        b = merged[f"{col}_ref"].to_numpy()
        rel_err = np.abs(a - b) / (np.abs(b) + 1e-9)
        p95 = np.percentile(rel_err, 95)
        assert p95 < tol, f"{col} p95 rel err {p95:.4g} >= {tol}"

    # pvalue correlation must be essentially 1.
    assert np.corrcoef(merged["pvalue"], merged["pvalue_ref"])[0, 1] > 0.9999

    # Signs agree on meaningful effects.
    meaningful = np.abs(merged["log2FoldChange_ref"]) > 0.1
    sign_match = np.sign(merged["log2FoldChange"][meaningful]) == \
        np.sign(merged["log2FoldChange_ref"][meaningful])
    assert sign_match.mean() > 0.995
