"""Benchmark harness, PyDESeq2 side (COMPETITOR — never a parity target).

PyDESeq2 is a third-party CPU/Python reimplementation of DESeq2. Here it is a
*competitor* we time and score against the R DESeq2 ground truth, exactly as we
do for cuDESeq2 — it is NOT the reference and its numerics are never used to
validate ours.

ISOLATION CONTRACT (do not break):
  * `pydeseq2` is imported ONLY inside this module, and lazily (inside the
    functions). It must NEVER be imported by `src/gpu_deseq/**`, by `run_cu.py`,
    or added to the package's runtime dependencies in `pyproject.toml`.
  * Nothing in this file is imported by the cuDESeq2 implementation.
  * PyDESeq2 is a benchmark-only requirement: `pip install pydeseq2` in the
    bench environment. If it is absent, this engine is skipped, not required.

For a prepared case (validation/data/<case>/) it both times the five pipeline
substeps and captures the intermediates (size factors, dispersions, raw LFC,
padj, apeGLM-shrunk LFC) in the same schema as run_cu, so bench.py can score
PyDESeq2-vs-R with the identical metrics used for cuDESeq2-vs-R.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("validation/data")
SUBSTEPS = ["normalization", "dispersion", "glm_fit", "significance", "lfc_shrink"]


def available() -> bool:
    """True iff pydeseq2 can be imported in this environment."""
    try:
        import pydeseq2  # noqa: F401
        return True
    except Exception:
        return False


def _load(case):
    d = DATA / case
    meta = json.loads((d / "meta.json").read_text())
    counts = pd.read_csv(d / "counts.csv", index_col=0)      # genes x samples
    coldata = pd.read_csv(d / "coldata.csv", index_col=0)    # samples x factors
    counts_sg = counts.T.astype(int)                          # samples x genes (pydeseq2 layout)
    # Force the contrast factor's reference level: with a formula-string design,
    # formulaic picks the reference from category order, so put `ref` first. This
    # makes the LFC coefficient formulaic's `factor[T.nonref]` (see `_coeff`).
    factor, ref = meta["factor"], meta["ref"]
    levels = [ref] + [l for l in pd.unique(coldata[factor]) if l != ref]
    coldata[factor] = pd.Categorical(coldata[factor], categories=levels)
    return meta, counts_sg, coldata


def _coeff(meta):
    return f'{meta["factor"]}[T.{meta["nonref"]}]'


def _new_dds(meta, counts_sg, coldata, n_cpus):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    inf = DefaultInference(n_cpus=n_cpus)
    return DeseqDataSet(
        counts=counts_sg, metadata=coldata, design=meta["design"],
        ref_level=[meta["factor"], meta["ref"]], inference=inf, quiet=True,
    ), inf


def _run(meta, counts_sg, coldata, n_cpus, timed):
    """Run PyDESeq2 end-to-end. If timed, return {substep: seconds}; else return
    the captured intermediates as pandas objects keyed like run_cu."""
    from pydeseq2.ds import DeseqStats
    genes = list(counts_sg.columns)
    samples = list(counts_sg.index)
    contrast = [meta["factor"], meta["nonref"], meta["ref"]]
    coeff = _coeff(meta)

    dds, inf = _new_dds(meta, counts_sg, coldata, n_cpus)
    t = {}

    s = time.perf_counter(); dds.fit_size_factors(); t["normalization"] = time.perf_counter() - s

    s = time.perf_counter()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    t["dispersion"] = time.perf_counter() - s

    s = time.perf_counter()
    dds.fit_LFC()
    dds.calculate_cooks()
    if dds.refit_cooks:
        dds.refit()
    t["glm_fit"] = time.perf_counter() - s

    s = time.perf_counter()
    stat = DeseqStats(dds, contrast=contrast, inference=inf, quiet=True)
    stat.summary()  # Wald test + Cook's filter + independent filtering + BH
    t["significance"] = time.perf_counter() - s

    raw_lfc = stat.results_df["log2FoldChange"].copy()
    padj = stat.results_df["padj"].copy()
    base_mean = stat.results_df["baseMean"].copy()
    stat_col = stat.results_df["stat"].copy()

    s = time.perf_counter()
    stat.lfc_shrink(coeff=coeff)  # apeGLM; mutates results_df["log2FoldChange"] in place
    t["lfc_shrink"] = time.perf_counter() - s
    shrunk_lfc = stat.results_df["log2FoldChange"].copy()

    if timed:
        t["total"] = sum(t.values())
        return t

    sf = np.asarray(dds.obs["size_factors"].values, dtype=float)
    disp = pd.Series(np.asarray(dds.var["dispersions"].values, dtype=float), index=genes)
    return {
        "sizeFactor":     pd.Series(sf, index=samples),
        "dispersion":     disp,
        "log2FoldChange": raw_lfc.reindex(genes),
        "stat":           stat_col.reindex(genes),
        "padj":           padj.reindex(genes),
        "baseMean":       base_mean.reindex(genes),
        "shrunk_lfc":     shrunk_lfc.reindex(genes),
    }


def time_pydeseq2(case, reps, n_cpus=None):
    meta, counts_sg, coldata = _load(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run(meta, counts_sg, coldata, n_cpus, timed=True)  # warm-up (joblib workers, imports)
        acc = {k: [] for k in SUBSTEPS}
        for _ in range(reps):
            t = _run(meta, counts_sg, coldata, n_cpus, timed=True)
            for k in SUBSTEPS:
                acc[k].append(t[k])
    out = {k: float(np.median(v)) * 1e3 for k, v in acc.items()}
    out["total"] = sum(out.values())
    return out


def capture_pydeseq2(case, n_cpus=None):
    meta, counts_sg, coldata = _load(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _run(meta, counts_sg, coldata, n_cpus, timed=False)


if __name__ == "__main__":
    import sys
    case = sys.argv[1] if len(sys.argv) > 1 else "pasilla"
    print(json.dumps(time_pydeseq2(case, reps=2, n_cpus=os.cpu_count()), indent=2))
