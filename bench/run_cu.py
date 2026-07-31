"""Benchmark harness, cuDESeq2 side.

For a prepared case (validation/data/<case>/) and an execution mode
(eager / graph / triton) this both:
  * times each of the five standard-pipeline substeps (normalization,
    dispersion, Wald fit plus Cook's-outlier replacement/refit, significance,
    lfc_shrink), median over `reps`; and
  * captures every per-substep intermediate (size factors, dispersions, LFC,
    Wald stat, padj, shrunk LFC) so equivalence can be verified.

Only the *dispersion* substep is affected by the mode flags; the other four run
the identical eager code in every mode (the harness makes that visible rather
than hiding it).

Imported by bench.py; also runnable standalone for one case/mode.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, "src")
from gpu_deseq import (DESeqDataset, fit_size_factors, fit_dispersions,
                       wald_test, results, lfc_shrink)
from gpu_deseq.api import _replace_outliers_and_refit_wald
import gpu_deseq._deseq2_core as _core

DATA = Path("validation/data")
SUBSTEPS = ["normalization", "dispersion", "glm_fit", "significance", "lfc_shrink"]
MODES = {"eager": {}, "graph": {"use_cuda_graph": True}, "triton": {"use_triton": True}}


def _load(case):
    d = DATA / case
    meta = json.loads((d / "meta.json").read_text())
    counts = pd.read_csv(d / "counts.csv", index_col=0)
    coldata = pd.read_csv(d / "coldata.csv", index_col=0)
    factor, ref, nonref = meta["factor"], meta["ref"], meta["nonref"]
    levels = list(pd.unique(coldata[factor]))
    coldata[factor] = pd.Categorical(coldata[factor], categories=[ref] + [l for l in levels if l != ref])
    contrast = f"{factor}[T.{nonref}]"
    return meta, counts, coldata, contrast


def _build(counts, coldata, meta, device):
    return DESeqDataset(counts.to_numpy(np.float64), coldata, design=meta["design"],
                        gene_ids=list(counts.index), sample_ids=list(counts.columns),
                        backend="torch").to(device)


def _run_substeps(dds, contrast, disp_kw, timed, device):
    """Run the 5 substeps in order; if `timed`, return {substep: seconds},
    else return the captured intermediates as a dict of pandas objects."""
    cuda = device == "cuda"
    def sync():
        if cuda: torch.cuda.synchronize()

    t = {}
    sync(); s = time.perf_counter(); fit_size_factors(dds);        sync(); t["normalization"] = time.perf_counter() - s
    sync(); s = time.perf_counter(); fit_dispersions(dds, **disp_kw); sync(); t["dispersion"] = time.perf_counter() - s
    sync(); s = time.perf_counter()
    fit = wald_test(dds, contrast=contrast)
    fit = _replace_outliers_and_refit_wald(
        dds, fit, use_cuda_graph=disp_kw.get("use_cuda_graph", False),
        use_triton=disp_kw.get("use_triton", False),
    )
    sync(); t["glm_fit"] = time.perf_counter() - s
    sync(); s = time.perf_counter(); res = results(fit);           sync(); t["significance"] = time.perf_counter() - s
    sync(); s = time.perf_counter()
    shr = results(lfc_shrink(fit, coeff=contrast), cooks_filter=False, independent_filter=False)
    sync(); t["lfc_shrink"] = time.perf_counter() - s

    if timed:
        return t
    genes = list(dds.gene_ids)
    return {
        "sizeFactor":  pd.Series(dds.size_factors.cpu().numpy(), index=list(dds.sample_ids)),
        "dispersion":  pd.Series(dds.dispersions.cpu().numpy(), index=genes),
        "log2FoldChange": res["log2FoldChange"].reindex(genes),
        "stat":        res["stat"].reindex(genes),
        "padj":        res["padj"].reindex(genes),
        "baseMean":    res["baseMean"].reindex(genes),
        "shrunk_lfc":  shr["log2FoldChange"].reindex(genes),
        "replaced":    pd.Series(
            fit.replaced_genes.cpu().numpy()
            if fit.replaced_genes is not None
            else np.zeros(len(genes), dtype=bool),
            index=genes,
        ),
    }


def time_mode(case, mode, device, reps):
    meta, counts, coldata, contrast = _load(case)
    disp_kw = MODES[mode]
    _core._GRAPH_CACHE.clear()
    # warm-up (compile / capture / cache), not timed
    _run_substeps(_build(counts, coldata, meta, device), contrast, disp_kw, timed=True, device=device)
    acc = {k: [] for k in SUBSTEPS}
    for _ in range(reps):
        dds = _build(counts, coldata, meta, device)
        t = _run_substeps(dds, contrast, disp_kw, timed=True, device=device)
        for k in SUBSTEPS:
            acc[k].append(t[k])
    out = {k: float(np.median(v)) * 1e3 for k, v in acc.items()}
    out["total"] = sum(out.values())
    return out


def capture_mode(case, mode, device):
    meta, counts, coldata, contrast = _load(case)
    _core._GRAPH_CACHE.clear()
    dds = _build(counts, coldata, meta, device)
    return _run_substeps(dds, contrast, disp_kw=MODES[mode], timed=False, device=device)


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "pasilla"
    mode = sys.argv[2] if len(sys.argv) > 2 else "triton"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps(time_mode(case, mode, dev, reps=3), indent=2))
