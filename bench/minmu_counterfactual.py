"""Generate the retained airway_cell Wald-SE minmu counterfactual."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpu_deseq import DESeqDataset, fit_dispersions, fit_size_factors, results, wald_test
from gpu_deseq import api


def _called_set(frame: pd.DataFrame) -> set[str]:
    return set(frame.index[(frame["padj"] < 0.05) & frame["padj"].notna()])


def _comparison(ours: set[str], reference: set[str]) -> dict[str, int | float]:
    union = ours | reference
    return {
        "calls": len(ours),
        "intersection_with_R": len(ours & reference),
        "R_only": len(reference - ours),
        "cuDESeq2_only": len(ours - reference),
        "jaccard_with_R": len(ours & reference) / len(union) if union else 1.0,
    }


def main() -> None:
    case = "airway_cell"
    data = ROOT / "validation/data" / case
    meta = json.loads((data / "meta.json").read_text())
    counts = pd.read_csv(data / "counts.csv", index_col=0)
    coldata = pd.read_csv(data / "coldata.csv", index_col=0)
    factor, ref, nonref = meta["factor"], meta["ref"], meta["nonref"]
    levels = list(pd.unique(coldata[factor]))
    coldata[factor] = pd.Categorical(
        coldata[factor], categories=[ref] + [x for x in levels if x != ref]
    )
    contrast_name = f"{factor}[T.{nonref}]"
    genes = list(counts.index)

    dds = DESeqDataset(
        counts.to_numpy(np.float64), coldata, design=meta["design"],
        gene_ids=genes, sample_ids=list(counts.columns), backend="torch",
    )
    fit_size_factors(dds)
    fit_dispersions(dds)
    fit = wald_test(dds, contrast=contrast_name)
    contrast = api._contrast_vector(fit, contrast_name)

    floored = _called_set(results(fit))

    original = api._wald_se_and_stat

    def unfloored_wald(coefficients, mu, dispersions, design, contrast, ridge=api._core.RIDGE):
        alpha = dispersions.unsqueeze(1)
        weights = mu / (1.0 + mu * alpha)
        information = torch.einsum("sp,gs,sq->gpq", design, weights, design)
        eye = ridge * torch.eye(
            information.shape[-1], dtype=information.dtype, device=information.device
        )
        inverse = torch.linalg.inv(information + eye)
        projected = torch.einsum("gpq,q->gp", inverse, contrast)
        se = torch.sqrt(
            torch.einsum("gp,gpq,gq->g", projected, information, projected)
            .clamp_min(1e-30)
        )
        return (coefficients @ contrast) / se, se

    try:
        api._wald_se_and_stat = unfloored_wald
        unfloored = _called_set(results(fit))
    finally:
        api._wald_se_and_stat = original

    r_results = pd.read_csv(
        ROOT / "bench/cache" / case / "r_results.csv", index_col="gene"
    )
    reference = set(
        r_results.index[(r_results["padj"] < 0.05) & r_results["padj"].notna()]
    )
    artifact = {
        "provenance": {
            "case": case,
            "device": "cpu",
            "design": meta["design"],
            "contrast": contrast_name,
            "called_set_threshold": 0.05,
            "results_independent_filter_alpha": 0.1,
            "reference": "bench/cache/airway_cell/r_results.csv",
            "method": "same fit; Wald covariance recomputed with and without mu >= 0.5",
        },
        "R_calls": len(reference),
        "with_minmu_floor": _comparison(floored, reference),
        "without_minmu_floor": _comparison(unfloored, reference),
    }
    destination = ROOT / "bench/results/minmu_counterfactual.json"
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print(destination)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
