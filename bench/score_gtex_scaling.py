#!/usr/bin/env python3
"""Score a captured large-GTEx cuDESeq2 result against R DESeq2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_pair(observed: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(observed) & np.isfinite(reference)
    return observed[mask], reference[mask]


def max_relative(observed: np.ndarray, reference: np.ndarray) -> float:
    observed, reference = finite_pair(observed, reference)
    return float(np.max(np.abs(observed - reference) / (np.abs(reference) + 1e-12)))


def p95_relative(observed: np.ndarray, reference: np.ndarray) -> float:
    observed, reference = finite_pair(observed, reference)
    return float(
        np.percentile(np.abs(observed - reference) / (np.abs(reference) + 1e-12), 95)
    )


def p95_absolute(observed: np.ndarray, reference: np.ndarray) -> float:
    observed, reference = finite_pair(observed, reference)
    return float(np.percentile(np.abs(observed - reference), 95))


def jaccard_005(observed: np.ndarray, reference: np.ndarray) -> float:
    observed_set = np.isfinite(observed) & (observed < 0.05)
    reference_set = np.isfinite(reference) & (reference < 0.05)
    union = int(np.count_nonzero(observed_set | reference_set))
    return (
        float(np.count_nonzero(observed_set & reference_set) / union)
        if union
        else 1.0
    )


def pearson(observed: np.ndarray, reference: np.ndarray) -> float:
    observed, reference = finite_pair(observed, reference)
    return float(np.corrcoef(observed, reference)[0, 1])


def spearman(observed: np.ndarray, reference: np.ndarray) -> float:
    observed, reference = finite_pair(observed, reference)
    return float(
        np.corrcoef(pd.Series(observed).rank(), pd.Series(reference).rank())[0, 1]
    )


def score(gpu_path: Path, r_dir: Path) -> dict[str, object]:
    gpu = np.load(gpu_path)
    genes = gpu["gene_id"].astype(str)
    size_factors = pd.read_csv(r_dir / "r_sizefactors.csv")["sizeFactor"].to_numpy()
    dispersions = pd.read_csv(r_dir / "r_dispersions.csv").set_index("gene")
    results = pd.read_csv(r_dir / "r_results.csv").set_index("gene")
    shrunk = pd.read_csv(r_dir / "r_shrink.csv").set_index("gene")
    for frame_name, frame in (
        ("dispersions", dispersions),
        ("results", results),
        ("shrink", shrunk),
    ):
        if set(frame.index) != set(genes):
            raise ValueError(f"{frame_name} gene IDs do not match GPU capture")
    dispersions = dispersions.reindex(genes)
    results = results.reindex(genes)
    shrunk = shrunk.reindex(genes)
    if size_factors.shape != gpu["size_factor"].shape:
        raise ValueError("size-factor lengths do not match")

    rows = [
        {
            "substep": "normalization",
            "metric": "max_relative_error",
            "value": max_relative(gpu["size_factor"], size_factors),
            "threshold": 1e-6,
            "higher_better": False,
        },
        {
            "substep": "dispersion",
            "metric": "p95_relative_error",
            "value": p95_relative(gpu["dispersion"], dispersions["dispersion"].to_numpy()),
            "threshold": 0.1,
            "higher_better": False,
        },
        {
            "substep": "glm_fit",
            "metric": "p95_absolute_lfc_error",
            "value": p95_absolute(
                gpu["log2_fold_change"], results["log2FoldChange"].to_numpy()
            ),
            "threshold": 0.01,
            "higher_better": False,
        },
        {
            "substep": "significance",
            "metric": "jaccard_padj_005",
            "value": jaccard_005(gpu["padj"], results["padj"].to_numpy()),
            "threshold": 0.95,
            "higher_better": True,
        },
        {
            "substep": "lfc_shrink",
            "metric": "pearson",
            "value": pearson(gpu["shrunk_lfc"], shrunk["log2FoldChange"].to_numpy()),
            "threshold": 0.9,
            "higher_better": True,
            "spearman": spearman(
                gpu["shrunk_lfc"], shrunk["log2FoldChange"].to_numpy()
            ),
        },
    ]
    for row in rows:
        row["pass"] = bool(
            row["value"] >= row["threshold"]
            if row["higher_better"]
            else row["value"] <= row["threshold"]
        )
    return {
        "gpu_capture_sha256": digest(gpu_path),
        "r_files_sha256": {
            path.name: digest(path)
            for path in sorted(r_dir.glob("r_*.csv"))
        },
        "n_genes": len(genes),
        "metrics": rows,
        "pass": all(row["pass"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_capture", type=Path)
    parser.add_argument("r_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.gpu_capture, args.r_directory)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
