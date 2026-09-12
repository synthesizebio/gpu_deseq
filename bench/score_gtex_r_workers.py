#!/usr/bin/env python3
"""Verify that serial and MulticoreParam GTEx DESeq2 outputs agree."""

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


def jaccard_005(observed: np.ndarray, reference: np.ndarray) -> float:
    observed_set = np.isfinite(observed) & (observed < 0.05)
    reference_set = np.isfinite(reference) & (reference < 0.05)
    union = int(np.count_nonzero(observed_set | reference_set))
    return float(np.count_nonzero(observed_set & reference_set) / union) if union else 1.0


def pearson(observed: np.ndarray, reference: np.ndarray) -> float:
    observed, reference = finite_pair(observed, reference)
    return float(np.corrcoef(observed, reference)[0, 1])


def aligned(
    serial_dir: Path, parallel_dir: Path, filename: str, index: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    serial = pd.read_csv(serial_dir / filename).set_index(index)
    parallel = pd.read_csv(parallel_dir / filename).set_index(index)
    if not serial.index.is_unique or not parallel.index.is_unique:
        raise ValueError(f"{filename} contains duplicate identifiers")
    if set(serial.index) != set(parallel.index):
        raise ValueError(f"{filename} identifiers do not match")
    return serial, parallel.reindex(serial.index)


def maximum_absolute(observed: np.ndarray, reference: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(reference)
    return float(np.max(np.abs(observed[mask] - reference[mask])))


def score(serial_dir: Path, parallel_dir: Path) -> dict[str, object]:
    serial_sf, parallel_sf = aligned(
        serial_dir, parallel_dir, "r_sizefactors.csv", "sample"
    )
    serial_disp, parallel_disp = aligned(
        serial_dir, parallel_dir, "r_refit_dispersions.csv", "gene"
    )
    serial_result, parallel_result = aligned(
        serial_dir, parallel_dir, "r_results.csv", "gene"
    )
    serial_shrink, parallel_shrink = aligned(
        serial_dir, parallel_dir, "r_shrink.csv", "gene"
    )

    rows = [
        {
            "substep": "normalization",
            "metric": "max_relative_error",
            "value": max_relative(
                parallel_sf["sizeFactor"].to_numpy(),
                serial_sf["sizeFactor"].to_numpy(),
            ),
            "threshold": 1e-10,
            "higher_better": False,
        },
        {
            "substep": "refit_dispersion",
            "metric": "max_relative_error",
            "value": max_relative(
                parallel_disp["dispersion"].to_numpy(),
                serial_disp["dispersion"].to_numpy(),
            ),
            "threshold": 1e-10,
            "higher_better": False,
        },
        {
            "substep": "glm_fit",
            "metric": "max_absolute_lfc_error",
            "value": maximum_absolute(
                parallel_result["log2FoldChange"].to_numpy(),
                serial_result["log2FoldChange"].to_numpy(),
            ),
            "threshold": 1e-10,
            "higher_better": False,
        },
        {
            "substep": "significance",
            "metric": "jaccard_padj_005",
            "value": jaccard_005(
                parallel_result["padj"].to_numpy(),
                serial_result["padj"].to_numpy(),
            ),
            "threshold": 1.0,
            "higher_better": True,
        },
        {
            "substep": "lfc_shrink",
            "metric": "pearson",
            "value": pearson(
                parallel_shrink["log2FoldChange"].to_numpy(),
                serial_shrink["log2FoldChange"].to_numpy(),
            ),
            "threshold": 0.9999999999,
            "higher_better": True,
            "max_absolute_lfc_error": maximum_absolute(
                parallel_shrink["log2FoldChange"].to_numpy(),
                serial_shrink["log2FoldChange"].to_numpy(),
            ),
        },
    ]
    for row in rows:
        row["pass"] = bool(
            row["value"] >= row["threshold"]
            if row["higher_better"]
            else row["value"] <= row["threshold"]
        )
    filenames = (
        "r_sizefactors.csv",
        "r_refit_dispersions.csv",
        "r_results.csv",
        "r_shrink.csv",
        "r_direct_timings.json",
    )
    return {
        "serial_files_sha256": {
            name: digest(serial_dir / name) for name in filenames
        },
        "parallel_files_sha256": {
            name: digest(parallel_dir / name) for name in filenames
        },
        "n_samples": len(serial_sf),
        "n_genes": len(serial_result),
        "metrics": rows,
        "pass": all(row["pass"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("serial_directory", type=Path)
    parser.add_argument("parallel_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.serial_directory, args.parallel_directory)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
