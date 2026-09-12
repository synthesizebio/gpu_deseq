from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bench.score_gtex_r_workers import score


def write_case(path: Path, offset: float = 0.0) -> None:
    path.mkdir()
    pd.DataFrame(
        {"sample": ["a", "b"], "sizeFactor": [1.0, 2.0]}
    ).to_csv(path / "r_sizefactors.csv", index=False)
    pd.DataFrame(
        {"gene": ["g1", "g2", "g3"], "dispersion": [0.1, 0.2, 0.3 + offset]}
    ).to_csv(path / "r_refit_dispersions.csv", index=False)
    pd.DataFrame(
        {
            "gene": ["g1", "g2", "g3"],
            "log2FoldChange": [1.0, -2.0, 0.2 + offset],
            "padj": [0.01, 0.2, np.nan],
        }
    ).to_csv(path / "r_results.csv", index=False)
    pd.DataFrame(
        {"gene": ["g1", "g2", "g3"], "log2FoldChange": [0.9, -1.8, 0.1 + offset]}
    ).to_csv(path / "r_shrink.csv", index=False)
    (path / "r_direct_timings.json").write_text(json.dumps({"status": "pass"}))


def test_identical_worker_outputs_pass(tmp_path: Path) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    write_case(serial)
    write_case(parallel)
    result = score(serial, parallel)
    assert result["pass"] is True
    assert result["n_samples"] == 2
    assert result["n_genes"] == 3


def test_worker_output_difference_fails_strict_gate(tmp_path: Path) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    write_case(serial)
    write_case(parallel, offset=1e-3)
    result = score(serial, parallel)
    assert result["pass"] is False
    failed = {row["substep"] for row in result["metrics"] if not row["pass"]}
    assert {"refit_dispersion", "glm_fit"}.issubset(failed)
