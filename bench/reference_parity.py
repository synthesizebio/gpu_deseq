"""Recompute standard-pipeline parity without rerunning GPU timings.

This is useful when the R reference version changes.  It runs one cuDESeq2
capture per dataset on the selected device, compares it with the outputs from
``run_r.R``, and writes a versioned artifact.  It does not alter the committed
GPU timing or execution-mode comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import bench
import run_cu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--only", default=None, help="comma-separated case subset")
    args = parser.parse_args()

    cases = sorted(
        path.name
        for path in bench.DATA.iterdir()
        if (path / "meta.json").exists()
    )
    if args.only:
        selected = set(args.only.split(","))
        cases = [case for case in cases if case in selected]

    output: dict[str, object] = {
        "device": args.device,
        "pipeline": "standard_wald_with_outlier_refit",
        "cases": {},
    }
    for case in cases:
        print(f"[{case}] reference parity on {args.device}", flush=True)
        capture = run_cu.capture_mode(case, "eager", args.device)
        pd.DataFrame(
            {
                key: capture[key]
                for key in (
                    "dispersion",
                    "log2FoldChange",
                    "stat",
                    "padj",
                    "baseMean",
                    "shrunk_lfc",
                    "replaced",
                )
            }
        ).to_csv(bench.CACHE / case / "cu_reference_results.csv")
        # parity_for_case also checks acceleration-mode agreement. Reusing the
        # same capture here intentionally limits this artifact to R-reference
        # parity; mode agreement remains in parity.json from an actual GPU run.
        rows = bench.parity_for_case(
            case, {"eager": capture, "graph": capture, "triton": capture}
        )
        timing_meta = json.loads(
            (bench.CACHE / case / "r_timings.json").read_text()
        )
        output["cases"][case] = {
            "r_version": timing_meta.get("r_version"),
            "deseq2_version": timing_meta.get("deseq2_version"),
            "apeglm_version": timing_meta.get("apeglm_version"),
            "metrics": [
                {
                    key: row[key]
                    for key in (
                        "substep",
                        "metric",
                        "value",
                        "tol",
                        "higher_better",
                        "pass",
                        "aux",
                    )
                    if key in row
                }
                for row in rows
            ],
        }

    destination = bench.RES / "reference_parity.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
