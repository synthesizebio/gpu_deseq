"""Recompute standard-pipeline parity without rerunning GPU timings.

This is useful when the R reference version changes.  It runs one cuDESeq2
capture per dataset on the selected device, compares it with the outputs from
``run_r.R``, and writes a versioned artifact. It captures every execution mode
but does not alter the committed GPU timing artifact.
"""
from __future__ import annotations

import argparse
import hashlib
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
    destination = bench.RES / "reference_parity.json"
    if args.only:
        selected = set(args.only.split(","))
        unknown = selected.difference(cases)
        if unknown:
            parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case in selected]

    if args.only and destination.exists():
        output: dict[str, object] = json.loads(destination.read_text())
        output.setdefault("cases", {})
        output["device"] = args.device
        output["pipeline"] = "standard_wald_with_outlier_refit"
        output["reference_modes"] = list(bench.MODES)
    else:
        output = {
            "device": args.device,
            "pipeline": "standard_wald_with_outlier_refit",
            "reference_modes": list(bench.MODES),
            "cases": {},
        }
    output["input_manifests"] = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in (
            ("source", Path("validation/data_sources.json")),
            ("prepared", Path("validation/prepared_data_manifest.json")),
        )
        if path.exists()
    }
    for case in cases:
        print(f"[{case}] reference parity on {args.device}", flush=True)
        captures = {
            mode: run_cu.capture_mode(case, mode, args.device)
            for mode in bench.MODES
        }
        capture = captures["eager"]
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
        rows = bench.parity_for_case(case, captures)
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
                        "values_vs_r",
                        "pass_by_mode",
                        "aux",
                    )
                    if key in row
                }
                for row in rows
            ],
        }

    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
