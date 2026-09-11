"""Recompute standard-pipeline parity without rerunning GPU timings.

This is useful when the R reference version changes.  It runs one cuDESeq2
capture per dataset on the selected device, compares it with the outputs from
``run_r.R``, and writes a versioned artifact. It captures every execution mode
but does not alter the committed GPU timing artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import subprocess

import pandas as pd
import torch

import run_cu


# Load the sibling orchestrator by path. ``bench`` is also a valid namespace
# package name once other benchmark helpers are imported as ``bench.*``; a
# plain ``import bench`` can therefore resolve to the package instead of this
# directory's ``bench.py`` in long-lived test or notebook processes.
_BENCH_SPEC = importlib.util.spec_from_file_location(
    "_gpu_deseq_benchmark_orchestrator", Path(__file__).with_name("bench.py")
)
assert _BENCH_SPEC is not None and _BENCH_SPEC.loader is not None
bench = importlib.util.module_from_spec(_BENCH_SPEC)
_BENCH_SPEC.loader.exec_module(bench)


def _input_reconstruction() -> dict[str, str]:
    return {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in (
            ("source_manifest_sha256", Path("validation/data_sources.json")),
            ("prepared_manifest_sha256", Path("validation/prepared_data_manifest.json")),
        )
        if path.exists()
    }


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _current_provenance(device: str) -> dict[str, object]:
    commit, dirty = _git_state()
    return {
        "pipeline": "standard_wald_with_outlier_refit",
        "device_argument": device,
        "reference_modes": list(bench.MODES),
        "input_reconstruction": _input_reconstruction(),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _case_has_all_modes(record: dict[str, object]) -> bool:
    expected = set(bench.MODES)
    metrics = record.get("metrics")
    if not isinstance(metrics, list):
        return False
    return all(
        isinstance(metric, dict)
        and set(metric.get("values_vs_r", {})) == expected
        and set(metric.get("pass_by_mode", {})) == expected
        for metric in metrics
    )


def _partial_update_is_compatible(
    output: dict[str, object], current: dict[str, object]
) -> bool:
    """Partial updates may preserve cases only under identical global provenance."""
    existing = output.get("provenance")
    if not isinstance(existing, dict) or existing != current:
        return False
    cases = output.get("cases")
    return isinstance(cases, dict) and all(
        isinstance(record, dict) and _case_has_all_modes(record)
        for record in cases.values()
    )


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

    current_provenance = _current_provenance(args.device)
    if args.only and destination.exists():
        output: dict[str, object] = json.loads(destination.read_text())
        if current_provenance["working_tree_dirty"] is not False:
            parser.error("--only requires a clean working tree")
        if not _partial_update_is_compatible(output, current_provenance):
            parser.error(
                "--only cannot preserve cases under changed or legacy global "
                "provenance; rerun without --only"
            )
    else:
        output = {
            "provenance": current_provenance,
            "cases": {},
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
