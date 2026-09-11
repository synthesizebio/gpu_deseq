from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_sweep_default_records_full_timing_protocol() -> None:
    module = _load_script(
        "bench_cuda_graph_contract", ROOT / "benchmarks/bench_cuda_graph.py"
    )

    assert module.SAMPLE_SWEEP_N_TIMED == 7
    provenance = module.provenance(n_timed=module.SAMPLE_SWEEP_N_TIMED)
    assert provenance["n_timed"] == 7
    assert provenance["n_warmup"] == 3


def test_reference_parity_rejects_stale_partial_run_provenance() -> None:
    sys.path.insert(0, str(ROOT / "bench"))
    try:
        module = _load_script(
            "reference_parity_contract", ROOT / "bench/reference_parity.py"
        )
    finally:
        sys.path.pop(0)

    modes = ["eager", "graph", "triton"]
    provenance = {
        "pipeline": "standard_wald_with_outlier_refit",
        "device_argument": "cuda",
        "reference_modes": modes,
        "input_reconstruction": {
            "source_manifest_sha256": "a" * 64,
            "prepared_manifest_sha256": "b" * 64,
        },
        "git_commit": "c" * 40,
        "working_tree_dirty": False,
        "python_version": "3.12.3",
        "torch_version": "2.7.1",
        "torch_cuda_version": "12.6",
    }
    output = {
        "provenance": provenance,
        "cases": {
            "pasilla": {
                "metrics": [
                    {
                        "values_vs_r": {mode: 1.0 for mode in modes},
                        "pass_by_mode": {mode: True for mode in modes},
                    }
                ]
            }
        },
    }

    assert module._partial_update_is_compatible(output, provenance)
    changed = dict(provenance)
    changed["input_reconstruction"] = {
        **provenance["input_reconstruction"],
        "prepared_manifest_sha256": "d" * 64,
    }
    assert not module._partial_update_is_compatible(output, changed)

    output["cases"]["pasilla"]["metrics"][0]["values_vs_r"].pop("triton")
    assert not module._partial_update_is_compatible(output, provenance)


def test_committed_paper_artifacts_pass_audit() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_paper_numbers.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL CHECKED MANUSCRIPT NUMBERS MATCH" in result.stdout


def test_regenerated_timing_schema_passes_paper_audit(tmp_path: Path) -> None:
    required_files = [
        "paper/main.tex",
        "validation/data_sources.json",
        "validation/prepared_data_manifest.json",
        "bench/results/timings.json",
        "bench/results/reference_parity.json",
        "bench/results/parity.json",
        "bench/results/r_parallel_a100_12worker.json",
        "benchmarks/results_a100_samplesweep_2026-09-11.json",
    ]
    required_files.extend(
        f"validation/results/{case}.json"
        for case in (
            "pasilla",
            "pasilla_2fac",
            "airway_dex",
            "airway_cell",
            "airway",
            "gtex_blood_muscle",
        )
    )
    for relative in required_files:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    timing_path = tmp_path / "bench/results/timings.json"
    timing = json.loads(timing_path.read_text())
    provenance = timing["provenance"]
    provenance.pop("direct_end_to_end")
    provenance["cu_total_definition"] = (
        "median direct wall time including DESeqDataset construction and "
        "host-to-device transfer"
    )
    provenance["cu_stage_total_definition"] = (
        "sum of independently measured stage medians"
    )
    for record in timing["r"].values():
        record["stage_total"] = record["total"]
        record["total_values_ms"] = [record["total"]]
    for modes in timing["cu"].values():
        for mode in ("eager", "graph", "triton"):
            record = modes[mode]
            record["stage_total"] = record["total"]
            record["total_values_ms"] = [record["total"]]
    timing_path.write_text(json.dumps(timing))

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_paper_numbers.py")],
        cwd=ROOT,
        env={**os.environ, "GPU_DESEQ_AUDIT_ROOT": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL CHECKED MANUSCRIPT NUMBERS MATCH" in result.stdout
