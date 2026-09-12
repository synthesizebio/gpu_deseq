#!/usr/bin/env python3
"""Assemble raw real-GTEx scaling outputs into a committed artifact and table."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TIMING_CASES = (
    "p2_300",
    "p2_600",
    "p2_912",
    "p3_fixed300",
    "p4_fixed300",
    "p5_fixed300",
    "p6_fixed300",
    "p6_all",
)
PROBE_CASES = ("p8_all", "p9_all", "p10_all", "p12_all")
R_CASES = ("p2_912", "p6_all")
DIRECT_WORKERS = (1, 12)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def timestamp(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()


def assemble(run_dir: Path) -> dict[str, Any]:
    source_manifest_path = ROOT / "validation" / "data_sources.json"
    prepared_manifest_path = ROOT / "validation" / "prepared_data_manifest.json"
    source_manifest = read(source_manifest_path)
    prepared_manifest = read(prepared_manifest_path)
    gtex_source = next(
        source
        for source in source_manifest["sources"]
        if source["id"] == "gtex_recount2_srp012682"
    )
    timings: dict[str, Any] = {}
    probes: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for case in TIMING_CASES:
        path = run_dir / f"{case}_gpu.json"
        record = read(path)
        if record["status"] != "pass":
            raise ValueError(f"timing case did not pass: {case}")
        if (record["warmups"], record["reps"], record["direct_reps"]) != (1, 5, 5):
            raise ValueError(f"timing contract mismatch: {case}")
        if record["working_tree_dirty_at_start"] is not False:
            raise ValueError(f"timing case was not run clean: {case}")
        timings[case] = record
        input_hashes[path.name] = digest(path)
    commits = {record["git_commit"] for record in timings.values()}
    if len(commits) != 1:
        raise ValueError(f"GPU timing cases do not share one commit: {commits}")

    for case in PROBE_CASES:
        path = run_dir / f"{case}_probe.json"
        record = read(path)
        expected = "pass" if case in {"p8_all", "p9_all"} else "oom"
        if record["status"] != expected:
            raise ValueError(f"unexpected probe result for {case}: {record['status']}")
        if record["working_tree_dirty_at_start"] is not False:
            raise ValueError(f"probe was not run clean: {case}")
        probes[case] = record
        input_hashes[path.name] = digest(path)

    r_endpoints: dict[str, Any] = {}
    for case in R_CASES:
        r_path = run_dir / f"{case}_r" / "r_timings.json"
        dispersion_path = run_dir / f"{case}_r" / "r_dispersion_stage.json"
        parity_path = run_dir / f"{case}_parity.json"
        r_record = read(r_path)
        dispersion_record = read(dispersion_path)
        parity = read(parity_path)
        if r_record["status"] != "pass" or parity["pass"] is not True:
            raise ValueError(f"R endpoint or parity failed: {case}")
        if r_record["reps"] != 1 or r_record["direct_reps"] != 0:
            raise ValueError(f"R endpoint is not the documented staged observation: {case}")
        if r_record["working_tree_dirty_at_start"] is not False:
            raise ValueError(f"R endpoint was not run clean: {case}")
        r_endpoints[case] = {
            "timing": r_record,
            "dispersion_capture": dispersion_record,
            "parity": parity,
            "stage_speedup_vs_gpu": (
                r_record["stage_total_ms"] / timings[case]["stage_total_ms"]
            ),
        }
        for path in (r_path, dispersion_path, parity_path):
            input_hashes[str(path.relative_to(run_dir))] = digest(path)

    r_direct_endpoints: dict[str, Any] = {}
    direct_commits: set[str] = set()
    for case in R_CASES:
        worker_records: dict[int, Any] = {}
        for workers in DIRECT_WORKERS:
            directory = run_dir / f"{case}_r_direct_{workers}"
            path = directory / "r_direct_timings.json"
            record = read(path)
            if record["status"] != "pass":
                raise ValueError(f"direct R endpoint did not pass: {case}/{workers}")
            if (record["workers"], record["warmups"], record["reps"]) != (
                workers,
                0,
                1,
            ):
                raise ValueError(f"direct R endpoint contract mismatch: {case}/{workers}")
            if record["working_tree_dirty_at_start"] is not False:
                raise ValueError(f"direct R endpoint was not run clean: {case}/{workers}")
            if record["gpu_case_sha256"] != digest(run_dir / f"{case}_gpu.json"):
                raise ValueError(f"direct R endpoint used wrong GPU cohort: {case}/{workers}")
            if record["source_rdata_sha256"] != gtex_source["sha256"]:
                raise ValueError(f"direct R endpoint used wrong source: {case}/{workers}")
            if record["parallel"] is not (workers > 1):
                raise ValueError(f"direct R endpoint parallel flag mismatch: {case}/{workers}")
            for filename, expected_hash in record["result_file_sha256"].items():
                if digest(directory / filename) != expected_hash:
                    raise ValueError(
                        f"direct R endpoint result hash mismatch: {case}/{workers}/{filename}"
                    )
            worker_records[workers] = record
            direct_commits.add(record["git_commit"])
            input_hashes[str(path.relative_to(run_dir))] = digest(path)

        parity_path = run_dir / f"{case}_r_worker_parity.json"
        worker_parity = read(parity_path)
        if worker_parity["pass"] is not True:
            raise ValueError(f"R worker parity failed: {case}")
        input_hashes[parity_path.name] = digest(parity_path)
        serial = worker_records[1]
        parallel = worker_records[12]
        gpu_ms = timings[case]["direct_median_ms"]
        r_direct_endpoints[case] = {
            "one_worker": serial,
            "twelve_workers": parallel,
            "worker_parity": worker_parity,
            "r_parallel_speedup": serial["direct_median_ms"]
            / parallel["direct_median_ms"],
            "gpu_speedup_vs_one_worker": serial["direct_median_ms"] / gpu_ms,
            "gpu_speedup_vs_twelve_workers": parallel["direct_median_ms"] / gpu_ms,
        }
    if len(direct_commits) != 1:
        raise ValueError(
            f"direct R endpoints do not share one clean harness commit: {direct_commits}"
        )

    files = [run_dir / name for name in input_hashes]
    observed_end = max(timestamp(path) for path in files)
    first_timing = run_dir / f"{TIMING_CASES[0]}_gpu.json"
    observed_start = dt.datetime.fromtimestamp(
        first_timing.stat().st_mtime - timings[TIMING_CASES[0]]["elapsed_s"],
        dt.timezone.utc,
    ).isoformat()
    representative = timings["p2_300"]
    return {
        "schema_version": 1,
        "provenance": {
            "source": "checksum-pinned recount2 SRP012682 GTEx RangedSummarizedExperiment",
            "source_manifest_sha256": digest(source_manifest_path),
            "source_rdata_sha256": gtex_source["sha256"],
            "paper_counts_sha256": prepared_manifest["cases"]["gtex_blood_muscle"]["files"]["counts.csv"]["sha256"],
            "n_source_samples": 9662,
            "n_source_tissues": 54,
            "n_genes": 54922,
            "matrix_metadata_sha256": representative["matrix_metadata_sha256"],
            "matrix_binary_sha256": representative["matrix_binary_sha256"],
            "gpu_timing_commit": next(iter(commits)),
            "gpu_name": representative["gpu_name"],
            "torch_version": representative["torch_version"],
            "torch_cuda_version": representative["torch_cuda_version"],
            "nvidia_driver_version": representative["nvidia_driver_version"],
            "observed_utc_start_from_output_mtime": observed_start,
            "observed_utc_end_from_output_mtime": observed_end,
            "gpu_timing_contract": (
                "one warm-up, five staged repetitions, and five direct repetitions; "
                "CUDA-synchronized medians; source-matrix extraction excluded"
            ),
            "oom_probe_contract": (
                "one cold staged workflow per fresh process; no separate direct run"
            ),
            "r_endpoint_contract": (
                "one cold, one-worker staged observation; controlled stage diagnostic "
                "and parity reference only, not the practical CPU baseline"
            ),
            "primary_cpu_baseline": "BiocParallel::MulticoreParam(12)",
            "r_direct_timing_commit": next(iter(direct_commits)),
            "r_direct_endpoint_contract": (
                "one cold direct observation in a fresh process for each worker count; "
                "DESeqDataSet construction + DESeq() + results() + lfcShrink(); "
                "SerialParam or MulticoreParam(12); BLAS and OpenMP pinned to one"
            ),
            "input_file_sha256": input_hashes,
        },
        "gpu_timings": timings,
        "oom_probes": probes,
        "r_endpoints": r_endpoints,
        "r_direct_endpoints": r_direct_endpoints,
    }


def render_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Real-GTEx scaling and A100 memory boundary",
        "",
        "The gene axis is fixed at 54,922. Direct GPU times are medians of five "
        "complete observations after one warm-up; stage sums add independently "
        "measured stage medians, and peak allocation is the maximum over the five "
        "staged repetitions. R endpoint rows are single cold observations and "
        "are not pooled with the main five-repetition headline benchmark.",
        "",
        "## Full-workflow GPU timing",
        "",
        "| case | samples | P | direct (s) | stage sum (s) | peak allocated (GiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in TIMING_CASES:
        record = artifact["gpu_timings"][case]
        lines.append(
            f"| {case} | {record['n_samples']} | {record['P']} | "
            f"{record['direct_median_ms'] / 1000:.3f} | "
            f"{record['stage_total_ms'] / 1000:.3f} | "
            f"{record['peak_allocated_bytes'] / 2**30:.2f} |"
        )
    lines += [
        "",
        "## Practical direct GPU versus 12-worker R endpoints",
        "",
        "The R modes use the same public `DESeqDataSetFromMatrix + DESeq + results "
        "+ lfcShrink` call path. Twelve-worker R is the practical CPU baseline; the "
        "one-worker observations remain in the JSON only as controlled diagnostics.",
        "",
        "| case | samples | P | GPU (s) | R 12 workers (s) | GPU vs R 12w | worker parity |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for case in R_CASES:
        gpu = artifact["gpu_timings"][case]
        endpoint = artifact["r_direct_endpoints"][case]
        lines.append(
            f"| {case} | {gpu['n_samples']} | {gpu['P']} | "
            f"{gpu['direct_median_ms'] / 1000:.3f} | "
            f"{endpoint['twelve_workers']['direct_median_ms'] / 1000:.3f} | "
            f"{endpoint['gpu_speedup_vs_twelve_workers']:.1f}× | "
            f"{'PASS' if endpoint['worker_parity']['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Controlled serial stage diagnostic",
        "",
        "These one-worker stage sums preserve matched boundaries for attribution and "
        "parity. They are intentionally not used as practical acceleration claims.",
        "",
        "| case | samples | P | R 1w stage sum (s) | GPU stage sum (s) | parity |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for case in R_CASES:
        record = artifact["gpu_timings"][case]
        endpoint = artifact["r_endpoints"][case]
        lines.append(
            f"| {case} | {record['n_samples']} | {record['P']} | "
            f"{endpoint['timing']['stage_total_ms'] / 1000:.3f} | "
            f"{record['stage_total_ms'] / 1000:.3f} | "
            f"{'PASS' if endpoint['parity']['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Fresh-process A100 feasibility boundary",
        "",
        "| case | samples | P | path | status | peak allocated (GiB) | failed stage |",
        "|---|---:|---:|---|:---:|---:|---|",
    ]
    for case in PROBE_CASES:
        record = artifact["oom_probes"][case]
        lines.append(
            f"| {case} | {record['n_samples']} | {record['P']} | "
            f"{record['mode_effective_for_dispersion']} | {record['status'].upper()} | "
            f"{record.get('peak_allocated_bytes', record.get('allocated_bytes', 0)) / 2**30:.2f} | "
            f"{record.get('failed_stage', '—')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    artifact = assemble(args.run_directory)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(artifact))
    print(f"wrote {args.output_json}")
    if args.markdown:
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
