#!/usr/bin/env python3
"""Benchmark full cuDESeq2 workflows on nested real GTEx cohorts.

The fixed 54,922-gene matrix is materialized once by
``bench/extract_gtex_scaling.R``. This runner deterministically selects nested
sample cohorts, records the exact source columns, times complete workflows,
tracks peak CUDA memory, and turns CUDA OOM into a structured result.

Run one case per process. This is important for OOM probes because CUDA
allocator state from a preceding case would otherwise change the boundary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "bench" / "cache" / "gtex_scaling" / "matrix"
PAPER_META = ROOT / "validation" / "data" / "gtex_blood_muscle" / "meta.json"
STAGES = (
    "normalization",
    "dispersion",
    "glm_fit",
    "significance",
    "lfc_shrink",
)
MODES = {
    "eager": {},
    "graph": {"use_cuda_graph": True},
    "triton": {"use_triton": True},
}
CASE_NAMES = (
    "p2_300",
    "p2_600",
    "p2_912",
    "p3_fixed300",
    "p4_fixed300",
    "p5_fixed300",
    "p6_fixed300",
    "p6_all",
    "p8_all",
    "p9_all",
    "p10_all",
    "p12_all",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def nvidia_driver_version() -> str | None:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def stable_order(rows: pd.DataFrame, seed: int = 1) -> list[int]:
    """Return zero-based source columns in a platform-stable hash order."""

    def key(row: Any) -> str:
        payload = f"{seed}|{row.tissue}|{row.sample_id}".encode()
        return hashlib.sha256(payload).hexdigest()

    ranked = sorted(rows.itertuples(index=False), key=key)
    return [int(row.source_column) - 1 for row in ranked]


def cohort_orders(samples: pd.DataFrame) -> tuple[list[str], dict[str, list[int]]]:
    """Build nested tissue/sample orders, anchored to the paper cohort."""
    counts = (
        samples.groupby("tissue", dropna=False)
        .size()
        .sort_values(ascending=False, kind="stable")
    )
    first = ["Whole Blood", "Muscle - Skeletal"]
    tissue_order = first + [
        str(tissue) for tissue in counts.index if tissue not in set(first)
    ]
    orders = {
        tissue: stable_order(samples[samples.tissue == tissue])
        for tissue in tissue_order
    }

    # Preserve the exact 150+150 paper cohort as the prefix of every P=2 run.
    paper = json.loads(PAPER_META.read_text())
    paper_columns = [int(value) - 1 for value in paper["selected_source_columns"]]
    tissue_by_column = samples.assign(
        source_column_zero_based=samples["source_column"].astype(int) - 1
    ).set_index("source_column_zero_based")["tissue"]
    for tissue in first:
        selected = [
            column for column in paper_columns if tissue_by_column.loc[column] == tissue
        ]
        selected_set = set(selected)
        orders[tissue] = selected + [
            column for column in orders[tissue] if column not in selected_set
        ]
    return tissue_order, orders


def case_spec(name: str, samples: pd.DataFrame) -> dict[str, Any]:
    tissues, orders = cohort_orders(samples)
    if name.startswith("p2_") and name.removeprefix("p2_").isdigit():
        total = int(name.removeprefix("p2_"))
        if total % 2:
            raise ValueError("balanced P=2 case must have an even total")
        k, per_group, use_all = 2, total // 2, False
    elif name.startswith("p") and name.endswith("_fixed300"):
        k = int(name[1:].split("_", 1)[0])
        per_group, use_all = 300, False
    elif name.startswith("p") and name.endswith("_all"):
        k = int(name[1:].split("_", 1)[0])
        per_group, use_all = None, True
    else:
        raise ValueError(f"unknown case {name!r}")
    if not 2 <= k <= len(tissues):
        raise ValueError(f"invalid tissue count {k}")

    selected_tissues = tissues[:k]
    columns: list[int] = []
    labels: list[str] = []
    group_counts: dict[str, int] = {}
    for tissue in selected_tissues:
        tissue_columns = orders[tissue]
        take = len(tissue_columns) if use_all else min(int(per_group), len(tissue_columns))
        chosen = tissue_columns[:take]
        columns.extend(chosen)
        labels.extend([tissue] * len(chosen))
        group_counts[tissue] = len(chosen)
    return {
        "name": name,
        "P": k,
        "n_samples": len(columns),
        "tissues": selected_tissues,
        "group_counts": group_counts,
        "source_columns_zero_based": columns,
        "labels": labels,
    }


def _matrix_paths(matrix_dir: Path) -> dict[str, Path]:
    return {
        "metadata": matrix_dir / "matrix_metadata.json",
        "samples": matrix_dir / "samples.csv",
        "genes": matrix_dir / "genes.txt",
        "binary": matrix_dir / "counts_int32_f.bin",
    }


def validate_matrix(matrix_dir: Path, *, full_hash: bool) -> dict[str, Any]:
    paths = _matrix_paths(matrix_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing GTEx matrix files: " + ", ".join(missing))
    metadata = json.loads(paths["metadata"].read_text())
    expected_bytes = int(metadata["n_genes"]) * int(metadata["n_samples"]) * 4
    if paths["binary"].stat().st_size != expected_bytes:
        raise ValueError("GTEx matrix byte length does not match metadata")
    if full_hash and digest(paths["binary"]) != metadata["binary_sha256"]:
        raise ValueError("GTEx matrix SHA-256 does not match metadata")
    return metadata


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def _run_workflow(
    counts: np.ndarray,
    coldata: pd.DataFrame,
    gene_ids: list[str],
    sample_ids: list[str],
    contrast: str,
    mode: str,
    *,
    timed_stages: bool,
    capture_arrays: bool = False,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "src"))
    from gpu_deseq import (
        DESeqDataset,
        fit_dispersions,
        fit_size_factors,
        lfc_shrink,
        results,
        wald_test,
    )
    from gpu_deseq.api import _replace_outliers_and_refit_wald

    mode_args = MODES[mode]
    stage_ms: dict[str, float] = {}
    stage_memory: dict[str, dict[str, int]] = {}

    def measure(name: str, function):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            value = function()
        except torch.OutOfMemoryError as error:
            error.add_note(f"cuDESeq2 GTEx stage: {name}")
            raise
        torch.cuda.synchronize()
        if timed_stages:
            stage_ms[name] = (time.perf_counter() - start) * 1000
            stage_memory[name] = _memory()
        return value

    try:
        dds = DESeqDataset(
            counts,
            coldata,
            design="~ tissue",
            gene_ids=gene_ids,
            sample_ids=sample_ids,
        ).to("cuda")
    except torch.OutOfMemoryError as error:
        error.add_note("cuDESeq2 GTEx stage: build")
        raise
    measure("normalization", lambda: fit_size_factors(dds))
    measure("dispersion", lambda: fit_dispersions(dds, **mode_args))

    def fit_wald():
        fit = wald_test(dds, contrast=contrast)
        return _replace_outliers_and_refit_wald(dds, fit, **mode_args)

    fit = measure("glm_fit", fit_wald)
    result = measure("significance", lambda: results(fit))
    shrunk = measure(
        "lfc_shrink",
        lambda: results(
            lfc_shrink(fit, coeff=contrast),
            cooks_filter=False,
            independent_filter=False,
        ),
    )
    summary = {
        "called_padj_005": int((result["padj"] < 0.05).fillna(False).sum()),
        "finite_shrunken_lfc": int(np.isfinite(shrunk["log2FoldChange"]).sum()),
    }
    if capture_arrays:
        summary["_arrays"] = {
            "size_factor": dds.size_factors.detach().cpu().numpy(),
            "dispersion": dds.dispersions.detach().cpu().numpy(),
            "base_mean": result["baseMean"].to_numpy(),
            "log2_fold_change": result["log2FoldChange"].to_numpy(),
            "lfc_se": result["lfcSE"].to_numpy(),
            "stat": result["stat"].to_numpy(),
            "pvalue": result["pvalue"].to_numpy(),
            "padj": result["padj"].to_numpy(),
            "shrunk_lfc": shrunk["log2FoldChange"].to_numpy(),
            "shrunk_lfc_se": shrunk["lfcSE"].to_numpy(),
        }
    if timed_stages:
        summary["stage_ms"] = stage_ms
        summary["stage_memory"] = stage_memory
    return summary


def run_case(
    name: str,
    mode: str,
    matrix_dir: Path,
    *,
    warmups: int,
    reps: int,
    direct_reps: int,
    full_hash: bool,
    capture_path: Path | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    metadata = validate_matrix(matrix_dir, full_hash=full_hash)
    paths = _matrix_paths(matrix_dir)
    samples = pd.read_csv(paths["samples"])
    genes = paths["genes"].read_text().splitlines()
    spec = case_spec(name, samples)
    columns = np.asarray(spec.pop("source_columns_zero_based"), dtype=np.int64)
    labels = spec.pop("labels")
    matrix = np.memmap(
        paths["binary"],
        mode="r",
        dtype="<i4",
        shape=(metadata["n_genes"], metadata["n_samples"]),
        order="F",
    )
    # Disk materialization is not part of the algorithm timing. Dataset
    # construction and host-to-device transfer remain inside direct timing.
    counts = np.asarray(matrix[:, columns], dtype=np.float64, order="C")
    samples_by_column = samples.set_index(samples["source_column"].astype(int) - 1)
    sample_ids = [str(samples_by_column.loc[column].sample_id) for column in columns]
    categories = list(spec["tissues"])
    coldata = pd.DataFrame(
        {"tissue": pd.Categorical(labels, categories=categories)},
        index=sample_ids,
    )
    contrast = f"tissue[T.{categories[1]}]"
    commit, dirty = git_state()
    record: dict[str, Any] = {
        **spec,
        "mode_requested": mode,
        "mode_effective_for_dispersion": mode if spec["P"] <= 6 else "eager_fallback",
        "contrast": contrast,
        "source_columns_one_based": [int(column) + 1 for column in columns],
        "source_sample_ids_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode()
        ).hexdigest(),
        "git_commit": commit,
        "working_tree_dirty_at_start": dirty,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "nvidia_driver_version": nvidia_driver_version(),
        "matrix_metadata_sha256": digest(paths["metadata"]),
        "matrix_binary_sha256": metadata["binary_sha256"],
        "warmups": warmups,
        "reps": reps,
        "direct_reps": direct_reps,
        "status": "running",
    }

    started = time.perf_counter()
    current_stage = "warmup"
    try:
        for _ in range(warmups):
            _release_cuda()
            _run_workflow(
                counts, coldata, genes, sample_ids, contrast, mode, timed_stages=False
            )

        stage_values = {stage: [] for stage in STAGES}
        memory_values: list[dict[str, Any]] = []
        last_summary: dict[str, Any] = {}
        current_stage = "staged_workflow"
        for _ in range(reps):
            _release_cuda()
            torch.cuda.reset_peak_memory_stats()
            last_summary = _run_workflow(
                counts,
                coldata,
                genes,
                sample_ids,
                contrast,
                mode,
                timed_stages=True,
                capture_arrays=capture_path is not None and len(memory_values) == reps - 1,
            )
            for stage in STAGES:
                stage_values[stage].append(last_summary["stage_ms"][stage])
            memory_values.append(last_summary["stage_memory"])

        arrays = last_summary.pop("_arrays", None)
        if arrays is not None and capture_path is not None:
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(capture_path, gene_id=np.asarray(genes), **arrays)
            record["capture_file"] = capture_path.name
            record["capture_sha256"] = digest(capture_path)

        direct_values: list[float] = []
        current_stage = "direct_workflow"
        for _ in range(direct_reps):
            _release_cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            direct_start = time.perf_counter()
            direct_summary = _run_workflow(
                counts, coldata, genes, sample_ids, contrast, mode, timed_stages=False
            )
            torch.cuda.synchronize()
            direct_values.append((time.perf_counter() - direct_start) * 1000)
            last_summary = direct_summary

        medians = {
            stage: float(np.median(values)) for stage, values in stage_values.items()
        }
        record.update(last_summary)
        record["stage_values_ms"] = stage_values
        record["stage_medians_ms"] = medians
        record["stage_total_ms"] = sum(medians.values())
        record["direct_values_ms"] = direct_values
        record["direct_median_ms"] = (
            float(np.median(direct_values)) if direct_values else None
        )
        record["memory_by_repetition"] = memory_values
        record["peak_allocated_bytes"] = max(
            memory[stage]["peak_allocated_bytes"]
            for memory in memory_values
            for stage in memory
        )
        record["peak_reserved_bytes"] = max(
            memory[stage]["peak_reserved_bytes"]
            for memory in memory_values
            for stage in memory
        )
        record["status"] = "pass"
    except (torch.OutOfMemoryError, RuntimeError) as error:
        is_oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
        if not is_oom:
            raise
        record["status"] = "oom"
        notes = getattr(error, "__notes__", [])
        stage_notes = [note for note in notes if note.startswith("cuDESeq2 GTEx stage: ")]
        record["failed_stage"] = (
            stage_notes[-1].split(": ", 1)[1] if stage_notes else current_stage
        )
        record["error"] = str(error)
        try:
            record.update(_memory())
        except Exception:
            pass
    finally:
        record["elapsed_s"] = time.perf_counter() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", choices=CASE_NAMES)
    parser.add_argument("--mode", choices=tuple(MODES), default="triton")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--direct-reps", type=int, default=5)
    parser.add_argument("--verify-matrix", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples_path = _matrix_paths(args.matrix_dir)["samples"]
    if args.list:
        samples = pd.read_csv(samples_path)
        for name in CASE_NAMES:
            spec = case_spec(name, samples)
            spec.pop("labels")
            spec.pop("source_columns_zero_based")
            print(json.dumps(spec))
        return
    if not args.case:
        parser.error("provide --list or --case CASE")
    record = run_case(
        args.case,
        args.mode,
        args.matrix_dir,
        warmups=args.warmups,
        reps=args.reps,
        direct_reps=args.direct_reps,
        full_hash=args.verify_matrix,
        capture_path=args.capture,
    )
    payload = json.dumps(record, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
