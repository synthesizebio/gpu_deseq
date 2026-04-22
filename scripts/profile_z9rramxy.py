"""Profile gpu_deseq end-to-end on the z9rramxy bulk_v4_holdout predictions.

Reads predictions.parquet from GCS (caching locally on first run), wrangles DE
tasks the same way text_to_rna's DEAnalysisStep does for the unpaired bulk_v4
holdout, then runs gpu_deseq on each task and reports per-phase timings.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gpu_deseq import (  # noqa: E402
    DESeqDataset,
    fit_dispersions,
    fit_size_factors,
    results,
    wald_test,
)

DEFAULT_RUN_ID = "z9rramxy"
DEFAULT_DATE = "2026_04_14"
DEFAULT_HOLDOUT = "bulk_v4_holdout"
DEFAULT_BUCKET = "bio-models"
DEFAULT_GCS_ACCOUNT = "ai-model-uploader@text-to-rna.iam.gserviceaccount.com"
DEFAULT_GENE_ORDER = Path("/home/max_synthesize_bio/text_to_rna/gene_order.json")
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "gpu_deseq_profile"
DEFAULT_MAX_GENES = 5000

COMPARISON_GROUP_COLUMNS = (
    "study",
    "de_context_id",
    "de_perturbation_id",
    "bioactive_entity_dose_value",
    "bioactive_entity_time_value_hours",
    "gene_perturbation_type",
    "cell_line_ontology_id",
)
CONTROL_CONTEXT_COLUMNS = ("study", "de_context_id")
MAX_CONTROLS_PER_COMPARISON = 8
MIN_REPLICATES = 2
MIN_TOTAL_COUNTS = 50
MIN_COUNT = 50
MIN_COUNT_SAMPLES = 2


def _gcs_uri(run_id: str, date: str, holdout: str, bucket: str) -> str:
    return f"gs://{bucket}/post_training_evals/{run_id}/{date}/holdout_eval/{holdout}/predictions.parquet"


def _cache_path(run_id: str, date: str, holdout: str, cache_root: Path) -> Path:
    return cache_root / run_id / date / holdout / "predictions.parquet"


def _ensure_predictions(
    gcs_uri: str,
    cache_path: Path,
    gcloud_account: str | None,
) -> float:
    if cache_path.exists():
        return 0.0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gcloud_account:
        subprocess.run(
            ["gcloud", "config", "set", "account", gcloud_account],
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    start = time.perf_counter()
    subprocess.run(
        ["gcloud", "storage", "cp", gcs_uri, str(cache_path)],
        check=True,
        env=env,
    )
    return time.perf_counter() - start


def _load_protein_coding_mask(gene_order_path: Path) -> np.ndarray:
    with open(gene_order_path) as handle:
        genes = json.load(handle)["bulk"]
    return np.array([not gene.startswith("ENSG") for gene in genes], dtype=bool)


def _read_predictions(parquet_path: Path) -> pd.DataFrame:
    ds = pa_ds.dataset(str(parquet_path), format="parquet", partitioning="hive")
    available = set(ds.schema.names)
    columns = [
        "counts",
        "counts_pred_prior",
        "sample_classification",
        "study",
        "de_context_id",
        "de_perturbation_id",
        "bioactive_entity_dose_value",
        "bioactive_entity_time_value_hours",
        "gene_perturbation_type",
        "cell_line_ontology_id",
    ]
    missing = [column for column in columns if column not in available]
    if missing:
        raise RuntimeError(f"predictions parquet is missing required columns: {missing}")
    return ds.to_table(columns=columns).to_pandas()


def _top_variable_gene_mask_streaming(
    count_vectors: pd.Series,
    max_genes: int,
) -> np.ndarray:
    mean_vec: np.ndarray | None = None
    m2: np.ndarray | None = None
    n = 0
    for values in count_vectors:
        arr = np.asarray(values, dtype=np.float64)
        if mean_vec is None:
            mean_vec = np.zeros_like(arr, dtype=np.float64)
            m2 = np.zeros_like(arr, dtype=np.float64)
        n += 1
        delta = arr - mean_vec
        mean_vec += delta / n
        delta2 = arr - mean_vec
        m2 += delta * delta2
    if mean_vec is None or m2 is None:
        raise RuntimeError("no count vectors available for variance filtering")
    variance = m2 / max(n, 1)
    top_idx = np.argsort(variance)[-max_genes:]
    mask = np.zeros_like(variance, dtype=bool)
    mask[top_idx] = True
    return mask


def _apply_gene_mask(predictions: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    trimmed = predictions.copy()
    for column in ("counts", "counts_pred_prior"):
        trimmed[column] = trimmed[column].apply(
            lambda values: np.asarray(values, dtype=np.float64)[mask]
        )
    return trimmed


def _normalize_key_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text in {"nan", "None", "NaN", "<NA>"}:
        return ""
    return text


def _available_group_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in COMPARISON_GROUP_COLUMNS if column in df.columns]


def _control_mask(df: pd.DataFrame) -> pd.Series:
    return df["sample_classification"].astype(str).str.lower().eq("control")


def _counts_to_int(values: object, *, round_first: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if round_first:
        arr = np.round(arr)
    return arr.astype(np.int64)


def _wrangle_counts_unpaired(df: pd.DataFrame) -> list[tuple[int, str, np.ndarray, list[str]]]:
    group_cols = _available_group_columns(df)
    if not group_cols:
        raise RuntimeError("no DE grouping columns were available in the predictions frame")

    ctrl = _control_mask(df)
    perturbed = df.loc[~ctrl]
    controls = df.loc[ctrl]
    context_cols = [column for column in CONTROL_CONTEXT_COLUMNS if column in df.columns]
    if controls.empty or not context_cols:
        raise RuntimeError("bulk_v4_holdout predictions missing controls or context columns")

    control_groups = {
        key: group.head(MAX_CONTROLS_PER_COMPARISON)
        for key, group in controls.groupby(context_cols, dropna=False)
    }
    tasks: list[tuple[int, str, np.ndarray, list[str]]] = []

    for group_id, (_, group) in enumerate(perturbed.groupby(group_cols, dropna=False), start=1):
        ctx_key = tuple(group.iloc[0].get(column, "") for column in context_cols)
        matched_controls = control_groups.get(ctx_key)
        if matched_controls is None:
            continue
        if len(group) < MIN_REPLICATES or len(matched_controls) < MIN_REPLICATES:
            continue

        observed_counts = np.stack(
            group["counts"].apply(_counts_to_int).tolist()
            + matched_controls["counts"].apply(_counts_to_int).tolist()
        ).T
        observed_conditions = ["perturbed"] * len(group) + ["control"] * len(matched_controls)
        tasks.append((group_id, "observed", observed_counts, observed_conditions))

        model_counts = np.stack(
            group["counts_pred_prior"].apply(lambda v: _counts_to_int(v, round_first=True)).tolist()
            + matched_controls["counts"].apply(_counts_to_int).tolist()
        ).T
        tasks.append((group_id, "model", model_counts, observed_conditions))

    return tasks


def _profile_task(
    count_matrix: np.ndarray,
    conditions: list[str],
    device: str,
) -> dict[str, object]:
    gene_totals = count_matrix.sum(axis=1)
    keep = (gene_totals >= MIN_TOTAL_COUNTS) & (
        (count_matrix >= MIN_COUNT).sum(axis=1) >= MIN_COUNT_SAMPLES
    )
    count_matrix = count_matrix[keep]
    positive_all = np.all(count_matrix > 0, axis=1)
    count_matrix = count_matrix[positive_all]
    n_genes_fit = int(count_matrix.shape[0])
    n_samples = int(count_matrix.shape[1]) if count_matrix.size else 0

    base = {
        "n_genes_fit": n_genes_fit,
        "n_samples": n_samples,
        "size_factors_s": 0.0,
        "dispersions_s": 0.0,
        "wald_s": 0.0,
        "results_s": 0.0,
        "total_s": 0.0,
        "significant_genes": 0,
        "device": device,
        "status": "skipped_no_genes",
    }
    if n_genes_fit == 0:
        return base

    coldata = pd.DataFrame({"condition": conditions})
    task_start = time.perf_counter()
    try:
        dds = DESeqDataset(count_matrix, coldata, design="~ condition", backend="torch").to(device)
        t0 = time.perf_counter()
        fit_size_factors(dds)
        size_factors_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        fit_dispersions(dds)
        dispersions_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        fit = wald_test(dds, contrast="condition[T.perturbed]")
        wald_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        frame = results(fit)
        results_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "total_s": time.perf_counter() - task_start,
            "status": "failed",
            "error": type(exc).__name__,
        }
    total_s = time.perf_counter() - task_start

    return {
        **base,
        "size_factors_s": size_factors_s,
        "dispersions_s": dispersions_s,
        "wald_s": wald_s,
        "results_s": results_s,
        "total_s": total_s,
        "significant_genes": int(frame["significant"].sum()),
        "status": "ok",
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def _phase_stats(task_summaries: list[dict[str, object]], key: str) -> dict[str, float]:
    vals = [float(s[key]) for s in task_summaries if s.get("status") == "ok"]
    if not vals:
        return {"count": 0, "total_s": 0.0, "mean_s": 0.0, "p50_s": 0.0, "p90_s": 0.0, "p99_s": 0.0, "max_s": 0.0}
    return {
        "count": len(vals),
        "total_s": float(sum(vals)),
        "mean_s": float(mean(vals)),
        "p50_s": float(median(vals)),
        "p90_s": _percentile(vals, 90),
        "p99_s": _percentile(vals, 99),
        "max_s": float(max(vals)),
    }


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def _device_label(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--gcloud-account", default=DEFAULT_GCS_ACCOUNT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--gene-order-json", type=Path, default=DEFAULT_GENE_ORDER)
    parser.add_argument("--max-genes", type=int, default=DEFAULT_MAX_GENES)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    gcs_uri = _gcs_uri(args.run_id, args.date, args.holdout, args.bucket)
    cache_path = _cache_path(args.run_id, args.date, args.holdout, args.cache_root)
    print(f"source: {gcs_uri}")
    print(f"cache:  {cache_path}")
    print(f"gene_order_json: {args.gene_order_json}")

    benchmark_start = time.perf_counter()
    download_s = _ensure_predictions(gcs_uri, cache_path, args.gcloud_account)
    print(f"download_s: {download_s:.2f}")

    read_start = time.perf_counter()
    predictions = _read_predictions(cache_path)
    read_s = time.perf_counter() - read_start
    print(f"loaded {len(predictions)} prediction rows in {read_s:.2f}s")

    filter_start = time.perf_counter()
    protein_coding_mask = _load_protein_coding_mask(args.gene_order_json)
    predictions = _apply_gene_mask(predictions, protein_coding_mask)
    variance_mask = _top_variable_gene_mask_streaming(predictions["counts"], args.max_genes)
    predictions = _apply_gene_mask(predictions, variance_mask)
    filter_s = time.perf_counter() - filter_start
    print(
        f"filtered to {int(protein_coding_mask.sum())} protein-coding "
        f"and {args.max_genes} top-variable in {filter_s:.2f}s"
    )

    wrangle_start = time.perf_counter()
    tasks = _wrangle_counts_unpaired(predictions)
    wrangle_s = time.perf_counter() - wrangle_start
    print(f"built {len(tasks)} DE tasks in {wrangle_s:.2f}s")

    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
        print(f"capped tasks to {len(tasks)} (--max-tasks)")

    device = _resolve_device(args.device)
    print(f"device: {_device_label(device)}")

    run_start = time.perf_counter()
    task_summaries: list[dict[str, object]] = []
    for idx, (_, source, count_matrix, conditions) in enumerate(tasks, start=1):
        summary = _profile_task(count_matrix, conditions, device)
        summary["source"] = source
        task_summaries.append(summary)
        if idx % 500 == 0 or idx == len(tasks):
            elapsed = time.perf_counter() - run_start
            failed = sum(s.get("status") == "failed" for s in task_summaries)
            print(
                f"completed {idx}/{len(tasks)} tasks on {device} in {elapsed:.2f}s "
                f"(latest source={source}, failures={failed})",
                flush=True,
            )
    run_s = time.perf_counter() - run_start

    ok_count = sum(s.get("status") == "ok" for s in task_summaries)
    failed_count = sum(s.get("status") == "failed" for s in task_summaries)
    skipped_count = sum(s.get("status") == "skipped_no_genes" for s in task_summaries)
    total_sig = sum(int(s.get("significant_genes", 0)) for s in task_summaries)

    report: dict[str, object] = {
        "run_id": args.run_id,
        "date": args.date,
        "holdout": args.holdout,
        "gcs_uri": gcs_uri,
        "cache_path": str(cache_path),
        "device": _device_label(device),
        "max_genes": args.max_genes,
        "protein_coding_genes": int(protein_coding_mask.sum()),
        "prediction_rows": int(len(predictions)),
        "task_count": len(task_summaries),
        "ok_task_count": ok_count,
        "failed_task_count": failed_count,
        "skipped_task_count": skipped_count,
        "total_significant_genes": total_sig,
        "download_s": round(download_s, 3),
        "read_s": round(read_s, 3),
        "filter_s": round(filter_s, 3),
        "wrangle_s": round(wrangle_s, 3),
        "run_s": round(run_s, 3),
        "total_s": round(time.perf_counter() - benchmark_start, 3),
        "phase_stats": {
            "size_factors": _phase_stats(task_summaries, "size_factors_s"),
            "dispersions": _phase_stats(task_summaries, "dispersions_s"),
            "wald": _phase_stats(task_summaries, "wald_s"),
            "results": _phase_stats(task_summaries, "results_s"),
            "total_per_task": _phase_stats(task_summaries, "total_s"),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
