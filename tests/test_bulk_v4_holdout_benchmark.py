from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import torch

from gpu_deseq import DESeqDataset, fit_dispersions, fit_size_factors, results, wald_test

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
DEFAULT_MAX_GENES = 5000
MIN_TOTAL_COUNTS = 50
MIN_COUNT = 50
MIN_COUNT_SAMPLES = 2

GPU_DESEQ_ROOT = Path(__file__).resolve().parents[1]
TEXT_TO_RNA_ROOT = GPU_DESEQ_ROOT.parent / "text_to_rna"
PREDICTIONS_PATH = (
    TEXT_TO_RNA_ROOT
    / "debug_outputs/holdout_eval_staging/bio-models/post_training_evals"
    / "z9rramxy/2026_04_14/holdout_eval/bulk_v4_holdout/predictions.parquet"
)
GENE_ORDER_PATH = TEXT_TO_RNA_ROOT / "gene_order.json"


def _load_protein_coding_mask() -> np.ndarray:
    with open(GENE_ORDER_PATH) as handle:
        genes = json.load(handle)["bulk"]
    return np.array([not gene.startswith("ENSG") for gene in genes], dtype=bool)


def _read_predictions() -> pd.DataFrame:
    ds = pa_ds.dataset(str(PREDICTIONS_PATH), format="parquet", partitioning="hive")
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
        raise AssertionError(f"predictions parquet is missing required columns: {missing}")
    return ds.to_table(columns=columns).to_pandas()


def _top_variable_gene_mask_streaming(
    count_vectors: pd.Series,
    max_genes: int,
) -> np.ndarray:
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    n = 0
    for values in count_vectors:
        arr = np.asarray(values, dtype=np.float64)
        if mean is None:
            mean = np.zeros_like(arr, dtype=np.float64)
            m2 = np.zeros_like(arr, dtype=np.float64)
        n += 1
        delta = arr - mean
        mean += delta / n
        delta2 = arr - mean
        m2 += delta * delta2
    if mean is None or m2 is None:
        raise AssertionError("no count vectors were available for variance filtering")
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


def _group_key(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_key_value(row.get(column)) for column in columns)


def _counts_to_int(values: object, *, round_first: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if round_first:
        arr = np.round(arr)
    return arr.astype(np.int64)


def _wrangle_counts_unpaired(df: pd.DataFrame) -> list[tuple[int, str, np.ndarray, list[str]]]:
    group_cols = _available_group_columns(df)
    if not group_cols:
        raise AssertionError("no DE grouping columns were available in the predictions frame")

    ctrl = _control_mask(df)
    perturbed = df.loc[~ctrl]
    controls = df.loc[ctrl]
    context_cols = [column for column in CONTROL_CONTEXT_COLUMNS if column in df.columns]
    if controls.empty or not context_cols:
        raise AssertionError("bulk_v4_holdout benchmark expected unpaired predictions with controls")

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

        observed = pd.concat(
            [
                pd.DataFrame(
                    {
                        "condition": "perturbed",
                        "counts": group["counts"].apply(_counts_to_int),
                    }
                ),
                pd.DataFrame(
                    {
                        "condition": "control",
                        "counts": matched_controls["counts"].apply(_counts_to_int),
                    }
                ),
            ],
            ignore_index=True,
        )
        model = pd.concat(
            [
                pd.DataFrame(
                    {
                        "condition": "perturbed",
                        "counts": group["counts_pred_prior"].apply(
                            lambda values: _counts_to_int(values, round_first=True)
                        ),
                    }
                ),
                pd.DataFrame(
                    {
                        "condition": "control",
                        "counts": matched_controls["counts"].apply(_counts_to_int),
                    }
                ),
            ],
            ignore_index=True,
        )

        tasks.append(
            (
                group_id,
                "observed",
                np.stack(observed["counts"].tolist()).T,
                observed["condition"].tolist(),
            )
        )
        tasks.append(
            (
                group_id,
                "model",
                np.stack(model["counts"].tolist()).T,
                model["condition"].tolist(),
            )
        )

    return tasks


def _run_gpu_deseq_task(
    count_matrix: np.ndarray,
    conditions: list[str],
    device: str,
) -> dict[str, int | float | str]:
    gene_totals = count_matrix.sum(axis=1)
    keep = (gene_totals >= MIN_TOTAL_COUNTS) & (
        (count_matrix >= MIN_COUNT).sum(axis=1) >= MIN_COUNT_SAMPLES
    )
    count_matrix = count_matrix[keep]
    positive_all = np.all(count_matrix > 0, axis=1)
    count_matrix = count_matrix[positive_all]
    if count_matrix.shape[0] == 0:
        return {"tested_genes": 0, "significant_genes": 0, "duration_s": 0.0, "device": device}

    coldata = pd.DataFrame({"condition": conditions})
    dds = DESeqDataset(count_matrix, coldata, design="~ condition", backend="torch").to(device)

    task_start = time.perf_counter()
    try:
        fit_size_factors(dds)
        fit_dispersions(dds)
        fit = wald_test(dds, contrast="condition[T.perturbed]")
        frame = results(fit)
    except Exception as exc:  # noqa: BLE001
        return {
            "tested_genes": 0,
            "significant_genes": 0,
            "duration_s": time.perf_counter() - task_start,
            "device": device,
            "status": "failed",
            "error": type(exc).__name__,
        }
    duration_s = time.perf_counter() - task_start

    return {
        "tested_genes": int(len(frame)),
        "significant_genes": int(frame["significant"].sum()),
        "duration_s": duration_s,
        "device": device,
        "status": "ok",
    }


def benchmark_bulk_v4_holdout() -> dict[str, int | float | str]:
    if not PREDICTIONS_PATH.exists():
        raise AssertionError(f"predictions file was not found: {PREDICTIONS_PATH}")
    if not GENE_ORDER_PATH.exists():
        raise AssertionError(f"gene order file was not found: {GENE_ORDER_PATH}")

    benchmark_start = time.perf_counter()

    read_start = time.perf_counter()
    predictions = _read_predictions()
    read_duration_s = time.perf_counter() - read_start
    print(f"loaded {len(predictions)} prediction rows in {read_duration_s:.2f}s")

    filter_start = time.perf_counter()
    protein_coding_mask = _load_protein_coding_mask()
    predictions = _apply_gene_mask(predictions, protein_coding_mask)
    variance_mask = _top_variable_gene_mask_streaming(predictions["counts"], DEFAULT_MAX_GENES)
    predictions = _apply_gene_mask(predictions, variance_mask)
    filter_duration_s = time.perf_counter() - filter_start
    print(
        "filtered genes "
        f"to {int(protein_coding_mask.sum())} protein-coding and {DEFAULT_MAX_GENES} top-variable "
        f"in {filter_duration_s:.2f}s"
    )

    wrangle_start = time.perf_counter()
    tasks = _wrangle_counts_unpaired(predictions)
    wrangle_duration_s = time.perf_counter() - wrangle_start
    print(f"built {len(tasks)} DE tasks in {wrangle_duration_s:.2f}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_start = time.perf_counter()
    task_summaries = []
    for idx, (_, source, count_matrix, conditions) in enumerate(tasks, start=1):
        task_summaries.append(_run_gpu_deseq_task(count_matrix, conditions, device))
        if idx % 500 == 0 or idx == len(tasks):
            elapsed = time.perf_counter() - run_start
            failed_so_far = sum(summary.get("status") == "failed" for summary in task_summaries)
            print(
                f"completed {idx}/{len(tasks)} tasks on {device} "
                f"in {elapsed:.2f}s (latest source={source}, failures={failed_so_far})"
            )
    run_duration_s = time.perf_counter() - run_start

    total_tested_genes = sum(int(summary["tested_genes"]) for summary in task_summaries)
    total_significant_genes = sum(int(summary["significant_genes"]) for summary in task_summaries)
    nonempty_tasks = sum(int(summary["tested_genes"]) > 0 for summary in task_summaries)
    failed_tasks = sum(summary.get("status") == "failed" for summary in task_summaries)

    result = {
        "run_id": "z9rramxy",
        "holdout": "bulk_v4_holdout",
        "prediction_rows": int(len(predictions)),
        "protein_coding_genes": int(protein_coding_mask.sum()),
        "top_variable_genes": DEFAULT_MAX_GENES,
        "task_count": int(len(tasks)),
        "nonempty_task_count": int(nonempty_tasks),
        "failed_task_count": int(failed_tasks),
        "total_tested_genes": int(total_tested_genes),
        "total_significant_genes": int(total_significant_genes),
        "device": device,
        "read_duration_s": round(read_duration_s, 3),
        "filter_duration_s": round(filter_duration_s, 3),
        "wrangle_duration_s": round(wrangle_duration_s, 3),
        "run_duration_s": round(run_duration_s, 3),
        "total_duration_s": round(time.perf_counter() - benchmark_start, 3),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def test_bulk_v4_holdout_benchmark() -> None:
    summary = benchmark_bulk_v4_holdout()
    assert summary["task_count"] > 0
    assert summary["nonempty_task_count"] > 0
    assert summary["total_tested_genes"] > 0
