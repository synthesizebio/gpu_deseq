"""Validate gpu_deseq against pydeseq2 on a sample of z9rramxy bulk_v4_holdout tasks.

Samples N DE comparisons from the same predictions.parquet the profile script
uses, runs both engines on the *exact same* filtered count matrix for each
task, and reports agreement metrics (sign, Pearson/Spearman on LFC, max
absolute LFC diff, significance-set Jaccard at alpha=0.1).

Both engines run with cooks_filter and independent_filter enabled (pydeseq2's
defaults) — the drop-in scenario.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import torch
from scipy.stats import pearsonr, spearmanr

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


# ---------- Data loading & wrangling (duplicated from profile_z9rramxy.py) ----------


def _gcs_uri(run_id: str, date: str, holdout: str, bucket: str) -> str:
    return f"gs://{bucket}/post_training_evals/{run_id}/{date}/holdout_eval/{holdout}/predictions.parquet"


def _cache_path(run_id: str, date: str, holdout: str, cache_root: Path) -> Path:
    return cache_root / run_id / date / holdout / "predictions.parquet"


def _ensure_predictions(gcs_uri: str, cache_path: Path, gcloud_account: str | None) -> None:
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gcloud_account:
        subprocess.run(
            ["gcloud", "config", "set", "account", gcloud_account],
            check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    subprocess.run(["gcloud", "storage", "cp", gcs_uri, str(cache_path)], check=True, env=env)


def _load_protein_coding_mask(gene_order_path: Path) -> np.ndarray:
    with open(gene_order_path) as handle:
        genes = json.load(handle)["bulk"]
    return np.array([not gene.startswith("ENSG") for gene in genes], dtype=bool)


def _read_predictions(parquet_path: Path) -> pd.DataFrame:
    ds = pa_ds.dataset(str(parquet_path), format="parquet", partitioning="hive")
    columns = [
        "counts", "counts_pred_prior", "sample_classification",
        "study", "de_context_id", "de_perturbation_id",
        "bioactive_entity_dose_value", "bioactive_entity_time_value_hours",
        "gene_perturbation_type", "cell_line_ontology_id",
    ]
    return ds.to_table(columns=columns).to_pandas()


def _top_variable_gene_mask_streaming(count_vectors: pd.Series, max_genes: int) -> np.ndarray:
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


def _available_group_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in COMPARISON_GROUP_COLUMNS if c in df.columns]


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
    context_cols = [c for c in CONTROL_CONTEXT_COLUMNS if c in df.columns]
    if controls.empty or not context_cols:
        raise RuntimeError("predictions missing controls or context columns")

    control_groups = {
        key: group.head(MAX_CONTROLS_PER_COMPARISON)
        for key, group in controls.groupby(context_cols, dropna=False)
    }
    tasks: list[tuple[int, str, np.ndarray, list[str]]] = []
    for group_id, (_, group) in enumerate(perturbed.groupby(group_cols, dropna=False), start=1):
        ctx_key = tuple(group.iloc[0].get(column, "") for column in context_cols)
        matched = control_groups.get(ctx_key)
        if matched is None:
            continue
        if len(group) < MIN_REPLICATES or len(matched) < MIN_REPLICATES:
            continue
        conditions = ["perturbed"] * len(group) + ["control"] * len(matched)
        observed = np.stack(
            group["counts"].apply(_counts_to_int).tolist()
            + matched["counts"].apply(_counts_to_int).tolist()
        ).T
        model = np.stack(
            group["counts_pred_prior"].apply(lambda v: _counts_to_int(v, round_first=True)).tolist()
            + matched["counts"].apply(_counts_to_int).tolist()
        ).T
        tasks.append((group_id, "observed", observed, conditions))
        tasks.append((group_id, "model", model, conditions))
    return tasks


# ---------- Per-task filter + DE engines ----------


def _filter_task_matrix(count_matrix: np.ndarray) -> np.ndarray:
    gene_totals = count_matrix.sum(axis=1)
    keep = (gene_totals >= MIN_TOTAL_COUNTS) & (
        (count_matrix >= MIN_COUNT).sum(axis=1) >= MIN_COUNT_SAMPLES
    )
    count_matrix = count_matrix[keep]
    positive_all = np.all(count_matrix > 0, axis=1)
    return count_matrix[positive_all]


def _run_gpu_deseq(count_matrix: np.ndarray, conditions: list[str], device: str) -> pd.DataFrame:
    coldata = pd.DataFrame({"condition": conditions})
    dds = DESeqDataset(count_matrix, coldata, design="~ condition", backend="torch").to(device)
    fit_size_factors(dds)
    fit_dispersions(dds)
    fit = wald_test(dds, contrast="condition[T.perturbed]")
    return results(fit)[["log2FoldChange", "pvalue", "padj"]]


def _run_pydeseq2(count_matrix: np.ndarray, conditions: list[str]) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    n_genes, n_samples = count_matrix.shape
    counts_df = pd.DataFrame(
        count_matrix.T.astype(np.int64),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"sample_{i}" for i in range(n_samples)],
    )
    metadata = pd.DataFrame({"condition": conditions}, index=counts_df.index)
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        dds = DeseqDataSet(
            counts=counts_df, metadata=metadata, design="~condition",
            quiet=True, n_cpus=1, low_memory=True,
        )
        dds.deseq2()
        stats = DeseqStats(
            dds, contrast=["condition", "perturbed", "control"],
            # match pydeseq2 production defaults
            quiet=True, n_cpus=1,
        )
        stats.summary()
    df = stats.results_df[["log2FoldChange", "pvalue", "padj"]].copy()
    df.index = [f"gene_{i}" for i in range(len(df))]
    return df


# ---------- Comparison metrics ----------


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xs, ys = x[mask], y[mask]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(xs, ys).statistic)
    return float(spearmanr(xs, ys).statistic)


def _compare_task(gpu: pd.DataFrame, ref: pd.DataFrame, alpha: float = 0.1) -> dict[str, float | int]:
    joined = gpu.join(ref, rsuffix="_ref", how="inner")
    lfc = joined["log2FoldChange"].to_numpy()
    lfc_ref = joined["log2FoldChange_ref"].to_numpy()
    pval = joined["pvalue"].to_numpy()
    pval_ref = joined["pvalue_ref"].to_numpy()
    padj = joined["padj"].to_numpy()
    padj_ref = joined["padj_ref"].to_numpy()

    finite = np.isfinite(lfc) & np.isfinite(lfc_ref)
    lfc_f, lfc_ref_f = lfc[finite], lfc_ref[finite]
    diff = np.abs(lfc_f - lfc_ref_f)

    sign_match = (np.sign(lfc_f) == np.sign(lfc_ref_f)).sum()
    strong = np.abs(lfc_ref_f) > 1.0
    sign_match_strong = int((np.sign(lfc_f[strong]) == np.sign(lfc_ref_f[strong])).sum())

    sig_gpu = set(joined.index[(padj <= alpha) & np.isfinite(padj)])
    sig_ref = set(joined.index[(padj_ref <= alpha) & np.isfinite(padj_ref)])

    return {
        "n_genes": int(len(joined)),
        "n_finite_lfc": int(finite.sum()),
        "lfc_pearson": _safe_corr(lfc_f, lfc_ref_f, "pearson"),
        "lfc_spearman": _safe_corr(lfc_f, lfc_ref_f, "spearman"),
        "lfc_max_abs_diff": float(diff.max()) if diff.size else float("nan"),
        "lfc_median_abs_diff": float(np.median(diff)) if diff.size else float("nan"),
        "lfc_p95_abs_diff": float(np.percentile(diff, 95)) if diff.size else float("nan"),
        "sign_agreement": int(sign_match),
        "sign_agreement_frac": float(sign_match / len(lfc_f)) if len(lfc_f) else float("nan"),
        "sign_agreement_strong": sign_match_strong,
        "sign_agreement_strong_total": int(strong.sum()),
        "pvalue_spearman": _safe_corr(pval, pval_ref, "spearman"),
        "neg_log10_padj_pearson": _safe_corr(
            -np.log10(np.clip(padj, 1e-300, 1.0)),
            -np.log10(np.clip(padj_ref, 1e-300, 1.0)),
            "pearson",
        ),
        "sig_jaccard": _jaccard(sig_gpu, sig_ref),
        "sig_gpu_count": int(len(sig_gpu)),
        "sig_ref_count": int(len(sig_ref)),
    }


def _aggregate(metrics: list[dict[str, float | int]], key: str) -> dict[str, float]:
    vals = [float(m[key]) for m in metrics if np.isfinite(float(m[key]))]
    if not vals:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p10": float("nan"), "min": float("nan")}
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "p50": float(median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "min": float(min(vals)),
    }


# ---------- CLI ----------


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


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
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=["observed", "model", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    gcs_uri = _gcs_uri(args.run_id, args.date, args.holdout, args.bucket)
    cache_path = _cache_path(args.run_id, args.date, args.holdout, args.cache_root)
    print(f"source:   {gcs_uri}", flush=True)
    print(f"cache:    {cache_path}", flush=True)
    _ensure_predictions(gcs_uri, cache_path, args.gcloud_account)

    t0 = time.perf_counter()
    predictions = _read_predictions(cache_path)
    print(f"read {len(predictions)} rows in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    pc_mask = _load_protein_coding_mask(args.gene_order_json)
    predictions = _apply_gene_mask(predictions, pc_mask)
    var_mask = _top_variable_gene_mask_streaming(predictions["counts"], args.max_genes)
    predictions = _apply_gene_mask(predictions, var_mask)
    print(f"filtered to {args.max_genes} top-variable genes in {time.perf_counter() - t0:.1f}s", flush=True)

    tasks = _wrangle_counts_unpaired(predictions)
    if args.source != "both":
        tasks = [t for t in tasks if t[1] == args.source]
    print(f"available tasks: {len(tasks)}", flush=True)

    rng = random.Random(args.seed)
    sample = rng.sample(tasks, min(args.n_tasks, len(tasks)))
    print(f"sampled {len(sample)} tasks (seed={args.seed})", flush=True)

    device = _resolve_device(args.device)
    print(f"device: {device}", flush=True)

    per_task: list[dict[str, object]] = []
    for idx, (group_id, source, count_matrix, conditions) in enumerate(sample, start=1):
        filtered = _filter_task_matrix(count_matrix)
        if filtered.shape[0] == 0:
            per_task.append({"group_id": group_id, "source": source, "status": "skipped_no_genes"})
            continue
        try:
            t_gpu = time.perf_counter()
            gpu_df = _run_gpu_deseq(filtered, conditions, device)
            gpu_s = time.perf_counter() - t_gpu

            t_ref = time.perf_counter()
            ref_df = _run_pydeseq2(filtered, conditions)
            ref_s = time.perf_counter() - t_ref

            metrics = _compare_task(gpu_df, ref_df)
            per_task.append({
                "group_id": group_id, "source": source, "status": "ok",
                "n_samples": int(filtered.shape[1]),
                "gpu_s": round(gpu_s, 3), "pydeseq2_s": round(ref_s, 3),
                **metrics,
            })
            print(
                f"[{idx}/{len(sample)}] group={group_id} {source:8s} "
                f"genes={metrics['n_genes']:>5} samples={filtered.shape[1]:>3} "
                f"pearson={metrics['lfc_pearson']:.4f} spearman={metrics['lfc_spearman']:.4f} "
                f"sign_frac={metrics['sign_agreement_frac']:.4f} "
                f"sig_jacc={metrics['sig_jaccard']:.3f} "
                f"max|dLFC|={metrics['lfc_max_abs_diff']:.3f} "
                f"gpu={gpu_s:.2f}s ref={ref_s:.2f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            per_task.append({
                "group_id": group_id, "source": source, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{idx}/{len(sample)}] group={group_id} {source} FAILED: {type(exc).__name__}: {exc}", flush=True)

    ok = [m for m in per_task if m.get("status") == "ok"]
    report = {
        "run_id": args.run_id, "date": args.date, "holdout": args.holdout,
        "device": device, "max_genes": args.max_genes,
        "n_tasks_sampled": len(sample), "n_tasks_ok": len(ok),
        "n_tasks_failed": sum(1 for m in per_task if m.get("status") == "failed"),
        "aggregate": {
            "lfc_pearson": _aggregate(ok, "lfc_pearson"),
            "lfc_spearman": _aggregate(ok, "lfc_spearman"),
            "lfc_max_abs_diff": _aggregate(ok, "lfc_max_abs_diff"),
            "lfc_median_abs_diff": _aggregate(ok, "lfc_median_abs_diff"),
            "lfc_p95_abs_diff": _aggregate(ok, "lfc_p95_abs_diff"),
            "sign_agreement_frac": _aggregate(ok, "sign_agreement_frac"),
            "pvalue_spearman": _aggregate(ok, "pvalue_spearman"),
            "neg_log10_padj_pearson": _aggregate(ok, "neg_log10_padj_pearson"),
            "sig_jaccard": _aggregate(ok, "sig_jaccard"),
        },
        "per_task": per_task,
    }
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote report to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
