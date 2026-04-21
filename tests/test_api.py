from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gpu_deseq import DESeqDataset, fit_dispersions, fit_size_factors, lrt_test, results, wald_test


def _synthetic_dataset() -> tuple[np.ndarray, pd.DataFrame]:
    counts = np.array(
        [
            [100, 120, 90, 350, 390, 410],
            [85, 78, 82, 88, 79, 91],
            [250, 240, 260, 120, 110, 130],
            [35, 40, 38, 36, 41, 39],
            [500, 520, 510, 530, 540, 535],
            [60, 62, 59, 160, 155, 170],
        ],
        dtype=float,
    )
    coldata = pd.DataFrame(
        {
            "batch": ["a", "a", "b", "a", "a", "b"],
            "condition": ["control", "control", "control", "treated", "treated", "treated"],
        }
    )
    return counts, coldata


def test_size_factor_estimation_centers_geometric_mean() -> None:
    counts = np.array(
        [
            [100, 200, 400],
            [50, 100, 200],
            [25, 50, 100],
        ],
        dtype=float,
    )
    coldata = pd.DataFrame({"condition": ["a", "a", "b"]})
    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")
    fit_size_factors(dds)
    expected = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(dds.size_factors.cpu().numpy(), expected, rtol=1e-6)


def test_formula_parsing_and_dispersion_fit() -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition", backend="torch")
    assert dds.design_columns == ["Intercept", "batch[T.b]", "condition[T.treated]"]
    fit_size_factors(dds)
    fit_dispersions(dds)
    assert torch.all(dds.dispersions_gene_wise > 0)
    assert torch.all(dds.dispersions > 0)


def test_wald_results_identify_directional_changes() -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition", backend="torch")
    fit_size_factors(dds)
    fit_dispersions(dds)
    wald = wald_test(dds, contrast="condition[T.treated]")
    res = results(wald)
    assert list(res.columns) == ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj", "significant"]
    assert res.loc["gene_0", "log2FoldChange"] > 0
    assert res.loc["gene_2", "log2FoldChange"] < 0
    assert np.all(np.diff(np.sort(res["padj"].to_numpy())) >= 0)


def test_numeric_contrast_matches_named_contrast() -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition", backend="torch")
    fit_size_factors(dds)
    fit_dispersions(dds)
    named = results(wald_test(dds, contrast="condition[T.treated]"))
    numeric = results(wald_test(dds, contrast=[0.0, 0.0, 1.0]))
    np.testing.assert_allclose(named["log2FoldChange"], numeric["log2FoldChange"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(named["pvalue"], numeric["pvalue"], rtol=1e-6, atol=1e-6)


def test_lrt_produces_valid_statistics() -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition", backend="torch")
    fit_size_factors(dds)
    fit_dispersions(dds)
    lrt = lrt_test(dds, reduced_design="~ batch")
    res = results(lrt)
    assert np.isfinite(res["stat"]).all()
    assert ((res["pvalue"] >= 0) & (res["pvalue"] <= 1)).all()
    assert ((res["padj"] >= 0) & (res["padj"] <= 1)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_gpu_consistency() -> None:
    counts, coldata = _synthetic_dataset()
    cpu = DESeqDataset(counts, coldata, design="~ batch + condition", backend="torch").to("cpu")
    fit_size_factors(cpu)
    fit_dispersions(cpu)
    cpu_res = results(wald_test(cpu, contrast="condition[T.treated]"))

    gpu = DESeqDataset(counts, coldata, design="~ batch + condition", backend="torch").to("cuda")
    fit_size_factors(gpu)
    fit_dispersions(gpu)
    gpu_res = results(wald_test(gpu, contrast="condition[T.treated]"))

    np.testing.assert_allclose(cpu_res["log2FoldChange"], gpu_res["log2FoldChange"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cpu_res["pvalue"], gpu_res["pvalue"], rtol=1e-4, atol=1e-4)


def test_reference_fixture_contract() -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition", backend="torch")
    fit_size_factors(dds)
    fit_dispersions(dds)
    res = results(wald_test(dds, contrast="condition[T.treated]")).sort_values("pvalue")
    top_two = list(res.index[:2])
    assert "gene_0" in top_two
    assert "gene_2" in top_two or "gene_5" in top_two
