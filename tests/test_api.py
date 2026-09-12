from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gpu_deseq import (
    DESeqDataset,
    deseq,
    fit_dispersions,
    fit_size_factors,
    lfc_shrink,
    lrt_test,
    results,
    wald_test,
)
from gpu_deseq._filters import (
    cooks_distance,
    cooks_distance_tensor,
    cooks_outlier_mask,
    robust_method_of_moments_disp,
    robust_method_of_moments_disp_tensor,
)
from gpu_deseq.api import _cooks_outlier_mask_tensor


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


def test_default_size_factors_require_explicit_poscounts_for_sparse_data() -> None:
    counts = np.array(
        [
            [0, 4, 16],
            [1, 0, 9],
            [8, 2, 0],
        ],
        dtype=float,
    )
    coldata = pd.DataFrame({"condition": ["a", "a", "b"]})
    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")

    with pytest.raises(ValueError, match="use method='poscounts'"):
        fit_size_factors(dds)


def test_poscounts_matches_deseq2_modified_geometric_means() -> None:
    # The all-one row has modified geometric mean one and must remain eligible.
    counts = np.array(
        [
            [0, 4, 16],
            [1, 1, 1],
            [8, 2, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    coldata = pd.DataFrame({"condition": ["a", "a", "b"]})
    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")
    fit_size_factors(dds, method="poscounts")

    positive = counts > 0
    log_geomeans = np.where(positive, np.log(np.clip(counts, 1, None)), 0).mean(axis=1)
    eligible = positive.any(axis=1)
    expected = []
    for sample in range(counts.shape[1]):
        mask = eligible & positive[:, sample]
        expected.append(np.exp(np.median(np.log(counts[mask, sample]) - log_geomeans[mask])))
    expected = np.asarray(expected)
    expected /= np.exp(np.log(expected).mean())

    np.testing.assert_allclose(dds.size_factors.cpu().numpy(), expected, rtol=1e-12)


def test_size_factor_method_is_validated() -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")
    with pytest.raises(ValueError, match="method must be"):
        fit_size_factors(dds, method="automatic")


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


def test_wald_handles_extreme_count_ranges_without_non_finite_results() -> None:
    counts = np.array(
        [
            [1, 3, 2, 500_000, 450_000, 520_000],
            [25, 30, 28, 40, 42, 39],
            [2_000, 1_900, 2_100, 4, 3, 5],
            [800, 900, 1_000, 1_100, 1_200, 1_300],
            [60_000, 70_000, 65_000, 62_000, 72_000, 69_000],
        ],
        dtype=float,
    )
    coldata = pd.DataFrame({"condition": ["control", "control", "control", "treated", "treated", "treated"]})
    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")
    fit_size_factors(dds)
    fit_dispersions(dds)
    res = results(wald_test(dds, contrast="condition[T.treated]"))

    assert np.isfinite(res["log2FoldChange"]).all()
    assert np.isfinite(res["lfcSE"]).all()
    assert np.isfinite(res["stat"]).all()
    assert np.isfinite(res["pvalue"]).all()
    assert np.isfinite(res["padj"]).all()


def test_deseq_replaces_and_refits_eligible_count_outlier() -> None:
    rng = np.random.default_rng(3)
    counts = rng.negative_binomial(
        20, 20 / (20 + 100), size=(100, 14)
    ).astype(float)
    counts[0, 0] = 100_000
    coldata = pd.DataFrame({"condition": ["control"] * 7 + ["treated"] * 7})

    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")
    fit = deseq(dds, contrast="condition[T.treated]")
    res = results(fit)

    assert fit.replaced_genes is not None and bool(fit.replaced_genes[0])
    assert fit.replaceable_samples is not None and bool(torch.all(fit.replaceable_samples))
    assert fit.replacement_counts is not None
    assert fit.replacement_counts[0, 0] < counts[0, 0]
    assert np.isfinite(res.loc["gene_0", "pvalue"])


def test_deseq_can_disable_count_outlier_replacement() -> None:
    rng = np.random.default_rng(3)
    counts = rng.negative_binomial(
        20, 20 / (20 + 100), size=(100, 14)
    ).astype(float)
    counts[0, 0] = 100_000
    coldata = pd.DataFrame({"condition": ["control"] * 7 + ["treated"] * 7})

    dds = DESeqDataset(counts, coldata, design="~ condition", backend="torch")
    fit = deseq(
        dds,
        contrast="condition[T.treated]",
        min_replicates_for_replace=None,
    )

    assert fit.replaced_genes is None
    assert fit.replacement_counts is None


@pytest.mark.parametrize(
    "bad_value,error",
    [
        (np.nan, "finite"),
        (np.inf, "finite"),
        (1.5, "integer"),
    ],
)
def test_dataset_rejects_non_finite_and_fractional_counts(
    bad_value: float, error: str
) -> None:
    counts = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    counts[0, 0] = bad_value
    coldata = pd.DataFrame({"condition": ["a", "a", "b", "b"]})

    with pytest.raises(ValueError, match=error):
        DESeqDataset(counts, coldata, design="~ condition")


def test_dataset_preserves_and_validates_dataframe_sample_labels() -> None:
    counts = pd.DataFrame(
        [[10, 11, 20, 21], [30, 31, 40, 41]],
        index=["gene_a", "gene_b"],
        columns=["sample_1", "sample_2", "sample_3", "sample_4"],
    )
    aligned = pd.DataFrame(
        {"condition": ["a", "a", "b", "b"]}, index=counts.columns
    )
    dds = DESeqDataset(counts, aligned, design="~ condition")
    assert dds.sample_ids == list(counts.columns)
    assert dds.gene_ids == list(counts.index)
    assert list(dds.coldata.index) == list(counts.columns)

    reversed_coldata = aligned.iloc[::-1]
    with pytest.raises(ValueError, match="same order"):
        DESeqDataset(counts, reversed_coldata, design="~ condition")


def test_dataset_rejects_rank_deficient_design() -> None:
    counts = np.arange(1, 161, dtype=float).reshape(20, 8)
    coldata = pd.DataFrame(
        {
            "condition": ["a"] * 4 + ["b"] * 4,
            "batch": ["x"] * 4 + ["y"] * 4,
        }
    )

    with pytest.raises(ValueError, match="not full rank"):
        DESeqDataset(counts, coldata, design="~ condition + batch")


def test_lrt_rejects_non_nested_reduced_design() -> None:
    counts, coldata = _synthetic_dataset()
    coldata = coldata.assign(x=np.linspace(-1.0, 1.0, len(coldata)))
    dds = DESeqDataset(counts, coldata, design="~ batch + condition").to("cpu")
    fit_size_factors(dds)
    fit_dispersions(dds)

    with pytest.raises(ValueError, match="nested"):
        lrt_test(dds, reduced_design="~ x")


def test_lrt_results_apply_independent_filtering_and_report_full_model_lfc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition").to("cpu")
    fit_size_factors(dds)
    fit_dispersions(dds)
    fit = lrt_test(dds, reduced_design="~ batch")

    sentinel = np.linspace(0.01, 0.06, dds.n_genes)
    calls: list[tuple[np.ndarray, np.ndarray, float]] = []

    def fake_filter(pvalue: np.ndarray, base_mean: np.ndarray, alpha: float) -> np.ndarray:
        calls.append((pvalue, base_mean, alpha))
        return sentinel

    monkeypatch.setattr("gpu_deseq._filters.independent_filtering", fake_filter)
    frame = results(fit, cooks_filter=False, independent_filter=True, alpha=0.05)

    assert len(calls) == 1
    np.testing.assert_allclose(frame["padj"], sentinel)
    assert np.isfinite(frame["log2FoldChange"]).all()
    assert np.isfinite(frame["lfcSE"]).all()


def test_apeglm_preserves_mle_inference_and_rejects_other_contrasts() -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition").to("cpu")
    fit_size_factors(dds)
    fit_dispersions(dds)
    fit = wald_test(dds, contrast="condition[T.treated]")
    raw = results(fit, cooks_filter=False, independent_filter=False)
    shrunk_fit = lfc_shrink(fit, coeff="condition[T.treated]")
    shrunk = results(shrunk_fit, cooks_filter=False, independent_filter=False)

    np.testing.assert_allclose(shrunk["stat"], raw["stat"])
    np.testing.assert_allclose(shrunk["pvalue"], raw["pvalue"])
    np.testing.assert_allclose(shrunk["padj"], raw["padj"])
    assert not np.allclose(shrunk["log2FoldChange"], raw["log2FoldChange"])

    with pytest.raises(ValueError, match="coefficient that was shrunk"):
        results(shrunk_fit, contrast="batch[T.b]")

    # A merely close vector is still a different contrast: accepting it would
    # rescale the reported LFC while leaving the stored shrunk SE unchanged.
    with pytest.raises(ValueError, match="coefficient that was shrunk"):
        results(shrunk_fit, contrast=[0.0, 0.0, 1.0 + 1e-8])


def test_lfc_shrink_reuses_preserved_size_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition").to("cpu")
    fit_size_factors(dds)
    fit_dispersions(dds)
    fit = wald_test(dds, contrast="condition[T.treated]")

    assert fit.size_factors is dds.size_factors

    def unexpected_reconstruction(*args: object, **kwargs: object) -> None:
        raise AssertionError("size factors should not be reconstructed")

    monkeypatch.setattr(torch, "nanmedian", unexpected_reconstruction)
    shrunk = lfc_shrink(fit, coeff="condition[T.treated]")

    assert shrunk.size_factors is dds.size_factors
    assert torch.isfinite(shrunk.coefficients[fit.non_zero_mask]).all()


def test_wald_inference_is_reused_by_shrinkage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts, coldata = _synthetic_dataset()
    dds = DESeqDataset(counts, coldata, design="~ batch + condition").to("cpu")
    fit_size_factors(dds)
    fit_dispersions(dds)
    fit = wald_test(dds, contrast="condition[T.treated]")

    from gpu_deseq import api as api_module

    original = api_module._wald_se_and_stat
    calls = 0

    def counted(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api_module, "_wald_se_and_stat", counted)
    raw = results(fit, cooks_filter=False, independent_filter=False)
    shrunk_fit = lfc_shrink(fit, coeff="condition[T.treated]")
    shrunk = results(shrunk_fit, cooks_filter=False, independent_filter=False)

    assert calls == 1
    np.testing.assert_allclose(shrunk["stat"], raw["stat"])
    np.testing.assert_allclose(shrunk["pvalue"], raw["pvalue"])


@pytest.mark.parametrize("apply_low_count_heuristic", [False, True])
def test_tensor_cooks_outlier_mask_matches_numpy(
    apply_low_count_heuristic: bool,
) -> None:
    design = pd.DataFrame(
        {
            "Intercept": np.ones(8),
            "condition[T.b]": [0.0] * 4 + [1.0] * 4,
        }
    )
    cooks = np.array(
        [
            [100.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 100.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1] * 8,
        ]
    )
    counts = np.array(
        [
            [1.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0],
            [12.0, 10.0, 11.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [5.0] * 8,
        ]
    )

    expected = cooks_outlier_mask(
        cooks.T,
        counts.T,
        design,
        num_vars=2,
        apply_low_count_heuristic=apply_low_count_heuristic,
    )
    actual = _cooks_outlier_mask_tensor(
        torch.as_tensor(cooks),
        torch.as_tensor(counts),
        design,
        num_vars=2,
        apply_low_count_heuristic=apply_low_count_heuristic,
    )

    np.testing.assert_array_equal(actual.numpy(), expected)


def test_combined_apeglm_loss_gradient_matches_separate_evaluations() -> None:
    from gpu_deseq import _shrink

    generator = torch.Generator().manual_seed(7)
    beta = torch.randn(11, 3, generator=generator, dtype=torch.float64)
    counts = torch.randint(
        0, 500, (11, 8), generator=generator, dtype=torch.int64
    ).to(torch.float64)
    size = torch.rand(11, generator=generator, dtype=torch.float64) * 20.0 + 0.1
    offset = torch.randn(8, generator=generator, dtype=torch.float64) * 0.2
    design = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    args = (beta, counts, size, offset, design, 15.0, 0.7, 2)

    expected_loss = _shrink._nbinom_apeglm_loss(*args)
    expected_grad = _shrink._nbinom_apeglm_grad(*args)
    actual_loss, actual_grad = _shrink._nbinom_apeglm_loss_grad(*args)

    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-14, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-14, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_apeglm_kernels_match_eager_cuda() -> None:
    """The CUDA fast path may change reduction order, but not the solution."""
    from gpu_deseq import _shrink

    generator = torch.Generator(device="cuda").manual_seed(23)
    beta = (
        torch.randn(
            127, 4, generator=generator, dtype=torch.float64, device="cuda"
        )
        * 0.1
    )
    counts = torch.randint(
        0, 500, (127, 31), generator=generator, dtype=torch.int64, device="cuda"
    ).to(torch.float64)
    size = (
        torch.rand(127, generator=generator, dtype=torch.float64, device="cuda")
        * 20.0
        + 0.1
    )
    offset = (
        torch.randn(31, generator=generator, dtype=torch.float64, device="cuda")
        * 0.2
    )
    design = torch.randn(
        31, 4, generator=generator, dtype=torch.float64, device="cuda"
    )
    scales = (
        torch.scalar_tensor(15.0, dtype=torch.float64, device="cuda"),
        torch.scalar_tensor(0.7, dtype=torch.float64, device="cuda"),
    )
    args = (beta, counts, size, offset, design, *scales, 3)
    eager_loss, eager_grad = _shrink._nbinom_apeglm_loss_grad(*args)
    compiled_loss, compiled_grad = _shrink._compiled_nbinom_apeglm_loss_grad(*args)
    eager_hess = _shrink._nbinom_apeglm_hess(*args)
    compiled_hess = _shrink._compiled_nbinom_apeglm_hess(*args)

    torch.testing.assert_close(compiled_loss, eager_loss, rtol=2e-13, atol=2e-10)
    torch.testing.assert_close(compiled_grad, eager_grad, rtol=2e-12, atol=2e-10)
    torch.testing.assert_close(compiled_hess, eager_hess, rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("replicated_cells", [False, True])
def test_tensor_cooks_distance_matches_numpy(replicated_cells: bool) -> None:
    generator = np.random.default_rng(19)
    n_samples = 8
    n_genes = 23
    if replicated_cells:
        design = pd.DataFrame(
            {
                "Intercept": np.ones(n_samples),
                "condition[T.b]": [0.0] * 4 + [1.0] * 4,
            }
        )
    else:
        # Every row is its own cell, exercising the plain trimmed-variance path.
        design = pd.DataFrame(
            {
                "Intercept": np.ones(n_samples),
                "sample": np.arange(n_samples, dtype=float),
            }
        )
    counts = generator.integers(1, 500, size=(n_genes, n_samples)).astype(float)
    size_factors = np.exp(generator.normal(0.0, 0.2, size=n_samples))
    normed = counts / size_factors[None, :]
    mu = np.maximum(counts * generator.lognormal(0.0, 0.1, counts.shape), 0.5)
    hat = generator.uniform(0.01, 0.25, size=counts.shape)

    expected_alpha = robust_method_of_moments_disp(normed.T, design)
    expected_cooks, _ = cooks_distance(
        counts.T, normed.T, mu.T, hat.T, design
    )
    actual_cooks, actual_alpha = cooks_distance_tensor(
        torch.as_tensor(counts),
        torch.as_tensor(normed),
        torch.as_tensor(mu),
        torch.as_tensor(hat),
        design,
    )
    direct_alpha = robust_method_of_moments_disp_tensor(
        torch.as_tensor(normed), design
    )

    np.testing.assert_allclose(actual_alpha.numpy(), expected_alpha, rtol=1e-13)
    np.testing.assert_allclose(direct_alpha.numpy(), expected_alpha, rtol=1e-13)
    np.testing.assert_allclose(actual_cooks.numpy().T, expected_cooks, rtol=1e-13)


def test_cooks_low_count_heuristic_is_explicitly_scoped() -> None:
    design = pd.DataFrame({"Intercept": np.ones(6), "x": np.arange(6)})
    cooks = np.zeros((6, 1))
    cooks[0, 0] = 1_000.0
    counts = np.array([[1.0], [10.0], [11.0], [12.0], [0.0], [0.0]])

    unscoped = cooks_outlier_mask(
        cooks, counts, design, num_vars=2, apply_low_count_heuristic=False
    )
    binary_factor_only = cooks_outlier_mask(
        cooks, counts, design, num_vars=2, apply_low_count_heuristic=True
    )

    assert bool(unscoped[0])
    assert not bool(binary_factor_only[0])

    binary_coldata = pd.DataFrame({"condition": ["a"] * 3 + ["b"] * 3})
    continuous_coldata = pd.DataFrame({"x": np.arange(6, dtype=float)})
    base_counts = np.arange(1, 61, dtype=float).reshape(10, 6)
    assert DESeqDataset(base_counts, binary_coldata, "~ condition").cooks_low_count_heuristic
    assert not DESeqDataset(base_counts, continuous_coldata, "~ x").cooks_low_count_heuristic
