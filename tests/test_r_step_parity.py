"""Step-by-step parity tests against R Bioconductor DESeq2.

Every kernel in our pipeline is checked against the matching R intermediate
extracted by scripts/generate_r_fixtures.R. End-to-end p-value agreement can
hide compensating errors at intermediate steps; these tests pin each step
independently.

Tolerances are calibrated from actual measured drift on the fixtures.
Tightening any tolerance is fine; loosening one means something regressed
and should be investigated.

Coverage:
- Single-factor Wald (small/medium/large)
- Multi-factor Wald (~batch + condition)
- Continuous covariate (~x)
- LRT on all fixture sizes and the multi-factor design
- fitType="local" and fitType="mean" trend per-gene values
- Cook's filter applied (cooksCutoff=TRUE in R)
- Independent filtering padj agreement
- apeGLM prior scale + shrunk LFC
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gpu_deseq import (
    DESeqDataset,
    fit_dispersions,
    fit_size_factors,
    lfc_shrink,
    lrt_test,
    results,
    wald_test,
)
from gpu_deseq.api import _wald_se_and_stat
from gpu_deseq._filters import cooks_distance, cooks_outlier_mask, independent_filtering


FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "r_deseq2"


@dataclass(frozen=True)
class FixtureSpec:
    label: str
    design: str
    our_contrast: str   # column name in our design matrix used for the contrast
    r_coef_name: str    # R DESeq2's resultsName for the same contrast


SINGLE_FACTOR = [
    FixtureSpec("small_12x200",  "~ condition", "condition[T.treated]", "condition_treated_vs_control"),
    FixtureSpec("medium_30x500", "~ condition", "condition[T.treated]", "condition_treated_vs_control"),
    FixtureSpec("large_60x2000", "~ condition", "condition[T.treated]", "condition_treated_vs_control"),
]
MULTI_FACTOR = [
    FixtureSpec("multi_60x1500", "~ batch + condition", "condition[T.treated]",
                "condition_treated_vs_control"),
]
CONTINUOUS = [
    FixtureSpec("continuous_60x1000", "~ x", "x", "x"),
]
ALL_FIXTURES = SINGLE_FACTOR + MULTI_FACTOR + CONTINUOUS

LRT_FIXTURES = [
    # (label, full_design, reduced_design)
    ("small_12x200",        "~ condition",         "~ 1"),
    ("medium_30x500",       "~ condition",         "~ 1"),
    ("large_60x2000",       "~ condition",         "~ 1"),
    ("multi_60x1500",       "~ batch + condition", "~ batch"),
    ("continuous_60x1000",  "~ x",                 "~ 1"),
]


def _spec_id(spec: FixtureSpec) -> str:
    return spec.label


def _load_inputs(label: str):
    path = FIXTURE_ROOT / label
    if not (path / "dispersions.csv").exists():
        pytest.skip(f"R intermediates missing for {label}; rerun scripts/generate_r_fixtures.R")
    counts = pd.read_csv(path / "counts.csv", index_col=0)
    coldata = pd.read_csv(path / "coldata.csv", index_col=0)
    return counts, coldata, path


def _build_dds(counts: pd.DataFrame, coldata: pd.DataFrame, design: str) -> DESeqDataset:
    return DESeqDataset(
        counts.to_numpy(dtype=np.float64),
        coldata,
        design=design,
        gene_ids=list(counts.index),
        sample_ids=list(counts.columns),
        backend="torch",
    )


def _run_pipeline_with_dispersions(counts, coldata, design: str,
                                    fit_type: str = "parametric") -> DESeqDataset:
    dds = _build_dds(counts, coldata, design)
    fit_size_factors(dds)
    fit_dispersions(dds, fit_type=fit_type)
    return dds


def _p95(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 95))


# ===========================================================================
# Step 1: size factors  — must match R bit-exact (deterministic algorithm)
# ===========================================================================


@pytest.mark.parametrize("spec", ALL_FIXTURES, ids=_spec_id)
def test_step_size_factors(spec: FixtureSpec) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _build_dds(counts, coldata, spec.design)
    fit_size_factors(dds)

    r_sf = pd.read_csv(path / "size_factors.csv")["size_factor"].to_numpy()
    our_sf = dds.size_factors.cpu().numpy()
    assert our_sf.shape == r_sf.shape
    rel = np.abs(our_sf - r_sf) / np.abs(r_sf)
    assert rel.max() < 1e-10, f"{spec.label}: size factor max rel err {rel.max():.4g}"


# ===========================================================================
# Step 2: gene-wise CR-MLE dispersion (dispGeneEst)
# ===========================================================================


@pytest.mark.parametrize(
    "spec,p95_rel,med_rel",
    [
        # Calibrated 2026-04-29: bit-exact (within FP rounding) parity with
        # R DESeq2 1.30.1 across all fixtures. Achieved by faithful port of
        # DESeq2 src/DESeq2.cpp::fitDisp (gradient ascent + Armijo line
        # search + periodic kappa halving + noIncrease revert + grid fallback).
        # Measured max rel-err: <1e-8 with median ~1e-15 across 5 fixtures.
        (SINGLE_FACTOR[0], 1e-7, 1e-9),
        (SINGLE_FACTOR[1], 1e-7, 1e-9),
        (SINGLE_FACTOR[2], 1e-7, 1e-9),
        (MULTI_FACTOR[0],  1e-7, 1e-9),
        (CONTINUOUS[0],    1e-7, 1e-9),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_disp_gene_est(spec: FixtureSpec, p95_rel: float, med_rel: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)

    r_disp = pd.read_csv(path / "dispersions.csv")
    r_gene = r_disp["dispGeneEst"].to_numpy()
    our_gene = dds.dispersions_gene_wise.cpu().numpy()
    mask = np.isfinite(r_gene) & np.isfinite(our_gene)
    rel = np.abs(our_gene[mask] - r_gene[mask]) / (np.abs(r_gene[mask]) + 1e-12)
    p95 = float(np.percentile(rel, 95))
    med = float(np.median(rel))
    assert p95 < p95_rel, f"{spec.label}: dispGeneEst p95 rel {p95:.4g} >= {p95_rel}"
    assert med < med_rel, f"{spec.label}: dispGeneEst median rel {med:.4g} >= {med_rel}"


# ===========================================================================
# Step 3: parametric trend (per-gene dispFit + recovered (a0, a1) coeffs)
# ===========================================================================


@pytest.mark.parametrize(
    "spec,p95_rel,coef_rel",
    [
        (SINGLE_FACTOR[0], 0.05, 0.10),
        (SINGLE_FACTOR[1], 0.05, 0.05),
        (SINGLE_FACTOR[2], 0.05, 0.05),
        (MULTI_FACTOR[0],  0.05, 0.05),
        (CONTINUOUS[0],    0.05, 0.05),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_disp_trend(spec: FixtureSpec, p95_rel: float, coef_rel: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)

    r_disp = pd.read_csv(path / "dispersions.csv")
    r_fun = pd.read_csv(path / "dispersion_function.csv").iloc[0]
    if r_fun["fit_type"] != "parametric":
        pytest.skip(f"{spec.label}: R fell back to {r_fun['fit_type']} trend")

    r_trend = r_disp["dispFit"].to_numpy()
    our_trend = dds.dispersion_trend.cpu().numpy()
    mask = np.isfinite(r_trend) & np.isfinite(our_trend)
    rel = np.abs(our_trend[mask] - r_trend[mask]) / (np.abs(r_trend[mask]) + 1e-12)
    p95 = _p95(rel)
    assert p95 < p95_rel, f"{spec.label}: dispFit p95 rel {p95:.4g} >= {p95_rel}"

    base_mean_r = r_disp["baseMean"].to_numpy()
    keep = mask & np.isfinite(base_mean_r) & (base_mean_r > 0)
    fitted = our_trend[keep]; base = base_mean_r[keep]
    if base.size >= 2:
        A = np.column_stack([np.ones_like(base), 1.0 / base])
        coef, *_ = np.linalg.lstsq(A, fitted, rcond=None)
        a0_ours, a1_ours = float(coef[0]), float(coef[1])
        a0_r, a1_r = float(r_fun["asymptDisp"]), float(r_fun["extraPois"])
        rel_a0 = abs(a0_ours - a0_r) / (abs(a0_r) + 1e-12)
        rel_a1 = abs(a1_ours - a1_r) / (abs(a1_r) + 1e-12)
        assert rel_a0 < coef_rel, f"{spec.label}: asymptDisp rel {rel_a0:.4g} >= {coef_rel}"
        assert rel_a1 < coef_rel, f"{spec.label}: extraPois  rel {rel_a1:.4g} >= {coef_rel}"


# ===========================================================================
# Step 4: MAP dispersion (post-prior, pre-outlier-rule)
# ===========================================================================


@pytest.mark.parametrize(
    "spec,p95_rel",
    [
        # Bit-exact (FP precision) after porting fitDisp with usePrior=TRUE.
        (SINGLE_FACTOR[0], 1e-5),
        (SINGLE_FACTOR[1], 1e-6),
        (SINGLE_FACTOR[2], 1e-6),
        (MULTI_FACTOR[0],  1e-6),
        (CONTINUOUS[0],    1e-5),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_disp_map(spec: FixtureSpec, p95_rel: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)

    r_disp = pd.read_csv(path / "dispersions.csv")
    r_map = r_disp["dispMAP"].to_numpy()
    our_map = dds.dispersions_map.cpu().numpy()
    mask = np.isfinite(r_map) & np.isfinite(our_map)
    rel = np.abs(our_map[mask] - r_map[mask]) / (np.abs(r_map[mask]) + 1e-12)
    p95 = _p95(rel)
    assert p95 < p95_rel, f"{spec.label}: dispMAP p95 rel {p95:.4g} >= {p95_rel}"


# ===========================================================================
# Step 5: final dispersion + outlier flag
# ===========================================================================


@pytest.mark.parametrize(
    "spec,p95_rel,outlier_jaccard_min",
    [
        # Bit-exact final dispersion = MAP everywhere except outlier-flagged genes.
        (SINGLE_FACTOR[0], 1e-5, 0.95),
        (SINGLE_FACTOR[1], 1e-6, 0.95),
        (SINGLE_FACTOR[2], 1e-6, 0.95),
        (MULTI_FACTOR[0],  1e-6, 0.95),
        (CONTINUOUS[0],    1e-5, 0.95),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_disp_final(spec: FixtureSpec, p95_rel: float,
                         outlier_jaccard_min: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)

    r_disp = pd.read_csv(path / "dispersions.csv")
    r_final = r_disp["dispersion"].to_numpy()
    r_out = r_disp["dispOutlier"].astype(bool).to_numpy()

    our_final = dds.dispersions.cpu().numpy()
    our_mle = dds.dispersions_gene_wise.cpu().numpy()
    our_map = dds.dispersions_map.cpu().numpy()
    our_out = np.isclose(our_final, our_mle) & ~np.isclose(our_final, our_map)

    mask = np.isfinite(r_final) & np.isfinite(our_final)
    rel = np.abs(our_final[mask] - r_final[mask]) / (np.abs(r_final[mask]) + 1e-12)
    p95 = _p95(rel)
    assert p95 < p95_rel, f"{spec.label}: final dispersion p95 rel {p95:.4g} >= {p95_rel}"

    if r_out.any() or our_out.any():
        inter = (r_out & our_out).sum()
        union = (r_out | our_out).sum()
        jacc = inter / union if union > 0 else 1.0
        assert jacc >= outlier_jaccard_min, (
            f"{spec.label}: outlier flag Jaccard {jacc:.3f} < {outlier_jaccard_min}")


# ===========================================================================
# Step 6: β coefficients (natural log == R log2 × ln 2)
# ===========================================================================


def _our_to_r_col(our_name: str, design_columns_r: list[str]) -> str | None:
    """Map our formulaic column name to R's resultsNames column.

    formulaic emits "condition[T.treated]" while R emits
    "condition_treated_vs_control"; for a continuous covariate "x" both agree.
    """
    if our_name == "Intercept":
        return "Intercept"
    if our_name in design_columns_r:
        return our_name
    if our_name.endswith("]") and "[T." in our_name:
        var, level = our_name.split("[T.", 1)
        level = level[:-1]
        # R uses "<var>_<level>_vs_<reference>"; we don't know reference, so try prefix.
        for col in design_columns_r:
            if col.startswith(f"{var}_{level}_vs_"):
                return col
    return None


@pytest.mark.parametrize(
    "spec,p95_abs_natlog",
    [
        # Bit-exact β post-IRLS once dispersions are bit-exact.
        (SINGLE_FACTOR[0], 1e-5),
        (SINGLE_FACTOR[1], 1e-5),
        (SINGLE_FACTOR[2], 1e-5),
        (MULTI_FACTOR[0],  1e-5),
        (CONTINUOUS[0],    1e-5),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_beta(spec: FixtureSpec, p95_abs_natlog: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)
    fit = wald_test(dds, contrast=spec.our_contrast)

    r_beta_log2 = pd.read_csv(path / "coefficients_log2.csv").set_index("gene")
    r_beta_nat = r_beta_log2 * np.log(2.0)

    our_beta = fit.coefficients.cpu().numpy()
    r_cols = list(r_beta_log2.columns)

    # Check every column of our design matrix that maps to an R column.
    for our_col in fit.design_columns:
        r_col = _our_to_r_col(our_col, r_cols)
        if r_col is None:
            continue
        ours = our_beta[:, fit.design_columns.index(our_col)]
        rs = r_beta_nat[r_col].to_numpy()
        m = np.isfinite(ours) & np.isfinite(rs)
        diff = np.abs(ours[m] - rs[m])
        p95 = _p95(diff)
        assert p95 < p95_abs_natlog, (
            f"{spec.label}: β[{our_col} → {r_col}] p95 |Δ natlog| {p95:.4g} "
            f">= {p95_abs_natlog}")


# ===========================================================================
# Step 7: μ matrix
# ===========================================================================


@pytest.mark.parametrize(
    "spec,p95_rel",
    [
        (SINGLE_FACTOR[0], 1e-5),
        (SINGLE_FACTOR[1], 1e-5),
        (SINGLE_FACTOR[2], 1e-5),
        (MULTI_FACTOR[0],  1e-5),
        (CONTINUOUS[0],    1e-5),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_mu(spec: FixtureSpec, p95_rel: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)
    fit = wald_test(dds, contrast=spec.our_contrast)

    r_mu = pd.read_csv(path / "mu.csv", index_col=0).to_numpy()
    our_mu = fit.mu.cpu().numpy()
    mask = np.isfinite(r_mu) & np.isfinite(our_mu) & (r_mu > 0)
    rel = np.abs(our_mu[mask] - r_mu[mask]) / r_mu[mask]
    p95 = _p95(rel)
    assert p95 < p95_rel, f"{spec.label}: μ p95 rel {p95:.4g} >= {p95_rel}"


# ===========================================================================
# Step 8: H (IRLS hat-diagonal)
# ===========================================================================


@pytest.mark.parametrize(
    "spec,p95_rel",
    [
        (SINGLE_FACTOR[0], 1e-5),
        (SINGLE_FACTOR[1], 1e-5),
        (SINGLE_FACTOR[2], 1e-5),
        (MULTI_FACTOR[0],  1e-5),
        (CONTINUOUS[0],    1e-5),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_hat_diag(spec: FixtureSpec, p95_rel: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)
    fit = wald_test(dds, contrast=spec.our_contrast)

    r_H = pd.read_csv(path / "H.csv", index_col=0).to_numpy()
    our_H = fit.hat_diagonals.cpu().numpy()
    mask = np.isfinite(r_H) & np.isfinite(our_H) & (np.abs(r_H) > 1e-6)
    rel = np.abs(our_H[mask] - r_H[mask]) / np.abs(r_H[mask])
    p95 = _p95(rel)
    assert p95 < p95_rel, f"{spec.label}: H p95 rel {p95:.4g} >= {p95_rel}"


# ===========================================================================
# Step 9: Cook's distance values
# ===========================================================================


@pytest.mark.parametrize(
    "spec,p95_rel",
    [
        (SINGLE_FACTOR[0], 1e-4),
        (SINGLE_FACTOR[1], 1e-5),
        (SINGLE_FACTOR[2], 1e-5),
        (MULTI_FACTOR[0],  1e-5),
        (CONTINUOUS[0],    1e-5),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_cooks(spec: FixtureSpec, p95_rel: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)
    fit = wald_test(dds, contrast=spec.our_contrast)

    nz_idx = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1).cpu().numpy()
    counts_sg = fit.counts[nz_idx].T.cpu().numpy()
    normed_sg = fit.normalized_counts[nz_idx].T.cpu().numpy()
    mu_sg = fit.mu[nz_idx].T.cpu().numpy()
    hat_sg = fit.hat_diagonals[nz_idx].T.cpu().numpy()
    our_cooks_sg, _ = cooks_distance(counts_sg, normed_sg, mu_sg, hat_sg, fit.design_df)
    our_cooks = our_cooks_sg.T  # (G_nz, S)

    r_cooks_full = pd.read_csv(path / "cooks.csv", index_col=0).to_numpy()
    r_cooks = r_cooks_full[nz_idx]
    mask = np.isfinite(r_cooks) & np.isfinite(our_cooks) & (r_cooks > 1e-6)
    rel = np.abs(our_cooks[mask] - r_cooks[mask]) / r_cooks[mask]
    p95 = _p95(rel)
    assert p95 < p95_rel, f"{spec.label}: Cook's p95 rel {p95:.4g} >= {p95_rel}"


# ===========================================================================
# Step 10: Wald SE
# ===========================================================================


@pytest.mark.parametrize(
    "spec,p95_rel",
    [
        # Bit-exact with one outlier on small_12x200 (single edge gene, near-0 SE).
        (SINGLE_FACTOR[0], 1e-5),
        (SINGLE_FACTOR[1], 1e-5),
        (SINGLE_FACTOR[2], 1e-5),
        (MULTI_FACTOR[0],  1e-5),
        (CONTINUOUS[0],    1e-5),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_wald_se(spec: FixtureSpec, p95_rel: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)
    fit = wald_test(dds, contrast=spec.our_contrast)

    r_se_log2 = pd.read_csv(path / "coefficient_se_log2.csv").set_index("gene")
    se_col = f"SE_{spec.r_coef_name}"
    r_se_nat = r_se_log2[se_col].to_numpy() * np.log(2.0)

    nz = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1)
    contrast_vec = torch.zeros(len(fit.design_columns), dtype=torch.float64,
                                device=fit.coefficients.device)
    contrast_vec[fit.design_columns.index(spec.our_contrast)] = 1.0
    _stat, our_se_nat_nz = _wald_se_and_stat(
        fit.coefficients[nz], fit.mu[nz], fit.dispersions[nz],
        fit.design_matrix, contrast_vec,
    )
    our_se_full = np.full(fit.coefficients.shape[0], np.nan, dtype=np.float64)
    our_se_full[nz.cpu().numpy()] = our_se_nat_nz.cpu().numpy()

    mask = np.isfinite(r_se_nat) & np.isfinite(our_se_full)
    rel = np.abs(our_se_full[mask] - r_se_nat[mask]) / (np.abs(r_se_nat[mask]) + 1e-12)
    p95 = _p95(rel)
    assert p95 < p95_rel, f"{spec.label}: Wald SE p95 rel {p95:.4g} >= {p95_rel}"


# ===========================================================================
# Step 11: independent filter — per-gene padj agreement
# ===========================================================================


@pytest.mark.parametrize(
    "spec,padj_p95_abs,sig_jaccard_min",
    [
        (SINGLE_FACTOR[0], 0.05,  0.80),
        (SINGLE_FACTOR[1], 0.05,  0.90),
        (SINGLE_FACTOR[2], 0.02,  0.95),
        (MULTI_FACTOR[0],  0.02,  0.95),
        (CONTINUOUS[0],    0.02,  0.95),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_indep_filter_padj(spec: FixtureSpec, padj_p95_abs: float,
                                 sig_jaccard_min: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)
    fit = wald_test(dds, contrast=spec.our_contrast)
    res_df = results(fit, cooks_filter=False, independent_filter=False)

    base_mean = res_df["baseMean"].to_numpy()
    pvalue = res_df["pvalue"].to_numpy()
    our_padj = independent_filtering(pvalue, base_mean, alpha=0.1)
    r_padj = pd.read_csv(path / "padj_filtered.csv").set_index("gene")["padj_filt"].to_numpy()

    our_sig = (our_padj < 0.1) & np.isfinite(our_padj)
    r_sig = (r_padj < 0.1) & np.isfinite(r_padj)
    if our_sig.any() or r_sig.any():
        jacc = (our_sig & r_sig).sum() / max((our_sig | r_sig).sum(), 1)
        assert jacc >= sig_jaccard_min, (
            f"{spec.label}: significance Jaccard {jacc:.3f} < {sig_jaccard_min}")

    both_finite = np.isfinite(our_padj) & np.isfinite(r_padj)
    if both_finite.any():
        diff = np.abs(our_padj[both_finite] - r_padj[both_finite])
        p95 = float(np.percentile(diff, 95))
        assert p95 < padj_p95_abs, (
            f"{spec.label}: padj p95 |Δ| {p95:.4g} >= {padj_p95_abs}")

    our_nan = np.isnan(our_padj) & np.isfinite(pvalue)
    r_keeps_sig = r_sig & np.isfinite(r_padj)
    lost = (our_nan & r_keeps_sig).sum()
    r_sig_n = r_keeps_sig.sum()
    if r_sig_n > 0:
        loss_frac = lost / r_sig_n
        assert loss_frac < 0.02, (
            f"{spec.label}: lost {lost}/{r_sig_n} R-significant genes via filter "
            f"({loss_frac:.3f} >= 0.02)")


# ===========================================================================
# Step 12: Cook's filter applied — pvalue agreement vs R cooksCutoff=TRUE
# ===========================================================================


@pytest.mark.parametrize(
    "spec,sig_jaccard_min",
    [
        (SINGLE_FACTOR[0], 0.85),
        (SINGLE_FACTOR[1], 0.95),
        (SINGLE_FACTOR[2], 0.95),
        (MULTI_FACTOR[0],  0.95),
        (CONTINUOUS[0],    0.95),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_cooks_filter_applied(spec: FixtureSpec, sig_jaccard_min: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)
    fit = wald_test(dds, contrast=spec.our_contrast)

    # Our pipeline: results with cooks_filter=True (sets pvalue=NaN for outliers).
    res_df = results(fit, cooks_filter=True, independent_filter=False)
    r_cf = pd.read_csv(path / "cooks_filtered.csv").set_index("gene")

    # Agreement on which genes have pvalue NaN'd out (Cook's outliers).
    our_nan = np.isnan(res_df["pvalue"].to_numpy())
    r_nan = np.isnan(r_cf["pvalue_cf"].to_numpy())
    if our_nan.any() or r_nan.any():
        jacc = (our_nan & r_nan).sum() / max((our_nan | r_nan).sum(), 1)
        # The outlier rule has discrete cutoffs and pydeseq2's robust dispersion
        # is close-but-not-identical to R's; require >= 0.5 Jaccard.
        assert jacc >= 0.5, (
            f"{spec.label}: Cook's-NaN Jaccard {jacc:.3f} < 0.5 "
            f"(ours={our_nan.sum()}, R={r_nan.sum()})")

    # Significance call agreement after BH on remaining pvalues.
    our_sig = (res_df["padj"].to_numpy() < 0.1) & np.isfinite(res_df["padj"].to_numpy())
    r_sig = (r_cf["padj_cf"].to_numpy() < 0.1) & np.isfinite(r_cf["padj_cf"].to_numpy())
    if our_sig.any() or r_sig.any():
        jacc = (our_sig & r_sig).sum() / max((our_sig | r_sig).sum(), 1)
        assert jacc >= sig_jaccard_min, (
            f"{spec.label}: significance Jaccard after Cook's filter "
            f"{jacc:.3f} < {sig_jaccard_min}")


# ===========================================================================
# Step 13: LRT — covers all fixtures (including multi-factor reduced=~batch)
# ===========================================================================


@pytest.mark.parametrize(
    "label,full_design,reduced_design,stat_p95_rel,p_corr_min",
    [
        ("small_12x200",       "~ condition",         "~ 1",     0.05, 0.999),
        ("medium_30x500",      "~ condition",         "~ 1",     0.05, 0.9999),
        ("large_60x2000",      "~ condition",         "~ 1",     0.05, 0.9999),
        ("multi_60x1500",      "~ batch + condition", "~ batch", 0.05, 0.9999),
        ("continuous_60x1000", "~ x",                 "~ 1",     0.05, 0.9999),
    ],
)
def test_step_lrt(label: str, full_design: str, reduced_design: str,
                  stat_p95_rel: float, p_corr_min: float) -> None:
    counts, coldata, path = _load_inputs(label)
    if not (path / "lrt_results.csv").exists():
        pytest.skip(f"{label}: no LRT fixture")

    dds = _run_pipeline_with_dispersions(counts, coldata, full_design)
    fit = lrt_test(dds, reduced_design=reduced_design)
    res_df = results(fit, cooks_filter=False, independent_filter=False)

    r_lrt = pd.read_csv(path / "lrt_results.csv").set_index("gene")
    merged = res_df.join(r_lrt, rsuffix="_r")
    mask = np.isfinite(merged["stat"]) & np.isfinite(merged["stat_r"])
    m = merged[mask]
    abs_diff = np.abs(m["stat"].to_numpy() - m["stat_r"].to_numpy())
    rel = abs_diff / (np.abs(m["stat_r"].to_numpy()) + 1e-9)
    p95 = float(np.percentile(rel, 95))
    assert p95 < stat_p95_rel, f"{label}: LRT stat p95 rel {p95:.4g} >= {stat_p95_rel}"

    pcorr = np.corrcoef(m["pvalue"], m["pvalue_r"])[0, 1]
    assert pcorr > p_corr_min, f"{label}: LRT pvalue Pearson {pcorr:.6f} <= {p_corr_min}"


# ===========================================================================
# Step 14: fitType="local" — per-gene dispFit
# ===========================================================================


@pytest.mark.parametrize("label", ["small_12x200", "medium_30x500", "large_60x2000"])
def test_step_local_trend(label: str) -> None:
    counts, coldata, path = _load_inputs(label)
    fixture_path = path / "local_dispfit.csv"
    if not fixture_path.exists():
        pytest.skip(f"{label}: no local fitType fixture")

    dds = _run_pipeline_with_dispersions(counts, coldata, "~ condition", fit_type="local")
    r_df = pd.read_csv(fixture_path).set_index("gene")
    r_trend = r_df["dispFit"].to_numpy()
    our_trend = dds.dispersion_trend.cpu().numpy()
    mask = np.isfinite(r_trend) & np.isfinite(our_trend)
    rel = np.abs(our_trend[mask] - r_trend[mask]) / (np.abs(r_trend[mask]) + 1e-12)
    # LOESS implementations differ between statsmodels and R locfit; medium/large
    # match well, small is noisy.
    tol = {"small_12x200": 0.30, "medium_30x500": 0.20, "large_60x2000": 0.15}[label]
    p95 = _p95(rel)
    assert p95 < tol, f"{label}: local dispFit p95 rel {p95:.4g} >= {tol}"


# ===========================================================================
# Step 15: fitType="mean" — per-gene dispFit (constant trimmed-mean)
# ===========================================================================


@pytest.mark.parametrize("label", ["small_12x200", "medium_30x500", "large_60x2000"])
def test_step_mean_trend(label: str) -> None:
    counts, coldata, path = _load_inputs(label)
    fixture_path = path / "mean_dispfit.csv"
    if not fixture_path.exists():
        pytest.skip(f"{label}: no mean fitType fixture")

    dds = _run_pipeline_with_dispersions(counts, coldata, "~ condition", fit_type="mean")
    r_df = pd.read_csv(fixture_path).set_index("gene")
    r_trend = r_df["dispFit"].to_numpy()
    our_trend = dds.dispersion_trend.cpu().numpy()
    mask = np.isfinite(r_trend) & np.isfinite(our_trend)
    # mean trend is a single constant; absolute agreement should be tight.
    rel = np.abs(our_trend[mask] - r_trend[mask]) / (np.abs(r_trend[mask]) + 1e-12)
    p95 = _p95(rel)
    assert p95 < 0.05, f"{label}: mean dispFit p95 rel {p95:.4g} >= 0.05"


# ===========================================================================
# Step 16: apeGLM prior scale + shrunk LFC + shrunk SE
# ===========================================================================


@pytest.mark.parametrize(
    "spec,prior_rel_tol,lfc_p95_abs,se_p95_rel",
    [
        # apeglm prior scale comes from EB; small fixture is noisy.
        (SINGLE_FACTOR[0], 0.30, 0.10, 0.05),
        (SINGLE_FACTOR[1], 0.20, 0.01, 0.01),
        (SINGLE_FACTOR[2], 0.10, 0.005, 0.005),
        (MULTI_FACTOR[0],  0.20, 0.02, 0.02),
        (CONTINUOUS[0],    0.20, 0.02, 0.02),
    ],
    ids=lambda s: s.label if isinstance(s, FixtureSpec) else str(s),
)
def test_step_apeglm(spec: FixtureSpec, prior_rel_tol: float,
                     lfc_p95_abs: float, se_p95_rel: float) -> None:
    counts, coldata, path = _load_inputs(spec.label)
    apeglm_pi = pd.read_csv(path / "apeglm_priorinfo.csv")
    if apeglm_pi.empty or pd.isna(apeglm_pi["prior_scale"].iloc[0]):
        pytest.skip(f"{spec.label}: no apeglm prior info")

    dds = _run_pipeline_with_dispersions(counts, coldata, spec.design)
    fit = wald_test(dds, contrast=spec.our_contrast)
    shrunk = lfc_shrink(fit, coeff=spec.our_contrast)

    # 1. Prior scale comparison. R reports prior.scale; apeglm uses scale = sqrt(var).
    # Our lfc_shrink stores prior_scale internally — recover it via the same
    # estimator (fit_prior_var on MLE β / SE_natlog).
    from gpu_deseq import _shrink as _shrink_core
    nz = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1)
    contrast_vec = torch.zeros(len(fit.design_columns), dtype=torch.float64,
                                device=fit.coefficients.device)
    contrast_vec[fit.design_columns.index(spec.our_contrast)] = 1.0
    _stat, se_nat_nz = _wald_se_and_stat(
        fit.coefficients[nz], fit.mu[nz], fit.dispersions[nz],
        fit.design_matrix, contrast_vec,
    )
    mle_beta_nat = fit.coefficients[nz, fit.design_columns.index(spec.our_contrast)].cpu().numpy()
    se_nat = se_nat_nz.cpu().numpy()
    our_prior_var = _shrink_core.fit_prior_var(mle_beta_nat, se_nat)
    our_prior_scale = float(min(np.sqrt(our_prior_var), 1.0))
    r_prior_scale = float(apeglm_pi["prior_scale"].iloc[0])
    rel = abs(our_prior_scale - r_prior_scale) / max(abs(r_prior_scale), 1e-9)
    assert rel < prior_rel_tol, (
        f"{spec.label}: apeglm prior_scale rel err {rel:.4g} (ours={our_prior_scale:.4f}, "
        f"R={r_prior_scale:.4f}) >= {prior_rel_tol}")

    # 2. Shrunk LFC and SE comparison via results table (R apeglm shipped in results.csv).
    our_res = results(shrunk, cooks_filter=False, independent_filter=False)
    r_res = pd.read_csv(path / "results.csv").set_index("gene")

    merged = our_res.join(r_res[["log2FoldChange_apeglm", "lfcSE_apeglm"]])
    finite = np.isfinite(merged[["log2FoldChange", "log2FoldChange_apeglm",
                                  "lfcSE", "lfcSE_apeglm"]]).all(axis=1)
    m = merged[finite]
    if m.empty:
        pytest.skip(f"{spec.label}: no finite apeglm rows")

    lfc_abs = np.abs(m["log2FoldChange"].to_numpy() - m["log2FoldChange_apeglm"].to_numpy())
    p95 = float(np.percentile(lfc_abs, 95))
    assert p95 < lfc_p95_abs, (
        f"{spec.label}: shrunk LFC p95 |abs| {p95:.4g} >= {lfc_p95_abs}")

    se_rel = np.abs(m["lfcSE"].to_numpy() - m["lfcSE_apeglm"].to_numpy()) / \
        (np.abs(m["lfcSE_apeglm"].to_numpy()) + 1e-9)
    p95_se = float(np.percentile(se_rel, 95))
    assert p95_se < se_p95_rel, (
        f"{spec.label}: shrunk SE p95 rel {p95_se:.4g} >= {se_p95_rel}")
