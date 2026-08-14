"""Regression tests for the `minmu` floor in the Wald standard error.

R forms the weights for its coefficient covariance from a mu already thresholded
at `minmu` (fitBeta.cpp), while `irls_batched` deliberately returns mu
*un*-thresholded because Cook's distance needs it raw. `_wald_se_and_stat` must
therefore apply the floor itself. When it did not, any sample with mu < MIN_MU
contributed ~0 weight, XtWX lost that sample, and the SE came out too large: 22%
high on genes where one contrasted group is entirely zero (which cost
`airway_cell` 5 of its 206 DE calls) and ~7% high on near-zero-count genes whose
dispersion is pinned at the ceiling.

The synthetic step-parity fixtures could not catch this: across all five of them
there are only two (gene, sample) cells with mu < 0.5 and the smallest mu is
0.396, so clamping changes nothing measurable. These tests construct the regime
directly instead, and pin both halves of the contract -- SE floors mu, Cook's
distance does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gpu_deseq import DESeqDataset, fit_size_factors, fit_dispersions, wald_test
from gpu_deseq.api import _wald_se_and_stat, _contrast_vector
import gpu_deseq._deseq2_core as _core


N_PER_GROUP = 4
#: Genes 0..2 have a structurally empty "treated" group -- the case that made the
#: missing floor change DE calls on real data. Gene 3 is an ordinary gene whose
#: mu stays well above MIN_MU in every sample and must be untouched by the floor.
EMPTY_GROUP_GENES = [0, 1, 2]
ORDINARY_GENE = 3


def _dataset(device="cpu"):
    rng = np.random.default_rng(0)
    n = 2 * N_PER_GROUP
    counts = rng.poisson(60.0, size=(40, n)).astype(np.float64)
    for g in EMPTY_GROUP_GENES:                 # entirely zero in the second group
        counts[g, :N_PER_GROUP] = [300, 180, 240, 210][: N_PER_GROUP]
        counts[g, N_PER_GROUP:] = 0.0
    coldata = pd.DataFrame(
        {"condition": pd.Categorical(["control"] * N_PER_GROUP + ["treated"] * N_PER_GROUP,
                                     categories=["control", "treated"])},
        index=[f"s{i}" for i in range(n)],
    )
    dds = DESeqDataset(counts, coldata, design="~ condition",
                       gene_ids=[f"g{i}" for i in range(counts.shape[0])],
                       sample_ids=list(coldata.index), backend="torch").to(device)
    fit_size_factors(dds)
    fit_dispersions(dds)
    return dds


def _fit():
    dds = _dataset()
    fit = wald_test(dds, contrast="condition[T.treated]")
    contrast = _contrast_vector(fit, "condition[T.treated]")
    return fit, contrast


def test_empty_group_actually_produces_mu_below_the_floor():
    """Guard the guard: if this regime stopped occurring the tests below would
    pass vacuously."""
    fit, _ = _fit()
    mu = fit.mu.detach().cpu().numpy()
    assert (mu[EMPTY_GROUP_GENES, N_PER_GROUP:] < _core.MIN_MU).all()
    assert (mu[ORDINARY_GENE] > _core.MIN_MU).all()


def test_wald_se_applies_the_minmu_floor():
    """SE must equal the value obtained from an explicitly pre-floored mu."""
    fit, contrast = _fit()
    args = (fit.coefficients, fit.dispersions, fit.design_matrix, contrast)
    _, se = _wald_se_and_stat(args[0], fit.mu, *args[1:])
    _, se_pre = _wald_se_and_stat(args[0], fit.mu.clamp_min(_core.MIN_MU), *args[1:])
    torch.testing.assert_close(se, se_pre, rtol=0, atol=0)


def test_omitting_the_floor_would_inflate_the_se_on_empty_group_genes():
    """The regression this file exists for: an unfloored mu inflates the SE, and
    only on genes that have a sample below the floor. Without this the clamp
    could be deleted and every other test would still pass."""
    fit, contrast = _fit()
    disp = fit.dispersions.unsqueeze(1)
    eye = _core.RIDGE * torch.eye(fit.design_matrix.shape[1], dtype=torch.float64)

    def se_from(mu):                                   # the un-floored formula
        W = mu / (1.0 + mu * disp)
        M = torch.einsum("sp,gs,sq->gpq", fit.design_matrix, W, fit.design_matrix)
        Hc = torch.einsum("gpq,q->gp", torch.linalg.inv(M + eye), contrast)
        return torch.sqrt(torch.einsum("gp,gpq,gq->g", Hc, M, Hc).clamp_min(1e-30))

    _, se = _wald_se_and_stat(fit.coefficients, fit.mu, fit.dispersions,
                              fit.design_matrix, contrast)
    se_unfloored = se_from(fit.mu)
    ratio = (se_unfloored / se).detach().cpu().numpy()

    assert (ratio[EMPTY_GROUP_GENES] > 1.10).all(), ratio[EMPTY_GROUP_GENES]
    assert ratio[ORDINARY_GENE] == pytest.approx(1.0, abs=1e-12)


def test_cooks_distance_still_receives_unfloored_mu():
    """The other half of the contract: `irls_batched` must keep returning raw mu,
    because R computes Cook's distance from the unthresholded fitted values. If
    the floor were pushed down into the fit instead of applied locally in the SE,
    this fails."""
    fit, _ = _fit()
    mu = fit.mu[EMPTY_GROUP_GENES, N_PER_GROUP:]
    assert (mu < _core.MIN_MU).all(), "mu was floored inside the GLM fit"
    assert (mu > 0).all()


def test_called_set_is_unchanged_by_the_floor_on_ordinary_genes():
    """The floor must not perturb genes that never approach it -- the reason this
    is safe to apply unconditionally."""
    dds = _dataset()
    fit = wald_test(dds, contrast="condition[T.treated]")
    contrast = _contrast_vector(fit, "condition[T.treated]")
    _, se = _wald_se_and_stat(fit.coefficients, fit.mu, fit.dispersions,
                             fit.design_matrix, contrast)
    ordinary = [g for g in range(dds.n_genes)
                if g not in EMPTY_GROUP_GENES
                and bool((fit.mu[g] >= _core.MIN_MU).all())]
    assert len(ordinary) > 30, "expected most synthetic genes to sit above the floor"
    _, se_pre = _wald_se_and_stat(fit.coefficients, fit.mu.clamp_min(_core.MIN_MU),
                                 fit.dispersions, fit.design_matrix, contrast)
    torch.testing.assert_close(se[ordinary], se_pre[ordinary], rtol=0, atol=0)
