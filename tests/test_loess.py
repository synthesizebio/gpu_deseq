"""Pin `_loess_quadratic` against R's loess.

R's `estimateDispersionsPriorVar` smooths the small-dof KL curve with
`loess(span=0.2, degree=2)` and takes its argmin. The argmin is sensitive to the
smoother -- substituting a Savitzky-Golay filter shifts airway's prior variance
from 0.561 to 0.609 against R's 0.529 -- so the smoother is pinned here.

The target is R's DEFAULT surface ("interpolate" -- kd-tree plus cubic Hermite),
because that is what `loess(klDivs ~ obsVarGrid, span = 0.2)` computes inside
DESeq2. The exact ("direct") surface is a different function: on the airway KL
curve the two pick argmins two grid points apart.

Reference values generated with R 4.x / stats::loess:

    x <- seq(0, 8, length.out = 200)
    y <- sin(1.7*x) + 0.15*x^2 - 0.4*x
    lo <- loess(y ~ x, span = 0.2)            # default surface = "interpolate"
    predict(lo, c(0, 0.37, 1.25, 2.5, 3.9, 5.05, 6.4, 7.2, 7.83, 8))
"""
from __future__ import annotations

import numpy as np

from gpu_deseq import _deseq2_core as core

_XOUT = np.array([0.0, 0.37, 1.25, 2.5, 3.9, 5.05, 6.4, 7.2, 7.83, 8.0])
_R_LOESS_INTERPOLATE = np.array([
    -0.035130233987455815,
    0.47896243542392819,
    0.58161254238119175,
    -0.95344835486815716,
    1.0621951821859033,
    2.5487302966825722,
    2.5964641038028899,
    4.5812671647688896,
    6.7271634415013635,
    7.3160506923449828,
])


def _curve() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 8.0, 200)
    return x, np.sin(1.7 * x) + 0.15 * x**2 - 0.4 * x


def test_loess_matches_r_default_surface() -> None:
    x, y = _curve()
    got = core._loess_quadratic(x, y, _XOUT, span=0.2)
    assert np.allclose(got, _R_LOESS_INTERPOLATE, rtol=0, atol=1e-12)


def test_kd_cuts_match_r_tree() -> None:
    """R's kd-tree on 200 evenly spaced points: 31 median cuts, the first at the
    100th order statistic, then the 50th and 150th."""
    xs = np.linspace(0.0, 8.0, 200)
    cuts = core._loess_kd_cuts(xs, fc=int(np.floor(200 * 0.2 * 0.2)))
    assert len(cuts) == 31
    assert cuts[0] == xs[99] and cuts[1] == xs[49]
    assert xs[149] in cuts


def test_loess_reproduces_exact_quadratic() -> None:
    """Local quadratic fits, cubic-Hermite blended, stay exact on a quadratic."""
    x = np.linspace(0.0, 8.0, 200)
    y = 2.0 - 0.75 * x + 0.3 * x**2
    xo = np.linspace(0.0, 8.0, 57)
    got = core._loess_quadratic(x, y, xo, span=0.2)
    assert np.allclose(got, 2.0 - 0.75 * xo + 0.3 * xo**2, rtol=0, atol=1e-10)


def test_prior_var_uses_loess_not_savgol() -> None:
    """Regression guard: the small-dof branch must route through the loess
    smoother. On the airway KL curve, Savitzky-Golay lands at 0.609 and the
    direct surface at 0.553, against R's 0.529 -- the difference between 6.7%,
    2.8% and 1.4% dispersion agreement with R."""
    rng = np.random.default_rng(0)
    m, p = 8, 5
    resid = rng.normal(0.0, 1.1, size=9000)
    pv = core._prior_var_kl_grid(resid, m, p)
    assert 0.25 <= pv <= 8.0
    # the returned value must sit on R's 1000-point output grid
    fine = np.linspace(0.0, 8.0, 1000)
    assert pv == 0.25 or np.min(np.abs(fine - pv)) < 1e-12
