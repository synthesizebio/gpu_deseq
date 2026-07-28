"""Pin the R random-number replay in `_r_rng` against R itself.

DESeq2's small-dof prior-variance estimator is a Monte-Carlo grid search seeded
with `set.seed(2)`. Reproducing its result means reproducing R's stream, so these
tests compare against R 4.x element by element, not distributionally.

References generated with:
    set.seed(2); runif(5); rnorm(3); rchisq(3, df = 3)
    set.seed(2); sum(runif(5000)); ...    (and rnorm / rchisq at df = 1, 2, 3)
"""
from __future__ import annotations

import numpy as np

from gpu_deseq import _r_rng


def test_unif_rand_matches_r_exactly() -> None:
    r_first5 = np.array([
        0.18488225992769003, 0.70237403595820069, 0.57332633482292295,
        0.16805192036554217, 0.94383933884091675,
    ])
    assert np.array_equal(_r_rng.unif_rand(5), r_first5)


def test_norm_rand_matches_r_exactly() -> None:
    """Requires R's INVERSION norm_rand on Wichura AS 241; SciPy's Cephes
    `ndtri` differs in the last bit and would fail this."""
    r_first3 = np.array([
        -0.89691454662498138, 0.18484918464674249, 1.5878453312088232,
    ])
    assert np.array_equal(_r_rng.norm_rand(3), r_first3)


def test_rchisq_matches_r_exactly() -> None:
    r_first3 = np.array([
        0.60839875872378502, 0.53883407524968197, 0.37812323397130138,
    ])
    assert np.array_equal(_r_rng.rchisq(3, 3), r_first3)


def test_long_streams_match_r() -> None:
    """5000 draws exercises the MT twist and both rgamma rejection branches
    (GS for df=1 -> shape 0.5, GD for df>=2); any divergence in the accept /
    reject path would desynchronise the stream and blow these up."""
    assert _r_rng.unif_rand(5000)[-1] == 0.75262894993647933
    assert _r_rng.norm_rand(5000)[-1] == -0.011654248993440903
    for df, last in ((1, 0.050499129472948719),
                     (2, 3.6772841788001469),
                     (3, 6.3020351512781403)):
        assert _r_rng.rchisq(5000, df)[-1] == last, df


def test_kl_grid_draws_match_r_stream() -> None:
    """The whole grid replay, including R's `rnorm(n, 0, 0)` short-circuit.

    R's rnorm returns mu without calling norm_rand when sigma == 0, so the
    x = 0 grid point consumes no normal variates. Drawing them anyway would
    desynchronise every later grid point -- this test is the regression guard.
    """
    r_sum_last = [
        (-33.075857194589901, -0.15571828684092992),
        (-38.735723829792022, 0.53752046012436172),
        (-57.856469603406438, -2.7611123020947712),
        (-42.019653875319378, -2.6057616436736679),
        (-53.015760614443629, 0.37611379208530105),
    ]
    got = _r_rng.kl_grid_draws(3, np.linspace(0.0, 8.0, 5), n_samp=100, seed=2)
    for i, (s, last) in enumerate(r_sum_last):
        assert abs(got[i].sum() - s) < 1e-11, i
        assert got[i, -1] == last, i
